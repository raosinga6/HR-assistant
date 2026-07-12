#!/usr/bin/env python3
"""
Grounded (RAG) answering over the HR policy corpus.

Fixes the fabrication problem of the pure fine-tuned path: instead of answering
from model weights, we retrieve the actual policy text and constrain the model
to answer ONLY from it, with numbered citations — and to refuse when the corpus
doesn't cover the question (e.g. sick leave, which the policy data never
mentions).

Corpus = data/non_instruction_data.txt (policy paragraphs)
       + data/instruction_dataset.jsonl (Q&A pairs, kept as "Q: ... A: ...")

Index  = MiniLM sentence embeddings + exact numpy cosine search, stored under
         hr_index/ (auto-built on first use; delete the directory to rebuild).
         At ~180 passages a vector database is unnecessary — and faiss-cpu's
         bundled libomp segfaults against torch's on macOS in this venv.

Generator = Qwen2.5-0.5B-Instruct by default. The instruct model follows the
grounding/refusal rules reliably; the project's SFT adapter was trained on bare
Q->A pairs and never learned to condition on provided context or to refuse.

CLI:
    python src/rag.py --question "How do I apply for compensatory off?"
    python src/rag.py --rebuild
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from generation import resolve_device
except ImportError:  # pragma: no cover - import shim
    from src.generation import resolve_device

DEFAULT_GEN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_INDEX_DIR = "hr_index"

# If the best-matching passage scores below this cosine similarity, the corpus
# doesn't cover the question — refuse without generating. Small models will
# otherwise "helpfully" adapt a similar-but-different procedure (e.g. answer
# sick leave with the compensatory-off process). Measured on this corpus:
# real matches score ~0.6+, fabrication bait ~0.25-0.43.
DEFAULT_MIN_SCORE = 0.45

REFUSAL = "I don't know based on the company's policy documents."

SYSTEM_PROMPT = (
    "You are the company's HR policy assistant. Answer the employee's question "
    "using ONLY the provided policy passages. Rules:\n"
    "1. Base your answer strictly on the passages. Do not use outside knowledge "
    "or invent procedures, team names, or timelines.\n"
    f"2. If the passages do not contain the answer, reply exactly: \"{REFUSAL}\"\n"
    "3. Cite the passages you used by number, e.g. [1] or [2].\n"
    "4. Be concise and practical."
)


# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------

def build_corpus(data_dir: str = "data") -> List[Dict]:
    """Collect retrieval passages from the HR policy data files."""
    data_dir = Path(data_dir)
    passages: List[Dict] = []

    # Policy paragraphs (blank-line separated; skip headings/short fragments)
    policy_file = data_dir / "non_instruction_data.txt"
    if policy_file.exists():
        for i, para in enumerate(re.split(r"\n\s*\n", policy_file.read_text())):
            para = " ".join(para.split())
            if len(para) >= 80:
                passages.append({"id": f"policy-{i}", "source": "policy document", "text": para})

    # Q&A pairs — real policy answers, phrased as questions employees ask
    qa_file = data_dir / "instruction_dataset.jsonl"
    if qa_file.exists():
        for i, line in enumerate(qa_file.open()):
            rec = json.loads(line)
            passages.append({
                "id": f"qa-{i}",
                "source": "policy Q&A",
                "text": f"Q: {rec['instruction']}\nA: {rec['response']}",
            })

    return passages


# --------------------------------------------------------------------------
# Prompting
# --------------------------------------------------------------------------

def format_context(chunks: List[Dict]) -> str:
    return "\n\n".join(f"[{i}] ({c['source']})\n{c['text']}"
                       for i, c in enumerate(chunks, 1))


def build_messages(question: str, chunks: List[Dict]) -> List[Dict]:
    user = f"Policy passages:\n\n{format_context(chunks)}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# --------------------------------------------------------------------------
# Index + pipeline
# --------------------------------------------------------------------------

class HRRag:
    """Retrieve HR policy passages and generate grounded, cited answers."""

    def __init__(self, index_dir: str = DEFAULT_INDEX_DIR, data_dir: str = "data",
                 model_path: Optional[str] = None, device: Optional[str] = None):
        import numpy as np
        from sentence_transformers import SentenceTransformer

        self.index_dir = Path(index_dir)
        self.embedder = SentenceTransformer(
            os.getenv("HR_EMBED_MODEL", DEFAULT_EMBED_MODEL))

        if not (self.index_dir / "embeddings.npy").exists():
            self._build_index(data_dir, np)

        self.embeddings = np.load(self.index_dir / "embeddings.npy")
        self.passages = [json.loads(l) for l in (self.index_dir / "metadata.jsonl").open()]

        # Generator (lazy device pick: cuda -> mps -> cpu)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = resolve_device(device)
        model_path = model_path or os.getenv("HR_RAG_MODEL", DEFAULT_GEN_MODEL)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)
        self.model.to(self.device)
        self.model.eval()

    def _build_index(self, data_dir: str, np):
        passages = build_corpus(data_dir)
        if not passages:
            raise FileNotFoundError(f"No policy data found under {data_dir}/")
        print(f"Building HR policy index: {len(passages)} passages ...", flush=True)
        emb = self.embedder.encode([p["text"] for p in passages],
                                   normalize_embeddings=True).astype(np.float32)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.index_dir / "embeddings.npy", emb)
        with (self.index_dir / "metadata.jsonl").open("w") as f:
            for p in passages:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

    @property
    def num_passages(self) -> int:
        return len(self.passages)

    def search(self, question: str, top_k: int = 4) -> List[Dict]:
        """Exact cosine search: embeddings are L2-normalized, so dot == cosine."""
        import numpy as np

        q = self.embedder.encode([question], normalize_embeddings=True).astype(np.float32)
        scores = self.embeddings @ q[0]
        top = np.argsort(-scores)[:top_k]
        out = []
        for idx in top:
            rec = dict(self.passages[int(idx)])
            rec["score"] = float(scores[idx])
            out.append(rec)
        return out

    def answer(self, question: str, top_k: int = 4, max_new_tokens: int = 256,
               temperature: float = 0.0,
               min_score: Optional[float] = None) -> Tuple[str, List[Dict]]:
        """Return (answer, sources). Deterministic by default: policy QA
        should give the same answer every time.

        If no retrieved passage reaches `min_score`, refuse immediately —
        don't hand weak context to a small model that will improvise.
        """
        import torch

        if min_score is None:
            min_score = float(os.getenv("HR_RAG_MIN_SCORE", DEFAULT_MIN_SCORE))

        chunks = self.search(question, top_k=top_k)
        if not chunks or chunks[0]["score"] < min_score:
            return REFUSAL, chunks
        prompt = self.tokenizer.apply_chat_template(
            build_messages(question, chunks), tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        do_sample = bool(temperature and temperature > 0.0)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs.update(temperature=temperature, top_p=0.9)

        with torch.no_grad():
            output = self.model.generate(**inputs, **gen_kwargs)

        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return answer, chunks


def main():
    parser = argparse.ArgumentParser(description="Grounded HR policy Q&A")
    parser.add_argument("--question", help="Question to answer")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the index")
    parser.add_argument("--index-dir", default=DEFAULT_INDEX_DIR)
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    if args.rebuild:
        import shutil
        shutil.rmtree(args.index_dir, ignore_errors=True)
        print("Index removed; it will rebuild on next use.")
        if not args.question:
            return

    if not args.question:
        parser.error("--question is required (or use --rebuild)")

    rag = HRRag(index_dir=args.index_dir)
    answer, sources = rag.answer(args.question, top_k=args.top_k)
    print(f"\nQ: {args.question}\n\nA: {answer}\n\nSources:")
    for i, s in enumerate(sources, 1):
        preview = s["text"][:90].replace("\n", " ")
        print(f"  [{i}] ({s['source']}, score {s['score']:.3f}) {preview}...")


if __name__ == "__main__":
    main()
