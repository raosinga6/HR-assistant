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


DEFAULT_AUDIT_LOG = "logs/audit_log.jsonl"


# --------------------------------------------------------------------------
# Audit log
# --------------------------------------------------------------------------

def append_audit_log(question: str, answer: str, sources: List[Dict],
                     meta: Dict, path: Optional[str] = None) -> Dict:
    """Append one answer event to the audit log (JSONL) and return the entry.

    Records what was asked, what was answered, which passages it was retrieved
    from (id + source_ref = file:line + score), and token/latency metrics.
    """
    import datetime

    path = path or os.getenv("HR_AUDIT_LOG", DEFAULT_AUDIT_LOG)
    entry = {
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "answer": answer,
        "refused": meta.get("refused", False),
        "sources": [
            {"n": i, "id": s.get("id"), "source_ref": s.get("source_ref"),
             "score": round(s.get("score", 0.0), 3)}
            for i, s in enumerate(sources, 1)
        ],
        "metrics": {k: meta.get(k) for k in
                    ("mode", "model", "device", "prompt_tokens",
                     "completion_tokens", "total_tokens", "latency_s", "top_score")},
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def read_audit_log(path: Optional[str] = None) -> List[Dict]:
    """Read all audit-log entries (oldest first). Missing file -> []."""
    path = path or os.getenv("HR_AUDIT_LOG", DEFAULT_AUDIT_LOG)
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------

def build_corpus(data_dir: str = "data") -> List[Dict]:
    """Collect retrieval passages from the HR policy data files.

    Every passage records `source_ref` (file:line) so an answer can be audited
    back to the exact line it was retrieved from.
    """
    data_dir = Path(data_dir)
    passages: List[Dict] = []

    # Policy paragraphs (blank-line separated), tracking the starting line so we
    # can cite e.g. "non_instruction_data.txt:12".
    policy_file = data_dir / "non_instruction_data.txt"
    if policy_file.exists():
        rel = str(policy_file)
        buf: List[str] = []
        start_line = None
        pi = 0
        lines = policy_file.read_text().split("\n")
        for lineno, line in enumerate(lines + [""], start=1):  # sentinel flush
            if line.strip():
                if start_line is None:
                    start_line = lineno
                buf.append(line)
            elif buf:
                para = " ".join(" ".join(buf).split())
                if len(para) >= 80:
                    passages.append({
                        "id": f"policy-{pi}", "source": "policy document",
                        "text": para, "source_ref": f"{rel}:{start_line}",
                    })
                    pi += 1
                buf, start_line = [], None

    # Q&A pairs — real policy answers, one JSONL line each.
    qa_file = data_dir / "instruction_dataset.jsonl"
    if qa_file.exists():
        rel = str(qa_file)
        for i, line in enumerate(qa_file.open()):
            rec = json.loads(line)
            passages.append({
                "id": f"qa-{i}",
                "source": "policy Q&A",
                "text": f"Q: {rec['instruction']}\nA: {rec['response']}",
                "source_ref": f"{rel}:{i + 1}",  # JSONL line number (1-based)
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
        self.model_name = model_path
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
               min_score: Optional[float] = None) -> Tuple[str, List[Dict], Dict]:
        """Return (answer, sources, meta).

        `meta` carries observability data: token counts, latency, device,
        whether the question was refused, and the top retrieval score.
        Deterministic by default (temperature=0). If no retrieved passage
        reaches `min_score`, refuse immediately — don't hand weak context to a
        small model that will improvise.
        """
        import time

        import torch

        if min_score is None:
            min_score = float(os.getenv("HR_RAG_MIN_SCORE", DEFAULT_MIN_SCORE))

        t0 = time.time()
        chunks = self.search(question, top_k=top_k)
        top_score = chunks[0]["score"] if chunks else 0.0

        meta = {
            "mode": "grounded", "model": self.model_name, "device": self.device,
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "top_score": round(top_score, 3), "refused": False,
            "retrieved": len(chunks), "used": 0,
        }

        if not chunks or top_score < min_score:
            meta["refused"] = True
            meta["latency_s"] = round(time.time() - t0, 2)
            return REFUSAL, chunks, meta

        # Ground the answer in only the passages that clear the relevance bar —
        # not all top_k. This keeps the model from drawing on weak matches and
        # keeps the displayed/cited sources meaningful. (At least chunks[0]
        # qualifies, since top_score >= min_score above.)
        relevant = [c for c in chunks if c["score"] >= min_score]

        prompt = self.tokenizer.apply_chat_template(
            build_messages(question, relevant), tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_tokens = int(inputs["input_ids"].shape[1])

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

        new_tokens = output[0][prompt_tokens:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        completion_tokens = int(new_tokens.shape[0])
        meta.update(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_s=round(time.time() - t0, 2),
            used=len(relevant),
        )
        return answer, relevant, meta


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
    answer, sources, meta = rag.answer(args.question, top_k=args.top_k)
    append_audit_log(args.question, answer, sources, meta)

    print(f"\nQ: {args.question}\n\nA: {answer}\n")
    if not meta["refused"]:
        print("Sources:")
        for i, s in enumerate(sources, 1):
            preview = s["text"][:80].replace("\n", " ")
            print(f"  [{i}] {s.get('source_ref')} (score {s['score']:.3f}) {preview}...")
    print(f"\nTokens: {meta['prompt_tokens']} prompt + {meta['completion_tokens']} "
          f"completion = {meta['total_tokens']} total | {meta['latency_s']}s | "
          f"{meta['device']} | top score {meta['top_score']}")


if __name__ == "__main__":
    main()
