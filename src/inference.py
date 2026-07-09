#!/usr/bin/env python3
"""
Inference script for the HR AI Assistant.
Loads the final DPO-aligned model and provides a generate_answer function.
"""

import argparse
import torch
import numpy as np
import mlx.core as mx
from pathlib import Path
from transformers import AutoTokenizer
from unsloth import FastLanguageModel


def _sample_next_token(logits: np.ndarray, temperature: float = 1.0, top_p: float = 0.9) -> int:
    if temperature == 0.0:
        return int(np.argmax(logits))

    scores = logits.astype(np.float64) / temperature
    scores -= np.max(scores)
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores)

    sorted_indices = np.argsort(-probs)
    cumulative_probs = np.cumsum(probs[sorted_indices])
    selected = sorted_indices[cumulative_probs <= top_p]
    if selected.size == 0:
        selected = sorted_indices[:1]

    top_probs = probs[selected]
    top_probs = top_probs / top_probs.sum()
    return int(np.random.choice(selected, p=top_probs))


def _tokenize_prompt(tokenizer, prompt: str):
    if callable(tokenizer):
        return tokenizer(prompt, return_tensors="pt")

    if hasattr(tokenizer, "_tokenizer"):
        inner = tokenizer._tokenizer
        if callable(inner):
            return inner(prompt, return_tensors="pt")
        if hasattr(inner, "encode"):
            ids = inner.encode(prompt)
            return {"input_ids": np.array([ids], dtype=np.int64)}

    if hasattr(tokenizer, "encode"):
        ids = tokenizer.encode(prompt)
        return {"input_ids": np.array([ids], dtype=np.int64)}

    raise ValueError("Tokenizer does not support tokenization for this model.")


def _decode_tokens(token_ids, tokenizer):
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(token_ids, skip_special_tokens=True)

    if hasattr(tokenizer, "_tokenizer") and hasattr(tokenizer._tokenizer, "decode"):
        return tokenizer._tokenizer.decode(token_ids, skip_special_tokens=True)

    raise ValueError("Tokenizer does not support decoding for this model.")


def _generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    inputs = _tokenize_prompt(tokenizer, prompt)
    token_ids = inputs["input_ids"]
    if hasattr(token_ids, "cpu"):
        token_ids = token_ids.cpu().numpy()
    token_ids = np.asarray(token_ids)
    token_ids = token_ids[0].tolist()

    for _ in range(max_new_tokens):
        tokens = mx.array(np.array([token_ids], dtype=np.int64))
        logits = np.asarray(model(tokens)[0, -1])
        next_token = _sample_next_token(logits, temperature=temperature, top_p=top_p)
        token_ids.append(next_token)
        if next_token == tokenizer.eos_token_id:
            break

    return _decode_tokens(token_ids, tokenizer)


class HRAssistant:
    """HR Domain-Specific AI Assistant using fine-tuned model."""

    def __init__(self, model_path: str = "models/dpo_aligned_merged", max_seq_length: int = 512, fallback: bool = False):
        """
        Initialize the HR Assistant.

        Args:
            model_path: Path to the merged model directory
            max_seq_length: Maximum sequence length for generation
        """
        self.model_path = model_path
        self.max_seq_length = max_seq_length
        self.model = None
        self.tokenizer = None
        self.fallback = fallback

        if not self.fallback:
            # Attempt to load model; if it fails and fallback=True was provided,
            # the exception will be handled by the caller. When running with
            # fallback mode we skip model loading entirely.
            self._load_model()

    def _load_model(self):
        """Load the fine-tuned model and tokenizer."""
        print(f"Loading model from {self.model_path}...")
        local_path = Path(self.model_path)

        if local_path.exists() and local_path.is_dir() and (local_path / "model_weights.safetensors").exists():
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name="unsloth/Qwen2.5-0.5B",
                max_seq_length=self.max_seq_length,
                dtype=None,
                load_in_4bit=True,
            )
            self.model.load_weights(str(local_path / "model_weights.safetensors"), strict=False)

            if (local_path / "tokenizer.json").exists() or (local_path / "tokenizer_config.json").exists():
                self.tokenizer = AutoTokenizer.from_pretrained(str(local_path), trust_remote_code=True)
        else:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_path,
                max_seq_length=self.max_seq_length,
                dtype=None,
                load_in_4bit=True,
            )
        FastLanguageModel.for_inference(self.model)
        print("Model loaded successfully!")

    def generate_answer(self, question: str, max_new_tokens: int = 200, temperature: float = 0.7) -> str:
        """
        Generate an answer for the given HR question.

        Args:
            question: The HR question to answer
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic)

        Returns:
            The generated answer string
        """
        # Fallback deterministic canned answers for UI/testing without a model
        if self.fallback:
            import re
            ql = question.strip().lower()

            def is_actionable(q: str) -> bool:
                keywords = ["how", "apply", "procedure", "process", "steps", "request", "submit"]
                return any(k in q for k in keywords)

            def fallback_sick_leave():
                return (
                    "1. Confirm eligibility and dates: check your contract or HR portal for leave entitlements.\n"
                    "2. Notify your manager immediately (email/Slack) with the dates and expected return.\n"
                    "3. Submit a formal request in the HR portal: Leave → Apply → Sick Leave; attach a doctor’s note if required.\n"
                    "4. If you don’t have a doctor’s note yet, mark the request as 'pending documentation' and follow up when available.\n"
                    "5. Expect confirmation within 1–3 business days; if delayed, contact HR at hr@example.com."
                )

            def fallback_actionable_generic():
                return (
                    "1. Identify the exact action needed and required documents.\n"
                    "2. Inform your manager and ask for any manager-level approval required.\n"
                    "3. Submit the request through the HR portal or form and attach documents.\n"
                    "4. Follow up with HR or your manager if no response within the stated timeline."
                )

            def fallback_policy():
                return (
                    "Policy summary: The company follows a hybrid work policy; employees need manager approval and must be available during core hours.\n"
                    "Practical implications: (1) Request WFH at least 24 hours in advance; (2) Use VPN and secure data handling; (3) Update team status when working remotely."
                )

            # Specific case: sick leave
            if "sick" in ql and "leave" in ql:
                return fallback_sick_leave()

            if is_actionable(ql):
                return fallback_actionable_generic()

            return fallback_policy()

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")

        # Intent detection: actionable vs policy questions
        def _is_actionable(q: str) -> bool:
            ql = q.lower()
            keywords = ["how", "how can", "how do", "apply", "application", "procedure", "process", "steps", "what is the process", "what is the procedure"]
            return any(k in ql for k in keywords)

        system_prompt = "You are a helpful HR assistant. Answer concisely in plain English. Do not output code or source files."
        if _is_actionable(question):
            instruction_template = (
                "Provide a clear, numbered step-by-step procedure the user can follow. "
                "Include required documents, expected timelines, who to contact, and an example. "
                "Keep each step short and actionable."
            )
        else:
            instruction_template = (
                "Provide a concise policy summary in plain language, then list practical implications and examples. "
                "If applicable, give 3 brief actionable notes."
            )

        prompt_text = f"{system_prompt}\n{instruction_template}\n\n### Instruction:\n{question}\n\n### Response:\n"

        # Reranking: generate several candidates and score them
        def _score_candidate(text: str, is_actionable_q: bool) -> float:
            score = 0.0
            # Penalize code-like output heavily
            code_indicators = ["#include", "using namespace", "int main", "std::", "<bits/stdc++>", "#include <", "printf("]
            if any(ind in text for ind in code_indicators):
                score -= 1000.0

            # Prefer numbered steps for actionable queries
            if is_actionable_q:
                if re.search(r"(^|\n)\s*\d+\s*[\.|\)]", text):
                    score += 10.0
                if "step" in text.lower() or "steps" in text.lower():
                    score += 3.0

            # Slight preference for concise answers (not extremely short)
            length = len(text.split())
            if length < 10:
                score -= 2.0
            if length > 1000:
                score -= 5.0

            # Small fluency proxy: penalize lots of repeated lines
            if re.search(r"(\n\s*\1\s*){3,}", text):
                score -= 5.0

            return score

        import re

        is_actionable_q = _is_actionable(question)
        candidates = []
        num_candidates = 5

        for i in range(num_candidates):
            gen = _generate_text(
                self.model,
                self.tokenizer,
                prompt_text,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
            )
            resp = gen.split("### Response:\n")[-1].strip()
            candidates.append(resp)

        # Score candidates and pick best
        best = None
        best_score = -1e9
        for c in candidates:
            s = _score_candidate(c, is_actionable_q)
            if s > best_score:
                best_score = s
                best = c

        # If best looks like code, attempt a deterministic retry (temp=0)
        if best is None:
            return ""

        if any(ind in best for ind in ["#include", "int main", "std::"]):
            gen = _generate_text(
                self.model,
                self.tokenizer,
                prompt_text,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                top_p=0.9,
            )
            best = gen.split("### Response:\n")[-1].strip()

        return best

    def interactive_mode(self):
        """Run interactive Q&A session."""
        print("\n=== HR AI Assistant ===")
        print("Type 'exit' or 'quit' to end the session.\n")

        while True:
            try:
                question = input("Question: ").strip()
                if question.lower() in ['exit', 'quit', 'q']:
                    print("Goodbye!")
                    break
                if not question:
                    continue

                print("Generating answer...")
                answer = self.generate_answer(question)
                print(f"\nAnswer: {answer}\n")
                print("-" * 60)

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="HR AI Assistant Inference")
    parser.add_argument("--model_path", type=str, default="models/dpo_aligned_merged",
                        help="Path to the merged model")
    parser.add_argument("--question", type=str, help="Single question to answer")
    parser.add_argument("--interactive", action="store_true", help="Run interactive mode")
    parser.add_argument("--max_tokens", type=int, default=200, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")

    parser.add_argument("--fallback", action="store_true", help="Use deterministic canned responses (no model load)")
    args = parser.parse_args()

    # Initialize assistant (may be fallback)
    assistant = HRAssistant(model_path=args.model_path, fallback=args.fallback)

    if args.question:
        # Single question mode
        answer = assistant.generate_answer(args.question, args.max_tokens, args.temperature)
        print(f"Q: {args.question}")
        print(f"A: {answer}")
    elif args.interactive:
        # Interactive mode
        assistant.interactive_mode()
    else:
        # Default: run demo questions
        demo_questions = [
            "How can I apply for sick leave?",
            "What is the work from home policy?",
            "How does reimbursement work?",
            "What is the notice period for resignation?",
            "What employee benefits are available?",
        ]

        print("\n=== HR AI Assistant Demo ===\n")
        for q in demo_questions:
            answer = assistant.generate_answer(q, args.max_tokens, args.temperature)
            print(f"Q: {q}")
            print(f"A: {answer}")
            print("-" * 60)


if __name__ == "__main__":
    main()