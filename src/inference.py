#!/usr/bin/env python3
"""
Inference script for the HR AI Assistant.

Loads a fine-tuned model (LoRA adapter or full model) via the portable
`transformers` + `peft` backend in `generation.py` and exposes a
`generate_answer` method. Also supports a model-free `fallback` mode with
deterministic canned answers for UI/testing.
"""

import argparse

# Backend import that works whether run as `src.inference` (tests, `python -m`)
# or `inference` (script with src/ on path).
try:  # pragma: no cover - import shim
    from generation import load_model, generate_answer as _generate_answer
except ImportError:  # pragma: no cover - import shim
    from src.generation import load_model, generate_answer as _generate_answer


# The SFT adapter is the first stage that was actually trained to *answer*
# questions (Stage 1 / non_instruction only adapts vocabulary and cannot answer).
DEFAULT_MODEL_PATH = "models/instruction_ft_adapter"


class HRAssistant:
    """HR Domain-Specific AI Assistant using a fine-tuned model."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, max_seq_length: int = 512, fallback: bool = False):
        """
        Args:
            model_path: Path to a LoRA adapter dir, a full model dir, or a hub id.
            max_seq_length: Retained for API compatibility.
            fallback: If True, skip model loading and use canned responses.
        """
        self.model_path = model_path
        self.max_seq_length = max_seq_length
        self.model = None
        self.tokenizer = None
        self.device = None
        self.fallback = fallback

        if not self.fallback:
            self._load_model()

    def _load_model(self):
        """Load the fine-tuned model and tokenizer via the portable backend."""
        print(f"Loading model from {self.model_path}...")
        self.model, self.tokenizer, self.device = load_model(self.model_path)
        print(f"Model loaded successfully on {self.device}!")

    def generate_answer(self, question: str, max_new_tokens: int = 200, temperature: float = 0.7) -> str:
        """Generate an answer for the given HR question."""
        if self.fallback:
            return self._fallback_answer(question)

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")

        return _generate_answer(
            self.model,
            self.tokenizer,
            self.device,
            question,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    @staticmethod
    def _fallback_answer(question: str) -> str:
        """Deterministic canned answers for UI/testing without a model."""
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

        if "sick" in ql and "leave" in ql:
            return fallback_sick_leave()
        if is_actionable(ql):
            return fallback_actionable_generic()
        return fallback_policy()

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
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                        help="Path to a LoRA adapter dir, full model dir, or hub id")
    parser.add_argument("--question", type=str, help="Single question to answer")
    parser.add_argument("--interactive", action="store_true", help="Run interactive mode")
    parser.add_argument("--max_tokens", type=int, default=200, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--fallback", action="store_true", help="Use deterministic canned responses (no model load)")
    args = parser.parse_args()

    assistant = HRAssistant(model_path=args.model_path, fallback=args.fallback)

    if args.question:
        answer = assistant.generate_answer(args.question, args.max_tokens, args.temperature)
        print(f"Q: {args.question}")
        print(f"A: {answer}")
    elif args.interactive:
        assistant.interactive_mode()
    else:
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
