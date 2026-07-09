#!/usr/bin/env python3
"""
Streamlit Web UI for HR AI Assistant.
Provides a chat interface for interacting with the fine-tuned HR model.
"""

import os

import streamlit as st
import numpy as np
from pathlib import Path

# NOTE: `mlx`, `transformers`, and `unsloth` are imported lazily inside the
# functions that need them. They are platform-specific (mlx = Apple Silicon,
# unsloth = CUDA) and must not be required just to start the app. This lets the
# app boot in fallback mode on any Linux host (e.g. Azure Container Apps).


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
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    import mlx.core as mx  # lazy: Apple-Silicon-only dependency

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


# Page configuration
st.set_page_config(
    page_title="HR AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "user_input" not in st.session_state:
    st.session_state.user_input = ""


@st.cache_resource
def load_model(model_path: str = "models/instruction_ft_adapter", max_seq_length: int = 512):
    """Load the fine-tuned model and tokenizer (cached)."""
    from transformers import AutoTokenizer  # lazy import
    from unsloth import FastLanguageModel  # lazy: CUDA-only dependency

    st.info(f"Loading model from: {model_path}")
    local_path = Path(model_path)

    if local_path.exists() and local_path.is_dir() and (local_path / "model_weights.safetensors").exists():
        # Load the base model first, then apply local adapter weights.
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Qwen2.5-0.5B",
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        model.load_weights(str(local_path / "model_weights.safetensors"), strict=False)

        if (local_path / "tokenizer.json").exists() or (local_path / "tokenizer_config.json").exists():
            tokenizer = AutoTokenizer.from_pretrained(str(local_path), trust_remote_code=True)
        return model, tokenizer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_answer(model, tokenizer, question: str, max_new_tokens: int = 200,
                    temperature: float = 0.7) -> str:
    """Generate an answer for the given HR question."""
    # Intent detection: treat 'how-to' and 'apply' questions as actionable
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

    prompt = f"{system_prompt}\n{instruction_template}\n\n### Instruction:\n{question}\n\n### Response:\n"

    # Rerank candidates: generate multiple completions and score them
    import re

    def _score_candidate(text: str, is_actionable_q: bool) -> float:
        score = 0.0
        code_indicators = ["#include", "using namespace", "int main", "std::", "<bits/stdc++>", "#include <", "printf("]
        if any(ind in text for ind in code_indicators):
            score -= 1000.0

        if is_actionable_q:
            if re.search(r"(^|\n)\s*\d+\s*[\.|\)]", text):
                score += 10.0
            if "step" in text.lower() or "steps" in text.lower():
                score += 3.0

        length = len(text.split())
        if length < 10:
            score -= 2.0
        if length > 1000:
            score -= 5.0

        return score

    from reranker import rerank

    is_actionable_q = _is_actionable(question)
    candidates = []
    num_candidates = 5
    for _ in range(num_candidates):
        gen = _generate_text(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
        )
        resp = gen.split("### Response:\n")[-1].strip()
        candidates.append(resp)

    best = rerank(candidates, question)

    if best is None:
        return ""

    if any(ind in best for ind in ["#include", "int main", "std::"]):
        gen = _generate_text(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_p=0.9,
        )
        best = gen.split("### Response:\n")[-1].strip()

    return best


def main():
    # Sidebar
    with st.sidebar:
        st.title("🤖 HR AI Assistant")
        st.markdown("---")
        st.markdown("""
        **Domain**: Human Resources
        **Model**: Qwen2.5-0.5B (DPO-aligned)
        **Fine-tuning**: 3-stage (QLoRA)
        """)
        st.markdown("---")

        # Model settings
        st.subheader("⚙️ Settings")
        # Defaults are configurable via environment variables so the same image
        # can be deployed in fallback mode (default) or pointed at a model.
        default_model_path = os.getenv("HR_MODEL_PATH", "models/non_instruction_ft_adapter")
        default_fallback = os.getenv("HR_FALLBACK", "0").lower() in ("1", "true", "yes")
        model_path = st.text_input("Model Path", value=default_model_path)
        use_fallback = st.checkbox("Use fallback responses (no model)", value=default_fallback)
        max_tokens = st.slider("Max Tokens", 50, 500, 200, 50)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)

        st.markdown("---")
        st.markdown("### Example Questions")
        example_questions = [
            "How can I apply for sick leave?",
            "What is the work from home policy?",
            "How does reimbursement work?",
            "What is the notice period for resignation?",
            "What employee benefits are available?",
            "How is overtime calculated?",
            "What is the onboarding process?",
            "How do I report a compliance concern?",
            "What is the attendance policy?",
            "How are performance reviews conducted?",
        ]
        for q in example_questions:
            if st.button(q, key=f"ex_{q[:20]}", use_container_width=True):
                st.session_state.user_input = q
                st.rerun() # koti changed

    # Main content
    st.title("HR Policy Assistant")
    st.markdown("Ask me anything about HR policies, leave, benefits, compliance, and more!")

    # Load model (or use fallback)
    model = None
    tokenizer = None
    if use_fallback:
        st.info("Using deterministic fallback responses (no model loaded)")
    else:
        try:
            with st.spinner("Loading model..."):
                model, tokenizer = load_model(model_path)
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.info("Make sure the model path is correct and the model files exist, or enable 'Use fallback responses' in the sidebar.")
            return

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask an HR question..."):
        st.session_state.user_input = prompt

    if st.session_state.user_input:
        prompt = st.session_state.user_input
        st.session_state.user_input = ""

        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    if use_fallback:
                        # Template-based fallback to avoid repetitive pattern matching
                        ql = prompt.strip().lower()

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
                            response = fallback_sick_leave()
                        elif is_actionable(ql):
                            response = fallback_actionable_generic()
                        else:
                            response = fallback_policy()
                    else:
                        response = generate_answer(model, tokenizer, prompt, max_tokens, temperature)

                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Error generating response: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


if __name__ == "__main__":
    main()