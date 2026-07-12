#!/usr/bin/env python3
"""
Streamlit Web UI for HR AI Assistant.

Provides a chat interface backed by the portable `transformers` + `peft`
generation backend (see generation.py). Also supports a model-free fallback
mode so the app runs anywhere (e.g. Azure Container Apps) without a GPU.
"""

import os

import streamlit as st

# Backend import works whether launched via `streamlit run src/app.py`
# (src/ on path) or imported as a package.
try:
    from generation import load_model as _load_model, generate_answer_with_meta
    from rag import append_audit_log
except ImportError:
    from src.generation import load_model as _load_model, generate_answer_with_meta
    from src.rag import append_audit_log


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
def load_model(model_path: str):
    """Load and cache the fine-tuned model, tokenizer, and device."""
    st.info(f"Loading model from: {model_path}")
    return _load_model(model_path)


@st.cache_resource
def load_rag():
    """Load and cache the grounded (RAG) pipeline over the policy corpus."""
    try:
        from rag import HRRag
    except ImportError:
        from src.rag import HRRag
    return HRRag()


def render_metrics(meta: dict):
    """Show token usage / latency for one answer as a compact metric row."""
    if not meta:
        return
    cols = st.columns(4)
    cols[0].metric("Prompt tokens", meta.get("prompt_tokens", 0))
    cols[1].metric("Completion tokens", meta.get("completion_tokens", 0))
    cols[2].metric("Total tokens", meta.get("total_tokens", 0))
    cols[3].metric("Latency (s)", meta.get("latency_s", 0))
    bits = [f"mode: **{meta.get('mode')}**", f"device: `{meta.get('device')}`"]
    if meta.get("top_score") is not None:
        bits.append(f"top match: `{meta.get('top_score')}`")
    if meta.get("refused"):
        bits.append("↩︎ **refused** (below retrieval threshold)")
    st.caption(" · ".join(bits))


def fallback_answer(question: str) -> str:
    """Deterministic template answers used when no model is loaded."""
    ql = question.strip().lower()

    def is_actionable(q: str) -> bool:
        keywords = ["how", "apply", "procedure", "process", "steps", "request", "submit"]
        return any(k in q for k in keywords)

    if "sick" in ql and "leave" in ql:
        return (
            "1. Confirm eligibility and dates: check your contract or HR portal for leave entitlements.\n"
            "2. Notify your manager immediately (email/Slack) with the dates and expected return.\n"
            "3. Submit a formal request in the HR portal: Leave → Apply → Sick Leave; attach a doctor’s note if required.\n"
            "4. If you don’t have a doctor’s note yet, mark the request as 'pending documentation' and follow up when available.\n"
            "5. Expect confirmation within 1–3 business days; if delayed, contact HR at hr@example.com."
        )
    if is_actionable(ql):
        return (
            "1. Identify the exact action needed and required documents.\n"
            "2. Inform your manager and ask for any manager-level approval required.\n"
            "3. Submit the request through the HR portal or form and attach documents.\n"
            "4. Follow up with HR or your manager if no response within the stated timeline."
        )
    return (
        "Policy summary: The company follows a hybrid work policy; employees need manager approval and must be available during core hours.\n"
        "Practical implications: (1) Request WFH at least 24 hours in advance; (2) Use VPN and secure data handling; (3) Update team status when working remotely."
    )


def main():
    # Sidebar
    with st.sidebar:
        st.title("🤖 HR AI Assistant")
        st.markdown("---")
        st.markdown("""
        **Domain**: Human Resources
        **Default**: grounded RAG over policy docs (cited, refuses off-corpus)
        **Model**: Qwen2.5-0.5B
        """)
        st.markdown("---")

        # Model settings. Defaults are configurable via environment variables so
        # the same image can run in fallback mode or be pointed at a model.
        st.subheader("⚙️ Settings")
        MODES = ["Grounded (RAG, cited)", "Fine-tuned model (ungrounded)",
                 "Fallback templates (no model)"]
        default_fallback = os.getenv("HR_FALLBACK", "0").lower() in ("1", "true", "yes")
        mode = st.radio(
            "Answer mode", MODES, index=2 if default_fallback else 0, key="answer_mode",
            help="Grounded: retrieves actual policy passages and answers only from "
                 "them — with citations, and a refusal when the policy doesn't "
                 "cover the question. Fine-tuned: answers from model weights "
                 "alone (can fabricate details).",
        )
        grounded = mode == MODES[0]
        use_fallback = mode == MODES[2]
        default_model_path = os.getenv("HR_MODEL_PATH", "models/instruction_ft_adapter")
        model_path = st.text_input("Model Path (fine-tuned mode)", value=default_model_path)
        max_tokens = st.slider("Max Tokens", 50, 500, 200, 50)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1,
                                help="0 = same answer every time (recommended for policy QA)")

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
                st.rerun()

        # Observability panel
        st.markdown("---")
        st.subheader("📊 Observability")
        st.metric("Tokens used this session", st.session_state.get("total_tokens", 0))
        audit = st.session_state.get("audit", [])
        st.caption(f"{len(audit)} question(s) logged → `{os.getenv('HR_AUDIT_LOG', 'logs/audit_log.jsonl')}`")
        if audit:
            with st.expander("🧾 Audit log (this session)"):
                for i, e in enumerate(reversed(audit), 1):
                    m = e["meta"]
                    st.markdown(f"**Q:** {e['q']}")
                    refs = ", ".join(s.get("source_ref", "?") for s in e["sources"]) or "—"
                    st.caption(
                        f"{m.get('total_tokens', 0)} tok · {m.get('latency_s', 0)}s · "
                        f"{'refused' if m.get('refused') else 'answered'} · retrieved from: {refs}"
                    )
                    if i < len(audit):
                        st.divider()

    # Main content
    st.title("HR Policy Assistant")
    st.markdown("Ask me anything about HR policies, leave, benefits, compliance, and more!")

    # Load the selected backend
    model = tokenizer = device = rag = None
    if use_fallback:
        st.info("Using deterministic fallback responses (no model loaded)")
    elif grounded:
        try:
            with st.spinner("Loading policy index and model..."):
                rag = load_rag()
            st.success(f"Grounded mode ready — {rag.num_passages} policy passages, device: {rag.device}")
        except Exception as e:
            st.error(f"Failed to load grounded pipeline: {e}")
            return
    else:
        try:
            with st.spinner("Loading model..."):
                model, tokenizer, device = load_model(model_path)
            st.success(f"Model loaded successfully on {device}!")
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.info("Make sure the model path is correct, or switch modes in the sidebar.")
            return

    # Chat history + observability state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "audit" not in st.session_state:
        st.session_state.audit = []
    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            for i, s in enumerate(message.get("sources", []), 1):
                ref = s.get("source_ref", s.get("source", ""))
                with st.expander(f"[{i}] {ref} (relevance {s['score']:.2f})"):
                    st.markdown(s["text"])
            if message.get("meta"):
                render_metrics(message["meta"])

    # Chat input
    if prompt := st.chat_input("Ask an HR question..."):
        st.session_state.user_input = prompt

    if st.session_state.user_input:
        prompt = st.session_state.user_input
        st.session_state.user_input = ""

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    sources, meta = [], {}
                    if use_fallback:
                        response = fallback_answer(prompt)
                        meta = {"mode": "fallback", "model": "templates", "device": "cpu",
                                "prompt_tokens": 0, "completion_tokens": 0,
                                "total_tokens": 0, "latency_s": 0, "top_score": None,
                                "refused": False}
                    elif grounded:
                        response, sources, meta = rag.answer(
                            prompt, max_new_tokens=max_tokens, temperature=temperature)
                    else:
                        response, meta = generate_answer_with_meta(
                            model, tokenizer, device, prompt,
                            max_new_tokens=max_tokens, temperature=temperature,
                            model_name=model_path)

                    st.markdown(response)
                    for i, s in enumerate(sources, 1):
                        ref = s.get("source_ref", s.get("source", ""))
                        with st.expander(f"[{i}] {ref} (relevance {s['score']:.2f})"):
                            st.markdown(s["text"])
                    render_metrics(meta)

                    # Persist to the audit log + running session totals.
                    append_audit_log(prompt, response, sources, meta)
                    st.session_state.audit.append(
                        {"q": prompt, "meta": meta, "sources": sources})
                    st.session_state.total_tokens += meta.get("total_tokens", 0)

                    msg = {"role": "assistant", "content": response, "meta": meta}
                    if sources:
                        msg["sources"] = sources
                    st.session_state.messages.append(msg)
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
