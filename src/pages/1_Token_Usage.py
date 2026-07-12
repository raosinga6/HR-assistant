#!/usr/bin/env python3
"""
Token Usage — a dedicated page listing every question asked and the tokens it
used, appended to the audit log as questions are raised.

This is a Streamlit multipage app page: it lives in src/pages/ and appears in
the sidebar navigation alongside the main chat page. It reads the persistent
audit log (logs/audit_log.jsonl) so it reflects every question, including those
from earlier sessions.
"""

import os
from pathlib import Path

import streamlit as st

try:
    from rag import read_audit_log
except ImportError:
    from src.rag import read_audit_log

st.set_page_config(page_title="Token Usage", page_icon="📊", layout="wide")

AUDIT_LOG = os.getenv("HR_AUDIT_LOG", "logs/audit_log.jsonl")

st.title("📊 Token Usage")
st.caption(f"Every question is appended here as it is asked · source: `{AUDIT_LOG}`")

left, right = st.columns([1, 6])
with left:
    if st.button("🔄 Refresh"):
        st.rerun()

entries = read_audit_log(AUDIT_LOG)
if not entries:
    st.info("No questions logged yet. Ask a question on the **HR Policy Assistant** "
            "page and it will appear here.")
    st.stop()

# ---- Summary metrics -------------------------------------------------------
def _sum(field):
    return sum(e["metrics"].get(field, 0) or 0 for e in entries)

total_q = len(entries)
prompt_tok = _sum("prompt_tokens")
completion_tok = _sum("completion_tokens")
total_tok = _sum("total_tokens")

c = st.columns(5)
c[0].metric("Questions", total_q)
c[1].metric("Prompt tokens", f"{prompt_tok:,}")
c[2].metric("Completion tokens", f"{completion_tok:,}")
c[3].metric("Total tokens", f"{total_tok:,}")
c[4].metric("Avg tokens / question", f"{total_tok // total_q if total_q else 0:,}")

# ---- Per-question table (newest first) -------------------------------------
rows = []
for i, e in enumerate(entries, 1):
    m = e["metrics"]
    rows.append({
        "#": i,
        "time": (e.get("ts", "") or "").replace("T", " "),
        "question": e.get("question", ""),
        "mode": m.get("mode"),
        "prompt": m.get("prompt_tokens", 0),
        "completion": m.get("completion_tokens", 0),
        "total": m.get("total_tokens", 0),
        "latency_s": m.get("latency_s"),
        "refused": "yes" if e.get("refused") else "",
        "retrieved from": ", ".join(s.get("source_ref", "") for s in e.get("sources", [])),
    })

st.subheader("Per-question log")

# Per-column explanations (shown on header hover, and spelled out below).
COLUMN_HELP = {
    "#": "Order in which the question was asked (1 = the first question logged).",
    "time": "When the question was asked (local time, YYYY-MM-DD HH:MM:SS).",
    "question": "The exact question the user asked.",
    "mode": "Answer mode used: grounded (RAG, cited), fine-tuned (model weights "
            "only), or fallback (fixed templates).",
    "prompt": "Prompt tokens — size of the input sent to the model (system rules + "
              "any retrieved passages + the question). Grounded mode is larger "
              "because the retrieved passages are included in the prompt.",
    "completion": "Completion tokens — how many tokens the model generated for the "
                  "answer.",
    "total": "Total tokens = prompt + completion. The overall model workload for "
             "this question.",
    "latency_s": "Seconds from question to answer (retrieval + generation). Refused "
                 "questions are fast because no answer is generated.",
    "refused": "'yes' if grounded mode declined to answer because no passage passed "
               "the retrieval-score threshold (no answer was generated).",
    "retrieved from": "The source passages the answer was grounded in, as "
                      "data/<file>:<line>. Empty for fine-tuned / fallback modes.",
}
column_config = {
    "#": st.column_config.NumberColumn("#", help=COLUMN_HELP["#"], width="small"),
    "time": st.column_config.TextColumn("time", help=COLUMN_HELP["time"]),
    "question": st.column_config.TextColumn("question", help=COLUMN_HELP["question"], width="large"),
    "mode": st.column_config.TextColumn("mode", help=COLUMN_HELP["mode"]),
    "prompt": st.column_config.NumberColumn("prompt", help=COLUMN_HELP["prompt"], format="%d"),
    "completion": st.column_config.NumberColumn("completion", help=COLUMN_HELP["completion"], format="%d"),
    "total": st.column_config.NumberColumn("total", help=COLUMN_HELP["total"], format="%d"),
    "latency_s": st.column_config.NumberColumn("latency (s)", help=COLUMN_HELP["latency_s"], format="%.2f"),
    "refused": st.column_config.TextColumn("refused", help=COLUMN_HELP["refused"]),
    "retrieved from": st.column_config.TextColumn("retrieved from", help=COLUMN_HELP["retrieved from"], width="large"),
}
st.dataframe(list(reversed(rows)), use_container_width=True, hide_index=True,
             column_config=column_config)

with st.expander("ℹ️ What do these columns mean?"):
    for col, desc in COLUMN_HELP.items():
        st.markdown(f"- **{col}** — {desc}")

# ---- Charts ----------------------------------------------------------------
totals = [r["total"] for r in rows]
col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Tokens per question")
    st.bar_chart({"total tokens": totals})
with col_b:
    st.subheader("Cumulative tokens")
    cumulative, running = [], 0
    for t in totals:
        running += t or 0
        cumulative.append(running)
    st.line_chart({"cumulative tokens": cumulative})

# ---- Download --------------------------------------------------------------
log_path = Path(AUDIT_LOG)
if log_path.exists():
    st.download_button("⬇️ Download audit log (JSONL)", log_path.read_text(),
                       file_name="audit_log.jsonl", mime="application/json")
