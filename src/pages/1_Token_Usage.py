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
st.dataframe(list(reversed(rows)), use_container_width=True, hide_index=True)

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
