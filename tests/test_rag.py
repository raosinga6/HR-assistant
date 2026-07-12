import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import json

from src.rag import (REFUSAL, SYSTEM_PROMPT, append_audit_log, build_corpus,
                     build_messages, format_context, read_audit_log)

CHUNKS = [
    {"id": "qa-1", "source": "policy Q&A", "text": "Q: Who approves comp off?\nA: Your reporting manager.", "score": 0.7},
    {"id": "policy-2", "source": "policy document", "text": "Compensatory off must be applied in the leave system.", "score": 0.6},
]


def test_build_corpus_reads_both_data_files():
    passages = build_corpus(os.path.join(ROOT, "data"))
    sources = {p["source"] for p in passages}
    assert sources == {"policy document", "policy Q&A"}
    # 120 Q&A pairs + policy paragraphs
    assert sum(1 for p in passages if p["source"] == "policy Q&A") == 120
    assert sum(1 for p in passages if p["source"] == "policy document") > 20
    assert all(p["text"].strip() for p in passages)


def test_format_context_numbers_passages():
    ctx = format_context(CHUNKS)
    assert "[1] (policy Q&A)" in ctx
    assert "[2] (policy document)" in ctx


def test_build_messages_contract():
    messages = build_messages("Who approves comp off?", CHUNKS)
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert "Question: Who approves comp off?" in messages[1]["content"]
    assert "[1] (policy Q&A)" in messages[1]["content"]


def test_system_prompt_enforces_grounding_and_refusal():
    assert "ONLY" in SYSTEM_PROMPT
    assert REFUSAL in SYSTEM_PROMPT
    assert "invent" in SYSTEM_PROMPT


def test_every_passage_has_a_source_ref():
    # The audit trail: each passage traces to a data file and line number.
    passages = build_corpus(os.path.join(ROOT, "data"))
    for p in passages:
        assert ":" in p["source_ref"]
        f, line = p["source_ref"].rsplit(":", 1)
        assert f.endswith((".jsonl", ".txt"))
        assert line.isdigit() and int(line) >= 1
    # Q&A source_ref line number matches its 1-based JSONL position
    qa = [p for p in passages if p["id"] == "qa-0"][0]
    assert qa["source_ref"].endswith("instruction_dataset.jsonl:1")


def test_append_audit_log_writes_jsonl(tmp_path):
    meta = {"mode": "grounded", "model": "m", "device": "cpu",
            "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
            "latency_s": 0.5, "top_score": 0.7, "refused": False}
    log = tmp_path / "audit.jsonl"
    entry = append_audit_log("Q?", "A.", CHUNKS, meta, path=str(log))

    assert entry["question"] == "Q?"
    assert entry["metrics"]["total_tokens"] == 15
    assert entry["sources"][0]["id"] == "qa-1"
    on_disk = json.loads(log.read_text().strip())
    assert on_disk == entry


def test_read_audit_log_roundtrip(tmp_path):
    log = str(tmp_path / "audit.jsonl")
    meta = {"mode": "grounded", "total_tokens": 15, "refused": False}
    append_audit_log("Q1?", "A1", CHUNKS, meta, path=log)
    append_audit_log("Q2?", "A2", CHUNKS, meta, path=log)

    entries = read_audit_log(log)
    assert len(entries) == 2
    assert [e["question"] for e in entries] == ["Q1?", "Q2?"]  # oldest first


def test_read_audit_log_missing_file_returns_empty(tmp_path):
    assert read_audit_log(str(tmp_path / "nope.jsonl")) == []
