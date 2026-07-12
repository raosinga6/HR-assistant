import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.rag import (REFUSAL, SYSTEM_PROMPT, build_corpus, build_messages,
                     format_context)

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
