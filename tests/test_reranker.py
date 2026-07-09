import os
import sys

# Ensure project src is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.reranker import rerank, default_weights


def test_rerank_prefers_numbered():
    candidates = [
        "This is a short note.",
        "1. Do this.\n2. Do that.\n3. Finish.",
        "#include <stdio.h>\nint main() { return 0; }",
    ]
    question = "How do I apply for sick leave?"
    best = rerank(candidates, question)
    assert "1." in best or "2." in best


def test_rerank_penalizes_code():
    candidates = [
        "#include <bits/stdc++.h>\nint main(){}",
        "Steps:\n1) Submit request.\n2) Get approval.",
    ]
    question = "How do I submit a request?"
    best = rerank(candidates, question)
    assert "#include" not in best


def test_weights_tuning():
    candidates = [
        "Short.",
        "1. Step one.\n2. Step two.\n3. Step three.",
    ]
    question = "How to proceed?"
    weights = default_weights()
    # Reduce numbered bonus to prefer shorter answers
    weights['numbered_bonus'] = 0.0
    best = rerank(candidates, question, weights=weights)
    # With numbered bonus removed, still should not pick the very short one
    assert best != "Short."
