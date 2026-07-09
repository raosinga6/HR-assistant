import os
import sys

# Ensure project src is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.reranker import rerank, default_weights, is_code_like


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


def test_is_code_like_detection():
    assert is_code_like("std::cout << x;")
    assert is_code_like("#include <stdio.h>")
    assert is_code_like("using namespace std;")
    assert not is_code_like("Please submit your request via the HR portal.")
    assert not is_code_like("1. Notify your manager.\n2. Submit the form.")


def test_rerank_empty_returns_empty_string():
    assert rerank([], "How do I apply?") == ""


def test_rerank_non_actionable_prefers_substantive_answer():
    # Non-actionable question: no numbered bonus, so length dominates and the
    # too-short candidate (<10 words) is penalized.
    candidates = [
        "Short.",
        "The company reviews its policy annually to stay compliant with laws and best practices.",
    ]
    best = rerank(candidates, "What is the policy review cadence?")
    assert best.startswith("The company reviews")


def test_rerank_ties_return_first_candidate():
    # Two equally-scored actionable answers -> the earlier one wins (strict >).
    candidates = [
        "1. Step one.\n2. Step two.\n3. Step three, all good here.",
        "1. Step alpha.\n2. Step beta.\n3. Step gamma, all good here.",
    ]
    best = rerank(candidates, "How do I submit a request?")
    assert best == candidates[0]


def test_rerank_all_code_still_returns_a_candidate():
    candidates = [
        "#include <bits/stdc++.h>\nint main(){}",
        "printf(\"hello\");",
    ]
    best = rerank(candidates, "How do I apply?")
    assert best in candidates


def test_weights_can_invert_code_preference():
    # Flip the code penalty into a reward and the code candidate should win.
    candidates = [
        "Submit your request through the HR portal and notify your manager.",
        "std::cout << 1;",
    ]
    weights = default_weights()
    weights["code_penalty"] = 1000.0
    best = rerank(candidates, "How do I submit a request?", weights=weights)
    assert "std::" in best
