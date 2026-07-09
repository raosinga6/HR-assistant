import re
from typing import List, Dict, Optional


def default_weights() -> Dict[str, float]:
    return {
        "code_penalty": -1000.0,
        "numbered_bonus": 10.0,
        "step_keyword_bonus": 3.0,
        "too_short_penalty": -2.0,
        "too_long_penalty": -5.0,
    }


def is_code_like(text: str) -> bool:
    code_indicators = ["#include", "using namespace", "int main", "std::", "<bits/stdc++>", "#include <", "printf("]
    return any(ind in text for ind in code_indicators)


def rerank(candidates: List[str], question: str, weights: Optional[Dict[str, float]] = None) -> str:
    """Rerank a list of candidate responses and return the best candidate.

    The scoring function is intentionally simple and tunable via `weights`.
    """
    if weights is None:
        weights = default_weights()

    ql = question.strip().lower()
    is_actionable = any(k in ql for k in ["how", "apply", "procedure", "process", "steps", "request", "submit"]) 

    def score_text(text: str) -> float:
        score = 0.0
        if is_code_like(text):
            score += weights.get("code_penalty", -1000.0)

        if is_actionable:
            if re.search(r"(^|\n)\s*\d+\s*[\.|\)]", text):
                score += weights.get("numbered_bonus", 10.0)
            if "step" in text.lower() or "steps" in text.lower():
                score += weights.get("step_keyword_bonus", 3.0)

        length = len(text.split())
        if length < 10:
            score += weights.get("too_short_penalty", -2.0)
        if length > 1000:
            score += weights.get("too_long_penalty", -5.0)

        # small tie-breaker: prefer earlier candidates slightly
        return score

    best = None
    best_score = float("-inf")
    for c in candidates:
        s = score_text(c)
        if s > best_score:
            best_score = s
            best = c

    return best if best is not None else ""
