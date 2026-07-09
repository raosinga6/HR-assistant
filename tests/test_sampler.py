import os
import sys

import numpy as np

# Ensure project root is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.inference import _sample_next_token


def test_greedy_is_argmax():
    # temperature == 0 must be deterministic and return the highest-logit index
    logits = np.array([0.1, 5.0, 0.2, -3.0])
    assert _sample_next_token(logits, temperature=0.0) == 1


def test_greedy_handles_negative_logits():
    logits = np.array([-10.0, -2.0, -8.0])
    assert _sample_next_token(logits, temperature=0.0) == 1


def test_top_p_collapses_to_argmax_on_peaked_distribution():
    # One token dominates: its probability alone exceeds top_p, so the nucleus
    # falls back to the single most-likely token -> effectively deterministic.
    logits = np.array([20.0, 0.0, 0.0, 0.0])
    for _ in range(50):
        assert _sample_next_token(logits, temperature=1.0, top_p=0.9) == 0


def test_returns_valid_in_range_index():
    np.random.seed(0)
    vocab = 8
    logits = np.random.randn(vocab)
    for _ in range(200):
        idx = _sample_next_token(logits, temperature=0.8, top_p=0.95)
        assert isinstance(idx, int)
        assert 0 <= idx < vocab


def test_low_temperature_favors_top_token():
    # With a clear winner and low temperature, sampling should almost always
    # pick the top token.
    np.random.seed(1)
    logits = np.array([3.0, 1.0, 0.5])
    picks = [_sample_next_token(logits, temperature=0.2, top_p=1.0) for _ in range(200)]
    assert picks.count(0) > picks.count(1)
    assert picks.count(0) > picks.count(2)
