import os
import sys

# Ensure project root is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.generation import build_prompt, extract_response, resolve_device


def test_build_prompt_matches_training_format():
    # Must exactly match the bare Alpaca format used during fine-tuning:
    # "### Instruction:\n{q}\n\n### Response:\n" and nothing more.
    assert build_prompt("How do I apply?") == "### Instruction:\nHow do I apply?\n\n### Response:\n"


def test_build_prompt_strips_whitespace():
    assert build_prompt("  spaced question  ").startswith("### Instruction:\nspaced question")


def test_extract_response_takes_text_after_marker():
    generated = "### Instruction:\nHow do I apply?\n\n### Response:\nSubmit the form."
    assert extract_response(generated) == "Submit the form."


def test_extract_response_uses_last_marker():
    generated = "### Response:\nfirst\n### Response:\nsecond"
    assert extract_response(generated) == "second"


def test_extract_response_without_marker_returns_stripped_text():
    assert extract_response("  just an answer  ") == "just an answer"


def test_resolve_device_respects_explicit_preference():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"


def test_resolve_device_reads_env(monkeypatch):
    monkeypatch.setenv("HR_DEVICE", "cpu")
    assert resolve_device() == "cpu"


def test_resolve_device_falls_back_to_known_value(monkeypatch):
    monkeypatch.delenv("HR_DEVICE", raising=False)
    assert resolve_device() in {"cuda", "mps", "cpu"}


def test_greedy_generation_uses_single_sequence():
    """Regression: greedy decoding (temperature=0) must not request multiple
    return sequences — transformers rejects num_return_sequences>1 without
    sampling. We capture the kwargs a fake model.generate receives."""
    import numpy as np

    from src import generation

    captured = {}

    class _Inputs(dict):
        def to(self, device):
            return self

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 0

        def __call__(self, text, return_tensors=None):
            return _Inputs(input_ids=np.zeros((1, 5)))  # 5 prompt tokens

        def batch_decode(self, seqs, skip_special_tokens=True):
            return ["### Response:\nok"] * len(seqs)

    class FakeModel:
        def generate(self, **kwargs):
            captured.update(kwargs)
            return [np.zeros(7) for _ in range(kwargs["num_return_sequences"])]

    out = generation.generate_candidates(
        FakeModel(), FakeTokenizer(), "cpu", "q?",
        num_candidates=3, temperature=0.0,
    )
    assert captured["do_sample"] is False
    assert captured["num_return_sequences"] == 1
    assert len(out) == 1

    generation.generate_candidates(
        FakeModel(), FakeTokenizer(), "cpu", "q?",
        num_candidates=3, temperature=0.7,
    )
    assert captured["do_sample"] is True
    assert captured["num_return_sequences"] == 3
