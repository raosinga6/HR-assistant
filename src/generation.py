#!/usr/bin/env python3
"""
Portable generation backend for the HR Assistant.

This replaces the previous Apple-Silicon-only (`mlx`) generation loop with a
standard `transformers` + `peft` implementation that runs on CPU, CUDA, or MPS.
It also fixes the prompt format to match how the model was actually fine-tuned.

Training used the bare Alpaca format (see notebooks/instruction_finetuning.ipynb):

    ### Instruction:\\n{instruction}\\n\\n### Response:\\n{response}<eos>

so inference must use the *same* prefix and nothing more. The old code prepended
a system prompt + an extra instruction template, which was out-of-distribution
and degraded output quality.
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

# Reranker import that works whether this module is imported as `src.generation`
# (tests, `python -m`) or `generation` (streamlit run src/app.py, src/ on path).
try:  # pragma: no cover - import shim
    from reranker import rerank, is_code_like
except ImportError:  # pragma: no cover - import shim
    from src.reranker import rerank, is_code_like


PROMPT_TEMPLATE = "### Instruction:\n{question}\n\n### Response:\n"
RESPONSE_MARKER = "### Response:\n"


def resolve_device(prefer: Optional[str] = None) -> str:
    """Pick a torch device: explicit override > env HR_DEVICE > cuda > mps > cpu."""
    import torch

    prefer = prefer or os.getenv("HR_DEVICE")
    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def build_prompt(question: str) -> str:
    """Format a question exactly as the model saw prompts during fine-tuning."""
    return PROMPT_TEMPLATE.format(question=question.strip())


def extract_response(generated: str) -> str:
    """Strip the prompt prefix, returning only the model's answer."""
    return generated.split(RESPONSE_MARKER)[-1].strip()


def load_model(model_path: str, device: Optional[str] = None):
    """Load a model + tokenizer for inference.

    Supports three kinds of `model_path`:
      * a PEFT/LoRA adapter directory (has adapter_config.json) -> base + adapter
      * a full model directory (has config.json)
      * a Hugging Face hub id

    Returns (model, tokenizer, device).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = resolve_device(device)
    dtype = torch.float16 if device == "cuda" else torch.float32
    path = Path(model_path)

    if (path / "adapter_config.json").exists():
        from peft import PeftModel

        cfg = json.loads((path / "adapter_config.json").read_text())
        base = cfg.get("base_model_name_or_path") or "Qwen/Qwen2.5-0.5B"
        tok_src = str(path) if (path / "tokenizer_config.json").exists() else base
        tokenizer = AutoTokenizer.from_pretrained(tok_src)
        model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype)
        model = PeftModel.from_pretrained(model, str(path))
    else:
        # Full model directory on disk, otherwise treat as a hub id.
        src = str(path) if path.exists() else model_path
        tokenizer = AutoTokenizer.from_pretrained(src)
        model = AutoModelForCausalLM.from_pretrained(src, dtype=dtype)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    model.eval()
    return model, tokenizer, device


def generate_candidates(
    model,
    tokenizer,
    device: str,
    question: str,
    num_candidates: int = 3,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> List[str]:
    """Generate N candidate answers with a single batched `model.generate` call."""
    import torch

    prompt = build_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    do_sample = bool(temperature and temperature > 0.0)
    # Greedy decoding is deterministic: transformers only allows (and only makes
    # sense to return) a single sequence. Sampling can return N distinct ones.
    n = num_candidates if do_sample else 1
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        num_return_sequences=n,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs.update(temperature=temperature, top_p=top_p)

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return [extract_response(t) for t in texts]


def generate_answer(
    model,
    tokenizer,
    device: str,
    question: str,
    num_candidates: int = 3,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
) -> str:
    """Generate several candidates, rerank, and return the best answer.

    If the winner still looks like source code, retry once deterministically.
    """
    candidates = generate_candidates(
        model, tokenizer, device, question,
        num_candidates=num_candidates,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    best = rerank(candidates, question)

    if best and is_code_like(best):
        retry = generate_candidates(
            model, tokenizer, device, question,
            num_candidates=1, max_new_tokens=max_new_tokens, temperature=0.0,
        )
        if retry and not is_code_like(retry[0]):
            best = retry[0]

    return best or ""
