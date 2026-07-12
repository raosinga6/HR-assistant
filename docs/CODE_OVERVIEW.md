# HR Assistant — Code Overview

A module-by-module tour for developers. Companion to
[USER_GUIDE.md](USER_GUIDE.md) (usage) and
[EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md) (why it was built this way).

---

## Repository layout

```
HR assistant/
├── src/
│   ├── generation.py      # Core: model loading + grounded generation backend
│   ├── inference.py       # HRAssistant class + CLI entry point
│   ├── app.py             # Streamlit chat UI
│   └── reranker.py        # Candidate answer scoring
├── models/
│   ├── non_instruction_ft_adapter/   # Stage 1 LoRA (domain adaptation)
│   ├── instruction_ft_adapter/       # Stage 2 LoRA (SFT) ← served by default
│   └── instruction_ft/               # Stage 2 training checkpoints
├── data/
│   ├── non_instruction_data.txt      # 60 HR policy paragraphs (stage 1)
│   ├── instruction_dataset.jsonl     # 120 Q&A pairs (stage 2)
│   └── preference_dataset.jsonl      # 60 chosen/rejected pairs (stage 3)
├── notebooks/             # The three fine-tuning stages (training code)
├── tests/                 # 23 unit tests (pytest)
├── docs/                  # This documentation
├── Dockerfile             # Slim fallback-mode image
├── Dockerfile.model       # CPU model-serving image
└── requirements*.txt      # full / app-slim / model-serving dependency sets
```

---

## `src/generation.py` — the generation backend

The heart of inference. Pure `transformers` + `peft` (portable: CPU/CUDA/MPS —
no mlx, no Unsloth at inference time).

| Function | Role |
|---|---|
| `resolve_device()` | Device pick: `HR_DEVICE` env → cuda → mps → cpu |
| `build_prompt(q)` | Formats the question **exactly** as trained: `### Instruction:\n{q}\n\n### Response:\n` — nothing else. Prompt/training mismatch was a major early bug (see EXECUTIVE_SUMMARY §5). |
| `load_model(path)` | Three path kinds: LoRA adapter dir (has `adapter_config.json` → loads base model + applies adapter via peft), full model dir, or HF hub id. Returns `(model, tokenizer, device)`. |
| `generate_candidates(...)` | One batched `model.generate()` call with `num_return_sequences=N` (default 3), temperature/top-p sampling, EOS stop. Strips the prompt prefix from each output. |
| `generate_answer(...)` | candidates → `reranker.rerank()` → best; if the winner looks like code, one deterministic retry (`temperature=0`). |

## `src/reranker.py` — answer selection

Scores each candidate with simple, tunable heuristics (`default_weights()`):

- **−1000** if code-like (`#include`, `std::` …) — the 0.5B model occasionally
  emits code; this filters it out.
- **+10 / +3** for numbered steps / "step" wording on actionable questions
  ("how", "apply", "submit"…).
- Small penalties for too-short (<10 words) or too-long (>1000 words) answers.

Returns the highest scorer; ties go to the earliest candidate. `is_code_like()`
is exported for the retry guard.

## `src/inference.py` — API + CLI

- `HRAssistant(model_path, fallback=False)` — loads the model via the backend
  (default `models/instruction_ft_adapter`); `fallback=True` skips loading.
- `generate_answer(question, max_new_tokens, temperature)` — one call per
  question; in fallback mode returns deterministic keyword-routed templates
  (sick-leave → actionable-generic → policy-summary).
- `main()` — CLI: `--question`, `--interactive`, `--fallback`, demo mode.

## `src/app.py` — Streamlit UI

Thin layer over the same backend: `@st.cache_resource` model loading, sidebar
settings (model path / fallback / tokens / temperature), chat history in
`st.session_state`, example-question buttons. `HR_MODEL_PATH` / `HR_FALLBACK`
env vars set the defaults so one container image serves both modes. All ML
imports are lazy — the app boots with zero ML dependencies in fallback mode.

## `tests/` — what's covered

| File | Covers |
|---|---|
| `test_generation.py` | Prompt format contract, response extraction, device resolution |
| `test_reranker.py` | Code detection, scoring, ties, weight overrides, empty input |
| `test_fallback.py` | Template routing, numbered steps, no model loaded in fallback |

Deliberately **not** covered: live `model.generate()` output (non-deterministic,
needs weights); exercised instead by the demo (`python src/inference.py`).

## Training code — `notebooks/`

Three notebooks, one per fine-tuning stage (they produced the adapters in
`models/`; inference never imports them):

1. `non_instruction_finetuning.ipynb` — causal-LM QLoRA on 60 raw HR paragraphs.
2. `instruction_finetuning.ipynb` — QLoRA SFT on 120 Alpaca-format Q&A pairs.
   **This defines the prompt format inference must match.**
3. `dpo_alignment.ipynb` — DPO on 60 preference pairs (β=0.1). Note: its output
   artifacts were never exported to `models/`, so the served model is the SFT
   adapter.

## Dependency sets

| File | For |
|---|---|
| `requirements.txt` | Development: full stack incl. training deps (unsloth, trl) + pytest |
| `requirements-app.txt` | Slim container: streamlit + numpy only (fallback mode) |
| `requirements-model.txt` | Model-serving container: torch/transformers/peft/streamlit |
