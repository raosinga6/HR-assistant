# HR Assistant — Code Overview

A module-by-module tour for developers. Companion to
[USER_GUIDE.md](USER_GUIDE.md) (usage) and
[EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md) (why it was built this way).

---

## Repository layout

```
HR assistant/
├── src/
│   ├── rag.py             # Grounded (RAG) answering — the app's DEFAULT mode
│   ├── generation.py      # Fine-tuned generation backend (model loading + generate)
│   ├── inference.py       # HRAssistant class + CLI entry point
│   ├── reranker.py        # Candidate answer scoring (fine-tuned mode)
│   ├── app.py             # Streamlit chat UI (multipage entrypoint)
│   └── pages/
│       └── 1_Token_Usage.py   # Token usage + audit-log page
├── models/
│   ├── non_instruction_ft_adapter/   # Stage 1 LoRA (domain adaptation)
│   ├── instruction_ft_adapter/       # Stage 2 LoRA (SFT) ← fine-tuned mode default
│   └── instruction_ft/               # Stage 2 training checkpoints
├── data/
│   ├── non_instruction_data.txt      # 60 HR policy paragraphs (stage 1 + RAG corpus)
│   ├── instruction_dataset.jsonl     # 120 Q&A pairs (stage 2 + RAG corpus)
│   └── preference_dataset.jsonl      # 60 chosen/rejected pairs (stage 3)
├── hr_index/              # Auto-built RAG index (embeddings.npy + metadata) — gitignored
├── logs/audit_log.jsonl   # Per-answer audit log — gitignored
├── notebooks/             # The three fine-tuning stages (training code)
├── tests/                 # 32 unit tests (pytest)
├── docs/                  # This documentation
├── .streamlit/config.toml # Server config (headless, watcher off)
├── Dockerfile             # Slim fallback-mode image
├── Dockerfile.model       # CPU model-serving image
└── requirements*.txt      # full / app-slim / model-serving dependency sets
```

## The three answer modes

The app answers in one of three modes (sidebar radio; **Grounded** is default):

| Mode | Module | Behavior |
|---|---|---|
| **Grounded (RAG)** | `src/rag.py` | Retrieve policy passages → answer only from them, cite, refuse if uncovered |
| **Fine-tuned** | `src/generation.py` + `src/inference.py` | Qwen2.5-0.5B + SFT adapter answering from weights (can fabricate) |
| **Fallback** | `src/app.py`, `src/inference.py` | Deterministic keyword-routed templates, zero ML deps |

---

## `src/generation.py` — the generation backend

The heart of inference. Pure `transformers` + `peft` (portable: CPU/CUDA/MPS —
no mlx, no Unsloth at inference time).

| Function | Role |
|---|---|
| `resolve_device()` | Device pick: `HR_DEVICE` env → cuda → mps → cpu |
| `build_prompt(q)` | Formats the question **exactly** as trained: `### Instruction:\n{q}\n\n### Response:\n` — nothing else. Prompt/training mismatch was a major early bug (see EXECUTIVE_SUMMARY §5). |
| `load_model(path)` | Three path kinds: LoRA adapter dir (has `adapter_config.json` → loads base model + applies adapter via peft), full model dir, or HF hub id. Returns `(model, tokenizer, device)`. |
| `generate_candidates(...)` | One batched `model.generate()` call with `num_return_sequences=N` (default 3), temperature/top-p sampling, EOS stop. Greedy (temp 0) forces N=1 — transformers rejects multiple greedy sequences. Strips the prompt prefix from each output. |
| `generate_answer(...)` | candidates → `reranker.rerank()` → best; if the winner looks like code, one deterministic retry (`temperature=0`). |
| `generate_answer_with_meta(...)` | Same, but also returns an observability `meta` dict (prompt/completion/total tokens, latency, device) — used by the UI. |

## `src/rag.py` — grounded (RAG) answering + observability

The app's default mode. `build_corpus()` turns `data/non_instruction_data.txt`
(policy paragraphs) + `data/instruction_dataset.jsonl` (Q&A pairs) into ~180
passages, each tagged with **`source_ref` = `data/<file>:<line>`** for audit
provenance. `HRRag` embeds them with MiniLM (index auto-built under `hr_index/`,
exact numpy cosine search — no vector DB needed at this scale) and generates
with Qwen2.5-0.5B-Instruct under a strict system prompt: answer only from the
passages, cite by number, refuse otherwise.

| Function / method | Role |
|---|---|
| `HRRag.search(q, top_k)` | Cosine top-k over normalized embeddings; each hit carries `source_ref` + `score`. |
| `HRRag.answer(...)` | Refuse-early if top score < `HR_RAG_MIN_SCORE` (default 0.45), else generate. Returns `(answer, sources, meta)` — `meta` has token counts, latency, device, `top_score`, `refused`. Deterministic by default. |
| `append_audit_log(...)` | Append one JSONL entry (ts, question, answer, refused, sources w/ `source_ref`+score, metrics) to `logs/audit_log.jsonl` (`HR_AUDIT_LOG`). |
| `read_audit_log(path)` | Read all entries oldest-first (missing file → `[]`); powers the Token Usage page. |

**Why the score threshold matters:** below ~0.45 the small model would adapt a
similar-but-wrong passage (e.g. answer *sick leave* with the *comp-off*
procedure). Refusing before generation is what stops the fabrication.

## `src/pages/1_Token_Usage.py` — Token Usage page

A Streamlit multipage page (any file under `src/pages/` becomes a sidebar nav
entry). Reads `logs/audit_log.jsonl` via `read_audit_log()` and shows summary
metrics, a per-question table (time, question, mode, tokens, latency, refusal,
`data:line` sources), tokens-per-question + cumulative charts, and a download.
Because it reads the persistent log, it reflects every question across sessions.

## `src/reranker.py` — answer selection (ungrounded mode)

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

## `src/app.py` — Streamlit UI (multipage entrypoint)

Chat page + backend router. A sidebar **Answer mode** radio picks Grounded /
Fine-tuned / Fallback; each backend is `@st.cache_resource`-loaded on demand.
The chat page stays lean — it renders the answer and, in grounded mode, the
**cited sources** (each labelled `data:line`). Every answer is written to the
audit log via `append_audit_log`; all token/usage detail is surfaced on the
**Token Usage** page rather than inline. `HR_MODEL_PATH` / `HR_FALLBACK` env vars
set defaults so one container image serves multiple modes. All ML imports are
lazy — the app boots with zero ML dependencies in fallback mode.

## `tests/` — what's covered

| File | Covers |
|---|---|
| `test_generation.py` | Prompt format contract, response extraction, device resolution, greedy N=1 kwargs |
| `test_reranker.py` | Code detection, scoring, ties, weight overrides, empty input |
| `test_fallback.py` | Template routing, numbered steps, no model loaded in fallback |
| `test_rag.py` | Corpus parsing, `source_ref` provenance, prompt/refusal contract, audit-log write + read |

**32 tests**, ~5s, no model download. Deliberately **not** covered: live
`model.generate()` output (non-deterministic, needs weights) — exercised instead
by running real questions (`python src/inference.py`, `python src/rag.py -q ...`).

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
