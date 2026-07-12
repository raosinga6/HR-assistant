# HR Assistant — User Guide

How to install, run, and use the HR AI Assistant. For how it works internally,
see [CODE_OVERVIEW.md](CODE_OVERVIEW.md); for the project story and model
rationale, see [EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md).

---

## 1. What it does

Answers HR policy questions (comp-off, policy reviews, work hours…) grounded
in the company policy corpus. Runs locally on CPU, NVIDIA GPU (CUDA), or Apple
Silicon (MPS) — no cloud calls.

**Three answer modes** (sidebar radio):

| Mode | How it answers | When to use |
|---|---|---|
| **Grounded (RAG, cited)** — default | Retrieves actual policy passages, answers only from them with expandable citations, and replies *"I don't know based on the company's policy documents."* when the corpus doesn't cover the question | Always, for trustworthy answers |
| Fine-tuned model (ungrounded) | Qwen2.5-0.5B + SFT adapter answering from weights alone | Comparing model behavior; **can fabricate details** |
| Fallback templates | Deterministic canned answers, zero ML dependencies | Demos, CI |

---

## 2. Install

```bash
git clone https://github.com/raosinga6/HR-assistant
cd HR-assistant
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requirements: Python 3.10+, ~2 GB disk (base model downloads from the Hugging
Face Hub on first run), ~4 GB RAM.

The fine-tuned LoRA adapter ships in `models/instruction_ft_adapter/`
(adapter weights + tokenizer). The base model (`unsloth/Qwen2.5-0.5B`) is
fetched automatically on first use.

---

## 3. Using the chat UI (recommended)

```bash
streamlit run src/app.py
```

Open http://localhost:8501, then ask questions or click the example questions
in the sidebar.

**Sidebar settings**

| Setting | Default | Meaning |
|---|---|---|
| Model Path | `models/instruction_ft_adapter` | LoRA adapter dir, full model dir, or HF hub id |
| Use fallback responses | off | Answer from deterministic templates without loading a model |
| Max Tokens | 200 | Answer length cap |
| Temperature | 0.7 | 0 = deterministic, higher = more varied |

---

## 4. Using the CLI

```bash
# One question
python src/inference.py --question "How can I apply for sick leave?"

# Interactive session (type 'exit' to quit)
python src/inference.py --interactive

# Demo: runs 5 sample HR questions
python src/inference.py

# Without a model (deterministic template answers — for demos/CI)
python src/inference.py --fallback --question "How can I apply for sick leave?"
```

Useful flags: `--model_path <dir|hub-id>`, `--max_tokens N`, `--temperature T`.

---

## 5. Observability

Every answer is instrumented:

- **Per-answer metrics** (shown under each response): prompt tokens, completion
  tokens, total tokens, latency, mode, device, and — in grounded mode — the top
  retrieval score.
- **Source provenance** (grounded mode): each cited passage shows the exact
  `data/<file>:<line>` it was retrieved from, expandable to the full text.
- **Session totals** (sidebar → 📊 Observability): running token count and an
  in-session audit-log view.
- **Persistent audit log**: every question is appended to
  `logs/audit_log.jsonl` (one JSON object per answer) with timestamp, question,
  answer, refusal flag, the retrieved sources (`id`, `source_ref`, `score`), and
  token/latency metrics. Point `HR_AUDIT_LOG` elsewhere to change the path.

Example audit entry (abridged):
```json
{"ts": "2026-07-13T00:46:45", "question": "Who approves my compensatory off request?",
 "answer": "Your compensatory off request must be approved by your reporting manager.",
 "refused": false,
 "sources": [{"n": 1, "id": "qa-24", "source_ref": "data/instruction_dataset.jsonl:25", "score": 0.68}],
 "metrics": {"mode": "grounded", "prompt_tokens": 409, "completion_tokens": 14,
             "total_tokens": 423, "latency_s": 3.12, "top_score": 0.68}}
```

## 6. Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `HR_MODEL_PATH` | `models/instruction_ft_adapter` | Model the Streamlit app loads |
| `HR_FALLBACK` | `0` | `1` = start the app in fallback mode (no model) |
| `HR_DEVICE` | auto (cuda → mps → cpu) | Force a torch device, e.g. `HR_DEVICE=cpu` |
| `HR_RAG_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | Generator for grounded mode |
| `HR_RAG_MIN_SCORE` | `0.45` | Retrieval score below which grounded mode refuses |
| `HR_AUDIT_LOG` | `logs/audit_log.jsonl` | Where the audit log is written |

---

## 6. Running the tests

```bash
pytest -q        # 23 tests, ~5s (no model download needed)
```

---

## 7. Deploying

See [AZURE_DEPLOYMENT.md](AZURE_DEPLOYMENT.md):
- **Fallback image** (`Dockerfile`) — tiny, no ML deps, deterministic answers.
- **Model image** (`Dockerfile.model`) — serves the fine-tuned model on CPU.

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| First answer is slow | Base model downloading / loading (~15–20s). Subsequent answers: ~4–12s on MPS, slower on CPU. |
| `Failed to load model` in the UI | Check Model Path exists (needs `adapter_config.json` for adapters); or tick "Use fallback responses". |
| Out-of-memory | Use `HR_DEVICE=cpu`, or lower `max_seq_length`; the 0.5B model needs ~2–3 GB free. |
| Answers look generic | Expected outside the 120 training Q&A pairs — see scope note in §1. Retrieval grounding (RAG) is the planned fix. |
| HF Hub rate-limit warnings | Harmless; set `HF_TOKEN` to silence and speed up downloads. |
