# HR AI Assistant — Executive Summary & Developer Onboarding

*A presentation-style walkthrough for new developers: what we built, why we
made each choice, how the model was tuned, what went wrong, and where it goes
next.*

---

## 1. The problem

HR teams answer the same policy questions over and over — leave applications,
work-from-home rules, reimbursements, notice periods. We built a **local,
domain-specific AI assistant** that answers these questions in company-policy
language, runs on commodity hardware, and never sends employee questions to an
external API.

**Deliverables:** a multipage chat UI (Streamlit) with **grounded retrieval
(RAG), citations, and per-question observability**, a CLI, a Python API
(`HRAssistant`), container images for Azure, and a 32-test suite.

---

## 2. Why Qwen2.5-0.5B (the model choice)

We needed a base model that a small team could fine-tune and serve **without a
GPU cluster**. Qwen2.5-0.5B won on four criteria:

| Criterion | Why it matters | Qwen2.5-0.5B |
|---|---|---|
| **Size (0.5B params)** | Trains with LoRA on a laptop; serves on CPU (~2–3 GB RAM) | ✅ smallest capable option |
| **Quality per parameter** | Best-in-class small model at the time; coherent English, follows instructions after SFT | ✅ |
| **License** | Commercial use without negotiation | ✅ Apache 2.0 |
| **Ecosystem** | First-class `transformers`/`peft` support; Unsloth-optimized variant for fast training | ✅ |

**The honest trade-off:** a 0.5B model has limited world knowledge and shallow
reasoning. That is acceptable here because the domain is narrow (HR policy) and
we added **retrieval grounding** (§4), which shifts the factual burden from the
model's weights to retrieved documents — the small model only has to *rephrase*
text it is shown, not *recall* facts. If we needed deeper reasoning without
retrieval, we'd step up to 1.5B–7B — at proportionally higher serving cost.

---

## 3. How it was tuned — the three-stage pipeline

**Data:** `strova-ai/hr-policies-qa-dataset` (644 HR policy Q&A pairs, Hugging
Face), from which we derived three training sets.

```
Base Qwen2.5-0.5B
   │  Stage 1 — Domain adaptation (60 raw HR paragraphs, causal LM)
   ▼      "learn HR vocabulary and phrasing"
non_instruction_ft_adapter
   │  Stage 2 — Instruction SFT (120 Q&A pairs, Alpaca format)
   ▼      "learn to ANSWER questions, not just continue text"
instruction_ft_adapter   ◄── the model served in production today
   │  Stage 3 — DPO alignment (60 chosen/rejected pairs, β=0.1)
   ▼      "prefer complete, professional answers over curt/wrong ones"
dpo_aligned (trained in notebook; artifacts not yet exported — see §5)
```

**Why three stages instead of one?** Each stage fixes a different failure mode:
raw pre-trained models *continue* text rather than *answer* it (stage 2 fixes
that); generic instruction models don't speak your policy language (stage 1);
and SFT models trained on few examples can be curt or waffly (stage 3
addresses tone/completeness using preference pairs rather than more labels).

**Why QLoRA instead of full fine-tuning?** Full fine-tuning of even a 0.5B
model updates all ~500M weights. LoRA trains small low-rank adapter matrices
(rank 16, α=32, dropout 0.05) on 7 attention/MLP projection modules — roughly
**1% of the parameters** — while the 4-bit-quantized base stays frozen. Result:
minutes of training on one consumer GPU, a ~35 MB adapter artifact instead of a
~1 GB model copy, and the base model is reusable across experiments.

**Training/inference contract (the most important lesson in this repo):** the
SFT data was formatted as

```
### Instruction:\n{question}\n\n### Response:\n{answer}
```

Inference **must** reproduce that prefix *exactly* — no extra system prompt, no
added instructions. Deviating from the training format puts the model
out-of-distribution and quality collapses (we learned this the hard way, §5).

---

## 4. How it works at inference time

The app has **three answer modes**; Grounded is the default.

**Grounded (RAG) — the trustworthy path:**
```
question ──► embed ──► retrieve top-k policy passages (MiniLM + cosine)
        ──► if best score < 0.45: REFUSE ("I don't know…")   ← anti-fabrication gate
        ──► else: prompt = system rules + passages + question
        ──► model.generate() (Qwen2.5-0.5B-Instruct), deterministic
        ──► answer + numbered citations (each traceable to data/<file>:<line>)
```

**Fine-tuned (ungrounded) — the original path, kept for comparison:**
```
question ──► build_prompt (exact training format)
        ──► model.generate() — candidates (transformers + peft)
        ──► reranker: penalize code, reward numbered steps ──► best answer
```

**Fallback:** deterministic keyword-routed templates, zero ML dependencies —
demos, CI, and the slim Azure container.

- Runs on **CPU, CUDA, or MPS** (auto-detected; `HR_DEVICE` overrides).
- **Observability:** every answer is written to `logs/audit_log.jsonl` (question,
  answer, refusal flag, retrieved sources with line-level provenance, token/latency
  metrics); the **Token Usage** page renders the running table and charts.
- Full data-flow diagrams: [ARCHITECTURE.md](ARCHITECTURE.md).

---

## 5. What went wrong, and what we fixed (learn from this)

The first working version produced garbage and "hallucinated" answers. A code
review traced it to four independent bugs — none of them the model's fault:

| Bug | Symptom | Fix |
|---|---|---|
| Generation loop fed **mlx arrays into a PyTorch model** (Apple-Silicon-only, hand-rolled sampling) | Incoherent output; unportable | Rewrote on `transformers` `model.generate()` (PR #3) |
| **Wrong default model** — served the stage-1 domain adapter, which was never taught to answer questions | Rambling continuations instead of answers | Default switched to the SFT adapter |
| **Prompt mismatch** — inference added a system prompt + templates the model never saw in training | Degraded, off-format answers | Bare training-format prompt |
| `strict=False` weight loading + a pointer to a **non-existent model dir** | Silently served base weights | Proper peft adapter loading; fail loudly |

**Takeaways for new developers:**
1. *Fine-tuning is a contract* — inference must match the training format byte-for-byte.
2. *Silent fallbacks hide disasters* — `strict=False` made a broken load look successful.
3. *"Hallucination" is often a systems bug* — profile the pipeline before blaming the model.
4. *Verify end-to-end* — unit tests all passed while the model path was completely broken; only running real questions exposed it.

---

## 6. Key things to note

For anyone using, reviewing, or extending this app:

1. **Default to Grounded mode.** It cites its sources and refuses when the policy
   doesn't cover a question. Fine-tuned mode answers from weights and *can
   fabricate* — the same question there once produced three contradictory
   sick-leave procedures. Use it only to demonstrate that contrast.
2. **The corpus is small and specific.** It covers what's in `data/` (comp-off,
   policy-review cadence, work hours…). It does **not** contain a sick-leave or
   work-from-home policy — so grounded mode correctly refuses those. That's a
   feature, not a bug. Add policy text to `data/` and delete `hr_index/` to grow
   coverage.
3. **The refusal threshold (`HR_RAG_MIN_SCORE`, 0.45) is the anti-fabrication
   gate.** Below it, grounded mode refuses *before* generating. Lower it and the
   small model starts improvising from weak matches; raise it and it refuses more.
4. **Every answer is auditable.** Citations trace to `data/<file>:<line>`, and
   `logs/audit_log.jsonl` records question, answer, sources, and token/latency
   metrics. The **Token Usage** page reads that log.
5. **Determinism:** grounded mode runs at `temperature=0` — same question, same
   answer. Good for policy QA; raise temperature only if you want variety.
6. **Prompt format is a contract** (§5). If you retrain, keep inference in sync.
7. **Two sibling projects share this design:**
   [wiki-rag-assistant](https://github.com/raosinga6/wiki-rag-assistant) proved
   the RAG pattern first; this repo applied it to HR data.

---

## 7. Current status & roadmap

**Working today** (verified on `main`): grounded, cited HR answers in ~2–12 s on
Apple Silicon; three answer modes; per-question observability + audit log; 32
passing tests; Azure-deployable containers (fallback + CPU model serving).

**Known limits:** the corpus is small (~180 passages), so coverage is narrow;
the 0.5B generator is fluent but shallow; DPO (stage 3) artifacts were never
exported, so the served generator is the SFT adapter, not the preference-aligned
model.

**Roadmap, in order of impact:**
1. ~~**Retrieval grounding (RAG)**~~ — ✅ **shipped** (`src/rag.py`): the app
   default. Pattern proven first in the sibling
   [wiki-rag-assistant](https://github.com/raosinga6/wiki-rag-assistant) project.
2. **Grow the policy corpus** — the biggest lever on how much it can answer.
3. **Export and serve the DPO-aligned model** (stage 3 artifacts).
4. **Evaluation harness** — regression-test answer quality on a held-out Q&A set
   (and measure refusal precision/recall), so changes are measurable not vibes.
5. **Larger base model (1.5B+)** if quality ceilings are hit after RAG.

---

## 8. Where to start as a new developer

1. Run it: [USER_GUIDE.md](USER_GUIDE.md) (5 minutes to first answer).
2. Read the code tour: [CODE_OVERVIEW.md](CODE_OVERVIEW.md) — start with
   `src/rag.py` (grounded default) then `src/generation.py` (fine-tuned path).
3. Skim `notebooks/instruction_finetuning.ipynb` to see where the prompt
   format contract comes from.
4. Deployment: [AZURE_DEPLOYMENT.md](AZURE_DEPLOYMENT.md).
