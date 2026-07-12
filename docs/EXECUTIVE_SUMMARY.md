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

**Deliverables:** a chat UI (Streamlit), a CLI, a Python API (`HRAssistant`),
container images for Azure, and a test suite.

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
the roadmap adds retrieval grounding (§6), which shifts the factual burden from
the model's weights to retrieved documents. If we needed deeper reasoning
without retrieval, we'd step up to 1.5B–7B — at proportionally higher serving
cost.

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

```
question ──► build_prompt (exact training format)
        ──► model.generate() — 3 sampled candidates (transformers + peft)
        ──► reranker: penalize code-like output, reward numbered steps
        ──► guard: deterministic retry if the winner still looks like code
        ──► answer
```

- Runs on **CPU, CUDA, or MPS** (auto-detected; `HR_DEVICE` overrides).
- A **fallback mode** serves deterministic template answers with zero ML
  dependencies — used for demos, CI, and the slim Azure container.
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

## 6. Current status & roadmap

**Working today** (verified on `main`): coherent HR answers in ~4–12 s on Apple
Silicon, 23 passing tests, Azure-deployable containers (fallback + CPU model
serving).

**Known limits:** answers come from 120 training pairs — the model is fluent
but can be generic outside them; it cannot cite sources; DPO artifacts were
never exported, so the preference-alignment stage isn't in the served model.

**Roadmap, in order of impact:**
1. ~~**Retrieval grounding (RAG)**~~ — ✅ **shipped** (`src/rag.py`): policy
   passages + Q&A pairs are embedded and retrieved per question; answers are
   constrained to retrieved text with citations; a retrieval-score threshold
   refuses out-of-corpus questions (e.g. sick leave — which the policy data
   never covers) instead of letting the model improvise. Grounded mode is the
   app default. Pattern proven first in the sibling
   [wiki-rag-assistant](https://github.com/raosinga6/wiki-rag-assistant) project.
2. **Export and serve the DPO-aligned model** (stage 3 artifacts).
3. **Evaluation harness** — regression-test answer quality on a held-out Q&A
   set, so model changes are measurable rather than vibes-based.
4. **Larger base model (1.5B+)** if quality ceilings are hit after RAG.

---

## 7. Where to start as a new developer

1. Run it: [USER_GUIDE.md](USER_GUIDE.md) (5 minutes to first answer).
2. Read the code tour: [CODE_OVERVIEW.md](CODE_OVERVIEW.md) — start with
   `src/generation.py` (~160 lines, the whole inference story).
3. Skim `notebooks/instruction_finetuning.ipynb` to see where the prompt
   format contract comes from.
4. Deployment: [AZURE_DEPLOYMENT.md](AZURE_DEPLOYMENT.md).
