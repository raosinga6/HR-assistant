# Fine-Tuning Explanation Report

## Why Full Fine-Tuning is Expensive

Full fine-tuning updates **all parameters** of a pre-trained model. For a 0.5B parameter model like Qwen2.5-0.5B:
- **Memory**: Requires storing gradients, optimizer states, and activations for all parameters (~6-8 GB for FP16, more for optimizer states)
- **Compute**: Backpropagation through entire network for every batch
- **Storage**: Need to save full model checkpoints (~1 GB per checkpoint)
- **Time**: Slower training due to full gradient computation

For larger models (7B, 13B, 70B), full fine-tuning requires enterprise-grade GPU clusters (A100 80GB, H100) making it inaccessible for most teams.

---

## What LoRA Does

**LoRA (Low-Rank Adaptation)** freezes the pre-trained model weights and injects trainable rank-decomposition matrices into the attention layers.

### How it Works
```
Original: W ∈ ℝ^(d×k)
LoRA:     W + ΔW = W + BA
Where:    B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), r << min(d,k)
```

- **Rank (r)**: Controls the number of trainable parameters. Lower r = fewer parameters but less capacity
- **Alpha (α)**: Scaling factor for LoRA updates. Effective learning rate = α/r × base_lr
- **Dropout**: Applied to LoRA layers for regularization

### Benefits
- **Parameters**: Only 0.1-1% of original parameters trained
- **Memory**: No gradients for frozen weights, no optimizer states for them
- **Storage**: Adapters are tiny (MBs vs GBs)
- **No inference latency**: Can merge weights: W_merged = W + BA

---

## What QLoRA Does

**QLoRA (Quantized LoRA)** combines 4-bit quantization with LoRA for even greater memory efficiency.

### Key Innovations
1. **4-bit NormalFloat (NF4)**: Quantizes weights to 4-bit with optimal distribution for normal weights
2. **Double Quantization**: Quantizes quantization constants (saves ~0.4 bits/parameter)
3. **Paged Optimizers**: Offloads optimizer states to CPU RAM when GPU memory full
4. **LoRA on Quantized Base**: Train LoRA adapters while base model stays in 4-bit

### Memory Comparison (Qwen2.5-0.5B)
| Method | GPU Memory (4-bit) |
|--------|-------------------|
| Full FT (FP16) | ~6 GB |
| LoRA (FP16) | ~3 GB |
| **QLoRA (4-bit)** | **~1.5 GB** |

---

## Why QLoRA is Useful on Limited GPU

- **Consumer GPUs**: Runs on 8GB VRAM (RTX 3070/4070, T4 Colab)
- **Free Tier Colab**: T4 (16GB) can train 1B-3B models with QLoRA
- **Cost**: 10-20x cheaper than full fine-tuning on cloud
- **Speed**: Similar training time to LoRA, faster than full FT
- **Quality**: Near full fine-tuning performance (within 1-2% on benchmarks)

---

## What is Non-Instruction Fine-Tuning?

**Non-instruction fine-tuning** (also called **domain adaptation** or **continued pre-training**) trains the model on raw domain text without instruction/response formatting.

### Purpose
- Learn domain vocabulary, terminology, writing style
- Absorb background knowledge and facts
- Adapt token distributions to domain
- Prepare model for instruction tuning

### Our Implementation
- **Data**: 60 paragraphs from HR policy documents
- **Format**: Raw text chunks (512 tokens, 50 overlap)
- **Objective**: Next-token prediction (causal LM)
- **Result**: Model "speaks HR" before learning to answer questions

---

## What is Instruction Fine-Tuning (SFT)?

**Supervised Fine-Tuning (SFT)** trains the model on instruction-response pairs to learn how to follow instructions and answer questions.

### Purpose
- Learn instruction-following behavior
- Map questions to appropriate answers
- Adopt desired response format and tone
- Specialize for specific use cases (HR Q&A)

### Our Implementation
- **Data**: 120 instruction-response pairs from strova-ai/hr-policies-qa-dataset
- **Format**: Alpaca-style (### Instruction:\n{question}\n\n### Response:\n{answer})
- **Objective**: Next-token prediction on response tokens only
- **Starting point**: Non-instruction fine-tuned model (or base model)
- **Result**: Model answers HR questions correctly

---

## What is DPO?

**Direct Preference Optimization (DPO)** aligns model outputs with human preferences without reinforcement learning.

### How it Works
Given a prompt x, chosen response y_w, and rejected response y_l:
```
L_DPO = -E[log σ(β * (log π_θ(y_w|x) - log π_ref(y_w|x) - log π_θ(y_l|x) + log π_ref(y_l|x)))]
```

- **π_θ**: Policy model (being trained)
- **π_ref**: Reference model (frozen SFT model)
- **β**: Temperature parameter (controls divergence from reference)
- **Implicit reward**: r(x,y) = β * log(π_θ(y|x)/π_ref(y|x))

### Benefits vs RLHF/PPO
- **No reward model needed**: Directly optimizes on preferences
- **No RL training**: Stable, no reward hacking
- **Simpler**: Single-stage training like SFT
- **Efficient**: Same compute as SFT

---

## Difference Between SFT and DPO

| Aspect | SFT | DPO |
|--------|-----|-----|
| **Data** | (instruction, response) | (prompt, chosen, rejected) |
| **Objective** | Maximize likelihood of response | Maximize margin between chosen/rejected |
| **Teaches** | How to answer | Which answer is better |
| **Reference Model** | None needed | Frozen SFT model as reference |
| **Loss** | Cross-entropy | Bradley-Terry preference loss |
| **Output** | Correct answers | Preferred answers (better tone, safety, completeness) |

---

## Hyperparameter Values Used

| Parameter | Non-Instruction FT | Instruction FT (SFT) | DPO |
|-----------|-------------------|---------------------|-----|
| **Base Model** | Qwen2.5-0.5B | Qwen2.5-0.5B (or non-inst FT) | SFT model |
| **Quantization** | 4-bit (QLoRA) | 4-bit (QLoRA) | 4-bit (QLoRA) |
| **LoRA Rank (r)** | 16 | 16 | 16 |
| **LoRA Alpha (α)** | 32 | 32 | 32 |
| **LoRA Dropout** | 0.05 | 0.05 | 0.05 |
| **Learning Rate** | 2e-4 | 2e-4 | 5e-5 |
| **Batch Size** | 2 (effective 8 with grad accum) | 2 (effective 8) | 2 (effective 8) |
| **Max Steps** | 100 | 200 | 100 |
| **Warmup Steps** | 10 | 20 | 10 |
| **Scheduler** | Cosine | Cosine | Cosine |
| **Optimizer** | AdamW 8-bit | AdamW 8-bit | AdamW 8-bit |
| **Weight Decay** | 0.01 | 0.01 | 0.01 |
| **Max Seq Length** | 512 | 512 | 512 |
| **DPO Beta** | N/A | N/A | 0.1 |
| **Target Modules** | q,k,v,o,gate,up,down_proj | q,k,v,o,gate,up,down_proj | q,k,v,o,gate,up,down_proj |
| **Gradient Checkpointing** | Unsloth | Unsloth | Unsloth |

---

## Rationale for Choices

- **Rank 16**: Good balance of capacity vs parameters (~0.5% trainable)
- **Alpha 32**: Standard scaling (α/r = 2), effective LR multiplier
- **Dropout 0.05**: Light regularization for small dataset
- **LR 2e-4**: Standard for LoRA, lower for DPO (5e-5) for stability
- **4-bit QLoRA**: Enables training on consumer GPUs
- **Max steps 100-200**: Small experiment, sufficient for demonstration
- **Beta 0.1**: Conservative DPO, keeps close to SFT reference

---

## Training Pipeline Summary

```
Base Model (Qwen2.5-0.5B)
        │
        ▼
┌───────────────────┐
│ Non-Instruction FT │  ← Domain adaptation (60 HR paragraphs)
│   (QLoRA, 100 steps)│
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Instruction FT   │  ← Learn to answer (120 Q&A pairs)
│    (QLoRA, 200 steps)│
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   DPO Alignment   │  ← Preference alignment (60 pref pairs)
│    (QLoRA, 100 steps)│
└───────────────────┘
        │
        ▼
Final HR Assistant Model
```

Each stage builds on the previous, progressively specializing the model for the HR domain.