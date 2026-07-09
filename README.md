# HR Domain-Specific AI Assistant

A complete HR AI assistant built by fine-tuning Qwen2.5-0.5B using Unsloth with a three-stage pipeline: Non-Instruction Fine-Tuning → Instruction Fine-Tuning (SFT) → DPO Preference Alignment.

## Project Overview

| Aspect | Details |
|--------|---------|
| **Domain** | Human Resources (HR Policies) |
| **Base Model** | Qwen2.5-0.5B (Unsloth optimized) |
| **Fine-Tuning Method** | QLoRA (4-bit quantization + LoRA) |
| **Stages** | 3 (Domain Adaptation → SFT → DPO) |
| **Framework** | Unsloth + Hugging Face Transformers + TRL |

## Business Problem

Building a domain-specific HR assistant that:
- Understands HR terminology (leave, reimbursement, WFH, compliance, etc.)
- Provides accurate, company-specific policy answers
- Outperforms generic base models on HR tasks
- Maintains professional tone and safety standards

## Dataset

**Source**: `strova-ai/hr-policies-qa-dataset` (644 Q&A pairs from Hugging Face)

### Derived Datasets
| Dataset | Size | Purpose |
|---------|------|---------|
| Non-Instruction | 60 paragraphs | Domain adaptation (raw HR text) |
| Instruction (SFT) | 120 Q&A pairs | Learn to answer HR questions |
| Preference (DPO) | 60 chosen/rejected pairs | Align responses to preferences |

## Three-Stage Fine-Tuning Pipeline

### Stage 1: Non-Instruction Fine-Tuning (Domain Adaptation)
- **Goal**: Adapt model to HR domain language and terminology
- **Data**: 60 paragraphs from HR policies
- **Method**: QLoRA on raw text (causal LM)
- **Notebook**: `notebooks/non_instruction_finetuning.ipynb`
- **Output**: `models/non_instruction_ft_adapter/`

### Stage 2: Instruction Fine-Tuning (SFT)
- **Goal**: Teach model to answer HR questions
- **Data**: 120 instruction-response pairs (Alpaca format)
- **Method**: QLoRA on formatted Q&A
- **Notebook**: `notebooks/instruction_finetuning.ipynb`
- **Output**: `models/instruction_ft_adapter/` + `models/instruction_ft_merged/`

### Stage 3: DPO Preference Alignment
- **Goal**: Improve response quality, tone, completeness
- **Data**: 60 preference pairs (chosen vs rejected)
- **Method**: DPO with β=0.1, reference = SFT model
- **Notebook**: `notebooks/dpo_alignment.ipynb`
- **Output**: `models/dpo_aligned_adapter/` + `models/dpo_aligned_merged/`

## LoRA/QLoRA Configuration

```python
# Shared across all stages
r = 16                    # Rank
lora_alpha = 32           # Alpha (scaling)
lora_dropout = 0.05       # Dropout
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
quantization = "4bit"     # QLoRA
```

### Training Hyperparameters

| Parameter | Stage 1 | Stage 2 | Stage 3 |
|-----------|---------|---------|---------|
| Learning Rate | 2e-4 | 2e-4 | 5e-5 |
| Max Steps | 100 | 200 | 100 |
| Batch Size | 2 | 2 | 2 |
| Grad Accum | 4 | 4 | 4 |
| Warmup Steps | 10 | 20 | 10 |
| Optimizer | AdamW 8-bit | AdamW 8-bit | AdamW 8-bit |
| Seq Length | 512 | 512 | 512 |

## Model Evaluation

### Test Questions (10 HR Domain Questions)
1. How can I apply for sick leave?
2. What is the work from home policy?
3. How does reimbursement work?
4. What is the notice period for resignation?
5. What employee benefits are available?
6. How is overtime calculated?
7. What is the onboarding process?
8. How do I report a compliance concern?
9. What is the attendance policy?
10. How are performance reviews conducted?

### Results Summary

| Model | Avg Score (1-5) | Domain Accuracy | Helpfulness |
|-------|-----------------|-----------------|-------------|
| Base (Qwen2.5-0.5B) | 2.25 | 1.0 | 1.0 |
| SFT | 4.38 | 5.0 | 5.0 |
| **DPO (Final)** | **5.0** | **5.0** | **5.0** |

**Key Finding**: DPO alignment provides consistent improvement over SFT across all criteria, especially tone, clarity, and professional quality.

### Sample Comparison

**Q: How can I apply for sick leave?**

| Model | Response |
|-------|----------|
| **Base** | You should contact your employer or HR department for sick leave procedures. |
| **SFT** | You can apply for sick leave through the HR portal by selecting the sick leave option and submitting the required details. |
| **DPO** | To apply for sick leave, log into the HR portal, navigate to the Leave section, select "Sick Leave," fill in the required details (dates, reason), and submit for manager approval. You'll receive a confirmation once approved. |

## Repository Structure

```
hr-ai-assistant/
│
├── data/
│   ├── non_instruction_data.txt      # 60 HR paragraphs
│   ├── instruction_dataset.jsonl     # 120 Q&A pairs
│   └── preference_dataset.jsonl      # 60 preference pairs
│
├── notebooks/
│   ├── non_instruction_finetuning.ipynb
│   ├── instruction_finetuning.ipynb
│   └── dpo_alignment.ipynb
│
├── reports/
│   ├── base_model_evaluation.md
│   ├── sft_model_comparison.md
│   ├── final_evaluation.md
│   └── fine_tuning_explanation.md
│
├── src/
│   ├── inference.py                  # Inference script
│   └── app.py                        # Streamlit web UI
│
├── models/                           # Generated (not in repo)
│   ├── non_instruction_ft_adapter/
│   ├── instruction_ft_adapter/
│   ├── instruction_ft_merged/
│   ├── dpo_aligned_adapter/
│   └── dpo_aligned_merged/
│
├── requirements.txt
└── README.md
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# For GPU training (requires CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Requirements

```
unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git
transformers>=4.36.0
datasets>=2.16.0
accelerate>=0.25.0
peft>=0.7.0
trl>=0.7.0
bitsandbytes>=0.41.0
streamlit>=1.28.0
```

## Running the Notebooks

### Option 1: Google Colab (Free GPU)
1. Upload notebooks to Colab
2. Enable GPU: Runtime → Change runtime type → GPU (T4)
3. Run cells sequentially

### Option 2: Local GPU
```bash
jupyter lab notebooks/
# Run each notebook in order
```

### Option 3: Cloud (RunPod, Lambda, etc.)
- Rent GPU instance (RTX 3090/4090, A100)
- Clone repo, install requirements, run notebooks

## Inference

### Using Python Script
```bash
python src/inference.py --model_path models/dpo_aligned_merged --question "How do I apply for leave?"
```

### Using Web UI
```bash
streamlit run src/app.py
```
Then open http://localhost:8501

## Challenges Faced

1. **Small Dataset Size**: Only 644 source examples - mitigated by data augmentation and careful curation
2. **GPU Memory**: Used QLoRA 4-bit to fit on consumer GPUs
3. **Training Stability**: Lower learning rate for DPO (5e-5 vs 2e-4) prevented divergence
4. **Evaluation**: No automated metrics for HR domain - used human evaluation criteria

## Future Improvements

- [ ] Expand dataset with more diverse HR scenarios
- [ ] Add RAG for real-time policy document retrieval
- [ ] Implement multi-turn conversation support
- [ ] Add evaluation benchmarks (HR-specific benchmarks)
- [ ] Try larger base models (Llama-3.2-3B, Qwen2.5-3B)
- [ ] Experiment with ORPO instead of DPO
- [ ] Add safety guardrails for sensitive HR topics
- [ ] Deploy as API with FastAPI

## Training Logs (Example)

```
# Stage 1: Non-Instruction FT
Step 10:  loss=2.341
Step 50:  loss=1.892
Step 100: loss=1.654

# Stage 2: Instruction FT
Step 25:  loss=2.103
Step 100: loss=1.456
Step 200: loss=1.234

# Stage 3: DPO
Step 20:  loss=0.567
Step 50:  loss=0.423
Step 100: loss=0.389
```

## License

MIT License - Feel free to use for learning and research.

## Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) for efficient fine-tuning
- [Hugging Face](https://huggingface.co) for transformers and datasets
- [strova-ai](https://huggingface.co/strova-ai) for HR policies dataset
- [TRL](https://github.com/huggingface/trl) for DPO implementation