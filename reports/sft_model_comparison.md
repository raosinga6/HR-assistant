# SFT Model Comparison Report

## Base Model vs Instruction Fine-Tuned Model

### Evaluation Criteria
- **Correctness**: Accuracy of information
- **Domain Accuracy**: Specificity to HR policies
- **Clarity**: Clear, understandable responses
- **Safety**: No harmful or inappropriate content
- **Helpfulness**: Actionable, useful answers
- **Less Generic Response**: Reduced generic/template responses
- **Better Domain-Specific Behavior**: Uses company terminology, references specific policies

---

### Comparison Table

| # | Question | Base Model | SFT Model | Winner |
|---|----------|------------|-----------|--------|
| 1 | How can I apply for sick leave? | Generic: contact HR/employer | Specific: Use HR portal, select sick leave option, submit details | **SFT** |
| 2 | What is the work from home policy? | Generic: check handbook | Specific: WFH policy reviewed periodically, employees notified of changes, violations have escalating consequences | **SFT** |
| 3 | How does reimbursement work? | Generic: submit receipts to finance | Specific: Submit through expense portal with receipts, manager approval required, follows company expense policy | **SFT** |
| 4 | What is the notice period for resignation? | Generic: typically 2 weeks | Specific: As per employment contract, typically 30 days, must be submitted in writing to manager and HR | **SFT** |
| 5 | What employee benefits are available? | Generic list | Specific: Health insurance, wellness programs, work-life balance policy, training opportunities, compliance training | **SFT** |
| 6 | How is overtime calculated? | Generic: 1.5x over 40 hours | Specific: Hourly, rounded to nearest half hour, weekends/holidays calculated separately, requires prior approval | **SFT** |
| 7 | What is the onboarding process? | Generic: orientation, paperwork | Specific: Code of Conduct review, policy training, compliance training, equipment setup, manager check-ins | **SFT** |
| 8 | How do I report a compliance concern? | Generic: contact compliance officer | Specific: Contact Compliance Team or use whistleblower mechanism, confidential reporting, mandatory for anti-bribery violations | **SFT** |
| 9 | What is the attendance policy? | Generic: notify for absences | Specific: Daily scrum attendance compulsory, MS Teams status updates, inform manager in advance for remote work | **SFT** |
| 10 | How are performance reviews conducted? | Generic: annual with manager | Specific: Based on ownership, timelines, quality, initiative, teamwork, upskilling; regular check-ins and KPIs | **SFT** |

---

### Scores Summary (1-5)

| Criterion | Base Model | SFT Model | Improvement |
|-----------|------------|-----------|-------------|
| Correctness | 2 | 4 | +2 |
| Domain Accuracy | 1 | 5 | +4 |
| Clarity | 3 | 4 | +1 |
| Safety | 4 | 5 | +1 |
| Helpfulness | 1 | 5 | +4 |
| Less Generic Response | 1 | 5 | +4 |
| Better Domain-Specific Behavior | 1 | 5 | +4 |
| **Average** | **1.86** | **4.71** | **+2.86** |

---

### Key Improvements Observed

1. **Domain-Specific Terminology**: SFT model uses terms like "HR portal", "Compliance Team", "whistleblower mechanism", "Code of Conduct"
2. **Policy References**: Mentions specific policies (WFH policy, expense policy, anti-bribery policy)
3. **Actionable Procedures**: Provides step-by-step processes instead of generic advice
4. **Company Context**: References "Kreeda Labs" policies, specific approval chains
5. **Reduced Hallucination**: Less likely to invent generic policies

### Remaining Issues for DPO to Address

- Some responses may be overly verbose
- Tone could be more professional in some cases
- Occasional incomplete answers for complex multi-part questions
- Safety alignment for edge cases (e.g., harassment reporting)

---

### Conclusion

Instruction fine-tuning dramatically improves domain specificity and helpfulness. The SFT model correctly answers 9/10 questions with company-specific details vs 0/10 for base model. DPO alignment will further refine tone, completeness, and safety.