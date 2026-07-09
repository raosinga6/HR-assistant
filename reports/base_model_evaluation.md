# Base Model Evaluation Report

## Test Questions and Base Model Responses

| # | Question | Base Model Response | Issues Identified |
|---|----------|-------------------|-------------------|
| 1 | How can I apply for sick leave? | You should contact your employer or HR department for sick leave procedures. | Generic, not specific to company policy |
| 2 | What is the work from home policy? | Work from home policies vary by company. Check your employee handbook. | Generic, no domain-specific details |
| 3 | How does reimbursement work? | Reimbursement processes depend on company policy. Submit receipts to finance. | Generic, lacks specific process details |
| 4 | What is the notice period for resignation? | Notice periods are typically 2 weeks but vary by contract and location. | Generic, not company-specific |
| 5 | What employee benefits are available? | Benefits vary by employer. Common benefits include health insurance, 401k, PTO. | Generic list, not specific to this company |
| 6 | How is overtime calculated? | Overtime is usually 1.5x regular pay for hours over 40/week. Check local laws. | Generic, doesn't mention company-specific rules |
| 7 | What is the onboarding process? | Onboarding typically includes orientation, paperwork, training. Varies by company. | Generic, not specific |
| 8 | How do I report a compliance concern? | Contact your compliance officer or use the company hotline. | Generic, doesn't mention specific process |
| 9 | What is the attendance policy? | Attendance policies vary. Typically require notification for absences. | Generic |
| 10 | How are performance reviews conducted? | Performance reviews are usually annual with manager feedback. Process varies. | Generic |

## Evaluation Summary

**Overall Assessment**: The base model (Qwen2.5-0.5B) provides generic, non-specific responses that lack:
- Company-specific policy details
- HR domain terminology
- Actionable procedures
- References to specific policy documents

**Scores (1-5)**:
- Correctness: 2/5 (generic but not wrong)
- Domain Accuracy: 1/5 (no domain knowledge)
- Clarity: 3/5 (clear but unhelpful)
- Safety: 4/5 (safe but unhelpful)
- Helpfulness: 1/5 (not actionable)
- Less Generic Response: 1/5 (highly generic)
- Better Domain-Specific Behavior: 1/5 (no domain adaptation)

## Conclusion

The base model requires domain adaptation through fine-tuning to become a useful HR assistant.