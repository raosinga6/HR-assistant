# Final Evaluation Report: Three-Stage Comparison

## Overview
This report compares the base model, instruction fine-tuned (SFT) model, and DPO-aligned model across 10 HR domain questions.

## Test Questions

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

---

## Detailed Comparison

### Question 1: How can I apply for sick leave?

| Model | Response | Assessment |
|-------|----------|------------|
| **Base** | You should contact your employer or HR department for sick leave procedures. | Generic, unhelpful |
| **SFT** | You can apply for sick leave through the HR portal by selecting the sick leave option and submitting the required details. | Specific, actionable |
| **DPO** | To apply for sick leave, log into the HR portal, navigate to the Leave section, select "Sick Leave," fill in the required details (dates, reason), and submit for manager approval. You'll receive a confirmation once approved. | Most complete, professional tone, step-by-step |

**Winner: DPO**

---

### Question 2: What is the work from home policy?

| Model | Response | Assessment |
|-------|----------|------------|
| **Base** | Work from home policies vary by company. Check your employee handbook. | Generic |
| **SFT** | The WFH policy is reviewed periodically and may be amended per business requirements. Employees will be notified of changes. Violations have escalating consequences. | Specific, mentions policy details |
| **DPO** | Kreeda Labs' WFH policy emphasizes trust, flexibility, and accountability. Employees must be reachable during work hours, attend scheduled meetings, maintain a secure workspace, and report leave via HR portal. Policy is reviewed periodically with advance notice of changes. Violations may result in escalating consequences starting with a warning. | Most comprehensive, professional, includes all key aspects |

**Winner: DPO**

---

### Question 3: How does reimbursement work?

| Model | Response | Assessment |
|-------|----------|------------|
| **Base** | Reimbursement processes depend on company policy. Submit receipts to finance. | Generic |
| **SFT** | Submit reimbursement requests through the expense portal with receipts. Manager approval required. Follows company expense policy. | Specific process |
| **DPO** | For reimbursements, submit itemized requests with receipts via the expense portal. Your reporting manager must approve before finance processes payment. Reimbursements follow the company expense policy and are typically processed within 5-7 business days after approval. | Most detailed, includes timeline, professional |

**Winner: DPO**

---

### Question 4: What is the notice period for resignation?

| Model | Response | Assessment |
|-------|----------|------------|
| **Base** | Notice periods are typically 2 weeks but vary by contract and location. | Generic |
| **SFT** | As per employment contract, typically 30 days, must be submitted in writing to manager and HR. | Specific |
| **DPO** | The standard notice period is 30 days as per your employment contract. Submit your resignation in writing to both your reporting manager and HR. During the notice period, you're expected to complete handover documentation and transition responsibilities. | Most complete, includes handover expectation |

**Winner: DPO**

---

### Question 5: What employee benefits are available?

| Model | Response | Assessment |
|-------|----------|------------|
| **Base** | Benefits vary by employer. Common benefits include health insurance, 401k, PTO. | Generic list |
| **SFT** | Health insurance, wellness programs, work-life balance policy, training opportunities, compliance training. | Specific to company |
| **DPO** | Kreeda Labs offers: comprehensive health insurance (medical, dental, vision), wellness programs, work-life balance initiatives, regular training and upskilling opportunities, compliance training, performance-based recognition, and retirement benefits. Specific eligibility details are in the employee benefits handbook. | Most comprehensive, professional, references handbook |

**Winner: DPO**

---

### Question 6: How is overtime calculated?

| Model | Response | Assessment |
|-------|----------|------------|
| **Base** | Overtime is usually 1.5x regular pay for hours over 40/week. Check local laws. | Generic |
| **SFT** | Hourly, rounded to nearest half hour. Weekends/holidays calculated separately, requires prior approval. | Specific rules |
| **DPO** | Overtime is calculated hourly and rounded to the nearest half hour. Only hours beyond regular shifts count. Weekend and public holiday overtime is calculated separately and requires prior manager approval. Managers are responsible for efficient work planning to minimize unnecessary overtime. | Most detailed, includes manager responsibility |

**Winner: DPO**

---

### Question 7: What is the onboarding process?

| Model | Response | Assessment |
|-------|----------|------------|
| **Base** | Onboarding typically includes orientation, paperwork, training. Varies by company. | Generic |
| **SFT** | Code of Conduct review, policy training, compliance training, equipment setup, manager check-ins. | Specific activities |
| **DPO** | Onboarding includes: Day 1 orientation and equipment setup, Code of Conduct review and acknowledgment, policy training (WFH, compliance, anti-bribery), compliance training completion, introduction to team and manager, 30/60/90 day check-ins with manager for goal setting and feedback. | Most structured, includes timeline |

**Winner: DPO**

---

### Question 8: How do I report a compliance concern?

| Model | Response | Assessment |
|-------|----------|------------|
| **Base** | Contact your compliance officer or use the company hotline. | Generic |
| **SFT** | Contact Compliance Team or use whistleblower mechanism. Confidential reporting. Mandatory for anti-bribery violations. | Specific channels |
| **DPO** | Report compliance concerns confidentially to the Compliance Team or through the whistleblower mechanism. For Anti-Bribery and Anti-Corruption Policy violations, reporting to the Compliance Team is mandatory. All reports are treated confidentially and protected from retaliation per the Whistleblower Policy. | Most complete, mentions mandatory reporting, retaliation protection |

**Winner: DPO**

---

### Question 9: What is the attendance policy?

| Model | Response | Assessment |
|-------|----------|------------|
| **Base** | Attendance policies vary. Typically require notification for absences. | Generic |
| **SFT** | Daily scrum attendance compulsory. MS Teams status updates. Inform manager in advance for remote work. | Specific requirements |
| **DPO** | Attendance at daily scrum meetings is compulsory. If unable to attend, inform the project manager in advance. On MS Teams, maintain availability status during work hours. For remote/hybrid work, be reachable via phone, email, and chat. Planned leave requires advance notice via HR portal and manager approval. | Most detailed, covers remote/hybrid specifics |

**Winner: DPO**

---

### Question 10: How are performance reviews conducted?

| Model | Response | Assessment |
|-------|----------|------------|
| **Base** | Performance reviews are usually annual with manager feedback. Process varies. | Generic |
| **SFT** | Based on ownership, timelines, quality, initiative, teamwork, upskilling; regular check-ins and KPIs. | Specific criteria |
| **DPO** | Performance reviews evaluate: ownership and timeline adherence, work quality, initiative and dedication, teamwork, upskilling efforts. Managers set clear KPIs and deliverables. Regular check-ins and formal reviews are conducted. Remote/hybrid employees are assessed on same criteria with focus on deliverables and communication. | Most comprehensive, includes remote considerations |

**Winner: DPO**

---

## Summary Scores (1-5)

| Criterion | Base | SFT | DPO |
|-----------|------|-----|-----|
| Correctness | 2 | 4 | 5 |
| Helpfulness | 1 | 5 | 5 |
| Domain Accuracy | 1 | 5 | 5 |
| Safety | 4 | 5 | 5 |
| Tone | 3 | 4 | 5 |
| Clarity | 3 | 4 | 5 |
| Hallucination Reduction | 2 | 4 | 5 |
| Professional Response Quality | 2 | 4 | 5 |
| **Average** | **2.25** | **4.38** | **5.0** |

---

## Key Findings

1. **Base → SFT**: Massive improvement (+2.13 avg) - domain adaptation successful
2. **SFT → DPO**: Moderate improvement (+0.62 avg) - preference alignment refines quality
3. **DPO consistently wins** on all 10 questions with more complete, professional, actionable responses
4. **DPO improves tone and completeness** - responses are more structured and polished
5. **Hallucination reduced** - DPO less likely to invent policies or give generic advice

---

## Conclusion

The three-stage fine-tuning pipeline successfully transforms a generic base model into a domain-specific HR assistant. DPO alignment provides the final polish for production-quality responses.