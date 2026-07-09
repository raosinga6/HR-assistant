import os
import sys

# Ensure project src is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.inference import HRAssistant


def test_fallback_sick_leave():
    assistant = HRAssistant(fallback=True)
    resp = assistant.generate_answer("How can I apply for sick leave?")
    assert "doctor" in resp.lower() or "sick" in resp.lower()


def test_fallback_policy():
    assistant = HRAssistant(fallback=True)
    resp = assistant.generate_answer("What is the work from home policy?")
    assert "policy" in resp.lower() or "work" in resp.lower()


def test_fallback_sick_leave_is_numbered_steps():
    assistant = HRAssistant(fallback=True)
    resp = assistant.generate_answer("How can I apply for sick leave?")
    # Sick-leave branch returns an actionable, numbered procedure.
    assert resp.strip().startswith("1.")
    assert "manager" in resp.lower()


def test_fallback_generic_actionable_branch():
    # Actionable but not sick-leave -> generic step-by-step, not the policy blurb.
    assistant = HRAssistant(fallback=True)
    resp = assistant.generate_answer("How do I submit a reimbursement request?")
    assert resp.strip().startswith("1.")
    assert "manager" in resp.lower()
    assert "policy summary" not in resp.lower()


def test_fallback_policy_branch_for_non_actionable():
    # No actionable keyword -> policy-summary template.
    assistant = HRAssistant(fallback=True)
    resp = assistant.generate_answer("What employee benefits are available?")
    assert "policy summary" in resp.lower()


def test_fallback_does_not_load_a_model():
    # Fallback mode must never touch the model/tokenizer.
    assistant = HRAssistant(fallback=True)
    assert assistant.model is None
    assert assistant.tokenizer is None
