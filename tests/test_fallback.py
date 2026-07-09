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
