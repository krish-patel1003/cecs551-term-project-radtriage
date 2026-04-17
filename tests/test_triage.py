from __future__ import annotations

from src.constants import CLASSES
from src.triage.scoring import score_case


def test_triage_routine_for_low_probs():
    probs = {c: 0.01 for c in CLASSES}
    out = score_case(probs)
    assert out["urgency_tier"] == "Routine"


def test_triage_emergent_for_pneumothorax_override():
    probs = {c: 0.05 for c in CLASSES}
    probs["Pneumothorax"] = 0.9
    out = score_case(probs)
    assert out["urgency_tier"] == "Emergent"


def test_triage_urgent_for_moderate_case():
    probs = {c: 0.01 for c in CLASSES}
    probs["Infiltration"] = 0.5
    probs["Effusion"] = 0.5
    out = score_case(probs)
    assert out["urgency_tier"] in {"Urgent", "Emergent"}
