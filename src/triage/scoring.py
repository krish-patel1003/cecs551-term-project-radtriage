from __future__ import annotations

from typing import Iterable

from src.constants import CLASSES
from src.triage.routing import map_score_to_tier
from src.triage.severity_weights import SEVERITY_WEIGHTS


def probs_to_dict(probs: dict | Iterable[float]) -> dict:
    if isinstance(probs, dict):
        return {k: float(v) for k, v in probs.items()}
    values = list(probs)
    return {cls: float(values[i]) for i, cls in enumerate(CLASSES)}


def urgency_score(probs: dict | Iterable[float], cap_at_one: bool = True) -> float:
    p = probs_to_dict(probs)
    score = sum(
        float(p.get(cls, 0.0)) * float(SEVERITY_WEIGHTS[cls]) for cls in CLASSES
    )
    return min(score, 1.0) if cap_at_one else score


def apply_override_rules(probs: dict, fallback_tier: str) -> str:
    if probs.get("Pneumothorax", 0.0) >= 0.60:
        return "Emergent"
    if probs.get("Edema", 0.0) >= 0.70 or probs.get("Pneumonia", 0.0) >= 0.70:
        return "Urgent"
    return fallback_tier


def score_case(probs: dict | Iterable[float], image_id: str = "") -> dict:
    p = probs_to_dict(probs)
    score = urgency_score(p)
    mapped = map_score_to_tier(score)
    tier = apply_override_rules(p, mapped)
    top_findings = sorted(p.items(), key=lambda x: x[1], reverse=True)[:3]
    confidence = float(top_findings[0][1]) if top_findings else 0.0

    return {
        "image_id": image_id,
        "probs": p,
        "top_findings": top_findings,
        "urgency_score": float(score),
        "urgency_tier": tier,
        "confidence": confidence,
    }
