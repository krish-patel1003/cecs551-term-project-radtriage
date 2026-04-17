from __future__ import annotations


def map_score_to_tier(
    score: float, emergent_thr: float = 0.75, urgent_thr: float = 0.4
) -> str:
    if score >= emergent_thr:
        return "Emergent"
    if score >= urgent_thr:
        return "Urgent"
    return "Routine"
