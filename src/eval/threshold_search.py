from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from src.constants import CLASSES, THRESHOLD_GRID


def _youden_j(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = float(((y_true == 1) & (y_pred == 1)).sum())
    tn = float(((y_true == 0) & (y_pred == 0)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    fn = float(((y_true == 1) & (y_pred == 0)).sum())
    sens = tp / (tp + fn + 1e-12)
    spec = tn / (tn + fp + 1e-12)
    return sens + spec - 1.0


def search_thresholds(
    y_true: np.ndarray, y_prob: np.ndarray, objective: str = "f1"
) -> dict:
    thresholds = {}
    for i, cls in enumerate(CLASSES):
        best_thr = 0.5
        best_score = -1.0
        for thr in THRESHOLD_GRID:
            pred = (y_prob[:, i] >= thr).astype(int)
            score = (
                float(f1_score(y_true[:, i], pred, zero_division=0))
                if objective == "f1"
                else _youden_j(y_true[:, i], pred)
            )
            if score > best_score:
                best_score = score
                best_thr = float(thr)
        thresholds[cls] = best_thr
    return thresholds


def save_thresholds(thresholds: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(thresholds, indent=2), encoding="utf-8")


def load_thresholds(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
