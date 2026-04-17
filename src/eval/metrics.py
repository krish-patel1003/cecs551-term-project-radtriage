from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.constants import CLASSES


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return float("nan")


def confusion_stats(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    tp = float(((y_true == 1) & (y_pred == 1)).sum())
    tn = float(((y_true == 0) & (y_pred == 0)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    fn = float(((y_true == 1) & (y_pred == 0)).sum())
    sensitivity = tp / (tp + fn + 1e-12)
    specificity = tn / (tn + fp + 1e-12)
    return sensitivity, specificity


def binarize(y_prob: np.ndarray, thresholds) -> np.ndarray:
    if np.isscalar(thresholds):
        return (y_prob >= float(thresholds)).astype(np.int64)
    thr = np.asarray(thresholds, dtype=np.float32).reshape(1, -1)
    return (y_prob >= thr).astype(np.int64)


def compute_all_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, thresholds=None
) -> dict:
    if thresholds is None:
        thresholds = 0.5
    y_pred = binarize(y_prob, thresholds)

    per_class = []
    aucs = []
    sensitivities = []
    specificities = []

    for i, cls in enumerate(CLASSES):
        auc = _safe_auc(y_true[:, i], y_prob[:, i])
        pr_auc = (
            float(average_precision_score(y_true[:, i], y_prob[:, i]))
            if len(np.unique(y_true[:, i])) > 1
            else float("nan")
        )
        sens, spec = confusion_stats(y_true[:, i], y_pred[:, i])
        prec = float(precision_score(y_true[:, i], y_pred[:, i], zero_division=0))
        rec = float(recall_score(y_true[:, i], y_pred[:, i], zero_division=0))
        f1 = float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0))

        per_class.append(
            {
                "class": cls,
                "roc_auc": auc,
                "pr_auc": pr_auc,
                "sensitivity": sens,
                "specificity": spec,
                "precision": prec,
                "recall": rec,
                "f1": f1,
            }
        )
        aucs.append(auc)
        sensitivities.append(sens)
        specificities.append(spec)

    macro_auc = float(np.nanmean(aucs))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))

    return {
        "macro_auc": macro_auc,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "mean_sensitivity": float(np.mean(sensitivities)),
        "mean_specificity": float(np.mean(specificities)),
        "per_class": per_class,
    }
