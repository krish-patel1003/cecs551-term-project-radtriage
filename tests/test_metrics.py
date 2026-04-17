from __future__ import annotations

import numpy as np

from src.eval.metrics import compute_all_metrics, confusion_stats


def test_metrics_ranges():
    y_true = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=int)
    y_prob = np.array([[0.9, 0.2], [0.1, 0.8], [0.7, 0.7], [0.3, 0.2]], dtype=float)

    y_true = np.pad(y_true, ((0, 0), (0, 12)))
    y_prob = np.pad(y_prob, ((0, 0), (0, 12)))

    metrics = compute_all_metrics(y_true, y_prob, thresholds=0.5)
    assert isinstance(metrics["macro_auc"], float)
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert 0.0 <= metrics["micro_f1"] <= 1.0


def test_sensitivity_specificity_formula():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0])
    sens, spec = confusion_stats(y_true, y_pred)
    assert np.isclose(sens, 0.5)
    assert np.isclose(spec, 0.5)
