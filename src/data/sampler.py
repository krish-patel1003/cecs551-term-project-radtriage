from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

from src.constants import CLASSES


def compute_pos_weight(targets: np.ndarray) -> np.ndarray:
    pos_counts = targets.sum(axis=0)
    neg_counts = len(targets) - pos_counts
    return neg_counts / (pos_counts + 1e-6)


def compute_sample_weights(
    targets: np.ndarray, pos_weight: np.ndarray, no_positive_weight: float | None = None
) -> np.ndarray:
    if no_positive_weight is None:
        no_positive_weight = float(np.clip(np.min(pos_weight) * 0.1, 0.05, 1.0))

    sample_weights = np.full(len(targets), no_positive_weight, dtype=np.float32)
    for i in range(len(targets)):
        positives = np.where(targets[i] == 1)[0]
        if len(positives):
            sample_weights[i] = float(np.max(pos_weight[positives]))
    return sample_weights


def targets_from_dataframe(df) -> np.ndarray:
    return df[CLASSES].values.astype(np.float32)


def build_weighted_sampler(sample_weights: np.ndarray) -> WeightedRandomSampler:
    weights = torch.tensor(sample_weights, dtype=torch.float32)
    return WeightedRandomSampler(
        weights=weights, num_samples=len(weights), replacement=True
    )
