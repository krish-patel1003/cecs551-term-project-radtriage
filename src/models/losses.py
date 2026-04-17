from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def build_bce_with_pos_weight(
    pos_weight: np.ndarray | None, device: torch.device
) -> nn.Module:
    if pos_weight is None:
        return nn.BCEWithLogitsLoss()
    tensor = torch.tensor(pos_weight, dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=tensor)
