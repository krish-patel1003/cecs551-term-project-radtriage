from __future__ import annotations

import numpy as np
import torch

from src.eval.metrics import compute_all_metrics


@torch.no_grad()
def validate(model, val_loader, criterion, device, thresholds=None) -> dict:
    model.eval()
    running_loss = 0.0
    total = 0
    y_true, y_prob = [], []

    for batch in val_loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        probs = torch.sigmoid(logits)

        bs = x.size(0)
        running_loss += loss.item() * bs
        total += bs
        y_true.append(y.detach().cpu().numpy())
        y_prob.append(probs.detach().cpu().numpy())

    y_true_np = np.concatenate(y_true, axis=0) if y_true else np.empty((0, 14))
    y_prob_np = np.concatenate(y_prob, axis=0) if y_prob else np.empty((0, 14))
    metrics = compute_all_metrics(y_true_np, y_prob_np, thresholds=thresholds)
    metrics["loss"] = running_loss / max(1, total)
    return metrics
