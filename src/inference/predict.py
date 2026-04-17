from __future__ import annotations

import torch

from src.constants import CLASSES


@torch.no_grad()
def predict_probs(model, input_tensor: torch.Tensor, device: torch.device) -> dict:
    model.eval()
    logits = model(input_tensor.to(device))
    probs = torch.sigmoid(logits)[0].detach().cpu().tolist()
    return {cls: float(probs[i]) for i, cls in enumerate(CLASSES)}
