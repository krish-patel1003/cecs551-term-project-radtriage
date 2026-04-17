from __future__ import annotations

import torch

from src.models.densenet import RadTriageDenseNet
from src.models.losses import build_bce_with_pos_weight


def test_model_forward_shape():
    model = RadTriageDenseNet(num_classes=14, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (2, 14)


def test_loss_backward():
    model = RadTriageDenseNet(num_classes=14, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    y = torch.randint(0, 2, (2, 14)).float()
    logits = model(x)
    criterion = build_bce_with_pos_weight(pos_weight=None, device=torch.device("cpu"))
    loss = criterion(logits, y)
    loss.backward()
    assert float(loss.item()) > 0.0
