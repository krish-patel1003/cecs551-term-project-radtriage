from __future__ import annotations

import torch.nn as nn
from torchvision.models import DenseNet121_Weights, densenet121


class RadTriageDenseNet(nn.Module):
    def __init__(
        self, num_classes: int = 14, pretrained: bool = True, dropout: float = 0.2
    ):
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = densenet121(weights=weights)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_features, num_classes)
        )

    def set_freeze_backbone(self, freeze_backbone: bool = True) -> None:
        for _, param in self.backbone.features.named_parameters():
            param.requires_grad = not freeze_backbone

    def forward(self, x):
        return self.backbone(x)
