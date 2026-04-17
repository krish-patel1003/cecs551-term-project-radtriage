from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from src.explainability.gradcam import GradCAM, overlay_heatmap
from src.models.densenet import RadTriageDenseNet


def test_gradcam_output_and_overlay(tmp_path: Path):
    model = RadTriageDenseNet(num_classes=14, pretrained=False)
    cam_generator = GradCAM(model)

    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    cam = cam_generator.generate(x, class_idx=7)

    assert cam.shape == (224, 224)
    assert cam.min() >= 0.0
    assert cam.max() <= 1.0

    original = Image.new("RGB", (224, 224), color=(128, 128, 128))
    overlay = overlay_heatmap(original, cam)
    out_path = tmp_path / "overlay.png"
    overlay.save(out_path)
    assert out_path.exists()
