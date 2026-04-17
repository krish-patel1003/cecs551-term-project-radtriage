from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.inference.pipeline import RadTriagePipeline
from src.models.densenet import RadTriageDenseNet


def test_inference_pipeline_keys(tmp_path: Path):
    arr = (np.random.rand(256, 256) * 255).astype("uint8")
    img_path = tmp_path / "sample.png"
    Image.fromarray(arr, mode="L").save(img_path)

    model = RadTriageDenseNet(num_classes=14, pretrained=False)
    pipeline = RadTriagePipeline(
        model=model, device="cpu", heatmap_dir=str(tmp_path / "heatmaps")
    )
    out = pipeline.predict_single(str(img_path))

    assert "probs" in out
    assert "urgency_score" in out
    assert "urgency_tier" in out
    assert "top_findings" in out
