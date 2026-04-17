from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.constants import CLASSES


@pytest.fixture
def sample_split_csv(tmp_path: Path) -> Path:
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    records = []
    rng = np.random.default_rng(0)
    for i in range(8):
        arr = (rng.uniform(0, 255, size=(256, 256))).astype("uint8")
        img_path = images_dir / f"{i:08d}_000.png"
        Image.fromarray(arr, mode="L").save(img_path)

        row = {
            "image_path": str(img_path),
            "patient_id": 1000 + i // 2,
            "labels_str": "Infiltration|Effusion" if i % 2 == 0 else "No Finding",
        }
        for cls in CLASSES:
            row[cls] = int(cls in {"Infiltration", "Effusion"} and i % 2 == 0)
        records.append(row)

    split_path = tmp_path / "split.csv"
    pd.DataFrame(records).to_csv(split_path, index=False)
    return split_path
