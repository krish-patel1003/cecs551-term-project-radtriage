from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from src.constants import CLASSES


class ChestXrayDataset(Dataset):
    def __init__(self, csv_path: str | Path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        missing = {"image_path", "patient_id", "labels_str", *CLASSES} - set(
            self.df.columns
        )
        if missing:
            raise ValueError(
                f"Missing required columns in split CSV: {sorted(missing)}"
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        path = str(row["image_path"])
        img = Image.open(path).convert("L")
        img = Image.merge("RGB", (img, img, img))
        if self.transform is not None:
            image_tensor = self.transform(img)
        else:
            image_tensor = ToTensor()(img)

        target = torch.tensor(
            row[CLASSES].astype("float32").values, dtype=torch.float32
        )
        return {
            "image": image_tensor.float(),
            "target": target,
            "image_id": Path(path).name,
            "patient_id": int(row["patient_id"]),
            "raw_path": path,
        }
