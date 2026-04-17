from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from src.data.transforms import get_eval_transform


def load_cxr_as_rgb(path: str | Path) -> Image.Image:
    img = Image.open(path).convert("L")
    return Image.merge("RGB", (img, img, img))


def preprocess_image(
    path: str | Path, transform=None
) -> tuple[Image.Image, torch.Tensor]:
    if transform is None:
        transform = get_eval_transform()
    img = load_cxr_as_rgb(path)
    tensor = transform(img).unsqueeze(0)
    return img, tensor
