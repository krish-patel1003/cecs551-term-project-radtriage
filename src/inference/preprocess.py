from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.constants import IMAGENET_MEAN, IMAGENET_STD


def get_eval_transform():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


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
