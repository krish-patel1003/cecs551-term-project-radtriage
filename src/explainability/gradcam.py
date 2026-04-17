from __future__ import annotations

from pathlib import Path

import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer or self.model.backbone.features.denseblock4
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, grad_input, grad_output):
            del grad_input
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        logits = self.model(input_tensor)
        score = logits[:, class_idx].sum()
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)

        grads = self.gradients
        acts = self.activations
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy()


def overlay_heatmap(
    original_image: Image.Image, cam: np.ndarray, alpha: float = 0.4
) -> Image.Image:
    base = original_image.convert("RGB")
    cam_img = Image.fromarray(np.uint8(cam * 255)).resize(base.size)
    cam_arr = np.array(cam_img).astype(np.float32) / 255.0
    heat = cm.jet(cam_arr)[..., :3]

    base_arr = np.array(base).astype(np.float32) / 255.0
    blend = (1 - alpha) * base_arr + alpha * heat
    blend = np.clip(blend * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(blend)


def save_cam_triplet(
    original: Image.Image, cam: np.ndarray, overlay: Image.Image, out_prefix: str | Path
) -> dict:
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    original_path = f"{out_prefix}_original.png"
    heatmap_path = f"{out_prefix}_heatmap.png"
    overlay_path = f"{out_prefix}_overlay.png"

    original.save(original_path)
    Image.fromarray(np.uint8(cam * 255)).save(heatmap_path)
    overlay.save(overlay_path)
    return {"original": original_path, "heatmap": heatmap_path, "overlay": overlay_path}
