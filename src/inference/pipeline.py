from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from src.constants import CLASSES
from src.explainability.gradcam import GradCAM, overlay_heatmap, save_cam_triplet
from src.inference.predict import predict_probs
from src.inference.preprocess import preprocess_image
from src.triage.scoring import score_case


class RadTriagePipeline:
    def __init__(
        self, model, device: str = "cuda", heatmap_dir: str = "outputs/heatmaps"
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.gradcam = GradCAM(self.model)
        self.heatmap_dir = Path(heatmap_dir)
        self.heatmap_dir.mkdir(parents=True, exist_ok=True)

    def predict_single(self, image_path: str, class_idx: int | None = None) -> dict:
        image_id = Path(image_path).name
        original, x = preprocess_image(image_path)
        probs = predict_probs(self.model, x, self.device)
        triage = score_case(probs, image_id=image_id)

        if class_idx is None:
            top_cls = triage["top_findings"][0][0]
            class_idx = CLASSES.index(top_cls)
        else:
            top_cls = CLASSES[class_idx]

        cam = self.gradcam.generate(x.to(self.device), class_idx)
        overlay = overlay_heatmap(original, cam)
        out_prefix = self.heatmap_dir / f"{Path(image_id).stem}_{top_cls}"
        paths = save_cam_triplet(original, cam, overlay, out_prefix)

        triage["heatmap_path"] = paths["overlay"]
        return triage

    def rank_worklist(self, studies: list[str]) -> pd.DataFrame:
        rows = []
        for idx, study_path in enumerate(studies):
            out = self.predict_single(study_path)
            rows.append(
                {
                    "study_id": Path(study_path).name,
                    "original_order": idx + 1,
                    "predicted_top_finding": out["top_findings"][0][0],
                    "top_probability": out["top_findings"][0][1],
                    "confidence": out["confidence"],
                    "urgency_score": out["urgency_score"],
                    "urgency_tier": out["urgency_tier"],
                    "heatmap_path": out["heatmap_path"],
                }
            )

        df = pd.DataFrame(rows)
        df = df.sort_values("urgency_score", ascending=False).reset_index(drop=True)
        df["new_rank"] = df.index + 1
        return df
