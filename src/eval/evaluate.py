from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import CLASSES
from src.data.dataset import ChestXrayDataset
from src.data.transforms import get_eval_transform
from src.eval.metrics import compute_all_metrics
from src.eval.threshold_search import (
    load_thresholds,
    search_thresholds,
    save_thresholds,
)
from src.models.densenet import RadTriageDenseNet


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    rows = []
    y_true, y_prob = [], []
    for batch in loader:
        x = batch["image"].to(device)
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()
        targets = batch["target"].cpu().numpy()

        for i in range(len(probs)):
            rec = {
                "image_id": batch["image_id"][i],
                "patient_id": int(batch["patient_id"][i]),
            }
            for c, cls in enumerate(CLASSES):
                rec[f"true_{cls}"] = float(targets[i, c])
                rec[f"prob_{cls}"] = float(probs[i, c])
            rows.append(rec)

        y_true.append(targets)
        y_prob.append(probs)

    return (
        pd.DataFrame(rows),
        np.concatenate(y_true, axis=0),
        np.concatenate(y_prob, axis=0),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--test-csv", type=str, required=True)
    p.add_argument("--val-preds", type=str, default="")
    p.add_argument("--thresholds", type=str, default="outputs/metrics/thresholds.json")
    p.add_argument("--out-dir", type=str, default="outputs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    cfg = checkpoint.get("config", {})
    model = RadTriageDenseNet(
        num_classes=len(CLASSES), pretrained=False, dropout=cfg.get("dropout", 0.2)
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    device = torch.device(
        cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    )
    model.to(device)

    test_ds = ChestXrayDataset(args.test_csv, transform=get_eval_transform())
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.get("batch_size", 32),
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
    )
    pred_df, y_true, y_prob = predict(model, test_loader, device)
    pred_df.to_csv(out_dir / "predictions" / "test_predictions.csv", index=False)

    thr_path = Path(args.thresholds)
    if thr_path.exists():
        thresholds = load_thresholds(thr_path)
    elif args.val_preds:
        val_df = pd.read_csv(args.val_preds)
        val_true = val_df[[f"true_{c}" for c in CLASSES]].values
        val_prob = val_df[[f"prob_{c}" for c in CLASSES]].values
        thresholds = search_thresholds(val_true, val_prob, objective="f1")
        save_thresholds(thresholds, thr_path)
    else:
        thresholds = {c: 0.5 for c in CLASSES}

    thr_list = [thresholds[c] for c in CLASSES]
    metrics = compute_all_metrics(y_true, y_prob, thresholds=thr_list)

    (out_dir / "metrics" / "test_metrics.json").write_text(
        json.dumps({k: v for k, v in metrics.items() if k != "per_class"}, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(metrics["per_class"]).to_csv(
        out_dir / "metrics" / "per_class_metrics.csv", index=False
    )
    print("Saved predictions and metrics.")


if __name__ == "__main__":
    main()
