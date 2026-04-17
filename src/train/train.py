from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CFG
from src.constants import CLASSES
from src.data.dataset import ChestXrayDataset
from src.data.sampler import (
    build_weighted_sampler,
    compute_pos_weight,
    compute_sample_weights,
)
from src.data.transforms import get_eval_transform, get_train_transform
from src.models.densenet import RadTriageDenseNet
from src.models.losses import build_bce_with_pos_weight
from src.seed import seed_everything
from src.train.engine import fit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=str, required=True)
    parser.add_argument("--val-csv", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="checkpoints")
    parser.add_argument("--freeze-backbone", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = dict(CFG)
    cfg["freeze_backbone"] = bool(args.freeze_backbone)
    seed_everything(cfg["seed"])

    train_ds = ChestXrayDataset(args.train_csv, transform=get_train_transform())
    val_ds = ChestXrayDataset(args.val_csv, transform=get_eval_transform())

    train_df = pd.read_csv(args.train_csv)
    targets_np = train_df[CLASSES].values.astype("float32")
    pos_weight = compute_pos_weight(targets_np)
    sample_weights = compute_sample_weights(targets_np, pos_weight)
    sampler = build_weighted_sampler(sample_weights)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        sampler=sampler,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    model = RadTriageDenseNet(
        num_classes=cfg["num_classes"], pretrained=True, dropout=cfg["dropout"]
    )
    model.set_freeze_backbone(cfg["freeze_backbone"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    criterion = build_bce_with_pos_weight(pos_weight, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    result = fit(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
        cfg,
        out_dir=args.out_dir,
    )
    print(
        f"Training complete. Best {cfg['monitor_metric']}: {result['best_metric']:.4f}"
    )


if __name__ == "__main__":
    main()
