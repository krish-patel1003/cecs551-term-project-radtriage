from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.constants import CLASSES
from src.data.dataset import ChestXrayDataset
from src.data.sampler import (
    build_weighted_sampler,
    compute_pos_weight,
    compute_sample_weights,
)
from src.data.transforms import get_eval_transform, get_train_transform
from src.eval.evaluate import predict
from src.eval.metrics import compute_all_metrics
from src.models.densenet import RadTriageDenseNet
from src.models.losses import build_bce_with_pos_weight
from src.train.engine import fit


def test_train_eval_smoke(tmp_path: Path, sample_split_csv: Path):
    train_ds = ChestXrayDataset(sample_split_csv, transform=get_train_transform())
    val_ds = ChestXrayDataset(sample_split_csv, transform=get_eval_transform())

    df = pd.read_csv(sample_split_csv)
    targets_np = df[CLASSES].values.astype("float32")
    pos_weight = compute_pos_weight(targets_np)
    sampler = build_weighted_sampler(compute_sample_weights(targets_np, pos_weight))

    train_loader = DataLoader(train_ds, batch_size=2, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0)

    cfg = {
        "device": "cpu",
        "epochs": 1,
        "monitor_metric": "macro_auc",
        "early_stopping_patience": 1,
        "mixed_precision": False,
        "num_classes": 14,
    }

    model = RadTriageDenseNet(num_classes=14, pretrained=False)
    criterion = build_bce_with_pos_weight(pos_weight, torch.device("cpu"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1
    )

    fit(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
        cfg,
        out_dir=str(tmp_path / "checkpoints"),
    )
    assert (tmp_path / "checkpoints" / "last.pt").exists()

    pred_df, y_true, y_prob = predict(model, val_loader, torch.device("cpu"))
    metrics = compute_all_metrics(y_true, y_prob, thresholds=0.5)

    assert len(pred_df) > 0
    assert "macro_auc" in metrics
