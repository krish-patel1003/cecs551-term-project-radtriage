from __future__ import annotations

import json
from pathlib import Path

import torch

from src.constants import CLASSES
from src.train.train_one_epoch import train_one_epoch
from src.train.validate import validate


def save_checkpoint(
    path: str | Path, epoch: int, model, optimizer, best_metric: float, cfg: dict
) -> None:
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,
        "config": cfg,
        "class_names": CLASSES,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    cfg: dict,
    out_dir: str = "checkpoints",
) -> dict:
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    model.to(device)

    scaler = torch.cuda.amp.GradScaler(
        enabled=cfg.get("mixed_precision", True) and device.type == "cuda"
    )
    best_metric = float("-inf")
    patience = 0
    history = []

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    for epoch in range(1, cfg["epochs"] + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            device,
            mixed_precision=cfg.get("mixed_precision", True),
        )
        val_metrics = validate(model, val_loader, criterion, device)
        monitor = float(val_metrics.get(cfg.get("monitor_metric", "macro_auc"), 0.0))

        scheduler.step(monitor)
        save_checkpoint(out_path / "last.pt", epoch, model, optimizer, best_metric, cfg)

        if monitor > best_metric:
            best_metric = monitor
            patience = 0
            save_checkpoint(
                out_path / "best_auc.pt", epoch, model, optimizer, best_metric, cfg
            )
        else:
            patience += 1

        history.append(
            {"epoch": epoch, **train_metrics, **val_metrics, "monitor": monitor}
        )
        if patience >= cfg.get("early_stopping_patience", 5):
            break

    return {"best_metric": best_metric, "history": history}
