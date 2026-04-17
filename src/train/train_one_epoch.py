from __future__ import annotations

import torch


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    scaler,
    device,
    mixed_precision: bool = True,
) -> dict:
    model.train()
    running_loss = 0.0
    total = 0
    use_amp = mixed_precision and str(device).startswith("cuda")

    for batch in train_loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = x.size(0)
        running_loss += loss.item() * bs
        total += bs

    return {"loss": running_loss / max(1, total)}
