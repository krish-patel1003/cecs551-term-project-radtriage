from __future__ import annotations

from dataclasses import asdict, dataclass

from src.constants import NUM_CLASSES


@dataclass
class TrainConfig:
    seed: int = 42
    img_size: int = 224
    batch_size: int = 32
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_classes: int = NUM_CLASSES
    num_workers: int = 4
    mixed_precision: bool = True
    early_stopping_patience: int = 5
    monitor_metric: str = "macro_auc"
    device: str = "cuda"
    freeze_backbone: bool = False
    dropout: float = 0.2

    def to_dict(self) -> dict:
        return asdict(self)


CFG = TrainConfig().to_dict()
