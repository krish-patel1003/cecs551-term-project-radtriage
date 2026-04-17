from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from src.constants import CLASSES
from src.data.dataset import ChestXrayDataset
from src.data.prepare_splits import make_patient_splits
from src.data.transforms import get_eval_transform


def test_dataset_basic(sample_split_csv: Path):
    ds = ChestXrayDataset(sample_split_csv, transform=get_eval_transform())
    assert len(ds) > 0

    sample = ds[0]
    assert sample["image"].shape == (3, 224, 224)
    assert sample["target"].shape == (14,)
    assert set(torch.unique(sample["target"]).tolist()).issubset({0.0, 1.0})
    assert isinstance(sample["patient_id"], int)


def test_split_leakage(sample_split_csv: Path):
    df = pd.read_csv(sample_split_csv)
    train_df, val_df, test_df = make_patient_splits(df, seed=42)

    train_patients = set(train_df["patient_id"].tolist())
    val_patients = set(val_df["patient_id"].tolist())
    test_patients = set(test_df["patient_id"].tolist())

    assert train_patients.isdisjoint(val_patients)
    assert train_patients.isdisjoint(test_patients)
    assert val_patients.isdisjoint(test_patients)
    assert list(df.columns).count(CLASSES[0]) == 1
