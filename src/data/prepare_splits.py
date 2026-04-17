from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import CLASS_TO_IDX, CLASSES, IMAGE_COL, LABEL_COL, PATIENT_COL


def encode_labels(labels_str: str) -> np.ndarray:
    target = np.zeros(len(CLASSES), dtype=np.int64)
    if not isinstance(labels_str, str):
        return target
    for finding in labels_str.split("|"):
        finding = finding.strip()
        if finding == "No Finding":
            continue
        idx = CLASS_TO_IDX.get(finding)
        if idx is not None:
            target[idx] = 1
    return target


def build_labeled_dataframe(
    csv_path: str | Path, images_dir: str | Path
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = {IMAGE_COL, LABEL_COL, PATIENT_COL} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = pd.DataFrame(
        {
            "image_path": [
                str(Path(images_dir) / name)
                for name in df[IMAGE_COL].astype(str).tolist()
            ],
            "patient_id": df[PATIENT_COL].astype(int),
            "labels_str": df[LABEL_COL].fillna("").astype(str),
        }
    )
    targets = np.vstack([encode_labels(x) for x in out["labels_str"].tolist()])
    for i, cls in enumerate(CLASSES):
        out[cls] = targets[:, i]
    return out


def make_patient_splits(
    df: pd.DataFrame, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    patient_ids = df["patient_id"].dropna().unique().astype(int)
    rng = np.random.default_rng(seed)
    rng.shuffle(patient_ids)

    n = len(patient_ids)
    n_train = int(0.7 * n)
    n_val = int(0.1 * n)

    train_patients = set(patient_ids[:n_train])
    val_patients = set(patient_ids[n_train : n_train + n_val])
    test_patients = set(patient_ids[n_train + n_val :])

    train_df = df[df["patient_id"].isin(train_patients)].reset_index(drop=True)
    val_df = df[df["patient_id"].isin(val_patients)].reset_index(drop=True)
    test_df = df[df["patient_id"].isin(test_patients)].reset_index(drop=True)
    return train_df, val_df, test_df


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: str | Path,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out / "train.csv", index=False)
    val_df.to_csv(out / "val.csv", index=False)
    test_df.to_csv(out / "test.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--images-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="data/splits")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = build_labeled_dataframe(args.csv, args.images_dir)
    train_df, val_df, test_df = make_patient_splits(df, seed=args.seed)
    save_splits(train_df, val_df, test_df, args.out_dir)
    print(f"Saved splits to {args.out_dir}")


if __name__ == "__main__":
    main()
