from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Optional

import pandas as pd
import streamlit as st
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import CLASSES
from src.inference.pipeline import RadTriagePipeline
from src.models.densenet import RadTriageDenseNet

DEFAULT_CHECKPOINT = str(PROJECT_ROOT / "src" / "best_radtriage_model_full_run2.pt")
DEFAULT_THRESHOLDS = str(PROJECT_ROOT / "src" / "best_thresholds_full.json")
DEFAULT_SAMPLE_ROOT = str(PROJECT_ROOT / "data" / "sample")


def load_thresholds(thresholds_path: str) -> dict:
    path = Path(thresholds_path)
    if not path.exists():
        return {cls: 0.5 for cls in CLASSES}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {cls: float(data.get(cls, 0.5)) for cls in CLASSES}


def download_sample_dataset() -> str:
    try:
        import kagglehub
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "kagglehub is not installed. Install it with: pip install kagglehub"
        ) from exc

    return kagglehub.dataset_download("nih-chest-xrays/sample")


def kagglehub_status() -> tuple[bool, str]:
    try:
        import kagglehub

        return True, f"kagglehub {kagglehub.__version__}"
    except Exception as exc:
        return False, str(exc)


def list_sample_images(sample_root: str) -> list[str]:
    root = Path(sample_root)
    if not root.exists():
        return []

    if (root / "sample" / "images").exists():
        root = root / "sample" / "images"
    elif (root / "images").exists():
        root = root / "images"

    files = []
    for pattern in ("**/*.png", "**/*.jpg", "**/*.jpeg"):
        files.extend(root.glob(pattern))
    files = sorted({str(p.resolve()) for p in files})
    return files


def get_sample_root(manual_path: str) -> Optional[str]:
    if manual_path:
        path = Path(manual_path)
        if path.exists():
            return str(path.resolve())
    default_path = Path(DEFAULT_SAMPLE_ROOT)
    if default_path.exists():
        return str(default_path.resolve())
    return st.session_state.get("sample_dataset_root")


@st.cache_data
def load_sample_labels(sample_root: str) -> dict[str, list[str]]:
    import pandas as pd

    root = Path(sample_root)
    candidates = [
        root / "sample_labels.csv",
        root / "sample" / "sample_labels.csv",
    ]
    csv_path = next((p for p in candidates if p.exists()), None)
    if csv_path is None:
        return {}

    df = pd.read_csv(csv_path)
    if "Image Index" not in df.columns or "Finding Labels" not in df.columns:
        return {}

    mapping: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        image_id = str(row["Image Index"])
        labels_str = str(row["Finding Labels"])
        labels = [
            x.strip()
            for x in labels_str.split("|")
            if x.strip() and x.strip() != "No Finding"
        ]
        mapping[image_id] = labels
    return mapping


def evaluate_case_correctness(
    probs: dict, thresholds: dict, true_labels: list[str]
) -> dict:
    predicted_labels = [
        cls for cls in CLASSES if probs[cls] >= float(thresholds.get(cls, 0.5))
    ]
    true_set = set(true_labels)
    pred_set = set(predicted_labels)
    exact_match = pred_set == true_set

    top_class = max(probs.items(), key=lambda x: x[1])[0]
    top_finding_correct = top_class in true_set if true_set else len(pred_set) == 0

    if not true_set and not pred_set:
        jaccard = 1.0
    elif not (true_set or pred_set):
        jaccard = 0.0
    else:
        jaccard = len(true_set & pred_set) / max(1, len(true_set | pred_set))

    return {
        "predicted_labels": predicted_labels,
        "exact_match": exact_match,
        "top_finding_correct": top_finding_correct,
        "jaccard": float(jaccard),
    }


@st.cache_resource
def load_pipeline(checkpoint_path: str = "") -> RadTriagePipeline:
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = RadTriageDenseNet(num_classes=len(CLASSES), pretrained=False)
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        elif isinstance(ckpt, dict):
            model.load_state_dict(ckpt)
        else:
            raise ValueError("Unsupported checkpoint format.")
    return RadTriagePipeline(model=model, device="cuda", heatmap_dir="outputs/heatmaps")


def render_single_case(
    pipeline: RadTriagePipeline,
    thresholds: dict,
    sample_images: list[str],
    sample_labels: dict[str, list[str]],
):
    st.subheader("Single Case")
    source = st.radio(
        "Image source", ["Upload image", "Choose sample image"], horizontal=True
    )

    img_path: Optional[Path] = None
    if source == "Upload image":
        up = st.file_uploader(
            "Upload chest X-ray", type=["png", "jpg", "jpeg"], key="single"
        )
        if up is None:
            return
        tmp_dir = Path("outputs/demo_cases")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        img_path = tmp_dir / up.name
        img_path.write_bytes(up.getvalue())
    else:
        if not sample_images:
            st.info(
                "No sample images found. Set a sample dataset directory in the sidebar or download with kagglehub."
            )
            return
        selected_path = st.selectbox(
            "Select sample image",
            options=sample_images,
            format_func=lambda p: str(Path(p).name),
            key="single_sample",
        )
        img_path = Path(selected_path)

    out = pipeline.predict_single(str(img_path))
    left, right = st.columns(2)
    with left:
        st.image(str(img_path), caption="Original image")
    with right:
        st.image(out["heatmap_path"], caption="Grad-CAM overlay")

    st.write(f"Urgency tier: **{out['urgency_tier']}**")
    st.write(f"Urgency score: {out['urgency_score']:.3f}")
    st.write(f"Confidence: {out['confidence']:.3f}")
    st.write("Top findings:", out["top_findings"])

    probs_df = pd.DataFrame(
        {"class": CLASSES, "probability": [out["probs"][c] for c in CLASSES]}
    )
    probs_df["threshold"] = probs_df["class"].map(thresholds)
    probs_df["predicted_positive"] = probs_df["probability"] >= probs_df["threshold"]
    probs_df = probs_df.sort_values("probability", ascending=False).reset_index(
        drop=True
    )

    positive_findings = probs_df[probs_df["predicted_positive"]]["class"].tolist()
    st.write(
        "Thresholded findings:", positive_findings if positive_findings else "None"
    )

    image_id = Path(img_path).name
    true_labels = sample_labels.get(image_id)
    if true_labels is not None:
        check = evaluate_case_correctness(out["probs"], thresholds, true_labels)
        st.write("Ground truth labels:", true_labels if true_labels else "No Finding")
        st.write(
            "Prediction check:",
            "Correct" if check["exact_match"] else "Partially/Not Correct",
        )
        st.write(
            "Top finding correct:", "Yes" if check["top_finding_correct"] else "No"
        )
        st.write("Label overlap (Jaccard):", f"{check['jaccard']:.3f}")

    st.dataframe(probs_df, width="stretch")


def render_worklist(pipeline: RadTriagePipeline, sample_images: list[str]):
    st.subheader("Worklist")
    source = st.radio(
        "Worklist source", ["Upload images", "Choose sample images"], horizontal=True
    )

    paths: list[str] = []
    if source == "Upload images":
        uploads = st.file_uploader(
            "Upload multiple studies",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="batch",
        )
        if not uploads:
            return
        tmp_dir = Path("outputs/demo_cases")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        for up in uploads:
            p = tmp_dir / up.name
            p.write_bytes(up.getvalue())
            paths.append(str(p))
    else:
        if not sample_images:
            st.info(
                "No sample images found. Set a sample dataset directory in the sidebar or download with kagglehub."
            )
            return
        selected = st.multiselect(
            "Select sample studies",
            options=sample_images,
            default=sample_images[: min(5, len(sample_images))],
            format_func=lambda p: str(Path(p).name),
            key="batch_sample",
        )
        if not selected:
            return
        paths = selected

    original_df = pd.DataFrame(
        {
            "study_id": [Path(p).name for p in paths],
            "original_order": list(range(1, len(paths) + 1)),
        }
    )
    ranked = pipeline.rank_worklist(paths)
    st.write("Original worklist")
    st.dataframe(original_df, width="stretch")
    st.write("AI-prioritized worklist")
    st.dataframe(ranked, width="stretch")


def render_metrics():
    st.subheader("Metrics")
    metrics_path = Path("outputs/metrics/test_metrics.json")
    per_class_path = Path("outputs/metrics/per_class_metrics.csv")
    if metrics_path.exists():
        st.json(json.loads(metrics_path.read_text(encoding="utf-8")))
    else:
        st.info("No test metrics found yet.")
    if per_class_path.exists():
        st.dataframe(pd.read_csv(per_class_path), width="stretch")


def main():
    st.set_page_config(page_title="RadTriage AI", layout="wide")
    st.title("RadTriage AI Dashboard")
    st.sidebar.caption(f"Python: {sys.executable}")
    ok, kh_status = kagglehub_status()
    if ok:
        st.sidebar.success(f"{kh_status} available")
    else:
        st.sidebar.error(f"kagglehub unavailable: {kh_status}")
    checkpoint_path = st.sidebar.text_input("Checkpoint path", value=DEFAULT_CHECKPOINT)
    thresholds_path = st.sidebar.text_input("Thresholds path", value=DEFAULT_THRESHOLDS)
    manual_sample_dir = st.sidebar.text_input(
        "Sample dataset path",
        value=DEFAULT_SAMPLE_ROOT,
    )
    if st.sidebar.button("Download sample subset with kagglehub"):
        try:
            st.session_state["sample_dataset_root"] = download_sample_dataset()
            st.sidebar.success(
                f"Downloaded sample subset: {st.session_state['sample_dataset_root']}"
            )
        except Exception as exc:
            st.sidebar.error(f"Download failed: {exc}")
            st.sidebar.info(
                "Run `pip install kagglehub` in your active .venv, then click again."
            )

    sample_root = get_sample_root(manual_sample_dir)
    sample_images = list_sample_images(sample_root) if sample_root else []
    sample_labels = load_sample_labels(sample_root) if sample_root else {}
    if sample_root:
        st.sidebar.caption(f"Sample root: {sample_root}")
        st.sidebar.caption(f"Sample images found: {len(sample_images)}")
        st.sidebar.caption(f"Sample labels loaded: {len(sample_labels)}")

    thresholds = load_thresholds(thresholds_path)
    try:
        pipeline = load_pipeline(checkpoint_path)
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        return
    st.sidebar.success("Loaded model and thresholds")

    tabs = st.tabs(["Single Case", "Worklist", "Metrics"])
    with tabs[0]:
        render_single_case(pipeline, thresholds, sample_images, sample_labels)
    with tabs[1]:
        render_worklist(pipeline, sample_images)
    with tabs[2]:
        render_metrics()


if __name__ == "__main__":
    main()
