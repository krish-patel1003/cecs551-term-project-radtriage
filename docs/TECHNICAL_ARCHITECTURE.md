# Technical Architecture and Code Walkthrough

## 1. System Overview

RadTriage AI is an inference-first clinical prototype that:

1. loads a pretrained DenseNet-121 checkpoint,
2. predicts 14 chest pathology probabilities,
3. computes a severity-weighted triage score and urgency tier,
4. generates Grad-CAM overlays,
5. displays case-level and worklist-level views in Streamlit.

Training and evaluation experimentation are intentionally externalized to `kaggle/` artifacts.

## 2. Runtime Architecture

High-level runtime flow:

```text
Input image
  -> preprocessing (grayscale -> RGB, resize/crop/normalize)
  -> DenseNet-121 forward pass (14 logits)
  -> sigmoid probabilities
  -> triage scoring + override rules
  -> Grad-CAM for top finding
  -> dashboard rendering (single case / worklist / metrics)
```

## 3. Module Responsibilities

### `src/constants.py`

- Defines the canonical 14-class order (`CLASSES`).
- Stores ImageNet normalization constants used by preprocessing.
- Provides schema constants shared across modules.

### `src/models/densenet.py`

- Defines `RadTriageDenseNet`.
- Uses `torchvision.models.densenet121` backbone.
- Replaces classifier head with:
  - `Dropout(0.2)`
  - `Linear(in_features, 14)`
- Outputs logits of shape `[B, 14]`.

### `src/inference/preprocess.py`

- Converts CXR grayscale PNG to 3-channel RGB by channel replication.
- Applies evaluation transform:
  - `Resize(256)`
  - `CenterCrop(224)`
  - `ToTensor()`
  - `Normalize(IMAGENET_MEAN, IMAGENET_STD)`

### `src/inference/predict.py`

- Runs model in `no_grad` mode.
- Applies sigmoid to logits.
- Returns class->probability dictionary.

### `src/triage/severity_weights.py`

- Encodes clinically motivated per-class severity coefficients.

### `src/triage/routing.py`

- Maps score to urgency tier:
  - `score >= 0.75` -> Emergent
  - `score >= 0.40` -> Urgent
  - otherwise -> Routine

### `src/triage/scoring.py`

- Computes weighted urgency score.
- Applies hard override rules:
  - Pneumothorax >= 0.60 -> Emergent
  - Edema >= 0.70 or Pneumonia >= 0.70 -> Urgent
- Returns:
  - `probs`, `top_findings`, `urgency_score`, `urgency_tier`, `confidence`.

### `src/explainability/gradcam.py`

- Hooks DenseNet target layer (`denseblock4`).
- Uses activation-gradient weighting to produce CAM.
- Normalizes CAM to `[0, 1]` and creates overlay image.
- Saves original / heatmap / overlay files.

### `src/inference/pipeline.py`

- Orchestrates end-to-end single-case inference.
- Generates Grad-CAM for top predicted finding by default.
- Provides `rank_worklist()` to reorder studies by urgency score.

### `src/app/streamlit_app.py`

- Loads model checkpoint and thresholds.
- Supports both uploaded images and local sample dataset.
- Displays prediction correctness against available sample labels.
- Renders:
  - Single Case tab
  - Worklist tab
  - Metrics tab (from exported CSV)

## 4. Runtime Data Contracts

### Single-case output contract

`RadTriagePipeline.predict_single()` returns:

- `image_id`
- `probs` (14-class probability mapping)
- `top_findings` (top 3 class-prob pairs)
- `urgency_score`
- `urgency_tier`
- `confidence`
- `heatmap_path`

### Worklist output contract

`RadTriagePipeline.rank_worklist()` returns dataframe columns:

- `study_id`
- `original_order`
- `predicted_top_finding`
- `top_probability`
- `confidence`
- `urgency_score`
- `urgency_tier`
- `heatmap_path`
- `new_rank`

## 5. Model and Artifact Loading

Default app artifact paths:

- checkpoint: `kaggle/kaggle_run_2/best_radtriage_model_full_run2.pt`
- thresholds: `kaggle/kaggle_run_1/best_thresholds_full.json`
- metrics CSV: `kaggle/kaggle_run_1/test_metrics_full.csv`

The checkpoint loader supports both:

- full checkpoint dict with `model_state_dict`
- raw state_dict format

## 6. Correctness Display in UI

For sample images with labels available in `sample_labels.csv`, the app computes:

- exact set match between predicted-positive labels and ground truth labels,
- whether the top predicted class is present in the ground truth,
- Jaccard overlap between predicted and true label sets.

This is presented as a prototype-level correctness indicator and not clinical validation.

## 7. Non-Goals in Current Runtime Repo

The runtime codebase intentionally excludes in-repo training pipeline code.

- Training, split generation, and experiment notebooks are preserved under `kaggle/`.
- Productionization tasks (model serving, auth, PHI controls, calibration studies) are out of current scope.
