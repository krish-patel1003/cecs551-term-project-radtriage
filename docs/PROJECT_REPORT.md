# RadTriage AI Project Report

## 1) Data

- Dataset family: NIH ChestX-ray14.
- Label space: 14 pathologies in fixed order:
  - Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule,
    Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis,
    Pleural_Thickening, Hernia.
- Sample subset used for interactive demo:
  - Images under `data/sample/sample/images/`
  - Labels under `data/sample/sample_labels.csv` (or nested equivalent)

## 2) Model Architecture

- Backbone: `torchvision.models.densenet121`.
- Transfer learning head:
  - Dropout(0.2)
  - Linear(in_features -> 14 logits)
- Multi-label output:
  - Raw logits -> sigmoid probabilities per class.

Implementation reference: `src/models/densenet.py`.

## 3) Inference + Triage Runtime

- Preprocess:
  - Grayscale CXR converted to 3-channel RGB
  - Resize 256 -> CenterCrop 224 -> ToTensor -> ImageNet normalization
- Prediction:
  - Sigmoid probabilities for 14 classes
  - Top findings extracted from class probabilities
- Triage:
  - Severity-weighted score from class probabilities
  - Rule-based escalation for critical findings (for example high Pneumothorax)
  - Tier output: Emergent / Urgent / Routine
- Explainability:
  - Grad-CAM on `denseblock4`
  - Overlay saved and shown in dashboard

Implementation references:

- `src/inference/preprocess.py`
- `src/inference/predict.py`
- `src/inference/pipeline.py`
- `src/triage/scoring.py`
- `src/explainability/gradcam.py`

## 4) Evaluation Results (Kaggle Run 1)

Source: `kaggle/kaggle_run_1/test_metrics_full.csv`

Aggregate scores (macro/mean over 14 classes):

- Macro AUC: **0.7770**
- Macro F1: **0.2750**
- Mean Sensitivity: **0.5569**
- Mean Specificity: **0.8242**

Top AUC classes:

- Hernia: 0.9407
- Emphysema: 0.8804
- Cardiomegaly: 0.8569
- Pneumothorax: 0.8304

Lower AUC classes:

- Infiltration: 0.6904
- Pneumonia: 0.6518

## 5) Thresholding

- Class-wise thresholds are loaded from:
  - `kaggle/kaggle_run_1/best_thresholds_full.json`
- Dashboard uses these thresholds to mark predicted-positive findings.

## 6) Artifacts

- Training/eval notebooks and exports:
  - `kaggle/kaggle_run_1/ai_term_project_full_dataset.ipynb`
  - `kaggle/kaggle_run_2/ai-term-project-full-dataset_run2.ipynb`
- Runtime model checkpoint:
  - `kaggle/kaggle_run_2/best_radtriage_model_full_run2.pt`
