# Prototype Progress Report

## Project

RadTriage AI: AI-assisted chest X-ray triage prototype for urgency-aware worklist prioritization.

## Current Stage

Inference-first prototype complete for interactive demo use.

## Objectives and Status

### Objective 1: End-to-end inference on a single CXR

- Status: Completed.
- Evidence:
  - Model checkpoint loading and prediction in `src/app/streamlit_app.py`.
  - Preprocess + predict flow in `src/inference/preprocess.py` and `src/inference/predict.py`.

### Objective 2: Deterministic urgency scoring and routing

- Status: Completed.
- Evidence:
  - Severity-weighted scoring and override rules in `src/triage/scoring.py`.
  - Tier mapping in `src/triage/routing.py`.

### Objective 3: Explainability visualization

- Status: Completed.
- Evidence:
  - Grad-CAM generation and overlay export in `src/explainability/gradcam.py`.

### Objective 4: Worklist reprioritization UI

- Status: Completed.
- Evidence:
  - Batch ranking in `src/inference/pipeline.py`.
  - Worklist tab in `src/app/streamlit_app.py`.

### Objective 5: Sample-based correctness checks

- Status: Completed (prototype-level).
- Evidence:
  - Ground-truth matching logic in `evaluate_case_correctness` (`src/app/streamlit_app.py`).

## Metrics Snapshot (from Kaggle artifacts)

Source: `kaggle/kaggle_run_1/test_metrics_full.csv`

- Macro AUC: 0.7770
- Macro F1: 0.2750
- Mean Sensitivity: 0.5569
- Mean Specificity: 0.8242

## Key Design Decisions

- Keep runtime repo lightweight and inference-focused.
- Externalize training/evaluation implementation details to Kaggle notebooks/artifacts.
- Use class-specific thresholds from validation outputs rather than global 0.5.
- Blend model-based triage score with safety-driven override rules.

## Risks and Limitations

- Clinical labels in ChestX-ray14 are weak labels and can contain noise.
- Class imbalance remains challenging for low-prevalence findings.
- Correctness check in UI is dataset-label agreement, not expert radiologist validation.
- Triage thresholds and severity weights need prospective calibration for deployment.

## Next Technical Milestones

1. Add confusion summaries and threshold diagnostics per class in dashboard.
2. Add reliability calibration plots (ECE/Brier) and confidence analysis.
3. Add regression tests for checkpoint compatibility and output schema.
4. Add reproducible script to regenerate report tables from Kaggle CSV artifacts.
5. Package API layer for non-Streamlit integration.

## Deliverables Present in Repo

- Runtime code under `src/`
- Kaggle artifacts and notebooks under `kaggle/`
- Technical docs under `docs/`
- Report-ready architecture and progress documentation (this file + technical architecture doc)
