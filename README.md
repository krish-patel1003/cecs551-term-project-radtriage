# RadTriage AI

Inference-first chest X-ray triage demo built around a pretrained DenseNet-121 checkpoint.

## What is in this repo

- `src/` contains only runtime code needed for prediction, triage scoring, Grad-CAM, and Streamlit UI.
- `kaggle/` contains Kaggle notebooks and exported training/evaluation artifacts used to produce the model.
- `data/sample/` can be used locally for demo images, but is excluded from GitHub.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run src/app/streamlit_app.py
```

## Default runtime artifacts

The app is preconfigured to use:

- Model: `kaggle/kaggle_run_2/best_radtriage_model_full_run2.pt`
- Thresholds: `kaggle/kaggle_run_1/best_thresholds_full.json`
- Metrics: `kaggle/kaggle_run_1/test_metrics_full.csv`

You can override these paths from the app sidebar.

## Documentation

See `docs/PROJECT_REPORT.md` for:

- Data summary and label schema
- Model architecture
- Evaluation results and score summary
