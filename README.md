# RadTriage AI

DenseNet-121 based multi-label chest X-ray triage prototype for NIH ChestX-ray14.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data layout

Place NIH metadata and images at:

- `data/raw/Data_Entry_2017.csv`
- `data/raw/images/`

Generate patient-level splits:

```bash
python -m src.data.prepare_splits --csv data/raw/Data_Entry_2017.csv --images-dir data/raw/images --out-dir data/splits
```

## Train

```bash
python -m src.train.train --train-csv data/splits/train.csv --val-csv data/splits/val.csv
```

## Evaluate

```bash
python -m src.eval.evaluate --checkpoint checkpoints/best_auc.pt --test-csv data/splits/test.csv --out-dir outputs
```

## Streamlit app

```bash
streamlit run src/app/streamlit_app.py
```

## GitHub notes

- Large/generated artifacts are gitignored (`data/sample/`, `outputs/`, checkpoint binaries, local model binaries).
- Keep required local model files in `src/` when running the demo app:
  - `src/best_radtriage_model_full_run2.pt`
  - `src/best_thresholds_full.json`
