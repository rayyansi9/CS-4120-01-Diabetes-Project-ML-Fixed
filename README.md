# CS-4120-01-Diabetes-Project-ML

We predict diabetes progression two ways: regression (raw score) and classification (high/low via median). One fixed split is reused everywhere, models are logged with MLflow, and evaluation reads saved artifacts—no hidden refits.

## Quick start
- Install deps: `pip install -r requirements.txt`
- Baselines (classical models): `python src/train_baselines.py`
- Neural nets (both tasks): `python src/train_nn.py`
- Plots + comparison tables: `python src/evaluate.py`
- Everything in one shot: `python src/run_all.py`
- Optional MLflow UI: `mlflow ui --backend-store-uri file:mlruns`

## What you get
- Plots in `reports/figures/` (learning curves, confusion matrix, residuals, permutation importance).
- Tables in `reports/tables/` (classical vs NN for classification and regression).
- Saved models in `models/` plus MLflow runs in `mlruns/`.

## Reproducibility
- Data: built-in sklearn diabetes dataset; no downloads needed. Split indices are fixed and saved to `reports/splits.npz`.
- Tracking: best run IDs live in `reports/best_runs.json`; `src/evaluate.py` loads those runs to make plots and tables.
- Seeds: `random_state=42` for splitting and permutation importance. Delete `reports/splits.npz` if you really need a fresh split, then rerun the baselines.

## Quick read on results (latest run)
- Regression: linear regression stays the most reliable on test RMSE; NN was close but a bit noisier.
- Classification: the tuned NN edges out the tree on ROC-AUC/F1; the tree still has slightly higher validation accuracy.
