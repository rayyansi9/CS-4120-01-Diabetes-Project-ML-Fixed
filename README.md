# CS-4120-01-Diabetes-Project-ML
Set up and running instructions

# Entire code is first being fixed on the feedback from midpoint report and then we're implement a NN.

This project predicts diabetes disease progression using the Diabetes Dataset, which is publicly available in scikit-learn. datasets. We train regression models to predict the continuous progression score and add a median-based label to build classification baselines.

# Running the pipeline
1) Install dependencies

pip install -r requirements.txt

2) Train the baselines (one shared split for regression & classification)

python src/train_baselines.py

# Quick Results Summary
- Creates a single stratified 70/15/15 train/val/test split (stored at `data/splits.npz`).
- Trains Linear Regression & Decision Tree Regressor plus Logistic Regression & Decision Tree Classifier once.
- Saves models to `models/` and regression/classification metrics (MAE, MSE, RMSE, Accuracy, F1 on val/test) to `reports/tables/`.

3) Generate plots from the saved models (no retraining)

python src/evaluationplots.py

- Loads the persisted models and uses the exact same split indices.
- Produces: target distribution, correlation heatmap, confusion matrix (test split), and residuals vs predicted (test split) saved under `reports/figures/`.

# Current best baselines
- Linear Regression remains the strongest regression baseline.
- Logistic Regression provides the most consistent classification performance.

# Notes
- Random seeds are fixed (random_state=42) and the split indices are now stored for reproducibility.
- No raw data is committed — the dataset loads automatically from scikit-learn.
- MLflow is available for future tracking but is not enabled by default in this stage.

# Diabetes Project — ML Baselines (Course Project)

This is a small course project that trains baseline regression and classification models on a diabetes dataset, saves models to `models/`, and writes evaluation tables to `reports/tables/` and figures to `reports/figures/`.

What I changed (short)
- Fixed a SyntaxError in `src/utils.py` (removed stray characters).
- Fixed `src/evaluationplots.py`:
  - moved `from __future__ import annotations` to the top,
  - added missing imports and a safe `load_model()` check,
  - added plotting for distribution, correlation heatmap, confusion matrix, and residuals plot.
- Added guidance to install missing dependencies (not committed automatically).
- Suggested adding these deps to `requirements.txt`: joblib, seaborn, matplotlib, pandas, scikit-learn.

Quick setup (macOS, recommended)
1. Create & activate a venv:
   ```
   /usr/local/bin/python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   - If you have a requirements file:
     ```
     .venv/bin/python -m pip install -r requirements.txt
     ```
   - Or install common deps:
     ```
     .venv/bin/python -m pip install joblib seaborn matplotlib pandas scikit-learn
     ```

Run scripts
- Train baselines (creates `models/` and `reports/tables/`):
  ```
  .venv/bin/python src/train_baselines.py
  ```
- Generate evaluation figures (needs trained models in `models/`):
  ```
  .venv/bin/python src/evaluationplots.py
  ```

Notes / troubleshooting
- If you see `ModuleNotFoundError`, install the missing package into the project venv.
- If `evaluationplots.py` errors about a missing model, run `train_baselines.py` first.
- Output locations:
  - Models: `models/`
  - Tables: `reports/tables/`
  - Figures: `reports/figures/`


