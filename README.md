# CS-4120-01-Diabetes-Project-ML

In this project we're trying to predict diabetes progression using 1) Regression is where we calculate the actual progression score and 2) Classification is where we label each case as high or low progression using the median value. After our midpoint feedback, we are now only using one fixed train/validation/test split for everything. In addition, all the models are logged with MLflow, and evaluation loads the saved runs directly, so nothing is being retrained behind.

## Start here
- Install packages: `pip install -r requirements.txt`
- Train Baselines (classical models): `python src/train_baselines.py`
- Train Neural networks (both tasks): `python src/train_nn.py`
- Generate Plots & comparison tables: `python src/evaluate.py`
- Can also now run the full pipeline at once by running: `python src/run_all.py`
- Optional can launch MLflow UI: `mlflow ui --backend-store-uri file:mlruns`
- Extra work : Refreshing older plots in notebook (kept for reference after working on midpoint feedback, residuals histogram only changed to residuals vs predicted too): `python notebooks/refresh_notebook_figures.py`

## Setup (to ensure no issues are faced before running)
- Create/activate a venv (recommended): `python3 -m venv .venv && source .venv/bin/activate` (or your IDE’s venv).
- Upgrade pip tooling if needed: `pip install --upgrade pip setuptools wheel`.
- Install packages: `pip install -r requirements.txt`.
- Run commands from the repo root so relative paths work.
- `mlruns/` is gitignored in the start, it will be created when you run the training scripts.

## Running in an IDE (e.g., PyCharm)
- Point the project to the `.venv` you created (Project Interpreter / Python env).
- Set the working directory of run configs to the repo root.
- Use the commands above (e.g., `python src/run_all.py` or `python src/evaluate.py`).
- If you see `ModuleNotFoundError`, please doubleccheck for the installation of `requirements.txt` into that interpreter.

## What and where's the output
- Plots in `reports/figures/` (learning curves, confusion matrix, residuals, permutation importance).
- Tables in `reports/tables/` (classical vs neural network results for classification and regression).
- Saved models in `models/` and all MLflow runs in `mlruns/`.

## Reproducibility
- Data: we use the built-in sklearn diabetes dataset, needs nothing extra to download.
  The train/val/test split is now fixed and saved in reports/splits.npz.
- Tracking: the best model IDs for tuning are stored in reports/best_runs.json.
  The evaluation script loads these exact runs to recreate the plots and tables.
- Seeds: random_state=42 is used for the split and for permutation importance.
  If you ever want a new split, you can delete reports/splits.npz and retrain the baselines.

## Quick summary of our results (based on the latest run)
Regression: Linear Regression stays the most dependable model on the test RMSE.
The neural network performed close to it but was a bit more variable.

Classification: The tuned neural network achieves higher ROC-AUC and slightly better F1.
The Decision Tree still has a marginally higher validation accuracy, but the NN is stronger overall for ranking and prediction.
