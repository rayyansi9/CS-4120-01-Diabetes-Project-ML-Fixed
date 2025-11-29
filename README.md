# CS-4120-01-Diabetes-Project-ML

In this project we're trying to predict diabetes progression using 1) Regression is where we calculate the actual progression score and 2) Classification is where we label each case as high or low progression using the median value. After our midpoint feedback, we are now only using one fixed train/validation/test split for everything. In addition, all the models are logged with MLflow, and evaluation loads the saved runs directly, so nothing is being retrained behind.

## Quick start
- Install deps: `pip install -r requirements.txt`
- Baselines (classical models): `python src/train_baselines.py`
- Neural nets (both tasks): `python src/train_nn.py`
- Plots + comparison tables: `python src/evaluate.py`
- Everything in one shot: `python src/run_all.py`
- Optional MLflow UI: `mlflow ui --backend-store-uri file:mlruns`
- Refresh legacy notebook plots (kept for reference): `python notebooks/refresh_notebook_figures.py`

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
