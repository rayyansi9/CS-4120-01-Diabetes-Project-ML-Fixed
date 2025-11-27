# CS-4120-01-Diabetes-Project-ML

Predicting diabetes progression with shared, reproducible splits for regression and classification baselines. We build a median-based label for classification, train the baselines once, save the fitted pipelines, and generate plots from those saved models.

## How to run
1) Install deps  
   ```bash
   pip install -r requirements.txt
   ```
2) Train classical baselines (single stratified 70/15/15 split reused for both tasks)  
   ```bash
   python3 src/train_baselines.py
   ```  
   Outputs: `models/*.joblib`, MLflow runs under `mlruns/`, `reports/tables/regression_results.csv`, `reports/tables/classification_results.csv`, `reports/best_runs.json` (best MLflow run ids).
3) Train neural nets (classification + regression) with learning curves logged to MLflow  
   ```bash
   python3 src/train_nn.py
   ```  
   Outputs: `reports/history/*.csv`, `reports/tables/nn_*_results.csv`, updates `reports/best_runs.json` with NN and overall-best runs.
4) Generate figures and comparison tables from MLflow artifacts (no refit)  
   ```bash
   python3 src/evaluate.py
   ```  
   Outputs: required 5 plots in `reports/figures/` and 2 comparison tables in `reports/tables/`.
5) One-shot run (baselines + NN + evaluation)  
   ```bash
   python3 src/run_all.py
   ```
6) Inspect MLflow UI (optional)  
   ```bash
   mlflow ui --backend-store-uri file:mlruns
   ```

## What’s in the results (latest run)
- Regression (val/test MAE | RMSE): Linear Regression 43.58 | 54.30 (val) and 43.36 | 54.25 (test); NN MLP 38.99 | 51.65 (val) and 45.41 | 56.49 (test). Linear remains the best test performer.
- Classification (val/test Accuracy | F1 | ROC AUC): Decision Tree 0.79 | 0.80 | 0.84 (val) and 0.70 | 0.72 | 0.78 (test); tuned NN MLP 0.78 | 0.78 | 0.884 (val) and 0.78 | 0.789 | 0.844 (test). NN wins on ROC AUC and F1; tree still slightly higher val accuracy. Logistic regression test F1 is 0.778 for threshold context.
- NN hyperparameter search: 3-config grid per task; best run IDs recorded in `reports/best_runs.json` (see `classification_nn` and `regression_nn` entries).
- Required plots (reports/figures): `plot1_classification_learning_curve.png`, `plot2_regression_learning_curve.png`, `plot3_confusion_matrix.png` (best final classifier), `plot4_residuals_vs_predicted.png` (best final regressor), `plot5_feature_importance.png` (permutation importance on best classifier).
- Required tables (reports/tables): `table1_classification_comparison.csv` (classical vs NN with Accuracy/F1/ROC-AUC) and `table2_regression_comparison.csv` (classical vs NN with MAE/RMSE).

## Reproducibility notes
- Splits are created once per run via `src/utils.get_split_indices` with `random_state=42` and stratification on the classification label; no external data files are needed.
- Saved artifacts live under `mlruns/` (MLflow), plus `reports/best_runs.json` capturing the chosen run IDs. `src/evaluationplots.py` loads models from those MLflow artifacts, so tables and plots come from the exact models that were trained.

## MLflow usage and splits policy
- Tracking: all runs are logged locally to `mlruns/` with tracking URI set to `file:mlruns`. Launch the UI with `mlflow ui --backend-store-uri file:mlruns`.
- Artifacts: best run IDs for classical and NN models are recorded in `reports/best_runs.json`; plots and comparison tables are generated from these artifacts—no refitting in evaluation.
- Splits: the train/val/test indices are persisted to `reports/splits.npz` when you run `python3 src/train_baselines.py` (or `python3 src/run_all.py`). This keeps evaluation perfectly aligned with training. If you want a fresh split, delete `reports/splits.npz` and rerun `python3 src/train_baselines.py` to regenerate. Avoid manual editing of this file.

## Interpretation and next steps
- Linear beats the tree on test RMSE, so signal looks mostly linear; try gradient boosting/random forest to capture mild curvature without overfitting.
- Classification NN improves ROC-AUC/F1; calibrate probabilities or tune thresholds to squeeze accuracy. Class-weighting could address remaining positive-class confusion.
