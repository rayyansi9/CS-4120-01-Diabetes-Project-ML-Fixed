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
   python3 src/evaluationplots.py
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
- Regression (val/test MAE | RMSE): Linear Regression 43.58 | 54.30 (val) and 43.36 | 54.25 (test); NN MLP 38.99 | 51.65 (val) and 45.41 | 56.49 (test). Classical linear still edges the NN on test RMSE.
- Classification (val/test Accuracy | F1 | ROC AUC): Decision Tree 0.79 | 0.80 | 0.84 (val) and 0.70 | 0.72 | 0.78 (test); NN MLP 0.76 | 0.76 | 0.86 (val) and 0.72 | 0.72 | 0.81 (test). NN wins on ROC AUC; tree slightly higher val accuracy but overfits.
- Required plots (reports/figures): `plot1_classification_learning_curve.png`, `plot2_regression_learning_curve.png`, `plot3_confusion_matrix.png` (best final classifier), `plot4_residuals_vs_predicted.png` (best final regressor), `plot5_feature_importance.png` (permutation importance on best classifier). Older exploratory plots remain but are not used in the final report.
- Required tables (reports/tables): `table1_classification_comparison.csv` (classical vs NN with Accuracy/F1/ROC-AUC) and `table2_regression_comparison.csv` (classical vs NN with MAE/RMSE).

## Reproducibility notes
- Splits are created once per run via `src/utils.get_split_indices` with `random_state=42` and stratification on the classification label; no external data files are needed.
- Saved artifacts live under `mlruns/` (MLflow), plus `reports/best_runs.json` capturing the chosen run IDs. `src/evaluationplots.py` loads models from those MLflow artifacts, so tables and plots come from the exact models that were trained.

## Interpretation and next steps
- Linear regression’s lower RMSE vs the tree indicates the relationships are largely linear; adding a modest nonlinear baseline (e.g., gradient boosting or random forest) is the next check before the planned neural net.
- Confusion matrix shows the model errs more on false positives; exploring thresholds or feature scaling/interaction terms may reduce that imbalance. EDA suggests feature correlations are moderate; engineered interactions or mild nonlinearities may help without overfitting.
