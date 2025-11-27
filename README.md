# CS-4120-01-Diabetes-Project-ML

Predicting diabetes progression with shared, reproducible splits for regression and classification baselines. We build a median-based label for classification, train the baselines once, save the fitted pipelines, and generate plots from those saved models.

## How to run
1) Install deps  
   ```bash
   pip install -r requirements.txt
   ```
2) Train baselines (single stratified 70/15/15 split reused for both tasks)  
   ```bash
   python3 src/train_baselines.py
   ```  
   Outputs: `models/*.joblib`, `reports/tables/regression_results.csv`, `reports/tables/classification_results.csv`.
3) Generate figures from the saved models (no refit)  
   ```bash
   python3 src/evaluationplots.py
   ```  
   Outputs: four plots in `reports/figures/`.
4) One-shot run  
   ```bash
   python3 src/run_all.py
   ```

## What’s in the results (latest run)
- Regression (val/test MAE | RMSE): Linear Regression 43.58 | 54.30 (val) and 43.36 | 54.25 (test) beats Decision Tree Regressor (56.27 | 69.57 val; 48.17 | 63.71 test), consistent with mostly linear tabular relationships and the tree’s capped depth.
- Classification (val/test Accuracy | F1 | ROC AUC): Logistic Regression 0.73 | 0.74 | 0.83 (val) and 0.76 | 0.78 | 0.84 (test); Decision Tree Classifier 0.79 | 0.80 | 0.84 (val) but drops to 0.70 | 0.72 | 0.78 (test), showing overfitting. Test confusion matrix for logistic regression: [[23 TN, 11 FP], [5 FN, 28 TP]]—more false positives than false negatives.
- Plots (test split, from saved models): target distribution, correlation heatmap, logistic regression confusion matrix, residuals vs predicted for linear regression (funnel/curvature check for bias/heteroscedasticity).
- Scale: target std ≈ 77.1; best RMSE ≈ 54, so the linear model reduces error well below baseline variance.

## Reproducibility notes
- Splits are created once per run via `src/utils.get_split_indices` with `random_state=42` and stratification on the classification label; no external data files are needed.
- Saved artifacts live under `models/` and are consumed by `src/evaluationplots.py`, so tables and plots come from the exact models that were trained.

## Interpretation and next steps
- Linear regression’s lower RMSE vs the tree indicates the relationships are largely linear; adding a modest nonlinear baseline (e.g., gradient boosting or random forest) is the next check before the planned neural net.
- Confusion matrix shows the model errs more on false positives; exploring thresholds or feature scaling/interaction terms may reduce that imbalance. EDA suggests feature correlations are moderate; engineered interactions or mild nonlinearities may help without overfitting.
