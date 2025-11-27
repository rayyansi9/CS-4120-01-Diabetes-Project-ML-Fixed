# Final Report — DiabetesG1

Authors: Muhammad Rayyan, Akaashdeep Anand  
Repository: https://github.com/rayyansi9/CS-4120-01-Diabetes-Project-ML-Fixed.git

We trained classical baselines and shallow neural networks on the sklearn diabetes dataset, tracked all runs with MLflow, and produced reproducible splits, figures, and tables. A small hyperparameter search lifted the NN classifier to val/test ROC-AUC 0.884/0.844 (+0.044/+0.066 over the best classical model). The NN regressor improved validation RMSE vs linear regression (51.65 vs 54.30) but still trails on test RMSE (+2.24).

## 1. Problem and deliverables
- Tasks: predict continuous diabetes progression (regression) and a median-split binary label (high vs low progression) for classification.
- Deliverables: single reproducible split file, two classical baselines per task, NN baselines, MLflow tracking, exactly five figures and two tables.

## 2. Data and preprocessing
- Dataset: sklearn diabetes (10 numeric features). No external raw files.
- Labeling: binary label uses target median = 140.5 (label = 1 if target >= median).
- Splits: one stratified train/val/test split (70/15/15) with `random_state=42`, persisted at `reports/splits.npz` and reused by all scripts.
- Preprocessing: classical models use raw features; NN pipelines standardize features with `StandardScaler` inside a sklearn `Pipeline`.

## 3. Models and preprocessing
- Regression baselines: Linear Regression; Decision Tree Regressor (max_depth=4).
- Classification baselines: Logistic Regression (lbfgs, max_iter=1000); Decision Tree Classifier (max_depth=4).
- NN classifier (sklearn MLP): hidden layers (64, 32), ReLU, Adam lr=1e-3, alpha=1e-4, batch_size=32, 40 epochs via `partial_fit`; StandardScaler + MLP pipeline; no dropout.
- NN regressor (sklearn MLP): hidden layers (64, 32), ReLU, Adam lr=1e-3, alpha=1e-4, batch_size=32, 60 epochs via `partial_fit`; StandardScaler + MLP pipeline; no dropout.

## 4. Hyperparameter tuning
- Method: small grid search (3 configs per task) logged in MLflow.
- Classification search: hidden sizes [(64,32), (128,), (64,32)], activations [relu, tanh], learning rates [1e-3, 5e-4], alpha [1e-4, 5e-4], batch_size [16,32], epochs [40–50]. Best: (64,32) ReLU, lr=1e-3, alpha=1e-4, batch_size=32, 40 epochs.
- Regression search: hidden sizes [(64,32), (128,), (64,32)], activations [relu, tanh], learning rates [1e-3, 5e-4], alpha [1e-4, 5e-4], batch_size [16,32], epochs [60–80]. Best: (64,32) ReLU, lr=1e-3, alpha=1e-4, batch_size=32, 60 epochs.

## 5. Experiment tracking and reproducibility
- MLflow tracking URI: `file:mlruns`; run IDs recorded in `reports/best_runs.json`.
- Artifacts: models logged under `model/`, histories under `reports/history/*.csv`, comparison tables under `reports/tables/`.
- Commands (macOS):  
  ```bash
  /usr/local/bin/python3 -m venv .venv
  source .venv/bin/activate
  .venv/bin/python -m pip install -r requirements.txt
  .venv/bin/python src/run_all.py
  mlflow ui --backend-store-uri file://$(pwd)/mlruns --port 5000
  ```

## 6. Results (latest run)
Tables (reports/tables):
- `table1_classification_comparison.csv`: Decision Tree vs MLP (Accuracy, F1, ROC-AUC).
- `table2_regression_comparison.csv`: Linear Regression vs MLP (MAE, RMSE).

Key numbers:
- Classification: Decision Tree val/test ROC-AUC 0.840/0.778; NN MLP val/test ROC-AUC 0.884/0.844. Test F1: logistic regression 0.778 vs NN 0.789.
- Regression: Linear Regression val/test RMSE 54.30/54.25; MLP val/test RMSE 51.65/56.49.
- Target context: target std = 77.01 → best test RMSE (54.25) is ~0.70 std.

Figures (reports/figures):
- `plot1_classification_learning_curve.png` — MLP classification metrics across epochs.
- `plot2_regression_learning_curve.png` — MLP regression loss metrics across epochs.
- `plot3_confusion_matrix.png` — Best final classifier on the test set.
- `plot4_residuals_vs_predicted.png` — Best final regressor residuals.
- `plot5_feature_importance.png` — Permutation importance for best classifier.

## 7. Analysis
- Classical vs NN: the tuned NN gains ROC-AUC and F1, suggesting better ranking and modest thresholded performance; calibration or threshold tuning could squeeze a bit more accuracy. The regression NN still overfits relative to linear regression on test, pointing to limited signal and small data.
- Confusion matrix: most confusion is on the positive class → consider class weights or threshold adjustment.
- Residuals: regression errors widen at higher predictions, hinting at mild heteroscedasticity; a tree-based ensemble might capture this curvature.
- Feature importance: BMI and blood pressure rank highly; adding interaction terms or monotonic constraints could stabilize performance.

## 8. Improvements vs midpoint
- Persisted and reused splits (`reports/splits.npz`); evaluation now loads saved models instead of refitting.
- Added ROC-AUC/F1 tracking and MLflow logging for all runs; comparison tables now include both metrics.
- Added small NN hyperparameter search; classification NN ROC-AUC improved from 0.865→0.884 (val) and 0.807→0.844 (test). Regression NN unchanged; linear regression remains the stronger test performer (54.25 vs 56.49 RMSE).

## 9. Risks, ethics, and mitigations
- Class imbalance: stratified splits; report F1 and ROC-AUC; consider class weights in future.
- Leakage: only raw sklearn features; single persisted split reused across scripts to avoid drift.
- Privacy/bias: dataset is de-identified; generalization to other populations is uncertain—flag for domain validation before deployment.

## 10. Limitations and next steps
- Small dataset and limited features constrain NN gains; try gradient boosting/random forest and stronger regularization or dropout for the NN.
- Add cross-validation or repeated splits to reduce variance; calibrate probabilities and tune thresholds for classification.
- Explore simple feature engineering (interactions, polynomial/monotonic constraints) and mild ensembling to stabilize regression performance.

## 11. PDF export
To satisfy the PDF submission requirement, convert this file with pandoc (if installed):
```bash
pandoc reports/final_report.md -o "final report GroupNN.pdf"
```
