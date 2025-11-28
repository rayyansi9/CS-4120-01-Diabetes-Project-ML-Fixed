# Final Report — DiabetesG1

Authors: Muhammad Rayyan, Akaashdeep Anand  
Repository: https://github.com/rayyansi9/CS-4120-01-Diabetes-Project-ML-Fixed.git

We tried two jobs: guess the diabetes score (a number) and guess if the score is “high” or “low” by splitting on the median. We used the built-in sklearn diabetes dataset, kept one fixed train/val/test split for everything, and logged all runs in MLflow so we can rerun or inspect later. The simple neural net classifier beats the classic models on ranking patients (ROC-AUC ~0.84 on test). For predicting the raw number, plain linear regression is still the safest pick on this small dataset.

## 1) What we did
- Data: sklearn diabetes (10 numeric features). No extra cleaning needed.
- Labels: “high” if the target is above the median (140.5), otherwise “low.”
- Split: one stratified 70/15/15 split saved in `reports/splits.npz` and reused everywhere.
- Outputs: 5 plots, 2 comparison tables, saved models, and a JSON pointing to the best runs.

## 2) Models (in plain English)
- Classic baselines: linear regression and a small decision tree for the number; logistic regression and a small decision tree for the high/low label.
- Neural nets: tiny two-layer MLPs (64 and 32 hidden units) with ReLU. We standardize the features first, then train for a few dozen passes. No dropout or other regularization beyond the built-in L2 weight decay (alpha). Optimizer is Adam with a small learning rate (around 1e-3); batch size is 16–32 depending on the search config.
- We tried a few settings (learning rates, sizes, activation, alpha, batch size, epochs) via a small grid and kept the best ones logged in MLflow.

## 3) How we measured things
- Classification: Accuracy and F1 (did we pick the right class) and ROC-AUC (did we rank high-risk people above low-risk people). ROC-AUC is our main lens because it doesn’t depend on one threshold.
- Regression: RMSE and MAE. RMSE lets us compare errors to the natural spread of the target.

## 4) What turned out best
- Classification: the MLP wins. ROC-AUC on test is ~0.84, a bit higher than the decision tree. F1 also nudges up (about 0.79 vs ~0.78).
- Regression: linear regression is steady. Its test RMSE is about 54, which is roughly 70% of the target’s natural spread (~77). The NN looked better on validation but slipped on test, so we stick with linear.

## 5) Plots and tables (all under `reports/`)
- Plots: learning curves for both NNs (`plot1`, `plot2`), test confusion matrix (`plot3`), residuals for the regressor (`plot4`), and feature importance for the classifier (`plot5`).
- Tables: `table1_classification_comparison.csv` (classic vs NN for high/low) and `table2_regression_comparison.csv` (classic vs NN for the number).

## 6) Takeaways in normal words
- The classifier NN picks up some nonlinear patterns, so it ranks patients better than the tree or logistic regression. Gains are small but consistent.
- The regression NN overfits a bit; with only 442 rows, the simple linear model generalizes better.
- BMI and blood pressure show up as the most important features for the classifier, which matches common sense.
- Errors on the regression side get wider for higher predictions, hinting that a tree ensemble or more data might help.

## 7) What changed since midpoint
- Added tiny NN baselines with a mini grid search (hidden sizes, activation, learning rate, alpha, batch size, epochs).
- Standardized inputs inside the NN pipelines (StandardScaler) to keep training stable.
- Logged everything in MLflow and saved the one split to `reports/splits.npz`, so evaluation reuses the exact data slices instead of refitting.
- Result: classification NN ROC-AUC improved a few points over the tree/logistic; regression NN did not beat linear on test, so linear stays best there.

## 8) Risks, ethics, and limits
- Small dataset → results move around; try cross-validation to steady them.
- Slight class imbalance → consider class weights or threshold tuning if deploying.
- Next experiments: gradient boosting or random forests for both tasks, plus a touch of regularization/dropout for the NNs to fight overfitting.
- Privacy/bias: dataset is de-identified but small; performance may not generalize to other groups—validate before use.
- Leakage: we only use the built-in features, and we reuse the same split for all runs to avoid accidental drift.
