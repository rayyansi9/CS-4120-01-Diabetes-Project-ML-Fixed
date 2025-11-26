# CS-4120-01-Diabetes-Project-ML
Set up and running instructions

#Entire code is first being fixed on the deefback from midpoint report and then we're implement a NN.

This project predicts diabetes disease progression using the Diabetes Dataset, which is publicly available in scikit-learn. datasets. We train regression models to predict the continuos progression score and add a media n-based label to build classification baselines.

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
