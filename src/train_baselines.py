# Training baseline regression and classification models once.

#This script creates a single, stratified train/val/test split that is
#shared across regression and classification tasks. Models are trained
#exactly once, saved to ``models/``, and evaluation metrics are written to
#CSV tables under ``reports/tables``.
##

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from data import add_class_label, load_diabetes_df
from utils import ensure_dirs, get_split_indices, regression_metrics


def main() -> None:
    df = load_diabetes_df()
    df, median_y = add_class_label(df)

    splits = get_split_indices(df)
    feature_cols = [col for col in df.columns if col not in {"target", "label"}]

    X = df[feature_cols]
    y_reg = df["target"]
    y_clf = df["label"]

    def subset(indices):
        return X.loc[indices], y_reg.loc[indices], y_clf.loc[indices]

    X_train, y_train_reg, y_train_clf = subset(splits["train"])
    X_val, y_val_reg, y_val_clf = subset(splits["val"])
    X_test, y_test_reg, y_test_clf = subset(splits["test"])

    ensure_dirs(["models", "reports/tables", "data"])

    regression_results = []
    regression_models = {
        "linear_regression": LinearRegression(),
        "decision_tree_regressor": DecisionTreeRegressor(random_state=42, max_depth=4),
    }

    for name, model in regression_models.items():
        model.fit(X_train, y_train_reg)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        val_metrics = regression_metrics(y_val_reg, val_pred)
        test_metrics = regression_metrics(y_test_reg, test_pred)

        regression_results.append(
            {
                "Model": name.replace("_", " ").title(),
                "Val MAE": val_metrics["mae"],
                "Val MSE": val_metrics["mse"],
                "Val RMSE": val_metrics["rmse"],
                "Test MAE": test_metrics["mae"],
                "Test MSE": test_metrics["mse"],
                "Test RMSE": test_metrics["rmse"],
            }
        )

        joblib.dump(model, Path("models") / f"{name}.joblib")

    reg_df = pd.DataFrame(regression_results)
    reg_df.to_csv("reports/tables/regression_results.csv", index=False)

    classification_results = []
    classification_models = {
        "logistic_regression": LogisticRegression(max_iter=2000),
        "decision_tree_classifier": DecisionTreeClassifier(random_state=42, max_depth=4),
    }

    for name, model in classification_models.items():
        model.fit(X_train, y_train_clf)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        classification_results.append(
            {
                "Model": name.replace("_", " ").title(),
                "Val Accuracy": accuracy_score(y_val_clf, val_pred),
                "Val F1": f1_score(y_val_clf, val_pred),
                "Test Accuracy": accuracy_score(y_test_clf, test_pred),
                "Test F1": f1_score(y_test_clf, test_pred),
            }
        )

        joblib.dump(model, Path("models") / f"{name}.joblib")

    clf_df = pd.DataFrame(classification_results)
    clf_df.to_csv("reports/tables/classification_results.csv", index=False)

    print("Saved models to models/ and metrics to reports/tables/.")
    print(f"Median target used for label thresholding: {median_y:.4f}")


if __name__ == "__main__":
    main()
