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
import numpy as np

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from data import add_class_label, load_diabetes_df
from utils import ensure_dirs, get_split_indices, regression_metrics

MODELS_DIR = Path("models")
TABLES_DIR = Path("reports/tables")


def main() -> None:
    df = load_diabetes_df()
    df, median_y = add_class_label(df)

    splits = get_split_indices(df)

    feature_cols = [c for c in df.columns if c not in {"target", "label"}]
    X = df[feature_cols]
    y_clf = df["label"]
    y_reg = df["target"]

    # Indices
    X_train = X.loc[splits["train"]]
    X_val = X.loc[splits["val"]]
    X_test = X.loc[splits["test"]]

    y_train_clf = y_clf.loc[splits["train"]]
    y_val_clf = y_clf.loc[splits["val"]]
    y_test_clf = y_clf.loc[splits["test"]]

    y_train_reg = y_reg.loc[splits["train"]]
    y_val_reg = y_reg.loc[splits["val"]]
    y_test_reg = y_reg.loc[splits["test"]]

    ensure_dirs([MODELS_DIR, TABLES_DIR])

    # Models
    regressors = {
        "linear_regression": LinearRegression(),
        "decision_tree_regressor": DecisionTreeRegressor(max_depth=4, random_state=42),
    }

    classifiers = {
        "logistic_regression": LogisticRegression(max_iter=1000, solver="lbfgs"),
        "decision_tree_classifier": DecisionTreeClassifier(max_depth=4, random_state=42),
    }

    # Train regressors and record metrics
    reg_rows = []
    for name, model in regressors.items():
        model.fit(X_train, y_train_reg)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        m_val = regression_metrics(y_val_reg, y_val_pred)
        m_test = regression_metrics(y_test_reg, y_test_pred)

        reg_rows.append(
            {
                "model": name,
                "val_mae": m_val["mae"],
                "val_rmse": m_val["rmse"],
                "test_mae": m_test["mae"],
                "test_rmse": m_test["rmse"],
            }
        )
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")

    # Train classifiers and record metrics
    clf_rows = []
    for name, model in classifiers.items():
        model.fit(X_train, y_train_clf)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        # predict_proba for ROC AUC (fallback if not available)
        try:
            y_val_proba = model.predict_proba(X_val)[:, 1]
            y_test_proba = model.predict_proba(X_test)[:, 1]
        except Exception:
            # fallback to decision function or binary predictions
            try:
                y_val_proba = model.decision_function(X_val)
                y_test_proba = model.decision_function(X_test)
            except Exception:
                y_val_proba = y_val_pred
                y_test_proba = y_test_pred

        clf_rows.append(
            {
                "model": name,
                "val_accuracy": accuracy_score(y_val_clf, y_val_pred),
                "val_f1": f1_score(y_val_clf, y_val_pred),
                "val_roc_auc": float(roc_auc_score(y_val_clf, y_val_proba)),
                "test_accuracy": accuracy_score(y_test_clf, y_test_pred),
                "test_f1": f1_score(y_test_clf, y_test_pred),
                "test_roc_auc": float(roc_auc_score(y_test_clf, y_test_proba)),
            }
        )
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")

    # Persist tables
    pd.DataFrame(reg_rows).to_csv(TABLES_DIR / "regression_results.csv", index=False)
    pd.DataFrame(clf_rows).to_csv(TABLES_DIR / "classification_results.csv", index=False)

    # Print a short summary
    target_std = float(y_reg.std())
    print("Saved models to models/ and metrics to reports/tables/")
    print(f"Median target used for label thresholding: {median_y:.4f}")
    print(f"Target standard deviation: {target_std:.4f}")


if __name__ == "__main__":
    main()
