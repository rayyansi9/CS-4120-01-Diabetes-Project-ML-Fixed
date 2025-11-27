# Train the classical regression and classification baselines once.
# One stratified train/val/test split is shared by both tasks, models land in ``models/``,
# and metrics go to CSVs under ``reports/tables``.

from __future__ import annotations

import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import mlflow
import mlflow.sklearn

from data import add_class_label, load_diabetes_df
from utils import ensure_dirs, get_split_indices, regression_metrics

MODELS_DIR = Path("models")
TABLES_DIR = Path("reports/tables")
BEST_RUN_PATH = Path("reports/best_runs.json")
MLRUNS_DIR = Path("mlruns")
SPLITS_PATH = Path("reports/splits.npz")
EXPERIMENT_NAME = "diabetes_baselines"


def main() -> None:
    df = load_diabetes_df()
    df, median_y = add_class_label(df)

    # Use the local mlruns folder for tracking
    mlflow.set_tracking_uri(f"file:{MLRUNS_DIR.resolve()}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    splits = get_split_indices(df)

    feature_cols = [c for c in df.columns if c not in {"target", "label"}]
    X = df[feature_cols]
    y_clf = df["label"]
    y_reg = df["target"]

    # Save the split so every script uses the same fold
    np.savez(SPLITS_PATH, train=splits["train"], val=splits["val"], test=splits["test"])

    # Grab the split slices
    X_train = X.loc[splits["train"]]
    X_val = X.loc[splits["val"]]
    X_test = X.loc[splits["test"]]

    y_train_clf = y_clf.loc[splits["train"]]
    y_val_clf = y_clf.loc[splits["val"]]
    y_test_clf = y_clf.loc[splits["test"]]

    y_train_reg = y_reg.loc[splits["train"]]
    y_val_reg = y_reg.loc[splits["val"]]
    y_test_reg = y_reg.loc[splits["test"]]

    ensure_dirs([MODELS_DIR, TABLES_DIR, BEST_RUN_PATH.parent, MLRUNS_DIR, SPLITS_PATH.parent])

    # Baseline models to try
    regressors = {
        "linear_regression": LinearRegression(),
        "decision_tree_regressor": DecisionTreeRegressor(max_depth=4, random_state=42),
    }

    classifiers = {
        "logistic_regression": LogisticRegression(max_iter=1000, solver="lbfgs"),
        "decision_tree_classifier": DecisionTreeClassifier(max_depth=4, random_state=42),
    }

    # Train regressors and log metrics
    reg_rows = []
    best_reg = None
    for name, model in regressors.items():
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_param("task", "regression")
            mlflow.log_param("model", name)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("val_size", 0.15)
            mlflow.log_param("test_size", 0.15)
            if hasattr(model, "get_params"):
                params = model.get_params()
                if "max_depth" in params:
                    mlflow.log_param("max_depth", params["max_depth"])

            model.fit(X_train, y_train_reg)
            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)

            m_val = regression_metrics(y_val_reg, y_val_pred)
            m_test = regression_metrics(y_test_reg, y_test_pred)

            mlflow.log_metrics(
                {
                    "val_mae": m_val["mae"],
                    "val_rmse": m_val["rmse"],
                    "test_mae": m_test["mae"],
                    "test_rmse": m_test["rmse"],
                }
            )
            mlflow.sklearn.log_model(model, artifact_path="model")

            row = {
                "model": name,
                "source": "classical",
                "val_mae": m_val["mae"],
                "val_rmse": m_val["rmse"],
                "test_mae": m_test["mae"],
                "test_rmse": m_test["rmse"],
                "run_id": run.info.run_id,
            }
            reg_rows.append(row)
            if best_reg is None or row["val_rmse"] < best_reg["val_rmse"]:
                best_reg = row

            joblib.dump(model, MODELS_DIR / f"{name}.joblib")

    # Train classifiers and record metrics
    clf_rows = []
    best_clf = None
    for name, model in classifiers.items():
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_param("task", "classification")
            mlflow.log_param("model", name)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("val_size", 0.15)
            mlflow.log_param("test_size", 0.15)
            if hasattr(model, "get_params"):
                params = model.get_params()
                if "max_depth" in params:
                    mlflow.log_param("max_depth", params["max_depth"])
                if "solver" in params:
                    mlflow.log_param("solver", params["solver"])
                if "max_iter" in params:
                    mlflow.log_param("max_iter", params["max_iter"])

            model.fit(X_train, y_train_clf)
            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)

            # Try predict_proba first for ROC AUC
            try:
                y_val_proba = model.predict_proba(X_val)[:, 1]
                y_test_proba = model.predict_proba(X_test)[:, 1]
            except Exception:
                # Fall back to decision_function or the raw preds
                try:
                    y_val_proba = model.decision_function(X_val)
                    y_test_proba = model.decision_function(X_test)
                except Exception:
                    y_val_proba = y_val_pred
                    y_test_proba = y_test_pred

            metrics = {
                "val_accuracy": accuracy_score(y_val_clf, y_val_pred),
                "val_f1": f1_score(y_val_clf, y_val_pred),
                "val_roc_auc": float(roc_auc_score(y_val_clf, y_val_proba)),
                "test_accuracy": accuracy_score(y_test_clf, y_test_pred),
                "test_f1": f1_score(y_test_clf, y_test_pred),
                "test_roc_auc": float(roc_auc_score(y_test_clf, y_test_proba)),
            }
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path="model")

            row = {"model": name, "source": "classical", **metrics, "run_id": run.info.run_id}
            clf_rows.append(row)
            if best_clf is None or row["val_roc_auc"] > best_clf["val_roc_auc"]:
                best_clf = row

            joblib.dump(model, MODELS_DIR / f"{name}.joblib")

    # Save results tables
    pd.DataFrame(reg_rows).to_csv(TABLES_DIR / "regression_results.csv", index=False)
    pd.DataFrame(clf_rows).to_csv(TABLES_DIR / "classification_results.csv", index=False)

    # Save a manifest so evaluation can reload the right runs
    manifest = {
        "split": {"val_size": 0.15, "test_size": 0.15, "random_state": 42},
        "label_threshold": median_y,
        "best_models": {
            "regression_classical": {
                "model": best_reg["model"],
                "source": "classical",
                "run_id": best_reg["run_id"],
                "metric": "val_rmse",
                "val_mae": best_reg["val_mae"],
                "val_rmse": best_reg["val_rmse"],
                "test_mae": best_reg["test_mae"],
                "test_rmse": best_reg["test_rmse"],
                "artifact_path": "model",
            },
            "classification_classical": {
                "model": best_clf["model"],
                "source": "classical",
                "run_id": best_clf["run_id"],
                "metric": "val_roc_auc",
                "val_accuracy": best_clf["val_accuracy"],
                "val_f1": best_clf["val_f1"],
                "val_roc_auc": best_clf["val_roc_auc"],
                "test_accuracy": best_clf["test_accuracy"],
                "test_f1": best_clf["test_f1"],
                "test_roc_auc": best_clf["test_roc_auc"],
                "artifact_path": "model",
            },
        },
    }
    BEST_RUN_PATH.write_text(json.dumps(manifest, indent=2))

    # Quick printout
    target_std = float(y_reg.std())
    print("Saved models to models/ and metrics to reports/tables/")
    print(f"Median target used for label thresholding: {median_y:.4f}")
    print(f"Target standard deviation: {target_std:.4f}")
    print(f"Best regression (classical): {manifest['best_models']['regression_classical']}")
    print(f"Best classification (classical): {manifest['best_models']['classification_classical']}")


if __name__ == "__main__":
    main()
