"""Train neural network baselines for regression and classification.

This script:
- Reuses the same stratified train/val/test split as classical baselines.
- Runs a small hyperparameter search over shallow MLPs with manual epoch control to log learning curves.
- Logs metrics and artifacts to MLflow under the `mlruns/` directory.
- Writes NN metric tables to `reports/tables/` and updates `reports/best_runs.json`
  with best NN runs plus overall best (classical vs NN) per task.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

from data import add_class_label, load_diabetes_df
from utils import ensure_dirs, get_split_indices

TABLES_DIR = Path("reports/tables")
HISTORY_DIR = Path("reports/history")
BEST_RUN_PATH = Path("reports/best_runs.json")
MLRUNS_DIR = Path("mlruns")
SPLITS_PATH = Path("reports/splits.npz")
EXPERIMENT_NAME = "diabetes_nn"


def load_manifest() -> Dict:
    """Load or initialize the run manifest shared with baselines."""
    if BEST_RUN_PATH.exists():
        return json.loads(BEST_RUN_PATH.read_text())
    raise FileNotFoundError(
        f"{BEST_RUN_PATH} not found. Run `python3 src/train_baselines.py` first."
    )


def load_splits(df):
    if SPLITS_PATH.exists():
        data = np.load(SPLITS_PATH)
        return {k: data[k] for k in ["train", "val", "test"]}
    return get_split_indices(df)


def log_history(name: str, history: List[Dict]) -> Path:
    df_hist = pd.DataFrame(history)
    ensure_dirs([HISTORY_DIR])
    path = HISTORY_DIR / f"{name}_history.csv"
    df_hist.to_csv(path, index=False)
    return path


def train_classifier_nn(X_train, y_train, X_val, y_val, params: Dict, random_state: int):
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)

    model = MLPClassifier(
        hidden_layer_sizes=params["hidden_layer_sizes"],
        activation=params["activation"],
        solver="adam",
        learning_rate_init=params["lr"],
        alpha=params["alpha"],
        max_iter=1,  # single epoch per fit
        warm_start=True,
        batch_size=params["batch_size"],
        random_state=random_state,
    )

    history = []
    best_model = None
    best_val = -np.inf

    for epoch in range(params["epochs"]):
        model.partial_fit(X_train_s, y_train, classes=np.array([0, 1]))

        train_pred = model.predict(X_train_s)
        train_proba = model.predict_proba(X_train_s)[:, 1]
        val_pred = model.predict(X_val_s)
        val_proba = model.predict_proba(X_val_s)[:, 1]

        train_acc = accuracy_score(y_train, train_pred)
        train_f1 = f1_score(y_train, train_pred)
        train_auc = roc_auc_score(y_train, train_proba)
        val_acc = accuracy_score(y_val, val_pred)
        val_f1 = f1_score(y_val, val_pred)
        val_auc = roc_auc_score(y_val, val_proba)

        history.append(
            {
                "epoch": epoch + 1,
                "train_accuracy": train_acc,
                "train_f1": train_f1,
                "train_roc_auc": train_auc,
                "val_accuracy": val_acc,
                "val_f1": val_f1,
                "val_roc_auc": val_auc,
            }
        )

        mlflow.log_metrics(
            {
                "train_accuracy": train_acc,
                "train_f1": train_f1,
                "train_roc_auc": train_auc,
                "val_accuracy": val_acc,
                "val_f1": val_f1,
                "val_roc_auc": val_auc,
            },
            step=epoch,
        )

        if val_auc > best_val:
            best_val = val_auc
            best_model = deepcopy(model)

    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([("scaler", scaler), ("model", best_model)])
    return pipeline, history


def train_regressor_nn(X_train, y_train, X_val, y_val, params: Dict, random_state: int):
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)

    model = MLPRegressor(
        hidden_layer_sizes=params["hidden_layer_sizes"],
        activation=params["activation"],
        solver="adam",
        learning_rate_init=params["lr"],
        alpha=params["alpha"],
        max_iter=1,
        warm_start=True,
        batch_size=params["batch_size"],
        random_state=random_state,
    )

    history = []
    best_model = None
    best_val = np.inf

    for epoch in range(params["epochs"]):
        model.partial_fit(X_train_s, y_train)

        train_pred = model.predict(X_train_s)
        val_pred = model.predict(X_val_s)

        train_mae = mean_absolute_error(y_train, train_pred)
        train_rmse = float(np.sqrt(mean_squared_error(y_train, train_pred)))
        val_mae = mean_absolute_error(y_val, val_pred)
        val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))

        history.append(
            {
                "epoch": epoch + 1,
                "train_mae": train_mae,
                "train_rmse": train_rmse,
                "val_mae": val_mae,
                "val_rmse": val_rmse,
            }
        )

        mlflow.log_metrics(
            {
                "train_mae": train_mae,
                "train_rmse": train_rmse,
                "val_mae": val_mae,
                "val_rmse": val_rmse,
            },
            step=epoch,
        )

        if val_rmse < best_val:
            best_val = val_rmse
            best_model = deepcopy(model)

    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([("scaler", scaler), ("model", best_model)])
    return pipeline, history


def main() -> None:
    df = load_diabetes_df()
    df, median_y = add_class_label(df)
    splits = load_splits(df)

    feature_cols = [c for c in df.columns if c not in {"target", "label"}]
    X = df[feature_cols]
    y_clf = df["label"]
    y_reg = df["target"]

    X_train, X_val, X_test = (
        X.loc[splits["train"]],
        X.loc[splits["val"]],
        X.loc[splits["test"]],
    )
    y_train_clf, y_val_clf, y_test_clf = (
        y_clf.loc[splits["train"]],
        y_clf.loc[splits["val"]],
        y_clf.loc[splits["test"]],
    )
    y_train_reg, y_val_reg, y_test_reg = (
        y_reg.loc[splits["train"]],
        y_reg.loc[splits["val"]],
        y_reg.loc[splits["test"]],
    )

    ensure_dirs([TABLES_DIR, HISTORY_DIR, MLRUNS_DIR])

    mlflow.set_tracking_uri(f"file:{MLRUNS_DIR.resolve()}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    random_state = 42
    clf_search = [
        {"hidden_layer_sizes": (64, 32), "activation": "relu", "lr": 0.001, "alpha": 1e-4, "batch_size": 32, "epochs": 40},
        {"hidden_layer_sizes": (128,), "activation": "relu", "lr": 0.0005, "alpha": 5e-4, "batch_size": 32, "epochs": 50},
        {"hidden_layer_sizes": (64, 32), "activation": "tanh", "lr": 0.001, "alpha": 1e-4, "batch_size": 16, "epochs": 50},
    ]
    reg_search = [
        {"hidden_layer_sizes": (64, 32), "activation": "relu", "lr": 0.001, "alpha": 1e-4, "batch_size": 32, "epochs": 60},
        {"hidden_layer_sizes": (128,), "activation": "relu", "lr": 0.0005, "alpha": 5e-4, "batch_size": 32, "epochs": 80},
        {"hidden_layer_sizes": (64, 32), "activation": "tanh", "lr": 0.001, "alpha": 1e-4, "batch_size": 16, "epochs": 70},
    ]

    # Classification NN (random/grid search over a few configs)
    best_clf_row = None
    best_clf_history = None
    for idx, clf_params in enumerate(clf_search, start=1):
        with mlflow.start_run(run_name=f"nn_classification_{idx}") as run:
            mlflow.log_params(
                {
                    "task": "classification",
                    "model": "mlp",
                    **clf_params,
                    "random_state": random_state,
                }
            )
            clf_pipeline, clf_history = train_classifier_nn(
                X_train.values, y_train_clf.values, X_val.values, y_val_clf.values, clf_params, random_state
            )

            clf_hist_path = log_history(f"classification_nn_{run.info.run_id}", clf_history)
            mlflow.log_artifact(clf_hist_path, artifact_path="history")
            mlflow.sklearn.log_model(clf_pipeline, artifact_path="model")

            y_val_pred = clf_pipeline.predict(X_val.values)
            y_val_proba = clf_pipeline.predict_proba(X_val.values)[:, 1]
            y_test_pred = clf_pipeline.predict(X_test.values)
            y_test_proba = clf_pipeline.predict_proba(X_test.values)[:, 1]

            clf_metrics = {
                "val_accuracy": accuracy_score(y_val_clf, y_val_pred),
                "val_f1": f1_score(y_val_clf, y_val_pred),
                "val_roc_auc": roc_auc_score(y_val_clf, y_val_proba),
                "test_accuracy": accuracy_score(y_test_clf, y_test_pred),
                "test_f1": f1_score(y_test_clf, y_test_pred),
                "test_roc_auc": roc_auc_score(y_test_clf, y_test_proba),
            }
            mlflow.log_metrics(clf_metrics)
            clf_row = {"model": f"mlp_classifier_{idx}", "source": "nn", "run_id": run.info.run_id, **clf_metrics, **clf_params}

            if best_clf_row is None or clf_row["val_roc_auc"] > best_clf_row["val_roc_auc"]:
                best_clf_row = clf_row
                best_clf_history = clf_history

    # Persist best classification history for plotting
    if best_clf_history is not None:
        log_history("classification_nn", best_clf_history)

    # Regression NN (random/grid search over a few configs)
    best_reg_row = None
    best_reg_history = None
    for idx, reg_params in enumerate(reg_search, start=1):
        with mlflow.start_run(run_name=f"nn_regression_{idx}") as run:
            mlflow.log_params(
                {
                    "task": "regression",
                    "model": "mlp",
                    **reg_params,
                    "random_state": random_state,
                }
            )
            reg_pipeline, reg_history = train_regressor_nn(
                X_train.values, y_train_reg.values, X_val.values, y_val_reg.values, reg_params, random_state
            )
            reg_hist_path = log_history(f"regression_nn_{run.info.run_id}", reg_history)
            mlflow.log_artifact(reg_hist_path, artifact_path="history")
            mlflow.sklearn.log_model(reg_pipeline, artifact_path="model")

            y_val_pred_reg = reg_pipeline.predict(X_val.values)
            y_test_pred_reg = reg_pipeline.predict(X_test.values)

            val_mse = mean_squared_error(y_val_reg, y_val_pred_reg)
            test_mse = mean_squared_error(y_test_reg, y_test_pred_reg)
            reg_metrics = {
                "val_mae": mean_absolute_error(y_val_reg, y_val_pred_reg),
                "val_rmse": float(np.sqrt(val_mse)),
                "test_mae": mean_absolute_error(y_test_reg, y_test_pred_reg),
                "test_rmse": float(np.sqrt(test_mse)),
            }
            mlflow.log_metrics(reg_metrics)
            reg_row = {"model": f"mlp_regressor_{idx}", "source": "nn", "run_id": run.info.run_id, **reg_metrics, **reg_params}

            if best_reg_row is None or reg_row["val_rmse"] < best_reg_row["val_rmse"]:
                best_reg_row = reg_row
                best_reg_history = reg_history

    if best_reg_history is not None:
        log_history("regression_nn", best_reg_history)

    # Persist NN metric tables for the selected best runs
    clf_row = {
        "model": "mlp_classifier",
        "source": "nn",
        "run_id": best_clf_row["run_id"],
        "val_accuracy": best_clf_row["val_accuracy"],
        "val_f1": best_clf_row["val_f1"],
        "val_roc_auc": best_clf_row["val_roc_auc"],
        "test_accuracy": best_clf_row["test_accuracy"],
        "test_f1": best_clf_row["test_f1"],
        "test_roc_auc": best_clf_row["test_roc_auc"],
    }
    reg_row = {
        "model": "mlp_regressor",
        "source": "nn",
        "run_id": best_reg_row["run_id"],
        "val_mae": best_reg_row["val_mae"],
        "val_rmse": best_reg_row["val_rmse"],
        "test_mae": best_reg_row["test_mae"],
        "test_rmse": best_reg_row["test_rmse"],
    }

    # Persist NN metric tables
    pd.DataFrame([clf_row]).to_csv(TABLES_DIR / "nn_classification_results.csv", index=False)
    pd.DataFrame([reg_row]).to_csv(TABLES_DIR / "nn_regression_results.csv", index=False)

    # Update manifest with NN runs and overall best selections
    manifest = load_manifest()
    best = manifest.get("best_models", {})

    best["classification_nn"] = {**clf_row, "artifact_path": "model"}
    best["regression_nn"] = {**reg_row, "artifact_path": "model"}

    # Compute overall best (by val_roc_auc for classification, val_rmse for regression)
    clf_candidates = [best["classification_classical"], best["classification_nn"]]
    reg_candidates = [best["regression_classical"], best["regression_nn"]]

    best_clf_overall = max(clf_candidates, key=lambda r: r["val_roc_auc"])
    best_reg_overall = min(reg_candidates, key=lambda r: r["val_rmse"])

    best["classification_best"] = best_clf_overall
    best["regression_best"] = best_reg_overall

    manifest["best_models"] = best
    BEST_RUN_PATH.write_text(json.dumps(manifest, indent=2))

    print("Logged NN runs to MLflow and updated reports/tables + best_runs.json")


if __name__ == "__main__":
    main()
