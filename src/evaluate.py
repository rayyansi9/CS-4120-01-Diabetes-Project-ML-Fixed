"""Evaluation entry point.

Loads best classical/NN models from MLflow artifacts and emits the required
plots and comparison tables from saved models (no refit).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Point matplotlib's cache somewhere we can write before importing it
REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")  # keeps plotting happy without a display
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import joblib
import numpy as np

from data import add_class_label, load_diabetes_df
from utils import ensure_dirs, get_split_indices

FIGURES_DIR = Path("reports/figures")
TABLES_DIR = Path("reports/tables")
HISTORY_DIR = Path("reports/history")
BEST_RUN_PATH = Path("reports/best_runs.json")
MLRUNS_DIR = Path("mlruns")
SPLITS_PATH = Path("reports/splits.npz")


def load_manifest(path: Path = BEST_RUN_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `python3 src/train_baselines.py` and `python3 src/train_nn.py` first."
        )
    return json.loads(path.read_text())


def load_history(name: str) -> pd.DataFrame:
    path = HISTORY_DIR / f"{name}_history.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing history file at {path}. Ensure `python3 src/train_nn.py` has run."
        )
    df = pd.read_csv(path)
    if "epoch" not in df.columns:
        df.insert(0, "epoch", np.arange(1, len(df) + 1))
    return df


def load_splits(df):
    if SPLITS_PATH.exists():
        data = np.load(SPLITS_PATH)
        return {k: data[k] for k in ["train", "val", "test"]}
    return get_split_indices(df)


def load_model_with_fallback(run_id: str, artifact_path: str, local_name: str):
    try:
        return mlflow.sklearn.load_model(f"runs:/{run_id}/{artifact_path}")
    except Exception as exc:
        local_path = Path("models") / f"{local_name}.joblib"
        if local_path.exists():
            print(f"MLflow load failed ({exc}); falling back to {local_path}")
            return joblib.load(local_path)
        raise


def plot_learning_curve_classification(df_hist: pd.DataFrame):
    plt.figure()
    plt.plot(df_hist["epoch"], df_hist["train_roc_auc"], label="Train ROC AUC", color="#4d4d4d")
    plt.plot(df_hist["epoch"], df_hist["val_roc_auc"], label="Val ROC AUC", color="#999999")
    plt.xlabel("Epoch")
    plt.ylabel("ROC AUC")
    plt.title("Classification NN Learning Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "plot1_classification_learning_curve.png")
    plt.close()


def plot_learning_curve_regression(df_hist: pd.DataFrame):
    plt.figure()
    plt.plot(df_hist["epoch"], df_hist["train_rmse"], label="Train RMSE", color="#4d4d4d")
    plt.plot(df_hist["epoch"], df_hist["val_rmse"], label="Val RMSE", color="#999999")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("Regression NN Learning Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "plot2_regression_learning_curve.png")
    plt.close()


def plot_confusion_matrix(model, X_test, y_test, title: str):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Greys")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "plot3_confusion_matrix.png")
    plt.close()


def plot_residuals(model, X_test, y_test, title: str):
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred
    plt.figure()
    sns.scatterplot(x=y_pred, y=residuals, color="#595959", alpha=0.7)
    plt.axhline(0, color="#d9d9d9", linestyle="--")
    plt.title(title)
    plt.xlabel("Predicted Progression")
    plt.ylabel("Residual (y_true - y_pred)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "plot4_residuals_vs_predicted.png")
    plt.close()


def plot_feature_importance(model, X_test, y_test, title: str, feature_names):
    result = permutation_importance(
        model, X_test, y_test, scoring="roc_auc", n_repeats=20, random_state=42
    )
    idx = result.importances_mean.argsort()[::-1]
    top_features = [feature_names[i] for i in idx]
    top_scores = result.importances_mean[idx]

    plt.figure(figsize=(8, 6))
    sns.barplot(x=top_scores, y=top_features, color="#888888")
    plt.title(title)
    plt.xlabel("Permutation Importance (Δ ROC AUC)")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "plot5_feature_importance.png")
    plt.close()


def main() -> None:
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(f"file:{(REPO_ROOT / 'mlruns').resolve()}")
    manifest = load_manifest()
    best = manifest["best_models"]

    # Load data and split
    df = load_diabetes_df()
    df, _ = add_class_label(df)
    splits = load_splits(df)
    feature_cols = [c for c in df.columns if c not in {"target", "label"}]
    X = df[feature_cols]
    y_clf = df["label"]
    y_reg = df["target"]
    X_test = X.loc[splits["test"]]
    y_test_clf = y_clf.loc[splits["test"]]
    y_test_reg = y_reg.loc[splits["test"]]

    ensure_dirs([FIGURES_DIR, TABLES_DIR])

    # Plot NN learning curves
    plot_learning_curve_classification(load_history("classification_nn"))
    plot_learning_curve_regression(load_history("regression_nn"))

    # Pull the best models
    clf_best = best["classification_best"]
    reg_best = best["regression_best"]
    clf_model = load_model_with_fallback(clf_best["run_id"], clf_best["artifact_path"], clf_best["model"])
    reg_model = load_model_with_fallback(reg_best["run_id"], reg_best["artifact_path"], reg_best["model"])

    # Plot confusion matrix and regression residuals
    plot_confusion_matrix(clf_model, X_test, y_test_clf, f"Confusion Matrix ({clf_best['model']}, Test)")
    plot_residuals(reg_model, X_test, y_test_reg, f"Residuals vs Predicted ({reg_best['model']}, Test)")

    # Permutation importance on the best classifier
    plot_feature_importance(
        clf_model,
        X_test,
        y_test_clf,
        "Permutation Importance (Best Classification Model)",
        feature_cols,
    )

    # Build comparison tables
    clf_table = pd.DataFrame(
        [
            {
                "model": best["classification_classical"]["model"],
                "source": "classical",
                "val_accuracy": best["classification_classical"]["val_accuracy"],
                "val_f1": best["classification_classical"]["val_f1"],
                "val_roc_auc": best["classification_classical"]["val_roc_auc"],
                "test_accuracy": best["classification_classical"]["test_accuracy"],
                "test_f1": best["classification_classical"]["test_f1"],
                "test_roc_auc": best["classification_classical"]["test_roc_auc"],
            },
            {
                "model": best["classification_nn"]["model"],
                "source": "nn",
                "val_accuracy": best["classification_nn"]["val_accuracy"],
                "val_f1": best["classification_nn"]["val_f1"],
                "val_roc_auc": best["classification_nn"]["val_roc_auc"],
                "test_accuracy": best["classification_nn"]["test_accuracy"],
                "test_f1": best["classification_nn"]["test_f1"],
                "test_roc_auc": best["classification_nn"]["test_roc_auc"],
            },
        ]
    )
    clf_table.to_csv(TABLES_DIR / "table1_classification_comparison.csv", index=False)

    reg_table = pd.DataFrame(
        [
            {
                "model": best["regression_classical"]["model"],
                "source": "classical",
                "val_mae": best["regression_classical"]["val_mae"],
                "val_rmse": best["regression_classical"]["val_rmse"],
                "test_mae": best["regression_classical"]["test_mae"],
                "test_rmse": best["regression_classical"]["test_rmse"],
            },
            {
                "model": best["regression_nn"]["model"],
                "source": "nn",
                "val_mae": best["regression_nn"]["val_mae"],
                "val_rmse": best["regression_nn"]["val_rmse"],
                "test_mae": best["regression_nn"]["test_mae"],
                "test_rmse": best["regression_nn"]["test_rmse"],
            },
        ]
    )
    reg_table.to_csv(TABLES_DIR / "table2_regression_comparison.csv", index=False)

    print(f"Saved plots to {FIGURES_DIR.resolve()} and tables to {TABLES_DIR.resolve()}")


if __name__ == "__main__":
    main()
