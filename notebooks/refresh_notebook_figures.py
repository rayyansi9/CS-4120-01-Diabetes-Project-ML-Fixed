"""Refresh the notebook figures using the current, single-source pipeline.

Keeps notebooks/figures aligned with the fixed splits and the saved best models,
without touching the main report assets.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent.parent / ".cache" / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Allow importing from src
sys.path.append("src")

from data import add_class_label, load_diabetes_df
from evaluate import (
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    PLOT_STYLE,
    load_manifest,
    load_model_with_fallback,
    load_splits,
)
from utils import ensure_dirs


def main() -> None:
    # Keep matplotlib happy in sandboxed environments
    repo_root = Path(__file__).resolve().parent.parent
    os.environ.setdefault("MPLCONFIGDIR", str(repo_root / ".cache" / "matplotlib"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    out_dir = Path("notebooks/figures")
    ensure_dirs([out_dir])

    # Load data and splits
    df = load_diabetes_df()
    df, _ = add_class_label(df)
    splits = load_splits(df)
    feature_cols = [c for c in df.columns if c not in {"target", "label"}]
    X = df[feature_cols]
    y_clf = df["label"]
    y_reg = df["target"]

    # Load best models from manifest/MLflow artifacts
    manifest = load_manifest()
    best = manifest["best_models"]
    clf_best = best["classification_best"]
    reg_best = best["regression_best"]
    clf_model = load_model_with_fallback(
        clf_best["run_id"], clf_best["artifact_path"], clf_best["model"]
    )
    reg_model = load_model_with_fallback(
        reg_best["run_id"], reg_best["artifact_path"], reg_best["model"]
    )

    # Target distribution (consistent label ordering)
    counts = df["label"].value_counts().sort_index()
    with plt.rc_context(PLOT_STYLE):
        plt.figure(figsize=(6.5, 4.5))
        sns.barplot(
            x=counts.index,
            y=counts.values,
            hue=counts.index,
            palette=[COLOR_SECONDARY, COLOR_PRIMARY],
            edgecolor="#23395b",
            linewidth=0.8,
            legend=False,
        )
        plt.title("Target Distribution (High vs Low Progression)")
        plt.xlabel("Label (0 = Low, 1 = High)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(out_dir / "target_distribution.png", dpi=300)
        plt.close()

    # Confusion matrix on test split (loaded model)
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    X_test = X.loc[splits["test"]]
    y_test_clf = y_clf.loc[splits["test"]]
    y_pred = clf_model.predict(X_test)
    cm = confusion_matrix(y_test_clf, y_pred)
    with plt.rc_context(PLOT_STYLE):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues")
        plt.title(f"Confusion Matrix ({clf_best['model']}, Test)")
        plt.tight_layout()
        plt.savefig(out_dir / "confusion_matrix.png", dpi=300)
        plt.close()

    # Residuals vs predicted for regression model
    X_test_reg = X.loc[splits["test"]]
    y_test_reg = y_reg.loc[splits["test"]]
    y_pred_reg = reg_model.predict(X_test_reg)
    residuals = y_test_reg - y_pred_reg
    with plt.rc_context(PLOT_STYLE):
        plt.figure(figsize=(6.5, 4.5))
        sns.scatterplot(
            x=y_pred_reg,
            y=residuals,
            color=COLOR_PRIMARY,
            alpha=0.7,
            edgecolor="white",
            linewidth=0.4,
        )
        plt.axhline(0, color=COLOR_SECONDARY, linestyle="--", linewidth=1)
        plt.title(f"Residuals vs Predicted ({reg_best['model']}, Test)")
        plt.xlabel("Predicted Progression")
        plt.ylabel("Residual (y_true - y_pred)")
        plt.tight_layout()
        plt.savefig(out_dir / "residuals_plot.png", dpi=300)
        plt.close()

    # Correlation heatmap over features + target/label
    corr_cols = feature_cols + ["target", "label"]
    corr = df[corr_cols].corr()
    with plt.rc_context(PLOT_STYLE):
        plt.figure(figsize=(9, 7))
        sns.heatmap(
            corr,
            cmap="coolwarm",
            center=0,
            square=True,
            cbar_kws={"shrink": 0.8, "label": "Pearson r"},
        )
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(out_dir / "correlation_heatmap.png", dpi=300)
        plt.close()

    print(f"Legacy notebook figures refreshed in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
