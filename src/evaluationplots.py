from __future__ import annotations

import os
from pathlib import Path

# Ensure matplotlib can write its cache in restricted environments before import.
# Use a repo-local cache to avoid home-directory permission issues in sandboxes.
REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".cache" / "matplotlib"))

import joblib
import seaborn as sns
import matplotlib
matplotlib.use("Agg")  # headless-safe backend for CLI/sandbox runs
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from data import add_class_label, load_diabetes_df
from utils import ensure_dirs, get_split_indices

FIGURES_DIR = Path("reports/figures")

def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing model at {path}. Run `python src/train_baselines.py` first."
        )
    return joblib.load(path)

def main() -> None:
    # Ensure matplotlib cache directory exists
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    df = load_diabetes_df()
    df, _ = add_class_label(df)

    splits = get_split_indices(df)

    # Features and targets
    feature_cols = [col for col in df.columns if col not in {"target", "label"}]
    X = df[feature_cols]
    y_clf = df["label"]
    y_reg = df["target"]

    # Test partition only (evaluation)
    X_test = X.loc[splits["test"]]
    y_test_clf = y_clf.loc[splits["test"]]
    y_test_reg = y_reg.loc[splits["test"]]

    ensure_dirs([FIGURES_DIR])

    # Target distribution bar plot
    plt.figure()
    df["label"].value_counts().sort_index().plot(
        kind="bar", color=["#d9d9d9", "#595959"]
    )
    plt.title("Target Distribution (High vs Low Progression)")
    plt.xlabel("Label (0 = Low, 1 = High)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "target_distribution.png")
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(numeric_only=True), cmap="Greys", annot=False)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png")
    plt.close()

    # Confusion matrix from saved logistic regression
    log_reg = load_model(Path("models/logistic_regression.joblib"))
    y_pred_clf = log_reg.predict(X_test)
    cm = confusion_matrix(y_test_clf, y_pred_clf)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Greys")
    plt.title("Confusion Matrix (Logistic Regression, Test Split)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png")
    plt.close()

    # Residuals vs predicted for saved linear regression
    lin_reg = load_model(Path("models/linear_regression.joblib"))
    y_pred_reg = lin_reg.predict(X_test)
    residuals = y_test_reg - y_pred_reg

    plt.figure()
    sns.scatterplot(x=y_pred_reg, y=residuals, color="#595959", alpha=0.7)
    plt.axhline(0, color="#d9d9d9", linestyle="--")
    plt.title("Residuals vs Predicted (Linear Regression, Test Split)")
    plt.xlabel("Predicted Progression")
    plt.ylabel("Residual (y_true - y_pred)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "residuals_vs_predicted.png")
    plt.close()

    print(f"Saved figures to {FIGURES_DIR.resolve()}")

if __name__ == "__main__":
    main()
