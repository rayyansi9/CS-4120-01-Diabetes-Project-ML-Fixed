# ...existing code...
from __future__ import annotations

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from data import add_class_label, load_diabetes_df
from utils import ensure_dirs, get_split_indices

FIGURES_DIR = Path("reports/figures")
# ...existing code...

def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing model at {path}. Run `python src/train_baselines.py` first."
        )
    return joblib.load(path)

def main() -> None:
    df = load_diabetes_df()
    df, _ = add_class_label(df)

    splits = get_split_indices(df)

    # Features and targets
    feature_cols = [col for col in df.columns if col not in {"target", "label"}]
    X = df[feature_cols]
    y_clf = df["label"]
    y_reg = df["target"]

    # Test partition only (evaluation as required)
    X_test = X.loc[splits["test"]]
    y_test_clf = y_clf.loc[splits["test"]]
    y_test_reg = y_reg.loc[splits["test"]]

    ensure_dirs([FIGURES_DIR])

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

    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(numeric_only=True), cmap="Greys", annot=False)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png")
    plt.close()

    log_reg = load_model(Path("models/logistic_regression.joblib"))
    y_pred_clf = log_reg.predict(X_test)
    cm = confusion_matrix(y_test_clf, y_pred_clf)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Greys")
    plt.title("Confusion Matrix (Logistic Regression, Test Split)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png")
    plt.close()

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
# ...existing code...ip
