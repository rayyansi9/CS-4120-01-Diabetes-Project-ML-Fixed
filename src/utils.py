#Helper functions for reusable splits, dirs, and metrics.

from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import os
import numpy as np


def ensure_dirs(paths: Iterable[os.PathLike | str]) -> None:
    """Create directories if they do not already exist."""

    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def get_split_indices(
    df,
    label_col: str = "label",
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    #
   # Create a single train/val/test split and return index arrays for each fold.
   # Stratifies on `label_col` to preserve class balance for classification.
    #
    if label_col not in df.columns:
        raise ValueError(f"{label_col} not in dataframe columns")

    idx = df.index.to_numpy()
    labels = df[label_col].to_numpy()

    # Split out the test set first
    idx_trainval, idx_test, y_trainval, y_test = train_test_split(
        idx, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    # Split the remaining data into train and val
    val_relative = val_size / (1.0 - test_size)
    idx_train, idx_val, _, _ = train_test_split(
        idx_trainval,
        y_trainval,
        test_size=val_relative,
        random_state=random_state,
        stratify=y_trainval,
    )

    return {"train": idx_train, "val": idx_val, "test": idx_test}


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "mse": float(mse)}


