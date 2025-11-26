
"""Utility helpers for reproducible training and evaluation.
This module centralizes split creation so regression and classification
experiments share identical train/val/test partitions. We also provide
lightweight helpers for directory creation and metric calculations to
keep the training and evaluation scripts compact and consistent.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


def ensure_dirs(paths: Iterable[os.PathLike | str]) -> None:
	"""Create directories if they do not already exist."""

	for path in paths:
		Path(path).mkdir(parents=True, exist_ok=True)


def get_split_indices(
	df,
	label_col: str = "label",
	val_size: float = 0.15,
	test_size: float = 0.15,
	random_state: int = 42,
	split_path: str = "data/splits.npz",
) -> Dict[str, np.ndarray]:
	"""Return (and persist) consistent stratified train/val/test indices.

	Splits are stratified on ``label_col`` to preserve class balance. The
	resulting indices are saved to ``split_path`` so training and evaluation
	re-use the exact same partitions and avoid silent misalignment between
	regression and classification targets.
	"""

	split_file = Path(split_path)
	if split_file.exists():
		data = np.load(split_file)
		return {"train": data["train"], "val": data["val"], "test": data["test"]}

	indices = df.index.to_numpy()

	train_val_idx, test_idx = train_test_split(
		indices,
		test_size=test_size,
		stratify=df[label_col],
		random_state=random_state,
	)

	adjusted_val_size = val_size / (1 - test_size)
	train_idx, val_idx = train_test_split(
		train_val_idx,
		test_size=adjusted_val_size,
		stratify=df.loc[train_val_idx, label_col],
		random_state=random_state,
	)

	split_file.parent.mkdir(parents=True, exist_ok=True)
	np.savez(split_file, train=train_idx, val=val_idx, test=test_idx)

	return {"train": train_idx, "val": val_idx, "test": test_idx}


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
	"""Compute MAE, MSE, and RMSE for convenience."""

	mse = mean_squared_error(y_true, y_pred)
	return {
		"mae": mean_absolute_error(y_true, y_pred),
		"mse": mse,
		"rmse": mse**0.5,
	}

