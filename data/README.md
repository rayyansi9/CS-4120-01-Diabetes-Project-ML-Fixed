Data source: we use the built-in scikit-learn `load_diabetes` dataset (`src/data.py`), so no download is required. Do not commit any external raw data here.

If you choose to experiment with other tabular diabetes datasets, place download instructions in this folder and keep raw files out of the repo (add to `.gitignore`). Splits are generated programmatically with a fixed seed; the persisted split indices live in `reports/splits.npz` after running `python3 src/train_baselines.py`.
