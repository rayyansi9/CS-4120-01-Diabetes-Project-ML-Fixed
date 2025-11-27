"""Wrapper entry point for evaluation.

Loads best classical/NN models from MLflow artifacts and emits the required
plots and comparison tables. Delegates to `evaluationplots.py`.
"""

from evaluationplots import main


if __name__ == "__main__":
    main()
