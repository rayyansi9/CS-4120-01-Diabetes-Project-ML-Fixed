from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PY = sys.executable  # uses current venv/python

def run(cmd: list):
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)

if __name__ == "__main__":
    run([PY, str(ROOT / "train_baselines.py")])
    run([PY, str(ROOT / "evaluationplots.py")])
    print("Done: trained models and generated figures/tables.")