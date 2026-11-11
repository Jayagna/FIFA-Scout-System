"""Run the whole pipeline: data -> panel -> forecaster -> evaluations.

    python scripts/run_all.py            # everything (the optimizer backtest takes about an hour)
    python scripts/run_all.py --quick    # skip the optimizer backtest
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = ["download_data.py", "build_dataset.py", "train_forecaster.py", "train_survival.py", "eval_similarity.py",
         "backtest_optimizer.py"]
# (scripts/fit_rating_formulas.py is run by hand: its output is committed in src/fcscout/)

if __name__ == "__main__":
    steps = STEPS[:-1] if "--quick" in sys.argv else STEPS
    for step in steps:
        print(f"\n=== {step}")
        subprocess.run([sys.executable, str(ROOT / "scripts" / step)], check=True)
