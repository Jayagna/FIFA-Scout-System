"""Train the stay-in-the-game model and add its scores to the latest forecasts.

Tests are by time, like the growth forecaster: the h-ahead model is scored on the
newest edition whose outcome is known (h=1: FC 25 -> FC 26, h=2: FC 24 -> FC 26),
trained only on transitions whose outcomes were known before it.

Writes reports/survival_metrics.json, models/survival_h{1,2}.joblib and adds
surv_t1 / surv_t2 to data/processed/forecasts_latest.csv.

    python scripts/train_survival.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fcscout.survival import (SurvivalModel, add_stay_targets, age_baseline, scores,  # noqa: E402
                              stays_col)

AGE_BANDS = [0, 23, 27, 30, 32, 34, 36, 50]
AGE_LABELS = ["<=23", "24-27", "28-30", "31-32", "33-34", "35-36", "37+"]


def main():
    panel = add_stay_targets(pd.read_pickle(ROOT / "data/processed/panel.pkl"))
    last = int(panel["edition"].max())
    first = int(panel["edition"].min())
    report, final = {}, {}
    for h in (1, 2):
        test_t = last - h
        tr = panel[panel["edition"].between(first, test_t - h) & panel[stays_col(h)].notna()]
        te = panel[(panel["edition"] == test_t) & panel[stays_col(h)].notna()]
        model = SurvivalModel(h).fit(tr)
        p = model.predict_proba(te)
        y = te[stays_col(h)].to_numpy()
        base = age_baseline(tr, te, h)
        band = pd.cut(te["age"], AGE_BANDS, labels=AGE_LABELS)
        calib = (pd.DataFrame({"band": band, "y": y, "p": p}).groupby("band", observed=True)
                 .agg(n=("y", "size"), actual=("y", "mean"), predicted=("p", "mean")).round(3))
        report[f"h{h}"] = {
            "test_edition": test_t, "target_edition": test_t + h, "n": int(len(te)),
            "n_train": int(len(tr)), "stay_rate": float(y.mean()),
            "model": scores(y, p), "age_and_rating_baseline": scores(y, base),
            "base_rate_brier": float(np.mean((y - tr[stays_col(h)].mean()) ** 2)),
            "by_age": calib.reset_index().to_dict(orient="records"),
        }
        print(f"h={h} test {test_t}->{test_t + h}: model {report[f'h{h}']['model']}, "
              f"age baseline {report[f'h{h}']['age_and_rating_baseline']}")
        print(calib.to_string())
        # final model: every transition with a known outcome
        final[h] = SurvivalModel(h).fit(panel[panel[stays_col(h)].notna()])
        (ROOT / "models").mkdir(exist_ok=True)
        joblib.dump(final[h], ROOT / f"models/survival_h{h}.joblib")

    (ROOT / "reports/survival_metrics.json").write_text(json.dumps(report, indent=2))
    latest = panel[panel["edition"] == last]
    fc_path = ROOT / "data/processed/forecasts_latest.csv"
    fc = pd.read_csv(fc_path).drop(columns=["surv_t1", "surv_t2"], errors="ignore")
    surv = pd.DataFrame({"id": latest["id"].to_numpy(),
                         **{f"surv_t{h}": final[h].predict_proba(latest).round(3) for h in (1, 2)}})
    fc.merge(surv, on="id", how="left").to_csv(fc_path, index=False)
    print("added surv_t1 / surv_t2 to", fc_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
