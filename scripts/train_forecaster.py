"""Backtest the growth forecaster on held-out editions, then train the final models.

Splits (strictly by time: training targets never come from the test edition or later)
  h=1  tune: train t=15..19 -> val t=20 (FIFA 21)   test: train t=15..20 -> t=21 (FIFA 22)
  h=2  tune: train t=15..16 -> val t=18 (FIFA 20)   test: train t=15..18 -> t=20 (FIFA 22)

Writes reports/forecast_metrics.json, models/growth_h{1,2}.joblib and
data/processed/forecasts_latest.csv (latest edition scored 1 and 2 editions ahead).
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fcscout.forecast import (GrowthModel, evaluate, fit_baselines, mae,  # noqa: E402
                              predict_all, rows_for, target_col)

GRID = [dict(max_leaf_nodes=l, min_samples_leaf=m) for l, m in itertools.product([15, 31, 63], [20, 80])]
SPLITS = {1: dict(tune=(range(15, 20), 20), test=(range(15, 21), 21)),
          2: dict(tune=(range(15, 17), 18), test=(range(15, 19), 20))}


def tune(panel, h, train_eds, val_ed):
    tr, va = rows_for(panel, train_eds, h), rows_for(panel, [val_ed], h)
    scores = []
    for params in GRID:
        m = GrowthModel(h, params).fit(tr, intervals=False)
        s = mae(va[target_col(h)], va["overall"] + m.predict_delta(va))
        scores.append((s, params))
        print(f"    h={h} {params} val MAE {s:.3f}")
    best = min(scores, key=lambda t: t[0])
    return best[1], best[0]


def main():
    t0 = time.time()
    panel = pd.read_pickle(ROOT / "data/processed/panel.pkl")
    metrics = {}
    final = {}
    for h, sp in SPLITS.items():
        print(f"== horizon {h}")
        params, val_mae = tune(panel, h, *sp["tune"])
        train_eds, test_ed = sp["test"]
        tr, te = rows_for(panel, train_eds, h), rows_for(panel, [test_ed], h)
        model = GrowthModel(h, params).fit(tr)
        preds = predict_all(fit_baselines(tr, h), te)
        band = model.predict(te)
        preds["model"] = band["pred"].to_numpy()
        ev = evaluate(te, preds, h)
        y = te[target_col(h)].to_numpy()
        ev["interval_80_coverage"] = float(np.mean((y >= np.floor(band["pred_lo"])) & (y <= np.ceil(band["pred_hi"]))))
        ev["best_params"] = params
        ev["val_mae"] = val_mae
        ev["train_editions"] = [int(e) for e in train_eds]
        ev["test_edition"] = int(test_ed)
        ev["n_train"] = int(len(tr))
        o = ev["overall"]
        ev["improvement_vs"] = {k: 1 - o["model"]["mae"] / o[k]["mae"] for k in o if k != "model"}
        metrics[f"h{h}"] = ev
        print(f"  test (t={test_ed} -> t={test_ed + h}):",
              {k: round(v["mae"], 3) for k, v in o.items()})
        # final model on every edition whose target is known
        last_ed = int(panel["edition"].max())
        final[h] = GrowthModel(h, params).fit(rows_for(panel, range(15, last_ed - h + 1), h))
        (ROOT / "models").mkdir(exist_ok=True)
        joblib.dump(final[h], ROOT / f"models/growth_h{h}.joblib")

    # importance proxy: how much the val MAE degrades without the history/club features
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/forecast_metrics.json").write_text(json.dumps(metrics, indent=2))

    latest = panel[panel["edition"] == panel["edition"].max()].copy()
    out = latest[["id", "edition", "name", "long_name", "age", "club", "league", "primary_pos",
                  "pos_group", "positions", "overall", "potential", "value_eur", "wage_eur"]].copy()
    for h, m in final.items():
        b = m.predict(latest)
        out[f"pred_t{h}"] = b["pred"].round(1).to_numpy()
        out[f"pred_t{h}_lo"] = b["pred_lo"].round(1).to_numpy()
        out[f"pred_t{h}_hi"] = b["pred_hi"].round(1).to_numpy()
    out["growth_t2"] = out["pred_t2"] - out["overall"]
    out.to_csv(ROOT / "data/processed/forecasts_latest.csv", index=False)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
