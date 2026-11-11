"""Backtest the growth forecaster on held-out editions, then train the final models.

Splits, strictly by time (training targets never come from the test edition or later).
With L = the newest edition (FC 26 = 26):
  h=1  tune: train t<=L-3 -> val t=L-2   test: train t<=L-2 -> t=L-1  (FC 25 -> FC 26)
  h=2  tune: train t<=L-6 -> val t=L-4   test: train t<=L-4 -> t=L-2  (FC 24 -> FC 26)

Also a rolling-origin check: for every edition t from FIFA 18 on, a model trained only
on targets known before t is scored on t -> t+h (same hyperparameters as the main test).
That shows how much one test window moves the result; see the README for why
FC 25 -> FC 26 is the odd one out (the FC 25 file is a late-season roster).

Writes reports/forecast_metrics.json, reports/forecast_rolling.json,
models/growth_h{1,2}.joblib and data/processed/forecasts_latest.csv (latest edition
scored 1 and 2 editions ahead).
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


def make_splits(last: int, first: int = 15) -> dict:
    """Test on the newest targets; tune on the edition before the test window."""
    out = {}
    for h in (1, 2):
        test_t = last - h
        val_t = test_t - h
        out[h] = dict(tune=(range(first, val_t - h + 1), val_t),
                      test=(range(first, test_t - h + 1), test_t))
    return out


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


def rolling(panel, h, params, first_test=18):
    """Score every test window t -> t+h with a model trained on targets known before t."""
    out = []
    last = int(panel["edition"].max())
    for t in range(first_test, last - h + 1):
        train_eds = range(int(panel["edition"].min()), t - h + 1)
        tr, te = rows_for(panel, train_eds, h), rows_for(panel, [t], h)
        if len(tr) == 0 or len(te) == 0:
            continue
        model = GrowthModel(h, params).fit(tr, intervals=False)
        preds = predict_all(fit_baselines(tr, h), te)
        preds["model"] = te["overall"].to_numpy() + model.predict_delta(te)
        y = te[target_col(h)].to_numpy()
        young = (te["age"] <= 21).to_numpy()
        out.append({"test_edition": t, "target_edition": t + h, "n": int(len(te)),
                    **{k: mae(y, v) for k, v in preds.items()},
                    "u21": {k: mae(y[young], v[young]) for k, v in preds.items()}})
        print(f"    rolling h={h} {t}->{t + h}:", {k: round(mae(y, v), 3) for k, v in preds.items()})
    return out


def main():
    t0 = time.time()
    panel = pd.read_pickle(ROOT / "data/processed/panel.pkl")
    metrics = {}
    final = {}
    splits = make_splits(int(panel["edition"].max()), int(panel["edition"].min()))
    for h, sp in splits.items():
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

    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/forecast_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if "--no-rolling" not in sys.argv:
        roll = {f"h{h}": rolling(panel, h, metrics[f"h{h}"]["best_params"]) for h in (1, 2)}
        (ROOT / "reports/forecast_rolling.json").write_text(json.dumps(roll, indent=2), encoding="utf-8")

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
