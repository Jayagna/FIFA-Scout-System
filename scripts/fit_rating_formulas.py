"""Recover EA's card-stat and position-rating formulas from data.

FC 24 (sofifa scrape) and FC 25 (game database) ship the 34 detailed attributes but
not the six card stats or the per-position ratings. EA computes both as weighted
sums of the attributes, so the weights can be recovered exactly enough with
non-negative least squares on an edition that publishes everything (FC 26), then
checked on editions the fit never saw (FIFA 22, FIFA 23).

Weights are snapped to whole hundredths (EA's own grid) when that costs no accuracy.
FIFA 22 is reproduced to within 0.1 rating points on every stat. The FIFA 23 file's
position ratings sit about 2 points higher for most positions because they include
sofifa's boost, so the loader recomputes them (see fcscout.data.load_sofifa_v2).

Writes src/fcscout/rating_formulas.json (used by fcscout.data.derive_ratings) and
reports/rating_formulas.json (accuracy on the held-out editions).

    python scripts/fit_rating_formulas.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fcscout.data import (FACE_STATS, GK_ATTRS, OUTFIELD_ATTRS, _V2_POSITION_FAMILIES,  # noqa: E402
                          detect_layout, _LOADERS)

FIT_EDITION, CHECK_EDITIONS = 26, (22, 23)
ATTRS = OUTFIELD_ATTRS + GK_ATTRS
FAMILIES = {fam.lower(): cols[len(cols) // 2] for fam, cols in _V2_POSITION_FAMILIES.items()}


def load(ed: int) -> pd.DataFrame:
    raw = ROOT / "data/raw"
    files = sorted(raw.glob(f"players_{ed}*.csv"))
    if not files:
        raise SystemExit(f"need an edition-{ed} file in data/raw (run scripts/download_data.py)")
    p = files[0]
    layout = detect_layout(p)
    if layout == "sofifa_v2":          # keep the file's own position ratings for the comparison
        return _LOADERS[layout](p, ed, position_ratings=True)
    return _LOADERS[layout](p, ed)


def targets() -> dict[str, str]:
    """name in the formula file -> column in the loaded table"""
    t = {fs: fs for fs in FACE_STATS}
    t.update({fam: f"pos_{col}" for fam, col in FAMILIES.items()})
    return t


def main():
    fit = load(FIT_EDITION)
    checks = {ed: load(ed) for ed in CHECK_EDITIONS}
    weights, report = {}, {}
    for name, col in targets().items():
        cols = OUTFIELD_ATTRS if name in FACE_STATS else ATTRS
        m = fit[col].notna() & fit[cols].notna().all(axis=1)
        if name in FACE_STATS:
            m &= fit["primary_pos"].ne("GK")
        if name == "gk":                 # outfield players' GK rating is not the keeper formula
            m &= fit["primary_pos"].eq("GK")
        X, y = fit.loc[m, cols].to_numpy(float), fit.loc[m, col].to_numpy(float)
        w, _ = nnls(X, y)
        # EA's weights are whole hundredths; snap to that grid unless it costs accuracy
        snapped = np.round(w, 2)
        err = lambda v: np.abs(np.floor(X @ v + 0.5) - y).mean()  # noqa: E731
        if err(snapped) <= err(w) + 0.002:
            w = snapped
        w = {c: round(float(v), 4) for c, v in zip(cols, w) if v > 0.004}
        weights[name] = w
        report[name] = {}
        for ed, d in {FIT_EDITION: fit, **checks}.items():
            mm = d[col].notna() & d[cols].notna().all(axis=1)
            if name in FACE_STATS:
                mm &= d["primary_pos"].ne("GK")
            if name == "gk":
                mm &= d["primary_pos"].eq("GK")
            X = sum(d.loc[mm, k] * v for k, v in w.items())
            pred = np.floor(X + 0.5)
            err = pred - d.loc[mm, col]
            report[name][str(ed)] = {"n": int(mm.sum()), "mae": round(float(err.abs().mean()), 3),
                                     "within_1": round(float((err.abs() <= 1).mean()), 4),
                                     "exact": round(float((err == 0).mean()), 4)}
        print(f"{name:9s}", "  ".join(f"{ed}: MAE {r['mae']:.2f} (±1: {r['within_1']:.1%})"
                                      for ed, r in report[name].items()))
    (ROOT / "src/fcscout/rating_formulas.json").write_text(
        json.dumps({"fitted_on": FIT_EDITION, "weights": weights}, indent=1))
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/rating_formulas.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
