"""Unit tests: parsing, optimizer exactness (vs brute force), similarity search, leakage.

Run:  python -m pytest -q      (or: python tests/test_core.py)
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fcscout.data import _pos_rating, _stars, load_kaggle_layout, load_sofifa_export  # noqa: E402
from fcscout.forecast import GrowthModel, rows_for  # noqa: E402
from fcscout.optimize import FORMATIONS, TransferSettings, optimize_squad, slot_ratings  # noqa: E402
from fcscout.similar import SimilarityIndex  # noqa: E402
from fcscout.data import OUTFIELD_ATTRS, GK_ATTRS  # noqa: E402


# ------------------------------------------------------------------ parsing
def test_rating_strings():
    s = pd.Series(["89+3", "91-1", "75", None, "42+11"])
    assert _pos_rating(s).tolist()[:3] == [89, 91, 75]
    assert np.isnan(_pos_rating(s)[3])
    assert _pos_rating(s)[4] == 42


def test_stars():
    assert _stars(pd.Series(["4 ★", "5★", 3])).tolist() == [4, 5, 3]


def test_two_layouts_same_schema(tmp_path):
    kag = pd.DataFrame({
        "sofifa_id": [1, 2], "short_name": ["A", "B"], "long_name": ["A a", "B b"], "age": [20, 30],
        "club_name": ["X", None], "overall": [70, 80], "potential": [80, 80],
        "value_eur": [1e6, 0], "wage_eur": [1e3, 2e3], "player_positions": ["ST, LW", "GK"],
        "work_rate": ["High/Low", "Medium/Medium"], "pace": [80, 50], "st": ["70+2", "20+1"],
        "attacking_crossing": ["60+1", 10],
    })
    exp = pd.DataFrame({
        "ID": [1, 2], "Name": ["A", "B"], "Age": [20, 30], "↓OVA": [70, 80], "POT": [80, 80],
        "Club": ["X", "Y"], "Contract Status": ["Contract", "Free"], "Positions": ["ST, LW", "GK"],
        "Value(Euro)": [1e6, 0], "Wage(Euro)": [1e3, 2e3], "A/W": ["High", "Medium"],
        "D/W": ["Low", "Medium"], "PAC": [80, 50], "Crossing": [60, 10],
    })
    a = load_kaggle_layout(_csv(tmp_path, "k.csv", kag), 22)
    b = load_sofifa_export(_csv(tmp_path, "e.csv", exp), 21)
    assert list(a.columns) == list(b.columns)
    for df in (a, b):
        assert df.loc[0, "work_rate_att"] == 3 and df.loc[0, "work_rate_def"] == 1
        assert np.isnan(df.loc[1, "value_eur"])          # 0 value -> missing
        assert np.isnan(df.loc[1, "pace"])               # keepers have no outfield face stats
        assert df.loc[0, "primary_pos"] == "ST"
        assert df.loc[0, "attacking_crossing"] == 60
    assert pd.isna(b.loc[1, "club"])                     # free agent
    assert a.loc[0, "pos_st"] == 70


def test_fc24_combined_layout_keeps_first_update(tmp_path):
    from fcscout.data import load_fc24_combined
    df = pd.DataFrame({
        "player_id": [7, 7, 7, 8], "fifa_version": [23.0, 23.0, 24.0, 24.0],
        "fifa_update": [2.0, 1.0, 1.0, 1.0], "short_name": ["X", "X", "X", "Y"],
        "overall": [71, 70, 74, 60], "potential": [80, 80, 82, 70], "age": [19, 19, 20, 25],
        "club_name": ["C", "C", "C", "D"], "league_name": ["L", "L", "L", "L"],
        "player_positions": ["ST", "ST", "ST", "CB"], "value_eur": [1e6] * 4, "wage_eur": [1e3] * 4,
        "club_contract_valid_until_year": [2026] * 4, "work_rate": ["High/High"] * 4,
        "defending_marking_awareness": [30, 30, 31, 70], "st": ["70+2"] * 4,
    })
    out = load_fc24_combined(_csv(tmp_path, "male_players.csv", df))
    assert sorted(out["edition"].unique().tolist()) == [23, 24]
    x23 = out[(out["id"] == 7) & (out["edition"] == 23)].iloc[0]
    assert x23["overall"] == 70                     # update 1 is kept
    assert x23["contract_end"] == 2026 and x23["defending_marking"] == 30


def _csv(tmp_path, name, df):
    p = Path(tmp_path) / name
    df.to_csv(p, index=False)
    return p


# ------------------------------------------------------------------ optimizer
def _random_players(rng, n, start_id, club):
    rows = []
    pos_cols = sorted({c for f in FORMATIONS.values() for _, c in f})
    prim = ["GK", "CB", "LB", "RB", "CDM", "CM", "LW", "RW", "ST"]
    for i in range(n):
        p = prim[i % len(prim)] if club == "Mine" else rng.choice(prim)
        base = rng.integers(55, 85)
        r = {f"pos_{c}": float(max(0, base - (0 if c.upper() == p else rng.integers(3, 25))))
             for c in pos_cols}
        if p != "GK":
            r["pos_gk"] = 20.0
        else:
            r.update({f"pos_{c}": 30.0 for c in pos_cols if c != "gk"})
        r.update(id=start_id + i, name=f"P{start_id + i}", club=club, primary_pos=p,
                 pos_group={"GK": "GK", "CB": "DEF", "LB": "DEF", "RB": "DEF", "CDM": "MID",
                            "CM": "MID", "LW": "FWD", "RW": "FWD", "ST": "FWD"}[p],
                 overall=base, potential=base + 3, age=int(rng.integers(18, 34)),
                 value_eur=float(rng.integers(1, 30)) * 1e6, wage_eur=float(rng.integers(5, 50)) * 1e3,
                 release_clause_eur=np.nan, on_loan=0)
        rows.append(r)
    return pd.DataFrame(rows)


def _brute_force(squad, market, s):
    R_sq = slot_ratings(squad, s.formation)
    R_mk = slot_ratings(market, s.formation)
    ask = market["value_eur"].to_numpy() * s.buy_markup
    wage_cap = squad["wage_eur"].sum() * 1.15
    best = -1
    for k in range(s.max_signings + 1):
        for combo in itertools.combinations(range(len(market)), k):
            combo = list(combo)
            if ask[combo].sum() > s.budget:
                continue
            if squad["wage_eur"].sum() + market["wage_eur"].to_numpy()[combo].sum() > wage_cap:
                continue
            R = np.vstack([R_sq, R_mk[combo]]) if combo else R_sq
            r, c = linear_sum_assignment(-R)
            best = max(best, R[r, c].sum())
    return best


def test_milp_matches_brute_force():
    rng = np.random.default_rng(1)
    for trial in range(15):
        squad = _random_players(rng, 13, 0, "Mine")
        market = _random_players(rng, 9, 1000, "Other")
        s = TransferSettings(formation="4-3-3", budget=float(rng.integers(10, 60)) * 1e6,
                             max_signings=2, min_depth={}, squad_min=0)
        plan = optimize_squad(squad, market, s)
        assert plan.status == "optimal"
        bf = _brute_force(squad, market, s)
        assert abs(plan.xi["rating_at_slot"].sum() - bf) < 1e-6, (trial, plan.xi_rating * 11, bf)
        assert plan.spend <= s.budget + 1e-6
        assert len(plan.buys) <= 2


def _brute_force_sales(squad, market, s):
    R_sq = slot_ratings(squad, s.formation)
    R_mk = slot_ratings(market, s.formation)
    ask = market["value_eur"].to_numpy() * s.buy_markup
    sell = squad["value_eur"].to_numpy() * s.sell_factor
    wq, wm = squad["wage_eur"].to_numpy(), market["wage_eur"].to_numpy()
    best = -1
    for ns in range(s.max_sales + 1):
        for sold in itertools.combinations(range(len(squad)), ns):
            kept = [i for i in range(len(squad)) if i not in sold]
            for k in range(s.max_signings + 1):
                for combo in itertools.combinations(range(len(market)), k):
                    combo = list(combo)
                    if ask[combo].sum() > s.budget + sell[list(sold)].sum() + 1e-6:
                        continue
                    if wq[kept].sum() + wm[combo].sum() > s.wage_budget:
                        continue
                    R = np.vstack([R_sq[kept], R_mk[combo]]) if combo else R_sq[kept]
                    if len(R) < R.shape[1]:
                        continue
                    r, c = linear_sum_assignment(-R)
                    best = max(best, R[r, c].sum())
    return best


def test_milp_matches_brute_force_with_sales():
    rng = np.random.default_rng(5)
    for trial in range(4):
        squad = _random_players(rng, 12, 0, "Mine")
        market = _random_players(rng, 7, 1000, "Other")
        s = TransferSettings(formation="4-4-2", budget=float(rng.integers(0, 25)) * 1e6,
                             max_signings=2, allow_sales=True, max_sales=1, min_depth={},
                             squad_min=0, squad_max=40,
                             wage_budget=float(squad["wage_eur"].sum() * 1.1))
        plan = optimize_squad(squad, market, s)
        bf = _brute_force_sales(squad, market, s)
        assert abs(plan.xi["rating_at_slot"].sum() - bf) < 1e-6, (trial, plan.xi_rating * 11, bf)


def test_sales_fund_signings():
    rng = np.random.default_rng(3)
    squad = _random_players(rng, 14, 0, "Mine")
    market = _random_players(rng, 30, 1000, "Other")
    s0 = TransferSettings(budget=0, max_signings=2, min_depth={}, squad_min=0)
    s1 = TransferSettings(budget=0, max_signings=2, allow_sales=True, max_sales=2,
                          min_depth={}, squad_min=0, wage_budget=1e9)
    p0, p1 = optimize_squad(squad, market, s0), optimize_squad(squad, market, s1)
    assert p0.spend == 0 and len(p0.buys) == 0
    assert p1.xi_rating >= p0.xi_rating
    assert p1.spend <= p1.income + 1e-6       # every euro spent came from a sale
    assert len(p1.sales) <= 2


# ------------------------------------------------------------------ similarity
def _attr_frame(n=300, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.integers(20, 95, size=(n, len(OUTFIELD_ATTRS))), columns=OUTFIELD_ATTRS)
    for c in GK_ATTRS:
        df[c] = rng.integers(5, 90, n)
    df["id"] = np.arange(n)
    df["primary_pos"] = np.where(np.arange(n) % 10 == 0, "GK", rng.choice(["ST", "CB", "CM"], n))
    df["club"] = "C"
    df["value_eur"] = rng.integers(1, 100, n) * 1e6
    df["overall"] = rng.integers(60, 90, n)
    df["age"] = rng.integers(18, 35, n)
    df["league"] = "L"
    return df


def test_self_is_nearest_and_filters():
    df = _attr_frame()
    idx = SimilarityIndex(df, use_faiss=False)
    for block, mask in (("outfield", df["primary_pos"] != "GK"), ("gk", df["primary_pos"] == "GK")):
        q = df[mask]
        _, rows = idx.search_vectors(idx.embed(q, block), block, k=1)
        assert (df["id"].to_numpy()[rows[:, 0]] == q["id"].to_numpy()).all()
    target = df[df["primary_pos"] == "ST"].iloc[0]
    res = idx.find(int(target["id"]), max_fee=50e6, max_age=25, k=20)
    assert len(res) > 0
    assert (res["fee"] <= 50e6).all() and (res["age"] <= 25).all()
    assert (res["primary_pos"] == "ST").all() and (res["id"] != target["id"]).all()
    assert res["similarity"].is_monotonic_decreasing


# ------------------------------------------------------------------ forecasting
def test_rows_for_never_leaks_future_editions():
    panel = pd.DataFrame({"edition": [15, 16, 17, 18], "overall_t1": [1.0, 2.0, np.nan, 3.0]})
    got = rows_for(panel, range(15, 17), 1)
    assert set(got["edition"]) == {15, 16}


def test_growth_model_handles_unseen_categories():
    from fcscout.features import FEATURES, CAT_FEATURES
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({c: rng.normal(size=n) for c in FEATURES if c not in CAT_FEATURES})
    df["pos_group"], df["primary_pos"], df["preferred_foot"] = "MID", "CM", "Right"
    df["overall"] = rng.integers(50, 80, n).astype(float)
    df["overall_t1"] = df["overall"] + 2 - 0.1 * df["age"]
    m = GrowthModel(1, {"max_iter": 50}).fit(df)
    new = df.head(5).copy()
    new["primary_pos"] = "LWB"          # never seen in training
    out = m.predict(new)
    assert out.notna().all().all() and (out["pred_lo"] <= out["pred"]).all() and (out["pred"] <= out["pred_hi"]).all()


if __name__ == "__main__":
    import tempfile
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                if "tmp_path" in fn.__code__.co_varnames[: fn.__code__.co_argcount]:
                    with tempfile.TemporaryDirectory() as d:
                        fn(d)
                else:
                    fn()
                print(f"ok    {name}")
            except Exception as e:  # noqa: BLE001
                failures += 1
                print(f"FAIL  {name}: {e!r}")
    sys.exit(1 if failures else 0)
