"""Two experiments for the squad builder. Writes reports/optimizer_metrics.json.

A) Exact MILP vs a hand-made greedy strategy (newest edition, FC 26)
   every Premier League club x budget {20, 50, 100}M x formation {4-3-3, 4-2-3-1},
   max 3 signings, no sales. Same constraints for both.

B) Decision backtest: do forecasts make better signings?
   Transfer window in edition t, 40 clubs each, EUR 30M,
   3 signings, 4-3-3. Signings are chosen three ways:
     now        rating at the position today
     potential  + the EA-potential baseline's predicted change
     forecast   + the growth model's predicted change
   Forecast/baseline models are trained only on editions before t.
   Each plan is then scored with the *actual* ratings from edition t+1:
   the best XI the final squad can field next season.

C) Whole squad vs starting XI only (newest edition, with sales allowed)
   every Premier League club x budget {20, 50}M, 4-3-3, max 3 signings, up to 4 sales.
   The same problem planned twice: XI only (bench_weight 0) and XI + 0.3 x second XI.
   Both final squads are scored the same way: best XI, then the best second XI behind it.
   Run on its own with --only-c (writes reports/optimizer_metrics_C.json).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fcscout.data import PREMIER_LEAGUE  # noqa: E402
from fcscout.forecast import GrowthModel, PotentialLinear, rows_for, target_col  # noqa: E402
from fcscout.optimize import (TransferSettings, greedy_plan, optimize_squad,  # noqa: E402
                              slot_ratings)


def experiment_a(panel: pd.DataFrame) -> dict:
    d = panel[panel["edition"] == panel["edition"].max()].copy()
    d["pos_group"] = d["pos_group"].astype(str)
    clubs = sorted(d.loc[d["league"] == PREMIER_LEAGUE, "club"].dropna().unique())
    rows = []
    for club in clubs:
        squad, market = d[d["club"] == club], d[d["club"] != club]
        for budget in (20e6, 50e6, 100e6):
            for formation in ("4-3-3", "4-2-3-1"):
                s = TransferSettings(formation=formation, budget=budget, max_signings=3)
                plan = optimize_squad(squad, market, s)
                g1 = greedy_plan(squad, market, s, rule="biggest_upgrade")
                g2 = greedy_plan(squad, market, s, rule="best_value")
                rows.append(dict(club=club, budget=budget, formation=formation,
                                 base=plan.baseline_xi_rating, milp=plan.xi_rating,
                                 greedy_best=max(g1, g2), greedy_upgrade=g1, greedy_value=g2,
                                 spend=plan.spend, n_candidates=plan.n_candidates,
                                 solve_s=plan.solve_seconds, status=plan.status))
        print(f"  A {club} done")
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "reports/optimizer_vs_greedy.csv", index=False)
    gain_milp = (df["milp"] - df["base"]) * 11      # total rating points added to the XI
    gain_greedy = (df["greedy_best"] - df["base"]) * 11
    tol = 1e-6
    return {
        "scenarios": int(len(df)),
        "all_optimal": bool((df["status"] == "optimal").all()),
        "milp_never_worse": bool((df["milp"] >= df["greedy_best"] - tol).all()),
        "milp_strictly_better_share": float((df["milp"] > df["greedy_best"] + tol).mean()),
        "mean_xi_points_added_milp": float(gain_milp.mean()),
        "mean_xi_points_added_greedy_best": float(gain_greedy.mean()),
        "extra_points_vs_greedy_mean": float((gain_milp - gain_greedy).mean()),
        "extra_gain_vs_greedy_pct": float((gain_milp.sum() - gain_greedy.sum()) / gain_greedy.sum()),
        "solve_seconds_median": float(df["solve_s"].median()),
        "solve_seconds_max": float(df["solve_s"].max()),
        "candidates_median": float(df["n_candidates"].median()),
    }


def lineups(squad: pd.DataFrame, formation: str) -> tuple[float, float, set, set]:
    """Best XI, then the best second XI behind it: (XI avg, second XI avg, XI ids, second XI ids)."""
    s = TransferSettings(formation=formation, budget=0, max_signings=0, bench_weight=1e-3,
                         min_depth={}, squad_min=0, squad_max=99, wage_budget=1e12)
    plan = optimize_squad(squad, squad.iloc[:0], s)
    return plan.xi_rating, plan.bench_rating, set(plan.xi["id"]), set(plan.bench["id"])


def experiment_c(panel: pd.DataFrame) -> dict:
    d = panel[panel["edition"] == panel["edition"].max()].copy()
    d["pos_group"] = d["pos_group"].astype(str)
    clubs = sorted(d.loc[d["league"] == PREMIER_LEAGUE, "club"].dropna().unique())
    rows = []
    for club in clubs:
        squad, market = d[d["club"] == club], d[d["club"] != club]
        _, _, _, backups = lineups(squad, "4-3-3")
        for budget in (20e6, 50e6):
            for w in (0.0, 0.3):
                s = TransferSettings(formation="4-3-3", budget=budget, max_signings=3, allow_sales=True,
                                     max_sales=4, bench_weight=w)
                plan = optimize_squad(squad, market, s)
                final = pd.concat([squad[~squad["id"].isin(plan.sales["id"])],
                                   market[market["id"].isin(plan.buys["id"])]])
                xi, bench, _, _ = lineups(final, "4-3-3")
                sold = set(plan.sales["id"])
                rows.append(dict(club=club, budget=budget, w=w, status=plan.status, xi=xi, bench=bench,
                                 n_sales=len(sold), backups_sold=len(sold & backups),
                                 sold_age=float(plan.sales["age"].mean()) if len(sold) else np.nan,
                                 solve_s=plan.solve_seconds))
        print(f"  C {club} done")
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "reports/whole_squad_vs_xi.csv", index=False)
    a, b = df[df["w"] == 0].reset_index(drop=True), df[df["w"] > 0].reset_index(drop=True)
    return {
        "scenarios": int(len(a)),
        "all_optimal": bool((df["status"] == "optimal").all()),
        "xi_only": {"xi_avg": float(a["xi"].mean()), "second_xi_avg": float(a["bench"].mean()),
                    "sales": int(a["n_sales"].sum()),
                    "sold_age_avg": float(a["sold_age"].mean())},
        "whole_squad": {"xi_avg": float(b["xi"].mean()), "second_xi_avg": float(b["bench"].mean()),
                        "sales": int(b["n_sales"].sum()),
                        "sold_age_avg": float(b["sold_age"].mean())},
        "xi_points_given_up_avg": float(((a["xi"] - b["xi"]) * 11).mean()),
        "second_xi_points_gained_avg": float(((b["bench"] - a["bench"]) * 11).mean()),
        "solve_seconds_median_whole_squad": float(b["solve_s"].median()),
    }


def best_xi_next(final_ids: list[int], nxt: pd.DataFrame, formation: str) -> tuple[float, int]:
    """Average XI rating next season for this squad (players gone from the game are unavailable)."""
    sq = nxt[nxt["id"].isin(final_ids)]
    missing = len(set(final_ids) - set(sq["id"]))
    R = slot_ratings(sq, formation)
    if len(sq) < R.shape[1]:
        return np.nan, missing
    r, c = linear_sum_assignment(-R)
    return float(R[r, c].mean()), missing


def experiment_b(panel: pd.DataFrame, h: int = 1, max_age: int | None = None,
                 n_clubs=40, budget=30e6) -> dict:
    # the same transfer windows as the first version of this project, so the
    # numbers stay comparable (all inside FIFA 15-22)
    editions = (18, 19) if h == 1 else (18, 20)
    out = {"horizon": h, "max_age": max_age, "editions": list(editions)}
    rows = []
    rng = np.random.default_rng(7)
    for t in editions:
        hist = rows_for(panel, range(15, t - h + 1), h)      # targets up to edition t only
        gm = GrowthModel(h, {"max_leaf_nodes": 31, "min_samples_leaf": 20}).fit(hist, intervals=False)
        y = hist[target_col(h)] - hist["overall"]
        pl = PotentialLinear().fit(hist, y)
        cur = panel[panel["edition"] == t].copy()
        cur["pos_group"] = cur["pos_group"].astype(str)
        nxt = panel[panel["edition"] == t + h].copy()
        boost = {"now": None,
                 "potential": pd.Series(pl.predict_delta(cur), index=cur.index),
                 "forecast": pd.Series(gm.predict_delta(cur), index=cur.index)}
        strength = cur.groupby("club")["club_strength"].first()
        size = cur.groupby("club").size()
        pool = strength[(strength >= 68) & (strength <= 78) & (size.reindex(strength.index) >= 18)]
        pool = pool.index.to_numpy()
        clubs = rng.choice(np.sort(pool), size=min(n_clubs, len(pool)), replace=False)
        for club in clubs:
            sq_mask = cur["club"] == club
            squad, market = cur[sq_mask], cur[~sq_mask]
            s = TransferSettings(formation="4-3-3", budget=budget, max_signings=3, max_age=max_age)
            base_next, _ = best_xi_next(squad["id"].tolist(), nxt, s.formation)
            for mode, b in boost.items():
                bs = None if b is None else b[sq_mask]
                bm = None if b is None else b[~sq_mask]
                plan = optimize_squad(squad, market, s, boost_squad=bs, boost_market=bm)
                ids = squad["id"].tolist() + plan.buys["id"].tolist()
                xi_next, missing = best_xi_next(ids, nxt, s.formation)
                buys_next = nxt.set_index("id").reindex(plan.buys["id"])["overall"]
                rows.append(dict(edition=t, club=club, mode=mode, xi_next=xi_next,
                                 base_next=base_next, n_buys=len(plan.buys),
                                 buys_age=plan.buys["age"].mean(),
                                 buys_ovr_now=plan.buys["overall"].mean(),
                                 buys_ovr_next=buys_next.mean(),
                                 buys_missing_next=int(buys_next.isna().sum()),
                                 spend=plan.spend, status=plan.status))
        print(f"  B edition {t}: {len(clubs)} clubs done")
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / f"reports/decision_backtest_h{h}_age{max_age}.csv", index=False)
    piv = df.pivot_table(index=["edition", "club"], columns="mode", values="xi_next")
    piv = piv.dropna()
    for mode in ("now", "potential", "forecast"):
        sub = df[df["mode"] == mode]
        out[mode] = {
            "next_season_xi_avg": float(sub["xi_next"].mean()),
            "xi_points_added_next_season": float(((sub["xi_next"] - sub["base_next"]) * 11).mean()),
            "signings_age_avg": float(sub["buys_age"].mean()),
            "signings_rating_now_avg": float(sub["buys_ovr_now"].mean()),
            "signings_rating_next_avg": float(sub["buys_ovr_next"].mean()),
            "signings_change_avg": float((sub["buys_ovr_next"] - sub["buys_ovr_now"]).mean()),
            "signings_gone_from_game": int(sub["buys_missing_next"].sum()),
            "spend_avg_m": float(sub["spend"].mean() / 1e6),
        }
    out["clubs"] = int(len(piv))
    out["forecast_beats_now_share"] = float((piv["forecast"] > piv["now"] + 1e-9).mean())
    out["forecast_ties_now_share"] = float((abs(piv["forecast"] - piv["now"]) <= 1e-9).mean())
    out["forecast_minus_now_xi_points"] = float(((piv["forecast"] - piv["now"]) * 11).mean())
    out["forecast_minus_potential_xi_points"] = float(((piv["forecast"] - piv["potential"]) * 11).mean())
    # paired bootstrap CI for forecast - now (XI points)
    diff = ((piv["forecast"] - piv["now"]) * 11).to_numpy()
    bs = [rng.choice(diff, len(diff)).mean() for _ in range(5000)]
    out["forecast_minus_now_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return out


def main():
    t0 = time.time()
    panel = pd.read_pickle(ROOT / "data/processed/panel.pkl")
    if "--only-c" in sys.argv:
        res = {"C_whole_squad_vs_xi": dict(experiment_c(panel), edition=int(panel["edition"].max()))}
        (ROOT / "reports/optimizer_metrics_C.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(json.dumps(res, indent=2))
        return
    prev = ROOT / "reports/optimizer_metrics_A.json"
    if "--skip-a" in sys.argv and prev.exists():
        res = json.loads(prev.read_text(encoding="utf-8"))
    else:
        res = {"A_milp_vs_greedy": dict(experiment_a(panel), edition=int(panel["edition"].max()))}
        prev.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    # every variant that was run is reported, not just the best one
    res["B_decision_backtest"] = {}
    for h, max_age in [(1, None), (1, 29), (2, None), (2, 29)]:
        key = f"h{h}_max_age_{max_age}"
        res["B_decision_backtest"][key] = experiment_b(panel, h=h, max_age=max_age)
        print(key, json.dumps(res["B_decision_backtest"][key], indent=2))
        (ROOT / "reports/optimizer_metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
