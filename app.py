"""Career Mode Scout - Streamlit app.

    streamlit run app.py

Needs the pipeline outputs (python scripts/run_all.py): data/processed/panel.pkl
and data/processed/forecasts_latest.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fcscout.optimize import FORMATIONS, TransferSettings, optimize_squad  # noqa: E402
from fcscout.similar import SimilarityIndex  # noqa: E402

EDITION_NAMES = {15: "FIFA 15", 16: "FIFA 16", 17: "FIFA 17", 18: "FIFA 18", 19: "FIFA 19",
                 20: "FIFA 20", 21: "FIFA 21", 22: "FIFA 22", 23: "FIFA 23", 24: "EA FC 24",
                 25: "EA FC 25", 26: "EA FC 26", 27: "EA FC 27"}


def ed_name(e: int) -> str:
    return EDITION_NAMES.get(int(e), f"edition {e}")


def money(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "-"
    return f"€{x / 1e6:.1f}M" if abs(x) >= 1e6 else f"€{x / 1e3:.0f}K"


@st.cache_data(show_spinner="Loading player data...")
def load():
    panel = pd.read_pickle(ROOT / "data/processed/panel.pkl")
    latest_ed = int(panel["edition"].max())
    cur = panel[panel["edition"] == latest_ed].copy()
    cur["pos_group"] = cur["pos_group"].astype(str)
    cur["primary_pos"] = cur["primary_pos"].astype(str)
    fc = pd.read_csv(ROOT / "data/processed/forecasts_latest.csv")
    keep = ["id", "pred_t1", "pred_t1_lo", "pred_t1_hi", "pred_t2", "pred_t2_lo", "pred_t2_hi"]
    cur = cur.merge(fc[keep], on="id", how="left")
    history = panel[["id", "edition", "overall", "potential"]].copy()
    return cur, history, latest_ed


@st.cache_resource(show_spinner="Building similarity index...")
def similarity_index(_cur: pd.DataFrame, edition: int) -> SimilarityIndex:
    return SimilarityIndex(_cur)


st.set_page_config(page_title="Career Mode Scout", layout="wide")
_needed = [ROOT / "data/processed/panel.pkl", ROOT / "data/processed/forecasts_latest.csv"]
_missing = [p.relative_to(ROOT).as_posix() for p in _needed if not p.exists()]
if _missing:
    st.title("Career Mode Scout")
    st.error("The player data and models haven't been built yet "
             f"(missing: {', '.join(_missing)}).\n\n"
             "Stop the app, run this once from the project folder (about 2 minutes), "
             "then start the app again:\n\n"
             "`python scripts/run_all.py --quick`")
    st.stop()
cur, history, latest = load()
leagues = sorted(cur["league"].dropna().unique().tolist())

st.title("Career Mode Scout")
st.caption(f"Player database: {ed_name(latest)} ({len(cur):,} players). Forecasts are for the "
           f"ratings EA gives these players in {ed_name(latest + 1)} and {ed_name(latest + 2)}.")

tab_kids, tab_squad, tab_repl, tab_player = st.tabs(
    ["Wonderkid finder", "Squad builder", "Replacement finder", "Player outlook"])

# ------------------------------------------------------------------ wonderkids
with tab_kids:
    st.subheader("Who will be rated highest in two years?")
    c1, c2, c3, c4 = st.columns(4)
    max_age = c1.slider("Max age", 16, 30, 21)
    max_value = c2.number_input("Max value (€M)", 0.0, 200.0, 15.0, step=1.0)
    groups = c3.multiselect("Position group", ["GK", "DEF", "MID", "FWD"], ["GK", "DEF", "MID", "FWD"])
    lg = c4.multiselect("Leagues (empty = all)", leagues, [])
    sort_by = st.radio("Rank by", ["Forecast in 2 editions", "Forecast growth", "EA potential"],
                       horizontal=True)
    f = cur[(cur["age"] <= max_age) & (cur["value_eur"].fillna(0) <= max_value * 1e6)
            & cur["pos_group"].isin(groups)]
    if lg:
        f = f[f["league"].isin(lg)]
    f = f.assign(growth=f["pred_t2"] - f["overall"])
    key = {"Forecast in 2 editions": "pred_t2", "Forecast growth": "growth",
           "EA potential": "potential"}[sort_by]
    f = f.sort_values(key, ascending=False).head(100)
    show = pd.DataFrame({
        "Player": f["name"], "Age": f["age"], "Pos": f["primary_pos"], "Club": f["club"],
        "OVR": f["overall"], "POT": f["potential"],
        f"{ed_name(latest + 1)} (80% range)": [f"{m:.0f} ({lo:.0f}-{hi:.0f})" for m, lo, hi in
                                               zip(f["pred_t1"], f["pred_t1_lo"], f["pred_t1_hi"])],
        f"{ed_name(latest + 2)} (80% range)": [f"{m:.0f} ({lo:.0f}-{hi:.0f})" for m, lo, hi in
                                               zip(f["pred_t2"], f["pred_t2_lo"], f["pred_t2_hi"])],
        "Value": f["value_eur"].map(money), "Wage/wk": f["wage_eur"].map(money),
    })
    st.dataframe(show, hide_index=True)
    st.caption("The forecast learns how EA actually re-rated similar players in past editions "
               "(age, attributes, club, minutes proxy, recent rating changes). It predicts next "
               "year's EA ratings, which follow real-life form; growth inside a career-mode save "
               "is driven by the game's own (hidden) potential.")

# ------------------------------------------------------------------ squad builder
with tab_squad:
    st.subheader("Best signings for your budget")
    c1, c2, c3 = st.columns(3)
    league = c1.selectbox("Your league", leagues,
                          index=leagues.index("English Premier League") if "English Premier League" in leagues else 0)
    clubs = sorted(cur.loc[cur["league"] == league, "club"].dropna().unique().tolist())
    club = c2.selectbox("Your club", clubs)
    formation = c3.selectbox("Formation", list(FORMATIONS))
    squad = cur[cur["club"] == club]
    bill = float(squad["wage_eur"].fillna(0).sum())

    c1, c2, c3, c4 = st.columns(4)
    budget = c1.number_input("Transfer budget (€M)", 0.0, 1000.0, 50.0, step=5.0)
    wage_budget = c2.number_input("Weekly wage budget (€K)", 0.0, 20000.0,
                                  float(round(bill * 1.15 / 1e3)), step=10.0,
                                  help=f"Current wage bill: {money(bill)}/week")
    max_sign = c3.slider("Max signings", 1, 6, 3)
    max_age_s = c4.slider("Max age of signings", 16, 40, 40)

    c1, c2, c3, c4 = st.columns(4)
    objective = c1.selectbox("Optimise for", ["Next season (forecast)", "Current ratings",
                                              "Two seasons (forecast)"])
    allow_sales = c2.checkbox("Allow sales to fund signings", False)
    max_sales = c3.slider("Max sales", 0, 6, 3, disabled=not allow_sales)
    markup = c4.slider("Asking price (x value)", 1.0, 2.0, 1.2, step=0.05)
    market_leagues = st.multiselect("Buy from these leagues (empty = anywhere)", leagues, [])

    if st.button("Find the best transfers", type="primary"):
        market = cur[cur["club"] != club]
        if market_leagues:
            market = market[market["league"].isin(market_leagues)]
        boost_col = {"Current ratings": None, "Next season (forecast)": "pred_t1",
                     "Two seasons (forecast)": "pred_t2"}[objective]
        bs = bm = None
        if boost_col:
            bs = (squad[boost_col] - squad["overall"]).fillna(0)
            bm = (market[boost_col] - market["overall"]).fillna(0)
        s = TransferSettings(formation=formation, budget=budget * 1e6,
                             wage_budget=wage_budget * 1e3, max_signings=max_sign,
                             allow_sales=allow_sales, max_sales=max_sales, buy_markup=markup,
                             max_age=None if max_age_s >= 40 else max_age_s)
        with st.spinner("Solving the transfer problem exactly..."):
            plan = optimize_squad(squad, market, s, boost_squad=bs, boost_market=bm)
        if plan.status != "optimal":
            st.error(f"No plan found: {plan.status}")
        else:
            m1, m2, m3, m4 = st.columns(4)
            xi_label = {"Current ratings": "XI rating", "Next season (forecast)": "XI rating next season",
                        "Two seasons (forecast)": "XI rating in two seasons"}[objective]
            m1.metric(xi_label, f"{plan.xi_rating:.1f}",
                      f"{plan.xi_rating - plan.baseline_xi_rating:+.1f} vs no transfers")
            m2.metric("Spent", money(plan.spend))
            m3.metric("Raised from sales", money(plan.income))
            m4.metric("Wage bill / week", money(plan.wage_bill))
            st.caption(f"Exact optimum over {plan.n_candidates:,} candidates left after "
                       f"provably-safe pruning; solved in {plan.solve_seconds:.1f}s. "
                       f"Ratings are position ratings{'' if boost_col is None else ' + forecast change'}.")
            left, right = st.columns([3, 2])
            with left:
                st.markdown("**Starting XI**")
                xi = plan.xi.assign(rating_at_slot=plan.xi["rating_at_slot"].round(1))
                st.dataframe(xi[["slot", "name", "age", "rating_at_slot", "overall", "source"]],
                             hide_index=True)
            with right:
                st.markdown("**Buy**")
                if len(plan.buys):
                    b = plan.buys.assign(fee=plan.buys["fee"].map(money),
                                         wage_eur=plan.buys["wage_eur"].map(money))
                    st.dataframe(b[["name", "age", "club", "primary_pos", "overall", "potential",
                                    "fee", "wage_eur"]], hide_index=True)
                else:
                    st.write("No signing improves the XI under these limits.")
                if len(plan.sales):
                    st.markdown("**Sell**")
                    sl = plan.sales.assign(fee=plan.sales["fee"].map(money))
                    st.dataframe(sl[["name", "age", "primary_pos", "overall", "fee"]], hide_index=True)

# ------------------------------------------------------------------ replacements
with tab_repl:
    st.subheader("Cheaper players with the same profile")
    idx = similarity_index(cur, latest)
    options = cur.sort_values("overall", ascending=False)
    labels = (options["name"] + " (" + options["club"].fillna("free agent") + ", "
              + options["overall"].astype(int).astype(str) + ")").tolist()
    choice = st.selectbox("Player to replace", range(len(labels)), format_func=lambda i: labels[i])
    target = options.iloc[choice]
    c1, c2, c3, c4 = st.columns(4)
    fee_cap = c1.number_input("Max fee (€M)", 0.0, 300.0,
                              float(round(0.5 * (target["value_eur"] or 0) / 1e6, 1)), step=1.0)
    age_cap = c2.slider("Max age", 16, 40, 40, key="repl_age")
    min_ovr = c3.slider("Min overall", 40, 99, int(max(40, target["overall"] - 8)))
    same_pos = c4.checkbox("Same position only", True)
    lg2 = st.multiselect("Leagues (empty = all)", leagues, [], key="repl_leagues")
    res = idx.find(int(target["id"]), max_fee=fee_cap * 1e6, max_age=age_cap,
                   min_overall=min_ovr, same_position=same_pos, leagues=lg2 or None,
                   exclude_club=target["club"] if isinstance(target["club"], str) else None, k=15)
    res = res.merge(cur[["id", "pred_t1"]], on="id", how="left", suffixes=("", "_fc"))
    out = pd.DataFrame({
        "Player": res["name"], "Age": res["age"], "Pos": res["primary_pos"], "Club": res["club"],
        "OVR": res["overall"], f"{ed_name(latest + 1)} forecast": res["pred_t1"].round(0),
        "Similarity": res["similarity"].round(3), "Fee": res["fee"].map(money),
        "Saving": res["saving"].map(money),
    })
    if len(out):
        st.dataframe(out, hide_index=True)
    else:
        st.info("Nobody matches these filters. Raise the fee cap or lower the minimum overall.")
    st.caption("Similarity = cosine similarity of standardised attribute profiles "
               f"({'FAISS' if idx.use_faiss else 'NumPy'} exact search).")

# ------------------------------------------------------------------ player outlook
with tab_player:
    st.subheader("Rating history and forecast")
    choice2 = st.selectbox("Player", range(len(labels)), format_func=lambda i: labels[i],
                           key="outlook_player")
    p = options.iloc[choice2]
    h = history[history["id"] == p["id"]].sort_values("edition")
    chart = pd.DataFrame({"edition": h["edition"], "Overall": h["overall"],
                          "Potential": h["potential"]}).set_index("edition")
    fc = pd.DataFrame({"edition": [latest, latest + 1, latest + 2],
                       "Forecast": [p["overall"], p["pred_t1"], p["pred_t2"]]}).set_index("edition")
    st.line_chart(chart.join(fc, how="outer"))
    c1, c2, c3 = st.columns(3)
    c1.metric(ed_name(latest), int(p["overall"]), f"POT {int(p['potential'])}", delta_color="off")
    c2.metric(ed_name(latest + 1), f"{p['pred_t1']:.0f}",
              f"80% range {p['pred_t1_lo']:.0f}-{p['pred_t1_hi']:.0f}", delta_color="off")
    c3.metric(ed_name(latest + 2), f"{p['pred_t2']:.0f}",
              f"80% range {p['pred_t2_lo']:.0f}-{p['pred_t2_hi']:.0f}", delta_color="off")
