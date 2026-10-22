"""Squad builder: choose who to buy (and optionally sell) to get the strongest XI.

Mixed-integer program, solved exactly with HiGHS (scipy.optimize.milp):

  variables  x_p in {0,1}  player p is in the final squad (current players: keep;
                           candidates: buy)
             y_pk in {0,1} player p starts in formation slot k
  maximise   sum_pk r_pk * y_pk  - tiny tie-breakers (spend less, sell less)
  s.t.       sum_p y_pk = 1                         every slot filled
             sum_k y_pk <= x_p                      a player starts at most once
             sum_buy cost_p - sum_sold sell_p <= transfer budget
             sum_squad wage_p <= wage budget
             #buys <= max_signings, #sales <= max_sales
             squad size and per-position depth limits

r_pk is the player's sofifa rating *at that position* (an ST playing LW is
rated as a LW), optionally shifted by the growth forecast. The market is first
reduced with two pruning rules that cannot remove the optimum (see
_slot_thresholds and _undominated), which keeps the solve to about a second.
A greedy baseline (what you would do by hand: biggest affordable upgrade
first) is included for comparison.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

FORMATIONS: dict[str, list[tuple[str, str]]] = {
    # slot name, sofifa position-rating column
    "4-3-3": [("GK", "gk"), ("LB", "lb"), ("CB1", "cb"), ("CB2", "cb"), ("RB", "rb"),
              ("CDM", "cdm"), ("CM1", "cm"), ("CM2", "cm"), ("LW", "lw"), ("ST", "st"), ("RW", "rw")],
    "4-2-3-1": [("GK", "gk"), ("LB", "lb"), ("CB1", "cb"), ("CB2", "cb"), ("RB", "rb"),
                ("CDM1", "cdm"), ("CDM2", "cdm"), ("LM", "lm"), ("CAM", "cam"), ("RM", "rm"),
                ("ST", "st")],
    "4-4-2": [("GK", "gk"), ("LB", "lb"), ("CB1", "cb"), ("CB2", "cb"), ("RB", "rb"),
              ("LM", "lm"), ("CM1", "cm"), ("CM2", "cm"), ("RM", "rm"), ("ST1", "st"), ("ST2", "st")],
    "3-5-2": [("GK", "gk"), ("CB1", "cb"), ("CB2", "cb"), ("CB3", "cb"), ("LWB", "lwb"),
              ("CDM", "cdm"), ("CM1", "cm"), ("CM2", "cm"), ("RWB", "rwb"), ("ST1", "st"),
              ("ST2", "st")],
}

DEPTH_GROUPS = {"GK": 2, "DEF": 6, "MID": 6, "FWD": 3}


@dataclass
class TransferSettings:
    formation: str = "4-3-3"
    budget: float = 50e6                 # transfer budget (EUR)
    wage_budget: float | None = None     # weekly wage budget (EUR); None = current bill * 1.15
    max_signings: int = 3
    allow_sales: bool = False
    max_sales: int = 3
    buy_markup: float = 1.0              # asking price = value * markup (capped by release clause)
    sell_factor: float = 0.9             # sale fee = value * factor
    squad_min: int = 18
    squad_max: int = 35
    max_age: int | None = None           # for signings only
    min_depth: dict = field(default_factory=lambda: dict(DEPTH_GROUPS))
    time_limit: float = 60.0             # seconds; the plan reports if it was hit


def buyable(market: pd.DataFrame) -> pd.DataFrame:
    """Players you can actually bid for: at a club, not on loan, with a known value and wage.
    (Free agents in these files have no wage data, so they are left out.)"""
    m = market[market["club"].notna() & market["value_eur"].gt(0) & market["wage_eur"].notna()]
    if "on_loan" in m:
        m = m[m["on_loan"].fillna(0).eq(0)]
    return m.reset_index(drop=True)


def _prepare(squad, market, boost_squad, boost_market):
    """Align optional per-player boosts (indexed like the input frames) with the
    filtered, re-indexed frames the solver works on."""
    squad = squad.assign(_boost=0.0 if boost_squad is None else boost_squad.reindex(squad.index))
    market = market.assign(_boost=0.0 if boost_market is None else boost_market.reindex(market.index))
    squad = squad.reset_index(drop=True)
    market = buyable(market)
    bs = None if boost_squad is None else squad["_boost"]
    bm = None if boost_market is None else market["_boost"]
    return squad, market, bs, bm


def slot_ratings(df: pd.DataFrame, formation: str, boost: pd.Series | None = None) -> np.ndarray:
    """rating matrix (players x slots). boost: optional per-player delta (e.g. forecast growth)."""
    cols = []
    for _, pos in FORMATIONS[formation]:
        col = f"pos_{pos}"
        if col in df and df[col].notna().any():
            r = df[col].astype(float)
            if pos == "gk":   # some editions have no gk column: keepers use overall
                r = r.fillna(df["overall"].where(df["primary_pos"].eq("GK")))
        else:
            r = df["overall"].where(df["primary_pos"].str.upper().eq(pos.upper()))
        cols.append(r.fillna(0).to_numpy())
    R = np.column_stack(cols)
    if boost is not None:
        R = R + boost.fillna(0).to_numpy()[:, None] * (R > 0)
    return R


def _costs(df: pd.DataFrame, s: TransferSettings):
    value = df["value_eur"].fillna(0).to_numpy(float)
    ask = value * s.buy_markup
    rc = df["release_clause_eur"].to_numpy(float) if "release_clause_eur" in df else np.full(len(df), np.nan)
    ask = np.where(np.isfinite(rc) & (rc > 0), np.minimum(ask, rc), ask)
    return ask, value * s.sell_factor, df["wage_eur"].fillna(0).to_numpy(float)


def _slot_thresholds(R_squad: np.ndarray, s: TransferSettings) -> np.ndarray:
    """T_k: a signing rated <= T_k at slot k is never needed there.

    If at least r = 11 + max_sales squad players rate >= x at slot k, then in any
    plan that starts a signing rated x at k, one of those r players is neither
    starting nor sold and can take the slot instead. So T_k is the r-th best squad
    rating at slot k (exact pruning, no optimality lost)."""
    r = 11 + (s.max_sales if s.allow_sales else 0)
    if len(R_squad) < r:
        return np.zeros(R_squad.shape[1])
    return -np.sort(-R_squad, axis=0)[r - 1]


def _undominated(R: np.ndarray, fee: np.ndarray, wage: np.ndarray, ids: np.ndarray, m: int,
                 chunk: int = 256) -> np.ndarray:
    """Mask of players with fewer than m dominators.

    d dominates c if d is at least as good at every slot and no more expensive in fee
    and wage (ties broken by id, so dominance is a strict order). With at most m
    signings, a player with >= m dominators can always be swapped for an unused one
    without losing rating or feasibility, so dropping him is exact."""
    n = len(R)
    keep = np.ones(n, bool)
    if n <= m:
        return keep
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        Rc = R[a:b, None, :]
        ge = (R[None, :, :] >= Rc).all(axis=2)
        ge &= fee[None, :] <= fee[a:b, None]
        ge &= wage[None, :] <= wage[a:b, None]
        strict = ((R[None, :, :] > Rc).any(axis=2) | (fee[None, :] < fee[a:b, None])
                  | (wage[None, :] < wage[a:b, None]) | (ids[None, :] < ids[a:b, None]))
        keep[a:b] = (ge & strict).sum(axis=1) < m
    return keep


def _candidate_pool(squad: pd.DataFrame, market: pd.DataFrame, s: TransferSettings,
                    R_squad: np.ndarray, R_market: np.ndarray, ask: np.ndarray,
                    wage: np.ndarray):
    """Exact reduction of the transfer market. Returns (pool indices, thresholds T)."""
    max_spend = s.budget + (s.sell_factor * squad["value_eur"].fillna(0).sort_values(ascending=False)
                            .head(s.max_sales).sum() if s.allow_sales else 0)
    ok = ask <= max_spend
    if s.max_age is not None:
        ok &= market["age"].to_numpy() <= s.max_age
    T = _slot_thresholds(R_squad, s)
    R_eff = np.where(R_market > T[None, :], R_market, 0.0)
    _, first = np.unique(R_eff, axis=1, return_index=True)   # duplicate slots (CB1/CB2)
    R_u = R_eff[:, np.sort(first)]
    useful = ok & (R_eff > 0).any(axis=1)
    groups = market["pos_group"].astype(str).to_numpy()
    ids = market["id"].to_numpy()
    keep = np.zeros(len(market), bool)
    m = max(s.max_signings, 1)
    for g in np.unique(groups):
        idx = np.where(useful & (groups == g))[0]
        if len(idx):
            keep[idx[_undominated(R_u[idx], ask[idx], wage[idx], ids[idx], m)]] = True
        if s.allow_sales:
            # bench fillers can matter when sales would break squad-size/depth limits:
            # keep the players not beaten on both fee and wage by m others
            idx = np.where(ok & (groups == g))[0]
            if len(idx):
                zero = np.zeros((len(idx), 1))
                keep[idx[_undominated(zero, ask[idx], wage[idx], ids[idx], m)]] = True
    return np.where(keep)[0], T


@dataclass
class Plan:
    status: str
    objective: float
    xi: pd.DataFrame
    buys: pd.DataFrame
    sales: pd.DataFrame
    spend: float
    income: float
    wage_bill: float
    xi_rating: float
    baseline_xi_rating: float
    n_candidates: int
    solve_seconds: float


def optimize_squad(squad: pd.DataFrame, market: pd.DataFrame, settings: TransferSettings,
                   boost_squad: pd.Series | None = None,
                   boost_market: pd.Series | None = None) -> Plan:
    import time

    s = settings
    squad, market, boost_squad, boost_market = _prepare(squad, market, boost_squad, boost_market)
    slots = FORMATIONS[s.formation]
    R_sq = slot_ratings(squad, s.formation, boost_squad)
    R_mk_all = slot_ratings(market, s.formation, boost_market)
    ask_all, _, wage_mk_all = _costs(market, s)
    pool, T = _candidate_pool(squad, market, s, R_sq, R_mk_all, ask_all, wage_mk_all)
    mk = market.iloc[pool].reset_index(drop=True)
    R_mk = R_mk_all[pool]
    ask, wage_mk = ask_all[pool], wage_mk_all[pool]
    _, sell, wage_sq = _costs(squad, s)

    nS, nC, K = len(squad), len(mk), len(slots)
    nP = nS + nC
    R = np.vstack([R_sq, R_mk]) if nC else R_sq
    # y variables only where they can matter: any squad player with a rating there,
    # signings only above the slot threshold (see _slot_thresholds)
    pairs = [(p, k) for p in range(nS) for k in range(K) if R[p, k] > 0]
    pairs += [(p, k) for p in range(nS, nP) for k in range(K) if R[p, k] > max(T[k], 0)]
    nY = len(pairs)
    n = nP + nY
    c = np.zeros(n)
    # tie-breakers, far below one rating point: cheaper buys, fewer sales
    c[nS:nP] = ask * 1e-10
    c[:nS] = -1e-3  # keeping a player is (very slightly) preferred over selling him
    for j, (p, k) in enumerate(pairs):
        c[nP + j] = -R[p, k]

    rows, cols, vals, lo, hi = [], [], [], [], []
    r = 0

    def add(coefs: dict[int, float], lb: float, ub: float):
        nonlocal r
        for col, v in coefs.items():
            rows.append(r)
            cols.append(col)
            vals.append(v)
        lo.append(lb)
        hi.append(ub)
        r += 1

    by_slot = {k: {} for k in range(K)}
    by_player = {p: {} for p in range(nP)}
    for j, (p, k) in enumerate(pairs):
        by_slot[k][nP + j] = 1.0
        by_player[p][nP + j] = 1.0
    for k in range(K):
        add(by_slot[k], 1, 1)
    for p in range(nP):
        if by_player[p]:
            add({**by_player[p], p: -1.0}, -np.inf, 0)
    # transfer budget: buys + kept*sell <= budget + total sell value
    budget_row = {nS + i: ask[i] for i in range(nC)}
    if s.allow_sales:
        budget_row.update({i: sell[i] for i in range(nS)})
        add(budget_row, -np.inf, s.budget + sell.sum())
    else:
        add(budget_row, -np.inf, s.budget)
    wage_budget = s.wage_budget if s.wage_budget is not None else wage_sq.sum() * 1.15
    add({**{i: wage_sq[i] for i in range(nS)}, **{nS + i: wage_mk[i] for i in range(nC)}},
        -np.inf, max(wage_budget, 0))
    add({nS + i: 1.0 for i in range(nC)}, 0, s.max_signings)
    size_lo = min(s.squad_min, nS)
    # without sales, a big squad must still be allowed to add the signings
    size_hi = max(s.squad_max, nS) if s.allow_sales else max(s.squad_max, nS + s.max_signings)
    add({i: 1.0 for i in range(nP)}, size_lo, size_hi)
    groups = np.concatenate([squad["pos_group"].astype(str).to_numpy(),
                             mk["pos_group"].astype(str).to_numpy()]) if nC else squad["pos_group"].astype(str).to_numpy()
    for g, d in s.min_depth.items():
        members = np.where(groups == g)[0]
        have = int((squad["pos_group"].astype(str) == g).sum())
        add({int(i): 1.0 for i in members}, min(d, have), np.inf)

    A = coo_matrix((vals, (rows, cols)), shape=(r, n)).tocsr()
    lb = np.zeros(n)
    ub = np.ones(n)
    if s.allow_sales:
        # sum_squad (1 - x) <= max_sales   <=>   -sum_squad x <= max_sales - nS
        A2 = coo_matrix(([-1.0] * nS, ([0] * nS, list(range(nS)))), shape=(1, n)).tocsr()
        cons = [LinearConstraint(A, lo, hi), LinearConstraint(A2, -np.inf, s.max_sales - nS)]
    else:
        lb[:nS] = 1.0  # keep everyone
        cons = [LinearConstraint(A, lo, hi)]
    t0 = time.time()
    res = milp(c, integrality=np.ones(n), bounds=Bounds(lb, ub), constraints=cons,
               options={"time_limit": s.time_limit, "mip_rel_gap": 1e-6})
    dt = time.time() - t0
    if res.x is None:
        empty_buys = pd.DataFrame(columns=["id", "name", "age", "club", "primary_pos", "overall",
                                           "potential", "value_eur", "wage_eur", "fee"])
        empty_sales = pd.DataFrame(columns=["id", "name", "age", "primary_pos", "overall",
                                            "value_eur", "fee"])
        return Plan(res.message, np.nan, pd.DataFrame(), empty_buys, empty_sales,
                    0, 0, 0, np.nan, _xi_only(squad, R_sq), nC, dt)
    x = np.round(res.x).astype(int)
    xi_rows = []
    allp = pd.concat([squad.assign(_src="squad"), mk.assign(_src="signing")], ignore_index=True)
    for j, (p, k) in enumerate(pairs):
        if x[nP + j]:
            row = allp.iloc[p]
            xi_rows.append({"slot": slots[k][0], "name": row["name"], "id": row["id"],
                            "rating_at_slot": R[p, k], "overall": row["overall"], "age": row["age"],
                            "source": row["_src"], "club": row.get("club")})
    order = {sl[0]: i for i, sl in enumerate(slots)}
    xi = pd.DataFrame(xi_rows).sort_values("slot", key=lambda s_: s_.map(order)).reset_index(drop=True)
    bought = np.where(x[nS:nP] == 1)[0]
    buys = mk.iloc[bought][["id", "name", "age", "club", "primary_pos", "overall", "potential",
                            "value_eur", "wage_eur"]].copy()
    buys["fee"] = ask[bought]
    sold = np.where(x[:nS] == 0)[0]
    sales = squad.iloc[sold][["id", "name", "age", "primary_pos", "overall", "value_eur"]].copy()
    sales["fee"] = sell[sold]
    wage_bill = float(wage_sq @ x[:nS] + (wage_mk @ x[nS:nP] if nC else 0))
    return Plan("optimal" if res.status == 0 else res.message, float(-res.fun), xi, buys, sales,
                float(buys["fee"].sum()), float(sales["fee"].sum()), wage_bill,
                float(xi["rating_at_slot"].mean()), _xi_only(squad, R_sq), nC, dt)


def _xi_only(squad: pd.DataFrame, R_sq: np.ndarray) -> float:
    """Best XI from the current squad alone (assignment problem, exact)."""
    from scipy.optimize import linear_sum_assignment
    if len(squad) < R_sq.shape[1]:
        return np.nan
    rows, cols = linear_sum_assignment(-R_sq)
    return float(R_sq[rows, cols].mean())


def greedy_plan(squad: pd.DataFrame, market: pd.DataFrame, settings: TransferSettings,
                boost_squad=None, boost_market=None, rule: str = "biggest_upgrade") -> float:
    """Hand-made approach: repeatedly buy the best affordable upgrade over the current XI.

    rule: 'biggest_upgrade' (largest rating gain) or 'best_value' (gain per euro).
    Returns the average XI rating after the signings. No sales (compare with
    optimize_squad(allow_sales=False)).
    """
    from scipy.optimize import linear_sum_assignment

    s = settings
    squad, market, boost_squad, boost_market = _prepare(squad, market, boost_squad, boost_market)
    R_sq = slot_ratings(squad, s.formation, boost_squad)
    R_mk = slot_ratings(market, s.formation, boost_market)
    ask, _, wage_mk = _costs(market, s)
    _, _, wage_sq = _costs(squad, s)
    ok = np.ones(len(market), bool)
    if s.max_age is not None:
        ok &= market["age"].to_numpy() <= s.max_age
    wage_budget = s.wage_budget if s.wage_budget is not None else wage_sq.sum() * 1.15
    budget, wages = s.budget, wage_sq.sum()
    R = R_sq.copy()
    bought = set()
    for _ in range(s.max_signings):
        rows, cols = linear_sum_assignment(-R)
        cur = np.zeros(R.shape[1])
        cur[cols] = R[rows, cols]
        afford = ok & (ask <= budget) & (wages + wage_mk <= wage_budget)
        if bought:
            afford[list(bought)] = False
        idx = np.where(afford)[0]
        if len(idx) == 0:
            break
        gains = R_mk[idx] - cur[None, :]
        best_gain = gains.max(axis=1)
        if rule == "best_value":
            score = best_gain / (ask[idx] + 1e5)
        else:
            score = best_gain - ask[idx] * 1e-10
        j = int(np.argmax(score))
        if best_gain[j] <= 0:
            break
        p = idx[j]
        bought.add(p)
        budget -= ask[p]
        wages += wage_mk[p]
        R = np.vstack([R, R_mk[p]])
    rows, cols = linear_sum_assignment(-R)
    return float(R[rows, cols].mean())
