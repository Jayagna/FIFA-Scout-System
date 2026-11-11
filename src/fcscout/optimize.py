"""Squad builder: choose who to buy (and optionally sell) to get the strongest squad.

Mixed-integer program, solved exactly with HiGHS (scipy.optimize.milp):

  variables  x_p in {0,1}  player p is in the final squad (current players: keep;
                           candidates: buy)
             y_pk in {0,1} player p starts in formation slot k
             z_pk in {0,1} player p is the first backup for slot k (the "second XI")
  maximise   sum_pk r_pk * y_pk + w * sum_pk r_pk * z_pk
             - tiny tie-breakers (spend less, sell less)
  s.t.       sum_p y_pk = 1, sum_p z_pk <= 1        every slot has a starter (and a backup)
             sum_k (y_pk + z_pk) <= x_p             a player fills one role at most
             sum_buy cost_p - sum_sold sell_p <= transfer budget
             sum_squad wage_p <= wage budget
             #buys <= max_signings, #sales <= max_sales
             squad size and per-position depth limits

w (bench_weight) is how much the second XI counts: backups play a real share of the
minutes (rotation, injuries, suspensions), so with w > 0 a useful backup is worth
keeping and a player in neither XI is free to sell to fund signings. w = 0 plans for
the starting XI only.

r_pk is the player's sofifa rating *at that position* (an ST playing LW is
rated as a LW), optionally shifted by the growth forecast. With a stay-in-the-game
probability s_p it becomes an expected rating, s_p * r_pk + (1 - s_p) * b_k, where b_k
is the squad's backup level at slot k (capped at r_pk, so leaving never helps).
Optional objective-only bonuses (transfer rumours, shuffle noise) change which plan
is chosen but not the ratings reported. Locked signings are forced in, banned ones
removed, and earlier plans can be excluded to produce alternatives. The market is first
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
    bench_weight: float = 0.0            # how much the second XI counts (0 = starting XI only)
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


def _prepare(squad, market, boost_squad, boost_market, extra_squad=None, extra_market=None):
    """Align optional per-player series (indexed like the input frames) with the
    filtered, re-indexed frames the solver works on. extra_* map a column name to a
    Series; missing players get NaN."""
    squad = squad.assign(_boost=0.0 if boost_squad is None else boost_squad.reindex(squad.index))
    market = market.assign(_boost=0.0 if boost_market is None else boost_market.reindex(market.index))
    for name, ser in (extra_squad or {}).items():
        squad = squad.assign(**{name: ser.reindex(squad.index)})
    for name, ser in (extra_market or {}).items():
        market = market.assign(**{name: ser.reindex(market.index)})
    squad = squad.reset_index(drop=True)
    market = buyable(market)
    bs = None if boost_squad is None else squad["_boost"]
    bm = None if boost_market is None else market["_boost"]
    return squad, market, bs, bm


def backup_level(R_squad: np.ndarray, depth: int = 12) -> np.ndarray:
    """b_k: the squad's backup rating at each slot (its depth-th best option there)."""
    if len(R_squad) == 0:
        return np.zeros(R_squad.shape[1])
    srt = -np.sort(-R_squad, axis=0)
    return srt[min(depth, len(R_squad)) - 1]


def expected_ratings(R: np.ndarray, stay: pd.Series | None, backup: np.ndarray) -> np.ndarray:
    """s * r + (1 - s) * min(b, r) where r > 0; players who can't play a slot stay at 0."""
    if stay is None:
        return R
    s_ = stay.fillna(1.0).clip(0, 1).to_numpy()[:, None]
    E = s_ * R + (1 - s_) * np.minimum(backup[None, :], R)
    return np.where(R > 0, E, 0.0)


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

    If at least r = L + max_sales squad players rate >= x at slot k (L = 11 lineup
    places, or 22 when the second XI counts), then in any plan that uses a signing
    rated x at k, one of those r players is in neither lineup nor sold and can take
    his place instead. So T_k is the r-th best squad rating at slot k (exact pruning,
    no optimality lost)."""
    r = (22 if s.bench_weight > 0 else 11) + (s.max_sales if s.allow_sales else 0)
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
                    wage: np.ndarray, locked: np.ndarray | None = None, n_excluded: int = 0):
    """Exact reduction of the transfer market.

    Returns (pool indices, thresholds T, slot_ok): slot_ok[i, k] says whether pool player
    i may be used at slot k at all. Two exact rules:
      * T_k (see _slot_thresholds): below it a signing is never needed at k;
      * slot-wise dominance: if at least m players of the same position group are at
        least as good *at slot k* and no dearer in fee or wage, then in any plan that
        uses p at k, one of them is not signed and can take p's place there (same
        role, same group, no more money), so the pair (p, k) can go.
    With n_excluded earlier plans ruled out, m = max_signings + n_excluded: of those
    dominators at least n_excluded + 1 are unused, each swap gives a different set of
    signings, and at most n_excluded sets are excluded, so one swap is always allowed.
    Locked players are always kept."""
    max_spend = s.budget + (s.sell_factor * squad["value_eur"].fillna(0).sort_values(ascending=False)
                            .head(s.max_sales).sum() if s.allow_sales else 0)
    ok = ask <= max_spend
    if s.max_age is not None:
        ok &= market["age"].to_numpy() <= s.max_age
    T = _slot_thresholds(R_squad, s)
    R_eff = np.where((R_market > T[None, :]) & ok[:, None], R_market, 0.0)
    groups = market["pos_group"].astype(str).to_numpy()
    ids = market["id"].to_numpy()
    keep = np.zeros(len(market), bool)
    slot_ok = np.zeros(R_market.shape, bool)
    m = max(s.max_signings, 1) + n_excluded
    _, first, inverse = np.unique(R_eff, axis=1, return_index=True, return_inverse=True)  # CB1/CB2
    for g in np.unique(groups):
        in_g = groups == g
        for u, col in enumerate(first):
            idx = np.where(in_g & (R_eff[:, col] > 0))[0]
            if not len(idx):
                continue
            good = idx[_undominated(R_eff[idx, col:col + 1], ask[idx], wage[idx], ids[idx], m)]
            slot_ok[np.ix_(good, np.where(inverse.ravel() == u)[0])] = True
    keep |= slot_ok.any(axis=1)
    for g in np.unique(groups):
        if s.allow_sales:
            # bench fillers can matter when sales would break squad-size/depth limits:
            # keep the players not beaten on both fee and wage by m others
            idx = np.where(ok & (groups == g))[0]
            if len(idx):
                zero = np.zeros((len(idx), 1))
                keep[idx[_undominated(zero, ask[idx], wage[idx], ids[idx], m)]] = True
    if locked is not None:
        keep |= locked
        slot_ok[locked] = R_market[locked] > 0      # locked players may play anywhere they can
    pool = np.where(keep)[0]
    return pool, T, slot_ok[pool]


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
    notes: list = field(default_factory=list)
    bench: pd.DataFrame = field(default_factory=pd.DataFrame)     # second XI (bench_weight > 0)
    surplus: pd.DataFrame = field(default_factory=pd.DataFrame)   # kept, but in neither XI
    bench_rating: float = float("nan")


def optimize_squad(squad: pd.DataFrame, market: pd.DataFrame, settings: TransferSettings,
                   boost_squad: pd.Series | None = None,
                   boost_market: pd.Series | None = None,
                   stay_squad: pd.Series | None = None,
                   stay_market: pd.Series | None = None,
                   bonus_market: pd.Series | None = None,
                   ban_ids=None, lock_ids=None, exclude_buy_sets=None) -> Plan:
    """boost_*: per-player rating change (forecast). stay_*: probability the player is
    still in the game at the horizon. bonus_market: objective-only extra value for
    signing a player (rumours, shuffle noise). ban_ids: never sign; lock_ids: must
    sign; exclude_buy_sets: lists of player ids, each an earlier plan's signings that
    must not be repeated exactly."""
    import time

    s = settings
    notes = []
    ban = set(int(i) for i in (ban_ids or []))
    if ban:
        market = market[~market["id"].isin(ban)]
    extra_sq = {"_stay": stay_squad} if stay_squad is not None else {}
    extra_mk = {}
    if stay_market is not None:
        extra_mk["_stay"] = stay_market
    if bonus_market is not None:
        extra_mk["_bonus"] = bonus_market
    squad, market, boost_squad, boost_market = _prepare(squad, market, boost_squad, boost_market,
                                                        extra_sq, extra_mk)
    slots = FORMATIONS[s.formation]
    R_sq_raw = slot_ratings(squad, s.formation, boost_squad)
    R_mk_raw = slot_ratings(market, s.formation, boost_market)
    use_stay = stay_squad is not None or stay_market is not None
    backup = backup_level(R_sq_raw)
    R_sq = expected_ratings(R_sq_raw, squad["_stay"] if "_stay" in squad else None, backup) if use_stay else R_sq_raw
    R_mk_true = expected_ratings(R_mk_raw, market["_stay"] if "_stay" in market else None, backup) if use_stay else R_mk_raw
    R_mk_obj = R_mk_true
    if "_bonus" in market:
        R_mk_obj = R_mk_true + market["_bonus"].fillna(0).to_numpy()[:, None] * (R_mk_true > 0)
    lock = set(int(i) for i in (lock_ids or []))
    locked_all = market["id"].isin(lock).to_numpy()
    missing = lock - set(market.loc[locked_all, "id"].astype(int))
    if missing:
        notes.append(f"{len(missing)} locked player(s) can't be bought (no club, value or wage in the data)")
    excluded = [set(int(i) for i in ids) for ids in (exclude_buy_sets or []) if len(ids)]
    ask_all, _, wage_mk_all = _costs(market, s)
    pool, T, slot_ok = _candidate_pool(squad, market, s, R_sq, R_mk_obj, ask_all, wage_mk_all,
                                       locked=locked_all, n_excluded=len(excluded))
    mk = market.iloc[pool].reset_index(drop=True)
    R_mk = R_mk_obj[pool]
    R_mk_report = R_mk_true[pool]
    R_mk_if_stays = R_mk_raw[pool]
    locked = locked_all[pool]
    ask, wage_mk = ask_all[pool], wage_mk_all[pool]
    _, sell, wage_sq = _costs(squad, s)

    nS, nC, K = len(squad), len(mk), len(slots)
    nP = nS + nC
    R = np.vstack([R_sq, R_mk]) if nC else R_sq
    R_true = np.vstack([R_sq, R_mk_report]) if nC else R_sq
    R_stays = np.vstack([R_sq_raw, R_mk_if_stays]) if nC else R_sq_raw
    # y variables only where they can matter: any squad player with a rating there,
    # signings only above the slot threshold (see _slot_thresholds); locked signings
    # wherever they can play
    one_role = [(p, k) for p in range(nS) for k in range(K) if R[p, k] > 0]
    one_role += [(p, k) for p in range(nS, nP) for k in range(K)
                 if slot_ok[p - nS, k] and R[p, k] > (0 if locked[p - nS] else max(T[k], 0))]
    w = float(s.bench_weight)
    roles = (0, 1) if w > 0 else (0,)
    pairs = [(p, k, role) for role in roles for p, k in one_role]   # role 0 = XI, 1 = second XI
    nY = len(pairs)
    n = nP + nY
    c = np.zeros(n)
    # tie-breakers, far below one rating point: cheaper buys, fewer sales
    c[nS:nP] = ask * 1e-10
    c[:nS] = -1e-3  # keeping a player is (very slightly) preferred over selling him
    for j, (p, k, role) in enumerate(pairs):
        c[nP + j] = -R[p, k] * (1.0 if role == 0 else w)

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

    by_slot = {(k, role): {} for k in range(K) for role in roles}
    by_player = {p: {} for p in range(nP)}
    for j, (p, k, role) in enumerate(pairs):
        by_slot[k, role][nP + j] = 1.0
        by_player[p][nP + j] = 1.0
    for (k, role), coefs in by_slot.items():
        if role == 0:
            add(coefs, 1, 1)                 # every slot has a starter
        elif coefs:
            add(coefs, 0, 1)                 # and at most one first backup
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
    add({nS + i: 1.0 for i in range(nC)}, 0, max(s.max_signings, int(locked.sum())))
    if int(locked.sum()) > s.max_signings:
        notes.append("more players locked than signings allowed; all locked players are signed")
    pool_ids = mk["id"].astype(int).to_numpy() if nC else np.array([], int)
    for ex in excluded:
        members = [nS + i for i in range(nC) if pool_ids[i] in ex]
        if len(members) == len(ex):        # otherwise that plan is impossible anyway
            add({m_: 1.0 for m_ in members}, -np.inf, len(ex) - 1)
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
    lb[nS:nP] = np.where(locked, 1.0, 0.0) if nC else lb[nS:nP]
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
                    0, 0, 0, np.nan, _xi_only(squad, R_sq), nC, dt, notes)
    x = np.round(res.x).astype(int)
    lineup_rows = {0: [], 1: []}
    allp = pd.concat([squad.assign(_src="squad"), mk.assign(_src="signing")], ignore_index=True)
    used = set()
    for j, (p, k, role) in enumerate(pairs):
        if x[nP + j]:
            row = allp.iloc[p]
            used.add(p)
            lineup_rows[role].append({"slot": slots[k][0], "name": row["name"], "id": row["id"],
                                      "rating_at_slot": R_true[p, k], "rating_if_stays": R_stays[p, k],
                                      "stay": float(row["_stay"]) if "_stay" in row and pd.notna(row["_stay"]) else 1.0,
                                      "overall": row["overall"], "age": row["age"],
                                      "source": row["_src"], "club": row.get("club")})
    order = {sl[0]: i for i, sl in enumerate(slots)}

    def lineup(rows_):
        if not rows_:
            return pd.DataFrame(columns=["slot", "name", "id", "rating_at_slot", "rating_if_stays", "stay",
                                         "overall", "age", "source", "club"])
        return pd.DataFrame(rows_).sort_values("slot", key=lambda s_: s_.map(order)).reset_index(drop=True)
    xi, bench = lineup(lineup_rows[0]), lineup(lineup_rows[1])
    bought = np.where(x[nS:nP] == 1)[0]
    buys = mk.iloc[bought][["id", "name", "age", "club", "primary_pos", "overall", "potential",
                            "value_eur", "wage_eur"]].copy()
    buys["fee"] = ask[bought]
    sold = np.where(x[:nS] == 0)[0]
    sales = squad.iloc[sold][["id", "name", "age", "primary_pos", "overall", "value_eur"]].copy()
    sales["fee"] = sell[sold]
    surplus = pd.DataFrame()
    if w > 0:
        spare = [i for i in range(nS) if x[i] == 1 and i not in used]
        surplus = squad.iloc[spare][["id", "name", "age", "primary_pos", "overall", "value_eur", "wage_eur"]].copy()
        surplus["fee"] = sell[spare]
        surplus = surplus.sort_values("fee", ascending=False)
    wage_bill = float(wage_sq @ x[:nS] + (wage_mk @ x[nS:nP] if nC else 0))
    return Plan("optimal" if res.status == 0 else res.message, float(-res.fun), xi, buys, sales,
                float(buys["fee"].sum()), float(sales["fee"].sum()), wage_bill,
                float(xi["rating_at_slot"].mean()), _xi_only(squad, R_sq), nC, dt, notes,
                bench=bench, surplus=surplus,
                bench_rating=float(bench["rating_at_slot"].mean()) if len(bench) else float("nan"))


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
