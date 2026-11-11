"""Everything the website asks for, as plain functions returning JSON-ready dicts.

The web server (server.py) is a thin layer over this module, so every response can
be tested without a browser or an HTTP client.
"""
from __future__ import annotations

import json
import math
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .data import FACE_STATS, GK_ATTRS, OUTFIELD_ATTRS, POS_RATING_COLS, PREMIER_LEAGUE
from .optimize import FORMATIONS, TransferSettings, optimize_squad
from .similar import SimilarityIndex

EDITION_NAMES = {15: "FIFA 15", 16: "FIFA 16", 17: "FIFA 17", 18: "FIFA 18", 19: "FIFA 19",
                 20: "FIFA 20", 21: "FIFA 21", 22: "FIFA 22", 23: "FIFA 23", 24: "EA FC 24",
                 25: "EA FC 25", 26: "EA FC 26", 27: "EA FC 27", 28: "EA FC 28"}

# where each formation slot stands on the pitch: x = 0 (left touchline) .. 100,
# y = 0 (own goal line) .. 100
SLOT_XY = {
    "GK": (50, 6), "LB": (14, 26), "RB": (86, 26), "CB1": (37, 20), "CB2": (63, 20),
    "CB3": (50, 19), "LWB": (10, 44), "RWB": (90, 44), "CDM": (50, 38), "CDM1": (37, 38),
    "CDM2": (63, 38), "CM1": (32, 52), "CM2": (68, 52), "LM": (13, 60), "RM": (87, 60),
    "CAM": (50, 64), "LW": (18, 78), "RW": (82, 78), "ST": (50, 86), "ST1": (38, 84),
    "ST2": (62, 84),
}
# 3-5-2 has three centre backs spread wider than a back four's pair
_SLOT_XY_352 = {"CB1": (27, 21), "CB2": (50, 18), "CB3": (73, 21)}

ATTRIBUTE_GROUPS = {
    "Attacking": ["attacking_crossing", "attacking_finishing", "attacking_heading_accuracy",
                  "attacking_short_passing", "attacking_volleys"],
    "Skill": ["skill_dribbling", "skill_curve", "skill_fk_accuracy", "skill_long_passing",
              "skill_ball_control"],
    "Movement": ["movement_acceleration", "movement_sprint_speed", "movement_agility",
                 "movement_reactions", "movement_balance"],
    "Power": ["power_shot_power", "power_jumping", "power_stamina", "power_strength",
              "power_long_shots"],
    "Mentality": ["mentality_aggression", "mentality_interceptions", "mentality_positioning",
                  "mentality_vision", "mentality_penalties", "mentality_composure"],
    "Defending": ["defending_marking", "defending_standing_tackle", "defending_sliding_tackle"],
    "Goalkeeping": GK_ATTRS,
}
_ATTR_LABEL = {
    "defending_marking": "Defensive awareness", "skill_fk_accuracy": "Free kick accuracy",
    "mentality_positioning": "Attacking positioning",
}


def attr_label(col: str) -> str:
    return _ATTR_LABEL.get(col, col.split("_", 1)[1].replace("_", " ").capitalize())


def edition_name(e: int) -> str:
    return EDITION_NAMES.get(int(e), f"Edition {e}")


def tier(overall: float) -> str:
    """Card colour as the game prints it: bronze < 65, silver 65-74, gold 75+."""
    if overall is None or not np.isfinite(overall):
        return "bronze"
    return "gold" if overall >= 75 else "silver" if overall >= 65 else "bronze"


def _fold(s: str) -> str:
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()


def clean(obj):
    """NaN/inf -> None and numpy scalars -> Python, recursively (JSON has no NaN)."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if not math.isfinite(f) else round(f, 4)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if obj is pd.NA or obj is pd.NaT:
        return None
    return obj


def load_rumours(path: Path, cur: pd.DataFrame) -> dict:
    """Read rumours.json (scripts/fetch_rumours.py) and index it by buying club. Links to
    players or clubs that aren't in the current edition are skipped."""
    path = Path(path)
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    ids = set(cur["id"].astype(int))
    clubs = set(cur["club"].dropna())
    by_club: dict = {}
    for link in raw.get("links", []):
        pid, club = int(link["player_id"]), link["to_club"]
        if pid in ids and club in clubs:
            by_club.setdefault(club, {})[pid] = link
    return {"generated": raw.get("generated"), "window_days": raw.get("window_days"),
            "sources": raw.get("sources", []), "by_club": by_club,
            "n_links": sum(len(v) for v in by_club.values())}


@dataclass
class Store:
    cur: pd.DataFrame                    # newest edition, with forecasts
    history: pd.DataFrame                # every edition: id, edition, overall, potential, club
    latest: int
    index: SimilarityIndex
    reports: dict = field(default_factory=dict)
    editions: dict = field(default_factory=dict)
    club_strength: dict = field(default_factory=dict)
    rumours: dict = field(default_factory=dict)

    def rumours_for(self, club: str) -> dict:
        """player id -> rumour link, for players linked with this club."""
        return self.rumours.get("by_club", {}).get(club, {})

    # ---------------------------------------------------------------- loading
    @classmethod
    def load(cls, root: Path | str) -> "Store":
        root = Path(root)
        panel = pd.read_pickle(root / "data/processed/panel.pkl")
        fc = pd.read_csv(root / "data/processed/forecasts_latest.csv")
        return cls.from_frames(panel, fc, reports_dir=root / "reports")

    @classmethod
    def from_frames(cls, panel: pd.DataFrame, forecasts: pd.DataFrame,
                    reports_dir: Path | None = None) -> "Store":
        latest = int(panel["edition"].max())
        cur = panel[panel["edition"] == latest].copy()
        for c in ("pos_group", "primary_pos", "preferred_foot"):
            if c in cur:
                cur[c] = cur[c].astype(str).replace({"nan": np.nan})
        keep = ["id", "pred_t1", "pred_t1_lo", "pred_t1_hi", "pred_t2", "pred_t2_lo", "pred_t2_hi",
                "surv_t1", "surv_t2"]
        cur = cur.merge(forecasts[[c for c in keep if c in forecasts]], on="id", how="left").reset_index(drop=True)
        cur["growth_t2"] = cur["pred_t2"] - cur["overall"]
        cur["_fold"] = (cur["name"].fillna("").map(_fold) + " | " + cur["long_name"].fillna("").map(_fold))
        hist_cols = ["id", "edition", "name", "overall", "potential", "club", "value_eur"]
        history = panel[[c for c in hist_cols if c in panel]].copy()
        reports = {}
        if reports_dir is not None:
            for name in ("forecast_metrics", "forecast_rolling", "optimizer_metrics",
                         "similarity_metrics", "rating_formulas"):
                p = Path(reports_dir) / f"{name}.json"
                if p.exists():
                    reports[name] = json.loads(p.read_text())
        editions = panel.groupby("edition").size().to_dict()
        ranked = cur.dropna(subset=["club"]).sort_values("overall", ascending=False)
        strength = ranked[ranked.groupby("club").cumcount() < 16].groupby("club")["overall"].mean().to_dict()
        rumours = load_rumours(Path(reports_dir).parent / "rumours/rumours.json", cur) if reports_dir else {}
        return cls(cur=cur, history=history, latest=latest, index=SimilarityIndex(cur),
                   reports=reports, editions=editions, club_strength=strength, rumours=rumours)

    def row(self, player_id: int) -> pd.Series:
        hit = self.cur.index[self.cur["id"] == int(player_id)]
        if len(hit) == 0:
            raise KeyError(player_id)
        return self.cur.loc[hit[0]]


# -------------------------------------------------------------------- shaping
def brief(r: pd.Series) -> dict:
    """The fields every player card needs."""
    return clean({
        "id": r["id"], "name": r["name"], "long_name": r.get("long_name"),
        "age": r["age"], "pos": r["primary_pos"], "positions": r.get("positions"),
        "club": r.get("club"), "league": r.get("league"), "nation": r.get("nationality"),
        "overall": r["overall"], "potential": r["potential"], "tier": tier(r["overall"]),
        "value": r.get("value_eur"), "wage": r.get("wage_eur"), "face": r.get("face_url"),
        "pred_t1": r.get("pred_t1"), "pred_t1_lo": r.get("pred_t1_lo"), "pred_t1_hi": r.get("pred_t1_hi"),
        "pred_t2": r.get("pred_t2"), "pred_t2_lo": r.get("pred_t2_lo"), "pred_t2_hi": r.get("pred_t2_hi"),
        "stay_t1": r.get("surv_t1"), "stay_t2": r.get("surv_t2"),
        "card": {s: r.get(s) for s in FACE_STATS},
    })


def _briefs(df: pd.DataFrame) -> list[dict]:
    return [brief(r) for _, r in df.iterrows()]


def formation_layout(name: str) -> list[dict]:
    xy = dict(SLOT_XY, **(_SLOT_XY_352 if name == "3-5-2" else {}))
    return [{"slot": s, "pos": p.upper(), "x": xy[s][0], "y": xy[s][1]} for s, p in FORMATIONS[name]]


def _league_order(cur: pd.DataFrame) -> list[str]:
    """Full leagues first, each group by strength (mean overall of each club's best 16).
    Some leagues are in the game with only their two or three biggest clubs, which
    would otherwise rank them above complete top flights."""
    top16 = (cur.dropna(subset=["league", "club"]).sort_values("overall", ascending=False)
             .groupby(["league", "club"]).head(16))
    g = top16.groupby("league").agg(strength=("overall", "mean"), clubs=("club", "nunique"))
    g["full"] = g["clubs"] >= 8
    return g.sort_values(["full", "strength"], ascending=[False, False]).index.tolist()


# -------------------------------------------------------------------- endpoints
def meta(store: Store) -> dict:
    cur = store.cur
    leagues = []
    for lg in _league_order(cur):
        clubs = (cur[cur["league"] == lg].groupby("club")["overall"]
                 .apply(lambda s: s.nlargest(16).mean()).sort_values(ascending=False))
        leagues.append({"name": lg, "clubs": clubs.index.tolist()})
    fm = store.reports.get("forecast_metrics", {})
    return clean({
        "edition": store.latest, "edition_name": edition_name(store.latest),
        "next_name": edition_name(store.latest + 1), "next2_name": edition_name(store.latest + 2),
        "players": len(cur), "player_seasons": int(sum(store.editions.values())),
        "editions": [{"edition": int(e), "name": edition_name(e), "players": int(n)}
                     for e, n in sorted(store.editions.items())],
        "leagues": leagues, "default_league": PREMIER_LEAGUE if PREMIER_LEAGUE in cur["league"].values else (leagues[0]["name"] if leagues else None),
        "formations": {f: formation_layout(f) for f in FORMATIONS},
        "rumours": {"generated": (store.rumours or {}).get("generated"),
                    "links": (store.rumours or {}).get("n_links", 0)},
        "forecast_test": {h: {"test": edition_name(v["test_edition"]) + " to " + edition_name(v["test_edition"] + int(h[1])),
                              "mae": v["overall"]["model"]["mae"],
                              "mae_potential": v["overall"]["ea_potential"]["mae"],
                              "mae_persistence": v["overall"]["persistence"]["mae"]}
                          for h, v in fm.items()},
    })


def search(store: Store, q: str, limit: int = 12) -> list[dict]:
    q = _fold(q).strip()
    if len(q) < 2:
        return []
    cur = store.cur
    hit = cur[cur["_fold"].str.contains(q, regex=False)]
    starts = hit["_fold"].str.startswith(q) | hit["_fold"].str.contains(" " + q, regex=False)
    hit = hit.assign(_s=starts.astype(int)).sort_values(["_s", "overall"], ascending=[False, False])
    return _briefs(hit.head(limit))


def wonderkids(store: Store, max_age: int = 21, max_value_m: float = 15.0,
               groups: list[str] | None = None, leagues: list[str] | None = None,
               sort: str = "pred_t2", limit: int = 60) -> dict:
    cur = store.cur
    f = cur[(cur["age"] <= max_age) & (cur["value_eur"].fillna(0) <= max_value_m * 1e6)]
    if groups:
        f = f[f["pos_group"].isin(groups)]
    if leagues:
        f = f[f["league"].isin(leagues)]
    key = {"pred_t2": "pred_t2", "growth": "growth_t2", "potential": "potential",
           "overall": "overall"}.get(sort, "pred_t2")
    f = f.sort_values([key, "overall"], ascending=False)
    return {"total": int(len(f)), "players": _briefs(f.head(limit))}


def risers(store: Store, limit: int = 10) -> list[dict]:
    """Under-21s the model rates highest two editions out (the home page strip)."""
    cur = store.cur
    f = cur[(cur["age"] <= 21) & cur["pred_t2"].notna()].sort_values("pred_t2", ascending=False)
    return _briefs(f.head(limit))


# which primary positions may fill a slot in the dream XI
_SLOT_FITS = {"gk": {"GK"}, "lb": {"LB", "LWB"}, "rb": {"RB", "RWB"}, "cb": {"CB"},
              "cdm": {"CDM", "CM"}, "cm": {"CM", "CDM", "CAM"}, "cam": {"CAM", "CM"},
              "lm": {"LM", "LW"}, "rm": {"RM", "RW"}, "lw": {"LW", "LM", "LF"},
              "rw": {"RW", "RM", "RF"}, "st": {"ST", "CF"}, "lwb": {"LWB", "LB"}, "rwb": {"RWB", "RB"}}


def dream_xi(store: Store, max_age: int = 21, formation: str = "4-3-3") -> list[dict]:
    """The best young XI two editions out: each slot takes the highest FC+2 forecast
    among players whose main position fits it (the home page's 3D pitch)."""
    pool = store.cur[(store.cur["age"] <= max_age) & store.cur["pred_t2"].notna()]
    pool = pool.sort_values("pred_t2", ascending=False)
    used, out = set(), []
    for slot in formation_layout(formation):
        fits = _SLOT_FITS[slot["pos"].lower()]
        cand = pool[pool["primary_pos"].isin(fits) & ~pool["id"].isin(used)]
        if cand.empty:
            continue
        r = cand.iloc[0]
        used.add(int(r["id"]))
        out.append(dict(slot, player=brief(r)))
    return out


def player(store: Store, player_id: int) -> dict:
    r = store.row(player_id)
    out = brief(r)
    is_gk = r["primary_pos"] == "GK"
    groups = {g: [{"key": c, "label": attr_label(c), "value": r.get(c)} for c in cols]
              for g, cols in ATTRIBUTE_GROUPS.items() if (g == "Goalkeeping") == is_gk or g == "Movement"}
    if not is_gk:
        groups = {g: v for g, v in groups.items() if g != "Goalkeeping"}
    h = store.history[store.history["id"] == r["id"]].sort_values("edition")
    hist = [{"edition": int(e), "name": edition_name(e), "overall": o, "potential": p, "club": c}
            for e, o, p, c in zip(h["edition"], h["overall"], h["potential"], h["club"])]
    forecast = [{"edition": store.latest + k, "name": edition_name(store.latest + k),
                 "pred": r.get(f"pred_t{k}"), "lo": r.get(f"pred_t{k}_lo"), "hi": r.get(f"pred_t{k}_hi")}
                for k in (1, 2)]
    similar = []
    try:
        res = store.index.find(int(r["id"]), same_position=True, k=6,
                               exclude_club=r["club"] if isinstance(r["club"], str) else None)
        similar = [dict(brief(store.row(i)), similarity=float(s)) for i, s in zip(res["id"], res["similarity"])]
    except KeyError:
        pass
    out.update(clean({
        "height": r.get("height_cm"), "weight": r.get("weight_kg"), "foot": r.get("preferred_foot"),
        "weak_foot": r.get("weak_foot"), "skill_moves": r.get("skill_moves"),
        "reputation": r.get("international_reputation"), "contract_end": r.get("contract_end"),
        "release_clause": r.get("release_clause_eur"), "on_loan": r.get("on_loan"),
        "attributes": groups,
        "positions_rated": {c.upper(): r.get(f"pos_{c}") for c in POS_RATING_COLS},
        "history": hist, "forecast": forecast, "similar": similar,
        "edition_name": edition_name(store.latest),
    }))
    return out


def club(store: Store, name: str) -> dict:
    sq = store.cur[store.cur["club"] == name].sort_values("overall", ascending=False)
    if sq.empty:
        raise KeyError(name)
    return clean({"club": name, "league": sq["league"].iloc[0],
                  "wage_bill": float(sq["wage_eur"].fillna(0).sum()),
                  "players": _briefs(sq)})


# build modes: which forecast(s) to plan on and which stay-in-the-game probability to use.
# "balanced" values a signing over the next two seasons: the average of the two
# forecasts and of the two stay probabilities.
MODES = {"now": ((), ()), "next": (("pred_t1",), ("surv_t1",)),
         "balanced": (("pred_t1", "pred_t2"), ("surv_t1", "surv_t2")),
         "future": (("pred_t2",), ("surv_t2",))}
MODE_ALIASES = {"two": "future", "win_now": "now"}
REALISM_GAP = 3.0        # skip players at clubs this much stronger (mean of best 16) than yours
SHUFFLE_NOISE = 1.5      # rating points of random preference per player when shuffling
                         # (a shuffled plan is within 3 * 1.5 / 11 = 0.4 XI points of the best)
RUMOUR_POINTS = 3.0      # extra rating points for a player strongly linked with the club


def _plan_dict(store: "Store", plan, layout: dict) -> dict:
    xi = []
    if plan.status == "optimal":
        for _, x in plan.xi.iterrows():
            xi.append(dict(layout[x["slot"]], player=brief(store.row(x["id"])),
                           rating=float(x["rating_at_slot"]), rating_if_stays=float(x["rating_if_stays"]),
                           stay=float(x["stay"]), signing=x["source"] == "signing"))
    buys = [dict(brief(store.row(i)), fee=float(f)) for i, f in zip(plan.buys["id"], plan.buys["fee"])]
    sales = [dict(brief(store.row(i)), fee=float(f)) for i, f in zip(plan.sales["id"], plan.sales["fee"])]
    return {"status": plan.status, "xi": xi, "buys": buys, "sales": sales,
            "xi_rating": plan.xi_rating, "spend": plan.spend, "income": plan.income,
            "wage_bill": plan.wage_bill, "candidates": plan.n_candidates,
            "solve_seconds": plan.solve_seconds, "notes": plan.notes}


def squad_plan(store: Store, club_name: str, formation: str = "4-3-3", budget_m: float = 50,
               wage_budget_k: float | None = None, max_signings: int = 3, max_age: int | None = None,
               objective: str = "balanced", allow_sales: bool = False, max_sales: int = 3,
               markup: float = 1.2, market_leagues: list[str] | None = None,
               realistic: bool = True, alternatives: int = 2, seed: int | None = None,
               ban_ids: list[int] | None = None, lock_ids: list[int] | None = None,
               rumours: bool = False) -> dict:
    """Best signings for a club, plus up to `alternatives` different plans.

    objective: "now" (today's ratings), "next" (next season's forecast, weighted by the
    chance each player is still in the game), "balanced" (average of the next two
    seasons, the default) or "future" (two seasons out).
    realistic: leave out players at much stronger clubs (they rarely move down).
    seed: shuffle - a small random preference per player, so repeated runs differ.
    rumours: favour players linked with this club in the transfer-rumour feed."""
    objective = MODE_ALIASES.get(objective, objective)
    if objective not in MODES:
        raise ValueError(f"unknown objective {objective}")
    cur = store.cur
    squad = cur[cur["club"] == club_name]
    if squad.empty:
        raise KeyError(club_name)
    if formation not in FORMATIONS:
        raise ValueError(f"unknown formation {formation}")
    lock = [int(i) for i in (lock_ids or [])]
    linked = store.rumours_for(club_name) if rumours else {}
    market = cur[cur["club"] != club_name]
    if market_leagues:
        market = market[market["league"].isin(market_leagues) | market["id"].isin(lock) | market["id"].isin(linked)]
    dropped_realism = 0
    if realistic and club_name in store.club_strength:
        own = store.club_strength[club_name]
        theirs = market["club"].map(store.club_strength).fillna(0)
        too_big = (theirs > own + REALISM_GAP) & ~market["id"].isin(lock) & ~market["id"].isin(linked)
        dropped_realism = int(too_big.sum())
        market = market[~too_big]
    boost_cols, stay_cols = MODES[objective]
    bs = bm = st_s = st_m = None
    if boost_cols:
        bs = (squad[list(boost_cols)].mean(axis=1) - squad["overall"]).fillna(0)
        bm = (market[list(boost_cols)].mean(axis=1) - market["overall"]).fillna(0)
    if stay_cols and all(c in cur for c in stay_cols):
        st_s, st_m = squad[list(stay_cols)].mean(axis=1), market[list(stay_cols)].mean(axis=1)
    bonus = pd.Series(0.0, index=market.index)
    if linked:
        bonus = bonus + market["id"].map({i: RUMOUR_POINTS * r["score"] for i, r in linked.items()}).fillna(0)
    if seed is not None:
        rng = np.random.default_rng(int(seed))
        bonus = bonus + rng.uniform(0, SHUFFLE_NOISE, len(market))
    bill = float(squad["wage_eur"].fillna(0).sum())
    wage_budget = bill * 1.15 if wage_budget_k is None else wage_budget_k * 1e3
    s = TransferSettings(formation=formation, budget=budget_m * 1e6, wage_budget=wage_budget,
                         max_signings=int(max_signings), allow_sales=bool(allow_sales),
                         max_sales=int(max_sales), buy_markup=float(markup),
                         max_age=None if not max_age or max_age >= 40 else int(max_age), time_limit=30)
    layout = {d["slot"]: d for d in formation_layout(formation)}
    kw = dict(boost_squad=bs, boost_market=bm, stay_squad=st_s, stay_market=st_m,
              bonus_market=bonus if (linked or seed is not None) else None,
              ban_ids=ban_ids, lock_ids=lock)
    plans, seen = [], []
    for _ in range(1 + max(0, int(alternatives))):
        plan = optimize_squad(squad, market, s, exclude_buy_sets=seen, **kw)
        if plan.status != "optimal" or (plans and plan.buys.empty):
            break
        plans.append(_plan_dict(store, plan, layout))
        if plan.buys.empty:
            break
        seen.append(plan.buys["id"].astype(int).tolist())
    first = plans[0] if plans else _plan_dict(store, plan, layout)
    for p in plans:
        for b in p["buys"]:
            if b["id"] in linked:
                b["rumour"] = linked[b["id"]]
    return clean({
        "status": first["status"], "club": club_name, "formation": formation, "objective": objective,
        "xi": first["xi"], "buys": first["buys"], "sales": first["sales"], "plans": plans,
        "xi_rating": first["xi_rating"], "baseline_xi_rating": plan.baseline_xi_rating,
        "spend": first["spend"], "income": first["income"], "wage_bill": first["wage_bill"],
        "wage_budget": wage_budget, "current_wage_bill": bill, "budget": budget_m * 1e6,
        "candidates": first["candidates"], "solve_seconds": sum(p["solve_seconds"] for p in plans),
        "realism_dropped": dropped_realism, "rumour_links": len(linked), "notes": first.get("notes", []),
    })


def rumours(store: Store, club: str) -> dict:
    """Players linked with a club in the rumour feed, strongest link first."""
    linked = store.rumours_for(club)
    rows = []
    for pid, link in sorted(linked.items(), key=lambda kv: -kv[1]["score"]):
        rows.append(dict(brief(store.row(pid)), rumour=link))
    meta_ = store.rumours or {}
    return clean({"club": club, "generated": meta_.get("generated"), "window_days": meta_.get("window_days"),
                  "total_links": meta_.get("n_links", 0), "players": rows})


def replacements(store: Store, player_id: int, max_fee_m: float | None = None,
                 max_age: int | None = None, min_overall: int | None = None,
                 same_position: bool = True, leagues: list[str] | None = None, k: int = 15) -> dict:
    target = store.row(player_id)
    if max_fee_m is None:
        max_fee_m = round(0.5 * float(target["value_eur"] or 0) / 1e6, 1)
    if min_overall is None:
        min_overall = int(max(40, target["overall"] - 8))
    res = store.index.find(int(player_id), max_fee=max_fee_m * 1e6, max_age=max_age,
                           min_overall=min_overall, same_position=same_position,
                           leagues=leagues or None,
                           exclude_club=target["club"] if isinstance(target["club"], str) else None, k=k)
    out = [dict(brief(store.row(i)), similarity=float(s), fee=float(f), saving=float(sv))
           for i, s, f, sv in zip(res["id"], res["similarity"], res["fee"], res["saving"])]
    return clean({"target": brief(target), "max_fee": max_fee_m * 1e6, "min_overall": min_overall,
                  "results": out})


def report(store: Store) -> dict:
    """The held-out results shown on the home page."""
    fm = store.reports.get("forecast_metrics", {})
    om = store.reports.get("optimizer_metrics", {})
    sm = store.reports.get("similarity_metrics", {})
    roll = store.reports.get("forecast_rolling", {})
    out = {"forecast": {}, "optimizer": om.get("A_milp_vs_greedy"), "similarity": sm,
           "rolling": {h: [dict(w, test=edition_name(w["test_edition"]), target=edition_name(w["target_edition"]))
                           for w in ws] for h, ws in roll.items()}}
    for h, v in fm.items():
        o = v["overall"]
        yr = v.get("young_ranking", {})
        out["forecast"][h] = {
            "test": f"{edition_name(v['test_edition'])} to {edition_name(v['test_edition'] + int(h[1]))}",
            "n": v["n"], "model": o["model"]["mae"], "ea_potential": o["ea_potential"]["mae"],
            "persistence": o["persistence"]["mae"], "age_curve": o["age_curve"]["mae"],
            "by_age": v.get("by_age"),
            "risers_hits_model": (yr.get("model") or {}).get("precision_top10pct_growth@100"),
            "risers_hits_potential": (yr.get("ea_gap") or {}).get("precision_top10pct_growth@100"),
            "risers_base_rate": yr.get("_base_rate"),
        }
    return clean(out)
