"""Turn the long player x edition table into a modelling panel.

One row = one player in edition t, with features known at t and the targets
overall_{t+1}, overall_{t+2} (NaN when the player is not in that edition).
Only columns present in *every* edition are used, so the test edition never
sees a feature the training editions lacked.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import FACE_STATS, GK_ATTRS, OUTFIELD_ATTRS, position_group

CAT_FEATURES = ["pos_group", "primary_pos", "preferred_foot"]

NUM_FEATURES = (
    ["age", "overall", "potential", "gap", "height_cm", "weight_kg", "bmi",
     "log_value", "log_wage", "log_value_rel", "international_reputation", "weak_foot",
     "skill_moves", "work_rate_att", "work_rate_def", "contract_years_left", "on_loan",
     "club_strength", "club_depth_rank", "group_rank_at_club", "ovr_vs_club",
     "has_club", "prev_delta", "prev_pot_delta", "prev2_delta", "editions_seen", "n_positions"]
    + FACE_STATS + OUTFIELD_ATTRS + GK_ATTRS
)

FEATURES = NUM_FEATURES + CAT_FEATURES


def _club_features(df: pd.DataFrame) -> pd.DataFrame:
    """Club strength and the player's place in the pecking order (a playing-time proxy)."""
    df = df.copy()
    key = ["edition", "club"]
    # a few editions list national-team-only players under their nation as "club",
    # and some clubs have only a couple of players in the game: treat both as no club
    size = df.groupby(key)["id"].transform("size")
    has = df["club"].notna() & size.ge(11)
    df["club_depth_rank"] = np.nan
    df["group_rank_at_club"] = np.nan
    df["club_strength"] = np.nan
    sub = df[has]
    rank = sub.groupby(key)["overall"].rank(ascending=False, method="first")
    grank = sub.groupby(key + ["pos_group"])["overall"].rank(ascending=False, method="first")
    top16 = (sub.sort_values("overall", ascending=False)
                .groupby(key).head(16)
                .groupby(key)["overall"].mean().rename("club_strength"))
    df.loc[has, "club_depth_rank"] = rank
    df.loc[has, "group_rank_at_club"] = grank
    df = df.drop(columns="club_strength").merge(top16, on=key, how="left")
    df["ovr_vs_club"] = df["overall"] - df["club_strength"]
    df["has_club"] = has.astype(int).values
    return df


def build_panel(players: pd.DataFrame) -> pd.DataFrame:
    df = players.copy()
    df["pos_group"] = position_group(df["primary_pos"])
    df["gap"] = df["potential"] - df["overall"]
    df["bmi"] = df["weight_kg"] / (df["height_cm"] / 100) ** 2
    df["log_value"] = np.log1p(df["value_eur"])
    df["log_wage"] = np.log1p(df["wage_eur"])
    # value relative to the edition median: removes edition-wide inflation of prices
    df["log_value_rel"] = df["log_value"] - df.groupby("edition")["log_value"].transform("median")
    df["contract_years_left"] = df["contract_end"] - (2000 + df["edition"])
    df["n_positions"] = df["positions"].astype(str).str.count(",") + 1
    df = _club_features(df)

    # history: previous editions of the same player
    df = df.sort_values(["id", "edition"]).reset_index(drop=True)
    g = df.groupby("id")
    prev_ed = g["edition"].shift(1)
    prev_ovr = g["overall"].shift(1)
    prev_pot = g["potential"].shift(1)
    prev2_ed = g["edition"].shift(2)
    prev2_ovr = g["overall"].shift(2)
    consecutive = prev_ed.eq(df["edition"] - 1)
    df["prev_delta"] = np.where(consecutive, df["overall"] - prev_ovr, np.nan)
    df["prev_pot_delta"] = np.where(consecutive, df["potential"] - prev_pot, np.nan)
    df["prev2_delta"] = np.where(consecutive & prev2_ed.eq(df["edition"] - 2),
                                 prev_ovr - prev2_ovr, np.nan)
    df["editions_seen"] = g.cumcount() + 1

    # targets
    for k in (1, 2, 3):
        nxt = df[["id", "edition", "overall", "potential"]].copy()
        nxt["edition"] = nxt["edition"] - k
        nxt = nxt.rename(columns={"overall": f"overall_t{k}", "potential": f"potential_t{k}"})
        df = df.merge(nxt, on=["id", "edition"], how="left")
    for c in CAT_FEATURES:
        df[c] = df[c].astype("category")
    return df


def feature_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    X = panel[FEATURES].copy()
    for c in CAT_FEATURES:
        X[c] = X[c].astype("category")
    return X
