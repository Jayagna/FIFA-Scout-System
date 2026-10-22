"""Load sofifa-derived player files from several FIFA editions into one schema.

Supported inputs (all scraped from sofifa.com):
  * stefanoleone992 Kaggle layout, FIFA 20 dataset   (players_15..20.csv: `club`, `team_position`)
  * stefanoleone992 Kaggle layout, FIFA 21/22 dataset (players_XX.csv:   `club_name`, `league_name`)
  * a cleaned sofifa web export for FIFA 21           (`ID`, `↓OVA`, `POT`, `Value(Euro)`, ...)
  * the combined FC 24 layout (male_players.csv with `fifa_version`), first update per edition

Every edition is mapped to the same columns (see STANDARD_COLUMNS) so the rest
of the code never has to care where a row came from.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

OUTFIELD_ATTRS = [
    "attacking_crossing", "attacking_finishing", "attacking_heading_accuracy",
    "attacking_short_passing", "attacking_volleys", "skill_dribbling", "skill_curve",
    "skill_fk_accuracy", "skill_long_passing", "skill_ball_control",
    "movement_acceleration", "movement_sprint_speed", "movement_agility",
    "movement_reactions", "movement_balance", "power_shot_power", "power_jumping",
    "power_stamina", "power_strength", "power_long_shots", "mentality_aggression",
    "mentality_interceptions", "mentality_positioning", "mentality_vision",
    "mentality_penalties", "mentality_composure", "defending_marking",
    "defending_standing_tackle", "defending_sliding_tackle",
]
GK_ATTRS = [
    "goalkeeping_diving", "goalkeeping_handling", "goalkeeping_kicking",
    "goalkeeping_positioning", "goalkeeping_reflexes",
]
FACE_STATS = ["pace", "shooting", "passing", "dribbling", "defending", "physic"]
POS_RATING_COLS = [
    "ls", "st", "rs", "lw", "lf", "cf", "rf", "rw", "lam", "cam", "ram", "lm", "lcm",
    "cm", "rcm", "rm", "lwb", "ldm", "cdm", "rdm", "rwb", "lb", "lcb", "cb", "rcb", "rb", "gk",
]

STANDARD_COLUMNS = (
    ["id", "edition", "name", "long_name", "age", "dob", "height_cm", "weight_kg",
     "nationality", "club", "league", "overall", "potential", "value_eur", "wage_eur",
     "release_clause_eur", "positions", "primary_pos", "preferred_foot",
     "international_reputation", "weak_foot", "skill_moves", "work_rate_att",
     "work_rate_def", "contract_end", "on_loan"]
    + FACE_STATS + OUTFIELD_ATTRS + GK_ATTRS + [f"pos_{c}" for c in POS_RATING_COLS]
)

# Mapping of the 21 sofifa web export onto the standard names.
_EXPORT_RENAME = {
    "ID": "id", "Name": "name", "LongName": "long_name", "Nationality": "nationality",
    "Age": "age", "↓OVA": "overall", "OVA": "overall", "POT": "potential", "Club": "club",
    "Positions": "positions", "Height(Cm)": "height_cm", "Weight(kg)": "weight_kg",
    "Preferred Foot": "preferred_foot", "Value(Euro)": "value_eur", "Wage(Euro)": "wage_eur",
    "Release Clause(Euro)": "release_clause_eur", "Contract End": "contract_end",
    "Crossing": "attacking_crossing", "Finishing": "attacking_finishing",
    "Heading Accuracy": "attacking_heading_accuracy", "Short Passing": "attacking_short_passing",
    "Volleys": "attacking_volleys", "Dribbling": "skill_dribbling", "Curve": "skill_curve",
    "FK Accuracy": "skill_fk_accuracy", "Long Passing": "skill_long_passing",
    "Ball Control": "skill_ball_control", "Acceleration": "movement_acceleration",
    "Sprint Speed": "movement_sprint_speed", "Agility": "movement_agility",
    "Reactions": "movement_reactions", "Balance": "movement_balance",
    "Shot Power": "power_shot_power", "Jumping": "power_jumping", "Stamina": "power_stamina",
    "Strength": "power_strength", "Long Shots": "power_long_shots",
    "Aggression": "mentality_aggression", "Interceptions": "mentality_interceptions",
    "Positioning": "mentality_positioning", "Vision": "mentality_vision",
    "Penalties": "mentality_penalties", "Composure": "mentality_composure",
    "Marking": "defending_marking", "Standing Tackle": "defending_standing_tackle",
    "Sliding Tackle": "defending_sliding_tackle", "GK Diving": "goalkeeping_diving",
    "GK Handling": "goalkeeping_handling", "GK Kicking": "goalkeeping_kicking",
    "GK Positioning": "goalkeeping_positioning", "GK Reflexes": "goalkeeping_reflexes",
    "PAC": "pace", "SHO": "shooting", "PAS": "passing", "DRI": "dribbling",
    "DEF": "defending", "PHY": "physic", "W/F": "weak_foot", "SM": "skill_moves",
    "IR": "international_reputation", "A/W": "work_rate_att", "D/W": "work_rate_def",
}

_WORK_RATE = {"Low": 1, "Medium": 2, "High": 3}


def _stars(s: pd.Series) -> pd.Series:
    """'4 ★' / '4★' / 4 -> 4.0"""
    return pd.to_numeric(s.astype(str).str.extract(r"(\d+)")[0], errors="coerce")


def _pos_rating(s: pd.Series) -> pd.Series:
    """'89+3' -> 89 (sofifa shows base rating plus a boost; we keep the base)."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    base = s.astype(str).str.extract(r"^\s*(\d+)")[0]
    return pd.to_numeric(base, errors="coerce")


def _primary_pos(positions: pd.Series) -> pd.Series:
    return positions.astype(str).str.split(",").str[0].str.strip().replace({"nan": np.nan})


def _finish(df: pd.DataFrame, edition: int) -> pd.DataFrame:
    """Shared clean-up after renaming, returns STANDARD_COLUMNS only."""
    df = df.copy()
    df["edition"] = edition
    for c in STANDARD_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    df["positions"] = df["positions"].astype(str).str.replace(r"\s+", "", regex=True).str.replace(",", ", ")
    df["primary_pos"] = _primary_pos(df["positions"])
    for c in ["weak_foot", "skill_moves", "international_reputation"]:
        df[c] = _stars(df[c])
    for c in ["work_rate_att", "work_rate_def"]:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(str).str.strip().map(_WORK_RATE)
    num = ["age", "height_cm", "weight_kg", "overall", "potential", "value_eur", "wage_eur",
           "release_clause_eur", "contract_end", "on_loan"]
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # some editions store attributes as '89+2' / '91-1' (value and change since the
    # previous roster update); the first number is the attribute value
    for c in FACE_STATS + OUTFIELD_ATTRS + GK_ATTRS:
        df[c] = _pos_rating(df[c])
    for c in POS_RATING_COLS:
        df[f"pos_{c}"] = _pos_rating(df[f"pos_{c}"])
    df["club"] = df["club"].replace({"": np.nan})
    # FIFA 15 files carry 0 for value/wage (not scraped); 0 is never a real value
    for c in ["value_eur", "wage_eur", "release_clause_eur"]:
        df.loc[df[c] <= 0, c] = np.nan
    # outfield face stats are meaningless for keepers (the sofifa export fills them
    # with GK stats, the Kaggle files leave them empty) -> make them consistently NaN
    is_gk = df["primary_pos"].eq("GK")
    df.loc[is_gk, FACE_STATS] = np.nan
    df = df.dropna(subset=["id", "overall", "potential", "age"])
    df["id"] = df["id"].astype(int)
    # a handful of players appear twice inside one edition file; keep the first row
    df = df.drop_duplicates(subset=["id"], keep="first")
    return df[STANDARD_COLUMNS].reset_index(drop=True)


def load_kaggle_layout(path: Path | str, edition: int) -> pd.DataFrame:
    """players_XX.csv from the stefanoleone992 FIFA 20/21/22 datasets."""
    df = pd.read_csv(path, low_memory=False)
    ren = {"sofifa_id": "id", "short_name": "name", "club_name": "club", "club": "club",
           "league_name": "league", "nationality_name": "nationality",
           "player_positions": "positions", "defending_marking_awareness": "defending_marking",
           "club_contract_valid_until": "contract_end", "contract_valid_until": "contract_end",
           "club_contract_valid_until_year": "contract_end"}
    df = df.rename(columns=ren)
    if "work_rate" in df.columns:
        wr = df["work_rate"].astype(str).str.split("/", expand=True)
        df["work_rate_att"], df["work_rate_def"] = wr[0], wr[1]
    loan_col = "club_loaned_from" if "club_loaned_from" in df.columns else "loaned_from"
    df["on_loan"] = df[loan_col].notna().astype(int) if loan_col in df.columns else 0
    for c in POS_RATING_COLS:
        if c in df.columns:
            df[f"pos_{c}"] = df[c]
    return _finish(df, edition)


def load_sofifa_export(path: Path | str, edition: int) -> pd.DataFrame:
    """Cleaned sofifa web export (ID, ↓OVA, POT, Value(Euro), ...)."""
    df = pd.read_csv(path, low_memory=False).rename(columns=_EXPORT_RENAME)
    if "Contract Status" in df.columns:
        df["on_loan"] = (df["Contract Status"] == "On Loan").astype(int)
        df.loc[df["Contract Status"] == "Free", "club"] = np.nan
    return _finish(df, edition)


def load_fc24_combined(path: Path | str) -> pd.DataFrame:
    """male_players.csv from the EA Sports FC 24 dataset: one row per player per
    roster update. Keeps the first update of every edition (closest to launch)."""
    df = pd.read_csv(path, low_memory=False)
    df = df.sort_values(["fifa_version", "fifa_update"])
    df = df.drop_duplicates(subset=["player_id", "fifa_version"], keep="first")
    df = df.rename(columns={"player_id": "sofifa_id"})
    out = []
    for ed, part in df.groupby("fifa_version"):
        tmp = Path(path).with_suffix(f".{int(ed)}.tmp.csv")
        part.to_csv(tmp, index=False)
        try:
            out.append(load_kaggle_layout(tmp, int(ed)))
        finally:
            tmp.unlink(missing_ok=True)
    return pd.concat(out, ignore_index=True)


def load_raw_dir(raw_dir: Path | str) -> pd.DataFrame:
    """Load every edition found in raw_dir into one long table (player x edition)."""
    raw_dir = Path(raw_dir)
    frames: dict[int, pd.DataFrame] = {}
    combined = raw_dir / "male_players.csv"
    if combined.exists():
        for ed, part in load_fc24_combined(combined).groupby("edition"):
            frames[int(ed)] = part
    for p in sorted(raw_dir.glob("players_*.csv")):
        m = re.fullmatch(r"players_(\d{2})(_sofifa_export)?\.csv", p.name)
        if not m:
            continue
        ed = int(m.group(1))
        if m.group(2):
            if ed in frames:          # a proper Kaggle-layout file wins over the export
                continue
            frames[ed] = load_sofifa_export(p, ed)
        else:
            frames[ed] = load_kaggle_layout(p, ed)
    if not frames:
        raise FileNotFoundError(f"No players_XX.csv files in {raw_dir}. Run scripts/download_data.py")
    df = pd.concat([frames[k] for k in sorted(frames)], ignore_index=True)
    return df


def position_group(primary_pos: pd.Series) -> pd.Series:
    groups = {
        "GK": "GK",
        "CB": "DEF", "LB": "DEF", "RB": "DEF", "LWB": "DEF", "RWB": "DEF",
        "CDM": "MID", "CM": "MID", "CAM": "MID", "LM": "MID", "RM": "MID",
        "LW": "FWD", "RW": "FWD", "ST": "FWD", "CF": "FWD", "LF": "FWD", "RF": "FWD",
    }
    return primary_pos.map(groups)
