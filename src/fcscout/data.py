"""Load player files from every FIFA / EA FC edition into one schema.

Supported inputs (sofifa.com scrapes unless noted). The layout of each file is
detected from its header, so the file name only has to carry the edition:

  data/raw/players_XX*.csv
    * Kaggle layout (stefanoleone992 / rovnez)  `sofifa_id` or `player_id`, `short_name`,
      `overall`, position columns `st`, `cam`, ...                   FIFA 15-20, 22, FC 26
    * sofifa web export, FIFA 21              `ID`, `↓OVA`, `POT`, `Value(Euro)`, ...
    * sofifa export v2, FIFA 23               `ID`, `Overall`, `ValueEUR`, `STRating`, ...
    * sofifa scrape (Playwright scraper), FC 24  `player_id`, `overall_rating`, `value` ("185M")
  data/raw/ea_db_XX/                          the game's own database tables (players.csv,
                                              teams.csv, leagues.csv, ...), FC 25
  data/raw/male_players.csv                   the combined FC 24 Kaggle file (all editions)

Every edition is mapped to the same columns (STANDARD_COLUMNS). After loading,
`load_raw_dir` fills the gaps some sources have, and flags what it filled:
  * face stats / position ratings missing  -> computed from the detailed attributes with
    EA's linear formulas (fitted on FC 26, checked on FIFA 22/23; see DERIVED_WEIGHTS)
  * values / wages partly missing           -> estimated (`money_estimated` = 1)
  * no league column                        -> taken from the same club in the next or
                                               previous edition
  * no player name                          -> taken from the same player's nearest edition
"""
from __future__ import annotations

import datetime as dt
import io
import re
import unicodedata
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
     "nationality", "club", "league", "league_id", "overall", "potential", "value_eur",
     "wage_eur", "release_clause_eur", "positions", "primary_pos", "preferred_foot",
     "international_reputation", "weak_foot", "skill_moves", "work_rate_att",
     "work_rate_def", "contract_end", "on_loan", "face_url", "ratings_derived",
     "money_estimated"]
    + FACE_STATS + OUTFIELD_ATTRS + GK_ATTRS + [f"pos_{c}" for c in POS_RATING_COLS]
)

# Leagues by EA/sofifa league id. Newer sofifa files use short names that clash across
# countries ("Super League" is Swiss, Greek, Chinese and Indian), so the id decides.
LEAGUE_NAMES = {
    1: "Superliga (Denmark)", 4: "Pro League (Belgium)", 7: "Série A (Brazil)",
    10: "Eredivisie (Netherlands)", 13: "Premier League (England)",
    14: "Championship (England)", 16: "Ligue 1 (France)", 17: "Ligue 2 (France)",
    19: "Bundesliga (Germany)", 20: "2. Bundesliga (Germany)", 31: "Serie A (Italy)",
    32: "Serie B (Italy)", 39: "MLS (USA)", 41: "Eliteserien (Norway)",
    50: "Premiership (Scotland)", 53: "La Liga (Spain)", 54: "La Liga 2 (Spain)",
    56: "Allsvenskan (Sweden)", 60: "League One (England)", 61: "League Two (England)",
    63: "Super League (Greece)", 64: "NB I (Hungary)", 65: "Premier Division (Ireland)",
    66: "Ekstraklasa (Poland)", 68: "Süper Lig (Türkiye)",
    80: "Bundesliga (Austria)", 83: "K League 1 (Korea)", 189: "Super League (Switzerland)",
    308: "Primeira Liga (Portugal)", 313: "Premyer Liqa (Azerbaijan)",
    317: "HNL (Croatia)", 318: "1. Division (Cyprus)", 319: "První liga (Czechia)",
    322: "Veikkausliiga (Finland)", 330: "Liga I (Romania)", 332: "Premier League (Ukraine)",
    335: "Primera División (Chile)", 336: "Primera A (Colombia)",
    337: "División Profesional (Paraguay)", 338: "Primera División (Uruguay)",
    350: "Pro League (Saudi Arabia)", 351: "A-League (Australia)",
    353: "Liga Profesional (Argentina)", 2012: "Super League (China)",
    2013: "Pro League (UAE)", 2017: "División Profesional (Bolivia)",
    2018: "Serie A (Ecuador)", 2019: "Primera División (Venezuela)", 2020: "Liga 1 (Peru)",
    2076: "3. Liga (Germany)", 2149: "Super League (India)",
}
# the Kaggle FIFA 15-22 files have country-qualified names without ids
_OLD_LEAGUE_IDS = {
    "English Premier League": 13, "Spain Primera Division": 53, "German 1. Bundesliga": 19,
    "Italian Serie A": 31, "French Ligue 1": 16, "English League Championship": 14,
    "Holland Eredivisie": 10, "Portuguese Liga ZON SAGRES": 308, "Liga NOS": 308,
    "USA Major League Soccer": 39, "Turkish Süper Lig": 68, "Belgian Jupiler Pro League": 4,
    "Scottish Premiership": 50, "Saudi Abdul L. Jameel League": 350,
}
PREMIER_LEAGUE = LEAGUE_NAMES[13]

_EA_POSITIONS = {
    0: "GK", 1: "CB", 2: "RWB", 3: "RB", 4: "CB", 5: "CB", 6: "CB", 7: "LB", 8: "LWB",
    9: "CDM", 10: "CDM", 11: "CDM", 12: "RM", 13: "CM", 14: "CM", 15: "CM", 16: "LM",
    17: "CAM", 18: "CAM", 19: "CAM", 20: "RF", 21: "CF", 22: "LF", 23: "RW", 24: "ST",
    25: "ST", 26: "ST", 27: "LW",
}

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

# sofifa export v2 (FIFA 23): CamelCase names, one rating per position family
_V2_RENAME = {
    "ID": "id", "Name": "name", "FullName": "long_name", "Age": "age", "Height": "height_cm",
    "Weight": "weight_kg", "PhotoUrl": "face_url", "Nationality": "nationality",
    "Overall": "overall", "Potential": "potential", "Positions": "positions", "Club": "club",
    "ValueEUR": "value_eur", "WageEUR": "wage_eur", "ReleaseClause": "release_clause_eur",
    "ContractUntil": "contract_end", "OnLoad": "on_loan", "PreferredFoot": "preferred_foot",
    "IntReputation": "international_reputation", "WeakFoot": "weak_foot",
    "SkillMoves": "skill_moves", "AttackingWorkRate": "work_rate_att",
    "DefensiveWorkRate": "work_rate_def", "PaceTotal": "pace", "ShootingTotal": "shooting",
    "PassingTotal": "passing", "DribblingTotal": "dribbling", "DefendingTotal": "defending",
    "PhysicalityTotal": "physic", "Crossing": "attacking_crossing",
    "Finishing": "attacking_finishing", "HeadingAccuracy": "attacking_heading_accuracy",
    "ShortPassing": "attacking_short_passing", "Volleys": "attacking_volleys",
    "Dribbling": "skill_dribbling", "Curve": "skill_curve", "FKAccuracy": "skill_fk_accuracy",
    "LongPassing": "skill_long_passing", "BallControl": "skill_ball_control",
    "Acceleration": "movement_acceleration", "SprintSpeed": "movement_sprint_speed",
    "Agility": "movement_agility", "Reactions": "movement_reactions",
    "Balance": "movement_balance", "ShotPower": "power_shot_power", "Jumping": "power_jumping",
    "Stamina": "power_stamina", "Strength": "power_strength", "LongShots": "power_long_shots",
    "Aggression": "mentality_aggression", "Interceptions": "mentality_interceptions",
    "Positioning": "mentality_positioning", "Vision": "mentality_vision",
    "Penalties": "mentality_penalties", "Composure": "mentality_composure",
    "Marking": "defending_marking", "StandingTackle": "defending_standing_tackle",
    "SlidingTackle": "defending_sliding_tackle", "GKDiving": "goalkeeping_diving",
    "GKHandling": "goalkeeping_handling", "GKKicking": "goalkeeping_kicking",
    "GKPositioning": "goalkeeping_positioning", "GKReflexes": "goalkeeping_reflexes",
}
_V2_POSITION_FAMILIES = {
    "ST": ["ls", "st", "rs"], "LW": ["lw"], "LF": ["lf"], "CF": ["cf"], "RF": ["rf"],
    "RW": ["rw"], "CAM": ["lam", "cam", "ram"], "LM": ["lm"], "CM": ["lcm", "cm", "rcm"],
    "RM": ["rm"], "LWB": ["lwb"], "CDM": ["ldm", "cdm", "rdm"], "RWB": ["rwb"], "LB": ["lb"],
    "CB": ["lcb", "cb", "rcb"], "RB": ["rb"], "GK": ["gk"],
}

# Playwright sofifa scraper (FC 24 file; newer runs prefix the attribute names)
_SCRAPE_ATTRS = {
    "crossing": "attacking_crossing", "finishing": "attacking_finishing",
    "heading_accuracy": "attacking_heading_accuracy", "short_passing": "attacking_short_passing",
    "volleys": "attacking_volleys", "dribbling": "skill_dribbling", "curve": "skill_curve",
    "fk_accuracy": "skill_fk_accuracy", "long_passing": "skill_long_passing",
    "ball_control": "skill_ball_control", "acceleration": "movement_acceleration",
    "sprint_speed": "movement_sprint_speed", "agility": "movement_agility",
    "reactions": "movement_reactions", "balance": "movement_balance",
    "shot_power": "power_shot_power", "jumping": "power_jumping", "stamina": "power_stamina",
    "strength": "power_strength", "long_shots": "power_long_shots",
    "aggression": "mentality_aggression", "interceptions": "mentality_interceptions",
    "positioning": "mentality_positioning", "att_positioning": "mentality_positioning",
    "vision": "mentality_vision", "penalties": "mentality_penalties",
    "composure": "mentality_composure", "defensive_awareness": "defending_marking",
    "standing_tackle": "defending_standing_tackle", "sliding_tackle": "defending_sliding_tackle",
    "gk_diving": "goalkeeping_diving", "gk_handling": "goalkeeping_handling",
    "gk_kicking": "goalkeeping_kicking", "gk_positioning": "goalkeeping_positioning",
    "gk_reflexes": "goalkeeping_reflexes",
}
_SCRAPE_RENAME = {
    "player_id": "id", "overall_rating": "overall", "value": "value_eur", "wage": "wage_eur",
    "release_clause": "release_clause_eur", "club_name": "club", "club_league_id": "league_id",
    "club_league_name": "league", "club_contract_valid_until": "contract_end",
    "country_name": "nationality", "full_name": "long_name", "image": "face_url",
}

# EA game database (FC 25): column names in players.csv
_EA_ATTRS = {
    "crossing": "attacking_crossing", "finishing": "attacking_finishing",
    "headingaccuracy": "attacking_heading_accuracy", "shortpassing": "attacking_short_passing",
    "volleys": "attacking_volleys", "dribbling": "skill_dribbling", "curve": "skill_curve",
    "freekickaccuracy": "skill_fk_accuracy", "longpassing": "skill_long_passing",
    "ballcontrol": "skill_ball_control", "acceleration": "movement_acceleration",
    "sprintspeed": "movement_sprint_speed", "agility": "movement_agility",
    "reactions": "movement_reactions", "balance": "movement_balance",
    "shotpower": "power_shot_power", "jumping": "power_jumping", "stamina": "power_stamina",
    "strength": "power_strength", "longshots": "power_long_shots",
    "aggression": "mentality_aggression", "interceptions": "mentality_interceptions",
    "positioning": "mentality_positioning", "vision": "mentality_vision",
    "penalties": "mentality_penalties", "composure": "mentality_composure",
    "defensiveawareness": "defending_marking", "standingtackle": "defending_standing_tackle",
    "slidingtackle": "defending_sliding_tackle", "gkdiving": "goalkeeping_diving",
    "gkhandling": "goalkeeping_handling", "gkkicking": "goalkeeping_kicking",
    "gkpositioning": "goalkeeping_positioning", "gkreflexes": "goalkeeping_reflexes",
}
# leagues in the game database that are not men's club leagues ("Men's ROW", 76, holds
# the legend squads such as Classic XI and Soccer Aid)
_EA_NON_CLUB_LEAGUES = {76, 78, 111, 382, 383, 384, 1003, 1014, 2000, 2028, 2092, 2240}

_WORK_RATE = {"Low": 1, "Medium": 2, "High": 3}


# ------------------------------------------------------------------ small parsers
def _stars(s: pd.Series) -> pd.Series:
    """'4 ★' / '4★' / 4 -> 4.0"""
    return pd.to_numeric(s.astype(str).str.extract(r"(\d+)")[0], errors="coerce")


def _pos_rating(s: pd.Series) -> pd.Series:
    """'89+3' -> 89 (sofifa shows base rating plus a boost; we keep the base)."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    base = s.astype(str).str.extract(r"^\s*(\d+)")[0]
    return pd.to_numeric(base, errors="coerce")


def _money(s: pd.Series) -> pd.Series:
    """'€115.5M' / '340K' / '€0' / 1.2e6 -> euros as float."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    m = s.astype(str).str.replace(",", "", regex=False).str.extract(r"([\d.]+)\s*([KMB]?)", flags=re.I)
    num = pd.to_numeric(m[0], errors="coerce")
    mult = m[1].str.upper().map({"K": 1e3, "M": 1e6, "B": 1e9, "": 1.0}).fillna(1.0)
    return num * mult


def _primary_pos(positions: pd.Series) -> pd.Series:
    return positions.astype(str).str.split(",").str[0].str.strip().replace({"nan": np.nan})


def season_start(edition: int) -> pd.Timestamp:
    """Reference date for ages. The Kaggle files compute every age on 1 July of the
    season's first year (checked: 100% of FIFA 20, FIFA 22 and FC 26 rows agree)."""
    return pd.Timestamp(2000 + int(edition) - 1, 7, 1)


def _age_at(dob: pd.Series, when: pd.Timestamp) -> pd.Series:
    d = pd.to_datetime(dob, errors="coerce")
    years = when.year - d.dt.year
    before_birthday = (d.dt.month > when.month) | ((d.dt.month == when.month) & (d.dt.day > when.day))
    return (years - before_birthday.astype(int)).astype(float)


def _read_csv_mixed(path: Path | str) -> pd.DataFrame:
    """Read a CSV whose lines mix UTF-8 and Windows-1252 text (the FC 24 scrape does)."""
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        lines = []
        for line in raw.split(b"\n"):
            try:
                lines.append(line.decode("utf-8"))
            except UnicodeDecodeError:
                lines.append(line.decode("cp1252", errors="replace"))
        text = "\n".join(lines)
    return pd.read_csv(io.StringIO(text), low_memory=False)


def _norm_club(s: pd.Series) -> pd.Series:
    """Club name key that survives accents, punctuation and 'FC'/'CF' suffixes."""
    def f(x):
        if not isinstance(x, str):
            return np.nan
        x = unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode().lower()
        x = re.sub(r"[^a-z0-9 ]+", " ", x)
        x = re.sub(r"\b(fc|cf|sc|ac|afc|cd|sv|club|de|the)\b", " ", x)
        return re.sub(r"\s+", " ", x).strip() or np.nan
    return s.map(f)


# ------------------------------------------------------------------ shared clean-up
def _finish(df: pd.DataFrame, edition: int) -> pd.DataFrame:
    """Shared clean-up after renaming, returns STANDARD_COLUMNS only."""
    df = df.copy()
    df["edition"] = edition
    for c in STANDARD_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    df["positions"] = df["positions"].astype(str).str.replace(r"\s+", "", regex=True).str.replace(",", ", ")
    df.loc[df["positions"].isin(["nan", "None", ""]), "positions"] = np.nan
    df["primary_pos"] = _primary_pos(df["positions"])
    for c in ["weak_foot", "skill_moves", "international_reputation"]:
        df[c] = _stars(df[c])
    for c in ["work_rate_att", "work_rate_def"]:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(str).str.strip().map(_WORK_RATE)
    for c in ["value_eur", "wage_eur", "release_clause_eur"]:
        df[c] = _money(df[c])
    num = ["age", "height_cm", "weight_kg", "overall", "potential", "contract_end", "on_loan",
           "league_id", "ratings_derived", "money_estimated"]
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ratings_derived"] = df["ratings_derived"].fillna(0).astype(int)
    df["money_estimated"] = df["money_estimated"].fillna(0).astype(int)
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


# ------------------------------------------------------------------ one loader per layout
def load_kaggle_layout(path: Path | str, edition: int) -> pd.DataFrame:
    """players_XX.csv from the stefanoleone992 FIFA 20/21/22 datasets and the FC 26
    file in the same layout (rovnez)."""
    df = pd.read_csv(path, low_memory=False) if not isinstance(path, pd.DataFrame) else path.copy()
    ren = {"sofifa_id": "id", "player_id": "id", "short_name": "name", "club_name": "club",
           "club": "club", "league_name": "league", "nationality_name": "nationality",
           "player_positions": "positions", "defending_marking_awareness": "defending_marking",
           "club_contract_valid_until": "contract_end", "contract_valid_until": "contract_end",
           "club_contract_valid_until_year": "contract_end", "player_face_url": "face_url"}
    df = df.rename(columns=ren)
    if "work_rate" in df.columns:
        wr = df["work_rate"].astype("string").str.split("/", n=1, expand=True)
        df["work_rate_att"] = wr[0] if 0 in wr.columns else np.nan
        df["work_rate_def"] = wr[1] if 1 in wr.columns else np.nan
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


def load_sofifa_v2(path: Path | str, edition: int, position_ratings: bool = False) -> pd.DataFrame:
    """sofifa export with CamelCase headers and one rating per position family (FIFA 23).

    Its position ratings include sofifa's boost (they often exceed the player's overall,
    which never happens in the other editions), so by default they are dropped and
    recomputed from the attributes like FC 24/25. The card stats are kept: they match
    EA's formula exactly."""
    df = pd.read_csv(path, low_memory=False)
    if position_ratings:
        pos = {f"pos_{c}": df[f"{fam}Rating"] for fam, cols in _V2_POSITION_FAMILIES.items()
               if f"{fam}Rating" in df.columns for c in cols}
        df = pd.concat([df, pd.DataFrame(pos)], axis=1)
    df = df.rename(columns=_V2_RENAME)
    if "on_loan" in df.columns:
        df["on_loan"] = df["on_loan"].astype(str).str.lower().isin(["true", "1", "yes"]).astype(int)
    if "face_url" in df.columns:          # the 60 px variant -> 120 px
        df["face_url"] = df["face_url"].astype(str).str.replace("_60.png", "_120.png", regex=False)
    return _finish(df, edition)


def load_sofifa_scrape(path: Path | str, edition: int) -> pd.DataFrame:
    """Output of the Playwright sofifa scraper (FC 24): no ages, money as '185M'."""
    df = _read_csv_mixed(path)
    ren = dict(_SCRAPE_RENAME)
    for k, v in _SCRAPE_ATTRS.items():
        for prefix in ("", "attacking_", "skill_", "movement_", "power_", "mentality_",
                       "defending_", "goalkeeping_"):
            ren[prefix + k] = v
    df = df.rename(columns=ren)
    df = df.loc[:, ~df.columns.duplicated()]
    df["long_name"] = df.get("long_name", df["name"])
    if "work_rate" in df.columns:
        wr = df["work_rate"].astype("string").str.split("/", n=1, expand=True)
        df["work_rate_att"] = wr[0] if 0 in wr.columns else np.nan
        df["work_rate_def"] = wr[1] if 1 in wr.columns else np.nan
    df["dob"] = pd.to_datetime(df["dob"], errors="coerce", format="mixed")
    df["age"] = _age_at(df["dob"], season_start(edition))
    df["dob"] = df["dob"].dt.strftime("%Y-%m-%d")
    df["face_url"] = np.nan        # rebuilt from the id for the right edition below
    return _finish(df, edition)


def load_ea_database(folder: Path | str, edition: int) -> pd.DataFrame:
    """Tables exported from the game's own database (players, teams, leagues and the
    links between them). Values and wages are not in the game database; they come from
    the sofifa.csv sheet next to it where available and are estimated otherwise."""
    folder = Path(folder)
    P = pd.read_csv(folder / "players.csv", low_memory=False)
    teams = pd.read_csv(folder / "teams.csv", low_memory=False, usecols=["teamid", "teamname"])
    leagues = pd.read_csv(folder / "leagues.csv")
    leagues = leagues[leagues["leaguename"].notna()]
    club_leagues = leagues[(leagues.get("iswomencompetition", 0) == 0)
                           & (leagues.get("isinternationalleague", 0) == 0)
                           & ~leagues["leagueid"].isin(_EA_NON_CLUB_LEAGUES)]
    ltl = pd.read_csv(folder / "leagueteamlinks.csv", usecols=["teamid", "leagueid"])
    tpl = pd.read_csv(folder / "teamplayerlinks.csv", usecols=["playerid", "teamid"])
    links = (tpl[tpl["playerid"] > 0].merge(ltl, on="teamid")
             .merge(club_leagues[["leagueid", "leaguename"]], on="leagueid")
             .drop_duplicates("playerid"))
    P = P[(P.get("gender", 0) == 0) & (P["playerid"] > 0)].merge(links, on="playerid")
    P = P.merge(teams, on="teamid", how="left")

    df = P.rename(columns={**_EA_ATTRS, "playerid": "id", "overallrating": "overall",
                           "height": "height_cm", "weight": "weight_kg",
                           "contractvaliduntil": "contract_end", "teamname": "club",
                           "leagueid": "league_id", "leaguename": "league",
                           "internationalrep": "international_reputation",
                           "weakfootabilitytypecode": "weak_foot"})
    df["skill_moves"] = P["skillmoves"] + 1                 # stored as 0-4
    df["preferred_foot"] = P["preferredfoot"].map({1: "Right", 2: "Left"})
    dob = pd.Timestamp(1582, 10, 14) + pd.to_timedelta(P["birthdate"], unit="D")
    df["dob"] = dob.dt.strftime("%Y-%m-%d")
    df["age"] = _age_at(dob, season_start(edition))
    pos = [P[f"preferredposition{i}"].map(_EA_POSITIONS) for i in range(1, 5)
           if f"preferredposition{i}" in P.columns]
    df["positions"] = pd.concat(pos, axis=1).apply(
        lambda r: ", ".join(dict.fromkeys(x for x in r if isinstance(x, str))), axis=1)
    if (folder / "countries.csv").exists():
        countries = pd.read_csv(folder / "countries.csv").set_index("countryid")["countryname"]
        df["nationality"] = P["nationality"].map(countries)
    names = {}
    if (folder / "models.csv").exists():
        names.update(pd.read_csv(folder / "models.csv").set_index("playerid")["playername"].to_dict())
    if (folder / "sofifa.csv").exists():
        sf = pd.read_csv(folder / "sofifa.csv", low_memory=False).drop_duplicates("playerid")
        sf = sf.set_index("playerid")
        names.update(sf["full_name"].dropna().str.strip(" -").to_dict())
        df["value_eur"] = df["id"].map(_money(sf["value"]))
        df["wage_eur"] = df["id"].map(_money(sf["wage"]))
        df["release_clause_eur"] = df["id"].map(_money(sf["release_clause"]))
    df["name"] = df["id"].map(names)
    df["long_name"] = df["name"]
    df["on_loan"] = np.nan                                   # loans are not in these tables
    return _finish(df, edition)


def load_fc24_combined(path: Path | str) -> pd.DataFrame:
    """male_players.csv from the EA Sports FC 24 dataset: one row per player per
    roster update. Keeps the first update of every edition (closest to launch)."""
    df = pd.read_csv(path, low_memory=False)
    df = df.sort_values(["fifa_version", "fifa_update"])
    df = df.drop_duplicates(subset=["player_id", "fifa_version"], keep="first")
    out = [load_kaggle_layout(part, int(ed)) for ed, part in df.groupby("fifa_version")]
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------------ layout detection
def detect_layout(path: Path | str) -> str:
    """Return the loader name for a players CSV, from its header."""
    if Path(path).is_dir():
        return "ea_database"
    head = Path(path).open("rb").readline().decode("utf-8", errors="replace")
    cols = {c.strip().strip('"') for c in head.strip().split(",")}
    if "fifa_version" in cols and "player_id" in cols and "short_name" in cols:
        return "kaggle"
    if "sofifa_id" in cols or "short_name" in cols:
        return "kaggle"
    if "STRating" in cols or "ValueEUR" in cols:
        return "sofifa_v2"
    if "↓OVA" in cols or "OVA" in cols or "Value(Euro)" in cols:
        return "sofifa_export"
    if "overall_rating" in cols and ("player_id" in cols or "playerid" in cols):
        return "sofifa_scrape"
    raise ValueError(f"unrecognised player file layout: {path}")


_LOADERS = {"kaggle": load_kaggle_layout, "sofifa_v2": load_sofifa_v2,
            "sofifa_scrape": load_sofifa_scrape, "ea_database": load_ea_database,
            "sofifa_export": load_sofifa_export}
# when two files cover the same edition, the most complete layout wins
_PRIORITY = {"kaggle": 0, "sofifa_v2": 1, "sofifa_scrape": 2, "ea_database": 3, "sofifa_export": 4}


def load_raw_dir(raw_dir: Path | str, fill: bool = True) -> pd.DataFrame:
    """Load every edition found in raw_dir into one long table (player x edition)."""
    raw_dir = Path(raw_dir)
    found: dict[int, tuple[int, pd.DataFrame]] = {}
    combined = raw_dir / "male_players.csv"
    if combined.exists():
        for ed, part in load_fc24_combined(combined).groupby("edition"):
            found[int(ed)] = (0, part)
    candidates = [p for p in raw_dir.glob("players_*.csv")] + [p for p in raw_dir.glob("ea_db_*") if p.is_dir()]
    for p in sorted(candidates):
        m = re.fullmatch(r"(?:players|ea_db)_(\d{2})(?:_[\w-]+)?(?:\.csv)?", p.name)
        if not m:
            continue
        ed = int(m.group(1))
        layout = detect_layout(p)
        rank = _PRIORITY[layout]
        if ed in found and found[ed][0] <= rank:
            continue
        found[ed] = (rank, _LOADERS[layout](p, ed))
    if not found:
        raise FileNotFoundError(f"No players_XX.csv files in {raw_dir}. Run scripts/download_data.py")
    df = pd.concat([found[k][1] for k in sorted(found)], ignore_index=True)
    return fill_gaps(df) if fill else df


# ------------------------------------------------------------------ gap filling
# EA computes the six card stats and every position rating as a weighted sum of the
# detailed attributes. The weights in rating_formulas.json were recovered by
# non-negative least squares on FC 26 and checked on FIFA 22 and FIFA 23, which
# publish both (scripts/fit_rating_formulas.py regenerates the file and reports the
# accuracy).
def _load_formulas() -> dict[str, dict[str, float]]:
    import json
    p = Path(__file__).with_name("rating_formulas.json")
    return json.loads(p.read_text(encoding="utf-8"))["weights"] if p.exists() else {}


DERIVED_WEIGHTS = _load_formulas()


def derive_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing face stats and position ratings from the detailed attributes."""
    if not DERIVED_WEIGHTS:
        return df
    df = df.copy()
    pos_cols = [f"pos_{c}" for c in POS_RATING_COLS]
    # only editions whose source lacks the ratings altogether; editions that publish
    # them keep their own values (including their blanks, e.g. keepers in FIFA 15-20)
    by_ed = df.groupby("edition")
    lacks_pos = df["edition"].map(by_ed[pos_cols].apply(lambda g: g.notna().any(axis=1).mean() < 0.5))
    lacks_face = df["edition"].map(by_ed[FACE_STATS].apply(lambda g: g.notna().any(axis=1).mean() < 0.5))
    has_attrs = df[OUTFIELD_ATTRS].notna().all(axis=1)
    need_pos = lacks_pos & has_attrs
    need_face = lacks_face & has_attrs & df["primary_pos"].ne("GK")
    for target, w in DERIVED_WEIGHTS.items():
        X = sum(df[a].fillna(0) * c for a, c in w.items())
        if target in FACE_STATS:
            df.loc[need_face, target] = np.floor(X[need_face] + 0.5)
        else:
            val = np.floor(X + 0.5)
            for c in ([target] if target == "gk" else _V2_SAME_RATING.get(target, [target])):
                df.loc[need_pos, f"pos_{c}"] = val[need_pos]
    df.loc[need_pos | need_face, "ratings_derived"] = 1
    return df


# positions that share one rating (left/centre/right variants of the same role)
_V2_SAME_RATING = {fam.lower(): cols for fam, cols in _V2_POSITION_FAMILIES.items()}


def fill_names(df: pd.DataFrame) -> pd.DataFrame:
    """Players without a name (game database rows) take it from their nearest edition."""
    df = df.copy()
    for col in ("name", "long_name"):
        known = df.dropna(subset=[col])[["id", "edition", col]]
        miss = df[col].isna()
        if not miss.any() or known.empty:
            continue
        q = df.loc[miss, ["id", "edition"]].reset_index()
        cand = q.merge(known, on="id", suffixes=("", "_k"))
        cand["dist"] = (cand["edition_k"] - cand["edition"]).abs() * 2 + (cand["edition_k"] < cand["edition"])
        best = cand.sort_values("dist").drop_duplicates("index").set_index("index")[col]
        df.loc[best.index, col] = best
    df["name"] = df["name"].fillna(df["long_name"])
    return df


def fill_leagues(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical league names (by id where known); editions without a league column
    take the league of the same club in the next, else the previous, edition."""
    df = df.copy()
    old = df["league_id"].isna() & df["league"].isin(_OLD_LEAGUE_IDS)
    df.loc[old, "league_id"] = df.loc[old, "league"].map(_OLD_LEAGUE_IDS)
    named = df["league_id"].isin(LEAGUE_NAMES)
    df.loc[named, "league"] = df.loc[named, "league_id"].astype(int).map(LEAGUE_NAMES)
    key = _norm_club(df["club"])
    original = df["league"].notna()          # copy only from editions that ship leagues
    for ed in sorted(df["edition"].unique()):
        rows = (df["edition"] == ed) & df["league"].isna() & df["club"].notna()
        if not rows.any() or original[df["edition"] == ed].mean() > 0.5:
            continue
        for other in (ed + 1, ed - 1):
            src = (df["edition"] == other) & original
            if not src.any():
                continue
            lut = (pd.DataFrame({"k": key[src], "league": df.loc[src, "league"],
                                 "league_id": df.loc[src, "league_id"]})
                   .dropna(subset=["k"]).drop_duplicates("k").set_index("k"))
            todo = rows & df["league"].isna()
            hit = key[todo].map(lut["league"])
            df.loc[hit.dropna().index, "league"] = hit.dropna()
            df.loc[hit.dropna().index, "league_id"] = key[hit.dropna().index].map(lut["league_id"])
    return df


def fill_face_urls(df: pd.DataFrame) -> pd.DataFrame:
    """sofifa serves every player's face at /players/<3 digits>/<3 digits>/<edition>_120.png."""
    df = df.copy()
    miss = df["face_url"].isna() | ~df["face_url"].astype(str).str.startswith("http")
    ids = df.loc[miss, "id"].astype(int)
    eds = df.loc[miss, "edition"].astype(int)
    df.loc[miss, "face_url"] = [f"https://cdn.sofifa.net/players/{i // 1000:03d}/{i % 1000:03d}/{e}_120.png"
                                for i, e in zip(ids, eds)]
    return df


def impute_money(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate missing values and wages in editions that have them only for some
    players (the FC 25 game database). A gradient-boosting model of log(value) on
    rating, potential, age, position and reputation is trained on the full editions
    around it, then shifted so its median error on the players whose value *is*
    known in that edition is zero. Wages also use the club's strength and league."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    df = df.copy()
    share = df.groupby("edition")["value_eur"].apply(lambda s: s.notna().mean())
    partial = [int(e) for e, s in share.items() if 0.05 < s < 0.9]
    complete = [int(e) for e, s in share.items() if s >= 0.9]
    if not partial or not complete:
        return df
    ranked = df.sort_values("overall", ascending=False)
    top16 = (ranked[ranked.groupby(["edition", "club"]).cumcount() < 16]
             .groupby(["edition", "club"])["overall"].mean().rename("club_strength").reset_index())
    strength = pd.Series(df[["edition", "club"]].merge(top16, on=["edition", "club"], how="left")
                         ["club_strength"].to_numpy(), index=df.index)
    X = pd.DataFrame({
        "overall": df["overall"], "potential": df["potential"], "age": df["age"],
        "gap": df["potential"] - df["overall"], "ir": df["international_reputation"],
        "years_left": df["contract_end"] - (2000 + df["edition"]),
        "gk": df["primary_pos"].eq("GK").astype(int),
        "group": position_group(df["primary_pos"]).map({"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}),
        "club_strength": strength, "league_id": df["league_id"],
    })
    for target, cols in (("value_eur", ["overall", "potential", "age", "gap", "ir", "years_left", "gk", "group"]),
                         ("wage_eur", ["overall", "potential", "age", "ir", "group", "club_strength", "league_id"])):
        for ed in partial:
            near = sorted(complete, key=lambda e: abs(e - ed))[:2]
            tr = df["edition"].isin(near) & df[target].notna()
            model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08,
                                                  categorical_features=[cols.index("league_id")] if "league_id" in cols else None,
                                                  random_state=0)
            Xtr = X.loc[tr, cols].copy()
            if "league_id" in cols:
                Xtr["league_id"] = Xtr["league_id"].fillna(-1).astype(int).clip(lower=0) % 10000
            model.fit(Xtr, np.log(df.loc[tr, target]))
            here = df["edition"] == ed
            Xh = X.loc[here, cols].copy()
            if "league_id" in cols:
                Xh["league_id"] = Xh["league_id"].fillna(-1).astype(int).clip(lower=0) % 10000
            pred = pd.Series(model.predict(Xh), index=Xh.index)
            known = here & df[target].notna()
            shift = float(np.median(np.log(df.loc[known, target]) - pred[known])) if known.any() else 0.0
            miss = here & df[target].isna()
            est = np.exp(pred[miss] + shift)
            # EA rounds values to steps that grow with the amount; mimic it loosely
            step = np.where(est >= 1e6, 1e5, np.where(est >= 1e5, 5e3, 1e3)) if target == "value_eur" \
                else np.where(est >= 1e4, 1e3, 100.0)
            df.loc[miss, target] = np.round(est / step) * step
            df.loc[miss, "money_estimated"] = 1
    return df


def fill_gaps(df: pd.DataFrame) -> pd.DataFrame:
    df = fill_names(df)
    df = fill_leagues(df)
    df = derive_ratings(df)
    df = impute_money(df)
    df = fill_face_urls(df)
    return df


def position_group(primary_pos: pd.Series) -> pd.Series:
    groups = {
        "GK": "GK",
        "CB": "DEF", "LB": "DEF", "RB": "DEF", "LWB": "DEF", "RWB": "DEF",
        "CDM": "MID", "CM": "MID", "CAM": "MID", "LM": "MID", "RM": "MID",
        "LW": "FWD", "RW": "FWD", "ST": "FWD", "CF": "FWD", "LF": "FWD", "RF": "FWD",
    }
    return primary_pos.map(groups)
