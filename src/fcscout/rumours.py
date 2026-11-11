"""Turn transfer-news headlines into (player, club) links the squad builder can use.

Only headlines are used (never article text): each one is checked for a player in
the current database, one or more clubs, and transfer wording. The club the player
already plays for is the seller; any other club named is a suitor, unless the
headline says the player is coming *from* it. Each link is scored by how reliable
the outlet is, how far along the deal sounds, and how recent the headline is.

Matching favours precision over recall: full names ("Lewis Hall") always count,
a surname on its own ("Hall") only for well-known players whose surname no other
player in the database shares.
"""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

# outlet reliability (by domain); anything else counts 0.3
SOURCE_WEIGHT = {
    "theathletic.com": 1.0, "nytimes.com": 1.0, "bbc.co.uk": 1.0, "bbc.com": 1.0,
    "theguardian.com": 0.8, "skysports.com": 0.9, "espn.com": 0.8, "espn.co.uk": 0.8,
    "reuters.com": 1.0, "apnews.com": 0.9, "telegraph.co.uk": 0.8, "thetimes.co.uk": 0.8,
    "thetimes.com": 0.8, "independent.co.uk": 0.6, "standard.co.uk": 0.6,
    "marca.com": 0.7, "as.com": 0.6, "mundodeportivo.com": 0.6, "sport.es": 0.5,
    "kicker.de": 0.8, "lequipe.fr": 0.8, "gazzetta.it": 0.7, "corrieredellosport.it": 0.5,
    "goal.com": 0.5, "90min.com": 0.4, "talksport.com": 0.4, "manchestereveningnews.co.uk": 0.5,
    "liverpoolecho.co.uk": 0.5, "football.london": 0.5, "dailymail.co.uk": 0.4,
    "mirror.co.uk": 0.3, "express.co.uk": 0.3, "thesun.co.uk": 0.3, "dailystar.co.uk": 0.2,
}
DEFAULT_WEIGHT = 0.3

# deal stage from the headline's wording: 3 = agreed / close, 2 = bid / talks, 1 = interest
STAGE_WORDS = {
    3: ["agree", "agreed", "agreement", "here we go", "medical", "close in", "closing in", "set to sign",
        "set to join", "completes", "complete move", "finalise", "finalize", "seal", "sealed",
        "personal terms"],
    2: ["bid", "offer", "talks", "negotiat", "approach", "swoop", "make move", "submit", "tabled",
        "opening offer", "enquir", "inquir"],
    1: ["interest", "interested", "target", "eye", "eyes", "eyeing", "monitor", "scout", "want",
        "wants", "keen", "consider", "linked", "chase", "race", "battle", "pursue", "move for",
        "shortlist", "admire", "transfer", "sign", "signing", "loan"],
}
NEGATIVE = ["reject", "turn down", "turned down", "turns down", "snub", "not for sale", "deny", "denies",
            "rule out", "rules out", "ruled out", "no interest", "end interest", "ends interest",
            "pull out", "pulls out", "walk away", "walks away", "cool interest", "cools interest", "abandon"]
# outgoing or contract news: the headline doesn't name a buyer, and with a roster that is
# a season old the player's real club may be the one named, so these are skipped
OUTGOING = ["sell", "sale", "exit", "leave", "leaving", "depart", "offload", "axe", "ditch",
            "renew", "extension", "new contract", "new deal", "contract talks", "release clause"]
SELLER_CUES = {"from", "leave", "leaves", "leaving", "quit", "quits", "exit", "exits", "depart", "departs"}

# headline names for clubs beyond what can be derived from the database's names
CLUB_ALIASES = {
    "Manchester United": ["man utd", "man united", "manchester utd"],
    "Manchester City": ["man city"],
    "Tottenham Hotspur": ["tottenham", "spurs"],
    "Newcastle United": ["newcastle", "toon"],
    "West Ham United": ["west ham", "hammers"],
    "Wolverhampton Wanderers": ["wolves", "wolverhampton"],
    "Brighton & Hove Albion": ["brighton"],
    "Nottingham Forest": ["nottingham forest", "forest"],
    "AFC Bournemouth": ["bournemouth", "cherries"],
    "Leeds United": ["leeds"],
    "Leicester City": ["leicester"],
    "Arsenal": ["gunners"],
    "Everton": ["toffees"],
    "Aston Villa": ["villa"],
    "Crystal Palace": ["palace"],
    "FC Barcelona": ["barcelona", "barca", "barça"],
    "Real Madrid": ["madrid", "los blancos"],
    "Atlético Madrid": ["atletico madrid", "atletico", "atleti"],
    "Paris Saint-Germain": ["psg", "paris saint germain", "paris sg"],
    "FC Bayern München": ["bayern munich", "bayern munchen", "bayern"],
    "Borussia Dortmund": ["dortmund", "bvb"],
    "Bayer 04 Leverkusen": ["leverkusen", "bayer leverkusen"],
    "RB Leipzig": ["leipzig"],
    "Inter": ["inter milan", "internazionale", "inter"],
    "AC Milan": ["ac milan", "milan"],
    "Juventus": ["juve"],
    "Napoli": ["napoli"],
    "Roma": ["as roma", "roma"],
    "Lazio": ["lazio"],
    "Atalanta": ["atalanta"],
    "Olympique de Marseille": ["marseille"],
    "Olympique Lyonnais": ["lyon"],
    "Galatasaray SK": ["galatasaray"],
    "Fenerbahçe SK": ["fenerbahce"],
    "Beşiktaş JK": ["besiktas"],
    "SL Benfica": ["benfica"],
    "Sporting CP": ["sporting lisbon", "sporting cp"],
    "FC Porto": ["porto"],
    "Real Betis Balompié": ["real betis", "betis"],
    "Athletic Club": ["athletic bilbao", "athletic club"],
    "Al Hilal": ["al hilal", "al-hilal"],
    "Al Nassr": ["al nassr", "al-nassr"],
    "Al Ittihad": ["al ittihad", "al-ittihad"],
    "Al Ahli": ["al ahli", "al-ahli"],
}
_STRIP = re.compile(r"\b(fc|afc|cf|sc|ac|sk|jk|cd|ud|sv|club|calcio)\b")


def fold(text: str) -> str:
    t = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower()
    t = t.replace("'s", "").replace("’s", "")
    return " " + re.sub(r"[^a-z0-9]+", " ", t).strip() + " "


def _phrase(t: str) -> str:
    return fold(t).strip()


@dataclass
class Matchers:
    players: dict = field(default_factory=dict)   # alias -> player id (unique aliases only)
    clubs: dict = field(default_factory=dict)     # alias -> club name
    club_of: dict = field(default_factory=dict)   # player id -> club
    name_of: dict = field(default_factory=dict)

    @classmethod
    def build(cls, cur: pd.DataFrame, min_overall: int = 68, surname_min_overall: int = 78) -> "Matchers":
        pool = cur[(cur["overall"] >= min_overall) & cur["club"].notna()]
        alias_ids: dict[str, set] = {}

        def add(alias, pid):
            a = _phrase(alias)
            if len(a) >= 4:
                alias_ids.setdefault(a, set()).add(int(pid))
        surnames: dict[str, set] = {}
        for pid, name, long_name, ovr in zip(pool["id"], pool["name"], pool["long_name"], pool["overall"]):
            name = str(name)
            long_tokens = _phrase(long_name).split() if isinstance(long_name, str) else []
            if "." in name:                                   # "L. Hall" -> surname "Hall"
                surname = name.split(".", 1)[1].strip()
                if long_tokens:
                    add(f"{long_tokens[0]} {surname}", pid)
            else:                                             # "Pedri", "Vinícius Jr."
                surname = name
                add(name, pid)
            if len(long_tokens) >= 2:
                add(f"{long_tokens[0]} {long_tokens[-1]}", pid)
            s_key = _phrase(surname)
            surnames.setdefault(s_key, set()).add(int(pid))
            if ovr >= surname_min_overall:
                alias_ids.setdefault("__surname__" + s_key, set()).add(int(pid))
        players = {a: next(iter(ids)) for a, ids in alias_ids.items() if len(ids) == 1 and not a.startswith("__")}
        for key, ids in alias_ids.items():
            if key.startswith("__surname__"):
                s_key = key[len("__surname__"):]
                if len(surnames.get(s_key, ())) == 1 and len(s_key) >= 4 and s_key not in players:
                    players[s_key] = next(iter(ids))
        clubs = {}
        for c in cur["club"].dropna().unique():
            for a in {_phrase(c), _STRIP.sub(" ", fold(c)).strip()} | set(CLUB_ALIASES.get(c, [])):
                a = re.sub(r"\s+", " ", _phrase(a))
                if len(a) >= 3:
                    clubs.setdefault(a, c)
        # player aliases that are also club aliases (e.g. a player called "Madrid") are dropped
        players = {a: p for a, p in players.items() if a not in clubs}
        return cls(players=players, clubs=clubs,
                   club_of=dict(zip(cur["id"].astype(int), cur["club"])),
                   name_of=dict(zip(cur["id"].astype(int), cur["name"])))


def _find(text: str, aliases: dict) -> list[tuple[int, str, object]]:
    """(position, alias, value) for every alias found as whole words, longest first,
    without overlaps."""
    hits, taken = [], []
    for a in sorted(aliases, key=len, reverse=True):
        for m in re.finditer(re.escape(f" {a} "), text):
            s, e = m.start(), m.end()
            if any(s < te and ts < e - 1 for ts, te in taken):
                continue
            taken.append((s, e))
            hits.append((s, a, aliases[a]))
    return sorted(hits)


def stage_of(text: str) -> int:
    for stage in (3, 2, 1):
        if any(f" {w}" in text for w in STAGE_WORDS[stage]):
            return stage
    return 0


def parse_headline(title: str, m: Matchers) -> list[dict]:
    """Links found in one headline: [{player_id, to_club, from_club, stage}]."""
    text = fold(title)
    if any(f" {w}" in text for w in NEGATIVE + OUTGOING):
        return []
    stage = stage_of(text)
    if stage == 0:
        return []
    players = _find(text, m.players)
    clubs = _find(text, m.clubs)
    if not players or not clubs:
        return []
    out = []
    for _, _, pid in players[:2]:
        current = m.club_of.get(pid)
        for pos, alias, club in clubs:
            if club == current:
                continue
            before = text[:pos].split()[-1:] if text[:pos].strip() else []
            if before and before[0] in SELLER_CUES:
                continue
            out.append({"player_id": int(pid), "to_club": club, "from_club": current, "stage": stage})
    # one link per (player, club)
    seen, uniq = set(), []
    for o in out:
        k = (o["player_id"], o["to_club"])
        if k not in seen:
            seen.add(k)
            uniq.append(o)
    return uniq


def domain_weight(domain: str) -> float:
    d = (domain or "").lower().removeprefix("www.")
    for known, w in SOURCE_WEIGHT.items():
        if d == known or d.endswith("." + known):
            return w
    return DEFAULT_WEIGHT


def score_links(articles: list[dict], m: Matchers, now: datetime | None = None,
                half_life_days: float = 45.0, keep_articles: int = 5) -> list[dict]:
    """articles: [{title, url, domain, source, date (ISO)}] -> scored links, strongest first.

    score = 1 - exp(-sum of outlet weight x stage weight x recency), so one solid report
    gives ~0.5 and several agreeing ones approach 1."""
    now = now or datetime.now(timezone.utc)
    stage_w = {1: 0.5, 2: 0.8, 3: 1.0}
    seen_titles = set()
    acc: dict[tuple, dict] = {}
    for a in articles:
        key = fold(a["title"]).strip()
        if not key or key in seen_titles:          # syndicated copies count once
            continue
        seen_titles.add(key)
        links = parse_headline(a["title"], m)
        if not links:
            continue
        when = _parse_date(a.get("date"))
        age = max(0.0, (now - when).total_seconds() / 86400) if when else 30.0
        rec = 0.5 ** (age / half_life_days)
        for link in links:
            k = (link["player_id"], link["to_club"])
            e = acc.setdefault(k, {**link, "raw": 0.0, "articles": []})
            e["stage"] = max(e["stage"], link["stage"])
            w = domain_weight(a.get("domain")) * stage_w[link["stage"]] * rec
            e["raw"] += w
            e["articles"].append({"title": a["title"], "url": a["url"], "source": a.get("source") or a.get("domain"),
                                  "date": when.date().isoformat() if when else None, "_w": w})
    out = []
    for e in acc.values():
        e["articles"].sort(key=lambda x: -x["_w"])
        arts = [{k: v for k, v in x.items() if k != "_w"} for x in e["articles"][:keep_articles]]
        out.append({"player_id": e["player_id"], "player": m.name_of.get(e["player_id"]),
                    "to_club": e["to_club"], "from_club": e["from_club"], "stage": e["stage"],
                    "score": round(1 - math.exp(-e["raw"]), 3), "n_articles": len(e["articles"]),
                    "articles": arts})
    return sorted(out, key=lambda x: -x["score"])


def _parse_date(s) -> datetime | None:
    if not s:
        return None
    s = str(s)
    for fmt in ("%Y%m%dT%H%M%SZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d"):
        try:
            d = datetime.strptime(s, fmt)
            return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None
