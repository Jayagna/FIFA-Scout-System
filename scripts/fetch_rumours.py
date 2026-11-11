"""Refresh the transfer-rumour feed (rumours/rumours.json).

Headlines come from two free sources that allow automated access and can be searched
over a date range (so a refresh every two months misses nothing):
  * GDELT DOC 2.0 API: worldwide news index, one search per big club, last ~2 months
  * The Guardian Open Platform: the football transfer-window tag, same window
    (set GUARDIAN_API_KEY for the full rate limit; the public "test" key works too)

Only headline, link, outlet and date are stored, never article text. Headlines are
matched to players and clubs in the newest edition of the database
(fcscout.rumours), scored by outlet, deal stage and recency, and written out.

Runs every two months from .github/workflows/rumours.yml, or by hand:
    python scripts/fetch_rumours.py                 # last 70 days
    python scripts/fetch_rumours.py --days 60 --clubs 40
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fcscout.rumours import CLUB_ALIASES, Matchers, score_links  # noqa: E402

OUT = ROOT / "rumours/rumours.json"
UA = "career-mode-scout/1.0 (+https://github.com/Jayagna/FIFA-Scout-System)"
GDELT = "https://api.gdeltproject.org/api/v2/doc/doc"
GUARDIAN = "https://content.guardianapis.com/search"


def http_get(url: str, tries: int = 2) -> bytes:
    for attempt in range(1, tries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == tries:
                raise
            print(f"    retrying ({e})")
            time.sleep(10 * attempt)
    raise RuntimeError("unreachable")


def latest_players() -> pd.DataFrame:
    """Newest edition: from the built dataset if present, else straight from data/raw."""
    built = ROOT / "data/processed/players.pkl"
    if built.exists():
        p = pd.read_pickle(built)
        return p[p["edition"] == p["edition"].max()].copy()
    from fcscout.data import load_raw_dir
    p = load_raw_dir(ROOT / "data/raw", fill=False)
    return p[p["edition"] == p["edition"].max()].copy()


def headline_name(club: str) -> str:
    """How headlines usually name a club: the first listed alias, else its name."""
    return (CLUB_ALIASES.get(club) or [club])[0]


def parse_gdelt(payload: bytes) -> list[dict]:
    try:
        data = json.loads(payload.decode("utf-8", errors="replace") or "{}")
    except json.JSONDecodeError:          # GDELT answers plain text on bad queries
        return []
    return [{"title": a.get("title", ""), "url": a.get("url", ""), "domain": a.get("domain", ""),
             "source": a.get("domain", ""), "date": a.get("seendate")}
            for a in data.get("articles", []) if a.get("title") and a.get("url")]


def fetch_gdelt(clubs: list[str], start: datetime, end: datetime, get=http_get, pause: float = 6.0) -> list[dict]:
    out, failures = [], 0
    for i, club in enumerate(clubs):
        if failures >= 3:
            print("  GDELT failed three times in a row; skipping the remaining clubs")
            break
        q = f'"{headline_name(club)}" (transfer OR bid OR target OR signing OR interest) sourcelang:english'
        url = GDELT + "?" + urllib.parse.urlencode({
            "query": q, "mode": "artlist", "format": "json", "maxrecords": 250, "sort": "datedesc",
            "startdatetime": start.strftime("%Y%m%d%H%M%S"), "enddatetime": end.strftime("%Y%m%d%H%M%S")})
        try:
            got = parse_gdelt(get(url))
            failures = 0
        except Exception as e:  # noqa: BLE001 - one failed club must not stop the run
            print(f"  GDELT {club}: failed ({e})")
            got = []
            failures += 1
        print(f"  GDELT {club}: {len(got)} headlines")
        out += got
        if pause and i < len(clubs) - 1:
            time.sleep(pause)               # GDELT asks for at most one request every 5 s
    return out


def parse_guardian(payload: bytes) -> tuple[list[dict], int]:
    resp = json.loads(payload.decode("utf-8")).get("response", {})
    rows = [{"title": r.get("webTitle", ""), "url": r.get("webUrl", ""), "domain": "theguardian.com",
             "source": "The Guardian", "date": r.get("webPublicationDate")}
            for r in resp.get("results", []) if r.get("webTitle")]
    return rows, int(resp.get("pages", 1) or 1)


def fetch_guardian(start: datetime, key: str, get=http_get, max_pages: int = 20) -> list[dict]:
    out, page, pages = [], 1, 1
    while page <= min(pages, max_pages):
        url = GUARDIAN + "?" + urllib.parse.urlencode({
            "tag": "football/transfer-window", "from-date": start.date().isoformat(),
            "order-by": "newest", "page-size": 50, "page": page, "api-key": key})
        try:
            rows, pages = parse_guardian(get(url))
        except Exception as e:  # noqa: BLE001
            print(f"  Guardian page {page}: failed ({e})")
            break
        out += rows
        page += 1
        time.sleep(1)
    print(f"  Guardian: {len(out)} headlines")
    return out


def build(articles: list[dict], cur: pd.DataFrame, days: int, sources: list[str], now: datetime) -> dict:
    m = Matchers.build(cur)
    links = score_links(articles, m, now=now)
    return {"generated": now.isoformat(timespec="seconds"), "window_days": days,
            "edition": int(cur["edition"].max()), "sources": sources,
            "headlines_scanned": len(articles), "links": links}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=70, help="how far back to look (GDELT keeps ~3 months)")
    ap.add_argument("--clubs", type=int, default=80, help="search this many of the strongest clubs on GDELT")
    args = ap.parse_args(argv)
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=args.days)
    cur = latest_players()
    ranked = cur.dropna(subset=["club"]).sort_values("overall", ascending=False)
    strength = ranked[ranked.groupby("club").cumcount() < 16].groupby("club")["overall"].mean()
    clubs = strength.sort_values(ascending=False).head(args.clubs).index.tolist()
    print(f"Searching {len(clubs)} clubs, {start.date()} to {now.date()}")
    articles = fetch_gdelt(clubs, start, now)
    articles += fetch_guardian(start, os.environ.get("GUARDIAN_API_KEY") or "test")
    if not articles:
        print("No headlines fetched (network problem?); keeping the previous rumours file.")
        return 1
    feed = build(articles, cur, args.days, ["GDELT", "The Guardian"], now)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(feed, indent=1, ensure_ascii=False))
    print(f"{len(feed['links'])} player-club links from {len(articles)} headlines -> {OUT.relative_to(ROOT)}")
    for link in feed["links"][:15]:
        print(f"  {link['score']:.2f}  {link['player']} -> {link['to_club']}  ({link['n_articles']} headlines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
