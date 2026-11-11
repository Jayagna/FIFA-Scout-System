"""Refresh the transfer-rumour feed (rumours/rumours.json).

Headlines come from two free sources that allow automated access and can be searched
over a date range (so a refresh every two months misses nothing):
  * GDELT DOC 2.0 API: worldwide news index, one search per big club, last ~2 months
    (a second search per club when the first hits GDELT's 250-article cap)
  * The Guardian Open Platform: the football transfer-window tag, same window; needs a
    free key in GUARDIAN_API_KEY (open-platform.theguardian.com/access), skipped without one

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


class RateLimited(Exception):
    """The source asked us to slow down (HTTP 429, or GDELT's plain-text notice)."""


def http_get(url: str, tries: int = 2) -> bytes:
    for attempt in range(1, tries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                raise RateLimited(str(e)) from e
            if attempt == tries or e.code < 500:
                raise
            print(f"    retrying ({e})")
            time.sleep(10 * attempt)
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
    text = payload.decode("utf-8", errors="replace")
    try:
        data = json.loads(text or "{}")
    except json.JSONDecodeError:          # GDELT answers plain text on bad queries...
        if "limit requests" in text.lower():
            raise RateLimited(text[:80]) from None   # ...and when asked too often
        return []
    return [{"title": a.get("title", ""), "url": a.get("url", ""), "domain": a.get("domain", ""),
             "source": a.get("domain", ""), "date": a.get("seendate")}
            for a in data.get("articles", []) if a.get("title") and a.get("url")]


GDELT_CAP = 250
BACKOFF = (20, 45, 90, 180)        # seconds to wait after each "slow down"


def gdelt_url(club: str, start: datetime, end: datetime) -> str:
    q = f'"{headline_name(club)}" (transfer OR bid OR target OR signing OR "keen on" OR swoop) sourcelang:english'
    return GDELT + "?" + urllib.parse.urlencode({
        "query": q, "mode": "artlist", "format": "json", "maxrecords": GDELT_CAP, "sort": "datedesc",
        "startdatetime": start.strftime("%Y%m%d%H%M%S"), "enddatetime": end.strftime("%Y%m%d%H%M%S")})


def _gdelt_time(seen: str | None) -> datetime | None:
    try:
        return datetime.strptime(seen or "", "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def fetch_gdelt(clubs: list[str], start: datetime, end: datetime, get=http_get, pause: float = 6.0,
                backoff=BACKOFF, sleep=time.sleep) -> list[dict]:
    """One search per club, newest first. GDELT returns at most 250 articles per search,
    which for a big club covers only the last few weeks, so when a search comes back
    full a second one covers the older part of the window. A "slow down" answer is
    waited out and retried; after three clubs in a row fail anyway, the rest are skipped."""
    out, failed_in_a_row = [], 0

    def one(url: str) -> list[dict] | None:
        for wait in (*backoff, None):
            try:
                return parse_gdelt(get(url))
            except RateLimited:
                if wait is None:
                    return None
                print(f"    rate limited; waiting {wait}s")
                sleep(wait)
            except Exception as e:  # noqa: BLE001 - one failed club must not stop the run
                print(f"    failed ({e})")
                return None
        return None

    for i, club in enumerate(clubs):
        if failed_in_a_row >= 3:
            print("  GDELT failed for three clubs in a row; skipping the remaining clubs")
            break
        got = one(gdelt_url(club, start, end))
        if got is None:
            failed_in_a_row += 1
            print(f"  GDELT {club}: failed")
            continue
        failed_in_a_row = 0
        oldest = min((t for t in map(_gdelt_time, (a["date"] for a in got)) if t), default=None)
        if len(got) >= GDELT_CAP and oldest and oldest - start > timedelta(days=3):
            if pause:
                sleep(pause)
            older = one(gdelt_url(club, start, oldest))
            got += older or []
        print(f"  GDELT {club}: {len(got)} headlines")
        out += got
        if pause and i < len(clubs) - 1:
            sleep(pause)               # GDELT asks for at most one request every 5 s
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
    sources = ["GDELT"]
    key = os.environ.get("GUARDIAN_API_KEY", "").strip()
    if key:
        guardian = fetch_guardian(start, key)
        articles += guardian
        if guardian:
            sources.append("The Guardian")
    else:
        print("  Guardian: skipped (set GUARDIAN_API_KEY to include it)")
    if not articles:
        print("No headlines fetched (network problem?); keeping the previous rumours file.")
        return 1
    feed = build(articles, cur, args.days, sources, now)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(feed, indent=1, ensure_ascii=False))
    print(f"{len(feed['links'])} player-club links from {len(articles)} headlines -> {OUT.relative_to(ROOT)}")
    for link in feed["links"][:15]:
        print(f"  {link['score']:.2f}  {link['player']} -> {link['to_club']}  ({link['n_articles']} headlines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
