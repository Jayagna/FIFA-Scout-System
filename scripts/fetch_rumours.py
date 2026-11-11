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
    python scripts/fetch_rumours.py --days 60 --clubs 40 --minutes 20
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fcscout.rumours import Matchers, score_links  # noqa: E402

OUT = ROOT / "rumours/rumours.json"
SEEN = ROOT / "rumours/headlines.json"   # this cycle's transfer headlines, so later runs can continue
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


# what to search GDELT for: names as headlines write them, at least 5 letters (GDELT
# rejects shorter phrases such as "PSG"), at most two per club
GDELT_TERMS = {
    "Manchester United": ["Manchester United", "Man Utd"], "Manchester City": ["Manchester City", "Man City"],
    "Tottenham Hotspur": ["Tottenham"], "Newcastle United": ["Newcastle"], "West Ham United": ["West Ham"],
    "Wolverhampton Wanderers": ["Wolves", "Wolverhampton"], "Brighton & Hove Albion": ["Brighton"],
    "AFC Bournemouth": ["Bournemouth"], "Nottingham Forest": ["Nottingham Forest"],
    "Paris Saint-Germain": ["Paris Saint-Germain", "Paris SG"], "FC Bayern München": ["Bayern Munich"],
    "Bayer 04 Leverkusen": ["Leverkusen"], "Borussia Dortmund": ["Dortmund"], "RB Leipzig": ["Leipzig"],
    "Inter": ["Inter Milan", "Internazionale"], "AC Milan": ["AC Milan"], "FC Barcelona": ["Barcelona"],
    "Atlético Madrid": ["Atletico Madrid"], "Olympique de Marseille": ["Marseille"],
    "Olympique Lyonnais": ["Olympique Lyonnais"], "Sporting CP": ["Sporting Lisbon", "Sporting CP"],
    "Fenerbahçe SK": ["Fenerbahce"], "Beşiktaş JK": ["Besiktas"], "Real Betis Balompié": ["Real Betis"],
    "Athletic Club": ["Athletic Bilbao", "Athletic Club"],
}
_CLUB_SUFFIX = re.compile(r"\b(FC|AFC|CF|SK|JK|SL|SC|CD|UD|SV|AS)\b")


def gdelt_terms(club: str) -> list[str]:
    if club in GDELT_TERMS:
        return GDELT_TERMS[club]
    plain = unicodedata.normalize("NFKD", club).encode("ascii", "ignore").decode()
    plain = re.sub(r"^\d+\.\s*", "", plain)                       # "1. FC Union Berlin"
    return [re.sub(r"\s+", " ", _CLUB_SUFFIX.sub(" ", plain)).strip() or plain]


def parse_gdelt(payload: bytes) -> list[dict]:
    text = payload.decode("utf-8", errors="replace")
    try:
        data = json.loads(text or "{}")
    except json.JSONDecodeError:          # GDELT answers plain text on bad queries...
        if "limit requests" in text.lower():
            raise RateLimited(text[:80]) from None   # ...and when asked too often
        if text.strip():
            print(f"    GDELT says: {text.strip()[:80]}")
        return []
    return [{"title": a.get("title", ""), "url": a.get("url", ""), "domain": a.get("domain", ""),
             "source": a.get("domain", ""), "date": a.get("seendate")}
            for a in data.get("articles", []) if a.get("title") and a.get("url")]


GDELT_CAP = 250
BACKOFF = (15, 30, 60)             # seconds to wait after each "slow down" for one search
PAUSE = 8.0                        # between searches; GDELT asks for at most one per 5 s
COOLDOWN = 240.0                   # pause when GDELT keeps refusing, club after club


def gdelt_url(club: str, start: datetime, end: datetime) -> str:
    terms = [f'"{t}"' for t in gdelt_terms(club)]
    who = terms[0] if len(terms) == 1 else "(" + " OR ".join(terms) + ")"
    q = f'{who} (transfer OR bid OR target OR signing OR "keen on" OR swoop) sourcelang:english'
    return GDELT + "?" + urllib.parse.urlencode({
        "query": q, "mode": "artlist", "format": "json", "maxrecords": GDELT_CAP, "sort": "datedesc",
        "startdatetime": start.strftime("%Y%m%d%H%M%S"), "enddatetime": end.strftime("%Y%m%d%H%M%S")})


def _gdelt_time(seen: str | None) -> datetime | None:
    try:
        return datetime.strptime(seen or "", "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def fetch_gdelt(clubs: list[str], start: datetime, end: datetime, get=http_get, pause: float = PAUSE,
                backoff=BACKOFF, sleep=time.sleep, budget_s: float | None = None,
                clock=time.monotonic, done: list | None = None, cooldown: float = COOLDOWN) -> list[dict]:
    """One search per club (strongest clubs first), newest articles first.

    GDELT answers at most 250 articles per search, which for a big club covers only the
    last few weeks, so once every club has had its search, clubs whose search came back
    full get a second one for the older part of the window.

    "Slow down" answers are waited out and retried; when two clubs in a row stay rate
    limited, everything pauses for `cooldown` seconds, and clubs that failed are tried
    once more at the end. The whole thing stops when budget_s seconds have passed (or
    after three real errors in a row, e.g. GDELT down) and returns what it has so far:
    a partial feed beats a run that times out and saves nothing. Clubs whose search
    worked are added to `done`, so the next run can carry on with the others."""
    out = []
    errors_in_a_row = limited_in_a_row = 0
    deadline = clock() + budget_s if budget_s else None

    def time_left(need: float = 0) -> bool:
        return deadline is None or clock() + need < deadline

    def one(url: str) -> tuple[list[dict] | None, bool]:
        """(articles, False) on success, (None, True) if still rate limited, (None, False) on error."""
        for wait in (*backoff, None):
            try:
                return parse_gdelt(get(url)), False
            except RateLimited:
                if wait is None or not time_left(wait):
                    return None, True
                print(f"    rate limited; waiting {wait}s")
                sleep(wait)
            except Exception as e:  # noqa: BLE001 - one failed club must not stop the run
                print(f"    failed ({e})")
                return None, False
        return None, True

    def run(jobs: list[tuple[str, datetime, datetime]], label: str, retry: bool) -> list[tuple[str, list[dict]]]:
        nonlocal errors_in_a_row, limited_in_a_row
        ok, failed = [], []
        queue = list(jobs)
        tried_again = set()
        k = 0
        while queue:
            club, s, e = queue.pop(0)
            if errors_in_a_row >= 3:
                print("  GDELT failed three times in a row; stopping")
                break
            if not time_left():
                print(f"  GDELT time budget used up ({len(ok)} {label} searches done, "
                      f"{len(queue) + 1} not reached)")
                break
            if limited_in_a_row >= 2:
                if not time_left(cooldown):
                    print("  GDELT keeps saying slow down and the time budget is nearly used up; stopping")
                    break
                print(f"  GDELT keeps saying slow down; pausing {cooldown:.0f}s")
                sleep(cooldown)
                limited_in_a_row = 0
            elif k and pause:
                sleep(pause)
            k += 1
            got, limited = one(gdelt_url(club, s, e))
            if got is None:
                if limited:
                    limited_in_a_row += 1
                else:
                    errors_in_a_row += 1
                print(f"  GDELT {club} ({label}): failed")
                if retry and club not in tried_again:
                    tried_again.add(club)
                    queue.append((club, s, e))          # one more try at the end
                continue
            errors_in_a_row = limited_in_a_row = 0
            print(f"  GDELT {club} ({label}): {len(got)} headlines")
            out.extend(got)
            ok.append((club, got))
        return ok

    first = run([(c, start, end) for c in clubs], "recent", retry=True)
    if done is not None:
        done.extend(club for club, _ in first)
    older = []
    for club, got in first:
        oldest = min((t for t in map(_gdelt_time, (a["date"] for a in got)) if t), default=None)
        if len(got) >= GDELT_CAP and oldest and oldest - start > timedelta(days=3):
            older.append((club, start, oldest))
    if older and errors_in_a_row < 3 and time_left(pause):
        if pause:
            sleep(pause)
        run(older, "older", retry=False)
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


def load_seen(path: Path, now: datetime, max_age_days: float) -> dict | None:
    """This cycle's saved headlines, if the last run was recent enough to continue."""
    try:
        seen = json.loads(path.read_text())
        updated = datetime.fromisoformat(seen["updated"])
        start = datetime.fromisoformat(seen["window_start"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if now - updated > timedelta(days=max_age_days):
        return None
    seen["window_start"] = start
    return seen


def merge_articles(*groups: list[dict]) -> list[dict]:
    """One copy per URL; only headlines with transfer wording are worth keeping."""
    from fcscout.rumours import fold, stage_of, strip_site_name
    out, urls = [], set()
    for group in groups:
        for a in group:
            if a.get("url") and a["url"] not in urls and stage_of(fold(strip_site_name(a.get("title", "")))):
                urls.add(a["url"])
                out.append({k: a.get(k) for k in ("title", "url", "domain", "source", "date")})
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
    ap.add_argument("--minutes", type=float, default=35, help="time budget for the GDELT searches")
    ap.add_argument("--continue-days", type=float, default=7,
                    help="carry on from rumours/headlines.json if the last run was this recent")
    ap.add_argument("--fresh", action="store_true", help="ignore rumours/headlines.json and start over")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)  # show progress live in the Actions log
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=args.days)
    cur = latest_players()
    ranked = cur.dropna(subset=["club"]).sort_values("overall", ascending=False)
    strength = ranked[ranked.groupby("club").cumcount() < 16].groupby("club")["overall"].mean()
    clubs = strength.sort_values(ascending=False).head(args.clubs).index.tolist()

    # GDELT allows few searches per hour from GitHub's shared machines, so a cycle's search
    # is spread over several runs (the workflow runs on three days in a row): each run
    # searches the clubs the previous ones didn't reach and adds to the saved headlines
    seen = None if args.fresh else load_seen(SEEN, now, args.continue_days)
    old, done = [], []
    if seen:
        start, old, done = seen["window_start"], seen["articles"], list(seen["clubs_done"])
        print(f"Continuing this cycle: {len(done)} of {len(clubs)} clubs already searched, "
              f"{len(old)} headlines saved")
    todo = [c for c in clubs if c not in set(done)]
    if seen and not todo:
        print("Every club has been searched this cycle; nothing to do until the next one.")
        return 0
    print(f"Searching {len(todo)} clubs, {start.date()} to {now.date()}")
    new = fetch_gdelt(todo, start, now, budget_s=args.minutes * 60, done=done) if todo else []
    sources = ["GDELT"]
    key = os.environ.get("GUARDIAN_API_KEY", "").strip()
    if key:
        guardian = fetch_guardian(start, key)
        new += guardian
        if guardian:
            sources.append("The Guardian")
    else:
        print("  Guardian: skipped (set GUARDIAN_API_KEY to include it)")
    if not new and not old:
        print("No headlines fetched (network problem?); keeping the previous rumours file.")
        return 1
    articles = merge_articles(old, new)
    OUT.parent.mkdir(exist_ok=True)
    SEEN.write_text(json.dumps({"window_start": start.isoformat(), "updated": now.isoformat(timespec="seconds"),
                                "clubs_done": done, "articles": articles}, ensure_ascii=False))
    feed = build(articles, cur, (now - start).days, sources, now)
    feed["clubs_searched"] = len(done)
    feed["clubs_total"] = len(clubs)
    OUT.write_text(json.dumps(feed, indent=1, ensure_ascii=False))
    print(f"{len(done)} of {len(clubs)} clubs searched this cycle; {len(feed['links'])} player-club links "
          f"from {len(articles)} transfer headlines -> rumours/{OUT.name}")
    for link in feed["links"][:15]:
        print(f"  {link['score']:.2f}  {link['player']} -> {link['to_club']}  ({link['n_articles']} headlines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
