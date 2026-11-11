"""Tests for the transfer-rumour feed: parsing both sources' responses, matching
headlines to players and clubs, scoring, and the squad builder using the links.

Run:  python -m pytest -q tests/test_rumours.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fcscout.rumours import Matchers, domain_weight, parse_headline, score_links  # noqa: E402

NOW = datetime(2026, 11, 10, tzinfo=timezone.utc)


def _db() -> pd.DataFrame:
    rows = [
        (1, "L. Hall", "Lewis Hall", "Newcastle United", 80),
        (2, "J. Bellingham", "Jude Victor William Bellingham", "Real Madrid", 90),
        (3, "F. Valverde", "Federico Santiago Valverde Dipetta", "Real Madrid", 89),
        (4, "Pedri", "Pedro González López", "FC Barcelona", 88),
        (5, "C. Gallagher", "Conor Gallagher", "Atlético Madrid", 81),
        (6, "S. Gallagher", "Sam Gallagher", "Blackburn Rovers", 70),
        (7, "B. Saka", "Bukayo Saka", "Arsenal", 88),
        (8, "M. Salah", "Mohamed Salah Hamed Ghaly", "Liverpool", 91),
        (20, "A. Keeper", "Alan Keeper", "Manchester United", 75),
        (21, "B. Back", "Bob Back", "Tottenham Hotspur", 74),
        (22, "C. Chels", "Carl Chels", "Chelsea", 76),
        (23, "Gabriel", "Gabriel dos Santos Magalhães", "Arsenal", 86),
    ]
    return pd.DataFrame(rows, columns=["id", "name", "long_name", "club", "overall"]).assign(edition=26)


def _m():
    return Matchers.build(_db())


def _links(title):
    m = _m()
    return [(m.name_of[l["player_id"]], l["to_club"], l["stage"]) for l in parse_headline(title, m)]


def test_full_names_nicknames_and_stages():
    assert _links("Man Utd make move for Newcastle's Lewis Hall") == [("L. Hall", "Manchester United", 2)]
    assert _links("Manchester United eye Hall as left-back target") == [("L. Hall", "Manchester United", 1)]
    assert _links("Spurs hold talks over Conor Gallagher") == [("C. Gallagher", "Tottenham Hotspur", 2)]
    assert _links("Barca agree terms with Federico Valverde") == [("F. Valverde", "FC Barcelona", 3)]
    assert _links("Arsenal and Chelsea in race for Jude Bellingham") == [
        ("J. Bellingham", "Arsenal", 1), ("J. Bellingham", "Chelsea", 1)]
    assert _links("Bayern Munich scout Pedri") == []          # Bayern isn't in this small database


def test_precision_rules():
    assert _links("Tottenham want Gallagher") == []             # two Gallaghers: surname is ambiguous
    assert _links("Liverpool reject bid for Salah") == []       # denials and rejections are skipped
    assert _links("Newcastle open to selling Lewis Hall") == []  # outgoing news names no buyer
    assert _links("Arsenal sign Saka to new contract") == []    # contract news
    assert _links("Chelsea sporting director talks up Saka") != [("B. Saka", "Sporting CP", 1)]
    assert _links("Lewis Hall stars as Newcastle win") == []    # no transfer wording
    # a club named after "from" is the seller, not a suitor
    assert _links("Man Utd want Conor Gallagher from Chelsea") == [("C. Gallagher", "Manchester United", 1)]
    # a one-word name next to another name is someone else (a Man Utd youth player here)
    assert _links("JJ Gabriel: Man Utd sensation reacts to Barcelona offer") == []
    assert _links("Man United latest stance on JJ Gabriel as transfer suitors circle") == []
    assert _links("Barcelona eye Arsenal\u2019s Gabriel") == [("Gabriel", "FC Barcelona", 1)]
    assert _links("Barcelona Eye Arsenal Defender Gabriel In Shock Move") == [("Gabriel", "FC Barcelona", 1)]


def test_scores_reward_reliable_recent_and_repeated_reports():
    m = _m()
    arts = [
        {"title": "Man Utd make move for Lewis Hall", "url": "https://a", "domain": "theathletic.com", "date": "2026-11-01T10:00:00Z"},
        {"title": "Manchester United in talks over Lewis Hall", "url": "https://b", "domain": "www.bbc.co.uk", "date": "20261105T080000Z"},
        {"title": "Man Utd make move for Lewis Hall", "url": "https://c", "domain": "mirror.co.uk", "date": "2026-11-02"},  # copy
        {"title": "Chelsea eye Bukayo Saka", "url": "https://d", "domain": "dailystar.co.uk", "date": "2026-08-01"},
    ]
    links = score_links(arts, m, now=NOW)
    assert [(l["player"], l["to_club"]) for l in links] == [("L. Hall", "Manchester United"), ("B. Saka", "Chelsea")]
    hall, saka = links
    assert hall["n_articles"] == 2 and hall["stage"] == 2 and hall["score"] > 0.6
    assert saka["score"] < 0.1                                   # one old tabloid line
    assert domain_weight("sport.bbc.co.uk") == 1.0 and domain_weight("unknown.blog") == 0.3


def test_source_payloads_parse():
    from fetch_rumours import parse_gdelt, parse_guardian
    gd = json.dumps({"articles": [
        {"url": "https://x/1", "url_mobile": "", "title": "Man Utd make move for Lewis Hall",
         "seendate": "20261101T101500Z", "socialimage": "", "domain": "theathletic.com",
         "language": "English", "sourcecountry": "United States"},
        {"url": "", "title": "no url"}]}).encode()
    assert [a["title"] for a in parse_gdelt(gd)] == ["Man Utd make move for Lewis Hall"]
    assert parse_gdelt(b"Your query was too short") == []
    gu = json.dumps({"response": {"status": "ok", "pages": 3, "results": [
        {"id": "football/2026/nov/03/x", "webTitle": "Football transfer rumours: Man Utd want Lewis Hall?",
         "webUrl": "https://www.theguardian.com/x", "webPublicationDate": "2026-11-03T08:00:00Z"}]}}).encode()
    rows, pages = parse_guardian(gu)
    assert pages == 3 and rows[0]["domain"] == "theguardian.com"


def test_fetchers_with_fake_network():
    from fetch_rumours import fetch_gdelt, fetch_guardian
    calls = []

    def fake_get(url):
        calls.append(url)
        if "gdeltproject" in url:
            return json.dumps({"articles": [{"url": "https://x", "title": "Spurs eye Conor Gallagher",
                                              "seendate": "20261101T000000Z", "domain": "skysports.com"}]}).encode()
        return json.dumps({"response": {"pages": 1, "results": [
            {"webTitle": "Arsenal and Chelsea in race for Jude Bellingham", "webUrl": "https://g",
             "webPublicationDate": "2026-11-02T00:00:00Z"}]}}).encode()
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    arts = fetch_gdelt(["Tottenham Hotspur", "Manchester United"], start, NOW, get=fake_get, pause=0)
    arts += fetch_guardian(start, "test", get=fake_get)
    assert any("%22tottenham%22" in c.lower() for c in calls)       # searched by headline name
    assert any("startdatetime=20260901" in c for c in calls)
    links = score_links(arts, _m(), now=NOW)
    assert {(l["player"], l["to_club"]) for l in links} == {
        ("C. Gallagher", "Tottenham Hotspur"), ("J. Bellingham", "Arsenal"), ("J. Bellingham", "Chelsea")}


def test_squad_builder_favours_rumoured_player(tmp_path):
    """The runner-up signing, once linked with the club, beats the unlinked best one."""
    from test_web import _league
    from fcscout import service
    from fcscout.features import build_panel
    panel = build_panel(_league())
    cur = panel[panel["edition"] == 26]
    fc = pd.DataFrame({"id": cur["id"], **{c: cur["overall"] for c in
                       ["pred_t1", "pred_t1_lo", "pred_t1_hi", "pred_t2", "pred_t2_lo", "pred_t2_hi"]}})
    store = service.Store.from_frames(panel, fc)
    plain = service.squad_plan(store, "Brentford", budget_m=200, max_signings=1, objective="now",
                               alternatives=1, realistic=False)
    best_plan, runner_up = plain["plans"]
    gap = (best_plan["xi_rating"] - runner_up["xi_rating"]) * 11       # in summed rating points
    assert 0 <= gap < service.RUMOUR_POINTS * 0.99
    target = runner_up["buys"][0]["id"]
    feed = {"generated": "2026-11-10T00:00:00+00:00", "window_days": 70, "sources": ["test"],
            "links": [{"player_id": target, "to_club": "Brentford", "from_club": None, "stage": 3, "score": 0.99,
                       "articles": [{"title": "t", "url": "https://u", "source": "s", "date": "2026-11-01"}]}]}
    path = tmp_path / "rumours.json"
    path.write_text(json.dumps(feed))
    store.rumours = service.load_rumours(path, store.cur)
    linked = service.squad_plan(store, "Brentford", budget_m=200, max_signings=1, objective="now",
                                alternatives=0, realistic=False, rumours=True)
    ids = [b["id"] for b in linked["buys"]]
    assert target in ids and linked["buys"][ids.index(target)]["rumour"]["score"] == 0.99
    assert service.rumours(store, "Brentford")["players"][0]["id"] == target


def test_gdelt_outage_stops_early():
    from fetch_rumours import fetch_gdelt
    calls = []

    def down(url):
        calls.append(url)
        raise TimeoutError("down")
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    assert fetch_gdelt([f"Club {i}" for i in range(20)], start, NOW, get=down, pause=0) == []
    assert len(calls) == 3


def test_gdelt_rate_limits_are_waited_out_and_full_searches_continue():
    from fetch_rumours import GDELT_CAP, RateLimited, fetch_gdelt
    calls, waits = [], []
    answers = iter([
        RateLimited("429"),                                          # club 1: slow down...
        b"Please limit requests to one every 5 seconds",             # ...plain-text version
        {"n": GDELT_CAP, "day": "20261101"},                         # then a full page
        {"n": 3, "day": "20260915"},                                 # older part of the window
        {"n": 2, "day": "20261020"},                                 # club 2: not full
    ])

    def get(url):
        calls.append(url)
        a = next(answers)
        if isinstance(a, Exception):
            raise a
        if isinstance(a, bytes):
            return a
        return json.dumps({"articles": [{"url": f"https://x/{len(calls)}/{i}", "title": "Spurs eye Conor Gallagher",
                                         "seendate": a["day"] + "T000000Z", "domain": "x.com"}
                                        for i in range(a["n"])]}).encode()
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    arts = fetch_gdelt(["Tottenham Hotspur", "Arsenal"], start, NOW, get=get, pause=0,
                       backoff=(1, 2), sleep=waits.append)
    assert waits == [1, 2] and len(calls) == 5
    assert len(arts) == GDELT_CAP + 3 + 2
    assert "enddatetime=20261101" in calls[3]                        # second search ends where the first stopped


def test_guardian_needs_a_key(monkeypatch, capsys):
    import fetch_rumours
    monkeypatch.delenv("GUARDIAN_API_KEY", raising=False)
    monkeypatch.setattr(fetch_rumours, "fetch_gdelt", lambda *a, **k: [])
    monkeypatch.setattr(fetch_rumours, "fetch_guardian", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    monkeypatch.setattr(fetch_rumours, "latest_players", _db)
    assert fetch_rumours.main(["--clubs", "2"]) == 1                 # nothing fetched: old file kept
    assert "Guardian: skipped" in capsys.readouterr().out
