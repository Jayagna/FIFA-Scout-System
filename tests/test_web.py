"""Tests for the website's backend: every service function on a small synthetic league,
then the real HTTP server (started on a free port) for routing, errors and static files.

Run:  python -m pytest -q tests/test_web.py
"""
from __future__ import annotations

import json
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from fcscout import service  # noqa: E402
from fcscout.data import GK_ATTRS, OUTFIELD_ATTRS, POS_RATING_COLS, STANDARD_COLUMNS  # noqa: E402
from fcscout.features import build_panel  # noqa: E402

POSITIONS = ["GK", "GK", "CB", "CB", "CB", "LB", "RB", "CDM", "CM", "CM", "CAM", "LW", "RW", "ST", "ST", "LM", "RM", "CB"]


def _league(seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    clubs = [("Arsenal", "Premier League (England)", 13), ("Chelsea", "Premier League (England)", 13),
             ("Brentford", "Premier League (England)", 13), ("Sevilla FC", "La Liga (Spain)", 53)]
    pid = 1
    for ed in (25, 26):
        pid = 1
        for club, league, lid in clubs:
            for k, pos in enumerate(POSITIONS):
                ovr = int(rng.integers(62, 86))
                r = {c: np.nan for c in STANDARD_COLUMNS}
                r.update({a: float(rng.integers(30, 90)) for a in OUTFIELD_ATTRS + GK_ATTRS})
                r.update({f"pos_{c}": float(max(20, ovr - rng.integers(0, 20))) for c in POS_RATING_COLS})
                if pos == "GK":
                    r.update({f"pos_{c}": 25.0 for c in POS_RATING_COLS}, pos_gk=float(ovr))
                else:
                    r[f"pos_{pos.lower()}"] = float(ovr)
                    r["pos_gk"] = 20.0
                r.update(id=pid, edition=ed, name=f"Player {pid}" if pid != 7 else "Jérôme Böateng",
                         long_name=f"Long Name {pid}", age=float(17 + (pid * 7) % 18) + (ed - 25),
                         club=club, league=league, league_id=lid, overall=ovr, potential=ovr + int(rng.integers(0, 10)),
                         value_eur=float(rng.integers(1, 60)) * 1e6, wage_eur=float(rng.integers(5, 150)) * 1e3,
                         positions=pos, primary_pos=pos, preferred_foot="Right", international_reputation=1,
                         weak_foot=3, skill_moves=3, contract_end=2028, on_loan=0, nationality="England",
                         face_url=f"https://cdn.sofifa.net/players/000/{pid:03d}/{ed}_120.png",
                         height_cm=180, weight_kg=75, ratings_derived=0, money_estimated=0)
                rows.append(r)
                pid += 1
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def store() -> service.Store:
    players = _league()
    panel = build_panel(players)
    cur = panel[panel["edition"] == 26]
    fc = pd.DataFrame({"id": cur["id"], "pred_t1": cur["overall"] + 1.0, "pred_t1_lo": cur["overall"] - 1.0,
                       "pred_t1_hi": cur["overall"] + 3.0, "pred_t2": cur["overall"] + 2.0,
                       "pred_t2_lo": cur["overall"] - 2.0, "pred_t2_hi": cur["overall"] + 5.0})
    return service.Store.from_frames(panel, fc)


def _json_ok(obj):
    json.dumps(obj, allow_nan=False)      # what the server does; NaN would raise
    return obj


# ------------------------------------------------------------------ service
def test_meta(store):
    m = _json_ok(service.meta(store))
    assert m["edition"] == 26 and m["next_name"] == "EA FC 27" and m["next2_name"] == "EA FC 28"
    assert m["leagues"][0]["name"] == "Premier League (England)"     # the full league comes first
    assert set(m["leagues"][0]["clubs"]) == {"Arsenal", "Chelsea", "Brentford"}
    assert m["default_league"] == "Premier League (England)"
    assert {s["slot"] for s in m["formations"]["4-3-3"]} >= {"GK", "ST", "LW", "RW"}


def test_search_folds_accents_and_ranks(store):
    hits = _json_ok(service.search(store, "jerome boa"))
    assert hits and hits[0]["name"] == "Jérôme Böateng"
    assert service.search(store, "j") == []                            # too short


def test_wonderkids_filters(store):
    r = _json_ok(service.wonderkids(store, max_age=21, max_value_m=30, groups=["MID", "FWD"]))
    assert r["players"]
    for p in r["players"]:
        assert p["age"] <= 21 and p["value"] <= 30e6 and p["pos"] in {"CDM", "CM", "CAM", "LW", "RW", "ST", "LM", "RM"}
    preds = [p["pred_t2"] for p in r["players"]]
    assert preds == sorted(preds, reverse=True)


def test_player_page(store):
    p = _json_ok(service.player(store, 7))
    assert p["name"] == "Jérôme Böateng" and len(p["history"]) == 2
    assert [f["name"] for f in p["forecast"]] == ["EA FC 27", "EA FC 28"]
    assert "ST" in p["positions_rated"] and p["similar"]
    with pytest.raises(KeyError):
        service.player(store, 999999)


def test_dream_xi_fills_every_slot_with_fitting_players(store):
    xi = _json_ok(service.dream_xi(store, max_age=40))
    assert len(xi) == 11 and len({s["player"]["id"] for s in xi}) == 11
    gk = next(s for s in xi if s["slot"] == "GK")
    assert gk["player"]["pos"] == "GK"


def test_squad_plan_is_optimal_and_within_budget(store):
    plan = _json_ok(service.squad_plan(store, "Brentford", budget_m=40, objective="now"))
    assert plan["status"] == "optimal" and len(plan["xi"]) == 11
    assert plan["spend"] <= 40e6 + 1e-6 and len(plan["buys"]) <= 3
    assert plan["xi_rating"] >= plan["baseline_xi_rating"] - 1e-9
    assert all(0 <= s["x"] <= 100 and 0 <= s["y"] <= 100 for s in plan["xi"])
    base = service.squad_plan(store, "Brentford", budget_m=0, max_signings=1, objective="now")
    assert base["buys"] == [] and abs(base["xi_rating"] - base["baseline_xi_rating"]) < 1e-9
    with pytest.raises(KeyError):
        service.squad_plan(store, "Nowhere FC")


def test_replacements_respect_caps(store):
    r = _json_ok(service.replacements(store, 7, max_fee_m=30, min_overall=60))
    assert r["target"]["id"] == 7
    for p in r["results"]:
        assert p["fee"] <= 30e6 and p["overall"] >= 60 and p["club"] != r["target"]["club"]


def test_clean_removes_nan_and_numpy():
    out = service.clean({"a": np.float64("nan"), "b": [np.int64(3), float("inf")], "c": np.bool_(True)})
    assert out == {"a": None, "b": [3, None], "c": True}


# ------------------------------------------------------------------ HTTP server
@pytest.fixture(scope="module")
def base_url(store):
    import uvicorn

    import server
    server._store = store                     # serve the synthetic league
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(server.app, host="127.0.0.1", port=port, log_level="error"))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    th.join(5)


def _get(url, data=None):
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read(), r.headers
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers


def test_http_api(base_url):
    st, body, _ = _get(f"{base_url}/api/meta")
    assert st == 200 and json.loads(body)["edition"] == 26
    st, body, _ = _get(f"{base_url}/api/wonderkids?max_age=40&max_value=100&groups=GK|DEF&leagues=La%20Liga%20(Spain)")
    ps = json.loads(body)["players"]
    assert st == 200 and ps and all(p["club"] == "Sevilla FC" for p in ps)
    st, body, _ = _get(f"{base_url}/api/squad", json.dumps({"club": "Chelsea", "budget": 25}).encode())
    assert st == 200 and json.loads(body)["status"] == "optimal"
    st, body, _ = _get(f"{base_url}/api/player/424242")
    assert st == 404 and json.loads(body)["error"] == "not_found"
    st, body, _ = _get(f"{base_url}/api/wonderkids?max_age=old")
    assert st == 400 and "max_age" in json.loads(body)["message"]
    st, body, _ = _get(f"{base_url}/api/squad", b"not json")
    assert st == 400


def test_http_site_files(base_url):
    st, body, hdr = _get(f"{base_url}/")
    assert st == 200 and b"Career Mode Scout" in body and b"js/main.js" in body
    st, _, hdr = _get(f"{base_url}/vendor/three/three.module.js")
    assert st == 200 and "max-age" in hdr.get("Cache-Control", "")
    st, _, _ = _get(f"{base_url}/js/scene/pitch.js")
    assert st == 200
