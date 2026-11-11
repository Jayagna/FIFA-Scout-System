"""Career Mode Scout - web server (JSON API + the website in web/).

    python server.py                 # http://localhost:8000
    PORT=7860 python server.py       # e.g. on Hugging Face Spaces

Needs the pipeline outputs (python scripts/run_all.py --quick):
data/processed/panel.pkl and data/processed/forecasts_latest.csv.
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fcscout import service  # noqa: E402

NEEDED = [ROOT / "data/processed/panel.pkl", ROOT / "data/processed/forecasts_latest.csv"]
_store: service.Store | None = None
_lock = threading.Lock()


def store() -> service.Store:
    global _store
    with _lock:
        if _store is None:
            missing = [p.relative_to(ROOT).as_posix() for p in NEEDED if not p.exists()]
            if missing:
                raise FileNotFoundError(", ".join(missing))
            _store = service.Store.load(ROOT)
        return _store


# ------------------------------------------------------------------ helpers
def _num(q, key, default=None, cast=float):
    v = q.get(key)
    if v in (None, ""):
        return default
    try:
        return cast(v)
    except ValueError as e:
        raise ValueError(f"'{key}' must be a number") from e


def _list(q, key):
    v = q.get(key)
    return [x for x in v.split("|") if x] if v else None


def _bool(q, key, default=False):
    v = q.get(key)
    return default if v is None else v.lower() in ("1", "true", "yes", "on")


def api(fn):
    """Run a handler; turn the usual failures into JSON errors a page can show."""
    def endpoint(request: Request):
        try:
            return JSONResponse(fn(request))
        except FileNotFoundError as e:
            return JSONResponse({"error": "not_built",
                                 "message": f"The player data and models haven't been built yet "
                                            f"(missing: {e}). Run `python scripts/run_all.py --quick` "
                                            f"from the project folder, then restart the server."},
                                status_code=503)
        except KeyError as e:
            return JSONResponse({"error": "not_found", "message": f"Nothing found for {e}."},
                                status_code=404)
        except ValueError as e:
            return JSONResponse({"error": "bad_request", "message": str(e)}, status_code=400)
    endpoint.__name__ = fn.__name__
    return endpoint


# ------------------------------------------------------------------ routes
@api
def meta(request):
    return service.meta(store())


@api
def search(request):
    q = request.query_params
    return service.search(store(), q.get("q", ""), limit=int(_num(q, "limit", 12)))


@api
def wonderkids(request):
    q = request.query_params
    return service.wonderkids(store(), max_age=int(_num(q, "max_age", 21)),
                              max_value_m=_num(q, "max_value", 15.0),
                              groups=_list(q, "groups"), leagues=_list(q, "leagues"),
                              sort=q.get("sort", "pred_t2"), limit=int(_num(q, "limit", 60)))


@api
def risers(request):
    return service.risers(store(), limit=int(_num(request.query_params, "limit", 10)))


@api
def dream_xi(request):
    return service.dream_xi(store(), max_age=int(_num(request.query_params, "max_age", 21)))


@api
def player(request):
    return service.player(store(), int(request.path_params["pid"]))


@api
def club(request):
    return service.club(store(), request.query_params.get("name", ""))


@api
def replacements(request):
    q = request.query_params
    return service.replacements(store(), int(_num(q, "id", cast=int)),
                                max_fee_m=_num(q, "max_fee"), max_age=_num(q, "max_age", cast=int),
                                min_overall=_num(q, "min_overall", cast=int),
                                same_position=_bool(q, "same_position", True),
                                leagues=_list(q, "leagues"))


@api
def rumours(request):
    return service.rumours(store(), request.query_params.get("club", ""))


@api
def report(request):
    return service.report(store())


async def squad(request: Request):
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return JSONResponse({"error": "bad_request", "message": "Send the settings as JSON."}, 400)
    from starlette.concurrency import run_in_threadpool

    def solve():
        return service.squad_plan(
            store(), body.get("club", ""), formation=body.get("formation", "4-3-3"),
            budget_m=float(body.get("budget", 50)), wage_budget_k=body.get("wage_budget"),
            max_signings=int(body.get("max_signings", 3)), max_age=body.get("max_age"),
            objective=body.get("objective", "balanced"), allow_sales=bool(body.get("allow_sales")),
            max_sales=int(body.get("max_sales", 3)), markup=float(body.get("markup", 1.2)),
            bench_weight=float(body.get("bench_weight", service.BENCH_WEIGHT)),
            market_leagues=body.get("market_leagues") or None,
            realistic=bool(body.get("realistic", True)), alternatives=int(body.get("alternatives", 2)),
            seed=body.get("seed"), ban_ids=[int(i) for i in body.get("ban_ids") or []],
            lock_ids=[int(i) for i in body.get("lock_ids") or []], rumours=bool(body.get("rumours")))
    return await run_in_threadpool(api(lambda _: solve()), request)


class WebFiles(StaticFiles):
    """The site: long cache for fonts and the 3D library, revalidate everything else."""
    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        if path.startswith(("fonts/", "vendor/")):
            resp.headers["Cache-Control"] = "public, max-age=604800"
        else:
            resp.headers["Cache-Control"] = "no-cache"
        return resp


routes = [
    Route("/api/meta", meta), Route("/api/search", search), Route("/api/wonderkids", wonderkids),
    Route("/api/risers", risers), Route("/api/dream-xi", dream_xi), Route("/api/player/{pid:int}", player), Route("/api/club", club),
    Route("/api/replacements", replacements), Route("/api/report", report),
    Route("/api/rumours", rumours),
    Route("/api/squad", squad, methods=["POST"]),
    Mount("/", WebFiles(directory=ROOT / "web", html=True), name="web"),
]
app = Starlette(routes=routes, middleware=[Middleware(GZipMiddleware, minimum_size=800)])


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0" if "PORT" in os.environ else "127.0.0.1")
    print(f"Career Mode Scout on http://{'localhost' if host == '127.0.0.1' else host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
