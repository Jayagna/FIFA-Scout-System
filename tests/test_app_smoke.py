"""Smoke test: execute app.py top to bottom with a stand-in for streamlit.

It checks the app's own logic (data loading, optimiser call, finder, tables) without
a browser. Needs the pipeline outputs. Run:  python tests/test_app_smoke.py
"""
from __future__ import annotations

import runpy
import sys
import types
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class _Widgets:
    def __init__(self, log, overrides):
        self.log = log
        self.overrides = overrides

    def _pick(self, label, default):
        return self.overrides.get(label, default)

    def selectbox(self, label, options, index=0, format_func=None, key=None, help=None, **kw):
        options = list(options)
        return self._pick(label, options[index] if options else None)

    def multiselect(self, label, options, default=None, key=None, **kw):
        return self._pick(label, list(default or []))

    def slider(self, label, min_value=None, max_value=None, value=None, step=None, key=None, **kw):
        return self._pick(label, value)

    def number_input(self, label, min_value=None, max_value=None, value=None, step=None, **kw):
        return self._pick(label, value)

    def radio(self, label, options, **kw):
        return self._pick(label, list(options)[0])

    def checkbox(self, label, value=False, **kw):
        return self._pick(label, value)

    def button(self, label, **kw):
        return True

    def metric(self, label, value, delta=None, **kw):
        self.log.append(("metric", label, value, delta))

    def dataframe(self, df, **kw):
        assert len(df.columns) > 0
        self.log.append(("dataframe", df.shape))

    def markdown(self, *a, **kw):
        pass

    write = caption = subheader = title = warning = markdown

    def info(self, msg, *a, **kw):
        self.log.append(("info", msg))

    def error(self, msg, *a, **kw):
        self.log.append(("error", msg))

    def line_chart(self, data, **kw):
        assert len(data) > 0
        self.log.append(("chart", data.shape))

    @contextmanager
    def spinner(self, *a, **kw):
        yield

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fake_streamlit(log, overrides):
    st = types.ModuleType("streamlit")
    w = _Widgets(log, overrides)
    for name in dir(w):
        if not name.startswith("_"):
            setattr(st, name, getattr(w, name))
    st.set_page_config = lambda **kw: None

    def cache(*args, **kw):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f
    st.cache_data = st.cache_resource = cache
    st.tabs = lambda names: [_Widgets(log, overrides) for _ in names]
    st.columns = lambda spec: [_Widgets(log, overrides) for _ in range(spec if isinstance(spec, int) else len(spec))]
    return st


def run(overrides=None):
    log = []
    sys.modules["streamlit"] = _fake_streamlit(log, overrides or {})
    try:
        runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
    finally:
        sys.modules.pop("streamlit", None)
    return log


def test_app_runs_default():
    log = run()
    assert not [e for e in log if e[0] == "error"], log
    # wonderkids, XI, buys, and the replacement table (or its "nobody matches" note)
    assert sum(e[0] in ("dataframe", "info") for e in log) >= 4
    assert any(e[0] == "metric" and e[1].startswith("XI rating") for e in log)


def test_app_runs_with_sales_and_current_ratings():
    log = run({"Allow sales to fund signings": True, "Optimise for": "Current ratings",
               "Formation": "4-2-3-1", "Your club": "Brentford", "Max age of signings": 26})
    assert not [e for e in log if e[0] == "error"], log
    assert any(e[0] == "metric" and e[1] == "XI rating" for e in log)


if __name__ == "__main__":
    for t in (test_app_runs_default, test_app_runs_with_sales_and_current_ratings):
        t()
        print("ok   ", t.__name__)
