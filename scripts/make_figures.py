"""Static figure for the README (reports/figures/forecast_backtest.png).

Left and middle: rolling-origin error, one point per test window, for the growth model
and the two baselines (each window's model only saw earlier editions). Right: how
many of each method's top-100 young picks became top-10% risers, two editions ahead.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# categorical slots 1-3 of the validated light palette (dataviz reference instance)
SERIES = {"model": ("Growth model", "#2a78d6"),
          "ea_potential": ("EA-potential baseline", "#eb6834"),
          "persistence": ("No change", "#1baf7a")}
NAMES = {e: str(e) for e in range(15, 30)}      # 15-23 = FIFA, 24+ = EA FC


def style(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, length=0, labelsize=8)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def rolling_panel(ax, windows, h):
    style(ax)
    x = list(range(len(windows)))
    labels = [f"{NAMES[w['test_edition']]}→{NAMES[w['target_edition']]}" for w in windows]
    for key, (label, color) in SERIES.items():
        ys = [w[key] for w in windows]
        ax.plot(x, ys, color=color, linewidth=2, marker="o", markersize=4.5, label=label,
                markeredgecolor=SURFACE, markeredgewidth=1.2)
        ax.text(x[-1] + 0.15, ys[-1], f"{ys[-1]:.2f}", color=INK2, fontsize=7.5, va="center")
    ax.set_xticks(x, labels, rotation=0)
    ax.set_xlim(-0.3, len(x) - 0.3)
    wins = sum(w["model"] < min(w["ea_potential"], w["persistence"]) for w in windows)
    ax.set_title(f"{h} edition{'s' if h > 1 else ''} ahead: model best in {wins} of {len(windows)} windows",
                 loc="left", color=INK, fontsize=10)
    ax.set_ylabel("Mean absolute error (rating points)", color=INK2, fontsize=8)
    ax.set_xlabel("Test window, edition → target (15–23 FIFA, 24–26 EA FC)", color=INK2, fontsize=8)


def main():
    roll = json.loads((ROOT / "reports/forecast_rolling.json").read_text())
    m = json.loads((ROOT / "reports/forecast_metrics.json").read_text())["h2"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1.35, 1.2, 1]})
    rolling_panel(axes[0], roll["h1"], 1)
    rolling_panel(axes[1], roll["h2"], 2)
    axes[0].legend(frameon=False, loc="lower left", fontsize=8, labelcolor=INK2)

    ax = axes[2]
    style(ax, "x")
    yr = m["young_ranking"]
    rows = [("Growth model", yr["model"]["precision_top10pct_growth@100"], SERIES["model"][1]),
            ("EA potential - overall", yr["ea_gap"]["precision_top10pct_growth@100"], SERIES["ea_potential"][1]),
            ("Random pick", yr["_base_rate"], "#8a8984")]
    for i, (lab, v, c) in enumerate(rows):
        ax.barh(i, v * 100, height=0.55, color=c)
        ax.text(v * 100 + 1, i, f"{v * 100:.0f}%", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(3), [r[0] for r in rows], color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, max(r[1] for r in rows) * 100 + 12)
    ax.set_xlabel("Top-100 U23 picks that became top-10% risers", color=INK2, fontsize=8)
    t = m["test_edition"]
    ax.set_title(f"Finding the risers, FC {t}→FC {t + 2} ({yr['_n_u23']:,} U23s)", loc="left",
                 color=INK, fontsize=10)
    fig.tight_layout()
    out = ROOT / "reports/figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "forecast_backtest.png", dpi=150, facecolor=SURFACE)
    print("wrote", out / "forecast_backtest.png")


if __name__ == "__main__":
    main()
