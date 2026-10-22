"""Static figures for the README (reports/figures/*.png)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"model": ("Growth model", "#2a78d6"),
          "ea_potential": ("EA-potential baseline", "#eb6834"),
          "persistence": ("No change", "#1baf7a")}


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    m = json.loads((ROOT / "reports/forecast_metrics.json").read_text())["h2"]
    groups = ["<=21", "22-25", "26-29", "30+"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [3, 2]})
    ax = axes[0]
    style(ax)
    y = np.arange(len(groups))
    h = 0.26
    for i, key in enumerate(SERIES):
        label, color = SERIES[key]
        vals = [m["by_age"][g][key] for g in groups]
        ax.barh(y + (i - 1) * h, vals, height=h - 0.03, color=color, label=label)
        for yy, v in zip(y + (i - 1) * h, vals):
            ax.text(v + 0.04, yy, f"{v:.2f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, [f"age {g}" for g in groups], color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("Mean absolute error, rating points (lower is better)", color=INK2)
    ax.set_title("Rating two editions ahead (FIFA 20 -> FIFA 22, held out)", loc="left",
                 color=INK, fontsize=11)
    ax.legend(frameon=False, loc="lower right", fontsize=8, labelcolor=INK2)

    ax = axes[1]
    style(ax)
    yr = m["young_ranking"]
    rows = [("Growth model", yr["model"]["precision_top10pct_growth@100"], "#2a78d6"),
            ("EA potential - overall", yr["ea_gap"]["precision_top10pct_growth@100"], "#eb6834"),
            ("Random pick", yr["_base_rate"], "#8a8984")]
    for i, (lab, v, c) in enumerate(rows):
        ax.barh(i, v * 100, height=0.55, color=c)
        ax.text(v * 100 + 1, i, f"{v * 100:.0f}%", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(3), [r[0] for r in rows], color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 60)
    ax.set_xlabel("Top-100 U23 picks that became top-10% risers", color=INK2)
    ax.set_title(f"Finding the risers ({yr['_n_u23']:,} U23 players)", loc="left", color=INK,
                 fontsize=11)
    fig.tight_layout()
    out = ROOT / "reports/figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "forecast_backtest.png", dpi=160, facecolor=SURFACE)
    print("wrote", out / "forecast_backtest.png")


if __name__ == "__main__":
    main()
