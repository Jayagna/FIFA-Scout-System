"""data/raw/*.csv  ->  data/processed/players.pkl (long table) + panel.pkl (features/targets)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fcscout.data import load_raw_dir  # noqa: E402
from fcscout.features import build_panel  # noqa: E402


def main():
    players = load_raw_dir(ROOT / "data/raw")
    panel = build_panel(players)
    out = ROOT / "data/processed"
    out.mkdir(parents=True, exist_ok=True)
    players.to_pickle(out / "players.pkl")
    panel.to_pickle(out / "panel.pkl")
    counts = players.groupby("edition").size()
    print("players per edition:", counts.to_dict())
    for h in (1, 2):
        col = f"overall_t{h}"
        print(f"rows with a {h}-edition-ahead target:", int(panel[col].notna().sum()))


if __name__ == "__main__":
    main()
