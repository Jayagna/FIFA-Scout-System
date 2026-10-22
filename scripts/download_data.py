"""Download the raw FIFA 15-22 player files into data/raw/.

Option A (default, no account needed): fetch the exact files this project was
built on from public GitHub mirrors of the Kaggle dataset, pinned to commits.

Option B: download "FIFA 22 complete player dataset" by stefanoleone992 from
Kaggle and copy players_15.csv ... players_22.csv into data/raw/. The loader
handles that layout too (it prefers players_21.csv over the sofifa export).

Usage:  python scripts/download_data.py
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

# (repo, commit, path in repo, local name, md5)
SOURCES = [
    ("tmclouisluk/FIFA-players-Data-Analysis", "959c7fcf0a6587acdb094de4c6b9561c4e013e68",
     [(f"data/players_{y}.csv", f"players_{y}.csv") for y in range(15, 21)]),
    ("Ruzsel/Fifa-21-Python-Data-Cleaning", "4577e09ca2c323b454ab3d2dc880eab738750131",
     [("Fifa21_fix.csv", "players_21_sofifa_export.csv")]),
    ("abineshta/FIFA-22-complete-player-dataset-EDA", "4f610dc59f001a79a148c1e5f6a082d304c16d7d",
     [("players_22.csv", "players_22.csv")]),
]

MD5 = {
    "players_15.csv": "2c0a446f9e8a6527d0e981a465b68014",
    "players_16.csv": "52b7c14ba57591fd7db13cdfcc5d7c8d",
    "players_17.csv": "0ee46fe7a33cb03c134dc352b1debc48",
    "players_18.csv": "065a4fa65383427da7a5b8685e99367e",
    "players_19.csv": "25bc867ae6c580021f07f410a412c635",
    "players_20.csv": "90eafb809be29307eb1e3ede2e45949f",
    "players_21_sofifa_export.csv": "fc51ec537f65732f427f8079b49abe3f",
    "players_22.csv": "d4e790640f118f43491548c166fbc5e6",
}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True, stdout=subprocess.DEVNULL)


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    todo = [(r, c, files) for r, c, files in SOURCES
            if any(not (RAW / local).exists() for _, local in files)]
    if not todo:
        print("All raw files already present.")
    for repo, commit, files in todo:
        print(f"Fetching {repo} @ {commit[:7]} ...")
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            run(["git", "-c", "gc.auto=0", "clone", "-q", "--filter=blob:none", "--no-checkout",
                 f"https://github.com/{repo}", str(tmp / "repo")])
            run(["git", "-c", "gc.auto=0", "checkout", "-q", commit, "--", *[p for p, _ in files]],
                cwd=tmp / "repo")
            for path, local in files:
                shutil.copy(tmp / "repo" / path, RAW / local)
    bad = 0
    for name, expected in MD5.items():
        p = RAW / name
        if not p.exists():
            print(f"  missing  {name}")
            bad += 1
        elif md5(p) != expected:
            print(f"  CHANGED  {name} (checksum differs from the build used for the README numbers)")
        else:
            print(f"  ok       {name}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
