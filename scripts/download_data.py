"""Download the raw player files (FIFA 15 - EA FC 26) into data/raw/.

Default: plain HTTPS downloads of the exact files this project was built on, from
public GitHub repositories, pinned to commits and checked by MD5. No git or Kaggle
account needed.

  edition   source (layout)                                         players
  15-20     Kaggle "FIFA 20 complete player dataset" (Kaggle)       15-18k each
  21        cleaned sofifa web export (sofifa export)               19.0k
  22        Kaggle "FIFA 22 complete player dataset" (Kaggle)       19.2k
  23        sofifa export with position families (sofifa v2)        18.4k
  24        Playwright sofifa scrape (sofifa scrape)                18.3k
  25        the game's own database tables (EA database)            16.9k men's club players
  26        Kaggle "FC 26 player data" by rovnez (Kaggle layout)    18.4k

Snapshot timing differs by edition: FIFA 23, FC 24 and FC 26 are launch rosters, the
FC 25 database is a late-season roster (mid-2025). See the README.

Alternative: put any edition in the Kaggle layout in data/raw/ as players_XX.csv
(e.g. from Kaggle) and it takes priority over the files below.

Usage:  python scripts/download_data.py
        python scripts/download_data.py --only players_26.csv    # just some files
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
# overridable for testing or for a local mirror
BASE = os.environ.get("FCSCOUT_RAW_BASE", "https://raw.githubusercontent.com")

# (repo, commit, [(path in repo, local name)])
SOURCES = [
    ("tmclouisluk/FIFA-players-Data-Analysis", "959c7fcf0a6587acdb094de4c6b9561c4e013e68",
     [(f"data/players_{y}.csv", f"players_{y}.csv") for y in range(15, 21)]),
    ("Ruzsel/Fifa-21-Python-Data-Cleaning", "4577e09ca2c323b454ab3d2dc880eab738750131",
     [("Fifa21_fix.csv", "players_21_sofifa_export.csv")]),
    ("abineshta/FIFA-22-complete-player-dataset-EDA", "4f610dc59f001a79a148c1e5f6a082d304c16d7d",
     [("players_22.csv", "players_22.csv")]),
    ("MithileshAnavkar/FOOTEM", "d361b4fe145054d973b0b6f0dcedbab9cbdb9a5a",
     [("Data/players_fifa23.csv", "players_23_sofifa_v2.csv")]),
    ("stephensoccer/EA-Sports-FC-24-Players-Data", "fa16d9bd06b5afd9c56fc8a1f282ece855a9edde",
     [("FC 24 Players Data.csv", "players_24_sofifa_scrape.csv")]),
    ("Radzim/football_data", "076dbb1e76413c93582385d8e56f18384dc6bee9",
     [(f"{t}.csv", f"ea_db_25/{t}.csv") for t in ("players", "teams", "leagues", "teamplayerlinks",
                                                   "leagueteamlinks", "countries", "sofifa", "models")]),
    ("ismailoksuz/EAFC26-DataHub", "8724af5803a787c9c262eee2ac8f942651b0e4b9",
     [("data/players.csv", "players_26.csv")]),
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
    "players_23_sofifa_v2.csv": "5f5b004fad76be0bb444177db60afe3e",
    "players_24_sofifa_scrape.csv": "8470ab5c8a3653668e055956696a8297",
    "ea_db_25/players.csv": "623f4367a2322d976f73cb5c08f467bf",
    "ea_db_25/teams.csv": "92d5bd242b945af56c7fc4c83c596501",
    "ea_db_25/leagues.csv": "4fd2e00df3e26e26fe33a30b7e5512cc",
    "ea_db_25/teamplayerlinks.csv": "874852e09324db4acc0d7aa9de585902",
    "ea_db_25/leagueteamlinks.csv": "d3fee592af247564099ea4b7b4ebaeaf",
    "ea_db_25/countries.csv": "9828e4388cc5ec368612fdfef9c8a80d",
    "ea_db_25/sofifa.csv": "bb2cdb140716d86fcb5e8523c39750f5",
    "ea_db_25/models.csv": "e2e1f5f429912a94a7152365610c7208",
    "players_26.csv": "764f4c62d94cdefa5fc99d41f6bee08f",
}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, tries: int = 3) -> None:
    """Download url to dest atomically (via a .part file), retrying on network errors."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, tries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "fc-career-scout"})
            with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            part.replace(dest)
            return
        except Exception as e:  # noqa: BLE001 - report and retry any network error
            part.unlink(missing_ok=True)
            if attempt == tries:
                raise RuntimeError(f"could not download {url}: {e}") from e
            print(f"    retrying ({e})")
            time.sleep(2 * attempt)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="download only these local file names")
    only = set(ap.parse_args(argv).only or [])
    RAW.mkdir(parents=True, exist_ok=True)
    failed, provenance = [], []
    for repo, commit, files in SOURCES:
        for path, local in files:
            if only and local not in only:
                continue
            dest = RAW / local
            url = f"{BASE}/{repo}/{commit}/{urllib.parse.quote(path)}"
            provenance.append({"file": local, "repo": repo, "commit": commit, "path": path})
            if dest.exists() and md5(dest) == MD5[local]:
                print(f"  ok       {local}")
                continue
            print(f"  fetching {local} ...", flush=True)
            try:
                fetch(url, dest)
            except RuntimeError as e:
                print(f"  FAILED   {e}")
                failed.append(local)
                continue
            if md5(dest) != MD5[local]:
                print(f"  CHANGED  {local}: checksum differs from the build used for the README numbers")
            else:
                print(f"  ok       {local}")
    if not only:
        (ROOT / "reports").mkdir(exist_ok=True)
        (ROOT / "reports/data_sources.json").write_text(json.dumps(provenance, indent=1), encoding="utf-8")
    if failed:
        print(f"\n{len(failed)} file(s) could not be downloaded: {', '.join(failed)}\n"
              "Check your internet connection and run this script again, or put the files "
              "in data/raw/ by hand (see the docstring at the top of this script).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
