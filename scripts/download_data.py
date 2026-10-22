"""Download the raw FIFA 15-22 player files into data/raw/.

Default: plain HTTPS downloads of the exact files this project was built on, from
public GitHub mirrors of the Kaggle dataset, pinned to commits and checked by MD5.
No git or Kaggle account needed.

Alternative: download "FIFA 22 complete player dataset" by stefanoleone992 from
Kaggle and copy players_15.csv ... players_22.csv into data/raw/. The loader
handles that layout too (it prefers players_21.csv over the sofifa export).

Usage:  python scripts/download_data.py
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
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


def fetch(url: str, dest: Path, tries: int = 3) -> None:
    """Download url to dest atomically (via a .part file), retrying on network errors."""
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


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    failed = []
    for repo, commit, files in SOURCES:
        for path, local in files:
            dest = RAW / local
            if dest.exists() and md5(dest) == MD5[local]:
                print(f"  ok       {local}")
                continue
            url = f"{BASE}/{repo}/{commit}/{path}"
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
    if failed:
        print(f"\n{len(failed)} file(s) could not be downloaded: {', '.join(failed)}\n"
              "Check your internet connection and run this script again, or put the Kaggle "
              "files in data/raw/ by hand (see the docstring at the top of this script).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
