"""Checks for the replacement finder. Writes reports/similarity_metrics.json.

1. Identity recall: embed each player's FIFA 21 attributes and search the FIFA 22
   pool. If the vectors capture "what kind of player this is", the nearest FIFA 22
   player should usually be the same person a year later.
2. Position agreement: share of a player's 10 nearest neighbours that play his
   primary position (versus the share expected by chance).
3. Cheaper look-alikes: for every Premier League player rated 80+, is there a
   same-position player within 3 rating points at <= 50% of his value?
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fcscout.similar import SimilarityIndex  # noqa: E402


def main():
    panel = pd.read_pickle(ROOT / "data/processed/panel.pkl")
    cur = panel[panel["edition"] == 22].reset_index(drop=True)
    prev = panel[panel["edition"] == 21].reset_index(drop=True)
    idx = SimilarityIndex(cur)
    out = {"faiss_used": idx.use_faiss}

    # 1. identity recall across editions
    rec = {}
    for block, gk in (("outfield", False), ("gk", True)):
        q = prev[prev["primary_pos"].astype(str).eq("GK") == gk]
        q = q[q["id"].isin(cur["id"])]
        Q = idx.embed(q, block)
        _, rows = idx.search_vectors(Q, block, k=10)
        found = cur["id"].to_numpy()[rows] == q["id"].to_numpy()[:, None]
        rec[block] = {"n": int(len(q)), "recall@1": float(found[:, 0].mean()),
                      "recall@10": float(found.any(axis=1).mean())}
    out["identity_recall_21_to_22"] = rec

    # 2. position agreement
    rng = np.random.default_rng(0)
    outf = np.where(~idx.is_gk)[0]
    sample = rng.choice(outf, 2000, replace=False)
    b = idx.blocks["outfield"]
    Q = b["Z"][np.searchsorted(b["idx"], sample)]
    _, rows = idx.search_vectors(Q, "outfield", k=11)
    rows = rows[:, 1:]  # drop self
    pos = cur["primary_pos"].astype(str).to_numpy()
    agree = (pos[rows] == pos[sample][:, None]).mean()
    share = pd.Series(pos[outf]).value_counts(normalize=True)
    chance = float((share ** 2).sum())
    out["position_agreement@10"] = {"model": float(agree), "chance": chance}

    # 3. cheaper look-alikes for Premier League 80+ players
    stars = cur[(cur["league"] == "English Premier League") & (cur["overall"] >= 80)]
    res = []
    for r in stars.itertuples():
        m = idx.find(r.id, max_fee=0.5 * r.value_eur, min_overall=r.overall - 3, k=1,
                     exclude_club=r.club)
        if len(m):
            res.append(dict(found=True, sim=float(m["similarity"].iloc[0]),
                            drop=float(r.overall - m["overall"].iloc[0]),
                            saving_pct=float(1 - m["value_eur"].iloc[0] / r.value_eur)))
        else:
            res.append(dict(found=False))
    res = pd.DataFrame(res)
    f = res[res["found"]]
    out["cheaper_lookalikes_pl_80plus"] = {
        "players": int(len(res)), "found_share": float(res["found"].mean()),
        "avg_similarity": float(f["sim"].mean()), "avg_rating_drop": float(f["drop"].mean()),
        "avg_saving_pct": float(f["saving_pct"].mean()),
    }
    (ROOT / "reports/similarity_metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
