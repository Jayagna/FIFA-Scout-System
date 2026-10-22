"""Replacement finder: players with a similar attribute profile, under a price cap.

Each player is a vector of his detailed attributes (29 outfield attributes, or the
goalkeeping ones for keepers), z-scored over the pool and L2-normalised, so the
inner product is cosine similarity. Search is exact (FAISS IndexFlatIP when
faiss is installed, otherwise the same computation in NumPy - identical results).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import GK_ATTRS, OUTFIELD_ATTRS

GK_PROFILE = GK_ATTRS + ["movement_reactions", "attacking_short_passing", "skill_long_passing",
                         "power_jumping", "power_strength", "mentality_composure"]

try:  # optional dependency
    import faiss  # type: ignore
    HAVE_FAISS = True
except Exception:  # pragma: no cover - depends on the environment
    faiss = None
    HAVE_FAISS = False


class SimilarityIndex:
    def __init__(self, players: pd.DataFrame, use_faiss: bool | None = None):
        self.players = players.reset_index(drop=True)
        self.use_faiss = HAVE_FAISS if use_faiss is None else (use_faiss and HAVE_FAISS)
        self.is_gk = self.players["primary_pos"].astype(str).eq("GK").to_numpy()
        self.blocks = {}
        for name, mask, cols in (("outfield", ~self.is_gk, OUTFIELD_ATTRS), ("gk", self.is_gk, GK_PROFILE)):
            idx = np.where(mask)[0]
            X = self.players.iloc[idx][cols].astype(float)
            mu, sd = X.mean(), X.std().replace(0, 1)
            Z = np.array(((X - mu) / sd).fillna(0), dtype=np.float32)
            Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
            index = None
            if self.use_faiss:
                index = faiss.IndexFlatIP(Z.shape[1])
                index.add(Z)
            self.blocks[name] = dict(idx=idx, Z=Z, mu=mu, sd=sd, cols=cols, index=index)

    def _block(self, row_pos: int):
        return self.blocks["gk" if self.is_gk[row_pos] else "outfield"]

    def embed(self, rows: pd.DataFrame, block: str) -> np.ndarray:
        b = self.blocks[block]
        Z = np.array(((rows[b["cols"]].astype(float) - b["mu"]) / b["sd"]).fillna(0), dtype=np.float32)
        return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)

    def search_vectors(self, Q: np.ndarray, block: str, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Top-k (similarities, row positions in self.players) for query vectors Q."""
        b = self.blocks[block]
        k = min(k, len(b["idx"]))
        if b["index"] is not None:
            sims, loc = b["index"].search(np.ascontiguousarray(Q, dtype=np.float32), k)
        else:
            S = Q @ b["Z"].T
            loc = np.argpartition(-S, kth=k - 1, axis=1)[:, :k]
            part = np.take_along_axis(S, loc, axis=1)
            order = np.argsort(-part, axis=1, kind="stable")
            loc = np.take_along_axis(loc, order, axis=1)
            sims = np.take_along_axis(part, order, axis=1)
        return sims, b["idx"][loc]

    def find(self, player_id: int, max_fee: float | None = None, max_age: int | None = None,
             min_overall: int | None = None, same_position: bool = True,
             leagues: list[str] | None = None, exclude_club: str | None = None,
             k: int = 10, markup: float = 1.0) -> pd.DataFrame:
        pos = np.where(self.players["id"].to_numpy() == player_id)[0]
        if len(pos) == 0:
            raise KeyError(f"player {player_id} not in index")
        p = int(pos[0])
        b = self._block(p)
        q = b["Z"][np.searchsorted(b["idx"], p)][None, :]
        sims, rows = self.search_vectors(q, "gk" if self.is_gk[p] else "outfield", k=len(b["idx"]))
        res = self.players.iloc[rows[0]].copy()
        res["similarity"] = sims[0]
        res["fee"] = res["value_eur"] * markup
        target = self.players.iloc[p]
        m = res["id"].ne(player_id) & res["club"].notna() & res["value_eur"].gt(0)
        if "on_loan" in res:
            m &= res["on_loan"].fillna(0).eq(0)
        if same_position:
            m &= res["primary_pos"].eq(target["primary_pos"])
        if max_fee is not None:
            m &= res["fee"].le(max_fee)
        if max_age is not None:
            m &= res["age"].le(max_age)
        if min_overall is not None:
            m &= res["overall"].ge(min_overall)
        if leagues:
            m &= res["league"].isin(leagues)
        if exclude_club:
            m &= res["club"].ne(exclude_club)
        out = res[m].head(k)
        out["saving"] = target["value_eur"] * markup - out["fee"]
        return out
