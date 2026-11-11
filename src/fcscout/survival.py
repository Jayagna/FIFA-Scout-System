"""Will this player still be in the game h editions from now?

The growth forecaster is trained only on players who appear again, so on its own it
never learns that a 37-year-old usually retires or leaves for an unlicensed league.
This classifier fills that gap: P(player appears in edition t+h), from the same
features. The squad builder multiplies it into each player's value (see
fcscout.optimize: expected rating = p * rating + (1 - p) * the backup's rating).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .features import CAT_FEATURES, FEATURES


def stays_col(h: int) -> str:
    return f"stays_t{h}"


def add_stay_targets(panel: pd.DataFrame, horizons=(1, 2)) -> pd.DataFrame:
    """stays_t{h} = 1 if the player is in edition t+h, 0 if not, NaN when edition t+h
    is not in the data yet."""
    df = panel.copy()
    last = int(df["edition"].max())
    for h in horizons:
        present = df[f"overall_t{h}"].notna().astype(float)
        df[stays_col(h)] = present.where(df["edition"] + h <= last)
    return df


@dataclass
class SurvivalModel:
    h: int
    params: dict = field(default_factory=dict)
    categories: dict = field(default_factory=dict)
    features: list = field(default_factory=list)

    def _X(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.features].copy()
        for c in CAT_FEATURES:
            v = X[c].astype(str)
            v = v.where(v.isin(self.categories[c]), "nan")
            X[c] = pd.Categorical(v, categories=self.categories[c])
        return X

    def fit(self, df: pd.DataFrame):
        df = df[df[stays_col(self.h)].notna()]
        self.categories = {c: sorted(df[c].dropna().astype(str).unique().tolist()) + ["nan"]
                           for c in CAT_FEATURES}
        self.features = [c for c in FEATURES if c in CAT_FEATURES or df[c].nunique(dropna=True) > 1]
        base = dict(learning_rate=0.05, max_iter=400, max_leaf_nodes=31, min_samples_leaf=60,
                    l2_regularization=1.0, categorical_features="from_dtype", random_state=0)
        base.update(self.params)
        self.model = HistGradientBoostingClassifier(**base).fit(self._X(df), df[stays_col(self.h)].astype(int))
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._X(df))[:, 1]


def age_baseline(train: pd.DataFrame, test: pd.DataFrame, h: int) -> np.ndarray:
    """Share of players of the same age (and rating band) who stayed, in the training data."""
    def key(d):
        return d["age"].clip(16, 40).astype(int).astype(str) + "|" + (d["overall"] // 5).astype(int).astype(str)
    rate = train.groupby(key(train))[stays_col(h)].mean()
    by_age = train.groupby(train["age"].clip(16, 40).astype(int))[stays_col(h)].mean()
    p = key(test).map(rate)
    p = p.fillna(test["age"].clip(16, 40).astype(int).map(by_age))
    return p.fillna(train[stays_col(h)].mean()).to_numpy()


def scores(y: np.ndarray, p: np.ndarray) -> dict:
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return {"auc": float(roc_auc_score(y, p)), "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p))}
