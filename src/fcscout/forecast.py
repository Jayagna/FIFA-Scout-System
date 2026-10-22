"""Rating growth forecaster: predict a player's overall rating 1 or 2 editions ahead.

The model predicts the *change* in overall (target - current) with gradient
boosting. It is evaluated with an edition-based split, never a random one: the
test edition's targets come from a game that was not available when the
training targets were.

Baselines, all fitted on the same training rows:
  persistence   next rating = current rating
  age_curve     current + mean change for that age
  ea_potential  current + linear model on EA's potential gap x age bucket
                (the strongest simple use of EA's own "Potential" number)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from .features import CAT_FEATURES, FEATURES

AGE_BUCKETS = [0, 19, 21, 23, 25, 27, 29, 31, 99]


def target_col(h: int) -> str:
    return f"overall_t{h}"


def rows_for(panel: pd.DataFrame, editions, h: int) -> pd.DataFrame:
    sub = panel[panel["edition"].isin(list(editions)) & panel[target_col(h)].notna()]
    return sub.reset_index(drop=True)


def _age_bucket(age: pd.Series) -> pd.Series:
    return pd.cut(age, AGE_BUCKETS, right=True, labels=False)


# ---------------------------------------------------------------- baselines
class AgeCurve:
    def fit(self, df: pd.DataFrame, y: pd.Series):
        a = df["age"].clip(16, 38)
        self.table = y.groupby(a).mean()
        self.default = float(y.mean())
        return self

    def predict_delta(self, df: pd.DataFrame) -> np.ndarray:
        a = df["age"].clip(16, 38)
        return a.map(self.table).fillna(self.default).to_numpy()


class PotentialLinear:
    """delta ~ sum_b 1[age in b] * (alpha_b + beta_b * gap): EA's potential, used well."""

    def _design(self, df: pd.DataFrame) -> np.ndarray:
        b = _age_bucket(df["age"]).fillna(0).astype(int).to_numpy()
        n_b = len(AGE_BUCKETS) - 1
        onehot = np.eye(n_b)[b]
        gap = df["gap"].to_numpy()[:, None]
        return np.hstack([onehot, onehot * gap])

    def fit(self, df: pd.DataFrame, y: pd.Series):
        self.lr = LinearRegression(fit_intercept=False).fit(self._design(df), y.to_numpy())
        return self

    def predict_delta(self, df: pd.DataFrame) -> np.ndarray:
        return self.lr.predict(self._design(df))


# ---------------------------------------------------------------- model
@dataclass
class GrowthModel:
    h: int
    params: dict = field(default_factory=dict)
    categories: dict = field(default_factory=dict)
    features: list = field(default_factory=list)

    def _X(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.features].copy()
        for c in CAT_FEATURES:
            v = X[c].astype(str)
            v = v.where(v.isin(self.categories[c]), "nan")   # unseen category -> missing
            X[c] = pd.Categorical(v, categories=self.categories[c])
        return X

    def fit(self, df: pd.DataFrame, intervals: bool = True):
        self.categories = {c: sorted(df[c].dropna().astype(str).unique().tolist()) + ["nan"]
                           for c in CAT_FEATURES}
        # drop columns that carry no information in these editions (e.g. composure
        # did not exist before FIFA 17)
        self.features = [c for c in FEATURES if c in CAT_FEATURES or df[c].nunique(dropna=True) > 1]
        y = df[target_col(self.h)] - df["overall"]
        X = self._X(df)
        base = dict(learning_rate=0.05, max_iter=600, max_leaf_nodes=31, min_samples_leaf=40,
                    l2_regularization=1.0, categorical_features="from_dtype", random_state=0)
        base.update(self.params)
        # absolute-error loss = median forecast; it was clearly better on the
        # validation editions because year-to-year rating shifts are skewed
        self.model = HistGradientBoostingRegressor(loss="absolute_error", **base).fit(X, y)
        self.q10 = self.q90 = None
        if intervals:
            qbase = {**base, "max_iter": min(base["max_iter"], 400)}
            self.q10 = HistGradientBoostingRegressor(loss="quantile", quantile=0.1, **qbase).fit(X, y)
            self.q90 = HistGradientBoostingRegressor(loss="quantile", quantile=0.9, **qbase).fit(X, y)
        return self

    def predict_delta(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self._X(df))

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._X(df)
        cur = df["overall"].to_numpy()
        mid = cur + self.model.predict(X)
        lo = cur + self.q10.predict(X)
        hi = cur + self.q90.predict(X)
        lo, hi = np.minimum(lo, mid), np.maximum(hi, mid)
        return pd.DataFrame({"pred": mid, "pred_lo": lo, "pred_hi": hi}, index=df.index)


# ---------------------------------------------------------------- evaluation
def mae(a, b) -> float:
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def evaluate(test: pd.DataFrame, preds: dict[str, np.ndarray], h: int) -> dict:
    """preds: name -> predicted overall at t+h (levels, not deltas)."""
    y = test[target_col(h)].to_numpy()
    out = {"n": int(len(test)), "overall": {}, "by_age": {}, "young_ranking": {}}
    for name, p in preds.items():
        out["overall"][name] = {"mae": mae(y, p), "rmse": rmse(y, p)}
    age_groups = {"<=21": test["age"] <= 21, "22-25": test["age"].between(22, 25),
                  "26-29": test["age"].between(26, 29), "30+": test["age"] >= 30}
    for g, m in age_groups.items():
        m = m.to_numpy()
        out["by_age"][g] = {"n": int(m.sum()),
                            **{name: mae(y[m], p[m]) for name, p in preds.items()}}
    # can it find the risers? U23 players ranked by predicted growth
    young = (test["age"] <= 23).to_numpy()
    actual_growth = (y - test["overall"].to_numpy())[young]
    rankers = {name: (p - test["overall"].to_numpy())[young] for name, p in preds.items()
               if name != "persistence"}
    rankers["ea_gap"] = test["gap"].to_numpy()[young]
    top_true = actual_growth >= np.quantile(actual_growth, 0.9)
    for name, score in rankers.items():
        order = np.argsort(-score, kind="stable")
        hits = {k: float(top_true[order[:k]].mean()) for k in (50, 100, 200)}
        out["young_ranking"][name] = {
            "spearman": float(spearmanr(score, actual_growth).statistic),
            "precision_top10pct_growth@50": hits[50],
            "precision_top10pct_growth@100": hits[100],
            "precision_top10pct_growth@200": hits[200],
            "mean_actual_growth_top100": float(actual_growth[order[:100]].mean()),
        }
    out["young_ranking"]["_n_u23"] = int(young.sum())
    out["young_ranking"]["_base_rate"] = float(top_true.mean())
    return out


def fit_baselines(train: pd.DataFrame, h: int):
    y = train[target_col(h)] - train["overall"]
    return {"age_curve": AgeCurve().fit(train, y), "ea_potential": PotentialLinear().fit(train, y)}


def predict_all(models: dict, test: pd.DataFrame) -> dict[str, np.ndarray]:
    cur = test["overall"].to_numpy()
    out = {"persistence": cur.astype(float)}
    for name, m in models.items():
        out[name] = cur + m.predict_delta(test)
    return out
