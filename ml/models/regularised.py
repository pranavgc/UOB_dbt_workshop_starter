"""
Ridge, Lasso and ElasticNet -- and why the coefficients should not be trusted.

The interesting result here is not that regularisation improves prediction. On
this data it barely does, because there are forty-odd well-behaved regressors and
tens of thousands of rows, which is not the regime where shrinkage earns its keep.

The interesting result is what it does to inference. Regularisation buys variance
reduction by paying in bias, and the bill lands on exactly the coefficients this
project is trying to recover. Two things are measured rather than asserted:

  1. Shrinkage path. The price elasticity is tracked as the penalty rises, so the
     drift away from the true value is visible instead of being a claim.

  2. Stability selection. Lasso is refitted on bootstrap resamples and the
     selection frequency of each feature recorded. With correlated regressors --
     and trailing-7-day and trailing-28-day demand are highly correlated by
     construction -- which of a correlated pair survives is close to a coin flip.
     A feature kept in 55% of resamples is not evidence of anything, and reading
     one Lasso fit as "these are the important variables" is the mistake this
     measurement exists to prevent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.features import design_matrix

SPEC = "controlled"


class ElasticNetDemand:
    """ElasticNet on log demand, with the penalty chosen by time-aware CV."""

    def __init__(self, l1_ratio=(0.1, 0.5, 0.9, 1.0), n_alphas: int = 30, seed: int = 20260914):
        self.model = Pipeline(
            [
                ("scale", StandardScaler()),
                # TimeSeriesSplit, not the default K-fold: choosing the penalty on
                # randomly shuffled folds is the same leakage the rest of the
                # project is careful to avoid, just hidden inside model selection.
                ("enet", ElasticNetCV(
                    l1_ratio=list(l1_ratio), alphas=n_alphas, cv=_ts_cv(),
                    random_state=seed, max_iter=5000, n_jobs=-1,
                )),
            ]
        )
        self.columns_: list[str] = []
        self.sigma2_: float = 0.0

    def fit(self, train: pd.DataFrame) -> ElasticNetDemand:
        rows = train[train["units_sold"] >= 1]
        X = design_matrix(rows, SPEC)
        self.columns_ = list(X.columns)
        y = np.log(rows["units_sold"].to_numpy(dtype=float))
        self.model.fit(X, y)
        resid = y - self.model.predict(X)
        self.sigma2_ = float(np.var(resid))
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        X = design_matrix(test, SPEC).reindex(columns=self.columns_, fill_value=0.0)
        return np.exp(self.model.predict(X) + self.sigma2_ / 2.0)

    @property
    def enet(self):
        return self.model.named_steps["enet"]

    def coef(self, name: str) -> float:
        """
        Coefficient on the ORIGINAL scale.

        The pipeline standardises first, so the fitted coefficient is per standard
        deviation and is not comparable to the simulator's parameters until it is
        divided back out. Forgetting this step is a quiet and common way to report
        a recovery error that is really a units error.
        """
        if name not in self.columns_:
            return float("nan")
        i = self.columns_.index(name)
        scale = self.model.named_steps["scale"].scale_[i]
        return float(self.enet.coef_[i] / scale)

    def chosen(self) -> dict:
        return {"alpha": float(self.enet.alpha_), "l1_ratio": float(self.enet.l1_ratio_)}


def _ts_cv(n_splits: int = 4):
    from sklearn.model_selection import TimeSeriesSplit

    return TimeSeriesSplit(n_splits=n_splits)


# --------------------------------------------------------------------------- #
def stability_selection(
    train: pd.DataFrame, alpha: float = 0.01, n_boot: int = 60, seed: int = 3
) -> pd.DataFrame:
    """
    How often does Lasso keep each feature across bootstrap resamples?

    Resampling is by store x sku series, not by row. Rows within a series are
    serially correlated, so a row-level bootstrap would treat 1,096 correlated
    observations as 1,096 independent ones and report a stability that is not
    there.
    """
    rows = train[train["units_sold"] >= 1]
    X_all = design_matrix(rows, SPEC)
    y_all = np.log(rows["units_sold"].to_numpy(dtype=float))
    series = (rows["store_id"] + "|" + rows["sku"]).to_numpy()
    unique = np.unique(series)

    rng = np.random.default_rng(seed)
    scaler = StandardScaler().fit(X_all)
    counts = np.zeros(X_all.shape[1])
    signs = np.zeros(X_all.shape[1])

    for _ in range(n_boot):
        picked = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(series == s) for s in picked])
        lasso = Lasso(alpha=alpha, max_iter=5000)
        lasso.fit(scaler.transform(X_all.iloc[idx]), y_all[idx])
        kept = np.abs(lasso.coef_) > 1e-8
        counts += kept
        signs += np.sign(lasso.coef_) * kept

    out = pd.DataFrame(
        {
            "feature": X_all.columns,
            "selection_frequency": (counts / n_boot).round(3),
            # A feature kept often but with an inconsistent sign is worse than one
            # dropped: the model is confident about magnitude and undecided about
            # direction.
            "sign_consistency": np.where(
                counts > 0, np.abs(signs) / np.maximum(counts, 1), np.nan
            ).round(3),
        }
    )
    return out.sort_values("selection_frequency", ascending=False).reset_index(drop=True)


def shrinkage_path(train: pd.DataFrame, alphas=(0.0001, 0.001, 0.005, 0.02, 0.05, 0.1)) -> pd.DataFrame:
    """Track the price elasticity as the penalty rises. Bias is not free."""
    rows = train[train["units_sold"] >= 1]
    X = design_matrix(rows, SPEC)
    y = np.log(rows["units_sold"].to_numpy(dtype=float))
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    i = list(X.columns).index("log_price_rel")

    out = []
    for a in alphas:
        lasso = Lasso(alpha=a, max_iter=5000).fit(Xs, y)
        out.append(
            {
                "alpha": a,
                "elasticity_jaffle": round(float(lasso.coef_[i] / scaler.scale_[i]), 4),
                "n_features_kept": int((np.abs(lasso.coef_) > 1e-8).sum()),
            }
        )
    return pd.DataFrame(out)
