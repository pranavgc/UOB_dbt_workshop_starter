"""
Gradient boosting on the raw feature block.

Fitted with a Poisson objective rather than squared error: the target is counts,
the variance grows with the mean, and a Gaussian loss would weight a busy
Saturday at a flagship store the same as a quiet Tuesday at a new one.

The booster gets no hand-built interactions -- no type-specific price term, no
squared temperature. If it beats the full OLS specification, it is because it
found that structure itself, which is exactly the comparison worth making
against a data-generating process whose nonlinearity is known in advance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.features import design_matrix

PARAMS = dict(
    objective="poisson",
    n_estimators=400,
    learning_rate=0.07,
    num_leaves=48,
    min_child_samples=40,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
    reg_lambda=1.0,
    verbose=-1,
    n_jobs=-1,
)


class GBMDemand:
    def __init__(self, seed: int = 20260914, **overrides) -> None:
        import lightgbm as lgb

        self.model = lgb.LGBMRegressor(**{**PARAMS, "random_state": seed, **overrides})
        self.columns_: list[str] = []

    def fit(self, train: pd.DataFrame) -> GBMDemand:
        X = design_matrix(train, "ml")
        self.columns_ = list(X.columns)
        cat = ["store_id_code", "sku_code"]
        X = self._add_categoricals(X, train)
        self.model.fit(X, train["units_sold"].to_numpy(dtype=float), categorical_feature=cat)
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        X = self._add_categoricals(design_matrix(test, "ml"), test)
        return np.clip(self.model.predict(X), 0.0, None)

    @staticmethod
    def _add_categoricals(X: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["store_id_code"] = pd.factorize(df["store_id"], sort=True)[0]
        X["sku_code"] = pd.factorize(df["sku"], sort=True)[0]
        return X

    def importances(self) -> pd.Series:
        names = list(self.model.feature_name_)
        return pd.Series(self.model.feature_importances_, index=names).sort_values(ascending=False)
