"""
Natural cubic splines on temperature.

The true weather response is quadratic with an optimum around 20 degrees: demand
falls away in both directions. A linear term cannot represent that shape at all,
and the resulting bias is knowable in advance rather than discovered afterwards.

Splines are the interpretable answer to it. Unlike a booster, the fitted curve can
be read off and plotted directly against the true response -- which is what
reports/figures/temperature_response.png does, and which is only possible because
the true response is known.

Everything else in the specification is unchanged, so the comparison against the
structural OLS isolates the effect of the functional form and nothing else.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import SplineTransformer

from ml.features import design_matrix

# Dropped from the design so the spline basis is the only temperature term.
_LINEAR_TEMP = ["temp_dev", "temp_dev_sq"]


class SplineTemperatureDemand:
    def __init__(self, n_knots: int = 6, degree: int = 3) -> None:
        self.spline = SplineTransformer(
            n_knots=n_knots, degree=degree, extrapolation="linear", include_bias=False
        )
        self.result = None
        self.columns_: list[str] = []

    def _design(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        X = design_matrix(df, "structural").drop(columns=_LINEAR_TEMP)
        t = df[["temperature_c"]].to_numpy(dtype=float)
        basis = self.spline.fit_transform(t) if fit else self.spline.transform(t)
        # B-spline bases sum to 1 at every point, so the full basis is exactly
        # collinear with the intercept. Dropping the first column is the standard
        # fix and leaves the fitted curve unchanged up to a constant -- which is
        # irrelevant here, since temperature_response() re-centres at 20 degrees.
        basis = basis[:, 1:]
        cols = [f"temp_spline_{i}" for i in range(basis.shape[1])]
        sp = pd.DataFrame(basis, columns=cols, index=X.index)
        # Beverages respond to temperature differently, so the basis is
        # interacted with the type rather than assumed common.
        for c in cols:
            sp[c + "_bev"] = sp[c] * df["is_beverage"].to_numpy()
        return pd.concat([X, sp], axis=1)

    def fit(self, train: pd.DataFrame) -> SplineTemperatureDemand:
        rows = train[train["units_sold"] >= 1]
        X = sm.add_constant(self._design(rows, fit=True), has_constant="add")
        self.columns_ = list(X.columns)
        y = np.log(rows["units_sold"].to_numpy(dtype=float))
        self.result = sm.OLS(y, X).fit(
            cov_type="cluster",
            cov_kwds={"groups": (rows["store_id"] + "|" + rows["sku"]).to_numpy()},
        )
        self.sigma2_ = float(np.var(self.result.resid, ddof=len(self.columns_)))
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        X = sm.add_constant(self._design(test, fit=False), has_constant="add")
        X = X.reindex(columns=self.columns_, fill_value=0.0)
        return np.exp(self.result.predict(X) + self.sigma2_ / 2.0)

    def temperature_response(self, temps: np.ndarray, is_beverage: int = 0) -> np.ndarray:
        """
        The fitted curve, centred at 20 degrees so it is directly comparable with
        the simulator's true response (which is zero at its optimum by construction).
        """
        basis = self.spline.transform(temps.reshape(-1, 1))[:, 1:]
        n = basis.shape[1]
        params = self.result.params
        contrib = np.zeros(len(temps))
        for i in range(n):
            b = params.get(f"temp_spline_{i}", 0.0)
            bb = params.get(f"temp_spline_{i}_bev", 0.0)
            contrib += basis[:, i] * (b + bb * is_beverage)
        ref = self.spline.transform(np.array([[20.0]]))[:, 1:]
        ref_val = sum(
            ref[0, i] * (params.get(f"temp_spline_{i}", 0.0)
                         + params.get(f"temp_spline_{i}_bev", 0.0) * is_beverage)
            for i in range(n)
        )
        return contrib - ref_val
