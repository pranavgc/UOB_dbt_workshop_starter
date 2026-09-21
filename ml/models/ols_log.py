"""
OLS on log demand.

The demand system is multiplicative, so logs are the natural scale: a coefficient
is then a proportional effect, directly comparable to the simulator's true
parameters. Two specifications are fitted, and the gap between them is the
substantive finding of the project.

    naive      -- one pooled price coefficient, a linear temperature term, and
                  `is_featured` entered with no control for why it was switched on.
    structural -- the form the demand system actually has: elasticity by product
                  type, quadratic temperature, sku fixed effects.
    controlled -- structural plus lagged trailing-demand controls, which proxy for
                  the persistent unobserved shock that drives featuring.

Standard errors are clustered by store x sku. Residuals are serially correlated
within a series -- the same weather, the same local trend, the same persistent
demand shock -- so classical OLS standard errors are far too narrow, and an
interval that is too narrow is worse than no interval at all.

Why store x sku and not store, when the shock is a store x product-type process?
Because there are only six stores. Cluster-robust variance is a large-sample
result in the NUMBER OF CLUSTERS, and six is nowhere near enough: the estimator
is badly downward-biased and the intervals are not trustworthy. Clustering at
store x sku gives 60 clusters, which is defensible, at the cost of not capturing
dependence across skus within a store and type. That residual dependence is a
known limitation, and it is the argument for the mixed-effects model on day 3.

Known limitation, addressed by the Negative Binomial GLM on day 2: OLS on
log(y) estimates E[log y], not log E[y]. With a mean around 15 units the gap is
small, but it is a real approximation and it is stated rather than hidden.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ml.features import design_matrix

# Rows selling nothing cannot be logged. They are 0.3% of the data; dropping them
# is a documented choice, not an oversight, and the GLM on day 2 keeps them.
_MIN_UNITS = 1


class OLSLogDemand:
    def __init__(self, spec: str = "controlled") -> None:
        self.spec = spec
        self.result: sm.regression.linear_model.RegressionResultsWrapper | None = None
        self.columns_: list[str] = []
        self.sigma2_: float = 0.0

    def fit(self, train: pd.DataFrame) -> OLSLogDemand:
        fit_rows = train[train["units_sold"] >= _MIN_UNITS]
        X = sm.add_constant(design_matrix(fit_rows, self.spec), has_constant="add")
        y = np.log(fit_rows["units_sold"].to_numpy(dtype=float))
        self.columns_ = list(X.columns)

        model = sm.OLS(y, X)
        self.result = model.fit(
            cov_type="cluster",
            cov_kwds={
                "groups": (fit_rows["store_id"] + "|" + fit_rows["sku"]).to_numpy()
            },
        )
        # Retransformation variance: E[y] = exp(Xb + sigma^2 / 2) under lognormal
        # errors. Omitting it biases every prediction low by a few percent.
        self.sigma2_ = float(np.var(self.result.resid, ddof=len(self.columns_)))
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        X = sm.add_constant(design_matrix(test, self.spec), has_constant="add")
        X = X.reindex(columns=self.columns_, fill_value=0.0)
        return np.exp(self.result.predict(X) + self.sigma2_ / 2.0)

    # ---- coefficient access, for the recovery table -------------------------
    def coef(self, name: str) -> float:
        return float(self.result.params.get(name, np.nan))

    def ci(self, name: str, alpha: float = 0.05) -> tuple[float, float]:
        if name not in self.result.params.index:
            return (np.nan, np.nan)
        lo, hi = self.result.conf_int(alpha=alpha).loc[name]
        return float(lo), float(hi)

    def elasticities(self) -> dict[str, tuple[float, float, float]]:
        """(estimate, ci_low, ci_high) for each product type's price elasticity."""
        out: dict[str, tuple[float, float, float]] = {}
        base = self.coef("log_price_rel")
        lo, hi = self.ci("log_price_rel")
        if self.spec == "naive":
            out["pooled"] = (base, lo, hi)
            return out

        out["jaffle"] = (base, lo, hi)
        # Beverage elasticity is the sum of the main effect and the interaction,
        # so its interval needs the full covariance, not the two margins.
        idx = [self.columns_.index("log_price_rel"), self.columns_.index("log_price_rel_bev")]
        c = np.zeros(len(self.columns_))
        c[idx] = 1.0
        est = float(c @ self.result.params.to_numpy())
        se = float(np.sqrt(c @ self.result.cov_params().to_numpy() @ c))
        out["beverage"] = (est, est - 1.96 * se, est + 1.96 * se)
        return out
