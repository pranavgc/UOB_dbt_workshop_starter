"""
Negative Binomial GLM.

The target is counts and its variance is roughly seven times its mean, so the
family matters. Poisson assumes variance equals mean; asserting that here would
understate every standard error and produce confidence intervals that are too
narrow to be worth printing.

The model is chosen on evidence rather than habit. `overdispersion()` returns the
Pearson chi-square per degree of freedom from a Poisson fit -- 1.0 if Poisson is
right -- and the dispersion parameter is estimated from the data rather than
guessed.

The GLM also fixes something the log-OLS could not: it models log E[y] directly
rather than E[log y], so zero-sales days stay in the sample instead of being
dropped to make the logarithm defined.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ml.features import design_matrix


class NegativeBinomialDemand:
    def __init__(self, spec: str = "structural") -> None:
        self.spec = spec
        self.result = None
        self.columns_: list[str] = []
        self.alpha_: float = 0.0
        self.poisson_dispersion_: float = 0.0

    def fit(self, train: pd.DataFrame) -> NegativeBinomialDemand:
        X = sm.add_constant(design_matrix(train, self.spec), has_constant="add")
        y = train["units_sold"].to_numpy(dtype=float)
        self.columns_ = list(X.columns)

        # Step 1: Poisson, purely to measure how badly its variance assumption fails.
        poisson = sm.GLM(y, X, family=sm.families.Poisson()).fit()
        self.poisson_dispersion_ = float(poisson.pearson_chi2 / poisson.df_resid)

        # Step 2: estimate the NB dispersion from the Poisson residuals via the
        # standard auxiliary regression (Cameron & Trivedi), then refit.
        mu = poisson.mu
        aux_y = ((y - mu) ** 2 - y) / mu
        self.alpha_ = float(np.clip(sm.OLS(aux_y, mu).fit().params[0], 1e-6, 10.0))

        self.result = sm.GLM(
            y, X, family=sm.families.NegativeBinomial(alpha=self.alpha_)
        ).fit(cov_type="cluster",
              cov_kwds={"groups": (train["store_id"] + "|" + train["sku"]).to_numpy()})
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        X = sm.add_constant(design_matrix(test, self.spec), has_constant="add")
        X = X.reindex(columns=self.columns_, fill_value=0.0)
        # A log link means the prediction is already E[y] -- no retransformation
        # correction, which is one fewer approximation than the log-OLS needs.
        return np.clip(self.result.predict(X), 0.0, None)

    def coef(self, name: str) -> float:
        return float(self.result.params.get(name, np.nan))

    def ci(self, name: str, alpha: float = 0.05) -> tuple[float, float]:
        if name not in self.result.params.index:
            return (np.nan, np.nan)
        lo, hi = self.result.conf_int(alpha=alpha).loc[name]
        return float(lo), float(hi)

    def elasticities(self) -> dict[str, tuple[float, float, float]]:
        out = {"jaffle": (self.coef("log_price_rel"), *self.ci("log_price_rel"))}
        idx = [self.columns_.index("log_price_rel"), self.columns_.index("log_price_rel_bev")]
        c = np.zeros(len(self.columns_))
        c[idx] = 1.0
        est = float(c @ self.result.params.to_numpy())
        se = float(np.sqrt(c @ self.result.cov_params().to_numpy() @ c))
        out["beverage"] = (est, est - 1.96 * se, est + 1.96 * se)
        return out

    def overdispersion(self) -> dict:
        return {
            "poisson_pearson_chi2_per_df": round(self.poisson_dispersion_, 3),
            "nb_alpha": round(self.alpha_, 4),
            "verdict": (
                "Poisson rejected: variance exceeds the mean"
                if self.poisson_dispersion_ > 1.5
                else "Poisson defensible"
            ),
        }
