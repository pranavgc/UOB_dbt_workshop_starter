"""
Regression diagnostics.

"Which OLS assumptions does your data violate, and what did you do about it?" is
the most common statistics question at this level, and the honest answer here is
that most of them fail. That is not a problem to hide; it is the reason the
specification looks the way it does.

Each test below returns a verdict and the action taken, so the output reads as a
set of decisions rather than a page of statistics with no consequence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera

from ml.features import design_matrix

# Regressors worth a collinearity check. The fixed-effect dummies are excluded:
# they are mechanically collinear with each other and a high VIF on a dummy block
# says nothing about whether the coefficients of interest are identified.
VIF_FEATURES = [
    "log_price_rel", "log_price_rel_bev", "is_featured",
    "temp_dev", "temp_dev_sq", "temp_dev_bev", "ramp",
    "fourier_sin1", "fourier_cos1", "fourier_sin2", "fourier_cos2",
    "log_trailing_7d", "log_trailing_28d",
    "log_type_trailing_7d", "log_type_trailing_28d",
]


def run(model, train: pd.DataFrame, spec: str = "controlled") -> list[dict]:
    rows = train[train["units_sold"] >= 1].reset_index(drop=True)
    X = sm.add_constant(design_matrix(rows, spec), has_constant="add")
    resid = np.asarray(model.result.resid, dtype=float)
    n = len(resid)
    out: list[dict] = []

    # ---- heteroskedasticity ------------------------------------------------
    lm, lm_p, _, _ = het_breuschpagan(resid, X.to_numpy())
    out.append(_row(
        "Heteroskedasticity", "Breusch-Pagan", f"LM = {lm:,.0f}, p = {lm_p:.2g}",
        lm_p < 0.05,
        "Standard errors clustered by store x sku, which is robust to both "
        "heteroskedasticity and within-series correlation.",
    ))

    # ---- serial correlation ------------------------------------------------
    dw = float(durbin_watson(resid))
    acf1 = _acf(resid, 1)
    acf7 = _acf(resid, 7)
    out.append(_row(
        "Serial correlation", "Durbin-Watson + residual ACF",
        f"DW = {dw:.3f}, ACF(1) = {acf1:.3f}, ACF(7) = {acf7:.3f}",
        abs(dw - 2.0) > 0.1 or abs(acf1) > 0.05,
        "Expected: a persistent latent demand shock sits in the error by "
        "construction. Handled by clustering, and by the lagged demand controls "
        "which absorb part of it. The residual dependence across skus within a "
        "store is what a mixed-effects model would address.",
    ))

    # ---- normality ---------------------------------------------------------
    jb, jb_p, skew, kurt = jarque_bera(resid)
    out.append(_row(
        "Normality of residuals", "Jarque-Bera",
        f"JB = {jb:,.0f}, skew = {skew:.2f}, kurtosis = {kurt:.2f}",
        jb_p < 0.05,
        "Not acted on. Normality is not needed for OLS to be unbiased, and at "
        f"n = {n:,} the CLT covers the inference. It would matter for prediction "
        "intervals, which is why those come from quantile methods instead.",
    ))

    # ---- multicollinearity -------------------------------------------------
    present = [c for c in VIF_FEATURES if c in X.columns]
    vifs = {c: float(variance_inflation_factor(X[present].to_numpy(), i))
            for i, c in enumerate(present)}
    worst = sorted(vifs.items(), key=lambda kv: -kv[1])[:3]
    out.append(_row(
        "Multicollinearity", "Variance inflation factor",
        "; ".join(f"{k} = {v:.1f}" for k, v in worst),
        worst[0][1] > 10,
        "The trailing-7-day and trailing-28-day demand features are correlated by "
        "construction -- one is a longer window over the same series. They are kept "
        "because dropping either weakens the de-confounding, and the cost is "
        "unstable individual coefficients, which stability_selection() quantifies.",
    ))

    # ---- influence ---------------------------------------------------------
    share = _influence_share(resid, X.shape[1], n)
    out.append(_row(
        "Influential observations", "Cook's distance > 4/n",
        f"{share:.2%} of rows",
        share > 0.05,
        "No rows removed. These are real high-demand days, not recording errors, "
        "and trimming them would bias the model toward quiet trading.",
    ))

    return out


def _row(assumption, test, statistic, violated, action):
    return {
        "assumption": assumption,
        "test": test,
        "statistic": statistic,
        "verdict": "VIOLATED" if violated else "ok",
        "action": action if violated else "None needed.",
    }


def _acf(x: np.ndarray, lag: int) -> float:
    x = x - x.mean()
    denom = float(np.dot(x, x))
    return float(np.dot(x[lag:], x[:-lag]) / denom) if denom else 0.0


def _influence_share(resid: np.ndarray, k: int, n: int) -> float:
    """
    Share of rows above the conventional 4/n Cook's distance threshold.

    Approximated from the standardised residual rather than computed exactly:
    statsmodels' exact influence measures build an n x n hat matrix, which at
    n > 40,000 needs more memory than the result is worth. The approximation
    treats leverage as uniform (k/n), which is reasonable for a design dominated
    by dummies and bounded continuous regressors.
    """
    s2 = float(np.dot(resid, resid)) / (n - k)
    h = k / n
    cook = (resid ** 2 / (k * s2)) * (h / (1 - h) ** 2)
    return float((cook > 4.0 / n).mean())


def to_markdown(rows: list[dict]) -> str:
    head = "| Assumption | Test | Statistic | Verdict | What was done |\n|---|---|---|---|---|\n"
    body = "\n".join(
        f"| {r['assumption']} | {r['test']} | {r['statistic']} | "
        f"{'**' + r['verdict'] + '**' if r['verdict'] == 'VIOLATED' else r['verdict']} | "
        f"{r['action']} |"
        for r in rows
    )
    return head + body
