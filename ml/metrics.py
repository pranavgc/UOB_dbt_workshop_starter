"""Scoring. Each metric is here because it answers a question the others cannot."""

from __future__ import annotations

import numpy as np


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(y - yhat)))


def smape(y: np.ndarray, yhat: np.ndarray) -> float:
    """
    Symmetric MAPE, in percent.

    Plain MAPE is unusable here: roughly 0.3% of store x sku x days sell nothing,
    and dividing by zero would either explode or force an arbitrary filter that
    quietly removes the hardest rows from the comparison. sMAPE bounds the
    per-row penalty instead.
    """
    denom = (np.abs(y) + np.abs(yhat)) / 2.0
    ok = denom > 0
    return float(100.0 * np.mean(np.abs(y[ok] - yhat[ok]) / denom[ok]))


def bias(y: np.ndarray, yhat: np.ndarray) -> float:
    """Mean signed error. A model can have good RMSE and still be systematically low."""
    return float(np.mean(yhat - y))


def score(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    return {"rmse": rmse(y, yhat), "mae": mae(y, yhat), "smape": smape(y, yhat),
            "bias": bias(y, yhat)}


def recovery_error(estimated: float, truth: float) -> float:
    """Percent error of a coefficient estimate against the simulator's true value."""
    return float(100.0 * (estimated - truth) / abs(truth))
