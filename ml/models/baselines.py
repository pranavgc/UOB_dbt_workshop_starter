"""
The bar every other model has to clear.

Both baselines forecast the whole test window from training data only -- no test
actuals, no recursive one-step-ahead updating -- so they are scored on exactly
the task the regressions are scored on. A model that cannot beat a weekday
average is not a model, and without these two rows in the results table an R^2 of
0.8 means nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def store_sku_mean(train: pd.DataFrame, test: pd.DataFrame, window_days: int = 28) -> np.ndarray:
    """Mean units per store x sku over the last `window_days` of training."""
    cutoff = train["order_date"].max() - pd.Timedelta(days=window_days - 1)
    recent = train[train["order_date"] >= cutoff]
    lookup = recent.groupby(["store_id", "sku"])["units_sold"].mean()
    fallback = float(recent["units_sold"].mean())
    keys = pd.MultiIndex.from_arrays([test["store_id"], test["sku"]])
    return lookup.reindex(keys).fillna(fallback).to_numpy()


def seasonal_naive(train: pd.DataFrame, test: pd.DataFrame, weeks: int = 8) -> np.ndarray:
    """
    Mean units per store x sku x weekday over the last `weeks` of training.

    This is the seasonal-naive forecast written properly for a multi-day horizon:
    a plain lag-7 would need actuals from inside the test window, which is not
    available to the models it is being compared against.
    """
    cutoff = train["order_date"].max() - pd.Timedelta(days=weeks * 7 - 1)
    recent = train[train["order_date"] >= cutoff]
    lookup = recent.groupby(["store_id", "sku", "dow"])["units_sold"].mean()
    coarse = recent.groupby(["store_id", "sku"])["units_sold"].mean()
    fallback = float(recent["units_sold"].mean())

    keys = pd.MultiIndex.from_arrays([test["store_id"], test["sku"], test["dow"]])
    out = lookup.reindex(keys).to_numpy()
    coarse_keys = pd.MultiIndex.from_arrays([test["store_id"], test["sku"]])
    backup = coarse.reindex(coarse_keys).fillna(fallback).to_numpy()
    return np.where(np.isnan(out), backup, out)
