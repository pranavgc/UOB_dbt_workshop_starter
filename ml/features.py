"""
Load the feature mart and build design matrices.

The mart is the contract. Nothing here recomputes a rolling window or reaches
back into the raw tables: if a feature is wrong, it is wrong in dbt, where a data
test can catch it. That separation is the point of keeping dbt in this project.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DUCKDB_PATH = ROOT / "data" / "jaffle.duckdb"
MART = "main.mart_ml_features"

# Features shared by every regression specification.
CALENDAR = ["fourier_sin1", "fourier_cos1", "fourier_sin2", "fourier_cos2", "ramp"]

# Lagged demand history. These are the controls that de-confound `is_featured`:
# placements are scheduled off trailing store x product-type demand, so
# conditioning on it breaks the correlation with the error term.
TRAILING = [
    "log_trailing_7d",
    "log_trailing_28d",
    "log_type_trailing_7d",
    "log_type_trailing_28d",
]


def load_features(db_path: Path | None = None) -> pd.DataFrame:
    """Read mart_ml_features and add the encodings the models need."""
    import duckdb

    path = db_path or DUCKDB_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m sim.generate` then `dbt build` first."
        )
    con = duckdb.connect(str(path), read_only=True)
    df = con.execute(f"select * from {MART} order by order_date, store_id, sku").df()
    con.close()

    df["order_date"] = pd.to_datetime(df["order_date"])
    df["is_featured"] = df["is_featured"].astype(int)
    df["is_beverage"] = (df["product_type"] == "beverage").astype(int)

    # Interaction that lets a single model carry a different elasticity per type.
    # The true DGP has one; a pooled coefficient is misspecified by construction.
    df["log_price_rel_bev"] = df["log_price_rel"] * df["is_beverage"]
    df["temp_dev_bev"] = df["temp_dev"] * df["is_beverage"]

    # Trailing demand enters in logs, matching the multiplicative demand system.
    # log1p rather than log because a store x sku can legitimately sell nothing.
    for src, dst in [
        ("trailing_7d_units", "log_trailing_7d"),
        ("trailing_28d_units", "log_trailing_28d"),
        ("type_trailing_7d_units", "log_type_trailing_7d"),
        ("type_trailing_28d_units", "log_type_trailing_28d"),
    ]:
        df[dst] = np.log1p(df[src])

    df["dow"] = df["day_of_week"].astype(int)
    return df


def drop_warmup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows whose trailing windows are not yet defined.

    The first 28 days of each store x sku have partial or null trailing features.
    Imputing them would inject a fabricated history into exactly the variables the
    de-confounding depends on, so they are dropped instead -- and the drop is
    applied identically to every model so the comparison stays fair.
    """
    need = ["trailing_7d_units", "trailing_28d_units",
            "type_trailing_7d_units", "type_trailing_28d_units"]
    mask = df[need].notna().all(axis=1)
    return df.loc[mask].reset_index(drop=True)


def design_matrix(df: pd.DataFrame, spec: str) -> pd.DataFrame:
    """
    Build the regressor block for a named specification.

    Three nested regressions, each fixing what the previous one got wrong.

    'naive'      -- what most people write first: one pooled price coefficient, a
                    linear temperature term, store and weekday effects, and
                    `is_featured` entered with no thought about why it was
                    switched on.
    'structural' -- the form the demand system actually has: elasticity
                    interacted with product type, a quadratic temperature term,
                    and sku fixed effects.
    'controlled' -- 'structural' plus the lagged trailing-demand controls that
                    proxy for the unobserved persistent shock driving featuring.
                    This is where the interesting trade-off shows up: the
                    controls reduce the bias in the featuring coefficient and
                    simultaneously damage two others. Controls are not free.
    'ml'         -- the raw feature block for the gradient booster, which gets no
                    hand-built interactions.
    """
    if spec == "naive":
        cols = ["log_price_rel", "is_featured", "temp_dev"] + CALENDAR
    elif spec in ("structural", "controlled"):
        cols = (
            # `is_beverage` is deliberately absent as a main effect: the sku
            # fixed effects below already determine product type, and including
            # both makes the design matrix rank-deficient. The interactions stay,
            # because those vary within a sku.
            ["log_price_rel", "log_price_rel_bev", "is_featured",
             "temp_dev", "temp_dev_sq", "temp_dev_bev"]
            + CALENDAR
            + (TRAILING if spec == "controlled" else [])
        )
    elif spec == "ml":
        cols = (
            ["log_price_rel", "is_featured", "is_beverage", "temp_dev", "temp_dev_sq",
             "dow", "day_of_year", "ramp", "days_since_last_featured"]
            + TRAILING
        )
    else:
        raise ValueError(f"unknown spec: {spec}")

    X = df[cols].copy()

    if spec in ("naive", "structural", "controlled"):
        # Store and weekday fixed effects, reference level dropped.
        X = pd.concat(
            [
                X,
                pd.get_dummies(df["store_name"], prefix="store", drop_first=True, dtype=float),
                pd.get_dummies(df["dow"], prefix="dow", drop_first=True, dtype=float),
            ],
            axis=1,
        )
    if spec in ("structural", "controlled"):
        X = pd.concat(
            [X, pd.get_dummies(df["sku"], prefix="sku", drop_first=True, dtype=float)],
            axis=1,
        )
    if spec == "ml":
        X["days_since_last_featured"] = X["days_since_last_featured"].fillna(999)

    return X.astype(float)
