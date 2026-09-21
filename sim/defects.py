"""
Deliberate data-quality defects.

Kept in its own module for a reason. The demand system in generate.py is a
statement about how customers behave; this file is a statement about how source
systems misbehave, and the two are separate concerns. It also means the defects
can be switched off (`defects.enabled: false`) to check that a pipeline failure
is really the defect and not something else.

Every defect here is one I have actually seen in a production feed. Each returns
a manifest entry recording exactly what it did, which lands in
sim/ground_truth.json, so the dbt tests downstream can be checked against what
was really injected rather than against a guess.

The pairing with the pipeline is the point:

    defect                        caught by                              repaired in
    ------------------------------------------------------------------------------------
    duplicate order rows          source `unique` (warn)                 int_orders_repaired
    cents reported as dollars     assert_order_revenue_reconciles        int_orders_repaired
    refunds as negative totals    accepted_range on staging              routed to stg refunds
    missing weather (MAR)         assert_weather_coverage                int_weather_imputed
    late-arriving orders          assert_load_lag_is_bounded (warn)      documented, not hidden
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def inject(tables: dict[str, pd.DataFrame], cfg: dict, rng: np.random.Generator,
           store_lookup: dict[str, str]) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    """Corrupt a clean feed. Returns the new tables and a manifest of what was done."""
    d = cfg.get("defects", {})
    if not d.get("enabled", False):
        return tables, [{"defect": "none", "note": "defects.enabled is false"}]

    manifest: list[dict] = []
    tables = dict(tables)

    for step in (
        lambda t: _duplicate_orders(t, d, rng),
        lambda t: _refunds(t, d, rng),
        lambda t: _unit_slip(t, d, store_lookup),
        lambda t: _missing_weather(t, d, rng),
        lambda t: _late_arrivals(t, d, rng),
    ):
        tables, entry = step(tables)
        manifest.append(entry)

    return tables, manifest


# --------------------------------------------------------------------------- #
def _duplicate_orders(tables, d, rng):
    """
    A double-submit bug in the till: the same order written twice.

    The line items are NOT duplicated, which is what makes this nasty. Joining
    items to orders before deduplicating fans out the join and inflates revenue
    by exactly the duplicate rate, silently and plausibly.
    """
    orders = tables["raw_orders"]
    rate = d.get("duplicate_order_rate", 0.0)
    n = int(round(len(orders) * rate))
    if n == 0:
        return tables, {"defect": "duplicate_orders", "rows": 0}

    picked = rng.choice(len(orders), size=n, replace=False)
    dupes = orders.iloc[picked].copy()
    tables["raw_orders"] = (
        pd.concat([orders, dupes], ignore_index=True)
        .sample(frac=1.0, random_state=int(rng.integers(0, 2**31)))
        .reset_index(drop=True)
    )
    return tables, {
        "defect": "duplicate_orders",
        "rows": int(n),
        "rate": rate,
        "detail": "exact duplicate order rows; line items not duplicated",
    }


def _refunds(tables, d, rng):
    """
    Refunds arriving in the same table as sales, with negative values and no lines.

    Dropping them silently would be wrong (the money is real) and leaving them in
    would corrupt demand. They get routed to their own staging model instead.
    """
    orders = tables["raw_orders"]
    rate = d.get("refund_rate", 0.0)
    n = int(round(len(orders) * rate))
    if n == 0:
        return tables, {"defect": "refunds", "rows": 0}

    src = orders.iloc[rng.choice(len(orders), size=n, replace=False)].copy()
    src["id"] = ["rf-" + str(i) for i in src["id"].str.slice(3)]
    for col in ("subtotal", "tax_paid", "order_total"):
        src[col] = -src[col]
    src["ordered_at"] = src["ordered_at"] + pd.to_timedelta(
        rng.integers(1, 10, size=n), unit="D"
    )
    tables["raw_orders"] = pd.concat([orders, src], ignore_index=True)
    return tables, {
        "defect": "refunds",
        "rows": int(n),
        "rate": rate,
        "detail": "negative subtotal/tax/total, no line items, id prefixed 'rf-'",
    }


def _unit_slip(tables, d, store_lookup):
    """
    One store-month has the cents conversion applied twice, so its header values
    are 100x too large.

    The classic migration bug. It is invisible to a row-level range check because
    the values are still positive and still numeric -- it only shows up when the
    order header stops agreeing with the sum of its own line items, which is
    precisely what the reconciliation test compares.

    Note the direction: too LARGE, not too small. A slip that shrinks values by
    100 rounds integer cents away and destroys information, so no repair could
    restore the original and every exact downstream check would have to be
    loosened into uselessness. Multiplying is lossless, so the repair is exact and
    the tests stay strict.
    """
    spec = d.get("unit_slip")
    if not spec:
        return tables, {"defect": "unit_slip", "rows": 0}

    orders = tables["raw_orders"]
    store_id = store_lookup[spec["store"]]
    month = pd.Period(spec["month"], freq="M")
    mask = (orders["store_id"] == store_id) & (
        orders["ordered_at"].dt.to_period("M") == month
    )
    for col in ("subtotal", "tax_paid", "order_total"):
        orders.loc[mask, col] = orders.loc[mask, col] * 100
    tables["raw_orders"] = orders
    return tables, {
        "defect": "unit_slip",
        "rows": int(mask.sum()),
        "store": spec["store"],
        "month": spec["month"],
        "detail": "order header values multiplied by 100; line items untouched",
    }


def _missing_weather(tables, d, rng):
    """
    Missing weather readings -- missing At Random, not Completely at Random.

    The station drops out far more often on extreme days, which is the realistic
    pattern and also the dangerous one: mean-imputing the gaps pulls exactly the
    observations that carry the temperature signal back toward the middle and
    biases the weather coefficient toward zero. The size of that bias is
    measured in ml/run.py rather than asserted.
    """
    spec = d.get("weather_missing")
    if not spec:
        return tables, {"defect": "missing_weather", "rows": 0}

    w = tables["raw_weather"].copy()
    z = (w["temperature_c"] - w["temperature_c"].mean()) / w["temperature_c"].std()
    extreme = z.abs() > spec.get("extreme_abs_z", 1.5)
    p = np.where(extreme, spec.get("extreme_rate", 0.14), spec.get("base_rate", 0.02))
    drop = rng.random(len(w)) < p

    w.loc[drop, "temperature_c"] = np.nan
    tables["raw_weather"] = w
    return tables, {
        "defect": "missing_weather",
        "rows": int(drop.sum()),
        "share_of_rows": round(float(drop.mean()), 4),
        "share_missing_among_extreme_days": round(float(drop[extreme].mean()), 4),
        "share_missing_among_normal_days": round(float(drop[~extreme].mean()), 4),
        "detail": "MAR: missingness depends on the value itself, so mean imputation biases",
    }


def _late_arrivals(tables, d, rng):
    """
    Orders that arrive in the warehouse days after they happened.

    Harmless if you know about it, quietly fatal if you don't: a feature built
    "as of" the order date uses rows that had not landed yet, so a backtest sees
    information production would not have had. `loaded_at` is emitted so the
    lag is visible rather than assumed away.
    """
    spec = d.get("late_arrival")
    orders = tables["raw_orders"]
    base = orders["ordered_at"].dt.normalize() + pd.Timedelta(days=1)
    if not spec:
        orders["loaded_at"] = base
        tables["raw_orders"] = orders
        return tables, {"defect": "late_arrivals", "rows": 0}

    n = len(orders)
    late = rng.random(n) < spec.get("rate", 0.03)
    extra = np.where(late, rng.integers(1, spec.get("max_lag_days", 4) + 1, size=n), 0)
    orders["loaded_at"] = base + pd.to_timedelta(extra, unit="D")
    tables["raw_orders"] = orders
    return tables, {
        "defect": "late_arrivals",
        "rows": int(late.sum()),
        "rate": spec.get("rate", 0.03),
        "max_lag_days": spec.get("max_lag_days", 4),
        "detail": "loaded_at lags ordered_at by 1 day normally, more for late rows",
    }
