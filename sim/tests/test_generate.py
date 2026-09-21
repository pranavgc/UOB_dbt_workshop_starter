"""
Tests for the simulator.

A generator that silently stops producing the structure it claims to produce
would invalidate every result downstream without failing anything, so the
properties the project depends on are asserted here rather than assumed.

These run on a short window so CI stays quick; the structure being checked does
not depend on the length of the window.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from sim.generate import CONFIG_PATH, generate

YEARS = 0.45


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """The feed as shipped, defects and all."""
    out = tmp_path_factory.mktemp("sim")
    return generate(years=YEARS, write_duckdb=False, output_dir=out)


@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    """
    The same feed with defect injection switched off.

    Some properties belong to the demand generator and some belong to the defect
    injector, and testing them against the same data would confuse the two. If the
    order headers stop matching their line items here, the generator is broken; if
    they stop matching in `run`, that is the injected unit slip doing its job.
    """
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    cfg["defects"]["enabled"] = False
    out = tmp_path_factory.mktemp("clean")
    path = out / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return generate(years=YEARS, write_duckdb=False, output_dir=out, config_path=path)


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load(CONFIG_PATH.read_text())


def test_is_reproducible(tmp_path_factory):
    """Same seed, same data. Without this, none of the results are re-checkable."""
    a = generate(years=0.2, write_duckdb=False, output_dir=tmp_path_factory.mktemp("a"))
    b = generate(years=0.2, write_duckdb=False, output_dir=tmp_path_factory.mktemp("b"))
    assert a["tables"] == b["tables"]
    assert (
        a["frames"]["raw_orders"]["order_total"].sum()
        == b["frames"]["raw_orders"]["order_total"].sum()
    )


def test_order_subtotal_equals_sum_of_line_prices(clean):
    """
    The property the dbt reconciliation test asserts downstream.

    Run against the CLEAN feed. On the shipped feed this deliberately fails for
    one store-month -- that is the injected unit slip, and catching it is the
    reconciliation test's whole purpose. Checking it here as well means a failure
    tells you where the problem is: here, the generator is wrong; downstream, the
    pipeline is.
    """
    orders = clean["frames"]["raw_orders"]
    items = clean["frames"]["raw_items"]
    prices = clean["frames"]["raw_price_snapshots"]

    lines = items.merge(
        orders[["id", "store_id", "ordered_at"]].rename(columns={"id": "order_id"}),
        on="order_id",
    )
    lines["order_date"] = lines["ordered_at"].dt.normalize()
    priced = lines.merge(
        prices.rename(columns={"price_date": "order_date"}),
        on=["store_id", "sku", "order_date"],
        how="left",
    )
    assert priced["price"].notna().all(), "every line must find a price snapshot"

    totals = priced.groupby("order_id")["price"].sum()
    joined = orders.set_index("id")["subtotal"].align(totals, join="inner")
    assert (joined[0] == joined[1]).all()


def test_counts_are_overdispersed(run):
    """Negative Binomial, not Poisson. The GLM choice on day 2 depends on this."""
    items = run["frames"]["raw_items"]
    orders = run["frames"]["raw_orders"]
    lines = items.merge(orders[["id", "store_id", "ordered_at"]].rename(columns={"id": "order_id"}),
                        on="order_id")
    lines["order_date"] = lines["ordered_at"].dt.normalize()
    daily = lines.groupby(["store_id", "sku", "order_date"]).size()
    assert daily.var() > 2.0 * daily.mean(), "variance should clearly exceed the mean"


def test_basket_size_matches_calibration(run, cfg):
    """Calibrated against the 686 real seed orders: mean basket 1.45 items."""
    target = cfg["baskets"]["mean_items_per_order"]
    observed = len(run["frames"]["raw_items"]) / len(run["frames"]["raw_orders"])
    assert observed == pytest.approx(target, abs=0.05)


def test_featuring_is_endogenous(run):
    """
    The confounder must actually be present.

    Featuring is supposed to follow weak trailing demand. If a refactor ever made
    assignment effectively random, the headline finding of the project would
    quietly become false while every other test still passed.
    """
    orders = run["frames"]["raw_orders"]
    items = run["frames"]["raw_items"]
    featured = run["frames"]["raw_featured_placements"]

    lines = items.merge(orders[["id", "store_id", "ordered_at"]].rename(columns={"id": "order_id"}),
                        on="order_id")
    lines["order_date"] = lines["ordered_at"].dt.normalize()
    daily = (
        lines.groupby(["store_id", "sku", "order_date"]).size()
        .rename("units").reset_index().sort_values("order_date")
    )
    # The rule reads demand BEFORE the campaign, so that is what has to be
    # measured. Demand *on* an opening day is already lifted by the treatment
    # itself and would show the opposite sign for the opposite reason.
    grouped = daily.groupby(["store_id", "sku"])["units"]
    daily["trailing_7"] = grouped.transform(lambda s: s.shift(1).rolling(7, min_periods=7).mean())
    daily["trailing_28"] = grouped.transform(lambda s: s.shift(1).rolling(28, min_periods=28).mean())
    daily["ratio"] = daily["trailing_7"] / daily["trailing_28"]
    daily = daily.dropna(subset=["ratio"])

    flags = featured.rename(columns={"placement_date": "order_date"})[
        ["store_id", "sku", "order_date"]
    ].drop_duplicates()
    flags["prev_day"] = flags["order_date"] - np.timedelta64(1, "D")
    ongoing = flags.merge(
        flags[["store_id", "sku", "order_date"]].rename(columns={"order_date": "prev_day"}),
        on=["store_id", "sku", "prev_day"], how="left", indicator=True,
    )
    opening_days = ongoing[ongoing["_merge"] == "left_only"][
        ["store_id", "sku", "order_date"]
    ]
    assert len(opening_days) >= 20, "too few campaigns to test the assignment rule"

    opens = daily.merge(opening_days, on=["store_id", "sku", "order_date"])
    assert len(opens) >= 20

    # Campaigns should launch after a soft patch: the trailing 7-day average
    # below the trailing 28-day average, on average.
    assert opens["ratio"].mean() < daily["ratio"].mean(), (
        "campaigns should follow weak trailing demand -- the assignment rule "
        "appears to have become random"
    )


def test_ground_truth_matches_config(run, cfg):
    """The published answers must be the ones actually used."""
    truth = run["truth"]["coefficients"]
    c = cfg["coefficients"]
    assert truth["beta_price_jaffle"] == c["beta_price"]["jaffle"]
    assert truth["beta_price_beverage"] == c["beta_price"]["beverage"]
    assert truth["gamma_featured"] == c["gamma_featured"]
    assert truth["ramp"] == c["ramp"]


def test_latent_shock_is_not_leaked(run):
    """
    The unobserved confounder must stay unobserved.

    If it ever appeared in an emitted table, every model could condition on it
    directly and the whole exercise would be vacuous.
    """
    for name, frame in run["frames"].items():
        cols = {c.lower() for c in frame.columns}
        assert not (cols & {"latent", "latent_shock", "lambda", "eta", "mu"}), (
            f"{name} leaks a latent quantity"
        )


def test_prices_move_within_a_store_and_sku(run):
    """
    The elasticity is identified only from within-series price movement, because
    the store and sku fixed effects absorb everything else. Too little of it and
    the estimates are noise -- which is exactly what happened in the first draft.
    """
    prices = run["frames"]["raw_price_snapshots"].copy()
    prices["log_price"] = np.log(prices["price"])
    within = prices.groupby(["store_id", "sku"])["log_price"].transform("mean")
    assert (prices["log_price"] - within).std() > 0.03


def test_injected_defects_are_present_and_declared(run):
    """
    The defect manifest must describe the data that was actually produced.

    Without this, a refactor could quietly stop injecting a defect while the
    manifest and the README kept claiming it was there, and every dbt test built
    to catch it would pass for the wrong reason.
    """
    manifest = {m["defect"]: m for m in run["truth"]["injected_defects"]}
    orders = run["frames"]["raw_orders"]
    weather = run["frames"]["raw_weather"]

    # duplicates really are duplicated
    assert manifest["duplicate_orders"]["rows"] > 0
    assert len(orders) - orders["id"].nunique() == manifest["duplicate_orders"]["rows"]

    # refunds really are negative and really have no line items
    refunds = orders[orders["subtotal"] < 0]
    assert len(refunds) == manifest["refunds"]["rows"]
    assert not refunds["id"].isin(run["frames"]["raw_items"]["order_id"]).any()

    # the unit slip really is 100x, in one store-month only
    slip = manifest["unit_slip"]
    assert slip["rows"] > 0
    slipped = orders[orders["subtotal"] > 20 * orders["subtotal"].median()]
    assert slipped["store_id"].nunique() == 1

    # weather missingness really is MAR, not MCAR
    mw = manifest["missing_weather"]
    assert weather["temperature_c"].isna().sum() == mw["rows"]
    assert mw["share_missing_among_extreme_days"] > 3 * mw["share_missing_among_normal_days"], (
        "missingness should depend on the value -- otherwise it is MCAR and the "
        "imputation-bias finding is not real"
    )
