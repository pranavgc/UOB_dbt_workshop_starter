"""
Calibration against the real Jaffle Shop orders.

This is the answer to the only serious objection a simulated-data project faces:
"you made the numbers up, so of course your models work." The reply is that the
generator's marginal distributions are held against the 686 genuine orders
preserved in calibration/real_seeds/, and here is the fit.

Two of the four checks are honest and two are not, and the difference matters:

  CALIBRATED (fitted to the real data, so agreement is expected, not evidence)
    - basket size: mean items per order
    - sku mix: per-sku popularity offsets were fitted from these shares
    - arrival times: the mixture was fitted to the real hour-of-day histogram

  EMERGENT (never fitted; agreement is a genuine out-of-sample check)
    - order value: falls out of basket size x sku mix x catalogue prices. Nothing
      in the generator targets it.

Reported honestly in both directions. The real sample is one store over sixteen
days in 2016, which is small and not representative of a six-store chain over
three years, so close agreement on a tail statistic would be luck rather than
skill -- and that caveat is in the output, not just in this docstring.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "calibration" / "real_seeds"
RAW = ROOT / "data" / "raw"


def _load_real() -> dict[str, pd.DataFrame]:
    orders = pd.read_csv(REAL / "raw_orders.csv")
    items = pd.read_csv(REAL / "raw_items.csv")
    orders.columns = [c.lower() for c in orders.columns]
    items.columns = [c.lower() for c in items.columns]
    orders["ordered_at"] = pd.to_datetime(orders["ordered_at"])
    return {"orders": orders, "items": items}


def _load_sim() -> dict[str, pd.DataFrame]:
    """
    Read the REPAIRED pipeline output, not the raw feed.

    This matters more than it looks. The raw feed carries the injected defects --
    duplicate rows, refunds, and one store-month inflated 100x by a unit slip --
    and comparing that against real data measures the defects rather than the
    demand model. The first run of this module did exactly that and reported a
    simulated mean order value of $19.70 against a real $10.53, with a median of
    only $8.48: a mean nowhere near its own median, which is the signature of a
    handful of enormous rows rather than a systematically different distribution.

    Calibrating against int_orders_repaired also makes this a check on the whole
    pipeline, which is the more useful thing to validate anyway.
    """
    import duckdb

    db = ROOT / "data" / "jaffle.duckdb"
    if db.exists():
        con = duckdb.connect(str(db), read_only=True)
        orders = con.execute(
            """
            select order_id as id, store_id, ordered_at,
                   order_subtotal_cents + order_tax_paid_cents as order_total
            from main.int_orders_repaired
            """
        ).df()
        items = con.execute(
            "select order_item_id, order_id, sku from main.int_order_lines_enriched"
        ).df()
        con.close()
        return {"orders": orders, "items": items}

    # Fallback for a parquet-only run: apply the same repairs inline.
    orders = pd.read_parquet(RAW / "raw_orders.parquet")
    items = pd.read_parquet(RAW / "raw_items.parquet")
    orders = orders[orders["subtotal"] > 0].drop_duplicates(subset="id")
    return {"orders": orders, "items": items}


def compare() -> dict:
    real, sim = _load_real(), _load_sim()
    out: dict[str, dict] = {}

    # ---- basket size (calibrated) ------------------------------------------
    r_basket = real["items"].groupby("order_id").size()
    s_basket = sim["items"].groupby("order_id").size()
    out["basket_size"] = {
        "status": "calibrated",
        "real_mean": round(float(r_basket.mean()), 3),
        "sim_mean": round(float(s_basket.mean()), 3),
        "real_max": int(r_basket.max()),
        "sim_max": int(s_basket.max()),
        **_ks(r_basket.to_numpy(), s_basket.to_numpy()),
    }

    # ---- sku mix (calibrated) ----------------------------------------------
    r_mix = real["items"]["sku"].value_counts(normalize=True).sort_index()
    s_mix = sim["items"]["sku"].value_counts(normalize=True).sort_index()
    aligned = pd.DataFrame({"real": r_mix, "sim": s_mix}).fillna(0.0)
    out["sku_mix"] = {
        "status": "calibrated",
        # Total variation distance: half the L1 gap between two distributions.
        # Reported instead of a chi-square because with 777k simulated rows any
        # chi-square test rejects on a difference too small to care about --
        # a large sample makes significance testing the wrong tool here.
        "total_variation_distance": round(float(0.5 * (aligned["real"] - aligned["sim"]).abs().sum()), 4),
        "real_beverage_share": round(float(r_mix[r_mix.index.str.startswith("BEV")].sum()), 4),
        "sim_beverage_share": round(float(s_mix[s_mix.index.str.startswith("BEV")].sum()), 4),
        "per_sku": aligned.round(4).to_dict(orient="index"),
    }

    # ---- arrival hour (calibrated) -----------------------------------------
    r_hour = real["orders"]["ordered_at"].dt.hour
    s_hour = sim["orders"]["ordered_at"].dt.hour
    out["arrival_hour"] = {
        "status": "calibrated",
        "real_median": int(r_hour.median()),
        "sim_median": int(s_hour.median()),
        "real_share_before_10am": round(float((r_hour < 10).mean()), 4),
        "sim_share_before_10am": round(float((s_hour < 10).mean()), 4),
        **_ks(r_hour.to_numpy(float), s_hour.to_numpy(float)),
    }

    # ---- order value (EMERGENT -- the real test) ----------------------------
    r_val = real["orders"]["order_total"].to_numpy(float) / 100.0
    s_val = sim["orders"]["order_total"].to_numpy(float) / 100.0
    out["order_value"] = {
        "status": "emergent",
        "note": (
            "Not fitted. Falls out of basket size, sku mix and catalogue prices. "
            "Agreement here is a genuine out-of-sample check; disagreement is a "
            "real finding, not a bug to tune away."
        ),
        "real_mean": round(float(r_val.mean()), 2),
        "sim_mean": round(float(s_val.mean()), 2),
        "real_median": round(float(np.median(r_val)), 2),
        "sim_median": round(float(np.median(s_val)), 2),
        "real_p90": round(float(np.percentile(r_val, 90)), 2),
        "sim_p90": round(float(np.percentile(s_val, 90)), 2),
        **_ks(r_val, s_val),
        # Level and shape are different claims and deserve separate numbers. The
        # real orders are one 2016 store; the simulation runs 2023-2025 with three
        # years of price inflation and regional multipliers up to 1.14, so the
        # LEVEL is expected to sit higher and matching it would be suspicious. What
        # the generator should get right is the SHAPE, so both samples are divided
        # by their own median and compared again.
        "ks_statistic_shape_only": _ks(
            r_val / np.median(r_val), s_val / np.median(s_val)
        )["ks_statistic"],
        "sim_over_real_median_ratio": round(
            float(np.median(s_val) / np.median(r_val)), 3
        ),
    }

    out["_caveat"] = {
        "real_orders": int(len(real["orders"])),
        "real_stores": int(real["orders"]["store_id"].nunique()),
        "real_days": int(real["orders"]["ordered_at"].dt.date.nunique()),
        "note": (
            "The real sample is one store over sixteen days in 2016. It is enough "
            "to anchor the shape of these distributions and not enough to validate "
            "a three-year six-store simulation. Prices have also moved since 2016, "
            "so the level of order value is expected to drift upward."
        ),
    }
    return out


def _ks(a: np.ndarray, b: np.ndarray) -> dict:
    """
    Two-sample Kolmogorov-Smirnov.

    The D statistic is what to read: it is the largest gap between the two
    cumulative distributions, on a 0-1 scale, and it does not care how many rows
    there are. The p-value is reported but should be ignored -- with hundreds of
    thousands of simulated rows against 686 real ones it is driven by sample size,
    not by whether the distributions are close enough to be useful.
    """
    res = stats.ks_2samp(a, b)
    return {"ks_statistic": round(float(res.statistic), 4),
            "ks_pvalue": float(f"{res.pvalue:.3g}")}


def write_report(result: dict, path: Path) -> Path:
    lines = [
        "# Simulator calibration",
        "",
        "Generated by `python -m ml.calibration`. Compares the generated feed against "
        f"the {result['_caveat']['real_orders']} real Jaffle Shop orders in "
        "`calibration/real_seeds/`.",
        "",
        "The KS **D statistic** is the number to read: the largest gap between the two "
        "cumulative distributions, on a 0-1 scale. P-values are shown for completeness "
        "and should be ignored -- against hundreds of thousands of simulated rows they "
        "measure sample size, not similarity.",
        "",
        "| Check | Status | Real | Simulated | KS D |",
        "|---|---|---:|---:|---:|",
        f"| Mean items per order | calibrated | {result['basket_size']['real_mean']} | "
        f"{result['basket_size']['sim_mean']} | {result['basket_size']['ks_statistic']} |",
        f"| Beverage share of units | calibrated | {result['sku_mix']['real_beverage_share']:.1%} | "
        f"{result['sku_mix']['sim_beverage_share']:.1%} | n/a (TVD "
        f"{result['sku_mix']['total_variation_distance']}) |",
        f"| Share of orders before 10am | calibrated | "
        f"{result['arrival_hour']['real_share_before_10am']:.1%} | "
        f"{result['arrival_hour']['sim_share_before_10am']:.1%} | "
        f"{result['arrival_hour']['ks_statistic']} |",
        f"| **Median order value** | **emergent** | ${result['order_value']['real_median']} | "
        f"${result['order_value']['sim_median']} | {result['order_value']['ks_statistic']} |",
        f"| **Mean order value** | **emergent** | ${result['order_value']['real_mean']} | "
        f"${result['order_value']['sim_mean']} | (as above) |",
        f"| **Order value, shape only** | **emergent** | \u2014 | "
        f"{result['order_value']['sim_over_real_median_ratio']}x the real median | "
        f"{result['order_value']['ks_statistic_shape_only']} |",
        "",
        "The last row divides each sample by its own median before comparing. Level and "
        "shape are separate claims: the simulation runs 2023-2025 with three years of "
        "price inflation off a 2016 catalogue and regional multipliers up to 1.14, so the "
        "level is *expected* to sit higher and matching it exactly would be suspicious. "
        "The shape is what the generator should get right.",
        "",
        "**Calibrated** rows were fitted to the real data, so agreement is expected and "
        "is not evidence of anything. **Emergent** rows were never fitted -- order value "
        "falls out of basket size, sku mix and catalogue prices -- so they are the only "
        "genuine out-of-sample check here.",
        "",
        "## Caveat",
        "",
        f"The real sample is {result['_caveat']['real_orders']} orders from "
        f"{result['_caveat']['real_stores']} store over {result['_caveat']['real_days']} days "
        "in 2016. That is enough to anchor the shape of these distributions and not enough "
        "to validate a three-year, six-store simulation. Prices have moved since 2016, so "
        "the level of order value is expected to drift upward.",
        "",
        "```json",
        json.dumps(result, indent=2),
        "```",
        "",
    ]
    path.write_text("\n".join(lines))
    return path


def main() -> None:
    result = compare()
    out = ROOT / "reports" / "calibration.md"
    out.parent.mkdir(exist_ok=True)
    write_report(result, out)
    for key in ("basket_size", "sku_mix", "arrival_hour", "order_value"):
        r = result[key]
        tag = r["status"].upper()
        d = r.get("ks_statistic", r.get("total_variation_distance"))
        print(f"  {key:<14} [{tag:<10}] D = {d}")
    print(f"  report -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
