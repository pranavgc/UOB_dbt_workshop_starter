"""
Generate the Jaffle Demand Lab source feed.

This is a structural simulator, not a random-number dump. It builds a log-linear
demand system with a known price elasticity, a known featuring lift, a known
seasonal shape and a known store-ramp effect, then emits the raw tables a real
point-of-sale system would hand a warehouse team.

The coefficients it uses are written to sim/ground_truth.json. Everything
downstream -- dbt, the feature mart, the model suite -- is then evaluated on
whether it can recover those numbers from the transactions alone.

Deliberately NOT emitted: the latent demand rate lambda. Only the observable
transactions, weather, prices and placements reach the warehouse, exactly as
they would in production.

Usage
-----
    python -m sim.generate                     # full run, writes parquet + duckdb
    python -m sim.generate --years 1           # shorter window while iterating
    python -m sim.generate --no-duckdb         # parquet only
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sim.defects import inject as inject_defects

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def uuid_array(rng: np.random.Generator, n: int) -> np.ndarray:
    """Vectorised RFC-4122-shaped identifiers. Deterministic under the seed."""
    raw = rng.integers(0, 256, size=(n, 16), dtype=np.uint8)
    hexed = np.array([f'{int.from_bytes(row.tobytes(), "big"):032x}' for row in raw])
    return np.char.add(
        np.char.add(
            np.char.add(
                np.char.add(
                    np.char.add([h[0:8] for h in hexed], "-"),
                    [h[8:12] for h in hexed],
                ),
                "-",
            ),
            np.char.add(np.char.add([h[12:16] for h in hexed], "-"), [h[16:20] for h in hexed]),
        ),
        np.char.add("-", [h[20:32] for h in hexed]),
    )


def truncated_geometric(rng: np.random.Generator, p: float, size: int, cap: int) -> np.ndarray:
    """Zero-truncated geometric basket sizes, clipped at `cap`."""
    return np.minimum(rng.geometric(p, size=size), cap)


@dataclass
class Catalogue:
    stores: pd.DataFrame
    products: pd.DataFrame
    supplies: pd.DataFrame


def load_catalogue() -> Catalogue:
    ref = ROOT / "seeds" / "reference"
    stores = pd.read_csv(ref / "raw_stores.csv")
    products = pd.read_csv(ref / "raw_products.csv")
    supplies = pd.read_csv(ref / "raw_supplies.csv")
    stores.columns = [c.lower() for c in stores.columns]
    products.columns = [c.lower() for c in products.columns]
    supplies.columns = [c.lower() for c in supplies.columns]
    return Catalogue(stores=stores, products=products, supplies=supplies)


# --------------------------------------------------------------------------- #
# components of the DGP
# --------------------------------------------------------------------------- #
def simulate_weather(cfg, rng, store_names, dates) -> np.ndarray:
    """(n_days, n_stores) daily mean temperature in degrees C."""
    n_days, n_stores = len(dates), len(store_names)
    doy = np.array([d.dayofyear for d in dates], dtype=float)
    temps = np.zeros((n_days, n_stores))
    rho = cfg["weather"]["ar1_rho"]
    sigma = cfg["weather"]["ar1_sigma"]

    for j, name in enumerate(store_names):
        c = cfg["weather"]["cities"][name]
        seasonal = c["annual_mean"] + c["annual_amplitude"] * np.cos(
            2 * np.pi * (doy - c["peak_doy"]) / 365.25
        )
        anomaly = np.zeros(n_days)
        eps = rng.normal(0.0, sigma, size=n_days)
        for t in range(1, n_days):
            anomaly[t] = rho * anomaly[t - 1] + eps[t]
        temps[:, j] = seasonal + anomaly
    return temps


def simulate_prices(cfg, rng, store_names, products, dates) -> np.ndarray:
    """
    (n_days, n_stores, n_skus) integer list price in cents.

    Three independent sources of variation, so that the elasticity is not
    identified off a single trend: chain-wide inflation, a drifting regional
    multiplier per store, and discrete per-sku repricing events.
    """
    n_days, n_stores, n_skus = len(dates), len(store_names), len(products)
    pr = cfg["pricing"]

    # chain-wide inflation, compounded daily from a monthly rate
    daily_infl = (1 + pr["monthly_inflation"]) ** (1 / 30.44)
    inflation = daily_infl ** np.arange(n_days)

    # per-store regional multiplier following a small random walk
    store_mult = np.zeros((n_days, n_stores))
    for j, name in enumerate(store_names):
        steps = rng.normal(0.0, pr["store_multiplier_drift_sd"], size=n_days)
        store_mult[:, j] = pr["store_multiplier"][name] * np.exp(np.cumsum(steps))

    # discrete repricing events per sku, as a piecewise-constant log deviation
    sku_dev = np.zeros((n_days, n_skus))
    lam = 1.0 / pr["reprice"]["mean_interval_days"]
    cap = pr["reprice"]["max_cumulative_log_dev"]
    for k in range(n_skus):
        dev, current = np.zeros(n_days), 0.0
        events = rng.random(n_days) < lam
        for t in range(n_days):
            if events[t]:
                current = float(np.clip(current + rng.normal(0, pr["reprice"]["jump_sd"]), -cap, cap))
            dev[t] = current
        sku_dev[:, k] = dev

    base = products["price"].to_numpy(dtype=float)  # cents
    price = (
        base[None, None, :]
        * inflation[:, None, None]
        * store_mult[:, :, None]
        * np.exp(sku_dev)[:, None, :]
    )
    return np.rint(price).astype(np.int64)


def build_design(cfg, dates, store_names, products, temps, open_day_index):
    """Everything in the linear predictor that does not depend on the simulation loop."""
    coef = cfg["coefficients"]
    n_days = len(dates)

    doy = np.array([d.dayofyear for d in dates], dtype=float)
    dow_idx = np.array([d.weekday() for d in dates])

    # annual seasonality, two harmonics
    f = coef["fourier"]
    season = (
        f["a1"] * np.sin(2 * np.pi * doy / 365.25)
        + f["b1"] * np.cos(2 * np.pi * doy / 365.25)
        + f["a2"] * np.sin(4 * np.pi * doy / 365.25)
        + f["b2"] * np.cos(4 * np.pi * doy / 365.25)
    )
    dow = np.array(coef["dow"], dtype=float)[dow_idx]

    alpha = np.array([coef["alpha_store"][n] for n in store_names], dtype=float)

    # Per-sku baseline popularity, fitted from the real seed mix. See config.yaml.
    pop = np.array(
        [coef.get("sku_popularity", {}).get(sku, 0.0) for sku in products["sku"]],
        dtype=float,
    )

    is_bev = (products["type"].to_numpy() == "beverage").astype(float)
    beta = np.where(is_bev == 1.0, coef["beta_price"]["beverage"], coef["beta_price"]["jaffle"])

    # temperature: quadratic penalty away from the optimum, plus a beverage tilt
    z = (temps - coef["temp_opt_c"]) / 10.0                     # (n_days, n_stores)
    temp_common = coef["temp_curvature"] * z ** 2               # (n_days, n_stores)
    temp_bev = coef["beverage_temp"] * z                        # (n_days, n_stores)

    # store ramp-up: 0 -> 1 over ramp_days after opening
    ramp_days = cfg["stores"]["ramp_days"]
    day_idx = np.arange(n_days)[:, None]
    since_open = day_idx - np.array(open_day_index)[None, :]
    ramp = np.clip(since_open / ramp_days, 0.0, 1.0)
    is_open = since_open >= 0

    # base part of the predictor, shape (n_days, n_stores, n_skus)
    base_eta = (
        alpha[None, :, None]
        + season[:, None, None]
        + dow[:, None, None]
        + coef["ramp"] * ramp[:, :, None]
        + temp_common[:, :, None]
        + temp_bev[:, :, None] * is_bev[None, None, :]
        + pop[None, None, :]
    )
    return base_eta, beta, is_bev, is_open, ramp, dow_idx, doy


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def generate(
    years: float | None = None,
    write_duckdb: bool = True,
    output_dir: Path | None = None,
    config_path: Path | None = None,
) -> dict:
    t0 = time.time()
    cfg = yaml.safe_load(Path(config_path or CONFIG_PATH).read_text())
    rng = np.random.default_rng(cfg["seed"])
    coef = cfg["coefficients"]

    cat = load_catalogue()
    stores = cat.stores.sort_values("opened_at").reset_index(drop=True)
    products = cat.products.sort_values("sku").reset_index(drop=True)
    store_names = stores["name"].tolist()
    store_ids = stores["id"].tolist()
    tax_rates = stores["tax_rate"].to_numpy(dtype=float)
    skus = products["sku"].tolist()
    n_stores, n_skus = len(store_names), len(skus)

    start = pd.Timestamp(cfg["window"]["start_date"])
    end = pd.Timestamp(cfg["window"]["end_date"])
    if years is not None:
        end = start + pd.Timedelta(days=int(round(365.25 * years)) - 1)
    dates = pd.date_range(start, end, freq="D")
    n_days = len(dates)

    # store opening dates come from config, not the catalogue -- see config.yaml
    open_day_index = []
    for name in store_names:
        opened = pd.Timestamp(cfg["stores"]["opened_at"][name])
        open_day_index.append(int((opened - start).days))

    # Tax rate in effect per store per day. The catalogue treats tax_rate as a
    # static store attribute; it is not, and the pipeline has to handle that.
    tax_by_day = np.tile(tax_rates, (n_days, 1))
    tax_changes = [
        {"store_id": store_ids[i], "effective_date": start, "tax_rate": float(tax_rates[i])}
        for i in range(n_stores)
    ]
    for change in cfg.get("tax", {}).get("changes", []):
        j = store_names.index(change["store"])
        eff = pd.Timestamp(change["effective_date"])
        if start <= eff <= end:
            tax_by_day[int((eff - start).days):, j] = float(change["new_rate"])
            tax_changes.append({
                "store_id": store_ids[j],
                "effective_date": eff,
                "tax_rate": float(change["new_rate"]),
            })

    arrivals = cfg["arrivals"]
    arr_w = np.array([c["weight"] for c in arrivals["components"]], dtype=float)
    arr_w = arr_w / arr_w.sum()
    arr_mu = np.array([c["mean_hour"] * 60 for c in arrivals["components"]], dtype=float)
    arr_sd = np.array([c["sd_hours"] * 60 for c in arrivals["components"]], dtype=float)
    open_min, close_min = arrivals["open_hour"] * 60, arrivals["close_hour"] * 60

    temps = simulate_weather(cfg, rng, store_names, dates)
    prices = simulate_prices(cfg, rng, store_names, products, dates)
    base_price = products["price"].to_numpy(dtype=float)

    base_eta, beta, is_bev, is_open, ramp, dow_idx, doy = build_design(
        cfg, dates, store_names, products, temps, open_day_index
    )

    # product types -> integer index, used by the featuring rule
    types = products["type"].to_numpy()
    type_names = sorted(set(types))
    type_idx = np.array([type_names.index(t) for t in types])
    n_types = len(type_names)

    promo = cfg["promotions"]
    nb_size = float(coef["nb_dispersion"])

    units = np.zeros((n_days, n_stores, n_skus), dtype=np.int64)
    featured = np.zeros((n_days, n_stores, n_skus), dtype=bool)
    campaign_id = np.full((n_days, n_stores, n_skus), "", dtype=object)

    # Persistent unobserved demand shock, per store x product type. Never emitted.
    latent = np.zeros((n_stores, n_types))
    rho = float(coef["latent_shock_rho"])
    sigma_u = float(coef["latent_shock_sigma"])

    active_until = np.full((n_stores, n_skus), -1, dtype=int)
    active_campaign = np.full((n_stores, n_skus), "", dtype=object)
    type_units_hist = np.zeros((n_days, n_stores, n_types))
    campaign_counter = 0

    # ---------------- the simulation loop -----------------------------------
    # Sequential because featuring at day t depends on realised demand before t.
    # That dependency is the confounder the project is built around.
    for t in range(n_days):
        # ---- advance the unobserved shock -----------------------------------
        latent = rho * latent + rng.normal(0.0, sigma_u, size=latent.shape)
        latent_by_sku = latent[:, type_idx]          # (n_stores, n_skus)

        # ---- weekly featuring decision, made from TRAILING performance only --
        if t % 7 == 0 and t >= promo["warmup_days"]:
            w7 = type_units_hist[max(0, t - 7):t].mean(axis=0)    # (n_stores, n_types)
            w28 = type_units_hist[max(0, t - 28):t].mean(axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(w28 > 0, w7 / w28, 1.0)
            weak = ratio < promo["weak_ratio_threshold"]
            prob = np.where(weak, promo["prob_if_weak"], promo["prob_if_normal"])
            fire = rng.random(prob.shape) < prob
            for s in range(n_stores):
                if t < open_day_index[s]:
                    continue
                for ty in range(n_types):
                    if not fire[s, ty]:
                        continue
                    campaign_counter += 1
                    cid = f"CMP-{campaign_counter:05d}"
                    candidates = np.flatnonzero(type_idx == ty)
                    chosen = rng.choice(
                        candidates,
                        size=min(promo["skus_per_campaign"], len(candidates)),
                        replace=False,
                    )
                    active_until[s, chosen] = t + promo["campaign_days"] - 1
                    active_campaign[s, chosen] = cid

        today_featured = active_until >= t
        featured[t] = today_featured
        campaign_id[t] = np.where(today_featured, active_campaign, "")

        # ---- demand ---------------------------------------------------------
        log_price_rel = np.log(prices[t] / base_price[None, :])   # (n_stores, n_skus)
        eta = (
            base_eta[t]
            + beta[None, :] * log_price_rel
            + coef["gamma_featured"] * today_featured
            + latent_by_sku
        )
        mu = np.exp(eta)
        mu = np.where(is_open[t][:, None], mu, 0.0)

        drawn = np.zeros_like(mu, dtype=np.int64)
        live = mu > 0
        if live.any():
            m = mu[live]
            p = nb_size / (nb_size + m)
            drawn[live] = rng.negative_binomial(nb_size, p)
        units[t] = drawn

        for ty in range(n_types):
            type_units_hist[t, :, ty] = drawn[:, type_idx == ty].sum(axis=1)

    print(f"  demand loop done in {time.time() - t0:.1f}s, {units.sum():,} units")

    # ---------------- customers ---------------------------------------------
    n_cust = cfg["customers"]["n_customers"]
    cust_ids = uuid_array(rng, n_cust)
    first = ["Alex", "Sam", "Jordan", "Riley", "Casey", "Morgan", "Taylor", "Jamie",
             "Avery", "Quinn", "Rowan", "Skyler", "Emerson", "Finley", "Hayden",
             "Kai", "Logan", "Micah", "Noel", "Parker", "Reese", "Sage", "Tatum"]
    last = ["Ahmed", "Byrne", "Chen", "Diaz", "Eriksen", "Fontaine", "Gupta", "Haddad",
            "Ivanov", "Jensen", "Kowalski", "Lindqvist", "Mbeki", "Nakamura", "Okafor",
            "Petrov", "Quaranta", "Rossi", "Silva", "Tanaka", "Ueda", "Vargas", "Weiss"]
    cust_names = [
        f"{first[i % len(first)]} {last[(i // len(first)) % len(last)]}" for i in range(n_cust)
    ]
    cust_weights = rng.dirichlet(np.full(n_cust, 1.0 / cfg["customers"]["repeat_concentration"]))

    # ---------------- assemble orders and line items -------------------------
    p_basket = 1.0 / cfg["baskets"]["mean_items_per_order"]
    cap = cfg["baskets"]["max_items_per_order"]

    order_rows, item_rows = [], []
    date_values = dates.to_numpy()

    for t in range(n_days):
        for s in range(n_stores):
            row = units[t, s]
            total = int(row.sum())
            if total == 0:
                continue
            pool = np.repeat(np.arange(n_skus), row)
            rng.shuffle(pool)

            sizes = truncated_geometric(rng, p_basket, size=int(total / p_basket * 1.2) + 8, cap=cap)
            edges = np.cumsum(sizes)
            keep = int(np.searchsorted(edges, total)) + 1
            sizes = sizes[:keep]
            sizes[-1] = total - (edges[keep - 2] if keep >= 2 else 0)
            sizes = sizes[sizes > 0]
            n_orders = len(sizes)
            if n_orders == 0:
                continue

            splits = np.split(pool, np.cumsum(sizes)[:-1])
            line_price = prices[t, s]

            subtotal = np.array([int(line_price[b].sum()) for b in splits], dtype=np.int64)
            tax = np.rint(subtotal * tax_by_day[t, s]).astype(np.int64)

            # intra-day arrival times, from the calibrated mixture in config.yaml
            pick = rng.choice(len(arr_w), size=n_orders, p=arr_w)
            minutes = rng.normal(arr_mu[pick], arr_sd[pick])
            minutes = np.clip(minutes, open_min, close_min).astype(int)
            ts = date_values[t] + minutes.astype("timedelta64[m]")

            oids = uuid_array(rng, n_orders)
            cids = rng.choice(cust_ids, size=n_orders, p=cust_weights)

            order_rows.append(
                pd.DataFrame(
                    {
                        "id": oids,
                        "customer": cids,
                        "ordered_at": ts,
                        "store_id": store_ids[s],
                        "subtotal": subtotal,
                        "tax_paid": tax,
                        "order_total": subtotal + tax,
                    }
                )
            )
            flat_sku = np.concatenate(splits)
            item_rows.append(
                pd.DataFrame(
                    {
                        "id": uuid_array(rng, total),
                        "order_id": np.repeat(oids, sizes),
                        "sku": [skus[i] for i in flat_sku],
                    }
                )
            )

    raw_orders = pd.concat(order_rows, ignore_index=True)
    raw_items = pd.concat(item_rows, ignore_index=True)
    raw_customers = pd.DataFrame({"id": cust_ids, "name": cust_names})

    # ---------------- the observable side tables -----------------------------
    dd = np.repeat(dates.values, n_stores)
    ss = np.tile(store_ids, n_days)
    raw_weather = pd.DataFrame(
        {"store_id": ss, "weather_date": dd, "temperature_c": np.round(temps.reshape(-1), 2)}
    )

    price_long = pd.DataFrame(
        {
            "store_id": np.repeat(np.tile(store_ids, n_days), n_skus),
            "sku": np.tile(skus, n_days * n_stores),
            "price_date": np.repeat(dates.values, n_stores * n_skus),
            "price": prices.reshape(-1),
        }
    )

    fmask = featured.reshape(-1)
    raw_featured = pd.DataFrame(
        {
            "store_id": np.repeat(np.tile(store_ids, n_days), n_skus)[fmask],
            "sku": np.tile(skus, n_days * n_stores)[fmask],
            "placement_date": np.repeat(dates.values, n_stores * n_skus)[fmask],
            "campaign_id": campaign_id.reshape(-1)[fmask].astype(str),
        }
    )

    # Opening dates are set in config.yaml, not taken from the store catalogue
    # (see the note there), so the simulator emits them as source metadata.
    # dim_store joins this to the catalogue rather than trusting the seed.
    raw_store_openings = pd.DataFrame(
        {
            "store_id": store_ids,
            "opened_at": [pd.Timestamp(cfg["stores"]["opened_at"][n]) for n in store_names],
        }
    )

    raw_tax_rate_changes = pd.DataFrame(tax_changes).sort_values(
        ["store_id", "effective_date"]
    ).reset_index(drop=True)

    tables = {
        "raw_store_openings": raw_store_openings,
        "raw_tax_rate_changes": raw_tax_rate_changes,
        "raw_customers": raw_customers,
        "raw_orders": raw_orders,
        "raw_items": raw_items,
        "raw_weather": raw_weather,
        "raw_price_snapshots": price_long,
        "raw_featured_placements": raw_featured,
    }

    # ---------------- inject the deliberate defects ---------------------------
    # Everything above this line is a clean feed. Everything a real source system
    # would get wrong happens here, and is recorded in the manifest.
    tables, defect_manifest = inject_defects(
        tables, cfg, rng, dict(zip(store_names, store_ids, strict=True))
    )

    # ---------------- write --------------------------------------------------
    out = cfg["output"]
    # `output_dir` lets the test suite generate into a tmp_path without
    # clobbering the developer's data/ directory or ground_truth.json.
    base = Path(output_dir) if output_dir else ROOT
    raw_dir = base / out["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(raw_dir / f"{name}.parquet", index=False)

    if write_duckdb:
        import duckdb

        db_path = base / out["duckdb_path"]
        db_path.parent.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(str(db_path))
        con.execute(f"create schema if not exists {out['raw_schema']}")
        for name, df in tables.items():
            con.register("_tmp", df)
            con.execute(f"create or replace table {out['raw_schema']}.{name} as select * from _tmp")
            con.unregister("_tmp")
        con.close()

    # ---------------- ground truth ------------------------------------------
    truth = {
        "_comment": (
            "True parameters of the data-generating process. Models are scored on "
            "recovering these, not only on held-out error. Never joined into the "
            "feature mart -- see README."
        ),
        "seed": cfg["seed"],
        "window": {"start": str(dates[0].date()), "end": str(dates[-1].date()), "days": n_days},
        "scale": {
            "orders": int(len(tables["raw_orders"])),
            "order_lines": int(len(tables["raw_items"])),
            "stores": n_stores,
            "skus": n_skus,
            "customers": n_cust,
        },
        "coefficients": {
            "alpha_store": coef["alpha_store"],
            "beta_price_jaffle": coef["beta_price"]["jaffle"],
            "beta_price_beverage": coef["beta_price"]["beverage"],
            "gamma_featured": coef["gamma_featured"],
            "sku_popularity": coef.get("sku_popularity", {}),
            "temp_opt_c": coef["temp_opt_c"],
            "temp_curvature": coef["temp_curvature"],
            "beverage_temp": coef["beverage_temp"],
            "dow": dict(zip(WEEKDAY_NAMES, coef["dow"], strict=True)),
            "fourier": coef["fourier"],
            "ramp": coef["ramp"],
            "nb_dispersion": coef["nb_dispersion"],
            "latent_shock_rho": coef["latent_shock_rho"],
            "latent_shock_sigma": coef["latent_shock_sigma"],
        },
        "injected_defects": defect_manifest,
        "identification_notes": {
            "beta_price": (
                "Identified off three independent price sources: chain inflation, a "
                "drifting per-store regional multiplier, and discrete per-sku repricing "
                "events. Elasticity differs by product type, so a pooled coefficient is "
                "misspecified and should land between the two true values."
            ),
            "gamma_featured": (
                "CONFOUNDED BY DESIGN. A persistent AR(1) demand shock (rho = 0.90) sits in "
                "the error term and is never emitted. Featuring is scheduled when trailing "
                "demand is weak, and because the shock persists, a weak last week implies a "
                "weak today -- so featuring is correlated with the CURRENT error and naive "
                "OLS understates the lift. Conditioning on the lagged trailing-demand "
                "features in mart_ml_features proxies for the shock and recovers it. Note "
                "that without persistence there would be no bias at all."
            ),
            "temp_curvature": (
                "The temperature response is quadratic. A linear term cannot represent it, "
                "so splines and gradient boosting should beat OLS on this feature for a "
                "reason that is knowable in advance."
            ),
            "nb_dispersion": (
                "Counts are Negative Binomial, so the variance exceeds the mean. A Poisson "
                "GLM is the wrong family and the dispersion statistic should say so."
            ),
        },
    }
    gt_path = base / out["ground_truth_path"]
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    gt_path.write_text(json.dumps(truth, indent=2))

    injected = sum(m.get("rows", 0) for m in defect_manifest)
    print(f"  wrote {len(tables['raw_orders']):,} orders / {len(tables['raw_items']):,} lines "
          f"in {time.time() - t0:.1f}s ({injected:,} rows carry an injected defect)")
    return {"tables": {k: len(v) for k, v in tables.items()},
            "frames": tables, "truth": truth}


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the Jaffle Demand Lab source feed.")
    ap.add_argument("--years", type=float, default=None, help="shorten the window while iterating")
    ap.add_argument("--no-duckdb", action="store_true", help="write parquet only")
    args = ap.parse_args()
    print("Generating source feed...")
    res = generate(years=args.years, write_duckdb=not args.no_duckdb)
    for name, n in res["tables"].items():
        print(f"    {name:<26} {n:>10,}")


if __name__ == "__main__":
    main()
