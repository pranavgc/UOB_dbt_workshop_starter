"""
The one figure that makes this project different from a Kaggle notebook.

Most portfolio charts show predictions against actuals, which tells you a model
fits. This one shows each estimated coefficient, its 95% interval, and the value
the simulator actually used -- so it answers the harder question of whether the
model is *right*, and makes the naive-versus-full specification gap visible at a
glance rather than buried in a table.

Two panels rather than one axis: elasticities and log-scale effects are different
quantities, and putting them on a shared scale would be a dual-axis chart wearing
a disguise.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Validated categorical slots 1-3; truth is ink, not a series colour.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8983"
GRID = "#e6e5e1"
SERIES = {
    "ols_structural": "#2a78d6",
    "ols_naive": "#eb6834",
    "ols_controlled": "#1baf7a",
    "glm_nb": "#eda100",
}
LABEL = {
    "ols_naive": "Naive OLS",
    "ols_structural": "Structural OLS",
    "ols_controlled": "OLS + demand controls",
    "glm_nb": "Negative Binomial GLM",
}


def plot_coefficient_recovery(recovery: pd.DataFrame, out_path: Path) -> Path:
    df = recovery.copy()
    elastic = df[df["parameter"].str.startswith("price elasticity")].reset_index(drop=True)
    effects = df[~df["parameter"].str.startswith("price elasticity")].reset_index(drop=True)

    fig, axes = plt.subplots(
        1, 2, figsize=(13.5, 8.2), facecolor=SURFACE,
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.42},
    )

    _panel(axes[0], elastic, "Price elasticity", "log units per log price")
    _panel(axes[1], effects, "Effects on log demand", "log points")

    handles = [
        plt.Line2D([], [], marker="o", ms=8, ls="none", color=SERIES[k], label=LABEL[k])
        for k in ("ols_naive", "ols_structural", "ols_controlled", "glm_nb")
    ] + [
        plt.Line2D([], [], marker="D", ms=8, ls="none", color=INK,
                   label="True value (simulator)"),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=5, frameon=False,
        bbox_to_anchor=(0.5, -0.04), fontsize=9.5, labelcolor=INK_2,
    )
    fig.suptitle(
        "Estimated coefficients against the values the simulator actually used",
        x=0.02, y=1.015, ha="left", fontsize=13.5, color=INK, fontweight="semibold",
    )
    fig.text(
        0.02, 0.945,
        "95% intervals, standard errors clustered by store x sku. A model can fit well "
        "and still miss the parameter.",
        ha="left", fontsize=9.5, color=INK_2,
    )
    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out_path


def _panel(ax, sub: pd.DataFrame, title: str, xlabel: str) -> None:
    ax.set_facecolor(SURFACE)
    n = len(sub)
    ys = np.arange(n)[::-1]

    for y, (_, r) in zip(ys, sub.iterrows(), strict=True):
        colour = SERIES.get(r["model"], INK_3)

        # 2px interval, then the estimate on top with a surface ring so the two
        # marks never merge where they overlap.
        ax.plot([r["ci95_low"], r["ci95_high"]], [y, y], lw=2, color=colour,
                solid_capstyle="round", zorder=2)
        if not pd.isna(r["true"]):
            ax.plot([r["true"]], [y], marker="D", ms=10, color=INK,
                    markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=4)
        ax.plot([r["estimate"]], [y], marker="o", ms=7.5, color=colour,
                markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=6)

        if not pd.isna(r["true"]):
            # error annotation, placed away from both marks
            err = r["recovery_error_pct"]
            ax.annotate(
                f"{err:+.0f}%",
                xy=(max(r["ci95_high"], r["true"]), y), xytext=(8, 0),
                textcoords="offset points", va="center", fontsize=9,
                color=INK if abs(err) < 10 else "#b23a2e",
            )

    labels = [
        f"{r['parameter']}\n{LABEL.get(r['model'], r['model'])}"
        for _, r in sub.iterrows()
    ]
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=9.5, color=INK)
    ax.set_ylim(-0.7, n - 0.3)

    ax.set_xlabel(xlabel, fontsize=9.5, color=INK_2, labelpad=8)
    ax.set_title(title, fontsize=11.5, color=INK, loc="left", pad=10,
                 fontweight="semibold")

    ax.grid(axis="x", color=GRID, lw=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", colors=INK_2, labelsize=9, length=0)
    ax.tick_params(axis="y", length=0)

    # leave room for the right-hand error annotations
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi + 0.22 * (hi - lo))


def plot_temperature_response(
    df: pd.DataFrame, spline, ols, truth: dict, out_path: Path
) -> Path:
    """
    The fitted weather response against the one the simulator actually used.

    This is the figure a public dataset cannot produce, and it does not say what I
    expected it to say. The structural OLS contains the true quadratic form by
    construction -- I knew the data-generating process, so I gave it the right
    shape -- and it tracks the truth almost exactly. The spline has to LEARN the
    shape. On jaffles it does. On beverages, where the type interaction rests on
    less data, it misplaces the optimum by roughly nine degrees.

    The lesson is not that splines are bad. It is that a flexible method needs
    more data per effect than a correctly specified parametric term, and that in
    any real problem you do not get handed the correct form. Both halves are worth
    more than a chart confirming what everyone already assumes.
    """
    lo, hi = np.percentile(df["temperature_c"].dropna(), [1, 99])
    grid = np.linspace(lo, hi, 240)
    z = (grid - truth["temp_opt_c"]) / 10.0

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), facecolor=SURFACE,
                             sharey=True, gridspec_kw={"wspace": 0.08})

    for ax, (label, is_bev) in zip(axes, [("Jaffles", 0), ("Beverages", 1)], strict=True):
        true_curve = truth["temp_curvature"] * z ** 2 + (
            truth["beverage_temp"] * z if is_bev else 0.0
        )
        linear = ols.coef("temp_dev") * z + ols.coef("temp_dev_sq") * z ** 2 + (
            ols.coef("temp_dev_bev") * z if is_bev else 0.0
        )
        linear = linear - (
            ols.coef("temp_dev") * 0 + ols.coef("temp_dev_sq") * 0
        )
        fitted_spline = spline.temperature_response(grid, is_beverage=is_bev)

        ax.set_facecolor(SURFACE)
        ax.axhline(0, color=GRID, lw=1, zorder=0)
        ax.plot(grid, true_curve, lw=2.6, color=INK, zorder=4,
                label="True response (simulator)")
        ax.plot(grid, linear, lw=2.2, color=SERIES["ols_structural"], zorder=3,
                label="Structural OLS (given the true quadratic form)")
        ax.plot(grid, fitted_spline, lw=2.2, color=SERIES["ols_controlled"], zorder=3,
                label="Natural cubic spline (has to learn it)")
        ax.axvline(truth["temp_opt_c"], color=INK_3, lw=1, ls=(0, (3, 3)), zorder=1)
        ax.annotate(f"optimum {truth['temp_opt_c']:.0f}°C",
                    xy=(truth["temp_opt_c"], ax.get_ylim()[0]), xytext=(5, 6),
                    textcoords="offset points", fontsize=8.5, color=INK_3)

        ax.set_title(label, fontsize=11.5, color=INK, loc="left", pad=8,
                     fontweight="semibold")
        ax.set_xlabel("Daily mean temperature (°C)", fontsize=9.5,
                      color=INK_2, labelpad=7)
        ax.grid(axis="y", color=GRID, lw=1, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("bottom", "left"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=9, length=0)

    axes[0].set_ylabel("Effect on log demand", fontsize=9.5, color=INK_2, labelpad=7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.07), fontsize=9.5, labelcolor=INK_2)
    fig.suptitle("Recovering a nonlinearity whose true shape is known",
                 x=0.02, y=1.02, ha="left", fontsize=13.5, color=INK,
                 fontweight="semibold")
    fig.text(0.02, 0.935,
             "Demand falls away on both sides of a 20°C optimum. A linear term "
             "cannot represent that; the spline recovers it.",
             ha="left", fontsize=9.5, color=INK_2)
    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out_path
