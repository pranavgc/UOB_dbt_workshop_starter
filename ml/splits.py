"""
Time-aware splitting.

The whole project is a forecasting problem, so a random split is not a mistake of
degree -- it is the wrong experiment. Training on Wednesday to predict the
preceding Tuesday tells you nothing about whether the model will work next month,
and it leaks: the trailing-demand features of a test row overlap the target of a
training row a few days earlier.

`rolling_origin` is what every model is scored on. `random_kfold` exists only so
the README can report, as a number, how much the wrong split would have
flattered us. It is never used to select a model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    name: str
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_end: pd.Timestamp
    test_end: pd.Timestamp


def holdout_split(
    df: pd.DataFrame, holdout_days: int = 90
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carve off the final period. Scored exactly once, at the end.

    Nothing in model selection, feature choice or hyper-parameters may look at
    this. That discipline is the only thing that makes the final number mean
    anything.
    """
    cutoff = df["order_date"].max() - pd.Timedelta(days=holdout_days - 1)
    development = df[df["order_date"] < cutoff].reset_index(drop=True)
    holdout = df[df["order_date"] >= cutoff].reset_index(drop=True)
    return development, holdout


def rolling_origin(
    df: pd.DataFrame,
    n_folds: int = 4,
    test_days: int = 60,
    min_train_days: int = 240,
) -> list[Fold]:
    """
    Expanding-window folds: train on everything up to t, test on (t, t + test_days].

    Each fold's training set grows, which mirrors how the model would actually be
    retrained in production, and no fold ever sees data from its own future.
    """
    dates = np.sort(df["order_date"].unique())
    if len(dates) < min_train_days + n_folds * test_days:
        raise ValueError(
            f"need at least {min_train_days + n_folds * test_days} days, have {len(dates)}"
        )

    folds: list[Fold] = []
    for k in range(n_folds):
        test_end_pos = len(dates) - (n_folds - 1 - k) * test_days - 1
        test_start_pos = test_end_pos - test_days + 1
        train_end = pd.Timestamp(dates[test_start_pos - 1])
        test_end = pd.Timestamp(dates[test_end_pos])

        train_idx = np.flatnonzero(df["order_date"].values <= np.datetime64(train_end))
        test_idx = np.flatnonzero(
            (df["order_date"].values > np.datetime64(train_end))
            & (df["order_date"].values <= np.datetime64(test_end))
        )
        folds.append(
            Fold(
                name=f"fold{k + 1}",
                train_idx=train_idx,
                test_idx=test_idx,
                train_end=train_end,
                test_end=test_end,
            )
        )
    return folds


def random_kfold(df: pd.DataFrame, n_folds: int = 4, seed: int = 7) -> list[Fold]:
    """
    Deliberately wrong: ignores time entirely.

    Used once, to measure the size of the error it would have caused. See
    ml/run.py -> leakage_experiment.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    chunks = np.array_split(order, n_folds)
    folds = []
    for k, test_idx in enumerate(chunks):
        train_idx = np.setdiff1d(order, test_idx)
        folds.append(
            Fold(
                name=f"kfold{k + 1}",
                train_idx=np.sort(train_idx),
                test_idx=np.sort(test_idx),
                train_end=pd.NaT,
                test_end=pd.NaT,
            )
        )
    return folds
