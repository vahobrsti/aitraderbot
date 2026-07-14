"""
Rolling-window sweep for metric normalisation.

Goal: decide, data-driven, which rolling window (e.g. 60d vs 90d) to use when
expressing a metric as a z-score relative to its own recent history.

"Better" is measured against the binary good-move labels already in the feature
CSV (there is no continuous forward return column). For each metric and window
we compute a z-score and score it on:

  - AUC          : separation of good-move days (0.5 = no signal; distance from
                   0.5 = strength; <0.5 = inversely predictive)
  - spread       : top-quintile minus bottom-quintile hit rate (signed)
  - monotonic    : whether hit rate is monotonic across z quintiles
  - autocorr1    : lag-1 autocorrelation of the z (higher = more stable)
  - flips_per_yr : sign changes of the z per year (lower = less whipsaw)

Everything is also computed per calendar year so a window can be checked for
consistency rather than one-shot in-sample fit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Metric name → base column the z-score is computed from.
# Bases mirror the columns the existing *_z_90 features are built on.
METRIC_BASES = {
    "mvrv_60d": "mvrv_60d",
    "sentiment": "sentiment_norm",
    "exchange_flow": "flow_sum_7",
    "mvrv_composite": "mvrv_composite_pct",
}

DEFAULT_WINDOWS = [30, 60, 90, 120, 180]
DEFAULT_LABELS = ["label_good_move_long", "label_good_move_short"]


def zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score of *series* against its own trailing *window*."""
    roll_mean = series.rolling(window, min_periods=max(10, window // 3)).mean()
    roll_std = series.rolling(window, min_periods=max(10, window // 3)).std()
    return (series - roll_mean) / (roll_std + 1e-9)


def rank_auc(scores: pd.Series, labels: pd.Series) -> float:
    """AUC via the Mann-Whitney rank formula (no sklearn dependency).

    Returns NaN if either class is empty.
    """
    mask = scores.notna() & labels.notna()
    s = scores[mask]
    y = labels[mask].astype(int)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = s.rank(method="average")
    sum_pos = ranks[y == 1].sum()
    return (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def quintile_spread(z: pd.Series, labels: pd.Series, n_buckets: int = 5):
    """Return (spread, monotonic) of hit rate across z quantile buckets.

    spread = top-bucket hit rate minus bottom-bucket hit rate (signed).
    monotonic = True if per-bucket hit rate is monotone (either direction).
    """
    mask = z.notna() & labels.notna()
    zz = z[mask]
    yy = labels[mask].astype(int)
    if len(zz) < n_buckets * 5:
        return np.nan, False
    try:
        buckets = pd.qcut(zz, n_buckets, labels=False, duplicates="drop")
    except ValueError:
        return np.nan, False
    means = yy.groupby(buckets).mean()
    if len(means) < 2:
        return np.nan, False
    ordered = means.sort_index()
    diffs = np.diff(ordered.values)
    monotonic = bool(np.all(diffs >= 0) or np.all(diffs <= 0))
    spread = float(ordered.iloc[-1] - ordered.iloc[0])
    return spread, monotonic


def stability(z: pd.Series) -> dict:
    """Persistence / whipsaw diagnostics for a z series."""
    zz = z.dropna()
    if len(zz) < 30:
        return {"autocorr1": np.nan, "flips_per_yr": np.nan}
    autocorr1 = float(zz.autocorr(lag=1))
    sign = np.sign(zz.values)
    flips = int((sign[1:] != sign[:-1]).sum())
    years = len(zz) / 365.0
    flips_per_yr = flips / years if years > 0 else np.nan
    return {"autocorr1": autocorr1, "flips_per_yr": flips_per_yr}


def evaluate(
    df: pd.DataFrame,
    metric: str,
    window: int,
    labels: list[str],
) -> dict:
    """Evaluate one (metric, window) pair across all requested labels."""
    base_col = METRIC_BASES[metric]
    z = zscore(df[base_col], window)
    row = {"metric": metric, "base_col": base_col, "window": window}
    row.update(stability(z))
    for label in labels:
        if label not in df.columns:
            continue
        short = label.replace("label_good_move_", "")
        row[f"auc_{short}"] = rank_auc(z, df[label])
        spread, mono = quintile_spread(z, df[label])
        row[f"spread_{short}"] = spread
        row[f"mono_{short}"] = mono
    return row


def sweep(
    df: pd.DataFrame,
    metrics: list[str] | None = None,
    windows: list[int] | None = None,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Full sweep over metrics × windows. Returns a tidy results DataFrame."""
    metrics = metrics or list(METRIC_BASES.keys())
    windows = windows or DEFAULT_WINDOWS
    labels = labels or DEFAULT_LABELS

    rows = []
    for metric in metrics:
        if METRIC_BASES[metric] not in df.columns:
            continue
        for window in windows:
            rows.append(evaluate(df, metric, window, labels))
    return pd.DataFrame(rows)


def sweep_by_year(
    df: pd.DataFrame,
    metric: str,
    window: int,
    label: str,
) -> pd.DataFrame:
    """Per-year AUC for one (metric, window, label) — robustness check."""
    base_col = METRIC_BASES[metric]
    # Compute z on the full series first so the rolling window has history,
    # then slice by year for scoring.
    z = zscore(df[base_col], window)
    rows = []
    for year, idx in df.groupby(df.index.year).groups.items():
        sub_z = z.loc[idx]
        sub_y = df.loc[idx, label]
        rows.append({"year": int(year), "n": int(sub_y.notna().sum()),
                     "auc": rank_auc(sub_z, sub_y)})
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
