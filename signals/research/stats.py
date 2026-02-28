"""
Descriptive statistics layer for the research pipeline.

All functions accept a research-table DataFrame (from fusion_table.py)
and return summary DataFrames with one row per group.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from signals.research.constants import MIN_SAMPLE_THRESHOLD


# ────────────────────────────────────────────────────────────────────
# Internal helpers
# ────────────────────────────────────────────────────────────────────
def _agg_group(group: pd.DataFrame) -> dict:
    """Aggregate a single group into a stats dict."""
    n = len(group)
    stats: dict = {"n": n}

    # Hit rates
    if "label_good_move_long" in group.columns:
        long_vals = group["label_good_move_long"].dropna()
        stats["long_hit_rate"] = long_vals.mean() if len(long_vals) else np.nan
    if "label_good_move_short" in group.columns:
        short_vals = group["label_good_move_short"].dropna()
        stats["short_hit_rate"] = short_vals.mean() if len(short_vals) else np.nan

    # Forward returns
    for col in ("ret_14d", "ret_7d", "ret_21d"):
        if col in group.columns:
            vals = group[col].dropna()
            stats[f"{col}_mean"] = vals.mean() if len(vals) else np.nan
            stats[f"{col}_median"] = vals.median() if len(vals) else np.nan
            stats[f"{col}_std"] = vals.std() if len(vals) else np.nan

    # MFE / MAE
    for col in ("mfe_14d", "mae_14d"):
        if col in group.columns:
            vals = group[col].dropna()
            stats[f"{col}_mean"] = vals.mean() if len(vals) else np.nan
            stats[f"{col}_median"] = vals.median() if len(vals) else np.nan

    return stats


def _build_stats_df(
    research_df: pd.DataFrame,
    group_cols: list[str],
    min_count: int = MIN_SAMPLE_THRESHOLD,
) -> pd.DataFrame:
    """Group by *group_cols*, compute stats, flag low-sample groups."""
    rows = []
    for name, grp in research_df.groupby(group_cols, observed=True):
        stats = _agg_group(grp)
        if isinstance(name, tuple):
            for i, col in enumerate(group_cols):
                stats[col] = name[i]
        else:
            stats[group_cols[0]] = name
        rows.append(stats)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["_flagged"] = out["n"] < min_count
    # Sort by sample size descending
    out = out.sort_values("n", ascending=False).reset_index(drop=True)
    return out


# ────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────
def compute_bucket_stats(
    research_df: pd.DataFrame,
    group_col: str,
    min_count: int = MIN_SAMPLE_THRESHOLD,
) -> pd.DataFrame:
    """Per-bucket stats for a single metric (e.g. ``mdia_bucket``).

    Returns one row per bucket with columns:
        n, long_hit_rate, short_hit_rate, ret_14d_mean, ret_14d_median,
        ret_14d_std, mfe_14d_mean, mfe_14d_median, mae_14d_mean,
        mae_14d_median, _flagged.
    """
    return _build_stats_df(research_df, [group_col], min_count)


def compute_combo_stats(
    research_df: pd.DataFrame,
    group_cols: list[str],
    min_count: int = MIN_SAMPLE_THRESHOLD,
) -> pd.DataFrame:
    """Stats grouped by a combination of bucket columns.

    Example: ``compute_combo_stats(rt, ['mdia_bucket', 'whale_bucket'])``
    """
    return _build_stats_df(research_df, group_cols, min_count)


def compute_state_stats(
    research_df: pd.DataFrame,
    min_count: int = MIN_SAMPLE_THRESHOLD,
) -> pd.DataFrame:
    """Stats grouped by ``fusion_state``."""
    return _build_stats_df(research_df, ["fusion_state"], min_count)


def compute_score_stats(
    research_df: pd.DataFrame,
    min_count: int = MIN_SAMPLE_THRESHOLD,
) -> pd.DataFrame:
    """Stats grouped by ``fusion_score``."""
    return _build_stats_df(research_df, ["fusion_score"], min_count)
