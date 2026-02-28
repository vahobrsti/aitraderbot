"""
Canonical bucket mapping.

Each function maps a feature row to exactly one mutually-exclusive bucket
string.  Priority order mirrors fusion.py — most specific condition first.
"""

from __future__ import annotations

import pandas as pd


# ────────────────────────────────────────────────────────────────────
# MDIA buckets
# ────────────────────────────────────────────────────────────────────
def map_mdia_bucket(row: pd.Series) -> str:
    """
    Maps to exactly one of:
      'strong_inflow', 'inflow', 'aging', 'neutral'

    Priority: strong_inflow > inflow > aging > neutral
    """
    if row.get("mdia_regime_strong_inflow", 0) == 1:
        return "strong_inflow"
    if row.get("mdia_regime_inflow", 0) == 1:
        return "inflow"
    if row.get("mdia_regime_aging", 0) == 1:
        return "aging"
    return "neutral"


# ────────────────────────────────────────────────────────────────────
# Whale buckets
# ────────────────────────────────────────────────────────────────────
def map_whale_bucket(row: pd.Series) -> str:
    """
    Maps to exactly one of:
      'broad_accum', 'strategic_accum', 'distribution_strong',
      'distribution', 'neutral'

    Priority: broad_accum > strategic_accum > distribution_strong
              > distribution > neutral
    """
    if row.get("whale_regime_broad_accum", 0) == 1:
        return "broad_accum"
    if row.get("whale_regime_strategic_accum", 0) == 1:
        return "strategic_accum"
    if row.get("whale_regime_distribution_strong", 0) == 1:
        return "distribution_strong"
    if row.get("whale_regime_distribution", 0) == 1:
        return "distribution"
    if row.get("whale_regime_mixed", 0) == 1:
        return "mixed"
    return "neutral"


# ────────────────────────────────────────────────────────────────────
# MVRV-LS buckets
# ────────────────────────────────────────────────────────────────────
def map_mvrv_ls_bucket(row: pd.Series) -> str:
    """
    Maps to exactly one of:
      'early_recovery', 'trend_confirm', 'call_confirm',
      'distribution_warning', 'put_confirm', 'neutral'

    Priority: early_recovery > trend_confirm > call_confirm
              > distribution_warning > put_confirm > neutral

    NOTE: early_recovery is checked before call_confirm because
    recovery is a *subset* of call_confirm (both require strong_uptrend,
    but recovery also requires level < 0).  Same order as
    compute_confidence_score() in fusion.py.
    """
    if row.get("mvrv_ls_regime_call_confirm_recovery", 0) == 1:
        return "early_recovery"
    if row.get("mvrv_ls_regime_call_confirm_trend", 0) == 1:
        return "trend_confirm"
    if row.get("mvrv_ls_regime_call_confirm", 0) == 1:
        return "call_confirm"
    if row.get("mvrv_ls_regime_distribution_warning", 0) == 1:
        return "distribution_warning"
    if row.get("mvrv_ls_regime_put_confirm", 0) == 1:
        return "put_confirm"
    return "neutral"


# ────────────────────────────────────────────────────────────────────
# Bulk helpers
# ────────────────────────────────────────────────────────────────────
def add_bucket_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``mdia_bucket``, ``whale_bucket``, ``mvrv_ls_bucket`` columns.

    Returns a **copy** — does not mutate the input.
    """
    out = df.copy()
    out["mdia_bucket"] = df.apply(map_mdia_bucket, axis=1)
    out["whale_bucket"] = df.apply(map_whale_bucket, axis=1)
    out["mvrv_ls_bucket"] = df.apply(map_mvrv_ls_bucket, axis=1)
    return out
