"""
Credit Spread Labels: Forward outcome classification for income spreads.

For each date, determines whether a credit spread would have expired safely
(short strike never breached) over the next N days.

Uses MVRV-derived strike boundaries (same logic as income_gate.py):
- Bull put floor: spot / mvrv_60d (cost basis), with P5 fallback when underwater
- Bear call ceiling: (spot / mvrv_60d) * mvrv_60d_p95_180d

Falls back to a fixed minimum OTM distance (4%) when MVRV data is unavailable.

Columns produced:
- label_bull_put_safe_14d: 1 if price never dropped below the MVRV floor (or 4% OTM)
- label_bear_call_safe_14d: 1 if price never rallied above the MVRV ceiling (or 4% OTM)
- label_bull_put_floor_pct: the floor distance as % below entry (for diagnostics)
- label_bear_call_ceiling_pct: the ceiling distance as % above entry (for diagnostics)
"""

import pandas as pd
import numpy as np


# Defaults aligned with IncomeGateConfig
_HORIZON_DAYS = 14
_MIN_OTM_PCT = 0.04  # Fallback: short strike is at least 4% OTM


def calculate(
    df: pd.DataFrame,
    horizon_days: int = _HORIZON_DAYS,
    min_otm_pct: float = _MIN_OTM_PCT,
) -> pd.DataFrame:
    """
    Compute credit spread hit labels using MVRV-derived boundaries.

    Args:
        df: DataFrame with DatetimeIndex. Must contain:
            - btc_close, btc_high, btc_low, btc_open
            - mvrv_60d (or mvrv_usd_60d)
            - mvrv_60d_p5_180d (optional, for underwater floor)
            - mvrv_60d_p95_180d (optional, for ceiling)
        horizon_days: Forward window in rows (default 14).
        min_otm_pct: Minimum OTM distance fallback (default 0.04 = 4%).

    Returns:
        DataFrame with label columns.
    """
    feats = {}

    # Entry price: next day's open (signal fires EOD, enter next open)
    entry_price = df["btc_open"].shift(-1)

    # Forward max high and min low over the next `horizon_days` observations
    # rolling(window).max/min then shift(-(window+1)) aligns [T+1 .. T+window]
    window = horizon_days
    fwd_max_high = df["btc_high"].rolling(window).max().shift(-(window + 1))
    fwd_min_low = df["btc_low"].rolling(window).min().shift(-(window + 1))

    # --- MVRV-60d (prefer computed feature column, fall back to raw) ---
    if "mvrv_60d" in df.columns:
        mvrv_60d = df["mvrv_60d"].astype(float)
    elif "mvrv_usd_60d" in df.columns:
        mvrv_60d = df["mvrv_usd_60d"].astype(float)
    else:
        mvrv_60d = pd.Series(np.nan, index=df.index)

    mvrv_60d_p5 = (
        df["mvrv_60d_p5_180d"].astype(float)
        if "mvrv_60d_p5_180d" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    mvrv_60d_p95 = (
        df["mvrv_60d_p95_180d"].astype(float)
        if "mvrv_60d_p95_180d" in df.columns
        else pd.Series(np.nan, index=df.index)
    )

    # --- Bull Put Floor ---
    # cost_basis = spot / mvrv_60d
    # If buyers profitable (cost_basis <= spot): floor = cost_basis
    # If buyers underwater (cost_basis > spot): floor = cost_basis * P5
    # Fallback: entry * (1 - min_otm_pct)
    cost_basis = entry_price / mvrv_60d

    # Determine which path: profitable vs underwater
    buyers_profitable = cost_basis <= entry_price

    # Floor when profitable: cost_basis itself
    # Floor when underwater: cost_basis * P5 (tighter bound)
    floor_profitable = cost_basis
    floor_underwater = cost_basis * mvrv_60d_p5
    # If P5 is NaN when underwater, fall back to cost_basis anyway
    floor_underwater = floor_underwater.where(mvrv_60d_p5.notna(), cost_basis)

    bull_put_floor = pd.Series(np.nan, index=df.index)
    bull_put_floor = bull_put_floor.where(
        ~buyers_profitable, floor_profitable
    )
    bull_put_floor = bull_put_floor.where(
        buyers_profitable, floor_underwater
    )

    # Fallback: when MVRV data is missing, use fixed 4% OTM
    fallback_floor = entry_price * (1 - min_otm_pct)
    mvrv_valid = mvrv_60d.notna() & (mvrv_60d > 0)
    bull_put_floor = bull_put_floor.where(mvrv_valid, fallback_floor)

    # --- Bear Call Ceiling ---
    # ceiling = cost_basis * P95
    # Fallback: entry * (1 + min_otm_pct)
    bear_call_ceiling = cost_basis * mvrv_60d_p95

    # Fallback when P95 is missing but MVRV-60d is valid: no ceiling computable
    # Use fixed OTM fallback
    fallback_ceiling = entry_price * (1 + min_otm_pct)
    ceiling_valid = mvrv_valid & mvrv_60d_p95.notna() & (mvrv_60d_p95 > 0)
    bear_call_ceiling = bear_call_ceiling.where(ceiling_valid, fallback_ceiling)

    # --- Labels ---
    # Bull put safe: price never dropped below the floor
    bull_put_safe = (fwd_min_low >= bull_put_floor).astype("Int64")
    # NaN where forward data is missing
    bull_put_safe = bull_put_safe.where(fwd_min_low.notna(), other=pd.NA)

    # Bear call safe: price never rallied above the ceiling
    bear_call_safe = (fwd_max_high <= bear_call_ceiling).astype("Int64")
    bear_call_safe = bear_call_safe.where(fwd_max_high.notna(), other=pd.NA)

    # --- Diagnostic: floor/ceiling as % distance from entry ---
    floor_pct = (entry_price - bull_put_floor) / entry_price
    ceiling_pct = (bear_call_ceiling - entry_price) / entry_price

    feats["label_bull_put_safe_14d"] = bull_put_safe
    feats["label_bear_call_safe_14d"] = bear_call_safe
    feats["label_bull_put_floor_pct"] = floor_pct
    feats["label_bear_call_ceiling_pct"] = ceiling_pct

    return pd.DataFrame(feats, index=df.index)
