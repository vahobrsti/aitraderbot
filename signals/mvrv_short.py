"""
MVRV Short Signal - Bear Market Tactical Short

Strategy: Short when mvrv_7d >= threshold in bear mode with mvrv_60d >= 1.0 overlay
Target: 4% price drop within 5 days
DCA: $30 initial (33%), $60 (67%) if price rises 4% against

Walk-forward validated across 2018, 2022, 2025-2026 bear markets.
Overall: 58% drop-first vs 31% rise-first (1.8:1 ratio)
"""
import pandas as pd
from dataclasses import dataclass
from typing import Optional


# Strategy constants
MVRV_7D_THRESHOLD = 1.02
MVRV_60D_THRESHOLD = 1.0  # Breakeven overlay
BEAR_MODE_START = 540
BEAR_MODE_END = 900
TARGET_PCT = 4.0
WINDOW_DAYS = 5
INITIAL_SIZE_PCT = 33  # 30/90 = 33%
DCA_SIZE_PCT = 67      # 60/90 = 67%


@dataclass
class MvrvShortSignal:
    """Result of MVRV short signal check."""
    active: bool
    is_bear_mode: bool
    cycle_day: Optional[int]
    mvrv_7d: Optional[float]
    mvrv_60d: Optional[float]
    btc_price: Optional[float]
    target_price: Optional[float]
    dca_trigger_price: Optional[float]
    reason: str
    
    # Execution parameters
    initial_size_pct: int = INITIAL_SIZE_PCT
    dca_size_pct: int = DCA_SIZE_PCT
    target_pct: float = TARGET_PCT
    window_days: int = WINDOW_DAYS


def is_bear_cycle_window(cycle_day: Optional[int]) -> bool:
    """Check if the current cycle day is in the bear window (540 - 900)."""
    if cycle_day is None or pd.isna(cycle_day):
        return False
    return BEAR_MODE_START <= cycle_day <= BEAR_MODE_END


def check_mvrv_short_signal(
    row: pd.Series,
    mvrv_7d_threshold: float = MVRV_7D_THRESHOLD,
    mvrv_60d_threshold: float = MVRV_60D_THRESHOLD,
) -> MvrvShortSignal:
    """
    Check if MVRV short signal conditions are met.
    
    Conditions:
    1. In bear mode (cycle days 540-900)
    2. mvrv_7d >= threshold (short-term holders profitable)
    3. mvrv_60d >= 1.0 (medium-term holders at breakeven or better)
    
    Args:
        row: Feature row with mvrv_usd_7d, mvrv_usd_60d, cycle_days_since_halving, btc_close
        mvrv_7d_threshold: MVRV 7d threshold (default 1.02)
        mvrv_60d_threshold: MVRV 60d overlay threshold (default 1.0)
    
    Returns:
        MvrvShortSignal with signal status and execution parameters
    """
    # Extract values
    cycle_day = row.get('cycle_days_since_halving', None)
    mvrv_7d = row.get('mvrv_7d', None)
    mvrv_60d = row.get('mvrv_60d', None)
    
    # Try alternate column names (raw data uses mvrv_usd_* prefix)
    if mvrv_7d is None:
        mvrv_7d = row.get('mvrv_usd_7d', None)
    if mvrv_60d is None:
        mvrv_60d = row.get('mvrv_usd_60d', None)
    
    btc_price = row.get('btc_close', None)
    
    # Handle NaN
    if cycle_day is not None and pd.isna(cycle_day):
        cycle_day = None
    if mvrv_7d is not None and pd.isna(mvrv_7d):
        mvrv_7d = None
    if mvrv_60d is not None and pd.isna(mvrv_60d):
        mvrv_60d = None
    if btc_price is not None and pd.isna(btc_price):
        btc_price = None
    
    # Check bear mode
    is_bear = is_bear_cycle_window(cycle_day)
    
    # Calculate targets if we have price
    target_price = btc_price * (1 - TARGET_PCT / 100) if btc_price else None
    dca_trigger = btc_price * (1 + TARGET_PCT / 100) if btc_price else None
    
    # Check conditions
    if not is_bear:
        return MvrvShortSignal(
            active=False,
            is_bear_mode=False,
            cycle_day=int(cycle_day) if cycle_day else None,
            mvrv_7d=mvrv_7d,
            mvrv_60d=mvrv_60d,
            btc_price=btc_price,
            target_price=target_price,
            dca_trigger_price=dca_trigger,
            reason=f"Not in bear mode (cycle day {cycle_day})"
        )
    
    if mvrv_7d is None:
        return MvrvShortSignal(
            active=False,
            is_bear_mode=True,
            cycle_day=int(cycle_day),
            mvrv_7d=None,
            mvrv_60d=mvrv_60d,
            btc_price=btc_price,
            target_price=target_price,
            dca_trigger_price=dca_trigger,
            reason="Missing MVRV 7d data"
        )
    
    if mvrv_7d < mvrv_7d_threshold:
        return MvrvShortSignal(
            active=False,
            is_bear_mode=True,
            cycle_day=int(cycle_day),
            mvrv_7d=mvrv_7d,
            mvrv_60d=mvrv_60d,
            btc_price=btc_price,
            target_price=target_price,
            dca_trigger_price=dca_trigger,
            reason=f"MVRV 7d below threshold ({mvrv_7d:.4f} < {mvrv_7d_threshold})"
        )
    
    if mvrv_60d is None:
        return MvrvShortSignal(
            active=False,
            is_bear_mode=True,
            cycle_day=int(cycle_day),
            mvrv_7d=mvrv_7d,
            mvrv_60d=None,
            btc_price=btc_price,
            target_price=target_price,
            dca_trigger_price=dca_trigger,
            reason="Missing MVRV 60d data"
        )
    
    if mvrv_60d < mvrv_60d_threshold:
        return MvrvShortSignal(
            active=False,
            is_bear_mode=True,
            cycle_day=int(cycle_day),
            mvrv_7d=mvrv_7d,
            mvrv_60d=mvrv_60d,
            btc_price=btc_price,
            target_price=target_price,
            dca_trigger_price=dca_trigger,
            reason=f"MVRV 60d below breakeven ({mvrv_60d:.4f} < {mvrv_60d_threshold})"
        )
    
    # All conditions met
    return MvrvShortSignal(
        active=True,
        is_bear_mode=True,
        cycle_day=int(cycle_day),
        mvrv_7d=mvrv_7d,
        mvrv_60d=mvrv_60d,
        btc_price=btc_price,
        target_price=target_price,
        dca_trigger_price=dca_trigger,
        reason=f"MVRV 7d={mvrv_7d:.4f} >= {mvrv_7d_threshold}, MVRV 60d={mvrv_60d:.4f} >= {mvrv_60d_threshold}"
    )


def get_signal_summary(signal: MvrvShortSignal) -> dict:
    """Convert MvrvShortSignal to a dictionary for API response."""
    return {
        "active": signal.active,
        "is_bear_mode": signal.is_bear_mode,
        "cycle_day": signal.cycle_day,
        "mvrv_7d": signal.mvrv_7d,
        "mvrv_60d": signal.mvrv_60d,
        "btc_price": signal.btc_price,
        "target_price": signal.target_price,
        "dca_trigger_price": signal.dca_trigger_price,
        "reason": signal.reason,
        "execution": {
            "initial_size_pct": signal.initial_size_pct,
            "dca_size_pct": signal.dca_size_pct,
            "target_pct": signal.target_pct,
            "window_days": signal.window_days,
        } if signal.active else None,
        "backtest_stats": {
            "win_rate": 0.58,
            "drop_rise_ratio": 1.8,
            "ev_per_trade_pct": 1.99,
        } if signal.active else None,
    }
