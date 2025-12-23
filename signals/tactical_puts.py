# features/signals/tactical_puts.py
"""
Tactical Puts Inside Bull Regimes (MVRV-60d Driven)

This module creates tactical PUT permissions during bull regimes when:
1. MVRV-60d is near its 60-day peak (profit saturation)
2. MVRV-60d is rolling over (falling/flattening)
3. Price is extended (optional guardrail)

Key principles:
- Conservative: won't spam puts
- Defined-risk: defaults to put spreads
- Non-destructive: doesn't change MarketState
- Compatible with overlay dominance rules

This does NOT flip calls to puts. It adds a small hedge/alpha permission
on top of the existing bull trade.
"""

import math
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from datetime import date, timedelta

from .fusion import MarketState, FusionResult
from .overlays import compute_near_peak_score


class TacticalPutStrategy(str, Enum):
    """Strategy type for tactical puts"""
    NONE = "NONE"
    PUT_SPREAD = "PUT_SPREAD"      # Default: defined risk
    PUT = "PUT"                    # Only for highest conviction (rare)
    PUT_CALENDAR = "PUT_CALENDAR"  # Optional, if supported


@dataclass
class TacticalPutResult:
    """Result of tactical put evaluation"""
    active: bool
    strength: int                 # 0=none, 1=partial, 2=full
    strategy: TacticalPutStrategy
    size_mult: float              # Applied on top of base size
    dte_mult: float               # Applied on top of base DTE
    reason: str
    
    @property
    def should_trade(self) -> bool:
        return self.active and self.strength > 0


# Bull states where tactical puts are allowed
BULL_STATES = {
    MarketState.STRONG_BULLISH,
    MarketState.MOMENTUM_CONTINUATION,
    MarketState.EARLY_RECOVERY
}


def in_bull_state(state: MarketState) -> bool:
    """Check if state is bullish"""
    return state in BULL_STATES


def compute_mvrv_60d_trend(row: pd.Series) -> tuple[bool, bool]:
    """
    Determine if MVRV-60d is rolling over (falling or flattening).
    
    Uses the mvrv_60d_pct_rank changes to detect trend.
    Falls back to simple comparison if specific trend features don't exist.
    
    Returns (is_falling, is_flattening)
    """
    # Check if explicit trend features exist
    if 'mvrv_60d_is_falling' in row.index:
        falling = row.get('mvrv_60d_is_falling', 0) == 1
        flattening = row.get('mvrv_60d_is_flattening', 0) == 1
        return falling, flattening
    
    # Fallback: use dist_from_max as proxy
    # If dist_from_max > 0.1 but score still high, it's rolling over
    dist = row.get('mvrv_60d_dist_from_max', None)
    pct = row.get('mvrv_60d_pct_rank', None)
    
    if dist is None or pct is None:
        return False, False
    
    if isinstance(dist, float) and math.isnan(dist):
        return False, False
    
    # Rolling over: score still elevated but starting to fall from peak
    # dist > 0.05 means we're at least 5% away from the peak
    # dist > 0.15 means we're clearly falling
    falling = dist > 0.15 and pct >= 0.60
    flattening = 0.05 < dist <= 0.15 and pct >= 0.70
    
    return falling, flattening


def check_price_extended(row: pd.Series) -> bool:
    """
    Check if price is extended (helps avoid early hedges).
    
    Uses available price extension features or defaults to True
    if no features available.
    """
    # Try various extension indicators
    price_z_60d = row.get('price_z_60d', None)
    if price_z_60d is not None and not (isinstance(price_z_60d, float) and math.isnan(price_z_60d)):
        if price_z_60d >= 1.0:
            return True
    
    # Try MVRV composite overheated as proxy for extended price
    mvrv_overheated = (
        row.get('mvrv_bucket_overvalued', 0) == 1 or
        row.get('mvrv_bucket_extreme_overvalued', 0) == 1
    )
    if mvrv_overheated:
        return True
    
    # Try near-top-any flag
    if row.get('mvrv_comp_near_top_any', 0) == 1:
        return True
    
    # Default: assume extended if MVRV-60d score is high
    score = compute_near_peak_score(row)
    if score is not None and score >= 0.80:
        return True
    
    return False


def check_melt_up_veto(row: pd.Series) -> bool:
    """
    Check for strong upward momentum that should veto tactical puts.
    
    Don't short into a strong squeeze.
    """
    # Check for strong recent returns (if available)
    ret_7d = row.get('ret_7d', None)
    if ret_7d is not None and not (isinstance(ret_7d, float) and math.isnan(ret_7d)):
        if ret_7d > 0.08:  # 8%+ in 7 days = melt-up
            return True
    
    # Check for strong uptrend in MVRV (if available)
    mvrv_rising = row.get('mvrv_is_rising', 0) == 1
    mvrv_60d_pct = row.get('mvrv_60d_pct_rank', 0.5)
    
    # If MVRV is still rising strongly, don't fight it
    if mvrv_rising and mvrv_60d_pct >= 0.95:
        return True
    
    return False


def cooldown_ok(last_put_trade_date: Optional[date], today: date, cooldown_days: int = 5) -> bool:
    """Check if cooldown period has passed since last put trade"""
    if last_put_trade_date is None:
        return True
    return (today - last_put_trade_date).days >= cooldown_days


def max_put_positions_reached(open_put_positions: int, max_puts: int = 1) -> bool:
    """Check if max put positions have been reached"""
    return open_put_positions >= max_puts


def tactical_put_inside_bull(
    fusion_result: FusionResult,
    row: pd.Series,
    *,
    cooldown_active: bool = False,
    already_has_put: bool = False,
    max_positions_reached: bool = False,
    full_threshold: float = 0.85,
    partial_threshold: float = 0.75,
) -> TacticalPutResult:
    """
    Tactical put permission inside bull regimes using MVRV-60d.
    
    This does NOT change MarketState. It adds a "hedge/alpha put" option
    that can be executed alongside the primary bull trade.
    
    Args:
        fusion_result: Result from fusion classification
        row: Feature row with MVRV-60d and other indicators
        cooldown_active: Whether cooldown period is active
        already_has_put: Whether there's already an open put position
        max_positions_reached: Whether max put positions reached
        full_threshold: Score threshold for full strength (default 0.85)
        partial_threshold: Score threshold for partial strength (default 0.75)
    
    Returns:
        TacticalPutResult with trade permission and parameters
    """
    
    # 0) Only consider in bull regimes
    if not in_bull_state(fusion_result.state):
        return TacticalPutResult(
            active=False, strength=0,
            strategy=TacticalPutStrategy.NONE,
            size_mult=1.0, dte_mult=1.0,
            reason="Not a bull state"
        )
    
    # 1) Portfolio / execution guardrails
    if cooldown_active:
        return TacticalPutResult(
            active=False, strength=0,
            strategy=TacticalPutStrategy.NONE,
            size_mult=1.0, dte_mult=1.0,
            reason="Cooldown active"
        )
    
    if max_positions_reached:
        return TacticalPutResult(
            active=False, strength=0,
            strategy=TacticalPutStrategy.NONE,
            size_mult=1.0, dte_mult=1.0,
            reason="Max put positions reached"
        )
    
    if already_has_put:
        return TacticalPutResult(
            active=False, strength=0,
            strategy=TacticalPutStrategy.NONE,
            size_mult=1.0, dte_mult=1.0,
            reason="Already have put exposure"
        )
    
    # 2) Compute MVRV-60d context
    score = compute_near_peak_score(row)
    if score is None:
        # Fail-closed in bull regimes (we don't need to short if data missing)
        return TacticalPutResult(
            active=False, strength=0,
            strategy=TacticalPutStrategy.NONE,
            size_mult=1.0, dte_mult=1.0,
            reason="Missing MVRV-60d score"
        )
    
    # 3) Check melt-up veto first
    if check_melt_up_veto(row):
        return TacticalPutResult(
            active=False, strength=0,
            strategy=TacticalPutStrategy.NONE,
            size_mult=1.0, dte_mult=1.0,
            reason=f"Melt-up veto active (score={score:.2f})"
        )
    
    # 4) Rolling-over condition (critical)
    is_falling, is_flattening = compute_mvrv_60d_trend(row)
    rolling_over = is_falling or is_flattening
    
    if not rolling_over:
        return TacticalPutResult(
            active=False, strength=0,
            strategy=TacticalPutStrategy.NONE,
            size_mult=1.0, dte_mult=1.0,
            reason=f"MVRV near-peak candidate but not rolling over (score={score:.2f})"
        )
    
    # 5) Price extension check
    price_extended = check_price_extended(row)
    
    # 6) Decision thresholds
    if score >= full_threshold and price_extended:
        # FULL tactical put
        return TacticalPutResult(
            active=True,
            strength=2,
            strategy=TacticalPutStrategy.PUT_SPREAD,
            size_mult=0.60,   # Small hedge/alpha, not a full reversal bet
            dte_mult=0.85,    # Shorter DTE to match pullback window
            reason=f"FULL: Bull regime + MVRV-60d near-peak & rolling over (score={score:.2f})"
        )
    
    if score >= partial_threshold and price_extended:
        # PARTIAL tactical put
        return TacticalPutResult(
            active=True,
            strength=1,
            strategy=TacticalPutStrategy.PUT_SPREAD,
            size_mult=0.40,
            dte_mult=0.80,
            reason=f"PARTIAL: Bull regime + MVRV-60d elevated & rolling over (score={score:.2f})"
        )
    
    # High score but price not extended, or score not high enough
    return TacticalPutResult(
        active=False, strength=0,
        strategy=TacticalPutStrategy.NONE,
        size_mult=1.0, dte_mult=1.0,
        reason=f"No tactical put: score={score:.2f}, price_extended={price_extended}, rolling_over={rolling_over}"
    )
