"""
IBIT wheel short-leg selection.

The wheel runs the income-gate signals as single-leg short positions on IBIT
instead of defined-risk BTC spreads:

- BULL_PUT_SPREAD  -> short put  (cash-secured put)
- BEAR_CALL_SPREAD -> short call (covered call)

Selection reuses the income-gate chain layer verbatim — the same delta band,
minimum-OTM distance, DTE window, and liquidity filters (``filter_option_chain``)
and the same low/medium/high delta tiers (``_TIER_DELTA_RANGES``). The only
difference from the spread path is that there is no long protective leg, so the
credit is the short leg's bid and there is no capped max-loss.

This module selects and describes; it does not place orders.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd

from signals.income_gate import (
    IncomeGateConfig,
    _TIER_DELTA_RANGES,
    filter_option_chain,
    normalize_chain_columns,
)

# Contract multiplier for IBIT options (confirmed via IBKR).
IBIT_MULTIPLIER = 100


@dataclass
class WheelLeg:
    """A single short-leg wheel position for one risk tier."""

    risk_tier: str            # "low" | "medium" | "high"
    side: str                 # "put" | "call"
    position: str             # "cash_secured_put" | "covered_call"
    strike: float
    delta: float
    credit: float             # premium collected (short leg bid), per share
    premium_usd: float        # credit x multiplier, per contract
    otm_pct: float            # distance from spot, as a fraction
    dte: int
    bid: float
    ask: float
    spread_pct: Optional[float] = None
    expiry: Optional[str] = None      # YYYY-MM-DD if available
    symbol: Optional[str] = None
    # Cash-secured puts only: cash to reserve per contract (strike x mult - premium)
    cash_reserve_usd: Optional[float] = None


def select_wheel_legs(
    chain_df: pd.DataFrame,
    side: str,
    spot_price: float,
    config: Optional[IncomeGateConfig] = None,
    strike_boundary: Optional[float] = None,
    dte_mode: str = "income",
    multiplier: int = IBIT_MULTIPLIER,
) -> list[WheelLeg]:
    """
    Select up to three short-leg wheel positions (one per risk tier) from an
    IBIT option chain, reusing the income-gate chain-layer filters.

    Args:
        chain_df: IBIT option chain (OptionSnapshot rows as a DataFrame).
        side: "put" (cash-secured put) or "call" (covered call).
        spot_price: Current IBIT spot price.
        config: Income gate configuration (defaults to IncomeGateConfig()).
        strike_boundary: Optional MVRV-derived IBIT proxy boundary. Puts keep
            strikes <= boundary; calls keep strikes >= boundary. None disables it
            (delta band and min-OTM still apply).
        dte_mode: "tactical" (9-21d) or "income" (21-45d).
        multiplier: Option contract multiplier (100 for IBIT).

    Returns:
        List[WheelLeg], 0 to 3 entries (one per tier that has a qualifying
        contract), ordered low -> medium -> high risk.
    """
    if side not in ("put", "call"):
        raise ValueError(f"side must be 'put' or 'call', got {side!r}")

    if config is None:
        config = IncomeGateConfig()

    normalized = normalize_chain_columns(chain_df)
    if normalized is None or normalized.empty:
        return []

    filtered = filter_option_chain(
        normalized, side, spot_price, config, dte_mode, strike_boundary=strike_boundary
    )
    if filtered.empty:
        return []

    position = "cash_secured_put" if side == "put" else "covered_call"
    legs: list[WheelLeg] = []

    for tier_name, (d_min, d_max) in _TIER_DELTA_RANGES.items():
        tier = filtered[(filtered["delta"] >= d_min) & (filtered["delta"] < d_max)]
        if tier.empty:
            continue

        # Most premium within the tier = highest short-leg bid (credit collected).
        best = tier.loc[tier["bid"].idxmax()]

        strike = float(best["strike"])
        credit = float(best["bid"])
        if side == "put":
            otm_pct = (spot_price - strike) / spot_price
            cash_reserve = round(strike * multiplier - credit * multiplier, 2)
        else:
            otm_pct = (strike - spot_price) / spot_price
            cash_reserve = None

        legs.append(WheelLeg(
            risk_tier=tier_name,
            side=side,
            position=position,
            strike=strike,
            delta=float(best["delta"]),
            credit=credit,
            premium_usd=round(credit * multiplier, 2),
            otm_pct=otm_pct,
            dte=int(best["dte"]),
            bid=credit,
            ask=float(best["ask"]),
            spread_pct=float(best["spread_pct"]) if "spread_pct" in best and pd.notna(best["spread_pct"]) else None,
            expiry=_format_expiry(best.get("expiry") if hasattr(best, "get") else best["expiry"] if "expiry" in best else None),
            symbol=str(best["symbol"]) if "symbol" in best and pd.notna(best["symbol"]) else None,
            cash_reserve_usd=cash_reserve,
        ))

    return legs


def wheel_side_for_decision(trade_decision: str) -> Optional[str]:
    """Map an income-gate decision to the wheel short-leg side."""
    d = (trade_decision or "").upper()
    if d == "BULL_PUT_SPREAD":
        return "put"
    if d == "BEAR_CALL_SPREAD":
        return "call"
    return None


def _format_expiry(raw) -> Optional[str]:
    if raw is None:
        return None
    try:
        if pd.isna(raw):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return pd.Timestamp(raw).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        if isinstance(raw, datetime):
            return raw.strftime("%Y-%m-%d")
        return str(raw)
