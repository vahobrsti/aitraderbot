# features/signals/options.py
"""
Signal Fusion Engine - Layer 2
Maps market states to optimal option structures.

Each market state has:
- Recommended structures (calls, puts, spreads, etc.)
- Strike selection guidance (ATM, OTM, ITM)
- DTE recommendations
- Position sizing guidance based on confidence
"""

from dataclasses import dataclass, replace
from enum import Enum
from typing import List, Optional
from .fusion import MarketState, Confidence


class OptionStructure(Enum):
    """Option structure types"""
    # Directional
    LONG_CALL = "long_call"
    LONG_PUT = "long_put"
    
    # Spreads
    CALL_SPREAD = "call_spread"          # Bull call spread
    PUT_SPREAD = "put_spread"            # Bear put spread
    
    # Calendars/Diagonals
    CALL_CALENDAR = "call_calendar"
    PUT_CALENDAR = "put_calendar"
    CALL_DIAGONAL = "call_diagonal"
    PUT_DIAGONAL = "put_diagonal"
    
    # Credit spreads
    SHORT_CALL_SPREAD = "short_call_spread"  # Bear call spread
    SHORT_PUT_SPREAD = "short_put_spread"    # Bull put spread
    
    # Advanced
    CALL_BACKSPREAD = "call_backspread"
    PUT_BACKSPREAD = "put_backspread"
    RATIO_CALL_SPREAD = "ratio_call_spread"
    BROKEN_WING_BUTTERFLY = "broken_wing_butterfly"
    
    # No action
    NO_TRADE = "no_trade"


class StrikeSelection(Enum):
    """Strike selection guidance"""
    ATM = "atm"              # At-the-money
    SLIGHT_ITM = "slight_itm"  # ~1-2% ITM / ~0.55-0.65 delta
    SLIGHT_OTM = "slight_otm"  # 5-10% OTM
    OTM = "otm"              # 10-20% OTM
    DEEP_OTM = "deep_otm"    # >20% OTM
    ITM = "itm"              # In-the-money


@dataclass
class DTEGuidance:
    """Days to expiration guidance"""
    min_dte: int
    max_dte: int
    optimal_dte: int
    
    def __str__(self):
        return f"{self.min_dte}-{self.max_dte}d (optimal: {self.optimal_dte}d)"


@dataclass
class PositionSizing:
    """Position sizing as % of portfolio"""
    high_confidence: float   # % when confidence is HIGH
    medium_confidence: float # % when confidence is MEDIUM
    low_confidence: float    # % when confidence is LOW (usually 0)
    
    def get_size(self, confidence: Confidence) -> float:
        if confidence == Confidence.HIGH:
            return self.high_confidence
        elif confidence == Confidence.MEDIUM:
            return self.medium_confidence
        else:
            return self.low_confidence


@dataclass
class SpreadGuidance:
    """Spread width and exit guidance"""
    width_pct: float
    take_profit_pct: float
    max_hold_days: int           # Hard time stop: close all remaining
    stop_loss_pct: float = 0.0   # Underlying move to exit (~40-50% of width)
    scale_down_day: int = 0      # Day to reduce position to 25%


def format_stop_loss_string(spread: 'SpreadGuidance') -> str:
    """Format stop loss as a human-readable string from SpreadGuidance.
    
    Single source of truth — used by get_strategy_summary,
    _get_strategy_summary_with_path_risk, and DECISION_STRATEGY_MAP.
    """
    if not spread or spread.stop_loss_pct <= 0:
        return ""
    return (
        f"{spread.stop_loss_pct*100:.1f}% stop | "
        f"scale to 25% on day {spread.scale_down_day} | "
        f"hard cut day {spread.max_hold_days}"
    )


@dataclass
class StrategyRecommendation:
    """Complete strategy recommendation"""
    market_state: MarketState
    primary_structures: List[OptionStructure]
    secondary_structures: List[OptionStructure]
    strike_guidance: StrikeSelection
    dte: DTEGuidance
    sizing: PositionSizing
    rationale: str
    spread: Optional[SpreadGuidance] = None
    campaign_bonus: Optional[str] = None  # If whale campaign confirmed


# === STRATEGY TEMPLATES ===
# Recalibrated from analyze_path_stats (14d horizon, 5% target, 4% invalidation, 569 trades)
# Key data: median TTH 3d, 73% hit rate, 39% overshoot→mean-revert path
#
# Stop loss calibration (per-state p75 MAE of winners):
# - stop_loss_pct ≈ 40-50% of spread width (range: 2.5-4.5%)
# - scale_down_day ≈ median TTH (reduce to 25% position)
# - max_hold_days = hard time stop (close all)
# - min_dte ≥ max_hold_days + 2d buffer

STRATEGY_MAP = {
    
    # 🚀 STRONG BULLISH (High Conviction)
    # Data: n=11, 63.6% hit, median TTH 4d, p75 TTH 6d, p75 MAE(W) 4.0%
    MarketState.STRONG_BULLISH: StrategyRecommendation(
        market_state=MarketState.STRONG_BULLISH,
        primary_structures=[
            OptionStructure.CALL_SPREAD,
            OptionStructure.LONG_CALL,
        ],
        secondary_structures=[],
        strike_guidance=StrikeSelection.SLIGHT_ITM,
        dte=DTEGuidance(min_dte=9, max_dte=14, optimal_dte=11),  # min_dte: max_hold(7)+2
        sizing=PositionSizing(high_confidence=0.07, medium_confidence=0.04, low_confidence=0.02),
        rationale="Fresh capital + smart money + exhaustion resolved. ATM strikes survive 3% shakeouts.",
        spread=SpreadGuidance(width_pct=0.09, take_profit_pct=0.70, max_hold_days=7, stop_loss_pct=0.04, scale_down_day=5),
        campaign_bonus="If whale_mega_campaign_accum=1: extend DTE to 21-30d",
    ),
    
    # 📈 EARLY RECOVERY (Asymmetric Upside)
    # Data: n=17, 88.2% hit, median TTH 3d, p75 TTH 5d, p75 MAE(W) 3.77%, 6.7% inv
    MarketState.EARLY_RECOVERY: StrategyRecommendation(
        market_state=MarketState.EARLY_RECOVERY,
        primary_structures=[
            OptionStructure.CALL_SPREAD,
            OptionStructure.LONG_CALL,
        ],
        secondary_structures=[],
        strike_guidance=StrikeSelection.SLIGHT_ITM,
        dte=DTEGuidance(min_dte=14, max_dte=30, optimal_dte=21),
        sizing=PositionSizing(high_confidence=0.05, medium_confidence=0.03, low_confidence=0.015),
        rationale="Direction likely right, timing uncertain. Defined-risk spreads.",
        spread=SpreadGuidance(width_pct=0.11, take_profit_pct=0.70, max_hold_days=7, stop_loss_pct=0.04, scale_down_day=4),
        campaign_bonus="If whale campaign: extend DTE to 21-30d",
    ),
    
    # 🔥 MOMENTUM CONTINUATION
    # Data: n=191, 81.7% hit, median TTH 4d, p75 TTH 8d, p75 MAE(W) 4.54%, 26.3% inv
    MarketState.MOMENTUM_CONTINUATION: StrategyRecommendation(
        market_state=MarketState.MOMENTUM_CONTINUATION,
        primary_structures=[
            OptionStructure.CALL_SPREAD,
            OptionStructure.LONG_CALL,
        ],
        secondary_structures=[],
        strike_guidance=StrikeSelection.SLIGHT_ITM,
        dte=DTEGuidance(min_dte=12, max_dte=21, optimal_dte=14),  # min_dte: max_hold(10)+2
        sizing=PositionSizing(high_confidence=0.04, medium_confidence=0.03, low_confidence=0.01),
        rationale="Trend continuation. 21.5% of winners hit days 8-14; wider DTE avoids theta decay.",
        spread=SpreadGuidance(width_pct=0.10, take_profit_pct=0.70, max_hold_days=10, stop_loss_pct=0.045, scale_down_day=6),
    ),
    
    # ⚠️ DISTRIBUTION RISK
    # Data: n=35, 65.7% hit, median TTH 2d, p75 TTH 5d, p75 MAE(W) 3.07%, 17.4% inv
    MarketState.DISTRIBUTION_RISK: StrategyRecommendation(
        market_state=MarketState.DISTRIBUTION_RISK,
        primary_structures=[
            OptionStructure.PUT_SPREAD,
        ],
        secondary_structures=[],
        strike_guidance=StrikeSelection.SLIGHT_ITM,
        dte=DTEGuidance(min_dte=8, max_dte=14, optimal_dte=12),  # min_dte: max_hold(6)+2
        sizing=PositionSizing(high_confidence=0.03, medium_confidence=0.02, low_confidence=0.0),
        rationale="Smart money exiting. Fast drops → put spreads with defined risk.",
        spread=SpreadGuidance(width_pct=0.08, take_profit_pct=0.70, max_hold_days=6, stop_loss_pct=0.035, scale_down_day=4),
    ),
    
    # 🐻 BEAR CONTINUATION
    # Data: n=7, 85.7% hit, median TTH 4d, p75 TTH 4d, p75 MAE(W) 2.81%, 16.7% inv
    # Small sample — no stop/width changes, sizing-only if warranted.
    MarketState.BEAR_CONTINUATION: StrategyRecommendation(
        market_state=MarketState.BEAR_CONTINUATION,
        primary_structures=[
            OptionStructure.PUT_SPREAD,
        ],
        secondary_structures=[
            OptionStructure.PUT_BACKSPREAD,
        ],
        strike_guidance=StrikeSelection.SLIGHT_ITM,
        dte=DTEGuidance(min_dte=8, max_dte=14, optimal_dte=12),  # min_dte: max_hold(6)+2
        sizing=PositionSizing(high_confidence=0.04, medium_confidence=0.02, low_confidence=0.0),
        rationale="Sellers in control. ATM put spreads survive 5%+ bounce before resuming.",
        spread=SpreadGuidance(width_pct=0.10, take_profit_pct=0.70, max_hold_days=6, stop_loss_pct=0.035, scale_down_day=4),
    ),
    
    # 🟢 BULL PROBE (Exploratory Long)
    # Data: n=83, 75.9% hit, median TTH 3d, p75 TTH 5d, p75 MAE(W) 3.54%, 19.0% inv
    MarketState.BULL_PROBE: StrategyRecommendation(
        market_state=MarketState.BULL_PROBE,
        primary_structures=[
            OptionStructure.CALL_SPREAD,
        ],
        secondary_structures=[],
        strike_guidance=StrikeSelection.SLIGHT_ITM,
        dte=DTEGuidance(min_dte=10, max_dte=14, optimal_dte=12),  # min_dte: max_hold(8)+2
        sizing=PositionSizing(high_confidence=0.02, medium_confidence=0.015, low_confidence=0.0),
        rationale="Early signs of buying. Small defined-risk probe. 16.9% of winners hit days 8-14.",
        spread=SpreadGuidance(width_pct=0.07, take_profit_pct=0.70, max_hold_days=8, stop_loss_pct=0.035, scale_down_day=4),
    ),
    
    # 🔍 BEAR PROBE (Exploratory Short)
    # Data: n=36, 55.6% hit, median TTH 6.5d, p75 TTH 9d, p75 MAE(W) 15.27%, 55% inv
    # Policy: cut sizing, keep stop unchanged. Path-risk (55%>>30%) pushes strike→ITM + DTE≥14d.
    MarketState.BEAR_PROBE: StrategyRecommendation(
        market_state=MarketState.BEAR_PROBE,
        primary_structures=[
            OptionStructure.PUT_SPREAD,
        ],
        secondary_structures=[],
        strike_guidance=StrikeSelection.SLIGHT_ITM,
        dte=DTEGuidance(min_dte=12, max_dte=16, optimal_dte=14),  # min_dte: max_hold(10)+2
        sizing=PositionSizing(high_confidence=0.015, medium_confidence=0.01, low_confidence=0.0),  # Cut from 2%/1.5%
        rationale="Early signs of distribution. Small probe; 55% inv rate → path-risk adjusts strike/DTE.",
        spread=SpreadGuidance(width_pct=0.07, take_profit_pct=0.70, max_hold_days=10, stop_loss_pct=0.04, scale_down_day=7),
    ),
    
    # --- BEAR MARKET RULES (Cycle Days 540-900) ---
    
    # Data: n=5, 40% hit — small sample, sizing-only change. Stop/width unchanged.
    MarketState.BEAR_EXHAUSTION_LONG: StrategyRecommendation(
        market_state=MarketState.BEAR_EXHAUSTION_LONG,
        primary_structures=[
            OptionStructure.CALL_SPREAD,
            OptionStructure.LONG_CALL,
        ],
        secondary_structures=[],
        strike_guidance=StrikeSelection.SLIGHT_ITM,
        dte=DTEGuidance(min_dte=8, max_dte=14, optimal_dte=11),  # min_dte: max_hold(6)+2
        sizing=PositionSizing(high_confidence=0.04, medium_confidence=0.02, low_confidence=0.01),  # Cut from 7%/4%/2%
        rationale="Holder capitulation verified by capital inflow. 40% hit rate → reduced sizing.",
        spread=SpreadGuidance(width_pct=0.09, take_profit_pct=0.70, max_hold_days=6, stop_loss_pct=0.04, scale_down_day=5),
    ),
    
    # Data: n=66, 50% hit, median TTH 3d, p75 TTH 7d, p75 MAE(W) 3.42%, 21.2% inv
    MarketState.BEAR_RALLY_LONG: StrategyRecommendation(
        market_state=MarketState.BEAR_RALLY_LONG,
        primary_structures=[
            OptionStructure.CALL_SPREAD,
        ],
        secondary_structures=[],
        strike_guidance=StrikeSelection.SLIGHT_ITM,
        dte=DTEGuidance(min_dte=10, max_dte=14, optimal_dte=12),  # min_dte: max_hold(8)+2
        sizing=PositionSizing(high_confidence=0.03, medium_confidence=0.015, low_confidence=0.01),  # Cut from 4%/2%
        rationale="Underwater holders getting temporary relief. 50% hit → reduced sizing.",
        spread=SpreadGuidance(width_pct=0.08, take_profit_pct=0.70, max_hold_days=8, stop_loss_pct=0.035, scale_down_day=4),
    ),
    
    # Data: n=4, 100% hit — small sample, no changes. Sizing-only rule.
    MarketState.BEAR_CONTINUATION_SHORT: StrategyRecommendation(
        market_state=MarketState.BEAR_CONTINUATION_SHORT,
        primary_structures=[
            OptionStructure.PUT_SPREAD,
        ],
        secondary_structures=[
            OptionStructure.PUT_BACKSPREAD,
        ],
        strike_guidance=StrikeSelection.SLIGHT_ITM,
        dte=DTEGuidance(min_dte=8, max_dte=14, optimal_dte=12),  # min_dte: max_hold(6)+2
        sizing=PositionSizing(high_confidence=0.05, medium_confidence=0.03, low_confidence=0.01),
        rationale="Profitable holders selling into weakness/aging flows. Strong downside expected.",
        spread=SpreadGuidance(width_pct=0.10, take_profit_pct=0.70, max_hold_days=6, stop_loss_pct=0.035, scale_down_day=4),
    ),
    
    # Data: n=59, 71.2% hit, median TTH 4d, p75 TTH 7.8d, p75 MAE(W) 4.24%, 28.6% inv
    MarketState.LATE_DISTRIBUTION_SHORT: StrategyRecommendation(
        market_state=MarketState.LATE_DISTRIBUTION_SHORT,
        primary_structures=[
            OptionStructure.PUT_SPREAD,
        ],
        secondary_structures=[],
        strike_guidance=StrikeSelection.SLIGHT_ITM,
        dte=DTEGuidance(min_dte=10, max_dte=14, optimal_dte=12),  # min_dte: max_hold(8)+2
        sizing=PositionSizing(high_confidence=0.03, medium_confidence=0.02, low_confidence=0.0),
        rationale="Breakeven/profitable holders losing conviction. Late stage short probe.",
        spread=SpreadGuidance(width_pct=0.08, take_profit_pct=0.70, max_hold_days=8, stop_loss_pct=0.04, scale_down_day=5),
    ),
    
    MarketState.TRANSITION_CHOP: StrategyRecommendation(
        market_state=MarketState.TRANSITION_CHOP,
        primary_structures=[
            OptionStructure.NO_TRADE,
        ],
        secondary_structures=[],
        strike_guidance=StrikeSelection.ATM,
        dte=DTEGuidance(min_dte=0, max_dte=0, optimal_dte=0),
        sizing=PositionSizing(high_confidence=0.0, medium_confidence=0.0, low_confidence=0.0),
        rationale="Conflicting valuation and flow data during bear market. Stand down.",
    ),
    
    # 🟡 NO TRADE
    MarketState.NO_TRADE: StrategyRecommendation(
        market_state=MarketState.NO_TRADE,
        primary_structures=[
            OptionStructure.NO_TRADE,
        ],
        secondary_structures=[],
        strike_guidance=StrikeSelection.ATM,  # N/A
        dte=DTEGuidance(min_dte=0, max_dte=0, optimal_dte=0),
        sizing=PositionSizing(high_confidence=0.0, medium_confidence=0.0, low_confidence=0.0),
        rationale="Conflicts everywhere. Market extracting premium from impatient traders.",
    ),
}


# === DECISION-BASED STRATEGY OVERRIDES ===
# For trade decisions that don't map 1:1 to a fusion MarketState.
# Keyed by trade_decision string (from services.py).
# SpreadGuidance objects ensure stop_loss strings stay in sync with params.
_OPTION_CALL_SPREAD = SpreadGuidance(width_pct=0.0, take_profit_pct=0.0, max_hold_days=7, stop_loss_pct=0.05, scale_down_day=4)
_OPTION_PUT_SPREAD = SpreadGuidance(width_pct=0.0, take_profit_pct=0.0, max_hold_days=6, stop_loss_pct=0.05, scale_down_day=4)
_TACTICAL_PUT_SPREAD = SpreadGuidance(width_pct=0.0, take_profit_pct=0.0, max_hold_days=6, stop_loss_pct=0.035, scale_down_day=4)

DECISION_STRATEGY_MAP = {
    "OPTION_CALL": {
        "primary_structures": "long_call, call_spread",
        "strike_guidance": "slight_itm",
        "dte_range": "7-14d",
        "rationale": "Rule: MVRV cheap + Sentiment fear. Exploratory long probe.",
        "stop_loss": format_stop_loss_string(_OPTION_CALL_SPREAD),
        "stop_loss_pct": _OPTION_CALL_SPREAD.stop_loss_pct,
        "scale_down_day": _OPTION_CALL_SPREAD.scale_down_day,
        "max_hold_days": _OPTION_CALL_SPREAD.max_hold_days,
        "spread_width_pct": _OPTION_CALL_SPREAD.width_pct,
        "take_profit_pct": _OPTION_CALL_SPREAD.take_profit_pct,
    },
    "OPTION_PUT": {
        "primary_structures": "long_put, put_spread",
        "strike_guidance": "slight_itm",
        "dte_range": "7-14d",
        "rationale": "Rule: MVRV overheated + Sentiment greed. Defined-risk short.",
        "stop_loss": format_stop_loss_string(_OPTION_PUT_SPREAD),
        "stop_loss_pct": _OPTION_PUT_SPREAD.stop_loss_pct,
        "scale_down_day": _OPTION_PUT_SPREAD.scale_down_day,
        "max_hold_days": _OPTION_PUT_SPREAD.max_hold_days,
        "spread_width_pct": _OPTION_PUT_SPREAD.width_pct,
        "take_profit_pct": _OPTION_PUT_SPREAD.take_profit_pct,
    },
    "TACTICAL_PUT": {
        "primary_structures": "put_spread",
        "strike_guidance": "slight_otm",
        "dte_range": "7-12d",
        "rationale": "Hedge inside bull: MVRV-60d near-peak & rolling over.",
        "stop_loss": format_stop_loss_string(_TACTICAL_PUT_SPREAD),
        "stop_loss_pct": _TACTICAL_PUT_SPREAD.stop_loss_pct,
        "scale_down_day": _TACTICAL_PUT_SPREAD.scale_down_day,
        "max_hold_days": _TACTICAL_PUT_SPREAD.max_hold_days,
        "spread_width_pct": _TACTICAL_PUT_SPREAD.width_pct,
        "take_profit_pct": _TACTICAL_PUT_SPREAD.take_profit_pct,
    },
}

# MVRV Short: Bear market tactical short with DCA
# 33% initial entry, 67% DCA if price rises 4% against
# Target: 4% drop within 5 days
_MVRV_SHORT_SPREAD = SpreadGuidance(
    width_pct=0.0,  # Not a spread, direct short
    take_profit_pct=0.04,  # 4% target
    max_hold_days=5,
    stop_loss_pct=0.04,  # DCA trigger (not a stop, but price rise threshold)
    scale_down_day=0,  # N/A - DCA strategy instead
)

DECISION_STRATEGY_MAP["MVRV_SHORT"] = {
    "primary_structures": "short_perp, short_future",
    "strike_guidance": "n/a",
    "dte_range": "5d max hold",
    "rationale": "Bear mode + MVRV 7d >= 1.02 + MVRV 60d >= 1.0. Short-term greed without capitulation cushion.",
    "stop_loss": "DCA at +4%, target -4%, 5d window",
    "stop_loss_pct": _MVRV_SHORT_SPREAD.stop_loss_pct,
    "scale_down_day": None,  # DCA strategy instead
    "max_hold_days": _MVRV_SHORT_SPREAD.max_hold_days,
    "spread_width_pct": None,
    "take_profit_pct": _MVRV_SHORT_SPREAD.take_profit_pct,
}

# Iron Condor: Range-bound strategy when chop gate passes
# 7d DTE, ±10% wings — optimised for capital preservation
_IRON_CONDOR_SPREAD = SpreadGuidance(
    width_pct=0.10,       # 10% wing width (10% OTM each side)
    take_profit_pct=0.50,  # Take profit at 50% of max credit
    max_hold_days=7,       # Aligned with 7-day range horizon
    stop_loss_pct=0.06,    # Exit if underlying moves 6% from entry
    scale_down_day=5,      # Reduce to 25% on day 5 if not yet profitable
)

DECISION_STRATEGY_MAP["IRON_CONDOR"] = {
    "primary_structures": "iron_condor",
    "strike_guidance": "otm",
    "dte_range": "7-14d",
    "rationale": "Range gate: chop state + neutral metrics + no directional signals. Sell premium, wide wings.",
    "stop_loss": format_stop_loss_string(_IRON_CONDOR_SPREAD),
    "stop_loss_pct": _IRON_CONDOR_SPREAD.stop_loss_pct,
    "scale_down_day": _IRON_CONDOR_SPREAD.scale_down_day,
    "max_hold_days": _IRON_CONDOR_SPREAD.max_hold_days,
    "spread_width_pct": _IRON_CONDOR_SPREAD.width_pct,
    "take_profit_pct": _IRON_CONDOR_SPREAD.take_profit_pct,
}

_EMPTY_STRATEGY = {
    "primary_structures": "",
    "strike_guidance": "",
    "dte_range": "",
    "rationale": "",
    "stop_loss": "",
    "stop_loss_pct": None,
    "scale_down_day": None,
    "max_hold_days": None,
    "spread_width_pct": None,
    "take_profit_pct": None,
}


def get_decision_strategy_summary(trade_decision: str) -> dict:
    """Get strategy summary for non-fusion trade decisions (OPTION_CALL, OPTION_PUT, TACTICAL_PUT)."""
    return DECISION_STRATEGY_MAP.get(trade_decision, _EMPTY_STRATEGY)


def get_strategy(state: MarketState) -> StrategyRecommendation:
    """Get strategy recommendation for a market state"""
    return STRATEGY_MAP[state]


def get_strategy_with_path_risk(
    state: MarketState,
    invalid_before_hit_rate: Optional[float] = None,
    same_day_ambiguous_rate: Optional[float] = None,
) -> StrategyRecommendation:
    """
    Return strategy with conservative adjustments for invalidation-heavy regimes.

    Rates should be decimals (e.g. 0.32 for 32%).
    """
    strategy = STRATEGY_MAP[state]
    if invalid_before_hit_rate is None and same_day_ambiguous_rate is None:
        return strategy

    inv_rate = invalid_before_hit_rate or 0.0
    amb_rate = same_day_ambiguous_rate or 0.0

    # If stops are frequently challenged before target, bias to more survivable setups.
    if inv_rate >= 0.30 or (inv_rate + amb_rate) >= 0.35:
        if strategy.strike_guidance == StrikeSelection.ATM:
            safer_strike = StrikeSelection.SLIGHT_ITM
        elif strategy.strike_guidance == StrikeSelection.SLIGHT_ITM:
            safer_strike = StrikeSelection.ITM
        else:
            safer_strike = strategy.strike_guidance
        safer_dte = DTEGuidance(
            min_dte=max(strategy.dte.min_dte, 10),
            max_dte=max(strategy.dte.max_dte, 14),
            optimal_dte=max(strategy.dte.optimal_dte, 14),
        )
        return replace(strategy, strike_guidance=safer_strike, dte=safer_dte)

    return strategy


def get_strategy_summary(state: MarketState) -> dict:
    """
    Get strategy summary as a structured dict for signal pipeline.
    
    Returns:
        dict with keys:
            - primary_structures: comma-separated list of primary structures
            - strike_guidance: ATM, OTM, etc.
            - dte_range: e.g., "45-90d"
            - rationale: strategy rationale text
    """
    strategy = STRATEGY_MAP.get(state)
    
    if strategy is None or state == MarketState.NO_TRADE:
        return {
            "primary_structures": "",
            "strike_guidance": "",
            "dte_range": "",
            "rationale": "",
            "stop_loss": "",
        }
    
    structures = ", ".join(s.value for s in strategy.primary_structures)
    dte_range = f"{strategy.dte.min_dte}-{strategy.dte.max_dte}d"
    
    stop_loss = format_stop_loss_string(strategy.spread)
    
    return {
        "primary_structures": structures,
        "strike_guidance": strategy.strike_guidance.value,
        "dte_range": dte_range,
        "rationale": strategy.rationale,
        "stop_loss": stop_loss,
    }


def get_position_size(state: MarketState, confidence: Confidence, portfolio_value: float) -> float:
    """
    Calculate dollar position size.
    
    Args:
        state: Current market state
        confidence: Confidence level
        portfolio_value: Total portfolio in dollars
    
    Returns:
        Dollar amount to allocate
    """
    strategy = get_strategy(state)
    pct = strategy.sizing.get_size(confidence)
    return portfolio_value * pct


def format_recommendation(state: MarketState, confidence: Confidence, 
                          whale_campaign: bool = False) -> str:
    """
    Format a human-readable recommendation.
    """
    strategy = STRATEGY_MAP[state]
    
    lines = [
        f"=== {state.value.upper()} ===",
        f"Confidence: {confidence.value.upper()}",
        f"",
        f"Primary Structures:",
    ]
    
    for struct in strategy.primary_structures:
        lines.append(f"  • {struct.value}")
    
    if strategy.secondary_structures:
        lines.append(f"Secondary Structures:")
        for struct in strategy.secondary_structures:
            lines.append(f"  • {struct.value}")
    
    lines.extend([
        f"",
        f"Strike: {strategy.strike_guidance.value}",
        f"DTE: {strategy.dte}",
        f"",
        f"Rationale: {strategy.rationale}",
    ])
    
    if whale_campaign and strategy.campaign_bonus:
        lines.append(f"Campaign Bonus: {strategy.campaign_bonus}")
    
    return "\n".join(lines)


# === TRADE SIGNAL GENERATION ===

@dataclass
class TradeSignal:
    """Actionable trade signal"""
    date: str
    market_state: MarketState
    confidence: Confidence
    fusion_score: int
    direction: str  # 'long', 'short', 'neutral'
    structures: List[OptionStructure]
    strike_guidance: StrikeSelection
    min_dte: int
    max_dte: int
    position_size_pct: float
    whale_campaign: bool
    rationale: str


def generate_trade_signal(row: dict, date: str) -> TradeSignal:
    """
    Generate actionable trade signal from a feature row.
    
    Args:
        row: Dict of features for a single day
        date: Date string
    
    Returns:
        TradeSignal object
    """
    from .fusion import fuse_signals
    import pandas as pd
    
    # Fuse signals
    result = fuse_signals(pd.Series(row))
    strategy = get_strategy(result.state)
    
    # Determine direction
    if result.state in [MarketState.STRONG_BULLISH, MarketState.EARLY_RECOVERY, 
                        MarketState.MOMENTUM_CONTINUATION, MarketState.BULL_PROBE,
                        MarketState.BEAR_EXHAUSTION_LONG, MarketState.BEAR_RALLY_LONG]:
        direction = 'long'
    elif result.state in [MarketState.DISTRIBUTION_RISK, MarketState.BEAR_CONTINUATION, MarketState.BEAR_PROBE,
                          MarketState.BEAR_CONTINUATION_SHORT, MarketState.LATE_DISTRIBUTION_SHORT]:
        direction = 'short'
    else:
        direction = 'neutral'
    
    # Check whale campaign
    whale_campaign = row.get('whale_mega_campaign_accum', 0) == 1 or \
                     row.get('whale_mega_campaign_distrib', 0) == 1

    # Runtime structure gating:
    # - Backspreads for robust structural states
    # - Credit spreads only in high-IV environment and reduced by policy outside this function
    structures = list(strategy.primary_structures)
    if result.state in [MarketState.STRONG_BULLISH, MarketState.BEAR_EXHAUSTION_LONG]:
        structures.append(OptionStructure.CALL_BACKSPREAD)
    if result.state in [MarketState.BEAR_CONTINUATION, MarketState.BEAR_CONTINUATION_SHORT]:
        structures.append(OptionStructure.PUT_BACKSPREAD)

    iv_keys = ("iv_percentile", "btc_iv_percentile", "options_iv_percentile")
    iv_percentile = next((float(row[k]) for k in iv_keys if k in row and row[k] is not None), None)
    if iv_percentile is not None and iv_percentile >= 0.85 and result.state in {
        MarketState.DISTRIBUTION_RISK,
        MarketState.BEAR_CONTINUATION,
        MarketState.BEAR_CONTINUATION_SHORT,
    }:
        structures.append(OptionStructure.SHORT_CALL_SPREAD)
    
    return TradeSignal(
        date=date,
        market_state=result.state,
        confidence=result.confidence,
        fusion_score=result.score,
        direction=direction,
        structures=structures,
        strike_guidance=strategy.strike_guidance,
        min_dte=strategy.dte.min_dte,
        max_dte=strategy.dte.max_dte,
        position_size_pct=strategy.sizing.get_size(result.confidence),
        whale_campaign=whale_campaign,
        rationale=strategy.rationale,
    )
