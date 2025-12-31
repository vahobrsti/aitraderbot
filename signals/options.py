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

from dataclasses import dataclass
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
    RATIO_CALL_SPREAD = "ratio_call_spread"
    BROKEN_WING_BUTTERFLY = "broken_wing_butterfly"
    
    # No action
    NO_TRADE = "no_trade"


class StrikeSelection(Enum):
    """Strike selection guidance"""
    ATM = "atm"              # At-the-money
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
class StrategyRecommendation:
    """Complete strategy recommendation"""
    market_state: MarketState
    primary_structures: List[OptionStructure]
    secondary_structures: List[OptionStructure]
    strike_guidance: StrikeSelection
    dte: DTEGuidance
    sizing: PositionSizing
    rationale: str
    campaign_bonus: Optional[str] = None  # If whale campaign confirmed


# === STRATEGY TEMPLATES ===

STRATEGY_MAP = {
    
    # 🚀 STRONG BULLISH (High Conviction)
    MarketState.STRONG_BULLISH: StrategyRecommendation(
        market_state=MarketState.STRONG_BULLISH,
        primary_structures=[
            OptionStructure.LONG_CALL,
            OptionStructure.CALL_SPREAD,
        ],
        secondary_structures=[
            OptionStructure.CALL_DIAGONAL,  # If IV elevated
        ],
        strike_guidance=StrikeSelection.SLIGHT_OTM,
        dte=DTEGuidance(min_dte=45, max_dte=90, optimal_dte=60),
        sizing=PositionSizing(high_confidence=0.05, medium_confidence=0.03, low_confidence=0.01),
        rationale="Fresh capital + smart money + exhaustion resolved. Size up.",
        campaign_bonus="If whale_mega_campaign_accum=1: extend DTE to 90-120d",
    ),
    
    # 📈 EARLY RECOVERY (Asymmetric Upside)
    MarketState.EARLY_RECOVERY: StrategyRecommendation(
        market_state=MarketState.EARLY_RECOVERY,
        primary_structures=[
            OptionStructure.CALL_SPREAD,
            OptionStructure.CALL_CALENDAR,
        ],
        secondary_structures=[
            OptionStructure.RATIO_CALL_SPREAD,  # Advanced
        ],
        strike_guidance=StrikeSelection.ATM,
        dte=DTEGuidance(min_dte=60, max_dte=120, optimal_dte=90),
        sizing=PositionSizing(high_confidence=0.04, medium_confidence=0.02, low_confidence=0.01),
        rationale="Direction likely right, timing uncertain. Options shine here.",
        campaign_bonus="Use long back leg (60-120d), short front leg (15-30d)",
    ),
    
    # 🔥 MOMENTUM CONTINUATION
    MarketState.MOMENTUM_CONTINUATION: StrategyRecommendation(
        market_state=MarketState.MOMENTUM_CONTINUATION,
        primary_structures=[
            OptionStructure.CALL_SPREAD,
            OptionStructure.CALL_DIAGONAL,
        ],
        secondary_structures=[
            OptionStructure.LONG_CALL,  # Shorter dated
        ],
        strike_guidance=StrikeSelection.SLIGHT_OTM,
        dte=DTEGuidance(min_dte=21, max_dte=45, optimal_dte=30),
        sizing=PositionSizing(high_confidence=0.03, medium_confidence=0.02, low_confidence=0.0),
        rationale="Trend continuation without strong sponsorship. Good trades, not lifetime trades.",
    ),
    
    # ⚠️ DISTRIBUTION RISK
    MarketState.DISTRIBUTION_RISK: StrategyRecommendation(
        market_state=MarketState.DISTRIBUTION_RISK,
        primary_structures=[
            OptionStructure.PUT_CALENDAR,
            OptionStructure.PUT_DIAGONAL,
        ],
        secondary_structures=[
            OptionStructure.SHORT_CALL_SPREAD,  # Advanced
        ],
        strike_guidance=StrikeSelection.ATM,
        dte=DTEGuidance(min_dte=45, max_dte=90, optimal_dte=60),
        sizing=PositionSizing(high_confidence=0.03, medium_confidence=0.02, low_confidence=0.0),
        rationale="Smart money exiting. Profit from stalls/reversals. Theta + convexity.",
        campaign_bonus="Short leg 7-21d, Long leg 45-90d",
    ),
    
    # 🐻 BEAR CONTINUATION
    MarketState.BEAR_CONTINUATION: StrategyRecommendation(
        market_state=MarketState.BEAR_CONTINUATION,
        primary_structures=[
            OptionStructure.PUT_SPREAD,
            OptionStructure.PUT_DIAGONAL,
        ],
        secondary_structures=[
            OptionStructure.BROKEN_WING_BUTTERFLY,
        ],
        strike_guidance=StrikeSelection.SLIGHT_OTM,
        dte=DTEGuidance(min_dte=30, max_dte=60, optimal_dte=45),
        sizing=PositionSizing(high_confidence=0.04, medium_confidence=0.02, low_confidence=0.0),
        rationale="No buyers, sellers in control. Directional downside with protection.",
    ),
    
    # � BULL PROBE (Exploratory Long)
    MarketState.BULL_PROBE: StrategyRecommendation(
        market_state=MarketState.BULL_PROBE,
        primary_structures=[
            OptionStructure.CALL_SPREAD,
        ],
        secondary_structures=[
            OptionStructure.CALL_DIAGONAL,
        ],
        strike_guidance=StrikeSelection.ATM,
        dte=DTEGuidance(min_dte=30, max_dte=60, optimal_dte=45),
        sizing=PositionSizing(high_confidence=0.02, medium_confidence=0.015, low_confidence=0.0),
        rationale="Early signs of buying. Small probe position with defined risk.",
    ),
    
    # 🔍 BEAR PROBE (Exploratory Short)
    MarketState.BEAR_PROBE: StrategyRecommendation(
        market_state=MarketState.BEAR_PROBE,
        primary_structures=[
            OptionStructure.PUT_SPREAD,
        ],
        secondary_structures=[
            OptionStructure.PUT_DIAGONAL,
        ],
        strike_guidance=StrikeSelection.ATM,
        dte=DTEGuidance(min_dte=30, max_dte=60, optimal_dte=45),
        sizing=PositionSizing(high_confidence=0.02, medium_confidence=0.015, low_confidence=0.0),
        rationale="Early signs of distribution. Small probe position with defined risk.",
    ),
    
    # �🟡 NO TRADE
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


def get_strategy(state: MarketState) -> StrategyRecommendation:
    """Get strategy recommendation for a market state"""
    return STRATEGY_MAP[state]


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
        }
    
    structures = ", ".join(s.value for s in strategy.primary_structures)
    dte_range = f"{strategy.dte.min_dte}-{strategy.dte.max_dte}d"
    
    return {
        "primary_structures": structures,
        "strike_guidance": strategy.strike_guidance.value,
        "dte_range": dte_range,
        "rationale": strategy.rationale,
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
                        MarketState.MOMENTUM_CONTINUATION]:
        direction = 'long'
    elif result.state in [MarketState.DISTRIBUTION_RISK, MarketState.BEAR_CONTINUATION]:
        direction = 'short'
    else:
        direction = 'neutral'
    
    # Check whale campaign
    whale_campaign = row.get('whale_mega_campaign_accum', 0) == 1 or \
                     row.get('whale_mega_campaign_distrib', 0) == 1
    
    return TradeSignal(
        date=date,
        market_state=result.state,
        confidence=result.confidence,
        fusion_score=result.score,
        direction=direction,
        structures=strategy.primary_structures,
        strike_guidance=strategy.strike_guidance,
        min_dte=strategy.dte.min_dte,
        max_dte=strategy.dte.max_dte,
        position_size_pct=strategy.sizing.get_size(result.confidence),
        whale_campaign=whale_campaign,
        rationale=strategy.rationale,
    )
