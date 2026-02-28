# features/signals/fusion.py
"""
Signal Fusion Engine - Layer 1
Turns orthogonal regime signals into unified market states with confidence scores.

Metrics and their roles:
- MDIA: Timing/impulse (is fresh capital entering NOW?)
- Whales: Intent/sponsorship (is smart money backing the move?)  
- MVRV LS: Macro confirmation (is market structurally ready?)
"""

import pandas as pd
from enum import Enum
from dataclasses import dataclass
from typing import Optional


class MarketState(Enum):
    """8 canonical market states for trading decisions"""
    STRONG_BULLISH = "strong_bullish"        # 🚀 High conviction long
    EARLY_RECOVERY = "early_recovery"        # 📈 Asymmetric upside
    MOMENTUM_CONTINUATION = "momentum"       # 🔥 Trend continuation
    BULL_PROBE = "bull_probe"                # 🎯 Timing + sponsorship, macro neutral
    DISTRIBUTION_RISK = "distribution_risk"  # ⚠️ Smart money exiting
    BEAR_CONTINUATION = "bear_continuation"  # 🐻 No buyers, sellers in control
    BEAR_PROBE = "bear_probe"                # 🔴 Selling + distribution
    NO_TRADE = "no_trade"                    # 🟡 Chop, conflicts everywhere


class Confidence(Enum):
    """Confidence levels for position sizing"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class FusionResult:
    """Result of signal fusion"""
    state: MarketState
    confidence: Confidence
    score: int
    components: dict  # Breakdown of what contributed
    short_source: Optional[str] = None  # Kept for backward compatibility
    

# Static mapping per MarketState instead of additive linear scoring
# Maintains backward compatibility with DB fusion_score integers
STATE_PROPERTIES = {
    MarketState.STRONG_BULLISH:        (Confidence.HIGH, 5),
    MarketState.EARLY_RECOVERY:        (Confidence.HIGH, 4),
    MarketState.MOMENTUM_CONTINUATION: (Confidence.MEDIUM, 3),
    MarketState.BULL_PROBE:            (Confidence.LOW, 1),
    MarketState.NO_TRADE:              (Confidence.LOW, 0),
    MarketState.BEAR_PROBE:            (Confidence.LOW, -2),
    MarketState.DISTRIBUTION_RISK:     (Confidence.MEDIUM, -3),
    MarketState.BEAR_CONTINUATION:     (Confidence.HIGH, -5),
}


def classify_market_state(row: pd.Series) -> MarketState:
    """
    Classify row into one of 8 canonical market states using strictly 
    hierarchical, empirically-derived rules based on research pipeline findings.
    """
    # MDIA
    mdia_strong = row.get('mdia_regime_strong_inflow', 0) == 1
    mdia_inflow = row.get('mdia_regime_inflow', 0) == 1 or mdia_strong
    mdia_aging = row.get('mdia_regime_aging', 0) == 1
    
    # Whales
    whale_broad = row.get('whale_regime_broad_accum', 0) == 1
    whale_strategic = row.get('whale_regime_strategic_accum', 0) == 1
    whale_mixed = row.get('whale_regime_mixed', 0) == 1
    whale_sponsored = whale_broad or whale_strategic
    whale_distrib_strong = row.get('whale_regime_distribution_strong', 0) == 1
    whale_distrib = row.get('whale_regime_distribution', 0) == 1 or whale_distrib_strong
    
    # MVRV LS
    mvrv_call = row.get('mvrv_ls_regime_call_confirm', 0) == 1
    mvrv_recovery = row.get('mvrv_ls_regime_call_confirm_recovery', 0) == 1
    mvrv_trend = row.get('mvrv_ls_regime_call_confirm_trend', 0) == 1
    mvrv_put = row.get('mvrv_ls_regime_put_confirm', 0) == 1
    mvrv_bear = row.get('mvrv_ls_regime_bear_continuation', 0) == 1
    mvrv_rollover = row.get('mvrv_ls_early_rollover', 0) == 1
    mvrv_weak_down = row.get('mvrv_ls_weak_downtrend', 0) == 1
    mvrv_distrib_warn = row.get('mvrv_ls_regime_distribution_warning', 0) == 1

    # Define MVRV macro buckets
    mvrv_macro_bullish = mvrv_call or mvrv_recovery or mvrv_trend
    mvrv_macro_bearish = mvrv_put or mvrv_bear or mvrv_rollover or mvrv_weak_down or mvrv_distrib_warn
    mvrv_macro_neutral = not mvrv_macro_bullish and not mvrv_macro_bearish
    
    # === RULE 1: STRONG BULLISH ===
    if mdia_strong and whale_sponsored and mvrv_macro_bullish:
        return MarketState.STRONG_BULLISH
        
    # === RULE 2: EARLY RECOVERY ===
    if mdia_inflow and whale_sponsored and mvrv_recovery:
        return MarketState.EARLY_RECOVERY
        
    # === RULE 3: BEAR CONTINUATION ===
    if not mdia_inflow and whale_distrib and (mvrv_put or mvrv_bear):
        return MarketState.BEAR_CONTINUATION
        
    # === RULE 4: BEAR PROBE ===
    if not mdia_inflow and whale_distrib_strong and mvrv_macro_neutral:
        return MarketState.BEAR_PROBE
        
    # === RULE 5: DISTRIBUTION RISK ===
    # Vetoed by mvrv_macro_bullish (don't short the bottom)
    if not mdia_inflow and whale_distrib and not mvrv_macro_bullish:
        return MarketState.DISTRIBUTION_RISK
        
    # === RULE 6: MOMENTUM CONTINUATION (Strict Macro Gate) ===
    # Cannot fire in a neutral or bearish MVRV regime. MUST have macro support.
    # Accepts whale_sponsored or whale_mixed.
    if mdia_inflow and (whale_sponsored or whale_mixed) and mvrv_macro_bullish:
        return MarketState.MOMENTUM_CONTINUATION
        
    # === RULE 7: BULL PROBE ===
    if mdia_inflow and whale_sponsored and mvrv_macro_neutral:
        return MarketState.BULL_PROBE
        
    return MarketState.NO_TRADE


def fuse_signals(row: pd.Series) -> FusionResult:
    """
    Main fusion function: takes a feature row and returns unified market state
    with its statically assigned confidence and score.
    """
    state = classify_market_state(row)
    confidence, score = STATE_PROPERTIES[state]
    
    # Track the basic components for UI explainability
    components = {
        'mdia_strong': int(row.get('mdia_regime_strong_inflow', 0) == 1),
        'mdia_inflow': int(row.get('mdia_regime_inflow', 0) == 1 or row.get('mdia_regime_strong_inflow', 0) == 1),
        'mdia_aging': int(row.get('mdia_regime_aging', 0) == 1),
        'whale_sponsored': int(row.get('whale_regime_broad_accum', 0) == 1 or row.get('whale_regime_strategic_accum', 0) == 1),
        'whale_mixed': int(row.get('whale_regime_mixed', 0) == 1),
        'whale_distrib': int(row.get('whale_regime_distribution', 0) == 1 or row.get('whale_regime_distribution_strong', 0) == 1),
        'whale_distrib_strong': int(row.get('whale_regime_distribution_strong', 0) == 1),
        'mvrv_macro_bullish': int(row.get('mvrv_ls_regime_call_confirm', 0) == 1 or row.get('mvrv_ls_regime_call_confirm_recovery', 0) == 1 or row.get('mvrv_ls_regime_call_confirm_trend', 0) == 1),
        'mvrv_macro_bearish': int(row.get('mvrv_ls_regime_put_confirm', 0) == 1 or row.get('mvrv_ls_regime_bear_continuation', 0) == 1 or row.get('mvrv_ls_early_rollover', 0) == 1 or row.get('mvrv_ls_weak_downtrend', 0) == 1 or row.get('mvrv_ls_regime_distribution_warning', 0) == 1),
    }
    components['mvrv_macro_neutral'] = int(not components['mvrv_macro_bullish'] and not components['mvrv_macro_bearish'])
    
    return FusionResult(
        state=state,
        confidence=confidence,
        score=score,
        components=components,
        short_source='rule' if state in [MarketState.BEAR_CONTINUATION, MarketState.DISTRIBUTION_RISK, MarketState.BEAR_PROBE] else None
    )


def fuse_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply fusion to entire feature DataFrame.
    Returns DataFrame with fusion columns added.
    """
    results = []
    
    for idx, row in df.iterrows():
        result = fuse_signals(row)
        results.append({
            'date': idx,
            'market_state': result.state.value,
            'confidence': result.confidence.value,
            'fusion_score': result.score,
        })
    
    fusion_df = pd.DataFrame(results).set_index('date')
    
    # Merge back with original
    return pd.concat([df, fusion_df], axis=1)


# === CONVENIENCE FEATURE GENERATION ===

def add_fusion_features(feats: dict) -> dict:
    """
    Add fusion features directly into feature dict during build.
    Called from feature_builder.py.
    """
    # Build temp DataFrame for classification
    temp_df = pd.DataFrame(feats)
    
    states = []
    scores = []
    confidences = []
    
    for idx in range(len(temp_df)):
        row = temp_df.iloc[idx]
        result = fuse_signals(row)
        states.append(result.state.value)
        scores.append(result.score)
        confidences.append(result.confidence.value)
    
    # Add as features
    feats['fusion_market_state'] = states
    feats['fusion_score'] = scores
    feats['fusion_confidence'] = confidences
    
    # Add binary state flags for ML
    for state in MarketState:
        feats[f'fusion_state_{state.value}'] = [
            1 if s == state.value else 0 for s in states
        ]
    
    return feats
