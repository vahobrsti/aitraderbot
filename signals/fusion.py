# features/signals/fusion.py
"""
Signal Fusion Engine - Layer 1
Turns orthogonal regime signals into unified market states with confidence scores.

Metrics and their roles:
- MDIA: Timing/impulse (is fresh capital entering NOW?)
- Whales: Intent/sponsorship (is smart money backing the move?)  
- MVRV LS: Macro confirmation (is market structurally ready?)

Think of it as:
- MDIA = ignition
- Whales = fuel
- MVRV LS = terrain
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
    BULL_PROBE = "bull_probe"                # 🎯 Timing + sponsorship, macro neutral (0.5x long)
    DISTRIBUTION_RISK = "distribution_risk"  # ⚠️ Smart money exiting
    BEAR_CONTINUATION = "bear_continuation"  # 🐻 No buyers, sellers in control
    BEAR_PROBE = "bear_probe"                # 🔴 Selling + distribution, macro neutral (0.5x short)
    NO_TRADE = "no_trade"                    # 🟡 Chop, conflicts everywhere


class Confidence(Enum):
    """Confidence levels for position sizing"""
    HIGH = "high"      # score >= 4
    MEDIUM = "medium"  # score 2-3
    LOW = "low"        # score < 2


@dataclass
class FusionResult:
    """Result of signal fusion"""
    state: MarketState
    confidence: Confidence
    score: int
    components: dict  # Breakdown of what contributed
    

def compute_confidence_score(row: pd.Series) -> tuple[int, dict]:
    """
    Compute confidence score from regime features.
    Returns (score, components_dict)
    
    Scoring:
    +2: MDIA strong_inflow, whale broad_accum, MVRV call_confirm
    +1: MDIA inflow (moderate), whale strategic, MVRV early_recovery
    -2: whale distribution
    -1: conflicts anywhere
    """
    score = 0
    components = {}
    
    # MDIA contribution
    if row.get('mdia_regime_strong_inflow', 0) == 1:
        score += 2
        components['mdia'] = '+2 (strong_inflow)'
    elif row.get('mdia_regime_inflow', 0) == 1:
        score += 1
        components['mdia'] = '+1 (inflow)'
    elif row.get('mdia_regime_distribution', 0) == 1:
        score -= 1
        components['mdia'] = '-1 (distribution)'
    else:
        components['mdia'] = '0 (neutral)'
    
    # Whale contribution
    if row.get('whale_regime_broad_accum', 0) == 1:
        score += 2
        components['whale'] = '+2 (broad_accum)'
    elif row.get('whale_regime_strategic_accum', 0) == 1:
        score += 1
        components['whale'] = '+1 (strategic)'
    elif row.get('whale_regime_distribution_strong', 0) == 1:
        score -= 2
        components['whale'] = '-2 (strong_distrib)'
    elif row.get('whale_regime_distribution', 0) == 1:
        score -= 1
        components['whale'] = '-1 (distribution)'
    else:
        components['whale'] = '0 (neutral)'
    
    # MVRV LS contribution
    # NOTE: Check recovery FIRST because it's a subset of call_confirm
    # (both require strong_uptrend, but recovery also requires level < 0)
    if row.get('mvrv_ls_regime_call_confirm_recovery', 0) == 1:
        score += 2  # Recovery is actually the best asymmetric setup
        components['mvrv_ls'] = '+2 (early_recovery)'
    elif row.get('mvrv_ls_regime_call_confirm_trend', 0) == 1:
        score += 1  # Trend continuation is good but less special
        components['mvrv_ls'] = '+1 (trend_confirm)'
    elif row.get('mvrv_ls_regime_call_confirm', 0) == 1:
        # Fallback: call_confirm is true but neither recovery nor trend flagged
        score += 1
        components['mvrv_ls'] = '+1 (call_confirm fallback)'
    elif row.get('mvrv_ls_regime_put_confirm', 0) == 1:
        score -= 2
        components['mvrv_ls'] = '-2 (put_confirm)'
    elif row.get('mvrv_ls_regime_distribution_warning', 0) == 1:
        score -= 1
        components['mvrv_ls'] = '-1 (distribution_warning)'
    else:
        components['mvrv_ls'] = '0 (neutral)'
    
    # Conflict penalty
    conflicts = 0
    if row.get('mvrv_ls_conflict', 0) == 1:
        conflicts += 1
    if row.get('whale_mega_conflict', 0) == 1:
        conflicts += 1
    if row.get('whale_small_conflict', 0) == 1:
        conflicts += 1
    
    if conflicts > 0:
        score -= conflicts
        components['conflicts'] = f'-{conflicts} (regime conflicts)'
    
    return score, components


def score_to_confidence(score: int) -> Confidence:
    """Convert numeric score to confidence level"""
    # Tuned thresholds for 1-3 trades/month with meaningful MEDIUM signals
    if score >= 4:
        return Confidence.HIGH
    elif score >= 2:
        return Confidence.MEDIUM
    else:
        return Confidence.LOW


def classify_market_state(row: pd.Series) -> MarketState:
    """
    Classify row into one of 6 canonical market states.
    Uses hierarchical rules (most specific first).
    """
    
    # Helper lookups
    mdia_strong = row.get('mdia_regime_strong_inflow', 0) == 1
    mdia_inflow = row.get('mdia_regime_inflow', 0) == 1 or mdia_strong
    mdia_distrib = row.get('mdia_regime_distribution', 0) == 1
    
    whale_broad = row.get('whale_regime_broad_accum', 0) == 1
    whale_strategic = row.get('whale_regime_strategic_accum', 0) == 1
    whale_sponsored = whale_broad or whale_strategic  # Any sponsorship
    whale_distrib = row.get('whale_regime_distribution', 0) == 1
    whale_mixed = row.get('whale_regime_mixed', 0) == 1
    whale_neutral = not whale_sponsored and not whale_distrib  # Neither accumulating nor distributing
    
    mvrv_call = row.get('mvrv_ls_regime_call_confirm', 0) == 1
    mvrv_recovery = row.get('mvrv_ls_regime_call_confirm_recovery', 0) == 1
    mvrv_trend = row.get('mvrv_ls_regime_call_confirm_trend', 0) == 1
    mvrv_weak_up = row.get('mvrv_ls_weak_uptrend', 0) == 1  # For looser MOMENTUM
    mvrv_put = row.get('mvrv_ls_regime_put_confirm', 0) == 1
    mvrv_bear = row.get('mvrv_ls_regime_bear_continuation', 0) == 1
    mvrv_rollover = row.get('mvrv_ls_early_rollover', 0) == 1
    mvrv_weak_down = row.get('mvrv_ls_weak_downtrend', 0) == 1
    mvrv_distrib_warn = row.get('mvrv_ls_regime_distribution_warning', 0) == 1
    
    # NOTE: Conflicts are now score-only (not classification gate)
    # This keeps more meaningful states instead of dumping into NO_TRADE
    # Conflicts still reduce score/confidence, which gates execution
    
    # === CLASSIFICATION RULES (most specific first) ===
    
    # 🚀 STRONG BULLISH: All aligned bullish
    if mdia_strong and whale_sponsored and mvrv_call:
        return MarketState.STRONG_BULLISH
    
    # 📈 EARLY RECOVERY: Smart money leading, structure turning
    if mdia_inflow and whale_sponsored and mvrv_recovery:
        return MarketState.EARLY_RECOVERY
    
    # 🐻 BEAR CONTINUATION: No buyers, sellers in control
    if (mdia_distrib or not mdia_inflow) and whale_distrib and (mvrv_put or mvrv_bear):
        return MarketState.BEAR_CONTINUATION
    
    # ⚠️ DISTRIBUTION RISK: Smart money exiting, structure cracking
    if whale_distrib and not mdia_strong and (mvrv_rollover or mvrv_weak_down or mvrv_distrib_warn):
        return MarketState.DISTRIBUTION_RISK
    
    # 🔥 MOMENTUM CONTINUATION: Trend continuation WITHOUT strong sponsorship
    # TUNED: Accept weak_uptrend OR full trend (not just full trend)
    mvrv_improving = mvrv_trend or mvrv_weak_up
    if mdia_inflow and (whale_mixed or whale_neutral) and mvrv_improving:
        return MarketState.MOMENTUM_CONTINUATION
    
    # === PROBE STATES: Timing/sponsorship align, but macro neutral ===
    # These are tradeable at 0.5x size with defined-risk strategies only
    
    # Define MVRV-LS macro terrain
    mvrv_bullish = mvrv_call or mvrv_recovery or mvrv_trend or mvrv_weak_up
    mvrv_bearish = mvrv_put or mvrv_bear or mvrv_rollover or mvrv_weak_down or mvrv_distrib_warn
    mvrv_neutral = (not mvrv_bullish) and (not mvrv_bearish)
    
    # 🎯 BULL PROBE: Timing + sponsorship, macro neutral
    # MDIA inflow + whales accumulating, MVRV doesn't confirm but isn't hostile
    if mdia_inflow and whale_sponsored and mvrv_neutral:
        return MarketState.BULL_PROBE
    
    # 🔴 BEAR PROBE: Selling pressure + distribution, macro neutral
    # Require explicit MDIA distribution (not just absence of inflow)
    if mdia_distrib and whale_distrib and mvrv_neutral:
        return MarketState.BEAR_PROBE
    
    # 🟡 NO TRADE: No alignment
    return MarketState.NO_TRADE


def fuse_signals(row: pd.Series) -> FusionResult:
    """
    Main fusion function: takes a feature row and returns unified market state.
    """
    state = classify_market_state(row)
    score, components = compute_confidence_score(row)
    confidence = score_to_confidence(score)
    
    return FusionResult(
        state=state,
        confidence=confidence,
        score=score,
        components=components
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
    
    Args:
        feats: The features dict being built
    
    Returns:
        feats dict with fusion columns added
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
