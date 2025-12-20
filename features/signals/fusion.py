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
    """6 canonical market states for trading decisions"""
    STRONG_BULLISH = "strong_bullish"        # 🚀 High conviction long
    EARLY_RECOVERY = "early_recovery"        # 📈 Asymmetric upside
    MOMENTUM_CONTINUATION = "momentum"       # 🔥 Trend continuation
    DISTRIBUTION_RISK = "distribution_risk"  # ⚠️ Smart money exiting
    BEAR_CONTINUATION = "bear_continuation"  # 🐻 No buyers, sellers in control
    NO_TRADE = "no_trade"                    # 🟡 Chop, conflicts everywhere


class Confidence(Enum):
    """Confidence levels for position sizing"""
    HIGH = "high"      # ≥5 score
    MEDIUM = "medium"  # 3-4 score
    LOW = "low"        # ≤2 score


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
    if row.get('mvrv_ls_regime_call_confirm', 0) == 1:
        score += 2
        components['mvrv_ls'] = '+2 (call_confirm)'
    elif row.get('mvrv_ls_regime_call_confirm_recovery', 0) == 1:
        score += 1
        components['mvrv_ls'] = '+1 (early_recovery)'
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
    if score >= 5:
        return Confidence.HIGH
    elif score >= 3:
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
    whale_distrib = row.get('whale_regime_distribution', 0) == 1
    whale_mixed = row.get('whale_regime_mixed', 0) == 1
    
    mvrv_call = row.get('mvrv_ls_regime_call_confirm', 0) == 1
    mvrv_recovery = row.get('mvrv_ls_regime_call_confirm_recovery', 0) == 1
    mvrv_trend = row.get('mvrv_ls_regime_call_confirm_trend', 0) == 1
    mvrv_put = row.get('mvrv_ls_regime_put_confirm', 0) == 1
    mvrv_bear = row.get('mvrv_ls_regime_bear_continuation', 0) == 1
    mvrv_rollover = row.get('mvrv_ls_early_rollover', 0) == 1
    mvrv_weak_down = row.get('mvrv_ls_weak_downtrend', 0) == 1
    mvrv_distrib_warn = row.get('mvrv_ls_regime_distribution_warning', 0) == 1
    
    # === CLASSIFICATION RULES (most specific first) ===
    
    # 🚀 STRONG BULLISH: All aligned bullish
    if mdia_strong and (whale_broad or whale_strategic) and mvrv_call:
        return MarketState.STRONG_BULLISH
    
    # 📈 EARLY RECOVERY: Smart money leading, structure turning
    if mdia_inflow and whale_strategic and mvrv_recovery:
        return MarketState.EARLY_RECOVERY
    
    # 🐻 BEAR CONTINUATION: No buyers, sellers in control
    if (mdia_distrib or not mdia_inflow) and whale_distrib and (mvrv_put or mvrv_bear):
        return MarketState.BEAR_CONTINUATION
    
    # ⚠️ DISTRIBUTION RISK: Smart money exiting, structure cracking
    if whale_distrib and not mdia_strong and (mvrv_rollover or mvrv_weak_down or mvrv_distrib_warn):
        return MarketState.DISTRIBUTION_RISK
    if mvrv_distrib_warn and whale_distrib:
        return MarketState.DISTRIBUTION_RISK
    
    # 🔥 MOMENTUM CONTINUATION: Trend continuation without strong sponsorship
    if mdia_inflow and (whale_mixed or not whale_distrib) and mvrv_trend:
        return MarketState.MOMENTUM_CONTINUATION
    
    # 🟡 NO TRADE: Conflicts or no alignment
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

def add_fusion_features(feats: dict, df_features: pd.DataFrame) -> dict:
    """
    Add fusion features directly into feature dict during build.
    Called from feature_builder.py.
    
    Args:
        feats: The features dict being built
        df_features: DataFrame of features built so far (converted from feats dict)
    
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
