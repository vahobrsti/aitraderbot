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
from signals.research.bucket_mapping import map_mvrv_60d_bucket


class MarketState(Enum):
    # --- All-Market / Default States ---
    STRONG_BULLISH = "strong_bullish"        # 🚀 High conviction long
    EARLY_RECOVERY = "early_recovery"        # 📈 Asymmetric upside
    MOMENTUM_CONTINUATION = "momentum"       # 🔥 Trend continuation
    BULL_PROBE = "bull_probe"                # 🎯 Timing + sponsorship, macro neutral
    DISTRIBUTION_RISK = "distribution_risk"  # ⚠️ Smart money exiting
    BEAR_CONTINUATION = "bear_continuation"  # 🐻 No buyers, sellers in control
    BEAR_PROBE = "bear_probe"                # 🔴 Selling + distribution
    NO_TRADE = "no_trade"                    # 🟡 Chop, conflicts everywhere

    # --- Bear-Market Specific States ---
    BEAR_EXHAUSTION_LONG = "bear_exhaustion_long"       # 🟢 High-conviction long near washed-out conditions
    BEAR_RALLY_LONG = "bear_rally_long"                 # 📈 tradable rebound starting
    BEAR_CONTINUATION_SHORT = "bear_continuation_short" # 🔴 High-conviction short, downside still active
    LATE_DISTRIBUTION_SHORT = "late_distribution_short" # 📉 weaker short, still bearish
    TRANSITION_CHOP = "transition_chop"                 # 🟡 Conflicting bear signals


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
    
    # Bear-Market Specific Mappings
    MarketState.BEAR_EXHAUSTION_LONG:    (Confidence.HIGH, 4),
    MarketState.BEAR_RALLY_LONG:         (Confidence.MEDIUM, 2),
    MarketState.TRANSITION_CHOP:         (Confidence.LOW, 0),
    MarketState.LATE_DISTRIBUTION_SHORT: (Confidence.MEDIUM, -2),
    MarketState.BEAR_CONTINUATION_SHORT: (Confidence.HIGH, -4),
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


def classify_bear_market_state(row: pd.Series) -> MarketState:
    """
    Classify row into a bear-market specific state.
    Anchors on mvrv_60d (valuation pain) for longs, and mdia (capital velocity) for shorts.
    """
    # 1. Valuation Pain (Anchor for Longs)
    m60_bucket = map_mvrv_60d_bucket(row)
    
    # 2. Capital Velocity (Anchor for Shorts)
    mdia_strong = row.get('mdia_regime_strong_inflow', 0) == 1
    mdia_inflow = row.get('mdia_regime_inflow', 0) == 1 or mdia_strong
    mdia_aging = row.get('mdia_regime_aging', 0) == 1
    
    # 3. Macro Structure (Confirmer)
    mvrv_call = row.get('mvrv_ls_regime_call_confirm', 0) == 1
    mvrv_recovery = row.get('mvrv_ls_regime_call_confirm_recovery', 0) == 1
    mvrv_trend = row.get('mvrv_ls_regime_call_confirm_trend', 0) == 1
    mvrv_put = row.get('mvrv_ls_regime_put_confirm', 0) == 1
    mvrv_bear = row.get('mvrv_ls_regime_bear_continuation', 0) == 1
    mvrv_rollover = row.get('mvrv_ls_early_rollover', 0) == 1
    mvrv_weak_down = row.get('mvrv_ls_weak_downtrend', 0) == 1
    mvrv_distrib_warn = row.get('mvrv_ls_regime_distribution_warning', 0) == 1

    mvrv_bull = mvrv_call or mvrv_recovery or mvrv_trend
    mvrv_bearish = mvrv_put or mvrv_bear or mvrv_rollover or mvrv_weak_down or mvrv_distrib_warn
    mvrv_neutral = not mvrv_bull and not mvrv_bearish

    # === RULE 1: BEAR EXHAUSTION LONG ===
    if m60_bucket == "deep_underwater" and mdia_inflow and not mvrv_bearish:
        return MarketState.BEAR_EXHAUSTION_LONG
        
    # === RULE 2: BEAR RALLY LONG ===
    if m60_bucket in {"deep_underwater", "underwater"} and mdia_inflow and (mvrv_neutral or mvrv_bull):
        return MarketState.BEAR_RALLY_LONG
        
    # === RULE 3: BEAR CONTINUATION SHORT ===
    if m60_bucket == "profitable" and not mdia_inflow and (mdia_aging or mvrv_bearish):
        return MarketState.BEAR_CONTINUATION_SHORT
        
    # === RULE 4: LATE DISTRIBUTION SHORT ===
    if m60_bucket in {"breakeven", "profitable"} and not mdia_inflow and (mvrv_bearish or mvrv_neutral):
        return MarketState.LATE_DISTRIBUTION_SHORT
        
    # === RULE 5: TRANSITION CHOP ===
    # Underwater valuation but no inflow confirmation
    if m60_bucket in {"deep_underwater", "underwater"} and not mdia_inflow:
        return MarketState.TRANSITION_CHOP
    # Profitable valuation but MDIA inflow conflicts with short
    if m60_bucket in {"breakeven", "profitable"} and mdia_inflow:
        return MarketState.TRANSITION_CHOP
    # Bulish macro structure but recent buyers are somehow profitable in a bear
    if mvrv_bull and m60_bucket == "profitable":
        return MarketState.TRANSITION_CHOP
        
    return MarketState.NO_TRADE


def _is_bear_cycle_window(row: pd.Series) -> bool:
    """Check if the current cycle day is in the bear window (540 - 900)."""
    cycle_day = row.get('cycle_days_since_halving', None)
    if cycle_day is None or pd.isna(cycle_day):
        return False
    return 540 <= cycle_day <= 900


def fuse_signals(row: pd.Series) -> FusionResult:
    """
    Main fusion function: takes a feature row and returns unified market state
    with its statically assigned confidence and score.
    """
    is_bear_mode = _is_bear_cycle_window(row)
    
    if is_bear_mode:
        state = classify_bear_market_state(row)
    else:
        state = classify_market_state(row)
        
    confidence, score = STATE_PROPERTIES[state]
    
    # Track the basic components for UI explainability
    components = {}
    components['bear_mode'] = bool(is_bear_mode)
    _cd = row.get('cycle_days_since_halving', None)
    components['cycle_day'] = float(_cd) if _cd is not None and not pd.isna(_cd) else None
    
    components['mdia_inflow'] = int(row.get('mdia_regime_inflow', 0) == 1 or row.get('mdia_regime_strong_inflow', 0) == 1)
    components['mdia_aging'] = int(row.get('mdia_regime_aging', 0) == 1)
    
    if is_bear_mode:
        _m60 = row.get('mvrv_60d', None)
        components['mvrv_60d_raw'] = float(_m60) if _m60 is not None and not pd.isna(_m60) else None
        components['mvrv_60d_bucket'] = map_mvrv_60d_bucket(row)
    else:
        components['mdia_strong'] = int(row.get('mdia_regime_strong_inflow', 0) == 1)
        components['whale_sponsored'] = int(row.get('whale_regime_broad_accum', 0) == 1 or row.get('whale_regime_strategic_accum', 0) == 1)
        components['whale_mixed'] = int(row.get('whale_regime_mixed', 0) == 1)
        components['whale_distrib'] = int(row.get('whale_regime_distribution', 0) == 1 or row.get('whale_regime_distribution_strong', 0) == 1)
        components['whale_distrib_strong'] = int(row.get('whale_regime_distribution_strong', 0) == 1)
        
    components['mvrv_macro_bullish'] = int(row.get('mvrv_ls_regime_call_confirm', 0) == 1 or row.get('mvrv_ls_regime_call_confirm_recovery', 0) == 1 or row.get('mvrv_ls_regime_call_confirm_trend', 0) == 1)
    components['mvrv_macro_bearish'] = int(row.get('mvrv_ls_regime_put_confirm', 0) == 1 or row.get('mvrv_ls_regime_bear_continuation', 0) == 1 or row.get('mvrv_ls_early_rollover', 0) == 1 or row.get('mvrv_ls_weak_downtrend', 0) == 1 or row.get('mvrv_ls_regime_distribution_warning', 0) == 1)
    components['mvrv_macro_neutral'] = int(not components['mvrv_macro_bullish'] and not components['mvrv_macro_bearish'])
    
    short_states = [
        MarketState.BEAR_CONTINUATION, MarketState.DISTRIBUTION_RISK, MarketState.BEAR_PROBE,
        MarketState.BEAR_CONTINUATION_SHORT, MarketState.LATE_DISTRIBUTION_SHORT
    ]
    
    return FusionResult(
        state=state,
        confidence=confidence,
        score=score,
        components=components,
        short_source='rule' if state in short_states else None
    )


def build_explain_trace(row: pd.Series, fusion_result: FusionResult = None) -> list[dict]:
    """
    Build a trace of the conditional classification logic for explainability.
    Returns a list of dicts containing the rule string, match status, and boolean details.
    """
    if fusion_result is None:
        fusion_result = fuse_signals(row)
        
    c = fusion_result.components
    is_bear_mode = c.get('bear_mode', False)
    
    mdia_strong = c.get('mdia_strong', 0) == 1
    mdia_inflow = c.get('mdia_inflow', 0) == 1
    mdia_non_inflow = not mdia_inflow
    
    whale_sponsored = c.get('whale_sponsored', 0) == 1
    whale_mixed = c.get('whale_mixed', 0) == 1
    whale_distrib = c.get('whale_distrib', 0) == 1
    whale_distrib_strong = c.get('whale_distrib_strong', 0) == 1
    
    mvrv_macro_bullish = c.get('mvrv_macro_bullish', 0) == 1
    mvrv_macro_bearish = c.get('mvrv_macro_bearish', 0) == 1
    mvrv_recovery = row.get('mvrv_ls_regime_call_confirm_recovery', 0) == 1
    mvrv_macro_neutral = c.get('mvrv_macro_neutral', 0) == 1
    mvrv_put_or_bear = row.get('mvrv_ls_regime_put_confirm', 0) == 1 or row.get('mvrv_ls_regime_bear_continuation', 0) == 1
    
    if is_bear_mode:
        m60_bucket = c.get('mvrv_60d_bucket', '')
        trace = [
            {"state": MarketState.BEAR_EXHAUSTION_LONG.value, "matched": bool(m60_bucket == "deep_underwater" and mdia_inflow and not mvrv_macro_bearish), "details": f"mvrv_60d={m60_bucket}, mdia_inflow={mdia_inflow}, not_macro_bearish={not mvrv_macro_bearish}"},
            {"state": MarketState.BEAR_RALLY_LONG.value, "matched": bool(m60_bucket in {"deep_underwater", "underwater"} and mdia_inflow and (mvrv_macro_neutral or mvrv_macro_bullish)), "details": f"mvrv_60d={m60_bucket}, mdia_inflow={mdia_inflow}, neutral_or_bull_macro={(mvrv_macro_neutral or mvrv_macro_bullish)}"},
            {"state": MarketState.BEAR_CONTINUATION_SHORT.value, "matched": bool(m60_bucket == "profitable" and mdia_non_inflow and (c.get('mdia_aging', 0)==1 or mvrv_macro_bearish)), "details": f"mvrv_60d={m60_bucket}, mdia_inflow={mdia_inflow}, aging_or_bear_macro={(c.get('mdia_aging', 0)==1 or mvrv_macro_bearish)}"},
            {"state": MarketState.LATE_DISTRIBUTION_SHORT.value, "matched": bool(m60_bucket in {"breakeven", "profitable"} and mdia_non_inflow and (mvrv_macro_bearish or mvrv_macro_neutral)), "details": f"mvrv_60d={m60_bucket}, mdia_inflow={mdia_inflow}, bear_or_neutral_macro={(mvrv_macro_bearish or mvrv_macro_neutral)}"},
            {"state": MarketState.TRANSITION_CHOP.value, "matched": bool((m60_bucket in {"deep_underwater", "underwater"} and mdia_non_inflow) or (m60_bucket in {"breakeven", "profitable"} and mdia_inflow) or (mvrv_macro_bullish and m60_bucket == "profitable")), "details": f"mvrv_60d={m60_bucket}, mdia_inflow={mdia_inflow}, macro_bullish={mvrv_macro_bullish}"},
        ]
    else:
        trace = [
            {"state": MarketState.STRONG_BULLISH.value, "matched": bool(mdia_strong and whale_sponsored and mvrv_macro_bullish), "details": f"mdia_strong={mdia_strong}, whale_sponsored={whale_sponsored}, macro_bullish={mvrv_macro_bullish}"},
            {"state": MarketState.EARLY_RECOVERY.value, "matched": bool(mdia_inflow and whale_sponsored and mvrv_recovery), "details": f"mdia_inflow={mdia_inflow}, whale_sponsored={whale_sponsored}, mvrv_recovery={mvrv_recovery}"},
            {"state": MarketState.BEAR_CONTINUATION.value, "matched": bool(mdia_non_inflow and whale_distrib and mvrv_put_or_bear), "details": f"not_mdia_inflow={mdia_non_inflow}, whale_distrib={whale_distrib}, mvrv_put/bear={mvrv_put_or_bear}"},
            {"state": MarketState.BEAR_PROBE.value, "matched": bool(mdia_non_inflow and whale_distrib_strong and mvrv_macro_neutral), "details": f"not_mdia_inflow={mdia_non_inflow}, whale_distrib_strong={whale_distrib_strong}, macro_neutral={mvrv_macro_neutral}"},
            {"state": MarketState.DISTRIBUTION_RISK.value, "matched": bool(mdia_non_inflow and whale_distrib and not mvrv_macro_bullish), "details": f"not_mdia_inflow={mdia_non_inflow}, whale_distrib={whale_distrib}, not_macro_bullish={not mvrv_macro_bullish}"},
            {"state": MarketState.MOMENTUM_CONTINUATION.value, "matched": bool(mdia_inflow and (whale_sponsored or whale_mixed) and mvrv_macro_bullish), "details": f"mdia_inflow={mdia_inflow}, whale_sponsored/mixed={(whale_sponsored or whale_mixed)}, macro_bullish={mvrv_macro_bullish}"},
            {"state": MarketState.BULL_PROBE.value, "matched": bool(mdia_inflow and whale_sponsored and mvrv_macro_neutral), "details": f"mdia_inflow={mdia_inflow}, whale_sponsored={whale_sponsored}, macro_neutral={mvrv_macro_neutral}"},
        ]
    return trace


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

def add_fusion_features(feats: pd.DataFrame | dict) -> pd.DataFrame:
    """
    Add fusion features directly into feature DataFrame during build.
    Called from feature_builder.py and analyze_hit_rate.py.
    
    Accepts either a dict or DataFrame. Returns a defragmented DataFrame
    with all fusion columns added via pd.concat to avoid PerformanceWarning.
    """
    # Convert to DataFrame if dict
    if isinstance(feats, dict):
        feats = pd.DataFrame(feats)
    
    states = []
    scores = []
    confidences = []
    
    for idx in range(len(feats)):
        row = feats.iloc[idx]
        result = fuse_signals(row)
        states.append(result.state.value)
        scores.append(result.score)
        confidences.append(result.confidence.value)
    
    # Build all new columns at once to avoid fragmentation
    new_cols = {
        'fusion_market_state': states,
        'fusion_score': scores,
        'fusion_confidence': confidences,
    }
    
    # Add binary state flags for ML
    for state in MarketState:
        new_cols[f'fusion_state_{state.value}'] = [
            1 if s == state.value else 0 for s in states
        ]
    
    # Create DataFrame with same index and concat once
    new_df = pd.DataFrame(new_cols, index=feats.index)
    return pd.concat([feats, new_df], axis=1)
