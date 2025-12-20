# features/signals/overlays.py
"""
Long-Only Overlays for Signal Fusion

These overlays modify confidence/DTE/sizing for long positions only.
They do NOT create new trades - they reshape conviction on existing signals.

Two overlay types:
1. Long Edge Overlay (Bullish Boost) - when sentiment + MVRV composite favor longs
2. Long Veto Overlay (Bullish Filter) - when sentiment + MVRV composite warn against longs

Mental model:
- fusion.py decides IF we should trade
- overlays.py decides HOW HARD we should press the bet (or not at all)
"""

import pandas as pd
from dataclasses import dataclass
from typing import Optional
from .fusion import MarketState, Confidence, FusionResult


@dataclass
class OverlayResult:
    """Result of applying overlays"""
    long_edge_active: bool      # Sentiment + MVRV composite favor longs
    long_veto_active: bool      # Sentiment + MVRV composite warn against longs
    short_veto_active: bool     # Fear + undervaluation warn against shorts
    score_adjustment: int       # +2/+1/0/-1/-2
    extended_dte: bool          # Should extend DTE for mean reversion
    reduced_size: bool          # Should reduce size due to veto
    reason: str                 # Human-readable explanation


def compute_long_edge_overlay(row: pd.Series) -> tuple[bool, str]:
    """
    Detect when sentiment + MVRV composite strongly favor longs.
    
    Long Edge triggers when BOTH:
    1. Sentiment is fear/extreme fear AND stabilizing or improving
    2. MVRV composite is undervalued AND turning up
    
    Returns (is_active, reason)
    """
    # Sentiment favorable conditions
    # LOOSENED: Accept rising OR flattening (stabilizing) when in fear
    sent_call_supportive = row.get('sent_regime_call_supportive', 0) == 1
    sent_mean_reversion = row.get('sent_regime_call_mean_reversion', 0) == 1
    
    # Additional: fear bucket + stabilizing (not falling further)
    sent_in_fear = (
        row.get('sent_bucket_fear', 0) == 1 or
        row.get('sent_bucket_extreme_fear', 0) == 1
    )
    sent_not_falling = row.get('sent_is_falling', 0) == 0  # Stabilizing or rising
    sent_fear_stabilizing = sent_in_fear and sent_not_falling
    
    sent_favorable = sent_call_supportive or sent_mean_reversion or sent_fear_stabilizing
    
    # MVRV composite favorable conditions
    mvrv_undervalued = (
        row.get('mvrv_bucket_deep_undervalued', 0) == 1 or
        row.get('mvrv_bucket_undervalued', 0) == 1
    )
    mvrv_rising = row.get('mvrv_is_rising', 0) == 1
    mvrv_not_falling = row.get('mvrv_is_falling', 0) == 0  # Rising or flat
    mvrv_favorable = mvrv_undervalued and mvrv_not_falling
    
    # Also check for regime_mvrv_call_backbone (undervalued + rising)
    mvrv_call_backbone = row.get('regime_mvrv_call_backbone', 0) == 1
    
    # FULL EDGE (+2): Both sentiment AND MVRV favorable
    if sent_favorable and (mvrv_favorable or mvrv_call_backbone):
        if sent_fear_stabilizing and not sent_call_supportive:
            return True, "FULL: Sentiment fear stabilizing + MVRV undervalued"
        return True, "FULL: Sentiment fear + MVRV undervalued rising"
    
    # PARTIAL EDGE (+1): Either sentiment OR MVRV favorable (but not both)
    if sent_favorable:
        return True, "PARTIAL: Sentiment favorable only"
    
    if mvrv_favorable or mvrv_call_backbone:
        return True, "PARTIAL: MVRV undervalued only"
    
    return False, ""


def compute_long_veto_overlay(row: pd.Series) -> tuple[bool, str]:
    """
    Detect when sentiment + MVRV composite warn against longs.
    
    Long Veto triggers when EITHER:
    1. Sentiment is extreme greed persisting (euphoria)
    2. MVRV composite is overvalued AND rolling over
    
    Returns (is_active, reason)
    """
    # Sentiment against longs
    sent_avoid_longs = row.get('sent_regime_avoid_longs', 0) == 1
    sent_put_supportive = row.get('sent_regime_put_supportive', 0) == 1
    
    # MVRV composite against longs
    mvrv_overvalued = (
        row.get('mvrv_bucket_overvalued', 0) == 1 or
        row.get('mvrv_bucket_extreme_overvalued', 0) == 1
    )
    mvrv_falling = row.get('mvrv_is_falling', 0) == 1
    mvrv_flattening = row.get('mvrv_is_flattening', 0) == 1
    
    # Check for reduce longs / put supportive regimes
    mvrv_reduce_longs = row.get('regime_mvrv_reduce_longs', 0) == 1
    mvrv_put_supportive = row.get('regime_mvrv_put_supportive', 0) == 1
    
    # Strong veto: both sentiment and MVRV against
    if sent_avoid_longs and (mvrv_overvalued and (mvrv_falling or mvrv_flattening)):
        return True, "Euphoria + MVRV overvalued rolling over (STRONG VETO)"
    
    if sent_avoid_longs and mvrv_put_supportive:
        return True, "Euphoria + MVRV put supportive (STRONG VETO)"
    
    # Moderate veto: just euphoria persisting
    if sent_avoid_longs:
        return True, "Extreme greed persisting (MODERATE VETO)"
    
    # Moderate veto: just MVRV extreme overvalued + falling
    if row.get('mvrv_bucket_extreme_overvalued', 0) == 1 and mvrv_falling:
        return True, "MVRV extreme overvalued + falling (MODERATE VETO)"
    
    return False, ""


def compute_short_veto_overlay(row: pd.Series, is_confirmed_bear: bool) -> tuple[bool, str]:
    """
    Detect when sentiment/MVRV conditions make shorts risky.
    
    Short Veto Logic:
    - (sentiment negative OR mvrv low) → reduce short conviction
    - (sentiment negative AND mvrv low) → hard veto
    
    BUT: Relaxed if market is in confirmed BEAR_CONTINUATION (structure says bear is real)
    
    Returns (is_active, reason)
    """
    # Sentiment negative (fear/extreme fear)
    sent_in_fear = (
        row.get('sent_bucket_fear', 0) == 1 or
        row.get('sent_bucket_extreme_fear', 0) == 1
    )
    
    # MVRV low (undervalued)
    mvrv_undervalued = (
        row.get('mvrv_bucket_deep_undervalued', 0) == 1 or
        row.get('mvrv_bucket_undervalued', 0) == 1
    )
    
    # Both conditions = HARD VETO (major snapback risk zone)
    both_conditions = sent_in_fear and mvrv_undervalued
    
    # Either condition = SOFT VETO (reduce size)
    either_condition = sent_in_fear or mvrv_undervalued
    
    # Context guard: relax veto if confirmed bear continuation
    if is_confirmed_bear:
        # In confirmed bear, only hard veto on BOTH conditions
        if both_conditions:
            return True, "HARD: Fear + undervalued (snapback risk even in bear)"
        # Single condition in confirmed bear = just a warning, not veto
        return False, ""
    
    # Not in confirmed bear: apply full veto logic
    if both_conditions:
        return True, "HARD: Fear + undervalued (major snapback risk)"
    
    if either_condition:
        if sent_in_fear:
            return True, "SOFT: Sentiment fear (potential seller exhaustion)"
        else:
            return True, "SOFT: MVRV undervalued (shorts paying up for risk)"
    
    return False, ""


def apply_overlays(fusion_result: FusionResult, row: pd.Series) -> OverlayResult:
    """
    Apply overlays to a fusion result.
    
    For LONG states: Apply long edge boost and long veto
    For SHORT states: Apply short veto (fear + undervaluation = snapback risk)
    """
    # Define state categories
    long_states = {
        MarketState.STRONG_BULLISH,
        MarketState.EARLY_RECOVERY,
        MarketState.MOMENTUM_CONTINUATION,
    }
    short_states = {
        MarketState.DISTRIBUTION_RISK,
        MarketState.BEAR_CONTINUATION,
    }
    
    # Neutral overlay for NO_TRADE
    if fusion_result.state == MarketState.NO_TRADE:
        return OverlayResult(
            long_edge_active=False,
            long_veto_active=False,
            short_veto_active=False,
            score_adjustment=0,
            extended_dte=False,
            reduced_size=False,
            reason="NO_TRADE state - overlays not applied"
        )
    
    # === LONG STATE OVERLAYS ===
    if fusion_result.state in long_states:
        edge_active, edge_reason = compute_long_edge_overlay(row)
        veto_active, veto_reason = compute_long_veto_overlay(row)
        
        score_adj = 0
        extended_dte = False
        reduced_size = False
        reason = ""
        
        if edge_active and not veto_active:
            if "FULL:" in edge_reason:
                score_adj = +2
                extended_dte = True
            else:  # PARTIAL
                score_adj = +1
                extended_dte = False
            reason = f"LONG EDGE: {edge_reason}"
            
        elif veto_active and not edge_active:
            if "STRONG" in veto_reason:
                score_adj = -2
            else:
                score_adj = -1
            reduced_size = True
            reason = f"LONG VETO: {veto_reason}"
            
        elif edge_active and veto_active:
            score_adj = 0
            reason = f"CONFLICT: {edge_reason} vs {veto_reason}"
        else:
            reason = "No overlay active"
        
        return OverlayResult(
            long_edge_active=edge_active,
            long_veto_active=veto_active,
            short_veto_active=False,
            score_adjustment=score_adj,
            extended_dte=extended_dte,
            reduced_size=reduced_size,
            reason=reason
        )
    
    # === SHORT STATE OVERLAYS ===
    if fusion_result.state in short_states:
        is_confirmed_bear = fusion_result.state == MarketState.BEAR_CONTINUATION
        short_veto, short_reason = compute_short_veto_overlay(row, is_confirmed_bear)
        
        score_adj = 0
        reduced_size = False
        reason = ""
        
        if short_veto:
            if "HARD:" in short_reason:
                score_adj = -2  # Strong reduction
            else:  # SOFT
                score_adj = -1  # Mild reduction
            reduced_size = True
            reason = f"SHORT VETO: {short_reason}"
        else:
            reason = "No short overlay active"
        
        return OverlayResult(
            long_edge_active=False,
            long_veto_active=False,
            short_veto_active=short_veto,
            score_adjustment=score_adj,
            extended_dte=False,
            reduced_size=reduced_size,
            reason=reason
        )
    
    # Fallback (shouldn't reach here)
    return OverlayResult(
        long_edge_active=False,
        long_veto_active=False,
        short_veto_active=False,
        score_adjustment=0,
        extended_dte=False,
        reduced_size=False,
        reason="Unknown state"
    )


# Keep old name for backwards compatibility
def apply_long_overlays(fusion_result: FusionResult, row: pd.Series) -> OverlayResult:
    """Backwards compatible alias for apply_overlays"""
    return apply_overlays(fusion_result, row)


def adjust_confidence_with_overlay(base_confidence: Confidence, overlay: OverlayResult) -> Confidence:
    """
    Adjust confidence level based on overlay result.
    """
    if overlay.score_adjustment == 0:
        return base_confidence
    
    # Map confidence to numeric
    conf_map = {Confidence.LOW: 0, Confidence.MEDIUM: 1, Confidence.HIGH: 2}
    reverse_map = {0: Confidence.LOW, 1: Confidence.MEDIUM, 2: Confidence.HIGH}
    
    current = conf_map[base_confidence]
    adjusted = max(0, min(2, current + (1 if overlay.score_adjustment > 0 else -1)))
    
    return reverse_map[adjusted]


def get_dte_multiplier(overlay: OverlayResult) -> float:
    """
    Get DTE multiplier based on overlay.
    
    Returns multiplier to apply to base DTE recommendation.
    """
    if overlay.extended_dte:
        return 1.5  # Extend DTE by 50%
    elif overlay.reduced_size:
        return 0.75  # Shorten DTE by 25%
    return 1.0


def get_size_multiplier(overlay: OverlayResult) -> float:
    """
    Get position size multiplier based on overlay.
    """
    if overlay.long_edge_active and not overlay.long_veto_active:
        return 1.25  # Increase size by 25%
    elif overlay.long_veto_active:
        return 0.5   # Cut size in half
    return 1.0
