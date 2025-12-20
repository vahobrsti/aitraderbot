# features/signals/overlays.py
"""
Signal Overlays for Long and Short Positions

These overlays modify confidence/DTE/sizing for both long and short positions.
They do NOT create new trades - they reshape conviction on existing signals.

Long overlays:
1. Long Edge Overlay (Bullish Boost) - when sentiment + MVRV composite favor longs
2. Long Veto Overlay (Bullish Filter) - when sentiment + MVRV composite warn against longs

Short overlays (MVRV-60d ONLY):
1. Short Edge Overlay - when MVRV-60d is near peak (shorts more reliable)
2. Short Veto Overlay - when MVRV-60d is far from peak (shorts less reliable)

Mental model:
- fusion.py decides IF we should trade
- overlays.py decides HOW HARD we should press the bet (or not at all)
"""

import pandas as pd
import math
from dataclasses import dataclass
from typing import Optional
from .fusion import MarketState, Confidence, FusionResult


@dataclass
class OverlayResult:
    """Result of applying overlays"""
    # Strength levels: 0 = none, 1 = partial/soft, 2 = full/hard
    edge_strength: int          # 0=none, 1=partial edge, 2=full edge (for longs)
    long_veto_strength: int     # 0=none, 1=moderate, 2=strong veto
    short_veto_strength: int    # 0=none, 1=soft, 2=hard veto
    short_edge_strength: int    # 0=none, 1=partial, 2=full edge (for shorts)
    score_adjustment: int       # +2/+1/0/-1/-2
    extended_dte: bool          # Should extend DTE for mean reversion
    reduced_size: bool          # Should reduce size due to veto
    reason: str                 # Human-readable explanation
    
    # Convenience properties for backwards compatibility
    @property
    def long_edge_active(self) -> bool:
        return self.edge_strength > 0
    
    @property
    def long_veto_active(self) -> bool:
        return self.long_veto_strength > 0
    
    @property
    def short_veto_active(self) -> bool:
        return self.short_veto_strength > 0
    
    @property
    def short_edge_active(self) -> bool:
        return self.short_edge_strength > 0


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


# ============================================================================
# SHORT OVERLAYS - MVRV-60d ONLY (no sentiment, no composite MVRV)
# ============================================================================

def compute_short_edge_overlay(row: pd.Series) -> tuple[int, str]:
    """
    Shorts are most reliable when MVRV-60d is stretched near its recent peak.
    
    Uses ONLY mvrv_60d_pct_rank and mvrv_60d_dist_from_max.
    No sentiment. No composite MVRV.
    
    Returns (edge_strength, reason), edge_strength ∈ {0,1,2}
    """
    pct = row.get("mvrv_60d_pct_rank", None)
    dist = row.get("mvrv_60d_dist_from_max", None)
    
    # If missing or NaN, no edge
    if pct is None or dist is None:
        return 0, ""
    if isinstance(pct, float) and math.isnan(pct):
        return 0, ""
    if isinstance(dist, float) and math.isnan(dist):
        return 0, ""
    
    # FULL edge (2): very near peak
    if pct >= 0.90 or dist <= 0.10:
        return 2, f"FULL: MVRV-60d near peak (pct={pct:.2f}, dist={dist:.2f})"
    
    # PARTIAL edge (1): elevated
    if pct >= 0.80 or dist <= 0.20:
        return 1, f"PARTIAL: MVRV-60d elevated (pct={pct:.2f}, dist={dist:.2f})"
    
    return 0, ""


def compute_short_veto_overlay(row: pd.Series, is_confirmed_bear: bool) -> tuple[bool, str]:
    """
    Detect when MVRV-60d conditions make shorts risky.
    
    Uses ONLY mvrv_60d_pct_rank and mvrv_60d_dist_from_max.
    No sentiment. No composite MVRV.
    
    Short Veto Logic:
    - HARD far-from-peak if (pct < 0.25) OR (dist > 0.75)
    - SOFT far-from-peak if (pct < 0.40) OR (dist > 0.60)
    
    Context guard: In confirmed BEAR_CONTINUATION, only apply HARD veto.
    
    Returns (is_active, reason)
    """
    pct = row.get("mvrv_60d_pct_rank", None)
    dist = row.get("mvrv_60d_dist_from_max", None)
    
    # If missing or NaN, do NOT veto (fail open)
    if pct is None or dist is None:
        return False, ""
    if isinstance(pct, float) and math.isnan(pct):
        return False, ""
    if isinstance(dist, float) and math.isnan(dist):
        return False, ""
    
    # Define far-from-peak thresholds
    hard_far_from_peak = (pct < 0.25) or (dist > 0.75)
    soft_far_from_peak = (pct < 0.40) or (dist > 0.60)
    
    # Context guard: in confirmed bear, only hard veto applies
    if is_confirmed_bear:
        if hard_far_from_peak:
            return True, f"HARD: MVRV-60d far from peak even in bear (pct={pct:.2f}, dist={dist:.2f})"
        return False, ""
    
    # Not in confirmed bear: apply both hard and soft veto
    if hard_far_from_peak:
        return True, f"HARD: MVRV-60d far from peak (pct={pct:.2f}, dist={dist:.2f})"
    
    if soft_far_from_peak:
        return True, f"SOFT: MVRV-60d not near peak (pct={pct:.2f}, dist={dist:.2f})"
    
    return False, ""


def apply_overlays(fusion_result: FusionResult, row: pd.Series) -> OverlayResult:
    """
    Apply overlays to a fusion result.
    
    For LONG states: Apply long edge boost and long veto (sentiment + composite MVRV)
    For SHORT states: Apply short edge/veto (MVRV-60d ONLY)
    
    Veto dominance rules:
    - STRONG/HARD veto always wins (overrides any edge)
    - Moderate/soft veto beats partial edge
    - Full edge can override moderate veto
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
            edge_strength=0,
            long_veto_strength=0,
            short_veto_strength=0,
            short_edge_strength=0,
            score_adjustment=0,
            extended_dte=False,
            reduced_size=False,
            reason="NO_TRADE state - overlays not applied"
        )
    
    # === LONG STATE OVERLAYS ===
    if fusion_result.state in long_states:
        edge_active, edge_reason = compute_long_edge_overlay(row)
        veto_active, veto_reason = compute_long_veto_overlay(row)
        
        # Determine strength levels
        edge_str = 0
        veto_str = 0
        
        if edge_active:
            edge_str = 2 if "FULL:" in edge_reason else 1
        if veto_active:
            veto_str = 2 if "STRONG" in veto_reason else 1
        
        # Apply veto dominance rules
        score_adj = 0
        extended_dte = False
        reduced_size = False
        reason = ""
        
        if veto_str == 2:
            # STRONG veto always wins
            score_adj = -2
            reduced_size = True
            reason = f"LONG VETO (STRONG wins): {veto_reason}"
        elif veto_str == 1 and edge_str < 2:
            # Moderate veto beats partial edge
            score_adj = -1
            reduced_size = True
            reason = f"LONG VETO (moderate wins over partial): {veto_reason}"
        elif edge_str == 2:
            # Full edge (can override moderate veto)
            score_adj = +2
            extended_dte = True
            reason = f"LONG EDGE (FULL): {edge_reason}"
        elif edge_str == 1 and veto_str == 0:
            # Partial edge, no veto
            score_adj = +1
            reason = f"LONG EDGE (PARTIAL): {edge_reason}"
        else:
            reason = "No overlay active"
        
        return OverlayResult(
            edge_strength=edge_str,
            long_veto_strength=veto_str,
            short_veto_strength=0,
            short_edge_strength=0,
            score_adjustment=score_adj,
            extended_dte=extended_dte,
            reduced_size=reduced_size,
            reason=reason
        )
    
    # === SHORT STATE OVERLAYS (MVRV-60d ONLY) ===
    if fusion_result.state in short_states:
        is_confirmed_bear = fusion_result.state == MarketState.BEAR_CONTINUATION
        
        # Compute both veto and edge
        veto_active, veto_reason = compute_short_veto_overlay(row, is_confirmed_bear)
        short_edge_str, short_edge_reason = compute_short_edge_overlay(row)
        
        # Determine veto strength
        veto_str = 0
        if veto_active:
            veto_str = 2 if "HARD:" in veto_reason else 1
        
        # Apply SHORT dominance rules
        score_adj = 0
        reduced_size = False
        reason = ""
        
        if veto_str == 2:
            # HARD veto always wins, edge ignored
            score_adj = -2
            reduced_size = True
            reason = f"SHORT VETO (HARD): {veto_reason}"
        elif veto_str == 1 and short_edge_str == 0:
            # Soft veto, no edge
            score_adj = -1
            reduced_size = True
            reason = f"SHORT VETO (SOFT): {veto_reason}"
        elif veto_str == 1 and short_edge_str > 0:
            # Soft veto + edge: mixed scenario
            reduced_size = True  # Risk control: keep reduced size
            if short_edge_str == 2:
                score_adj = 0  # Full edge cancels soft veto score
            else:
                score_adj = -1  # Partial edge: still cautious
            reason = f"SHORT MIXED: {veto_reason} + {short_edge_reason}"
        elif veto_str == 0 and short_edge_str == 2:
            # No veto, full edge
            score_adj = +2
            reduced_size = False
            reason = f"SHORT EDGE (FULL): {short_edge_reason}"
        elif veto_str == 0 and short_edge_str == 1:
            # No veto, partial edge
            score_adj = +1
            reduced_size = False
            reason = f"SHORT EDGE (PARTIAL): {short_edge_reason}"
        else:
            reason = "No short overlay active"
        
        return OverlayResult(
            edge_strength=0,
            long_veto_strength=0,
            short_veto_strength=veto_str,
            short_edge_strength=short_edge_str,
            score_adjustment=score_adj,
            extended_dte=False,
            reduced_size=reduced_size,
            reason=reason
        )
    
    # Fallback (shouldn't reach here)
    return OverlayResult(
        edge_strength=0,
        long_veto_strength=0,
        short_veto_strength=0,
        short_edge_strength=0,
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
    Note: Shorts do not extend DTE based on short edge.
    """
    if overlay.extended_dte:
        return 1.5  # Extend DTE by 50%
    elif overlay.reduced_size:
        return 0.75  # Shorten DTE by 25%
    return 1.0


def get_size_multiplier(overlay: OverlayResult) -> float:
    """
    Get position size multiplier based on overlay.
    
    Uses reduced_size flag which works for both long and short veto.
    Uses edge_strength (long) and short_edge_strength for boosts.
    """
    # 1) Veto sizing dominates (reduced_size = True)
    if overlay.reduced_size:
        max_veto = max(overlay.long_veto_strength, overlay.short_veto_strength)
        if max_veto == 2:
            return 0.0   # Hard/strong veto: no trade
        else:
            return 0.5   # Soft/moderate veto: cut size in half
    
    # 2) Long edge boost
    if overlay.edge_strength == 2:
        return 1.25  # Full long edge: increase size by 25%
    elif overlay.edge_strength == 1:
        return 1.1   # Partial long edge: modest increase
    
    # 3) Short edge boost
    if overlay.short_edge_strength == 2:
        return 1.15  # Full short edge: increase size by 15%
    elif overlay.short_edge_strength == 1:
        return 1.05  # Partial short edge: slight increase
    
    return 1.0
