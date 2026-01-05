# features/signals/overlays.py
"""
Signal Overlays for Long and Short Positions

These overlays modify confidence/DTE/sizing for both long and short positions.
They do NOT create new trades - they reshape conviction on existing signals.

Long overlays:
1. Long Edge Overlay (Bullish Boost) - when sentiment + MVRV composite favor longs
2. Long Veto Overlay (Bullish Filter) - when sentiment + MVRV composite warn against longs

Short overlays (MVRV-60d ONLY):
Uses a blended "near_peak_score" from mvrv_60d_pct_rank and mvrv_60d_dist_from_max.
1. Short Edge Overlay - when near_peak_score is high (shorts more reliable)
2. Short Veto Overlay - when near_peak_score is low (shorts less reliable)

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
    score_adjustment: int       # +2/+1/0/-1/-2 (for logging/interpretability)
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
# Uses a blended "near_peak_score" for smoother behavior
# ============================================================================

def compute_near_peak_score(row: pd.Series) -> Optional[float]:
    """
    Compute a blended near-peak score from MVRV-60d features.
    
    Blends mvrv_60d_pct_rank and mvrv_60d_dist_from_max into a single 0-1 score.
    Higher score = nearer to 60-day peak = shorts more reliable.
    
    Returns None if data is missing/NaN.
    """
    pct = row.get("mvrv_60d_pct_rank", None)
    dist = row.get("mvrv_60d_dist_from_max", None)
    
    # Check for missing or NaN
    if pct is None or dist is None:
        return None
    if isinstance(pct, float) and math.isnan(pct):
        return None
    if isinstance(dist, float) and math.isnan(dist):
        return None
    
    # Blend: 50% pct_rank (higher is better) + 50% (1 - dist) (lower dist is better)
    near_peak_score = 0.5 * pct + 0.5 * (1 - dist)
    
    return near_peak_score


def compute_short_overlay(row: pd.Series, is_confirmed_bear: bool) -> tuple[int, int, str]:
    """
    Compute short overlay using blended near_peak_score.
    
    Edge and veto are MUTUALLY EXCLUSIVE (no mixed states).
    
    Thresholds:
    - >= 0.85: Full edge (shorts very reliable)
    - >= 0.75: Partial edge (shorts reliable)
    - <= 0.35: Soft veto (shorts less reliable) - only for DISTRIBUTION_RISK
    - <= 0.25: Hard veto (shorts unreliable)
    
    ABSOLUTE LEVEL CHECK (mvrv_60d):
    - mvrv_60d < 1.0: HARD VETO - short-term holders underwater, shorting is dangerous
    
    For DISTRIBUTION_RISK: fail-closed (require score >= 0.35 to trade)
    For BEAR_CONTINUATION: fail-open (only hard veto, structure confirms bear)
    
    Returns (edge_strength, veto_strength, reason)
    """
    # === ABSOLUTE LEVEL CHECK (Priority) ===
    # When MVRV-60d < 1.0, short-term holders are at a loss - shorting is dangerous
    mvrv_60d_raw = row.get("mvrv_60d", None)
    if mvrv_60d_raw is not None and not pd.isna(mvrv_60d_raw):
        if mvrv_60d_raw < 0.95:
            return 0, 2, f"HARD: MVRV-60d < 0.95 (holders underwater: {mvrv_60d_raw:.2f})"
    
    # === RELATIVE NEAR-PEAK SCORE (Secondary) ===
    score = compute_near_peak_score(row)
    
    # Handle missing data
    if score is None:
        if is_confirmed_bear:
            # BEAR_CONTINUATION: fail-open (structure confirms bear)
            return 0, 0, "Missing MVRV-60d data (fail-open in confirmed bear)"
        else:
            # DISTRIBUTION_RISK: fail-closed (conservative)
            return 0, 2, "HARD: Missing MVRV-60d data (fail-closed)"
    
    # Edge/veto logic (mutually exclusive)
    if score >= 0.85:
        return 2, 0, f"FULL EDGE: MVRV-60d near peak (score={score:.2f})"
    
    if score >= 0.75:
        return 1, 0, f"PARTIAL EDGE: MVRV-60d elevated (score={score:.2f})"
    
    # Below 0.75: check for vetoes
    if score <= 0.25:
        # Hard veto applies to both states
        if is_confirmed_bear:
            return 0, 2, f"HARD: MVRV-60d far from peak even in bear (score={score:.2f})"
        else:
            return 0, 2, f"HARD: MVRV-60d far from peak (score={score:.2f})"
    
    if score <= 0.35:
        # Soft veto only for DISTRIBUTION_RISK
        if not is_confirmed_bear:
            return 0, 1, f"SOFT: MVRV-60d not near peak (score={score:.2f})"
    
    # No edge, no veto (0.35 < score < 0.75)
    return 0, 0, f"No short overlay active (score={score:.2f})"


def apply_overlays(fusion_result: FusionResult, row: pd.Series) -> OverlayResult:
    """
    Apply overlays to a fusion result.
    
    For LONG states: Apply long edge boost and long veto (sentiment + composite MVRV)
    For SHORT states: Apply short edge/veto (MVRV-60d ONLY, blended score)
    
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
        MarketState.BULL_PROBE,  # Trade smaller, but still a long state
    }
    short_states = {
        MarketState.DISTRIBUTION_RISK,
        MarketState.BEAR_CONTINUATION,
        MarketState.BEAR_PROBE,  # Trade smaller, but still a short state
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
    
    # === SHORT STATE OVERLAYS (MVRV-60d ONLY, blended score) ===
    if fusion_result.state in short_states:
        is_confirmed_bear = fusion_result.state == MarketState.BEAR_CONTINUATION
        
        # Compute overlay using blended score (edge/veto mutually exclusive)
        short_edge_str, veto_str, reason = compute_short_overlay(row, is_confirmed_bear)
        
        # Determine score_adjustment and flags
        score_adj = 0
        reduced_size = False
        extended_dte = False
        
        if veto_str == 2:
            # Hard veto
            score_adj = -2
            reduced_size = True
        elif veto_str == 1:
            # Soft veto
            score_adj = -1
            reduced_size = True
        elif short_edge_str == 2:
            # Full edge
            score_adj = +2
            extended_dte = True  # Let mean reversion play out
        elif short_edge_str == 1:
            # Partial edge
            score_adj = +1
        
        return OverlayResult(
            edge_strength=0,
            long_veto_strength=0,
            short_veto_strength=veto_str,
            short_edge_strength=short_edge_str,
            score_adjustment=score_adj,
            extended_dte=extended_dte,
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
    
    Uses edge/veto strengths directly (not just score_adjustment).
    """
    conf_map = {Confidence.LOW: 0, Confidence.MEDIUM: 1, Confidence.HIGH: 2}
    reverse_map = {0: Confidence.LOW, 1: Confidence.MEDIUM, 2: Confidence.HIGH}
    
    current = conf_map[base_confidence]
    
    # Hard/strong veto -> force LOW
    if overlay.long_veto_strength == 2 or overlay.short_veto_strength == 2:
        return Confidence.LOW
    
    # Soft/moderate veto -> down 1 notch
    if overlay.long_veto_strength == 1 or overlay.short_veto_strength == 1:
        adjusted = max(0, current - 1)
        return reverse_map[adjusted]
    
    # Full edge -> up 1 notch
    if overlay.edge_strength == 2 or overlay.short_edge_strength == 2:
        adjusted = min(2, current + 1)
        return reverse_map[adjusted]
    
    # Partial edge -> up 1 notch only if not already LOW
    if overlay.edge_strength == 1 or overlay.short_edge_strength == 1:
        if current > 0:  # Avoid overconfidence from LOW
            adjusted = min(2, current + 1)
            return reverse_map[adjusted]
    
    return base_confidence


def get_dte_multiplier(overlay: OverlayResult) -> float:
    """
    Get DTE multiplier based on overlay.
    
    For longs: extend DTE on full edge (mean reversion)
    For shorts: extend DTE on full edge (let mean reversion play out)
    For vetoes: shorten DTE (reduce exposure)
    """
    # Full edge (long or short) extends DTE
    if overlay.extended_dte:
        if overlay.short_edge_strength == 2:
            return 1.15  # Short full edge: slight extension
        else:
            return 1.5   # Long full edge: extend by 50%
    
    # Partial short edge: slight extension
    if overlay.short_edge_strength == 1:
        return 1.05
    
    # Veto shortens DTE
    if overlay.reduced_size:
        max_veto = max(overlay.long_veto_strength, overlay.short_veto_strength)
        if max_veto == 1:
            return 0.85  # Soft veto: shorten by 15%
        else:
            return 0.75  # Hard veto: shorten by 25%
    
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
