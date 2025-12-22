# features/signals/option_b.py
"""
Option B: Tactical Probes During Structural Neutrality

This is a SEPARATE, SUBORDINATE trading lane that operates ONLY when:
1. Core fusion system says NO_TRADE
2. MVRV-LS is NEUTRAL (not bearish, not permissive)
3. Local price stretch justifies a cheap, time-boxed bet

CRITICAL PRINCIPLES:
- Option B is NOT a relaxation of fusion rules
- It is parallel logic with hard ceilings
- It should LOSE OFTEN but NEVER MATTER MUCH
- If Option B ever outperforms Option A, you're using it wrong

WHAT MVRV-60D IS GOOD AT (for Option B):
- Identifying local overextension
- Capturing distribution risk near highs
- Avoiding shorting bottoms
- Providing short-dated put asymmetry

WHAT MVRV-60D IS NOT (keep out of fusion):
- Cycle phase indicator
- Macro readiness indicator
- Multi-month trend quality

EXECUTION CONSTRAINTS (non-negotiable):
- Size: max 25% of core trade
- Confidence: always LOW
- DTE: 7-21 days only
- No extensions, no pyramiding
- Mostly PUT-biased
"""

import math
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .fusion import MarketState, FusionResult
from .overlays import compute_near_peak_score


class OptionBStrategy(str, Enum):
    """Strategy types for Option B"""
    NONE = "NONE"
    PUT_SPREAD = "PUT_SPREAD"      # Default: cheap, defined risk
    PUT = "PUT"                    # Rare: high conviction local top


@dataclass
class OptionBResult:
    """
    Result of Option B evaluation.
    
    Option B trades are tactical probes, not convictions.
    They are allowed to lose often, but never allowed to matter much.
    """
    active: bool
    strategy: OptionBStrategy
    size_mult: float              # Max 0.25, relative to core trade
    dte_range: tuple              # (min_dte, max_dte), always short
    reason: str
    gate_status: dict             # Which gates passed/failed
    
    # Hard-coded confidence (always LOW, never upgraded)
    @property
    def confidence(self) -> str:
        return "LOW"
    
    @property
    def should_trade(self) -> bool:
        return self.active


# =============================================================================
# GATE FUNCTIONS (each is a go/no-go check)
# =============================================================================

def check_gate_0_structural_veto(fusion_result: FusionResult, row: pd.Series) -> tuple[bool, str]:
    """
    Gate 0: Structural Veto (ABSOLUTE)
    
    Option B is FORBIDDEN unless:
    - fusion_state == NO_TRADE
    - mvrv_ls_status == NEUTRAL
    - mvrv_ls_bearish == False
    
    This guarantees we're not trading against structure.
    """
    # Must be NO_TRADE
    if fusion_result.state != MarketState.NO_TRADE:
        return False, f"GATE_0_FAIL: fusion_state={fusion_result.state.value}, not NO_TRADE"
    
    # Derive LS status (same logic as diagnose_notrade)
    ls_permissive = (
        row.get('mvrv_ls_regime_call_confirm', 0) == 1 or
        row.get('mvrv_ls_regime_call_confirm_recovery', 0) == 1 or
        row.get('mvrv_ls_weak_uptrend', 0) == 1
    )
    ls_bearish = (
        row.get('mvrv_ls_regime_distribution_warning', 0) == 1 or
        row.get('mvrv_ls_regime_put_confirm', 0) == 1
    )
    
    if ls_permissive:
        return False, "GATE_0_FAIL: LS is PERMISSIVE (use Option A instead)"
    
    if ls_bearish:
        return False, "GATE_0_FAIL: LS is BEARISH (don't probe, wait for structure)"
    
    # LS is truly NEUTRAL - Option B allowed
    return True, "GATE_0_PASS: NO_TRADE + LS neutral"


def check_gate_1_context_sanity(row: pd.Series) -> tuple[bool, str]:
    """
    Gate 1: Context Sanity
    
    Require at least one of:
    - MDIA != DISTRIBUTION
    - Whale != STRONG_DISTRIBUTION
    
    This avoids probing during crisis moments.
    """
    mdia_distrib = row.get('mdia_regime_distribution', 0) == 1
    whale_strong_distrib = row.get('whale_regime_distribution_strong', 0) == 1
    
    # If BOTH are distributing heavily, it's a crisis - stay out
    if mdia_distrib and whale_strong_distrib:
        return False, "GATE_1_FAIL: Both MDIA and Whale strongly distributing (crisis)"
    
    return True, "GATE_1_PASS: No crisis conditions"


def check_gate_2_tactical_trigger(row: pd.Series, threshold: float = 0.75) -> tuple[bool, float, str]:
    """
    Gate 2: Tactical Trigger (MVRV-60d near-peak)
    
    This is where mvrv_60d lives - detecting local stretch.
    
    Returns (passed, near_peak_score, reason)
    """
    score = compute_near_peak_score(row)
    
    if score is None:
        return False, 0.0, "GATE_2_FAIL: Missing MVRV-60d data"
    
    if score >= 0.85:
        return True, score, f"GATE_2_PASS: Very near local peak (score={score:.2f})"
    
    if score >= threshold:
        return True, score, f"GATE_2_PASS: Elevated local stretch (score={score:.2f})"
    
    return False, score, f"GATE_2_FAIL: Not stretched (score={score:.2f})"


def check_streak_requirement(notrade_streak_days: int, min_streak: int = 14) -> tuple[bool, str]:
    """
    Optional: Streak-based activation
    
    Only allow Option B after prolonged NO_TRADE periods.
    This reduces churn and improves signal-to-noise.
    
    Recommended: min_streak >= 14 days
    """
    if notrade_streak_days >= min_streak:
        return True, f"STREAK_PASS: {notrade_streak_days} days >= {min_streak} minimum"
    
    return False, f"STREAK_FAIL: {notrade_streak_days} days < {min_streak} minimum"


# =============================================================================
# MAIN OPTION B EVALUATION
# =============================================================================

def evaluate_option_b(
    fusion_result: FusionResult,
    row: pd.Series,
    *,
    notrade_streak_days: int = 0,
    require_streak: bool = True,
    min_streak_days: int = 14,
    near_peak_threshold: float = 0.75,
) -> OptionBResult:
    """
    Evaluate Option B: Tactical probes during structural neutrality.
    
    This function is the ONLY entry point for Option B decisions.
    It enforces all gates and constraints.
    
    Args:
        fusion_result: Result from core fusion (must be NO_TRADE)
        row: Feature row with MVRV-60d and other indicators
        notrade_streak_days: How many consecutive NO_TRADE days (for streak mode)
        require_streak: If True, only allow after min_streak_days of NO_TRADE
        min_streak_days: Minimum NO_TRADE streak before Option B activates
        near_peak_threshold: Minimum near_peak_score for tactical trigger
    
    Returns:
        OptionBResult with trade parameters or rejection reason
    """
    gate_status = {}
    
    # === GATE 0: Structural Veto (absolute) ===
    g0_pass, g0_reason = check_gate_0_structural_veto(fusion_result, row)
    gate_status['gate_0_structural'] = g0_pass
    
    if not g0_pass:
        return OptionBResult(
            active=False,
            strategy=OptionBStrategy.NONE,
            size_mult=0.0,
            dte_range=(0, 0),
            reason=g0_reason,
            gate_status=gate_status,
        )
    
    # === STREAK CHECK (optional but recommended) ===
    if require_streak:
        streak_pass, streak_reason = check_streak_requirement(notrade_streak_days, min_streak_days)
        gate_status['streak_check'] = streak_pass
        
        if not streak_pass:
            return OptionBResult(
                active=False,
                strategy=OptionBStrategy.NONE,
                size_mult=0.0,
                dte_range=(0, 0),
                reason=streak_reason,
                gate_status=gate_status,
            )
    else:
        gate_status['streak_check'] = 'SKIPPED'
    
    # === GATE 1: Context Sanity ===
    g1_pass, g1_reason = check_gate_1_context_sanity(row)
    gate_status['gate_1_context'] = g1_pass
    
    if not g1_pass:
        return OptionBResult(
            active=False,
            strategy=OptionBStrategy.NONE,
            size_mult=0.0,
            dte_range=(0, 0),
            reason=g1_reason,
            gate_status=gate_status,
        )
    
    # === GATE 2: Tactical Trigger ===
    g2_pass, near_peak_score, g2_reason = check_gate_2_tactical_trigger(row, near_peak_threshold)
    gate_status['gate_2_tactical'] = g2_pass
    gate_status['near_peak_score'] = near_peak_score
    
    if not g2_pass:
        return OptionBResult(
            active=False,
            strategy=OptionBStrategy.NONE,
            size_mult=0.0,
            dte_range=(0, 0),
            reason=g2_reason,
            gate_status=gate_status,
        )
    
    # === ALL GATES PASSED - OPTION B TRADE ALLOWED ===
    
    # Determine size based on conviction (capped at 0.25)
    if near_peak_score >= 0.85:
        size_mult = 0.25  # Maximum for very stretched
    elif near_peak_score >= 0.80:
        size_mult = 0.20
    else:
        size_mult = 0.15  # Minimum for threshold-level stretch
    
    return OptionBResult(
        active=True,
        strategy=OptionBStrategy.PUT_SPREAD,
        size_mult=size_mult,
        dte_range=(7, 21),  # Always short-dated
        reason=f"OPTION_B_ACTIVE: {g2_reason}",
        gate_status=gate_status,
    )


# =============================================================================
# HELPER FOR STREAK CALCULATION
# =============================================================================

def calculate_notrade_streak(fusion_states: list, current_idx: int) -> int:
    """
    Calculate how many consecutive NO_TRADE days up to current_idx.
    
    Args:
        fusion_states: List of fusion state values (as strings)
        current_idx: Current position in list
    
    Returns:
        Number of consecutive NO_TRADE days ending at current_idx
    """
    if current_idx < 0 or current_idx >= len(fusion_states):
        return 0
    
    if fusion_states[current_idx] != 'no_trade':
        return 0
    
    streak = 0
    for i in range(current_idx, -1, -1):
        if fusion_states[i] == 'no_trade':
            streak += 1
        else:
            break
    
    return streak
