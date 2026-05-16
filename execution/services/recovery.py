"""
Recovery Decision Engine for Trade Execution.

This service provides recovery decision logic for adverse trades based on the
recovery policy parameters. It integrates with the existing trade execution
pipeline to provide clear decision logic for when to flip, cut, or hold trades.

The service uses policy-derived parameters for:
- Checkpoint timing (TTH p75 values)
- Adverse thresholds (MAE p75 / 2)
- Recovery thresholds (flip/cut/target values)

Usage:
    from execution.services.recovery import RecoveryDecisionEngine
    from execution.services.policy import get_policy
    
    engine = RecoveryDecisionEngine(get_policy())
    
    # Check if trade should be flipped
    should_flip = engine.should_flip("CALL", entry_price, current_price, days_held)
    
    # Check if trade should be cut
    should_cut = engine.should_cut("CALL", entry_price, current_price, days_held)
    
    # Get recovery target for flipped trade
    target = engine.get_recovery_target("CALL")
"""

from dataclasses import dataclass
from typing import Optional, Tuple
from decimal import Decimal
import logging

from execution.services.policy import PolicyVersion, get_policy

logger = logging.getLogger(__name__)


@dataclass
class RecoveryDecision:
    """
    Recovery decision result.
    
    Contains the decision (FLIP, CUT, HOLD) along with supporting metrics
    and rationale for the decision.
    """
    action: str  # "FLIP", "CUT", or "HOLD"
    confidence: float  # 0.0 to 1.0 confidence in the decision
    checkpoint_return: float  # Return at checkpoint (negative = adverse)
    recovery_potential: float  # Estimated recovery potential (0.0 to 1.0)
    rationale: str  # Human-readable explanation of the decision
    
    @property
    def should_flip(self) -> bool:
        """True if the decision is to flip the trade direction."""
        return self.action == "FLIP"
    
    @property
    def should_cut(self) -> bool:
        """True if the decision is to cut losses."""
        return self.action == "CUT"
    
    @property
    def should_hold(self) -> bool:
        """True if the decision is to hold the original position."""
        return self.action == "HOLD"


class RecoveryDecisionEngine:
    """
    Recovery Decision Engine for trade execution.
    
    Provides recovery decision logic for adverse trades based on policy parameters.
    Integrates with the existing trade execution pipeline to determine when to
    flip, cut, or hold trades that are performing poorly.
    
    The engine uses three key thresholds from the policy:
    1. Adverse threshold: When to consider a trade for recovery (MAE p75 / 2)
    2. Flip threshold: Recovery potential above which to flip direction
    3. Cut threshold: Recovery potential below which to cut losses
    """
    
    def __init__(self, policy: Optional[PolicyVersion] = None):
        """
        Initialize the recovery decision engine.
        
        Args:
            policy: Policy version to use. If None, uses current active policy.
        """
        self.policy = policy or get_policy()
        
        # Cache commonly used values
        self._checkpoint_cache = {}
        self._threshold_cache = {}
    
    def should_flip(
        self,
        signal_type: str,
        entry_price: float,
        current_price: float,
        days_held: int,
        recovery_mfe: Optional[float] = None,
    ) -> bool:
        """
        Determine if a trade should be flipped to the opposite direction.
        
        A trade should be flipped when:
        1. It's adverse at the checkpoint day (price moved against original direction)
        2. The estimated recovery potential exceeds the flip threshold
        
        Args:
            signal_type: Signal type (e.g., "CALL", "PUT", "MVRV_SHORT")
            entry_price: Original entry price
            current_price: Current market price
            days_held: Number of days the trade has been held
            recovery_mfe: Optional pre-calculated recovery MFE
            
        Returns:
            True if the trade should be flipped
        """
        decision = self.get_recovery_decision(
            signal_type, entry_price, current_price, days_held, recovery_mfe
        )
        return decision.should_flip
    
    def should_cut(
        self,
        signal_type: str,
        entry_price: float,
        current_price: float,
        days_held: int,
        recovery_mfe: Optional[float] = None,
    ) -> bool:
        """
        Determine if a trade should be cut (close at loss).
        
        A trade should be cut when:
        1. It's adverse at the checkpoint day
        2. The estimated recovery potential is below the cut threshold
        
        Args:
            signal_type: Signal type (e.g., "CALL", "PUT", "MVRV_SHORT")
            entry_price: Original entry price
            current_price: Current market price
            days_held: Number of days the trade has been held
            recovery_mfe: Optional pre-calculated recovery MFE
            
        Returns:
            True if the trade should be cut
        """
        decision = self.get_recovery_decision(
            signal_type, entry_price, current_price, days_held, recovery_mfe
        )
        return decision.should_cut
    
    def get_recovery_target(self, signal_type: str) -> float:
        """
        Get the recovery target for a signal type.
        
        The recovery target is typically lower than the original target
        since recovery trades have shorter time horizons.
        
        Args:
            signal_type: Signal type (e.g., "CALL", "PUT", "MVRV_SHORT")
            
        Returns:
            Recovery target as decimal (e.g., 0.03 for 3%)
        """
        return self.policy.get_recovery_target(signal_type)
    
    def get_recovery_decision(
        self,
        signal_type: str,
        entry_price: float,
        current_price: float,
        days_held: int,
        recovery_mfe: Optional[float] = None,
    ) -> RecoveryDecision:
        """
        Get comprehensive recovery decision for a trade.
        
        Analyzes the trade's current state and returns a decision with
        supporting metrics and rationale.
        
        Args:
            signal_type: Signal type (e.g., "CALL", "PUT", "MVRV_SHORT")
            entry_price: Original entry price
            current_price: Current market price
            days_held: Number of days the trade has been held
            recovery_mfe: Optional pre-calculated recovery MFE
            
        Returns:
            RecoveryDecision with action, confidence, and rationale
        """
        try:
            # Get checkpoint day from policy
            checkpoint_day = self._get_checkpoint_day(signal_type)
            
            # Check if we're at or past checkpoint
            if days_held < checkpoint_day:
                return RecoveryDecision(
                    action="HOLD",
                    confidence=1.0,
                    checkpoint_return=0.0,
                    recovery_potential=0.0,
                    rationale=f"Trade has not reached checkpoint day {checkpoint_day} (currently day {days_held})"
                )
            
            # Calculate current return
            direction = self._get_direction(signal_type)
            if direction == "LONG":
                current_return = current_price / entry_price - 1.0
            else:  # SHORT
                current_return = 1.0 - current_price / entry_price
            
            # Get adverse threshold
            adverse_threshold = self._get_adverse_threshold(signal_type)
            
            # Check if trade is adverse
            is_adverse = current_return < -adverse_threshold
            
            if not is_adverse:
                return RecoveryDecision(
                    action="HOLD",
                    confidence=0.8,
                    checkpoint_return=current_return,
                    recovery_potential=0.5,  # Neutral since not adverse
                    rationale=f"Trade is not adverse (return: {current_return:.2%}, threshold: {-adverse_threshold:.2%})"
                )
            
            # Trade is adverse - evaluate recovery potential
            if recovery_mfe is None:
                # Estimate recovery potential based on historical patterns
                recovery_potential = self._estimate_recovery_potential(
                    signal_type, current_return, days_held
                )
            else:
                recovery_potential = recovery_mfe
            
            # Get recovery thresholds
            flip_threshold = self.policy.get_recovery_flip_threshold(signal_type)
            cut_threshold = self.policy.get_recovery_cut_threshold(signal_type)
            
            # Make decision based on recovery potential
            if recovery_potential > flip_threshold:
                action = "FLIP"
                confidence = min(0.9, 0.5 + (recovery_potential - flip_threshold) * 5)
                rationale = (
                    f"High recovery potential ({recovery_potential:.2%}) exceeds flip threshold "
                    f"({flip_threshold:.2%}). Recommend flipping direction."
                )
            elif recovery_potential < cut_threshold:
                action = "CUT"
                confidence = min(0.9, 0.5 + (cut_threshold - recovery_potential) * 5)
                rationale = (
                    f"Low recovery potential ({recovery_potential:.2%}) below cut threshold "
                    f"({cut_threshold:.2%}). Recommend cutting losses."
                )
            else:
                action = "HOLD"
                confidence = 0.6  # Moderate confidence in neutral zone
                rationale = (
                    f"Moderate recovery potential ({recovery_potential:.2%}) between thresholds "
                    f"({cut_threshold:.2%} - {flip_threshold:.2%}). Recommend holding."
                )
            
            return RecoveryDecision(
                action=action,
                confidence=confidence,
                checkpoint_return=current_return,
                recovery_potential=recovery_potential,
                rationale=rationale
            )
            
        except Exception as e:
            logger.error(f"Error in recovery decision for {signal_type}: {e}")
            return RecoveryDecision(
                action="HOLD",
                confidence=0.1,
                checkpoint_return=0.0,
                recovery_potential=0.0,
                rationale=f"Error in analysis: {str(e)}. Defaulting to HOLD."
            )
    
    def get_checkpoint_day(self, signal_type: str) -> int:
        """
        Get the checkpoint day for a signal type.
        
        The checkpoint day is when we evaluate if a trade is adverse and
        consider recovery actions. Based on TTH p75 values from policy.
        
        Args:
            signal_type: Signal type (e.g., "CALL", "PUT", "MVRV_SHORT")
            
        Returns:
            Checkpoint day (1-indexed from trade entry)
        """
        return self._get_checkpoint_day(signal_type)
    
    def get_adverse_threshold(self, signal_type: str) -> float:
        """
        Get the adverse threshold for a signal type.
        
        The adverse threshold determines when a trade is considered
        "adverse" and eligible for recovery analysis.
        
        Args:
            signal_type: Signal type (e.g., "CALL", "PUT", "MVRV_SHORT")
            
        Returns:
            Adverse threshold as decimal (e.g., 0.0235 for 2.35%)
        """
        return self._get_adverse_threshold(signal_type)
    
    def is_recovery_candidate(
        self,
        signal_type: str,
        entry_price: float,
        current_price: float,
        days_held: int,
    ) -> bool:
        """
        Check if a trade qualifies as a recovery candidate.
        
        A trade qualifies when:
        1. It has reached the checkpoint day
        2. The price has moved adversely beyond the threshold
        
        Args:
            signal_type: Signal type (e.g., "CALL", "PUT", "MVRV_SHORT")
            entry_price: Original entry price
            current_price: Current market price
            days_held: Number of days the trade has been held
            
        Returns:
            True if the trade is a recovery candidate
        """
        # Check if at checkpoint
        checkpoint_day = self._get_checkpoint_day(signal_type)
        if days_held < checkpoint_day:
            return False
        
        # Check if adverse
        direction = self._get_direction(signal_type)
        if direction == "LONG":
            current_return = current_price / entry_price - 1.0
        else:  # SHORT
            current_return = 1.0 - current_price / entry_price
        
        adverse_threshold = self._get_adverse_threshold(signal_type)
        return current_return < -adverse_threshold
    
    def _get_checkpoint_day(self, signal_type: str) -> int:
        """Get checkpoint day with caching."""
        if signal_type not in self._checkpoint_cache:
            # Map signal types to their TTH p75 checkpoint days
            # These values come from the policy's path profiles
            tth_p75_map = {
                "CALL": 7,
                "LONG": 7,  # Alias for CALL
                "PUT": 6,
                "PRIMARY_SHORT": 6,  # Alias for PUT
                "OPTION_CALL": 5,
                "OPTION_PUT": 3,
                "TACTICAL_PUT": 6,  # 5.5 rounded up
                "BULL_PROBE": 5,
                "BEAR_PROBE": 9,  # 8.5 rounded up
                "MVRV_SHORT": 10,
                "IRON_CONDOR": 7,
            }
            
            # Try to get from policy path profile, fallback to hardcoded
            try:
                profile = self.policy.get_path_profile(signal_type)
                # For now, use the mapping above. Could be enhanced to derive from policy data
                checkpoint_day = tth_p75_map.get(signal_type, 7)
            except Exception:
                checkpoint_day = tth_p75_map.get(signal_type, 7)
            
            self._checkpoint_cache[signal_type] = checkpoint_day
        
        return self._checkpoint_cache[signal_type]
    
    def _get_adverse_threshold(self, signal_type: str) -> float:
        """Get adverse threshold with caching."""
        if signal_type not in self._threshold_cache:
            try:
                profile = self.policy.get_path_profile(signal_type)
                mae_p75 = profile.get("mae_p75", 0.04)  # Default 4%
                threshold = mae_p75 / 2
            except Exception:
                # Fallback to hardcoded values
                mae_fallback = {
                    "CALL": 0.0471,
                    "LONG": 0.0471,
                    "PUT": 0.0441,
                    "PRIMARY_SHORT": 0.0441,
                    "OPTION_CALL": 0.0848,
                    "OPTION_PUT": 0.0682,
                    "TACTICAL_PUT": 0.0308,
                    "BULL_PROBE": 0.0384,
                    "BEAR_PROBE": 0.0653,
                    "MVRV_SHORT": 0.0719,
                    "IRON_CONDOR": 0.0676,
                }
                threshold = mae_fallback.get(signal_type, 0.04) / 2
            
            self._threshold_cache[signal_type] = threshold
        
        return self._threshold_cache[signal_type]
    
    def _get_direction(self, signal_type: str) -> str:
        """Get trade direction for a signal type."""
        long_types = {"CALL", "LONG", "OPTION_CALL", "BULL_PROBE"}
        return "LONG" if signal_type in long_types else "SHORT"
    
    def _estimate_recovery_potential(
        self,
        signal_type: str,
        current_return: float,
        days_held: int,
    ) -> float:
        """
        Estimate recovery potential based on historical patterns.
        
        This is a simplified estimation. In a full implementation, this would
        use historical recovery data or ML models to predict recovery likelihood.
        
        Args:
            signal_type: Signal type
            current_return: Current return (negative for adverse trades)
            days_held: Days held
            
        Returns:
            Estimated recovery potential (0.0 to 1.0)
        """
        # Simple heuristic based on signal type characteristics
        # In practice, this would use historical recovery analysis data
        
        # Base recovery potential from policy analysis findings
        # These are approximate values based on the recovery analyzer results
        base_recovery_rates = {
            "OPTION_CALL": 0.80,    # 100% edge found in analysis
            "PRIMARY_SHORT": 0.70,  # 45.5% edge
            "TACTICAL_PUT": 0.65,   # 44.4% edge
            "BEAR_PROBE": 0.60,     # 33.3% edge
            "CALL": 0.55,           # 27.3% edge
            "LONG": 0.55,           # Same as CALL
            "BULL_PROBE": 0.50,     # 22.2% edge
            "OPTION_PUT": 0.45,     # 16.7% edge
            "MVRV_SHORT": 0.40,     # 14.3% edge
        }
        
        base_rate = base_recovery_rates.get(signal_type, 0.50)
        
        # Adjust based on severity of adverse move
        # More adverse = lower recovery potential
        adverse_severity = abs(current_return)
        severity_penalty = min(adverse_severity * 2, 0.3)  # Max 30% penalty
        
        # Adjust based on time held
        # Later in trade = lower recovery potential
        checkpoint_day = self._get_checkpoint_day(signal_type)
        time_factor = max(0.7, 1.0 - (days_held - checkpoint_day) * 0.05)
        
        # Calculate final recovery potential
        recovery_potential = base_rate * (1 - severity_penalty) * time_factor
        
        # Ensure bounds
        return max(0.0, min(1.0, recovery_potential))


# Convenience functions for direct use
def should_flip_trade(
    signal_type: str,
    entry_price: float,
    current_price: float,
    days_held: int,
    policy: Optional[PolicyVersion] = None,
) -> bool:
    """
    Convenience function to check if a trade should be flipped.
    
    Args:
        signal_type: Signal type (e.g., "CALL", "PUT", "MVRV_SHORT")
        entry_price: Original entry price
        current_price: Current market price
        days_held: Number of days the trade has been held
        policy: Optional policy version (uses current if None)
        
    Returns:
        True if the trade should be flipped
    """
    engine = RecoveryDecisionEngine(policy)
    return engine.should_flip(signal_type, entry_price, current_price, days_held)


def should_cut_trade(
    signal_type: str,
    entry_price: float,
    current_price: float,
    days_held: int,
    policy: Optional[PolicyVersion] = None,
) -> bool:
    """
    Convenience function to check if a trade should be cut.
    
    Args:
        signal_type: Signal type (e.g., "CALL", "PUT", "MVRV_SHORT")
        entry_price: Original entry price
        current_price: Current market price
        days_held: Number of days the trade has been held
        policy: Optional policy version (uses current if None)
        
    Returns:
        True if the trade should be cut
    """
    engine = RecoveryDecisionEngine(policy)
    return engine.should_cut(signal_type, entry_price, current_price, days_held)


def get_recovery_decision(
    signal_type: str,
    entry_price: float,
    current_price: float,
    days_held: int,
    policy: Optional[PolicyVersion] = None,
) -> RecoveryDecision:
    """
    Convenience function to get a comprehensive recovery decision.
    
    Args:
        signal_type: Signal type (e.g., "CALL", "PUT", "MVRV_SHORT")
        entry_price: Original entry price
        current_price: Current market price
        days_held: Number of days the trade has been held
        policy: Optional policy version (uses current if None)
        
    Returns:
        RecoveryDecision with action, confidence, and rationale
    """
    engine = RecoveryDecisionEngine(policy)
    return engine.get_recovery_decision(signal_type, entry_price, current_price, days_held)