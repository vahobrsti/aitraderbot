"""
Unit tests for the Recovery Decision Engine.

Tests the recovery service functionality including decision logic,
threshold calculations, and integration with policy parameters.
"""

import unittest
from unittest.mock import Mock, patch
from decimal import Decimal

from execution.services.recovery import (
    RecoveryDecisionEngine,
    RecoveryDecision,
    should_flip_trade,
    should_cut_trade,
    get_recovery_decision,
)
from execution.services.policy import PolicyVersion, RecoveryConfig


class TestRecoveryDecisionEngine(unittest.TestCase):
    """Test cases for RecoveryDecisionEngine."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock policy with test recovery configurations
        self.mock_policy = Mock(spec=PolicyVersion)
        
        # Mock recovery configurations
        self.mock_policy.get_recovery_config.return_value = RecoveryConfig(
            recovery_flip_threshold=0.05,  # 5%
            recovery_cut_threshold=0.02,   # 2%
            recovery_target=0.03,          # 3%
        )
        
        self.mock_policy.get_recovery_flip_threshold.return_value = 0.05
        self.mock_policy.get_recovery_cut_threshold.return_value = 0.02
        self.mock_policy.get_recovery_target.return_value = 0.03
        
        # Mock path profiles
        self.mock_policy.get_path_profile.return_value = {
            "mae_p75": 0.04,  # 4% MAE p75
            "shakeout_pct": 0.20,
            "invalidation_pct": 0.25,
            "clean_win_pct": 0.75,
        }
        self.mock_policy.has_path_profile.return_value = True
        
        self.engine = RecoveryDecisionEngine(self.mock_policy)
    
    def test_initialization(self):
        """Test engine initialization."""
        self.assertIsNotNone(self.engine)
        self.assertEqual(self.engine.policy, self.mock_policy)
        self.assertEqual(self.engine._checkpoint_cache, {})
        self.assertEqual(self.engine._threshold_cache, {})
    
    def test_get_recovery_target(self):
        """Test recovery target retrieval."""
        target = self.engine.get_recovery_target("CALL")
        self.assertEqual(target, 0.03)
        self.mock_policy.get_recovery_target.assert_called_once_with("CALL")
    
    def test_get_checkpoint_day(self):
        """Test checkpoint day calculation."""
        # Test known signal types
        self.assertEqual(self.engine.get_checkpoint_day("CALL"), 7)
        self.assertEqual(self.engine.get_checkpoint_day("PUT"), 6)
        self.assertEqual(self.engine.get_checkpoint_day("OPTION_CALL"), 5)
        self.assertEqual(self.engine.get_checkpoint_day("OPTION_PUT"), 3)
        self.assertEqual(self.engine.get_checkpoint_day("MVRV_SHORT"), 10)
        
        # Test unknown signal type (should default to 7)
        self.assertEqual(self.engine.get_checkpoint_day("UNKNOWN_TYPE"), 7)
    
    def test_get_adverse_threshold(self):
        """Test adverse threshold calculation."""
        threshold = self.engine.get_adverse_threshold("CALL")
        # Mock policy returns mae_p75=0.04, has_path_profile=True
        # So threshold = 0.04 / 2 = 0.02
        self.assertAlmostEqual(threshold, 0.04 / 2, places=4)
    
    def test_is_recovery_candidate_before_checkpoint(self):
        """Test recovery candidate check before checkpoint day."""
        # CALL has checkpoint day 7, so day 5 should not be a candidate
        is_candidate = self.engine.is_recovery_candidate("CALL", 100.0, 95.0, 5)
        self.assertFalse(is_candidate)
    
    def test_is_recovery_candidate_not_adverse(self):
        """Test recovery candidate check when trade is not adverse."""
        # CALL is LONG, price went up (not adverse)
        is_candidate = self.engine.is_recovery_candidate("CALL", 100.0, 105.0, 7)
        self.assertFalse(is_candidate)
    
    def test_is_recovery_candidate_adverse_long(self):
        """Test recovery candidate check for adverse LONG trade."""
        # CALL is LONG, price down 3% (adverse threshold is 2%)
        is_candidate = self.engine.is_recovery_candidate("CALL", 100.0, 97.0, 7)
        self.assertTrue(is_candidate)
    
    def test_is_recovery_candidate_adverse_short(self):
        """Test recovery candidate check for adverse SHORT trade."""
        # PUT is SHORT, price up 3% (adverse threshold is 2%)
        is_candidate = self.engine.is_recovery_candidate("PUT", 100.0, 103.0, 6)
        self.assertTrue(is_candidate)
    
    def test_should_flip_high_recovery_potential(self):
        """Test flip decision with high recovery potential."""
        # Mock high recovery potential (8% > 5% flip threshold)
        with patch.object(self.engine, '_estimate_recovery_potential', return_value=0.08):
            should_flip = self.engine.should_flip("CALL", 100.0, 97.0, 7)
            self.assertTrue(should_flip)
    
    def test_should_flip_low_recovery_potential(self):
        """Test flip decision with low recovery potential."""
        # Mock low recovery potential (3% < 5% flip threshold)
        with patch.object(self.engine, '_estimate_recovery_potential', return_value=0.03):
            should_flip = self.engine.should_flip("CALL", 100.0, 97.0, 7)
            self.assertFalse(should_flip)
    
    def test_should_cut_low_recovery_potential(self):
        """Test cut decision with low recovery potential."""
        # Mock very low recovery potential (1% < 2% cut threshold)
        with patch.object(self.engine, '_estimate_recovery_potential', return_value=0.01):
            should_cut = self.engine.should_cut("CALL", 100.0, 97.0, 7)
            self.assertTrue(should_cut)
    
    def test_should_cut_high_recovery_potential(self):
        """Test cut decision with high recovery potential."""
        # Mock high recovery potential (4% > 2% cut threshold)
        with patch.object(self.engine, '_estimate_recovery_potential', return_value=0.04):
            should_cut = self.engine.should_cut("CALL", 100.0, 97.0, 7)
            self.assertFalse(should_cut)
    
    def test_get_recovery_decision_before_checkpoint(self):
        """Test recovery decision before checkpoint day."""
        decision = self.engine.get_recovery_decision("CALL", 100.0, 95.0, 5)
        
        self.assertEqual(decision.action, "HOLD")
        self.assertEqual(decision.confidence, 1.0)
        self.assertIn("checkpoint day", decision.rationale)
    
    def test_get_recovery_decision_not_adverse(self):
        """Test recovery decision when trade is not adverse."""
        decision = self.engine.get_recovery_decision("CALL", 100.0, 101.0, 7)
        
        self.assertEqual(decision.action, "HOLD")
        self.assertEqual(decision.confidence, 0.8)
        self.assertIn("not adverse", decision.rationale)
    
    def test_get_recovery_decision_flip(self):
        """Test recovery decision recommending flip."""
        with patch.object(self.engine, '_estimate_recovery_potential', return_value=0.08):
            decision = self.engine.get_recovery_decision("CALL", 100.0, 97.0, 7)
            
            self.assertEqual(decision.action, "FLIP")
            self.assertGreater(decision.confidence, 0.5)
            self.assertIn("flip", decision.rationale.lower())
    
    def test_get_recovery_decision_cut(self):
        """Test recovery decision recommending cut."""
        with patch.object(self.engine, '_estimate_recovery_potential', return_value=0.01):
            decision = self.engine.get_recovery_decision("CALL", 100.0, 97.0, 7)
            
            self.assertEqual(decision.action, "CUT")
            self.assertGreater(decision.confidence, 0.5)
            self.assertIn("cut", decision.rationale.lower())
    
    def test_get_recovery_decision_hold(self):
        """Test recovery decision recommending hold."""
        with patch.object(self.engine, '_estimate_recovery_potential', return_value=0.035):
            decision = self.engine.get_recovery_decision("CALL", 100.0, 97.0, 7)
            
            self.assertEqual(decision.action, "HOLD")
            self.assertEqual(decision.confidence, 0.6)
            self.assertIn("holding", decision.rationale.lower())
    
    def test_estimate_recovery_potential_option_call(self):
        """Test recovery potential estimation for OPTION_CALL."""
        # OPTION_CALL should have highest base recovery MFE (0.08)
        potential = self.engine._estimate_recovery_potential("OPTION_CALL", -0.03, 5)
        self.assertGreater(potential, 0.05)  # Should exceed flip threshold
    
    def test_estimate_recovery_potential_mvrv_short(self):
        """Test recovery potential estimation for MVRV_SHORT."""
        # MVRV_SHORT has weakest flip edge (+14.3%), base MFE just above threshold
        # With severity factor for -0.03 adverse (mildly adverse = 0.9), result is ~0.047
        potential = self.engine._estimate_recovery_potential("MVRV_SHORT", -0.03, 10)
        # Should be close to but may be slightly above or below flip threshold
        self.assertLess(potential, 0.06)  # Not a strong flip signal
        self.assertGreater(potential, 0.03)  # But not a cut signal either
    
    def test_estimate_recovery_potential_severity_penalty(self):
        """Test that more adverse moves get lower recovery potential."""
        # Less adverse move
        potential_mild = self.engine._estimate_recovery_potential("CALL", -0.02, 7)
        
        # More adverse move
        potential_severe = self.engine._estimate_recovery_potential("CALL", -0.10, 7)
        
        self.assertGreater(potential_mild, potential_severe)
    
    def test_estimate_recovery_potential_time_factor(self):
        """Test that later trades get lower recovery potential."""
        # Earlier in trade lifecycle
        potential_early = self.engine._estimate_recovery_potential("CALL", -0.03, 7)
        
        # Later in trade lifecycle
        potential_late = self.engine._estimate_recovery_potential("CALL", -0.03, 12)
        
        self.assertGreater(potential_early, potential_late)
    
    def test_direction_mapping(self):
        """Test signal type to direction mapping."""
        # Test LONG types
        self.assertEqual(self.engine._get_direction("CALL"), "LONG")
        self.assertEqual(self.engine._get_direction("LONG"), "LONG")
        self.assertEqual(self.engine._get_direction("OPTION_CALL"), "LONG")
        self.assertEqual(self.engine._get_direction("BULL_PROBE"), "LONG")
        
        # Test SHORT types
        self.assertEqual(self.engine._get_direction("PUT"), "SHORT")
        self.assertEqual(self.engine._get_direction("PRIMARY_SHORT"), "SHORT")
        self.assertEqual(self.engine._get_direction("OPTION_PUT"), "SHORT")
        self.assertEqual(self.engine._get_direction("BEAR_PROBE"), "SHORT")
        self.assertEqual(self.engine._get_direction("TACTICAL_PUT"), "SHORT")
        self.assertEqual(self.engine._get_direction("MVRV_SHORT"), "SHORT")
    
    def test_caching(self):
        """Test that checkpoint days and thresholds are cached."""
        # First call should populate cache
        checkpoint1 = self.engine._get_checkpoint_day("CALL")
        threshold1 = self.engine._get_adverse_threshold("CALL")
        
        # Second call should use cache
        checkpoint2 = self.engine._get_checkpoint_day("CALL")
        threshold2 = self.engine._get_adverse_threshold("CALL")
        
        self.assertEqual(checkpoint1, checkpoint2)
        self.assertEqual(threshold1, threshold2)
        
        # Check cache is populated
        self.assertIn("CALL", self.engine._checkpoint_cache)
        self.assertIn("CALL", self.engine._threshold_cache)
    
    def test_error_handling(self):
        """Test error handling in recovery decision."""
        # Create a new engine with a policy that raises exceptions
        error_policy = Mock(spec=PolicyVersion)
        error_policy.get_path_profile.side_effect = Exception("Policy error")
        error_policy.get_recovery_flip_threshold.side_effect = Exception("Policy error")
        error_policy.get_recovery_cut_threshold.side_effect = Exception("Policy error")
        
        error_engine = RecoveryDecisionEngine(error_policy)
        
        # Should not crash and should return safe default
        decision = error_engine.get_recovery_decision("CALL", 100.0, 97.0, 7)
        
        self.assertEqual(decision.action, "HOLD")
        self.assertEqual(decision.confidence, 0.1)
        self.assertIn("Error", decision.rationale)


class TestRecoveryDecision(unittest.TestCase):
    """Test cases for RecoveryDecision dataclass."""
    
    def test_decision_properties(self):
        """Test RecoveryDecision properties."""
        # Test FLIP decision
        flip_decision = RecoveryDecision(
            action="FLIP",
            confidence=0.8,
            checkpoint_return=-0.03,
            recovery_potential=0.07,
            rationale="High recovery potential"
        )
        
        self.assertTrue(flip_decision.should_flip)
        self.assertFalse(flip_decision.should_cut)
        self.assertFalse(flip_decision.should_hold)
        
        # Test CUT decision
        cut_decision = RecoveryDecision(
            action="CUT",
            confidence=0.9,
            checkpoint_return=-0.05,
            recovery_potential=0.01,
            rationale="Low recovery potential"
        )
        
        self.assertFalse(cut_decision.should_flip)
        self.assertTrue(cut_decision.should_cut)
        self.assertFalse(cut_decision.should_hold)
        
        # Test HOLD decision
        hold_decision = RecoveryDecision(
            action="HOLD",
            confidence=0.6,
            checkpoint_return=-0.02,
            recovery_potential=0.035,
            rationale="Moderate recovery potential"
        )
        
        self.assertFalse(hold_decision.should_flip)
        self.assertFalse(hold_decision.should_cut)
        self.assertTrue(hold_decision.should_hold)


class TestConvenienceFunctions(unittest.TestCase):
    """Test cases for convenience functions."""
    
    @patch('execution.services.recovery.RecoveryDecisionEngine')
    def test_should_flip_trade(self, mock_engine_class):
        """Test should_flip_trade convenience function."""
        mock_engine = Mock()
        mock_engine.should_flip.return_value = True
        mock_engine_class.return_value = mock_engine
        
        result = should_flip_trade("CALL", 100.0, 97.0, 7)
        
        self.assertTrue(result)
        mock_engine.should_flip.assert_called_once_with("CALL", 100.0, 97.0, 7)
    
    @patch('execution.services.recovery.RecoveryDecisionEngine')
    def test_should_cut_trade(self, mock_engine_class):
        """Test should_cut_trade convenience function."""
        mock_engine = Mock()
        mock_engine.should_cut.return_value = True
        mock_engine_class.return_value = mock_engine
        
        result = should_cut_trade("PUT", 100.0, 103.0, 6)
        
        self.assertTrue(result)
        mock_engine.should_cut.assert_called_once_with("PUT", 100.0, 103.0, 6)
    
    @patch('execution.services.recovery.RecoveryDecisionEngine')
    def test_get_recovery_decision_function(self, mock_engine_class):
        """Test get_recovery_decision convenience function."""
        mock_engine = Mock()
        mock_decision = RecoveryDecision(
            action="FLIP",
            confidence=0.8,
            checkpoint_return=-0.03,
            recovery_potential=0.07,
            rationale="Test decision"
        )
        mock_engine.get_recovery_decision.return_value = mock_decision
        mock_engine_class.return_value = mock_engine
        
        result = get_recovery_decision("CALL", 100.0, 97.0, 7)
        
        self.assertEqual(result, mock_decision)
        mock_engine.get_recovery_decision.assert_called_once_with("CALL", 100.0, 97.0, 7)


if __name__ == '__main__':
    unittest.main()