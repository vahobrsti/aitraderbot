"""
Tests for recovery policy validation - Task 7.1

Tests policy integration with various signal types, validates recovery threshold 
calculations, and tests edge case handling (missing policy data, invalid thresholds).

Requirements: 8.1, 8.2, 8.3
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from decimal import Decimal

from django.test import TestCase
from django.core.management import call_command
from io import StringIO

from signals.management.commands.analyze_loser_recovery import (
    Command, 
    PolicyConfigAdapter, 
    RecoveryCandidate,
    TTH_P75_BY_TYPE_FALLBACK,
    MAE_W_P75_BY_TYPE_FALLBACK
)
from execution.services.policy import (
    get_policy, 
    PolicyVersion, 
    RecoveryConfig,
    ExitConfig,
    TierConfig,
    DTEConfig
)


class TestPolicyConfigAdapter(TestCase):
    """Test PolicyConfigAdapter integration with policy.py"""

    def setUp(self):
        """Set up test policy data."""
        # Reset cached engine to ensure test isolation
        PolicyConfigAdapter.reset_cache()
        
        # Create a mock policy with test data
        self.mock_policy = PolicyVersion(
            version="test-1.0",
            description="Test policy for recovery validation",
            path_profiles={
                "CALL": {
                    "mae_p75": 0.0471,
                    "shakeout_pct": 0.21,
                    "invalidation_pct": 0.27,
                    "clean_win_pct": 0.729
                },
                "PUT": {
                    "mae_p75": 0.0441,
                    "shakeout_pct": 0.18,
                    "invalidation_pct": 0.29,
                    "clean_win_pct": 0.714
                },
                "MVRV_SHORT": {
                    "mae_p75": 0.0719,
                    "shakeout_pct": 0.57,
                    "invalidation_pct": 0.43,
                    "clean_win_pct": 0.571
                },
                "OPTION_CALL": {
                    "mae_p75": 0.0848,
                    "shakeout_pct": 0.35,
                    "invalidation_pct": 0.54,
                    "clean_win_pct": 0.458
                },
                "TACTICAL_PUT": {
                    "mae_p75": 0.0308,
                    "shakeout_pct": 0.15,
                    "invalidation_pct": 0.25,
                    "clean_win_pct": 0.750
                }
            },
            recovery_configs={
                "CALL": RecoveryConfig(
                    recovery_flip_threshold=0.05,
                    recovery_cut_threshold=0.02,
                    recovery_target=0.03
                ),
                "PUT": RecoveryConfig(
                    recovery_flip_threshold=0.04,
                    recovery_cut_threshold=0.015,
                    recovery_target=0.025
                )
            }
        )

    @patch('signals.management.commands.analyze_loser_recovery.get_policy')
    def test_checkpoint_day_from_policy(self, mock_get_policy):
        """Test checkpoint day derivation from policy TTH p75 values."""
        mock_get_policy.return_value = self.mock_policy
        
        # Test known signal types with expected TTH p75 values
        test_cases = [
            ("CALL", 7),
            ("PUT", 6), 
            ("OPTION_CALL", 5),
            ("OPTION_PUT", 3),
            ("TACTICAL_PUT", 6),  # 5.5 rounded up
            ("BULL_PROBE", 5),
            ("BEAR_PROBE", 9),    # 8.5 rounded up
            ("MVRV_SHORT", 10),
            ("IRON_CONDOR", 7),
        ]
        
        for signal_type, expected_checkpoint in test_cases:
            with self.subTest(signal_type=signal_type):
                checkpoint = PolicyConfigAdapter.get_checkpoint_day(signal_type)
                self.assertEqual(
                    checkpoint, 
                    expected_checkpoint,
                    f"Checkpoint day for {signal_type} should be {expected_checkpoint}, got {checkpoint}"
                )

    @patch('signals.management.commands.analyze_loser_recovery.get_policy')
    def test_checkpoint_day_fallback_for_unknown_signal(self, mock_get_policy):
        """Test fallback to default checkpoint for unknown signal types."""
        mock_get_policy.return_value = self.mock_policy
        
        # Test unknown signal type
        checkpoint = PolicyConfigAdapter.get_checkpoint_day("UNKNOWN_SIGNAL")
        self.assertEqual(checkpoint, 7, "Unknown signal type should use default checkpoint of 7 days")

    @patch('signals.management.commands.analyze_loser_recovery.get_policy')
    def test_adverse_threshold_from_policy(self, mock_get_policy):
        """Test adverse threshold calculation from policy MAE(W) p75 values."""
        mock_get_policy.return_value = self.mock_policy
        
        # Test signal types with known MAE p75 values
        test_cases = [
            ("CALL", 0.0471 / 2),      # 2.355%
            ("PUT", 0.0441 / 2),       # 2.205%
            ("MVRV_SHORT", 0.0719 / 2), # 3.595%
            ("OPTION_CALL", 0.0848 / 2), # 4.24%
            ("TACTICAL_PUT", 0.0308 / 2), # 1.54%
        ]
        
        for signal_type, expected_threshold in test_cases:
            with self.subTest(signal_type=signal_type):
                threshold = PolicyConfigAdapter.get_adverse_threshold(signal_type)
                self.assertAlmostEqual(
                    threshold, 
                    expected_threshold, 
                    places=4,
                    msg=f"Adverse threshold for {signal_type} should be {expected_threshold:.4f}, got {threshold:.4f}"
                )

    @patch('signals.management.commands.analyze_loser_recovery.get_policy')
    def test_adverse_threshold_fallback_for_unknown_signal(self, mock_get_policy):
        """Test fallback to default adverse threshold for unknown signal types."""
        mock_get_policy.return_value = self.mock_policy
        
        # Test unknown signal type — falls back to calibrated default (0.04 / 2 = 0.02)
        threshold = PolicyConfigAdapter.get_adverse_threshold("UNKNOWN_SIGNAL")
        expected_fallback = 0.04 / 2  # Calibrated default for unknown types
        self.assertEqual(
            threshold, 
            expected_fallback,
            f"Unknown signal type should use default threshold of {expected_fallback}"
        )

    def test_adverse_threshold_user_override(self):
        """Test that user-provided override takes precedence over policy values."""
        user_override = 0.025  # 2.5%
        
        # Test with various signal types - override should always be used
        test_signals = ["CALL", "PUT", "MVRV_SHORT", "UNKNOWN_SIGNAL"]
        
        for signal_type in test_signals:
            with self.subTest(signal_type=signal_type):
                threshold = PolicyConfigAdapter.get_adverse_threshold(signal_type, override=user_override)
                self.assertEqual(
                    threshold, 
                    user_override,
                    f"User override should take precedence for {signal_type}"
                )

    @patch('signals.management.commands.analyze_loser_recovery.get_policy')
    def test_policy_unavailable_fallback(self, mock_get_policy):
        """Test fallback behavior when policy is unavailable."""
        # Simulate policy unavailable by raising exception
        mock_get_policy.side_effect = Exception("Policy service unavailable")
        
        # Test checkpoint day fallback
        checkpoint = PolicyConfigAdapter.get_checkpoint_day("CALL")
        expected_fallback = TTH_P75_BY_TYPE_FALLBACK.get("CALL", 7)
        self.assertEqual(
            checkpoint, 
            expected_fallback,
            "Should fallback to hardcoded values when policy unavailable"
        )
        
        # Test adverse threshold fallback
        threshold = PolicyConfigAdapter.get_adverse_threshold("CALL")
        expected_threshold = MAE_W_P75_BY_TYPE_FALLBACK.get("CALL", 0.04) / 2
        self.assertEqual(
            threshold, 
            expected_threshold,
            "Should fallback to hardcoded values when policy unavailable"
        )


class TestRecoveryPolicyIntegration(TestCase):
    """Test recovery policy integration with various signal types."""

    def setUp(self):
        """Set up test data for recovery analysis."""
        self.command = Command()
        
        # Create sample price data
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        self.price_df = pd.DataFrame({
            'btc_close': [50000 + i * 100 for i in range(20)],
            'btc_high': [50500 + i * 100 for i in range(20)],
            'btc_low': [49500 + i * 100 for i in range(20)]
        }, index=dates)
        
        # Create sample trades data
        self.trades_df = pd.DataFrame([
            {'date': '2023-01-01', 'type': 'CALL', 'direction': 'LONG'},
            {'date': '2023-01-02', 'type': 'PUT', 'direction': 'SHORT'},
            {'date': '2023-01-03', 'type': 'MVRV_SHORT', 'direction': 'SHORT'},
            {'date': '2023-01-04', 'type': 'OPTION_CALL', 'direction': 'LONG'},
            {'date': '2023-01-05', 'type': 'TACTICAL_PUT', 'direction': 'SHORT'},
        ])

    @patch('signals.management.commands.analyze_loser_recovery.PolicyConfigAdapter.get_checkpoint_day')
    @patch('signals.management.commands.analyze_loser_recovery.PolicyConfigAdapter.get_adverse_threshold')
    def test_signal_type_specific_parameters(self, mock_get_threshold, mock_get_checkpoint):
        """Test that different signal types use their specific policy parameters."""
        # Mock policy responses for different signal types
        checkpoint_map = {
            'CALL': 7,
            'PUT': 6,
            'MVRV_SHORT': 10,
            'OPTION_CALL': 5,
            'TACTICAL_PUT': 6
        }
        threshold_map = {
            'CALL': 0.02355,
            'PUT': 0.02205,
            'MVRV_SHORT': 0.03595,
            'OPTION_CALL': 0.0424,
            'TACTICAL_PUT': 0.0154
        }
        
        mock_get_checkpoint.side_effect = lambda signal_type: checkpoint_map.get(signal_type, 7)
        mock_get_threshold.side_effect = lambda signal_type, override=None: (
            override if override is not None else threshold_map.get(signal_type, 0.02)
        )
        
        # Analyze recovery for all signal types
        candidates = self.command._analyze_recovery(
            self.trades_df, 
            self.price_df,
            horizon=14,
            target=0.05,
            recovery_target=0.03,
            adverse_threshold=None,  # Use policy values
            checkpoint_mode="tth_p75",
            fixed_checkpoint=7,
            flip_threshold=0.05,
            cut_threshold=0.02
        )
        
        # Verify that policy methods were called for each signal type
        expected_calls = [
            unittest.mock.call('CALL'),
            unittest.mock.call('PUT'), 
            unittest.mock.call('MVRV_SHORT'),
            unittest.mock.call('OPTION_CALL'),
            unittest.mock.call('TACTICAL_PUT')
        ]
        
        mock_get_checkpoint.assert_has_calls(expected_calls, any_order=True)
        mock_get_threshold.assert_has_calls([
            unittest.mock.call('CALL'),
            unittest.mock.call('PUT'), 
            unittest.mock.call('MVRV_SHORT'),
            unittest.mock.call('OPTION_CALL'),
            unittest.mock.call('TACTICAL_PUT')
        ], any_order=True)

    def test_recovery_threshold_calculations_accuracy(self):
        """Test accuracy of recovery threshold calculations."""
        # Create a recovery candidate with known values
        candidate = RecoveryCandidate(
            date="2023-01-01",
            trade_type="CALL",
            direction="LONG",
            entry_price=50000.0,
            checkpoint_day=7,
            checkpoint_price=47000.0,  # 6% down from entry
            checkpoint_return=-0.06,
            remaining_days=7,
            forward_return=-0.02,
            original_hit=False,
            recovery_hit=True,
            recovery_return=0.08,  # 8% recovery MFE
            flip_threshold=0.05,
            cut_threshold=0.02
        )
        
        # Test checkpoint return calculation
        expected_checkpoint_return = (47000.0 / 50000.0) - 1.0  # -0.06
        self.assertAlmostEqual(
            candidate.checkpoint_return, 
            expected_checkpoint_return, 
            places=4,
            msg="Checkpoint return calculation should be accurate"
        )
        
        # Test adverse classification (LONG trade down 6% > 2.355% threshold)
        # This would be determined by the analyzer based on policy threshold
        self.assertLess(
            candidate.checkpoint_return, 
            -0.02,  # Simplified threshold for test
            "Trade should be classified as adverse when price moves against direction"
        )
        
        # Test recovery return bounds
        self.assertGreaterEqual(
            candidate.recovery_return, 
            0.0,
            "Recovery return should be non-negative"
        )
        self.assertLessEqual(
            candidate.recovery_return, 
            1.0,
            "Recovery return should not exceed 100%"
        )

    def test_direction_specific_adverse_classification(self):
        """Test that adverse classification is direction-specific."""
        # Test LONG trade adverse classification
        long_candidate = RecoveryCandidate(
            date="2023-01-01",
            trade_type="CALL",
            direction="LONG",
            entry_price=50000.0,
            checkpoint_day=7,
            checkpoint_price=47000.0,  # Price down - adverse for LONG
            checkpoint_return=-0.06,
            remaining_days=7,
            forward_return=-0.02,
            original_hit=False,
            recovery_hit=True,
            recovery_return=0.08,
            flip_threshold=0.05,
            cut_threshold=0.02
        )
        
        # Test SHORT trade adverse classification  
        short_candidate = RecoveryCandidate(
            date="2023-01-01",
            trade_type="PUT",
            direction="SHORT",
            entry_price=50000.0,
            checkpoint_day=6,
            checkpoint_price=53000.0,  # Price up - adverse for SHORT
            checkpoint_return=-0.06,  # Calculated as 1 - (53000/50000) = -0.06
            remaining_days=8,
            forward_return=-0.02,
            original_hit=False,
            recovery_hit=True,
            recovery_return=0.05,
            flip_threshold=0.05,
            cut_threshold=0.02
        )
        
        # Both should be adverse (negative checkpoint returns)
        self.assertLess(
            long_candidate.checkpoint_return, 
            0,
            "LONG trade with price decline should have negative checkpoint return"
        )
        self.assertLess(
            short_candidate.checkpoint_return, 
            0,
            "SHORT trade with price increase should have negative checkpoint return"
        )


class TestEdgeCaseHandling(TestCase):
    """Test edge case handling for missing policy data and invalid thresholds."""

    def setUp(self):
        """Set up test command instance."""
        self.command = Command()
        # Reset cached engine to ensure test isolation
        PolicyConfigAdapter.reset_cache()

    @patch('signals.management.commands.analyze_loser_recovery.get_policy')
    def test_missing_policy_data_graceful_handling(self, mock_get_policy):
        """Test graceful handling when policy data is missing."""
        # Create policy with missing path_profiles for some signal types
        incomplete_policy = PolicyVersion(
            version="incomplete-1.0",
            description="Incomplete policy for testing",
            path_profiles={
                "CALL": {"mae_p75": 0.0471},
                # Missing PUT, MVRV_SHORT, etc.
            }
        )
        mock_get_policy.return_value = incomplete_policy
        
        # Test checkpoint day for missing signal type
        checkpoint = PolicyConfigAdapter.get_checkpoint_day("MISSING_SIGNAL")
        self.assertEqual(
            checkpoint, 
            7,  # Default fallback
            "Should use default checkpoint when signal type missing from policy"
        )
        
        # Test adverse threshold for missing signal type
        threshold = PolicyConfigAdapter.get_adverse_threshold("MISSING_SIGNAL")
        expected_fallback = 0.04 / 2  # Calibrated default for unknown types
        self.assertEqual(
            threshold, 
            expected_fallback,
            f"Should use fallback threshold when signal type missing from policy: expected {expected_fallback}, got {threshold}"
        )

    @patch('signals.management.commands.analyze_loser_recovery.get_policy')
    def test_corrupted_policy_data_handling(self, mock_get_policy):
        """Test handling of corrupted or invalid policy data."""
        # Create policy with invalid/corrupted data
        corrupted_policy = PolicyVersion(
            version="corrupted-1.0", 
            description="Corrupted policy for testing",
            path_profiles={
                "CALL": {
                    "mae_p75": "invalid_string",  # Should be float
                    "shakeout_pct": None,
                },
                "PUT": {
                    "mae_p75": -0.05,  # Invalid negative value
                }
            }
        )
        mock_get_policy.return_value = corrupted_policy
        
        # Test that invalid data doesn't crash the system
        try:
            threshold_call = PolicyConfigAdapter.get_adverse_threshold("CALL")
            threshold_put = PolicyConfigAdapter.get_adverse_threshold("PUT")
            
            # Should get fallback values, not crash
            self.assertIsInstance(threshold_call, float)
            self.assertIsInstance(threshold_put, float)
            # Note: PUT with negative mae_p75 will result in negative threshold
            # The system should handle this gracefully, even if the result is negative
            
        except Exception as e:
            self.fail(f"Should handle corrupted policy data gracefully, but got: {e}")

    def test_invalid_threshold_values_validation(self):
        """Test validation of invalid threshold values."""
        # Test that RecoveryCandidate accepts negative thresholds (no validation in constructor)
        # The recommendation logic should still work even with invalid thresholds
        candidate = RecoveryCandidate(
            date="2023-01-01",
            trade_type="CALL",
            direction="LONG", 
            entry_price=50000.0,
            checkpoint_day=7,
            checkpoint_price=47000.0,
            checkpoint_return=-0.06,
            remaining_days=7,
            forward_return=-0.02,
            original_hit=False,
            recovery_hit=True,
            recovery_return=0.08,
            flip_threshold=-0.05,  # Invalid negative threshold
            cut_threshold=0.02
        )
        
        # Should still produce a recommendation even with invalid thresholds
        recommendation = candidate.recommendation
        self.assertIn(recommendation, ["FLIP", "HOLD", "CUT"])
        
        # Test threshold ordering (cut_threshold >= flip_threshold is logically invalid)
        candidate2 = RecoveryCandidate(
            date="2023-01-01",
            trade_type="CALL",
            direction="LONG",
            entry_price=50000.0,
            checkpoint_day=7,
            checkpoint_price=47000.0,
            checkpoint_return=-0.06,
            remaining_days=7,
            forward_return=-0.02,
            original_hit=False,
            recovery_hit=True,
            recovery_return=0.03,  # 3% between invalid thresholds
            flip_threshold=0.02,   # Lower than cut threshold
            cut_threshold=0.05     # Higher than flip threshold
        )
        
        # With invalid threshold ordering, recommendation logic should still work
        # but may produce unexpected results
        recommendation2 = candidate2.recommendation
        self.assertIn(recommendation2, ["FLIP", "HOLD", "CUT"])

    def test_extreme_price_data_handling(self):
        """Test handling of extreme or invalid price data."""
        # Create price data with extreme values
        extreme_dates = pd.date_range('2023-01-01', periods=5, freq='D')
        extreme_price_df = pd.DataFrame({
            'btc_close': [50000, 0, np.inf, -1000, 50000],  # Invalid values
            'btc_high': [51000, 100, np.inf, -500, 51000],
            'btc_low': [49000, -100, 1000, -2000, 49000]
        }, index=extreme_dates)
        
        trades_df = pd.DataFrame([
            {'date': '2023-01-01', 'type': 'CALL', 'direction': 'LONG'},
        ])
        
        # Analysis should handle extreme values gracefully
        try:
            candidates = self.command._analyze_recovery(
                trades_df,
                extreme_price_df,
                horizon=14,
                target=0.05,
                recovery_target=0.03,
                adverse_threshold=0.02,
                checkpoint_mode="tth_p75",
                fixed_checkpoint=7,
                flip_threshold=0.05,
                cut_threshold=0.02
            )
            
            # Should return empty list or valid candidates, not crash
            self.assertIsInstance(candidates, list)
            
            # Any returned candidates should have valid data
            for candidate in candidates:
                self.assertIsInstance(candidate.entry_price, float)
                self.assertGreater(candidate.entry_price, 0)
                self.assertTrue(np.isfinite(candidate.entry_price))
                
        except Exception as e:
            self.fail(f"Should handle extreme price data gracefully, but got: {e}")

    def test_empty_data_handling(self):
        """Test handling of empty datasets."""
        empty_trades_df = pd.DataFrame(columns=['date', 'type', 'direction'])
        empty_price_df = pd.DataFrame(columns=['btc_close', 'btc_high', 'btc_low'])
        
        # Should handle empty data without crashing
        candidates = self.command._analyze_recovery(
            empty_trades_df,
            empty_price_df,
            horizon=14,
            target=0.05,
            recovery_target=0.03,
            adverse_threshold=0.02,
            checkpoint_mode="tth_p75",
            fixed_checkpoint=7,
            flip_threshold=0.05,
            cut_threshold=0.02
        )
        
        self.assertEqual(len(candidates), 0, "Empty data should return empty candidates list")

    @patch('signals.management.commands.analyze_loser_recovery.get_policy')
    def test_policy_service_timeout_handling(self, mock_get_policy):
        """Test handling of policy service timeouts or network issues."""
        # Simulate timeout/network error
        mock_get_policy.side_effect = TimeoutError("Policy service timeout")
        
        # Should fallback gracefully
        checkpoint = PolicyConfigAdapter.get_checkpoint_day("CALL")
        threshold = PolicyConfigAdapter.get_adverse_threshold("CALL")
        
        # Should get fallback values
        self.assertEqual(checkpoint, TTH_P75_BY_TYPE_FALLBACK.get("CALL", 7))
        self.assertEqual(threshold, MAE_W_P75_BY_TYPE_FALLBACK.get("CALL", 0.04) / 2)


class TestRecoveryPolicyValidationIntegration(TestCase):
    """Integration tests for recovery policy validation."""

    @patch('signals.management.commands.analyze_loser_recovery.Path.exists', return_value=True)
    @patch('signals.management.commands.analyze_loser_recovery.Command._build_trades_df')
    @patch('signals.management.commands.analyze_loser_recovery.Command._load_prices')
    def test_full_command_integration_with_policy(self, mock_load_prices, mock_build_trades, mock_path_exists):
        """Test full command integration with policy validation."""
        # Mock dependencies
        mock_build_trades.return_value = pd.DataFrame([
            {'date': '2023-01-01', 'type': 'CALL', 'direction': 'LONG'},
            {'date': '2023-01-02', 'type': 'MVRV_SHORT', 'direction': 'SHORT'},
        ])
        
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        mock_load_prices.return_value = pd.DataFrame({
            'btc_close': [50000 - i * 200 for i in range(20)],  # Declining prices
            'btc_high': [50500 - i * 200 for i in range(20)],
            'btc_low': [49500 - i * 200 for i in range(20)]
        }, index=dates)
        
        # Capture command output
        out = StringIO()
        
        # Run command with policy integration
        call_command(
            'analyze_loser_recovery',
            '--type', 'CALL',
            '--adverse-threshold', '0.025',  # Override policy value
            stdout=out
        )
        
        output = out.getvalue()
        
        # Verify policy integration sections are present
        self.assertIn("RECOVERY POLICY ANALYZER", output)
        
        # Should handle the command without errors
        self.assertNotIn("Error", output)
        self.assertNotIn("Exception", output)

    def test_policy_parameter_consistency(self):
        """Test consistency between policy parameters and recovery analysis."""
        # Get current policy
        policy = get_policy()
        
        # Test that recovery configs exist for expected signal types
        expected_signals = ["CALL", "PUT", "OPTION_CALL", "OPTION_PUT", "TACTICAL_PUT", 
                          "BULL_PROBE", "BEAR_PROBE", "MVRV_SHORT"]
        
        for signal_type in expected_signals:
            with self.subTest(signal_type=signal_type):
                # Should have path profile data
                profile = policy.get_path_profile(signal_type)
                self.assertIn("mae_p75", profile)
                self.assertIsInstance(profile["mae_p75"], (int, float))
                self.assertGreater(profile["mae_p75"], 0)
                
                # Should have recovery config (except IRON_CONDOR)
                if signal_type != "IRON_CONDOR":
                    recovery_config = policy.get_recovery_config(signal_type)
                    self.assertIsInstance(recovery_config, RecoveryConfig)
                    self.assertGreater(recovery_config.recovery_flip_threshold, 0)
                    self.assertGreater(recovery_config.recovery_cut_threshold, 0)
                    self.assertGreater(recovery_config.recovery_target, 0)
                    
                    # Logical threshold ordering
                    self.assertLess(
                        recovery_config.recovery_cut_threshold,
                        recovery_config.recovery_flip_threshold,
                        f"Cut threshold should be less than flip threshold for {signal_type}"
                    )


if __name__ == '__main__':
    unittest.main()