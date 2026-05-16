"""
Tests for RecoveryCandidate three-way recommendation system.
"""

import unittest
from signals.management.commands.analyze_loser_recovery import RecoveryCandidate


class TestRecoveryCandidate(unittest.TestCase):
    """Test the RecoveryCandidate recommendation logic."""

    def test_flip_recommendation(self):
        """Test FLIP recommendation for high recovery MFE."""
        candidate = RecoveryCandidate(
            date="2023-01-01",
            trade_type="LONG",
            direction="LONG",
            entry_price=50000.0,
            checkpoint_day=7,
            checkpoint_price=47000.0,
            checkpoint_return=-0.06,
            remaining_days=7,
            forward_return=-0.02,
            original_hit=False,
            recovery_hit=True,
            recovery_return=0.08,  # 8% > 5% default flip threshold
            flip_threshold=0.05,
            cut_threshold=0.02,
        )
        self.assertEqual(candidate.recommendation, "FLIP")

    def test_hold_recommendation(self):
        """Test HOLD recommendation for moderate recovery MFE."""
        candidate = RecoveryCandidate(
            date="2023-01-01",
            trade_type="LONG",
            direction="LONG",
            entry_price=50000.0,
            checkpoint_day=7,
            checkpoint_price=47000.0,
            checkpoint_return=-0.06,
            remaining_days=7,
            forward_return=-0.02,
            original_hit=False,
            recovery_hit=True,
            recovery_return=0.03,  # 3% between 2% and 5%
            flip_threshold=0.05,
            cut_threshold=0.02,
        )
        self.assertEqual(candidate.recommendation, "HOLD")

    def test_cut_recommendation(self):
        """Test CUT recommendation for low recovery MFE."""
        candidate = RecoveryCandidate(
            date="2023-01-01",
            trade_type="LONG",
            direction="LONG",
            entry_price=50000.0,
            checkpoint_day=7,
            checkpoint_price=47000.0,
            checkpoint_return=-0.06,
            remaining_days=7,
            forward_return=-0.02,
            original_hit=False,
            recovery_hit=False,
            recovery_return=0.01,  # 1% < 2% default cut threshold
            flip_threshold=0.05,
            cut_threshold=0.02,
        )
        self.assertEqual(candidate.recommendation, "CUT")

    def test_custom_thresholds(self):
        """Test custom threshold configuration."""
        # Test with custom thresholds: flip > 6%, cut < 1%
        candidate = RecoveryCandidate(
            date="2023-01-01",
            trade_type="LONG",
            direction="LONG",
            entry_price=50000.0,
            checkpoint_day=7,
            checkpoint_price=47000.0,
            checkpoint_return=-0.06,
            remaining_days=7,
            forward_return=-0.02,
            original_hit=False,
            recovery_hit=True,
            recovery_return=0.03,  # 3% between 1% and 6%
            flip_threshold=0.06,
            cut_threshold=0.01,
        )
        self.assertEqual(candidate.recommendation, "HOLD")

        # Same recovery return but with different thresholds should give FLIP
        candidate.flip_threshold = 0.02  # Lower flip threshold
        self.assertEqual(candidate.recommendation, "FLIP")

        # Same recovery return but with different thresholds should give CUT
        candidate.flip_threshold = 0.06
        candidate.cut_threshold = 0.04  # Higher cut threshold
        self.assertEqual(candidate.recommendation, "CUT")


if __name__ == "__main__":
    unittest.main()