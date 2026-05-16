"""
Test the enhanced MFE analysis functionality in analyze_loser_recovery command.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from django.test import TestCase
from django.core.management import call_command
from io import StringIO

from signals.management.commands.analyze_loser_recovery import Command, RecoveryCandidate


class TestRecoveryMFEAnalysis(TestCase):
    """Test the enhanced MFE distribution analysis functionality."""

    def setUp(self):
        """Set up test data."""
        self.command = Command()
        
        # Create sample recovery candidates for testing
        self.sample_candidates = [
            RecoveryCandidate(
                date="2022-01-01",
                trade_type="MVRV_SHORT",
                direction="SHORT",
                entry_price=50000.0,
                checkpoint_day=10,
                checkpoint_price=47000.0,
                checkpoint_return=-0.06,
                remaining_days=4,
                forward_return=-0.02,
                original_hit=False,
                recovery_hit=True,
                recovery_return=0.08,  # 8% recovery MFE
                flip_threshold=0.05,
                cut_threshold=0.02,
            ),
            RecoveryCandidate(
                date="2022-01-15",
                trade_type="LONG",
                direction="LONG",
                entry_price=45000.0,
                checkpoint_day=7,
                checkpoint_price=43000.0,
                checkpoint_return=-0.044,
                remaining_days=7,
                forward_return=0.01,
                original_hit=True,
                recovery_hit=False,
                recovery_return=0.015,  # 1.5% recovery MFE
                flip_threshold=0.05,
                cut_threshold=0.02,
            ),
            RecoveryCandidate(
                date="2022-02-01",
                trade_type="BULL_PROBE",
                direction="LONG",
                entry_price=40000.0,
                checkpoint_day=5,
                checkpoint_price=38000.0,
                checkpoint_return=-0.05,
                remaining_days=9,
                forward_return=-0.01,
                original_hit=False,
                recovery_hit=False,
                recovery_return=0.005,  # 0.5% recovery MFE
                flip_threshold=0.05,
                cut_threshold=0.02,
            ),
        ]

    def test_recovery_candidate_recommendation_logic(self):
        """Test the recommendation logic for recovery candidates."""
        # Test FLIP recommendation (recovery_return > flip_threshold)
        flip_candidate = self.sample_candidates[0]
        self.assertEqual(flip_candidate.recommendation, "FLIP")
        
        # Test CUT recommendation (recovery_return < cut_threshold)
        cut_candidate = self.sample_candidates[2]
        self.assertEqual(cut_candidate.recommendation, "CUT")
        
        # Test HOLD recommendation (cut_threshold <= recovery_return <= flip_threshold)
        hold_candidate = self.sample_candidates[1]
        self.assertEqual(hold_candidate.recommendation, "CUT")  # 1.5% < 2% threshold

    def test_analyze_optimal_thresholds(self):
        """Test the optimal threshold analysis functionality."""
        # Create DataFrame from sample candidates
        df = pd.DataFrame([vars(c) for c in self.sample_candidates])
        
        # Test the optimal threshold analysis
        results = self.command._analyze_optimal_thresholds(df)
        
        # Verify results structure
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        
        # Check that results are sorted by score (descending)
        scores = [r['score'] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        
        # Verify result structure
        first_result = results[0]
        required_keys = [
            'cut_threshold', 'flip_threshold', 'win_rate', 
            'flip_count', 'hold_count', 'cut_count', 'score'
        ]
        for key in required_keys:
            self.assertIn(key, first_result)
            
        # Verify threshold constraints
        for result in results:
            self.assertLess(result['cut_threshold'], result['flip_threshold'])
            self.assertGreaterEqual(result['win_rate'], 0)
            self.assertLessEqual(result['win_rate'], 100)

    def test_mfe_distribution_calculations(self):
        """Test MFE distribution statistical calculations."""
        recovery_values = [c.recovery_return for c in self.sample_candidates]
        
        # Test percentile calculations
        p25 = np.percentile(recovery_values, 25)
        p50 = np.percentile(recovery_values, 50)
        p75 = np.percentile(recovery_values, 75)
        
        self.assertLessEqual(p25, p50)
        self.assertLessEqual(p50, p75)
        
        # Test quartile calculations
        iqr = p75 - p25
        self.assertGreaterEqual(iqr, 0)
        
        # Test basic statistics
        mean_mfe = np.mean(recovery_values)
        std_mfe = np.std(recovery_values)
        min_mfe = np.min(recovery_values)
        max_mfe = np.max(recovery_values)
        
        self.assertGreaterEqual(mean_mfe, min_mfe)
        self.assertLessEqual(mean_mfe, max_mfe)
        self.assertGreaterEqual(std_mfe, 0)

    @patch('signals.management.commands.analyze_loser_recovery.Command._build_trades_df')
    @patch('signals.management.commands.analyze_loser_recovery.Command._load_prices')
    @patch('signals.management.commands.analyze_loser_recovery.Command._analyze_recovery')
    @patch('signals.management.commands.analyze_loser_recovery.Path.exists', return_value=True)
    def test_command_integration(self, mock_path_exists, mock_analyze, mock_prices, mock_trades):
        """Test the command integration with enhanced MFE analysis."""
        # Mock the dependencies
        mock_trades.return_value = pd.DataFrame([
            {'date': '2022-01-01', 'type': 'MVRV_SHORT', 'direction': 'SHORT'}
        ])
        mock_prices.return_value = pd.DataFrame({
            'btc_close': [50000, 47000, 48000],
            'btc_high': [51000, 48000, 49000],
            'btc_low': [49000, 46000, 47000]
        }, index=pd.date_range('2022-01-01', periods=3))
        mock_analyze.return_value = self.sample_candidates
        
        # Capture output
        out = StringIO()
        
        # Run command
        call_command('analyze_loser_recovery', '--type', 'MVRV_SHORT', stdout=out)
        
        output = out.getvalue()
        
        # Verify enhanced MFE analysis sections are present
        self.assertIn("RECOVERY MFE DISTRIBUTION ANALYSIS", output)
        self.assertIn("OPTIMAL THRESHOLD ANALYSIS", output)
        self.assertIn("Quartile Analysis:", output)
        self.assertIn("Additional Statistics:", output)
        self.assertIn("THRESHOLD RECOMMENDATIONS:", output)
        self.assertIn("Data-Driven Insights:", output)

    def test_threshold_performance_scoring(self):
        """Test the threshold performance scoring methodology."""
        df = pd.DataFrame([vars(c) for c in self.sample_candidates])
        results = self.command._analyze_optimal_thresholds(df)
        
        # Test that scoring methodology is working
        for result in results:
            # Score should be based on win rate, cut rate, and flip rate balance
            expected_base_score = result['win_rate']
            cut_bonus = min(result['cut_rate'] * 0.5, 10)
            flip_penalty = abs(result['flip_rate'] - 40.0) * 0.1
            expected_score = expected_base_score + cut_bonus - flip_penalty
            
            # Allow small floating point differences
            self.assertAlmostEqual(result['score'], expected_score, places=2)


if __name__ == '__main__':
    unittest.main()