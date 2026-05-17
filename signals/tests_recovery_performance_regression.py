"""
Performance regression tests for recovery policy analyzer - Task 7.2

Tests to ensure enhanced command maintains existing functionality, validates all
existing CLI arguments still work, and ensures output format compatibility.

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6
"""

import unittest
import time
import tempfile
import os
from unittest.mock import patch, MagicMock
from io import StringIO
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
from django.test import TestCase
from django.core.management import call_command
from django.core.management.base import CommandError

from signals.management.commands.analyze_loser_recovery import (
    Command, 
    RecoveryCandidate,
    PolicyConfigAdapter
)


class TestPerformanceRegression(TestCase):
    """Test performance regression for enhanced recovery analyzer command."""

    def setUp(self):
        """Set up test data and mock dependencies."""
        self.command = Command()
        
        # Create temporary CSV file for testing
        self.temp_dir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.temp_dir, "test_features.csv")
        
        # Create sample features CSV
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        features_df = pd.DataFrame({
            'btc_close': np.random.normal(50000, 5000, 100),
            'signal_option_call': np.random.choice([0, 1], 100, p=[0.9, 0.1]),
            'signal_option_put': np.random.choice([0, 1], 100, p=[0.9, 0.1]),
            # Add other required features
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100),
        }, index=dates)
        features_df.to_csv(self.csv_path)
        
        # Create temporary model files
        self.long_model_path = os.path.join(self.temp_dir, "long_model.joblib")
        self.short_model_path = os.path.join(self.temp_dir, "short_model.joblib")
        
        # Create mock models
        from sklearn.ensemble import RandomForestClassifier
        mock_model = RandomForestClassifier(n_estimators=10, random_state=42)
        mock_model.fit(np.random.randn(100, 3), np.random.choice([0, 1], 100))
        
        mock_bundle = {
            'model': mock_model,
            'feature_names': ['feature1', 'feature2', 'feature3']
        }
        
        joblib.dump(mock_bundle, self.long_model_path)
        joblib.dump(mock_bundle, self.short_model_path)
        
        # Create sample price data
        self.price_dates = pd.date_range('2023-01-01', periods=120, freq='D')
        self.price_df = pd.DataFrame({
            'btc_close': 50000 + np.cumsum(np.random.randn(120) * 100),
            'btc_high': 50000 + np.cumsum(np.random.randn(120) * 100) + 500,
            'btc_low': 50000 + np.cumsum(np.random.randn(120) * 100) - 500,
        }, index=self.price_dates)
        
        # Ensure prices are positive and high >= close >= low
        self.price_df['btc_close'] = np.abs(self.price_df['btc_close'])
        self.price_df['btc_high'] = np.maximum(self.price_df['btc_high'], self.price_df['btc_close'])
        self.price_df['btc_low'] = np.minimum(self.price_df['btc_low'], self.price_df['btc_close'])

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('signals.management.commands.analyze_loser_recovery.Command._load_prices')
    @patch('signals.management.commands.analyze_loser_recovery.Command._build_trades_df')
    def test_basic_command_execution_performance(self, mock_build_trades, mock_load_prices):
        """Test basic command execution maintains acceptable performance."""
        # Mock dependencies
        mock_build_trades.return_value = pd.DataFrame([
            {'date': '2023-01-01', 'type': 'CALL', 'direction': 'LONG'},
            {'date': '2023-01-02', 'type': 'PUT', 'direction': 'SHORT'},
            {'date': '2023-01-03', 'type': 'MVRV_SHORT', 'direction': 'SHORT'},
        ])
        mock_load_prices.return_value = self.price_df
        
        # Measure execution time
        start_time = time.time()
        
        out = StringIO()
        call_command(
            'analyze_loser_recovery',
            '--csv', self.csv_path,
            '--long-model', self.long_model_path,
            '--short-model', self.short_model_path,
            stdout=out
        )
        
        execution_time = time.time() - start_time
        
        # Performance assertion - should complete within reasonable time
        # (generous limit to avoid CI flakiness on slow runners)
        self.assertLess(execution_time, 120.0, "Command should complete within 120 seconds")
        
        # Verify output was generated
        output = out.getvalue()
        self.assertGreater(len(output), 0, "Command should produce output")

    @patch('signals.management.commands.analyze_loser_recovery.Command._load_prices')
    @patch('signals.management.commands.analyze_loser_recovery.Command._build_trades_df')
    def test_all_cli_arguments_compatibility(self, mock_build_trades, mock_load_prices):
        """Test all existing CLI arguments still work correctly."""
        # Mock dependencies
        mock_build_trades.return_value = pd.DataFrame([
            {'date': '2023-01-01', 'type': 'CALL', 'direction': 'LONG'},
            {'date': '2023-01-02', 'type': 'PUT', 'direction': 'SHORT'},
        ])
        mock_load_prices.return_value = self.price_df
        
        # Test all CLI argument combinations
        test_cases = [
            # Basic arguments
            {
                'args': ['--csv', self.csv_path, '--long-model', self.long_model_path, '--short-model', self.short_model_path],
                'description': 'Basic arguments'
            },
            # Year filter
            {
                'args': ['--csv', self.csv_path, '--year', '2023', '--long-model', self.long_model_path, '--short-model', self.short_model_path],
                'description': 'Year filter'
            },
            # Type filter
            {
                'args': ['--csv', self.csv_path, '--type', 'CALL', '--long-model', self.long_model_path, '--short-model', self.short_model_path],
                'description': 'Type filter'
            },
            # No overlay
            {
                'args': ['--csv', self.csv_path, '--no-overlay', '--long-model', self.long_model_path, '--short-model', self.short_model_path],
                'description': 'No overlay flag'
            },
            # No cooldown
            {
                'args': ['--csv', self.csv_path, '--no-cooldown', '--long-model', self.long_model_path, '--short-model', self.short_model_path],
                'description': 'No cooldown flag'
            },
            # Custom horizon
            {
                'args': ['--csv', self.csv_path, '--horizon', '21', '--long-model', self.long_model_path, '--short-model', self.short_model_path],
                'description': 'Custom horizon'
            },
            # Custom target
            {
                'args': ['--csv', self.csv_path, '--target', '0.08', '--long-model', self.long_model_path, '--short-model', self.short_model_path],
                'description': 'Custom target'
            },
            # Custom adverse threshold
            {
                'args': ['--csv', self.csv_path, '--adverse-threshold', '0.03', '--long-model', self.long_model_path, '--short-model', self.short_model_path],
                'description': 'Custom adverse threshold'
            },
            # Custom recovery target
            {
                'args': ['--csv', self.csv_path, '--recovery-target', '0.04', '--long-model', self.long_model_path, '--short-model', self.short_model_path],
                'description': 'Custom recovery target'
            },
            # New recovery thresholds
            {
                'args': ['--csv', self.csv_path, '--recovery-flip-threshold', '0.06', '--recovery-cut-threshold', '0.015', '--long-model', self.long_model_path, '--short-model', self.short_model_path],
                'description': 'Recovery thresholds'
            },
            # Checkpoint mode
            {
                'args': ['--csv', self.csv_path, '--checkpoint-mode', 'fixed', '--fixed-checkpoint', '5', '--long-model', self.long_model_path, '--short-model', self.short_model_path],
                'description': 'Fixed checkpoint mode'
            },
            # Show threshold impact
            {
                'args': ['--csv', self.csv_path, '--show-threshold-impact', '--long-model', self.long_model_path, '--short-model', self.short_model_path],
                'description': 'Show threshold impact'
            },
            # Simulate policy
            {
                'args': ['--csv', self.csv_path, '--simulate-policy', '--long-model', self.long_model_path, '--short-model', self.short_model_path],
                'description': 'Simulate policy'
            },
            # Sensitivity analysis
            {
                'args': ['--csv', self.csv_path, '--sensitivity-analysis', '--long-model', self.long_model_path, '--short-model', self.short_model_path],
                'description': 'Sensitivity analysis'
            },
        ]
        
        for test_case in test_cases:
            with self.subTest(description=test_case['description']):
                out = StringIO()
                err = StringIO()
                
                try:
                    call_command('analyze_loser_recovery', *test_case['args'], stdout=out, stderr=err)
                    
                    # Verify no errors
                    error_output = err.getvalue()
                    self.assertEqual(error_output, "", f"No errors expected for {test_case['description']}")
                    
                    # Verify output was generated
                    output = out.getvalue()
                    self.assertGreater(len(output), 0, f"Output expected for {test_case['description']}")
                    
                except Exception as e:
                    self.fail(f"CLI argument test failed for {test_case['description']}: {e}")

    @patch('signals.management.commands.analyze_loser_recovery.Command._load_prices')
    @patch('signals.management.commands.analyze_loser_recovery.Command._build_trades_df')
    def test_output_format_compatibility(self, mock_build_trades, mock_load_prices):
        """Test output format maintains compatibility with existing expectations."""
        # Mock dependencies with known data that will generate recovery candidates
        mock_build_trades.return_value = pd.DataFrame([
            {'date': '2023-01-01', 'type': 'CALL', 'direction': 'LONG'},
            {'date': '2023-01-02', 'type': 'PUT', 'direction': 'SHORT'},
            {'date': '2023-01-03', 'type': 'MVRV_SHORT', 'direction': 'SHORT'},
        ])
        
        # Create price data that will generate adverse trades (declining for LONG, rising for SHORT)
        adverse_price_df = pd.DataFrame({
            'btc_close': [50000, 47000, 44000, 41000, 38000, 35000, 32000, 29000, 26000, 23000, 20000, 18000, 16000, 14000, 12000] + [50000] * 105,
            'btc_high': [50500, 47500, 44500, 41500, 38500, 35500, 32500, 29500, 26500, 23500, 20500, 18500, 16500, 14500, 12500] + [50500] * 105,
            'btc_low': [49500, 46500, 43500, 40500, 37500, 34500, 31500, 28500, 25500, 22500, 19500, 17500, 15500, 13500, 11500] + [49500] * 105,
        }, index=self.price_dates)
        
        mock_load_prices.return_value = adverse_price_df
        
        out = StringIO()
        call_command(
            'analyze_loser_recovery',
            '--csv', self.csv_path,
            '--long-model', self.long_model_path,
            '--short-model', self.short_model_path,
            stdout=out
        )
        
        output = out.getvalue()
        
        # Test that output is generated (even if no recovery candidates)
        self.assertGreater(len(output), 0, "Should generate some output")
        
        # Test expected output sections are present when there are recovery candidates
        # The command should generate recovery candidates with this adverse price data
        expected_sections = [
            "RECOVERY POLICY ANALYZER",
            "SUMMARY BY TRADE TYPE", 
            "OVERALL STATISTICS",
            "DECISION MATRIX"
        ]
        
        for section in expected_sections:
            self.assertIn(section, output, f"Expected section '{section}' should be present in output")
        
        # Test output format structure
        lines = output.split('\n')
        
        # Should have header sections
        header_lines = [line for line in lines if line.startswith('=')]
        self.assertGreater(len(header_lines), 0, "Should have header sections")
        
        # Should not have error messages
        error_indicators = ['Error', 'Exception', 'Traceback', 'Failed']
        for indicator in error_indicators:
            self.assertNotIn(indicator, output, f"Should not contain error indicator: {indicator}")

    @patch('signals.management.commands.analyze_loser_recovery.Command._load_prices')
    @patch('signals.management.commands.analyze_loser_recovery.Command._build_trades_df')
    def test_backward_compatibility_with_existing_functionality(self, mock_build_trades, mock_load_prices):
        """Test backward compatibility with existing analyzer functionality."""
        # Create realistic test data that would generate recovery candidates
        mock_build_trades.return_value = pd.DataFrame([
            {'date': '2023-01-01', 'type': 'CALL', 'direction': 'LONG'},
            {'date': '2023-01-05', 'type': 'PUT', 'direction': 'SHORT'},
            {'date': '2023-01-10', 'type': 'MVRV_SHORT', 'direction': 'SHORT'},
        ])
        
        # Create price data that will generate adverse trades
        adverse_price_df = self.price_df.copy()
        # Make first trade adverse (LONG with declining prices)
        adverse_price_df.loc['2023-01-01':'2023-01-08', 'btc_close'] = [50000, 48000, 47000, 46000, 45000, 44000, 43000, 42000]
        adverse_price_df.loc['2023-01-01':'2023-01-08', 'btc_high'] = [50500, 48500, 47500, 46500, 45500, 44500, 43500, 42500]
        adverse_price_df.loc['2023-01-01':'2023-01-08', 'btc_low'] = [49500, 47500, 46500, 45500, 44500, 43500, 42500, 41500]
        
        mock_load_prices.return_value = adverse_price_df
        
        out = StringIO()
        call_command(
            'analyze_loser_recovery',
            '--csv', self.csv_path,
            '--long-model', self.long_model_path,
            '--short-model', self.short_model_path,
            stdout=out
        )
        
        output = out.getvalue()
        
        # Test core functionality still works
        # 1. Recovery candidate identification
        self.assertIn("recovery candidates", output.lower(), "Should identify recovery candidates")
        
        # 2. Decision matrix calculation
        self.assertIn("decision matrix", output.lower(), "Should show decision matrix")
        
        # 3. Recommendation generation
        recommendation_keywords = ["flip", "hold", "cut"]
        has_recommendations = any(keyword in output.lower() for keyword in recommendation_keywords)
        self.assertTrue(has_recommendations, "Should generate recommendations")
        
        # 4. Statistical analysis
        self.assertIn("%", output, "Should show percentage statistics")
        
        # 5. Trade type breakdown
        self.assertIn("CALL", output, "Should show CALL trades")

    def test_error_handling_maintains_stability(self):
        """Test error handling maintains system stability."""
        # Test with non-existent CSV file
        out = StringIO()
        err = StringIO()
        
        try:
            call_command(
                'analyze_loser_recovery',
                '--csv', 'nonexistent.csv',
                '--long-model', self.long_model_path,
                '--short-model', self.short_model_path,
                stdout=out,
                stderr=err
            )
            # If no exception is raised, check that appropriate message is shown
            output = out.getvalue() + err.getvalue()
            self.assertIn("CSV not found", output, "Should show appropriate error message")
        except SystemExit:
            # Command may exit with error code, which is acceptable
            error_output = err.getvalue()
            self.assertIn("CSV not found", error_output, "Should show appropriate error message")
        except Exception as e:
            # Any other exception should contain appropriate error message
            self.assertIn("CSV", str(e), "Error should be related to CSV file")

    @patch('signals.management.commands.analyze_loser_recovery.Command._load_prices')
    @patch('signals.management.commands.analyze_loser_recovery.Command._build_trades_df')
    def test_memory_usage_regression(self, mock_build_trades, mock_load_prices):
        """Test memory usage doesn't regress with enhanced functionality."""
        try:
            import psutil
            import os
            
            # Create larger dataset to test memory usage
            large_trades_df = pd.DataFrame([
                {'date': f'2023-01-{i:02d}', 'type': 'CALL', 'direction': 'LONG'}
                for i in range(1, 32)  # 31 trades
            ])
            mock_build_trades.return_value = large_trades_df
            mock_load_prices.return_value = self.price_df
            
            # Measure memory before
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            out = StringIO()
            call_command(
                'analyze_loser_recovery',
                '--csv', self.csv_path,
                '--long-model', self.long_model_path,
                '--short-model', self.short_model_path,
                stdout=out
            )
            
            # Measure memory after
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = memory_after - memory_before
            
            # Memory increase should be reasonable (less than 500MB for this test)
            # Generous limit to avoid CI flakiness across different runner configurations
            self.assertLess(memory_increase, 500, "Memory usage should not increase excessively")
            
        except ImportError:
            # Skip test if psutil is not available
            self.skipTest("psutil not available for memory testing")

    @patch('signals.management.commands.analyze_loser_recovery.Command._load_prices')
    @patch('signals.management.commands.analyze_loser_recovery.Command._build_trades_df')
    def test_concurrent_execution_safety(self, mock_build_trades, mock_load_prices):
        """Test command can handle concurrent execution safely."""
        import threading
        import queue
        
        mock_build_trades.return_value = pd.DataFrame([
            {'date': '2023-01-01', 'type': 'CALL', 'direction': 'LONG'},
        ])
        mock_load_prices.return_value = self.price_df
        
        results = queue.Queue()
        errors = queue.Queue()
        
        def run_command():
            try:
                out = StringIO()
                call_command(
                    'analyze_loser_recovery',
                    '--csv', self.csv_path,
                    '--long-model', self.long_model_path,
                    '--short-model', self.short_model_path,
                    stdout=out
                )
                results.put(out.getvalue())
            except Exception as e:
                errors.put(str(e))
        
        # Run multiple threads concurrently
        threads = []
        for i in range(3):
            thread = threading.Thread(target=run_command)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=60)  # 60 second timeout
        
        # Check results
        self.assertEqual(errors.qsize(), 0, "No errors should occur during concurrent execution")
        self.assertEqual(results.qsize(), 3, "All threads should complete successfully")

    @patch('signals.management.commands.analyze_loser_recovery.Command._load_prices')
    @patch('signals.management.commands.analyze_loser_recovery.Command._build_trades_df')
    def test_large_dataset_performance(self, mock_build_trades, mock_load_prices):
        """Test performance with larger datasets."""
        # Create large dataset
        large_trades_df = pd.DataFrame([
            {
                'date': f'2023-{month:02d}-{day:02d}', 
                'type': np.random.choice(['CALL', 'PUT', 'MVRV_SHORT']), 
                'direction': np.random.choice(['LONG', 'SHORT'])
            }
            for month in range(1, 13) for day in range(1, 29)  # ~336 trades
        ])
        mock_build_trades.return_value = large_trades_df
        
        # Create corresponding large price dataset
        large_price_dates = pd.date_range('2023-01-01', periods=400, freq='D')
        large_price_df = pd.DataFrame({
            'btc_close': 50000 + np.cumsum(np.random.randn(400) * 100),
            'btc_high': 50000 + np.cumsum(np.random.randn(400) * 100) + 500,
            'btc_low': 50000 + np.cumsum(np.random.randn(400) * 100) - 500,
        }, index=large_price_dates)
        
        # Ensure valid price data
        large_price_df['btc_close'] = np.abs(large_price_df['btc_close'])
        large_price_df['btc_high'] = np.maximum(large_price_df['btc_high'], large_price_df['btc_close'])
        large_price_df['btc_low'] = np.minimum(large_price_df['btc_low'], large_price_df['btc_close'])
        
        mock_load_prices.return_value = large_price_df
        
        # Measure execution time
        start_time = time.time()
        
        out = StringIO()
        call_command(
            'analyze_loser_recovery',
            '--csv', self.csv_path,
            '--long-model', self.long_model_path,
            '--short-model', self.short_model_path,
            stdout=out
        )
        
        execution_time = time.time() - start_time
        
        # Should complete within reasonable time even with large dataset
        # (generous limit to avoid CI flakiness on slow runners)
        self.assertLess(execution_time, 300.0, "Large dataset should complete within 5 minutes")
        
        # Verify output was generated
        output = out.getvalue()
        self.assertGreater(len(output), 0, "Should produce output for large dataset")

    def test_policy_integration_backward_compatibility(self):
        """Test policy integration doesn't break existing functionality."""
        # Test PolicyConfigAdapter methods work correctly
        adapter = PolicyConfigAdapter()
        
        # Test checkpoint day retrieval
        checkpoint = adapter.get_checkpoint_day("CALL")
        self.assertIsInstance(checkpoint, int, "Checkpoint day should be integer")
        self.assertGreater(checkpoint, 0, "Checkpoint day should be positive")
        
        # Test adverse threshold retrieval
        threshold = adapter.get_adverse_threshold("CALL")
        self.assertIsInstance(threshold, float, "Adverse threshold should be float")
        self.assertGreater(threshold, 0, "Adverse threshold should be positive")
        
        # Test user override functionality
        override_threshold = adapter.get_adverse_threshold("CALL", override=0.025)
        self.assertEqual(override_threshold, 0.025, "User override should take precedence")

    @patch('signals.management.commands.analyze_loser_recovery.Command._load_prices')
    @patch('signals.management.commands.analyze_loser_recovery.Command._build_trades_df')
    def test_enhanced_features_dont_break_basic_functionality(self, mock_build_trades, mock_load_prices):
        """Test enhanced features don't interfere with basic functionality."""
        mock_build_trades.return_value = pd.DataFrame([
            {'date': '2023-01-01', 'type': 'CALL', 'direction': 'LONG'},
        ])
        
        # Create price data that will generate adverse trades
        adverse_price_df = pd.DataFrame({
            'btc_close': [50000, 47000, 44000, 41000, 38000, 35000, 32000, 29000, 26000, 23000, 20000, 18000, 16000, 14000, 12000] + [50000] * 105,
            'btc_high': [50500, 47500, 44500, 41500, 38500, 35500, 32500, 29500, 26500, 23500, 20500, 18500, 16500, 14500, 12500] + [50500] * 105,
            'btc_low': [49500, 46500, 43500, 40500, 37500, 34500, 31500, 28500, 25500, 22500, 19500, 17500, 15500, 13500, 11500] + [49500] * 105,
        }, index=self.price_dates)
        
        mock_load_prices.return_value = adverse_price_df
        
        # Test basic functionality still works with enhanced features enabled
        enhanced_args = [
            '--csv', self.csv_path,
            '--long-model', self.long_model_path,
            '--short-model', self.short_model_path,
            '--show-threshold-impact',
            '--simulate-policy',
            '--sensitivity-analysis',
            '--recovery-flip-threshold', '0.06',
            '--recovery-cut-threshold', '0.015'
        ]
        
        out = StringIO()
        call_command('analyze_loser_recovery', *enhanced_args, stdout=out)
        
        output = out.getvalue()
        
        # Test that command completes without errors
        self.assertGreater(len(output), 0, "Should produce output")
        
        # Should not have error messages
        error_indicators = ['Error', 'Exception', 'Traceback', 'Failed']
        for indicator in error_indicators:
            self.assertNotIn(indicator, output, f"Should not contain error indicator: {indicator}")
        
        # With adverse price data, recovery candidates should be found
        # Basic sections should be present
        basic_sections = [
            "RECOVERY POLICY ANALYZER",
            "SUMMARY BY TRADE TYPE",
            "DECISION MATRIX"
        ]
        
        for section in basic_sections:
            self.assertIn(section, output, f"Basic section '{section}' should still be present")
        
        # Enhanced sections should also be present
        enhanced_sections = [
            "THRESHOLD SENSITIVITY ANALYSIS",
            "POLICY BACKTESTING SIMULATION"
        ]
        
        for section in enhanced_sections:
            self.assertIn(section, output, f"Enhanced section '{section}' should be present")

    def test_data_validation_maintains_robustness(self):
        """Test data validation maintains robustness with enhanced features."""
        # Test RecoveryCandidate validation
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
            flip_threshold=0.05,
            cut_threshold=0.02
        )
        
        # Test recommendation logic
        recommendation = candidate.recommendation
        self.assertIn(recommendation, ["FLIP", "HOLD", "CUT"], "Should generate valid recommendation")
        
        # Test with edge case values
        edge_candidate = RecoveryCandidate(
            date="2023-01-01",
            trade_type="UNKNOWN",
            direction="LONG",
            entry_price=1.0,  # Very small price
            checkpoint_day=1,
            checkpoint_price=0.5,
            checkpoint_return=-0.5,
            remaining_days=1,
            forward_return=0.0,
            original_hit=False,
            recovery_hit=False,
            recovery_return=0.0,
            flip_threshold=0.05,
            cut_threshold=0.02
        )
        
        # Should handle edge cases gracefully
        edge_recommendation = edge_candidate.recommendation
        self.assertIn(edge_recommendation, ["FLIP", "HOLD", "CUT"], "Should handle edge cases")


if __name__ == '__main__':
    unittest.main()