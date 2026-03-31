"""
Unit tests for signals/mvrv_short.py

Tests the MVRV short signal logic for bear market tactical shorts.
"""
import math
from django.test import SimpleTestCase
import pandas as pd

from .mvrv_short import (
    MVRV_7D_THRESHOLD,
    MVRV_60D_THRESHOLD,
    BEAR_MODE_START,
    BEAR_MODE_END,
    TARGET_PCT,
    WINDOW_DAYS,
    INITIAL_SIZE_PCT,
    DCA_SIZE_PCT,
    is_bear_cycle_window,
    check_mvrv_short_signal,
    get_signal_summary,
    MvrvShortSignal,
)


def make_row(**kwargs) -> pd.Series:
    """Helper to create a feature row with defaults."""
    defaults = {
        'cycle_days_since_halving': 700,  # In bear mode by default
        'mvrv_usd_7d': 1.05,
        'mvrv_usd_60d': 1.02,
        'btc_close': 50000,
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


class TestIsBearCycleWindow(SimpleTestCase):
    """Test is_bear_cycle_window function."""
    
    def test_in_bear_window(self):
        """Cycle day 700 is in bear window (540-900)."""
        self.assertTrue(is_bear_cycle_window(700))
    
    def test_at_bear_start(self):
        """Cycle day 540 is at bear window start."""
        self.assertTrue(is_bear_cycle_window(540))
    
    def test_at_bear_end(self):
        """Cycle day 900 is at bear window end."""
        self.assertTrue(is_bear_cycle_window(900))
    
    def test_before_bear_window(self):
        """Cycle day 539 is before bear window."""
        self.assertFalse(is_bear_cycle_window(539))
    
    def test_after_bear_window(self):
        """Cycle day 901 is after bear window."""
        self.assertFalse(is_bear_cycle_window(901))
    
    def test_none_cycle_day(self):
        """None cycle day returns False."""
        self.assertFalse(is_bear_cycle_window(None))
    
    def test_nan_cycle_day(self):
        """NaN cycle day returns False."""
        self.assertFalse(is_bear_cycle_window(float('nan')))


class TestCheckMvrvShortSignal(SimpleTestCase):
    """Test check_mvrv_short_signal function."""
    
    def test_signal_active_all_conditions_met(self):
        """Signal active when all conditions met."""
        row = make_row(
            cycle_days_since_halving=700,
            mvrv_usd_7d=1.05,
            mvrv_usd_60d=1.02,
            btc_close=50000,
        )
        signal = check_mvrv_short_signal(row)
        
        self.assertTrue(signal.active)
        self.assertTrue(signal.is_bear_mode)
        self.assertEqual(signal.cycle_day, 700)
        self.assertEqual(signal.mvrv_7d, 1.05)
        self.assertEqual(signal.mvrv_60d, 1.02)
        self.assertEqual(signal.btc_price, 50000)
        self.assertEqual(signal.target_price, 48000)  # 50000 * 0.96
        self.assertEqual(signal.dca_trigger_price, 52000)  # 50000 * 1.04
    
    def test_signal_inactive_not_bear_mode(self):
        """Signal inactive when not in bear mode."""
        row = make_row(cycle_days_since_halving=400)
        signal = check_mvrv_short_signal(row)
        
        self.assertFalse(signal.active)
        self.assertFalse(signal.is_bear_mode)
        self.assertIn("Not in bear mode", signal.reason)
    
    def test_signal_inactive_mvrv_7d_below_threshold(self):
        """Signal inactive when MVRV 7d below threshold."""
        row = make_row(mvrv_usd_7d=1.01)  # Below 1.02
        signal = check_mvrv_short_signal(row)
        
        self.assertFalse(signal.active)
        self.assertTrue(signal.is_bear_mode)
        self.assertIn("MVRV 7d below threshold", signal.reason)
    
    def test_signal_inactive_mvrv_60d_below_threshold(self):
        """Signal inactive when MVRV 60d below breakeven."""
        row = make_row(mvrv_usd_60d=0.98)  # Below 1.0
        signal = check_mvrv_short_signal(row)
        
        self.assertFalse(signal.active)
        self.assertTrue(signal.is_bear_mode)
        self.assertIn("MVRV 60d below breakeven", signal.reason)
    
    def test_signal_inactive_missing_mvrv_7d(self):
        """Signal inactive when MVRV 7d is missing."""
        row = make_row(mvrv_usd_7d=None)
        signal = check_mvrv_short_signal(row)
        
        self.assertFalse(signal.active)
        self.assertIn("Missing MVRV 7d", signal.reason)
    
    def test_signal_inactive_missing_mvrv_60d(self):
        """Signal inactive when MVRV 60d is missing."""
        row = make_row(mvrv_usd_60d=None)
        signal = check_mvrv_short_signal(row)
        
        self.assertFalse(signal.active)
        self.assertIn("Missing MVRV 60d", signal.reason)
    
    def test_signal_at_exact_thresholds(self):
        """Signal active at exact threshold values."""
        row = make_row(
            mvrv_usd_7d=1.02,  # Exactly at threshold
            mvrv_usd_60d=1.0,  # Exactly at breakeven
        )
        signal = check_mvrv_short_signal(row)
        
        self.assertTrue(signal.active)
    
    def test_custom_thresholds(self):
        """Custom thresholds are respected."""
        row = make_row(
            mvrv_usd_7d=1.04,
            mvrv_usd_60d=1.03,
        )
        
        # With default thresholds, should be active
        signal = check_mvrv_short_signal(row)
        self.assertTrue(signal.active)
        
        # With higher thresholds, should be inactive
        signal = check_mvrv_short_signal(row, mvrv_7d_threshold=1.05)
        self.assertFalse(signal.active)
        
        signal = check_mvrv_short_signal(row, mvrv_60d_threshold=1.05)
        self.assertFalse(signal.active)
    
    def test_nan_values_handled(self):
        """NaN values are handled as missing."""
        row = make_row(mvrv_usd_7d=float('nan'))
        signal = check_mvrv_short_signal(row)
        
        self.assertFalse(signal.active)
        self.assertIn("Missing MVRV 7d", signal.reason)
    
    def test_alternate_column_names(self):
        """Alternate column names (mvrv_7d, mvrv_60d) are supported."""
        row = pd.Series({
            'cycle_days_since_halving': 700,
            'mvrv_7d': 1.05,  # Alternate name
            'mvrv_60d': 1.02,  # Alternate name
            'btc_close': 50000,
        })
        signal = check_mvrv_short_signal(row)
        
        self.assertTrue(signal.active)


class TestGetSignalSummary(SimpleTestCase):
    """Test get_signal_summary function."""
    
    def test_active_signal_summary(self):
        """Active signal includes execution and backtest stats."""
        signal = MvrvShortSignal(
            active=True,
            is_bear_mode=True,
            cycle_day=700,
            mvrv_7d=1.05,
            mvrv_60d=1.02,
            btc_price=50000,
            target_price=48000,
            dca_trigger_price=52000,
            reason="Test reason",
        )
        summary = get_signal_summary(signal)
        
        self.assertTrue(summary['active'])
        self.assertIsNotNone(summary['execution'])
        self.assertEqual(summary['execution']['initial_size_pct'], INITIAL_SIZE_PCT)
        self.assertEqual(summary['execution']['dca_size_pct'], DCA_SIZE_PCT)
        self.assertIsNotNone(summary['backtest_stats'])
        self.assertEqual(summary['backtest_stats']['win_rate'], 0.58)
    
    def test_inactive_signal_summary(self):
        """Inactive signal has no execution or backtest stats."""
        signal = MvrvShortSignal(
            active=False,
            is_bear_mode=False,
            cycle_day=400,
            mvrv_7d=1.05,
            mvrv_60d=1.02,
            btc_price=50000,
            target_price=48000,
            dca_trigger_price=52000,
            reason="Not in bear mode",
        )
        summary = get_signal_summary(signal)
        
        self.assertFalse(summary['active'])
        self.assertIsNone(summary['execution'])
        self.assertIsNone(summary['backtest_stats'])


class TestConstants(SimpleTestCase):
    """Test that constants are set correctly."""
    
    def test_default_thresholds(self):
        """Default thresholds match documented values."""
        self.assertEqual(MVRV_7D_THRESHOLD, 1.02)
        self.assertEqual(MVRV_60D_THRESHOLD, 1.0)
    
    def test_bear_mode_window(self):
        """Bear mode window is 540-900."""
        self.assertEqual(BEAR_MODE_START, 540)
        self.assertEqual(BEAR_MODE_END, 900)
    
    def test_execution_params(self):
        """Execution parameters match documented values."""
        self.assertEqual(TARGET_PCT, 4.0)
        self.assertEqual(WINDOW_DAYS, 5)
        self.assertEqual(INITIAL_SIZE_PCT, 33)
        self.assertEqual(DCA_SIZE_PCT, 67)
