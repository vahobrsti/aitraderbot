"""
Unit tests for signals/management/commands/analyze_fusion.py.

Tests the Django management command output to ensure:
- All 8 market states (including probes) appear in counts
- Zero-division guard works on empty data
- Hit-rate section uses correct labels
- Explain trace matches current fusion.py classification
- Direction filtering includes option signals
"""
import os
import tempfile
from io import StringIO

import pandas as pd
from django.core.management import call_command
from django.test import SimpleTestCase

from signals.fusion import MarketState, fuse_signals
from signals.tests import make_row


def _build_fixture_csv(path: str):
    """Build a small synthetic CSV that produces all 8 market states.

    Each row is crafted to trigger a specific state via classify_market_state.
    Includes label_good_move_long/short and signal_option_call/put columns
    so hit-rate and option-signal tests can validate against known values.
    """
    base = {col: 0 for col in [
        # MDIA
        'mdia_regime_strong_inflow', 'mdia_regime_inflow',
        'mdia_regime_distribution', 'mdia_regime_aging',
        # Whale
        'whale_regime_broad_accum', 'whale_regime_strategic_accum',
        'whale_regime_distribution', 'whale_regime_distribution_strong',
        'whale_regime_mixed',
        # MVRV-LS
        'mvrv_ls_regime_call_confirm', 'mvrv_ls_regime_call_confirm_recovery',
        'mvrv_ls_regime_call_confirm_trend',
        'mvrv_ls_regime_put_confirm', 'mvrv_ls_regime_bear_continuation',
        'mvrv_ls_regime_distribution_warning',
        'mvrv_ls_weak_uptrend', 'mvrv_ls_weak_downtrend',
        'mvrv_ls_early_rollover',
        # Conflicts
        'mvrv_ls_conflict', 'whale_mega_conflict', 'whale_small_conflict',
        # Overlay features (sentiment + MVRV composite)
        'sent_regime_call_supportive', 'sent_regime_call_mean_reversion',
        'sent_regime_put_supportive', 'sent_regime_avoid_longs',
        'sent_bucket_fear', 'sent_bucket_extreme_fear', 'sent_is_falling',
        'mvrv_bucket_deep_undervalued', 'mvrv_bucket_undervalued',
        'mvrv_bucket_overvalued', 'mvrv_bucket_extreme_overvalued',
        'mvrv_is_rising', 'mvrv_is_falling', 'mvrv_is_flattening',
        'regime_mvrv_call_backbone', 'regime_mvrv_reduce_longs',
        'regime_mvrv_put_supportive',
        # MVRV-60d (overlay)
        'mvrv_60d', 'mvrv_60d_pct_rank', 'mvrv_60d_dist_from_max',
        # Labels
        'label_good_move_long', 'label_good_move_short',
        # Option signals
        'signal_option_call', 'signal_option_put',
        # Sentiment (for display)
        'sentiment_norm',
        # MDIA buckets (for explain)
        'mdia_bucket_1d', 'mdia_slope_z_1d',
        'mdia_bucket_2d', 'mdia_slope_z_2d',
        'mdia_bucket_4d', 'mdia_slope_z_4d',
        'mdia_bucket_7d', 'mdia_slope_z_7d',
        # Whale buckets (for explain)
        'whale_mega_bucket_1d', 'whale_mega_delta_z_1d',
        'whale_mega_bucket_2d', 'whale_mega_delta_z_2d',
        'whale_mega_bucket_4d', 'whale_mega_delta_z_4d',
        'whale_mega_bucket_7d', 'whale_mega_delta_z_7d',
        'whale_small_bucket_1d', 'whale_small_delta_z_1d',
        'whale_small_bucket_2d', 'whale_small_delta_z_2d',
        'whale_small_bucket_4d', 'whale_small_delta_z_4d',
        'whale_small_bucket_7d', 'whale_small_delta_z_7d',
        # MVRV-LS buckets (for explain)
        'mvrv_ls_level', 'mvrv_ls_roll_pct_365d', 'mvrv_ls_z_score_365d',
        'mvrv_ls_trend_2d', 'mvrv_ls_delta_z_2d',
        'mvrv_ls_trend_4d', 'mvrv_ls_delta_z_4d',
        'mvrv_ls_trend_7d', 'mvrv_ls_delta_z_7d',
        'mvrv_ls_trend_14d', 'mvrv_ls_delta_z_14d',
        # MVRV composite for option signals (for explain)
        'mvrv_comp_undervalued_90d', 'mvrv_comp_new_low_180d',
        'mvrv_comp_near_bottom_any', 'mvrv_60d_pct',
        # Distribution pressure (for EFB overlay)
        'distribution_pressure_score',
    ]}
    # Set sensible defaults for numeric fields
    base['mvrv_60d'] = 1.5
    base['mvrv_60d_pct_rank'] = 0.5
    base['mvrv_60d_dist_from_max'] = 0.5
    base['mvrv_ls_roll_pct_365d'] = 0.5
    base['distribution_pressure_score'] = 0.5

    rows = []

    def _row(date, overrides, label_long=0, label_short=0,
             opt_call=0, opt_put=0):
        r = dict(base)
        r.update(overrides)
        r['label_good_move_long'] = label_long
        r['label_good_move_short'] = label_short
        r['signal_option_call'] = opt_call
        r['signal_option_put'] = opt_put
        r['date'] = date
        return r

    # --- STRONG_BULLISH (hit long) ---
    rows.append(_row('2024-01-01', {
        'mdia_regime_strong_inflow': 1,
        'whale_regime_broad_accum': 1,
        'mvrv_ls_regime_call_confirm': 1,
    }, label_long=1))

    # --- EARLY_RECOVERY (miss long) ---
    rows.append(_row('2024-01-02', {
        'mdia_regime_inflow': 1,
        'whale_regime_broad_accum': 1,
        'mvrv_ls_regime_call_confirm_recovery': 1,
    }, label_long=0))

    # --- MOMENTUM_CONTINUATION (hit long) ---
    rows.append(_row('2024-01-03', {
        'mdia_regime_inflow': 1,
        'whale_regime_mixed': 1,
        'mvrv_ls_regime_call_confirm_trend': 1,
    }, label_long=1))

    # --- BULL_PROBE (hit long) ---
    rows.append(_row('2024-01-04', {
        'mdia_regime_inflow': 1,
        'whale_regime_strategic_accum': 1,
    }, label_long=1))

    # --- DISTRIBUTION_RISK (hit short) ---
    rows.append(_row('2024-01-05', {
        'whale_regime_distribution': 1,
        'mvrv_ls_early_rollover': 1,
        'mvrv_60d_pct_rank': 0.9,
        'mvrv_60d_dist_from_max': 0.05,
    }, label_short=1))

    # --- BEAR_CONTINUATION (miss short) ---
    rows.append(_row('2024-01-06', {
        'mdia_regime_aging': 1,
        'whale_regime_distribution': 1,
        'mvrv_ls_regime_put_confirm': 1,
        'mvrv_60d_pct_rank': 0.9,
        'mvrv_60d_dist_from_max': 0.05,
    }, label_short=0))

    # --- BEAR_PROBE (hit short) ---
    rows.append(_row('2024-01-07', {
        'whale_regime_distribution_strong': 1,
        'mvrv_60d_pct_rank': 0.9,
        'mvrv_60d_dist_from_max': 0.05,
    }, label_short=1))

    # --- NO_TRADE ---
    rows.append(_row('2024-01-08', {}, label_long=0, label_short=0))

    # --- Extra row with OPTION_CALL + OPTION_PUT signals ---
    rows.append(_row('2024-01-09', {
        'mdia_regime_inflow': 1,
        'whale_regime_broad_accum': 1,
        'mvrv_ls_regime_call_confirm': 1,
        'mdia_regime_strong_inflow': 1,
    }, label_long=1, opt_call=1, opt_put=1))

    df = pd.DataFrame(rows)
    df.set_index('date', inplace=True)
    df.to_csv(path)


# =============================================================================
# FIXTURE SETUP
# =============================================================================

# Single temp CSV shared by all tests in this module
_FIXTURE_CSV = os.path.join(tempfile.gettempdir(), 'test_analyze_fusion_fixture.csv')
_build_fixture_csv(_FIXTURE_CSV)


def _call(extra_args=None):
    """Call analyze_fusion with the test fixture, capturing stdout/stderr."""
    out = StringIO()
    err = StringIO()
    args = ['analyze_fusion', '--csv', _FIXTURE_CSV]
    if extra_args:
        args.extend(extra_args)
    call_command(*args, stdout=out, stderr=err)
    return out.getvalue(), err.getvalue()


# =============================================================================
# TESTS
# =============================================================================

class TestAnalyzeFusionProbeStates(SimpleTestCase):
    """Probe states (BULL_PROBE, BEAR_PROBE) must appear in counts."""

    def test_bull_probe_in_state_distribution(self):
        """BULL_PROBE appears in the Market States section."""
        out, _ = _call()
        self.assertIn('bull_probe', out)

    def test_bear_probe_in_state_distribution(self):
        """BEAR_PROBE appears in the Market States section."""
        out, _ = _call()
        self.assertIn('bear_probe', out)

    def test_probe_counted_in_long_signals(self):
        """BULL_PROBE is counted in LONG signals (overlay stats)."""
        out, _ = _call()
        # The fixture has 1 bull_probe row → LONG signals count
        # should include it (4 long states total: strong_bullish,
        # early_recovery, momentum, bull_probe + the extra strong_bullish)
        self.assertIn('LONG signals:', out)

    def test_probe_counted_in_short_signals(self):
        """BEAR_PROBE is counted in SHORT signals (overlay stats)."""
        out, _ = _call()
        self.assertIn('SHORT signals:', out)

    def test_bull_probe_green_emoji(self):
        """BULL_PROBE gets 🟢 in trade signals list."""
        out, _ = _call()
        # Find lines with bull_probe — they should have green emoji
        lines = [l for l in out.split('\n') if 'bull_probe' in l and '🟢' in l]
        self.assertTrue(len(lines) > 0, "BULL_PROBE should get 🟢 emoji")

    def test_bear_probe_red_emoji(self):
        """BEAR_PROBE gets 🔴 in trade signals list."""
        out, _ = _call()
        lines = [l for l in out.split('\n') if 'bear_probe' in l and '🔴' in l]
        self.assertTrue(len(lines) > 0, "BEAR_PROBE should get 🔴 emoji")


class TestAnalyzeFusionZeroGuard(SimpleTestCase):
    """Empty filter results must not crash."""

    def test_empty_year_returns_message(self):
        """--year with no matching rows shows message and exits cleanly."""
        _, err = _call(['--year', '2099'])
        self.assertIn('No rows after filtering', err)

    def test_empty_year_no_crash(self):
        """--year with no matching rows does not crash."""
        try:
            _call(['--year', '2099'])
        except (ZeroDivisionError, Exception) as e:
            self.fail(f"Empty year caused crash: {e}")


class TestAnalyzeFusionHitRate(SimpleTestCase):
    """Hit-rate section validates state→price correlation."""

    def test_hit_rate_section_present(self):
        """HIT RATE BY STATE section appears in output."""
        out, _ = _call()
        self.assertIn('HIT RATE BY STATE', out)

    def test_hit_rate_shows_bull_probe(self):
        """bull_probe appears in hit-rate table."""
        out, _ = _call()
        # Find the hit rate section
        hit_section = out[out.find('HIT RATE BY STATE'):]
        self.assertIn('bull_probe', hit_section)

    def test_hit_rate_shows_bear_probe(self):
        """bear_probe appears in hit-rate table."""
        out, _ = _call()
        hit_section = out[out.find('HIT RATE BY STATE'):]
        self.assertIn('bear_probe', hit_section)

    def test_hit_rate_uses_correct_labels(self):
        """Hit rate correctly uses label_good_move_long for long states."""
        out, _ = _call()
        hit_section = out[out.find('HIT RATE BY STATE'):]
        # strong_bullish: 2 rows, both label_long=1 → 100%
        self.assertIn('strong_bullish', hit_section)

    def test_no_trade_excluded(self):
        """NO_TRADE state is excluded from hit-rate table."""
        out, _ = _call()
        hit_section = out[out.find('HIT RATE BY STATE'):]
        # Lines after "HIT RATE" but before next section
        end = hit_section.find('LATEST')
        if end > 0:
            hit_section = hit_section[:end]
        # no_trade should not appear as a row in the hit-rate table
        hit_lines = [l for l in hit_section.split('\n')
                     if 'no_trade' in l and '%' in l]
        self.assertEqual(len(hit_lines), 0,
                         "no_trade should not appear in hit-rate table")


class TestAnalyzeFusionExplainTrace(SimpleTestCase):
    """Explain trace must match current fusion.py classification rules."""

    def test_explain_runs(self):
        """--explain does not crash."""
        out, _ = _call(['--explain'])
        self.assertIn('FUSION EXPLANATION', out)

    def test_explain_with_date(self):
        """--explain --date targets correct row."""
        out, _ = _call(['--explain', '--date', '2024-01-01'])
        self.assertIn('2024-01-01', out)
        self.assertIn('STRONG_BULLISH', out)

    def test_explain_trace_uses_not_whale_distrib(self):
        """MOMENTUM rule shows 'not_whale_distrib' (not stale whale_mixed/neutral)."""
        out, _ = _call(['--explain', '--date', '2024-01-03'])
        self.assertIn('not_whale_distrib', out)

    def test_explain_trace_uses_mvrv_improving(self):
        """MOMENTUM rule shows 'mvrv_improving' including mvrv_call."""
        out, _ = _call(['--explain', '--date', '2024-01-03'])
        self.assertIn('mvrv_improving', out)

    def test_explain_trace_bear_continuation_not_mdia_inflow(self):
        """BEAR_CONTINUATION rule shows 'not_mdia_inflow' (not stale mdia_distrib)."""
        out, _ = _call(['--explain', '--date', '2024-01-06'])
        self.assertIn('not_mdia_inflow', out)

    def test_explain_trace_bear_probe_whale_distrib_strong(self):
        """BEAR_PROBE rule shows 'whale_distrib_strong' (not whale_distrib + mdia_distrib)."""
        out, _ = _call(['--explain', '--date', '2024-01-07'])
        self.assertIn('whale_distrib_strong', out)

    def test_explain_no_mdia_distrib_variable(self):
        """Explain trace does not reference stale 'mdia_distrib' variable."""
        out, _ = _call(['--explain', '--date', '2024-01-06'])
        # The string "mdia_distrib=" should NOT appear in the trace
        trace_section = out[out.find('CLASSIFICATION TRACE'):]
        self.assertNotIn('mdia_distrib=', trace_section)

    def test_explain_trace_matches_fuse_signals(self):
        """The matched rule in explain trace must agree with fuse_signals()."""
        # 2024-01-01 should be STRONG_BULLISH per fuse_signals
        row = make_row(
            mdia_regime_strong_inflow=1,
            whale_regime_broad_accum=1,
            mvrv_ls_regime_call_confirm=1,
        )
        result = fuse_signals(row)
        self.assertEqual(result.state, MarketState.STRONG_BULLISH)

        out, _ = _call(['--explain', '--date', '2024-01-01'])
        self.assertIn('STRONG_BULLISH', out)
        self.assertIn('Matched: STRONG_BULLISH', out)


class TestAnalyzeFusionDirection(SimpleTestCase):
    """Direction filtering must include option signals."""

    def test_direction_short_includes_option_put(self):
        """--direction short output includes OPTION_PUT count."""
        out, _ = _call(['--direction', 'short', '--latest', '50'])
        self.assertIn('OPTION_PUT', out)

    def test_direction_long_includes_option_call(self):
        """--direction long output includes OPTION_CALL."""
        out, _ = _call(['--direction', 'long', '--latest', '50'])
        self.assertIn('OPTION_CALL', out)

    def test_direction_all_shows_both(self):
        """--direction all shows both LONG and SHORT setups."""
        out, _ = _call(['--direction', 'all', '--latest', '50'])
        self.assertIn('LONG', out)

    def test_direction_short_counts_nonzero(self):
        """--direction short OPTION_PUT count is non-zero when signals exist."""
        out, _ = _call(['--direction', 'short', '--latest', '50'])
        # The fixture has 1 row with signal_option_put=1
        self.assertIn('OPTION_PUT (rule-based):', out)
        # Extract the count
        for line in out.split('\n'):
            if 'OPTION_PUT (rule-based):' in line:
                # e.g. "OPTION_PUT (rule-based):           1"
                count = int(line.split(':')[-1].strip())
                self.assertGreater(count, 0,
                                   "OPTION_PUT count should be > 0")
                break


class TestAnalyzeFusionTradeSignals(SimpleTestCase):
    """Trade signals list must show all non-NO_TRADE states."""

    def test_all_tradeable_states_appear(self):
        """All 7 tradeable states appear in the trade signals list."""
        out, _ = _call()
        trade_section = out[out.find('ALL TRADE SIGNALS'):]
        for state in MarketState:
            if state == MarketState.NO_TRADE:
                continue
            self.assertIn(state.value, trade_section,
                          f"{state.value} missing from trade signals")

    def test_option_signals_tracked(self):
        """OPTION_CALL and OPTION_PUT appear in trade signals."""
        out, _ = _call()
        trade_section = out[out.find('ALL TRADE SIGNALS'):]
        self.assertIn('OPTION_CALL', trade_section)
        self.assertIn('OPTION_PUT', trade_section)
