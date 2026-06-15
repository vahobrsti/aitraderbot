"""
Tests for signals/income_gate.py — scoring, vetoes, and MVRV boundaries.

Covers:
- Bull put scoring components and max score
- Bear call scoring components and max score
- Bull put veto conditions (each individually)
- Bear call veto conditions (each individually)
- compute_strike_boundaries floor/ceiling logic
- normalize_chain_columns (happy path + missing columns)
"""
import pandas as pd
from django.test import SimpleTestCase

from .income_gate import (
    IncomeGateConfig,
    compute_bull_put_score,
    compute_bear_call_score,
    check_bull_put_vetoes,
    check_bear_call_vetoes,
    compute_strike_boundaries,
    normalize_chain_columns,
)


def make_row(**kwargs) -> pd.Series:
    """Helper to create a feature row with defaults of 0."""
    defaults = {
        # MDIA
        "mdia_regime_inflow": 0,
        "mdia_regime_strong_inflow": 0,
        "mdia_regime_aging": 0,
        # Whale
        "whale_regime_broad_accum": 0,
        "whale_regime_strategic_accum": 0,
        "whale_regime_distribution": 0,
        "whale_regime_distribution_strong": 0,
        # MVRV-LS
        "mvrv_ls_regime_call_confirm": 0,
        "mvrv_ls_regime_call_confirm_recovery": 0,
        "mvrv_ls_regime_call_confirm_trend": 0,
        "mvrv_ls_regime_put_confirm": 0,
        "mvrv_ls_regime_bear_continuation": 0,
        "mvrv_ls_regime_distribution_warning": 0,
        "mvrv_ls_early_rollover": 0,
        "mvrv_ls_weak_downtrend": 0,
        # Signals
        "signal_option_put": 0,
        "signal_option_call": 0,
        # Sentiment
        "sent_bucket_extreme_greed": 0,
        "sent_extreme_greed_persist_5d": 0,
        "sent_bucket_greed": 0,
        "sent_is_flattening": 0,
        # MVRV raw values
        "mvrv_60d": None,
        "mvrv_60d_p10_180d": None,
        "mvrv_60d_p90_180d": None,
        "mvrv_composite": None,
        "mvrv_composite_p90_180d": None,
        "mvrv_composite_pct": None,
        "mvrv_comp_max_180d": None,
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


# ============================================================
# BULL PUT SCORING
# ============================================================


class TestBullPutScore(SimpleTestCase):
    """Tests for compute_bull_put_score."""

    def test_perfect_score(self):
        """All components active → max score of 100."""
        row = make_row(
            mdia_regime_inflow=1,
            whale_regime_broad_accum=1,
            mvrv_ls_regime_call_confirm=1,
            signal_option_put=0,
            sent_bucket_extreme_greed=0,
            whale_regime_distribution=0,
            whale_regime_distribution_strong=0,
        )
        score, components = compute_bull_put_score(row)
        self.assertEqual(score, 100.0)
        self.assertEqual(components["mdia_inflow"], 25.0)
        self.assertEqual(components["whale_sponsored"], 20.0)
        self.assertEqual(components["mvrv_macro_bullish"], 20.0)
        self.assertEqual(components["no_option_put"], 15.0)
        self.assertEqual(components["sentiment_safe"], 10.0)
        self.assertEqual(components["whale_not_distributing"], 10.0)

    def test_zero_score(self):
        """No components active + conflicts → zero score."""
        row = make_row(
            mdia_regime_inflow=0,
            mdia_regime_strong_inflow=0,
            whale_regime_broad_accum=0,
            whale_regime_strategic_accum=0,
            mvrv_ls_regime_call_confirm=0,
            signal_option_put=1,  # blocks no_option_put
            sent_bucket_extreme_greed=1,  # blocks sentiment_safe
            whale_regime_distribution=1,  # blocks whale_not_distributing
        )
        score, components = compute_bull_put_score(row)
        self.assertEqual(score, 0.0)

    def test_strong_inflow_triggers_mdia(self):
        """mdia_regime_strong_inflow alone activates MDIA component."""
        row = make_row(mdia_regime_strong_inflow=1)
        score, components = compute_bull_put_score(row)
        self.assertEqual(components["mdia_inflow"], 25.0)

    def test_strategic_accum_triggers_whale(self):
        """whale_regime_strategic_accum alone activates whale component."""
        row = make_row(whale_regime_strategic_accum=1)
        score, components = compute_bull_put_score(row)
        self.assertEqual(components["whale_sponsored"], 20.0)

    def test_mvrv_recovery_triggers_bullish(self):
        """mvrv_ls_regime_call_confirm_recovery activates macro bullish."""
        row = make_row(mvrv_ls_regime_call_confirm_recovery=1)
        score, components = compute_bull_put_score(row)
        self.assertEqual(components["mvrv_macro_bullish"], 20.0)

    def test_distribution_strong_blocks_whale_not_distributing(self):
        """whale_regime_distribution_strong blocks the whale_not_distributing bonus."""
        row = make_row(whale_regime_distribution_strong=1)
        _, components = compute_bull_put_score(row)
        self.assertEqual(components["whale_not_distributing"], 0.0)


# ============================================================
# BEAR CALL SCORING
# ============================================================


class TestBearCallScore(SimpleTestCase):
    """Tests for compute_bear_call_score."""

    def test_perfect_score(self):
        """All components active → max score of 100."""
        row = make_row(
            mdia_regime_inflow=0,
            mdia_regime_strong_inflow=0,
            whale_regime_distribution=1,
            mvrv_ls_regime_put_confirm=1,
            signal_option_call=0,
            sent_bucket_greed=1,
            whale_regime_broad_accum=0,
            whale_regime_strategic_accum=0,
        )
        score, components = compute_bear_call_score(row)
        self.assertEqual(score, 100.0)
        self.assertEqual(components["no_mdia_inflow_or_aging"], 25.0)
        self.assertEqual(components["whale_distribution"], 20.0)
        self.assertEqual(components["mvrv_macro_bearish"], 20.0)
        self.assertEqual(components["no_option_call"], 15.0)
        self.assertEqual(components["sentiment_greed_or_flat"], 10.0)
        self.assertEqual(components["whale_not_accumulating"], 10.0)

    def test_zero_score(self):
        """All components inactive → zero score."""
        row = make_row(
            mdia_regime_inflow=1,
            mdia_regime_strong_inflow=1,
            whale_regime_distribution=0,
            whale_regime_distribution_strong=0,
            mvrv_ls_regime_put_confirm=0,
            signal_option_call=1,
            sent_bucket_greed=0,
            sent_is_flattening=0,
            whale_regime_broad_accum=1,
        )
        score, _ = compute_bear_call_score(row)
        self.assertEqual(score, 0.0)

    def test_aging_triggers_no_inflow(self):
        """mdia_regime_aging alone activates the no_mdia_inflow_or_aging component."""
        row = make_row(mdia_regime_inflow=1, mdia_regime_aging=1)
        _, components = compute_bear_call_score(row)
        # aging OR no_inflow — aging is present so component fires
        self.assertEqual(components["no_mdia_inflow_or_aging"], 25.0)

    def test_flattening_triggers_sentiment(self):
        """sent_is_flattening activates sentiment_greed_or_flat."""
        row = make_row(sent_is_flattening=1)
        _, components = compute_bear_call_score(row)
        self.assertEqual(components["sentiment_greed_or_flat"], 10.0)

    def test_bear_continuation_triggers_bearish(self):
        """mvrv_ls_regime_bear_continuation activates macro bearish."""
        row = make_row(mvrv_ls_regime_bear_continuation=1)
        _, components = compute_bear_call_score(row)
        self.assertEqual(components["mvrv_macro_bearish"], 20.0)


# ============================================================
# BULL PUT VETOES
# ============================================================


class TestBullPutVetoes(SimpleTestCase):
    """Tests for check_bull_put_vetoes."""

    def test_no_vetoes_clean_row(self):
        """Clean row with valid MVRV → no vetoes."""
        row = make_row(mvrv_60d=1.2)
        vetoes = check_bull_put_vetoes(row)
        self.assertEqual(vetoes, [])

    def test_mvrv_floor_invalid_when_none(self):
        """Missing MVRV-60d triggers MVRV_FLOOR_INVALID."""
        row = make_row(mvrv_60d=None)
        vetoes = check_bull_put_vetoes(row)
        self.assertIn("MVRV_FLOOR_INVALID", vetoes)

    def test_mvrv_floor_invalid_when_zero(self):
        """MVRV-60d = 0 triggers MVRV_FLOOR_INVALID."""
        row = make_row(mvrv_60d=0)
        vetoes = check_bull_put_vetoes(row)
        self.assertIn("MVRV_FLOOR_INVALID", vetoes)

    def test_mvrv_below_p10_band(self):
        """MVRV below P10 triggers MVRV_BELOW_P10_BAND."""
        row = make_row(mvrv_60d=0.8, mvrv_60d_p10_180d=0.9)
        vetoes = check_bull_put_vetoes(row)
        self.assertIn("MVRV_BELOW_P10_BAND", vetoes)

    def test_option_put_active(self):
        """Active put signal triggers OPTION_PUT_ACTIVE."""
        row = make_row(mvrv_60d=1.2, signal_option_put=1)
        vetoes = check_bull_put_vetoes(row)
        self.assertIn("OPTION_PUT_ACTIVE", vetoes)

    def test_extreme_greed_persist(self):
        """Persistent extreme greed triggers EXTREME_GREED_PERSIST_5D."""
        row = make_row(mvrv_60d=1.2, sent_extreme_greed_persist_5d=1)
        vetoes = check_bull_put_vetoes(row)
        self.assertIn("EXTREME_GREED_PERSIST_5D", vetoes)

    def test_mvrv_macro_bearish_put_confirm(self):
        """MVRV put confirm triggers MVRV_MACRO_BEARISH."""
        row = make_row(mvrv_60d=1.2, mvrv_ls_regime_put_confirm=1)
        vetoes = check_bull_put_vetoes(row)
        self.assertIn("MVRV_MACRO_BEARISH", vetoes)

    def test_mvrv_macro_bearish_early_rollover(self):
        """MVRV early rollover triggers MVRV_MACRO_BEARISH."""
        row = make_row(mvrv_60d=1.2, mvrv_ls_early_rollover=1)
        vetoes = check_bull_put_vetoes(row)
        self.assertIn("MVRV_MACRO_BEARISH", vetoes)

    def test_whale_distrib_strong(self):
        """Strong whale distribution triggers WHALE_DISTRIB_STRONG."""
        row = make_row(mvrv_60d=1.2, whale_regime_distribution_strong=1)
        vetoes = check_bull_put_vetoes(row)
        self.assertIn("WHALE_DISTRIB_STRONG", vetoes)

    def test_higher_priority_signal(self):
        """Higher priority flag triggers HIGHER_PRIORITY_SIGNAL."""
        row = make_row(mvrv_60d=1.2)
        vetoes = check_bull_put_vetoes(row, higher_priority_active=True)
        self.assertIn("HIGHER_PRIORITY_SIGNAL", vetoes)

    def test_atr_expansion(self):
        """ATR expansion above threshold triggers veto."""
        row = make_row(mvrv_60d=1.2)
        vetoes = check_bull_put_vetoes(row, atr_ratio=2.0)
        self.assertTrue(any("ATR_EXPANSION" in v for v in vetoes))

    def test_fusion_state_bearish(self):
        """Bearish fusion state triggers FUSION_STATE_BEARISH."""
        row = make_row(mvrv_60d=1.2)
        vetoes = check_bull_put_vetoes(row, fusion_state="bear_continuation")
        self.assertIn("FUSION_STATE_BEARISH", vetoes)

    def test_condor_precedence(self):
        """Condor eligibility triggers CONDOR_PRECEDENCE."""
        row = make_row(mvrv_60d=1.2)
        vetoes = check_bull_put_vetoes(row, condor_eligible=True)
        self.assertIn("CONDOR_PRECEDENCE", vetoes)


# ============================================================
# BEAR CALL VETOES
# ============================================================


class TestBearCallVetoes(SimpleTestCase):
    """Tests for check_bear_call_vetoes."""

    def test_no_vetoes_clean_row(self):
        """Clean row → no vetoes."""
        row = make_row(mvrv_composite=1.2, mvrv_composite_p90_180d=1.5)
        vetoes = check_bear_call_vetoes(row)
        self.assertEqual(vetoes, [])

    def test_mvrv_ceiling_invalid(self):
        """P90 <= current composite triggers MVRV_CEILING_INVALID."""
        row = make_row(mvrv_composite=1.5, mvrv_composite_p90_180d=1.4)
        vetoes = check_bear_call_vetoes(row)
        self.assertIn("MVRV_CEILING_INVALID", vetoes)

    def test_mvrv_above_p90_band(self):
        """MVRV-60d above P90 triggers MVRV_ABOVE_P90_BAND."""
        row = make_row(mvrv_60d=2.0, mvrv_60d_p90_180d=1.8)
        vetoes = check_bear_call_vetoes(row)
        self.assertIn("MVRV_ABOVE_P90_BAND", vetoes)

    def test_option_call_active(self):
        """Active call signal triggers OPTION_CALL_ACTIVE."""
        row = make_row(signal_option_call=1)
        vetoes = check_bear_call_vetoes(row)
        self.assertIn("OPTION_CALL_ACTIVE", vetoes)

    def test_mdia_strong_inflow(self):
        """Strong MDIA inflow triggers MDIA_STRONG_INFLOW."""
        row = make_row(mdia_regime_strong_inflow=1)
        vetoes = check_bear_call_vetoes(row)
        self.assertIn("MDIA_STRONG_INFLOW", vetoes)

    def test_mvrv_macro_bullish(self):
        """MVRV call confirm triggers MVRV_MACRO_BULLISH."""
        row = make_row(mvrv_ls_regime_call_confirm=1)
        vetoes = check_bear_call_vetoes(row)
        self.assertIn("MVRV_MACRO_BULLISH", vetoes)

    def test_whale_sponsored(self):
        """Whale accumulation triggers WHALE_SPONSORED."""
        row = make_row(whale_regime_broad_accum=1)
        vetoes = check_bear_call_vetoes(row)
        self.assertIn("WHALE_SPONSORED", vetoes)

    def test_higher_priority_signal(self):
        """Higher priority flag triggers HIGHER_PRIORITY_SIGNAL."""
        row = make_row()
        vetoes = check_bear_call_vetoes(row, higher_priority_active=True)
        self.assertIn("HIGHER_PRIORITY_SIGNAL", vetoes)

    def test_atr_expansion(self):
        """ATR expansion above threshold triggers veto."""
        row = make_row()
        vetoes = check_bear_call_vetoes(row, atr_ratio=2.0)
        self.assertTrue(any("ATR_EXPANSION" in v for v in vetoes))

    def test_fusion_state_bullish(self):
        """Bullish fusion state triggers FUSION_STATE_BULLISH."""
        row = make_row()
        vetoes = check_bear_call_vetoes(row, fusion_state="strong_bullish")
        self.assertIn("FUSION_STATE_BULLISH", vetoes)

    def test_condor_precedence(self):
        """Condor eligibility triggers CONDOR_PRECEDENCE."""
        row = make_row()
        vetoes = check_bear_call_vetoes(row, condor_eligible=True)
        self.assertIn("CONDOR_PRECEDENCE", vetoes)


# ============================================================
# MVRV STRIKE BOUNDARIES
# ============================================================


class TestStrikeBoundaries(SimpleTestCase):
    """Tests for compute_strike_boundaries."""

    def test_buyers_profitable_floor(self):
        """MVRV >= 1 → floor = cost_basis = spot/mvrv (below spot)."""
        floor, ceiling = compute_strike_boundaries(
            spot_price=100000, mvrv_60d=1.25
        )
        # cost_basis = 100000 / 1.25 = 80000
        self.assertAlmostEqual(floor, 80000.0)
        self.assertIsNone(ceiling)

    def test_buyers_underwater_floor_with_p10(self):
        """MVRV < 1 → cost_basis > spot → floor = cost_basis × P10."""
        floor, ceiling = compute_strike_boundaries(
            spot_price=100000, mvrv_60d=0.8, mvrv_60d_p10_180d=0.7
        )
        # cost_basis = 100000 / 0.8 = 125000 (above spot)
        # floor = 125000 × 0.7 = 87500
        self.assertAlmostEqual(floor, 87500.0)

    def test_buyers_underwater_floor_without_p10(self):
        """MVRV < 1, no P10 → floor = cost_basis (raw fallback)."""
        floor, _ = compute_strike_boundaries(
            spot_price=100000, mvrv_60d=0.8
        )
        # cost_basis = 125000
        self.assertAlmostEqual(floor, 125000.0)

    def test_ceiling_with_p90(self):
        """P90 provided → ceiling = cost_basis × P90."""
        floor, ceiling = compute_strike_boundaries(
            spot_price=100000, mvrv_60d=1.2, mvrv_60d_p90_180d=1.5
        )
        # cost_basis = 100000 / 1.2 ≈ 83333.33
        # ceiling = 83333.33 × 1.5 = 125000
        self.assertAlmostEqual(ceiling, 125000.0, places=0)

    def test_no_mvrv_returns_none(self):
        """No MVRV data → both None."""
        floor, ceiling = compute_strike_boundaries(
            spot_price=100000, mvrv_60d=None
        )
        self.assertIsNone(floor)
        self.assertIsNone(ceiling)

    def test_zero_mvrv_returns_none(self):
        """MVRV = 0 → both None (division guard)."""
        floor, ceiling = compute_strike_boundaries(
            spot_price=100000, mvrv_60d=0
        )
        self.assertIsNone(floor)
        self.assertIsNone(ceiling)


# ============================================================
# CHAIN NORMALIZATION
# ============================================================


class TestNormalizeChainColumns(SimpleTestCase):
    """Tests for normalize_chain_columns."""

    def test_happy_path(self):
        """Standard DataFrame normalizes correctly."""
        df = pd.DataFrame({
            "strike": [95000, 90000],
            "option_type": ["Put", "Put"],
            "delta": [-0.25, -0.15],
            "bid": [500, 200],
            "ask": [550, 230],
            "days_to_expiry": [14, 14],
        })
        result = normalize_chain_columns(df)
        self.assertIsNotNone(result)
        self.assertIn("side", result.columns)
        self.assertIn("dte", result.columns)
        # side normalized to lowercase
        self.assertEqual(result["side"].iloc[0], "put")
        # delta converted to absolute
        self.assertTrue((result["delta"] >= 0).all())
        # spread_pct computed
        self.assertIn("spread_pct", result.columns)

    def test_missing_required_columns(self):
        """Missing required columns → returns None."""
        df = pd.DataFrame({"strike": [95000], "bid": [500]})
        result = normalize_chain_columns(df)
        self.assertIsNone(result)

    def test_empty_dataframe(self):
        """Empty DataFrame → returns None."""
        result = normalize_chain_columns(pd.DataFrame())
        self.assertIsNone(result)

    def test_none_input(self):
        """None input → returns None."""
        result = normalize_chain_columns(None)
        self.assertIsNone(result)
