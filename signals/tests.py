"""
Comprehensive tests for signals/fusion.py and signals/overlays.py.

These tests ensure the critical trading logic (market state classification,
confidence scoring, and overlay edge/veto rules) work correctly.
"""
import math
from django.test import SimpleTestCase
import pandas as pd

from .fusion import (
    MarketState,
    Confidence,
    FusionResult,
    classify_market_state,
    classify_bear_market_state,
    fuse_signals,
)
from .overlays import (
    OverlayResult,
    compute_near_peak_score,
    compute_long_edge_overlay,
    compute_long_veto_overlay,
    compute_short_overlay,
    apply_overlays,
    adjust_confidence_with_overlay,
    get_size_multiplier,
    get_dte_multiplier,
)


def make_row(**kwargs) -> pd.Series:
    """Helper to create a feature row with defaults of 0."""
    defaults = {
        # MDIA regimes
        'mdia_regime_strong_inflow': 0,
        'mdia_regime_inflow': 0,
        # 'mdia_regime_distribution' is legacy alias for 'aging'
        'mdia_regime_distribution': 0,
        'mdia_regime_aging': 0,
        # Whale regimes
        'whale_regime_broad_accum': 0,
        'whale_regime_strategic_accum': 0,
        'whale_regime_distribution_strong': 0,
        'whale_regime_distribution': 0,
        'whale_regime_mixed': 0,
        # MVRV-LS regimes
        'mvrv_ls_regime_call_confirm': 0,
        'mvrv_ls_regime_call_confirm_recovery': 0,
        'mvrv_ls_regime_call_confirm_trend': 0,
        'mvrv_ls_regime_put_confirm': 0,
        'mvrv_ls_regime_bear_continuation': 0,
        'mvrv_ls_regime_distribution_warning': 0,
        'mvrv_ls_weak_uptrend': 0,
        'mvrv_ls_weak_downtrend': 0,
        'mvrv_ls_early_rollover': 0,
        # Conflicts
        'mvrv_ls_conflict': 0,
        'whale_mega_conflict': 0,
        'whale_small_conflict': 0,
        # Sentiment regimes (for overlays)
        'sent_regime_call_supportive': 0,
        'sent_regime_call_mean_reversion': 0,
        'sent_regime_put_supportive': 0,
        'sent_regime_avoid_longs': 0,
        'sent_bucket_fear': 0,
        'sent_bucket_extreme_fear': 0,
        'sent_is_falling': 0,
        # MVRV composite (for overlays)
        'mvrv_bucket_deep_undervalued': 0,
        'mvrv_bucket_undervalued': 0,
        'mvrv_bucket_overvalued': 0,
        'mvrv_bucket_extreme_overvalued': 0,
        'mvrv_is_rising': 0,
        'mvrv_is_falling': 0,
        'mvrv_is_flattening': 0,
        'regime_mvrv_call_backbone': 0,
        'regime_mvrv_reduce_longs': 0,
        'regime_mvrv_put_supportive': 0,
        # MVRV-60d (for short overlays)
        'mvrv_60d_pct_rank': 0.5,
        'mvrv_60d_dist_from_max': 0.5,
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


# =============================================================================
# FUSION TESTS
# =============================================================================

class TestMarketStateClassification(SimpleTestCase):
    """Test classify_market_state function for all 8 market states."""
    
    def test_strong_bullish(self):
        """STRONG_BULLISH: Strong MDIA inflow + whale sponsorship + MVRV call confirm."""
        row = make_row(
            mdia_regime_strong_inflow=1,
            whale_regime_broad_accum=1,
            mvrv_ls_regime_call_confirm=1,
        )
        self.assertEqual(classify_market_state(row), MarketState.STRONG_BULLISH)
    
    def test_strong_bullish_strategic_whale(self):
        """STRONG_BULLISH with strategic (not broad) whale accumulation."""
        row = make_row(
            mdia_regime_strong_inflow=1,
            whale_regime_strategic_accum=1,
            mvrv_ls_regime_call_confirm=1,
        )
        self.assertEqual(classify_market_state(row), MarketState.STRONG_BULLISH)
    
    def test_early_recovery(self):
        """EARLY_RECOVERY: MDIA inflow + whale sponsorship + MVRV recovery."""
        row = make_row(
            mdia_regime_inflow=1,
            whale_regime_broad_accum=1,
            mvrv_ls_regime_call_confirm_recovery=1,
        )
        self.assertEqual(classify_market_state(row), MarketState.EARLY_RECOVERY)
    
    def test_momentum_continuation(self):
        """MOMENTUM: MDIA inflow + whale neutral/mixed + MVRV improving."""
        row = make_row(
            mdia_regime_inflow=1,
            whale_regime_mixed=1,
            mvrv_ls_regime_call_confirm_trend=1,
        )
        self.assertEqual(classify_market_state(row), MarketState.MOMENTUM_CONTINUATION)
    
    def test_momentum_with_weak_uptrend_rejected(self):
        """MOMENTUM no longer accepts weak_uptrend (looser MVRV condition) per strict v1 macro rules."""
        row = make_row(
            mdia_regime_inflow=1,
            whale_regime_mixed=1,
            mvrv_ls_weak_uptrend=1,
        )
        self.assertEqual(classify_market_state(row), MarketState.NO_TRADE)
    
    def test_bull_probe(self):
        """BULL_PROBE: MDIA inflow + whale sponsorship + MVRV neutral."""
        row = make_row(
            mdia_regime_inflow=1,
            whale_regime_strategic_accum=1,
            # MVRV neutral (nothing set)
        )
        self.assertEqual(classify_market_state(row), MarketState.BULL_PROBE)
    
    def test_bear_continuation(self):
        """BEAR_CONTINUATION: No inflow + whale distrib + MVRV bear."""
        row = make_row(
            # Using aging (neutral) instead of inflow
            mdia_regime_aging=1,
            whale_regime_distribution=1,
            mvrv_ls_regime_put_confirm=1,
        )
        self.assertEqual(classify_market_state(row), MarketState.BEAR_CONTINUATION)
    
    def test_distribution_risk(self):
        """DISTRIBUTION_RISK: Whale distrib + MVRV rollover warning."""
        row = make_row(
            whale_regime_distribution=1,
            mvrv_ls_early_rollover=1,
        )
        self.assertEqual(classify_market_state(row), MarketState.DISTRIBUTION_RISK)
    
    def test_bear_probe(self):
        """BEAR_PROBE: MDIA aging + whale STRONG distrib + MVRV neutral."""
        row = make_row(
            mdia_regime_aging=1,  # Rising MDIA = Aging (Neutral)
            whale_regime_distribution_strong=1,
            # MVRV neutral (nothing set)
        )
        self.assertEqual(classify_market_state(row), MarketState.BEAR_PROBE)
    
    def test_no_trade_default(self):
        """NO_TRADE: No alignment (all features neutral)."""
        row = make_row()  # All defaults to 0
        self.assertEqual(classify_market_state(row), MarketState.NO_TRADE)
    
    def test_no_trade_conflicting_signals(self):
        """NO_TRADE: Conflicting bullish MDIA but bearish whales."""
        row = make_row(
            mdia_regime_inflow=1,
            whale_regime_distribution=1,
            # No MVRV confirmation
        )
        # This doesn't fit any bullish or bearish pattern cleanly
        self.assertEqual(classify_market_state(row), MarketState.NO_TRADE)
    
    def test_known_gap_fixed_inflow_sponsored_trend(self):
        """FIXED GAP: inflow + whale_sponsored + mvrv_trend → MOMENTUM."""
        row = make_row(
            mdia_regime_inflow=1,
            whale_regime_strategic_accum=1,
            mvrv_ls_regime_call_confirm_trend=1,
        )
        state = classify_market_state(row)
        self.assertEqual(state, MarketState.MOMENTUM_CONTINUATION)


class TestBearMarketStateClassification(SimpleTestCase):
    """Test classify_bear_market_state function for the 5 bear-specific market states."""
    
    def test_bear_exhaustion_long(self):
        """BEAR_EXHAUSTION_LONG: deep_underwater + inflow + NOT bearish macro."""
        row = make_row(
            mvrv_60d=0.80,  # deep_underwater (< 0.85)
            mdia_regime_inflow=1,
            mvrv_ls_regime_put_confirm=0,  # neutral macro
        )
        self.assertEqual(classify_bear_market_state(row), MarketState.BEAR_EXHAUSTION_LONG)

    def test_bear_rally_long(self):
        """BEAR_RALLY_LONG: underwater + inflow + neutral/bullish macro."""
        row = make_row(
            mvrv_60d=0.95,  # underwater (0.85-1.0)
            mdia_regime_inflow=1,
            mvrv_ls_regime_put_confirm=0,  # neutral macro
        )
        self.assertEqual(classify_bear_market_state(row), MarketState.BEAR_RALLY_LONG)

    def test_bear_continuation_short(self):
        """BEAR_CONTINUATION_SHORT: profitable + NO inflow + (aging or bearish macro)."""
        row = make_row(
            mvrv_60d=1.15,  # profitable (>= 1.1)
            mdia_regime_inflow=0,
            mdia_regime_aging=1,
        )
        self.assertEqual(classify_bear_market_state(row), MarketState.BEAR_CONTINUATION_SHORT)

    def test_late_distribution_short(self):
        """LATE_DISTRIBUTION_SHORT: breakeven + NO inflow + neutral macro."""
        row = make_row(
            mvrv_60d=1.05,  # breakeven (1.0-1.1)
            mdia_regime_inflow=0,
            mvrv_ls_regime_put_confirm=0,  # neutral macro
        )
        self.assertEqual(classify_bear_market_state(row), MarketState.LATE_DISTRIBUTION_SHORT)

    def test_transition_chop(self):
        """TRANSITION_CHOP: profitable + inflow."""
        row = make_row(
            mvrv_60d=1.15,  # profitable
            mdia_regime_inflow=1,  # conflicting inflow
        )
        self.assertEqual(classify_bear_market_state(row), MarketState.TRANSITION_CHOP)
        
    def test_no_trade_fallback(self):
        """NO_TRADE: unknown mvrv bucket or unhandled combo."""
        row = make_row(
            mvrv_60d=None,
        )
        self.assertEqual(classify_bear_market_state(row), MarketState.NO_TRADE)


class TestFuseSignals(SimpleTestCase):
    """Test fuse_signals integration."""
    
    def test_fuse_strong_bullish(self):
        """Integration test for STRONG_BULLISH with statically assigned HIGH confidence & score 5."""
        row = make_row(
            mdia_regime_strong_inflow=1,
            whale_regime_broad_accum=1,
            mvrv_ls_regime_call_confirm=1,
        )
        result = fuse_signals(row)
        self.assertEqual(result.state, MarketState.STRONG_BULLISH)
        self.assertEqual(result.confidence, Confidence.HIGH)
        self.assertEqual(result.score, 5)


# =============================================================================
# OVERLAY TESTS
# =============================================================================

class TestNearPeakScore(SimpleTestCase):
    """Test compute_near_peak_score blending."""
    
    def test_peak_score(self):
        """At peak: pct_rank=1, dist=0 -> score=1."""
        row = make_row(mvrv_60d_pct_rank=1.0, mvrv_60d_dist_from_max=0.0)
        self.assertAlmostEqual(compute_near_peak_score(row), 1.0)
    
    def test_trough_score(self):
        """At trough: pct_rank=0, dist=1 -> score=0."""
        row = make_row(mvrv_60d_pct_rank=0.0, mvrv_60d_dist_from_max=1.0)
        self.assertAlmostEqual(compute_near_peak_score(row), 0.0)
    
    def test_middle_score(self):
        """Middle: pct_rank=0.5, dist=0.5 -> score=0.5."""
        row = make_row(mvrv_60d_pct_rank=0.5, mvrv_60d_dist_from_max=0.5)
        self.assertAlmostEqual(compute_near_peak_score(row), 0.5)
    
    def test_missing_data_returns_none(self):
        """Missing data returns None."""
        row = make_row()
        row['mvrv_60d_pct_rank'] = None
        self.assertIsNone(compute_near_peak_score(row))
    
    def test_nan_returns_none(self):
        """NaN returns None."""
        row = make_row()
        row['mvrv_60d_pct_rank'] = float('nan')
        self.assertIsNone(compute_near_peak_score(row))


class TestLongEdgeOverlay(SimpleTestCase):
    """Test compute_long_edge_overlay."""
    
    def test_full_edge_sentiment_plus_mvrv(self):
        """Full edge when both sentiment fear and MVRV undervalued."""
        row = make_row(
            sent_regime_call_supportive=1,
            mvrv_bucket_undervalued=1,
            mvrv_is_falling=0,  # Not falling
        )
        active, reason = compute_long_edge_overlay(row)
        self.assertTrue(active)
        self.assertIn("FULL:", reason)
    
    def test_partial_edge_sentiment_only(self):
        """Partial edge when only sentiment is favorable."""
        row = make_row(
            sent_regime_call_mean_reversion=1,
        )
        active, reason = compute_long_edge_overlay(row)
        self.assertTrue(active)
        self.assertIn("PARTIAL:", reason)
    
    def test_partial_edge_mvrv_only(self):
        """Partial edge when only MVRV is favorable."""
        row = make_row(
            mvrv_bucket_deep_undervalued=1,
            mvrv_is_falling=0,
        )
        active, reason = compute_long_edge_overlay(row)
        self.assertTrue(active)
        self.assertIn("PARTIAL:", reason)
    
    def test_no_edge_neutral(self):
        """No edge when nothing favorable."""
        row = make_row()
        active, reason = compute_long_edge_overlay(row)
        self.assertFalse(active)
        self.assertEqual(reason, "")


class TestLongVetoOverlay(SimpleTestCase):
    """Test compute_long_veto_overlay."""
    
    def test_strong_veto_euphoria_plus_mvrv(self):
        """Strong veto when euphoria + MVRV overvalued rolling over."""
        row = make_row(
            sent_regime_avoid_longs=1,
            mvrv_bucket_overvalued=1,
            mvrv_is_falling=1,
        )
        active, reason = compute_long_veto_overlay(row)
        self.assertTrue(active)
        self.assertIn("STRONG VETO", reason)
    
    def test_moderate_veto_euphoria_only(self):
        """Moderate veto when just euphoria."""
        row = make_row(
            sent_regime_avoid_longs=1,
        )
        active, reason = compute_long_veto_overlay(row)
        self.assertTrue(active)
        self.assertIn("MODERATE VETO", reason)
    
    def test_moderate_veto_mvrv_extreme(self):
        """Moderate veto when MVRV extreme overvalued + falling."""
        row = make_row(
            mvrv_bucket_extreme_overvalued=1,
            mvrv_is_falling=1,
        )
        active, reason = compute_long_veto_overlay(row)
        self.assertTrue(active)
    
    def test_no_veto_neutral(self):
        """No veto when conditions not met."""
        row = make_row()
        active, reason = compute_long_veto_overlay(row)
        self.assertFalse(active)


class TestShortOverlay(SimpleTestCase):
    """Test compute_short_overlay thresholds."""
    
    def test_full_edge_above_085(self):
        """Full edge when score >= 0.85."""
        row = make_row(mvrv_60d_pct_rank=0.9, mvrv_60d_dist_from_max=0.05)
        edge, veto, reason = compute_short_overlay(row, is_confirmed_bear=False)
        self.assertEqual(edge, 2)
        self.assertEqual(veto, 0)
        self.assertIn("FULL EDGE", reason)
    
    def test_partial_edge_above_075(self):
        """Partial edge when 0.75 <= score < 0.85."""
        row = make_row(mvrv_60d_pct_rank=0.8, mvrv_60d_dist_from_max=0.2)
        edge, veto, reason = compute_short_overlay(row, is_confirmed_bear=False)
        self.assertEqual(edge, 1)
        self.assertEqual(veto, 0)
        self.assertIn("PARTIAL EDGE", reason)
    
    def test_hard_veto_below_025(self):
        """Hard veto when score <= 0.25."""
        row = make_row(mvrv_60d_pct_rank=0.2, mvrv_60d_dist_from_max=0.8)
        edge, veto, reason = compute_short_overlay(row, is_confirmed_bear=False)
        self.assertEqual(edge, 0)
        self.assertEqual(veto, 2)
        self.assertIn("HARD", reason)
    
    def test_soft_veto_distribution_risk(self):
        """Soft veto for DISTRIBUTION_RISK when 0.25 < score <= 0.35."""
        row = make_row(mvrv_60d_pct_rank=0.3, mvrv_60d_dist_from_max=0.7)
        edge, veto, reason = compute_short_overlay(row, is_confirmed_bear=False)
        self.assertEqual(edge, 0)
        self.assertEqual(veto, 1)
        self.assertIn("SOFT", reason)
    
    def test_no_soft_veto_bear_continuation(self):
        """Bear continuation doesn't get soft veto (only hard)."""
        row = make_row(mvrv_60d_pct_rank=0.3, mvrv_60d_dist_from_max=0.7)
        edge, veto, reason = compute_short_overlay(row, is_confirmed_bear=True)
        # Score ~0.3 is above hard veto (0.25) but below soft (0.35)
        # For confirmed bear, soft veto doesn't apply
        self.assertEqual(veto, 0)
    
    def test_missing_data_fail_closed_distribution(self):
        """Missing data = hard veto for DISTRIBUTION_RISK (fail-closed)."""
        row = make_row()
        row['mvrv_60d_pct_rank'] = None
        edge, veto, reason = compute_short_overlay(row, is_confirmed_bear=False)
        self.assertEqual(veto, 2)
    
    def test_missing_data_fail_open_bear(self):
        """Missing data = no veto for BEAR_CONTINUATION (fail-open)."""
        row = make_row()
        row['mvrv_60d_pct_rank'] = None
        edge, veto, reason = compute_short_overlay(row, is_confirmed_bear=True)
        self.assertEqual(veto, 0)


class TestApplyOverlays(SimpleTestCase):
    """Test apply_overlays integration."""
    
    def test_no_trade_no_overlay(self):
        """NO_TRADE state doesn't apply overlays."""
        fusion_result = FusionResult(
            state=MarketState.NO_TRADE,
            confidence=Confidence.LOW,
            score=0,
            components={},
        )
        row = make_row()
        overlay = apply_overlays(fusion_result, row)
        self.assertFalse(overlay.reduced_size)
    
    def test_long_state_with_edge(self):
        """Long state with favorable conditions gets edge boost."""
        fusion_result = FusionResult(
            state=MarketState.STRONG_BULLISH,
            confidence=Confidence.HIGH,
            score=6,
            components={},
        )
        row = make_row(
            sent_regime_call_supportive=1,
            mvrv_bucket_undervalued=1,
        )
        overlay = apply_overlays(fusion_result, row)
        self.assertGreater(overlay.edge_strength, 0)
    
    def test_long_state_with_strong_veto(self):
        """Strong veto overrides edge."""
        fusion_result = FusionResult(
            state=MarketState.MOMENTUM_CONTINUATION,
            confidence=Confidence.MEDIUM,
            score=3,
            components={},
        )
        row = make_row(
            sent_regime_avoid_longs=1,
            mvrv_bucket_overvalued=1,
            mvrv_is_falling=1,
        )
        overlay = apply_overlays(fusion_result, row)
        self.assertEqual(overlay.long_veto_strength, 2)
        self.assertTrue(overlay.reduced_size)
    
    def test_short_state_with_edge(self):
        """Short state near peak gets edge boost."""
        fusion_result = FusionResult(
            state=MarketState.DISTRIBUTION_RISK,
            confidence=Confidence.MEDIUM,
            score=-2,
            components={},
        )
        row = make_row(mvrv_60d_pct_rank=0.9, mvrv_60d_dist_from_max=0.05)
        overlay = apply_overlays(fusion_result, row)
        self.assertEqual(overlay.short_edge_strength, 2)
        self.assertTrue(overlay.extended_dte)


class TestSizeMultiplier(SimpleTestCase):
    """Test get_size_multiplier."""
    
    def test_hard_veto_zero_size(self):
        """Hard/strong veto returns 0 (no trade)."""
        overlay = OverlayResult(
            edge_strength=0,
            long_veto_strength=2,
            short_veto_strength=0,
            short_edge_strength=0,
            extended_dte=False,
            reduced_size=True,
            reason="test",
        )
        self.assertEqual(get_size_multiplier(overlay), 0.0)
    
    def test_soft_veto_half_size(self):
        """Soft/moderate veto returns 0.5."""
        overlay = OverlayResult(
            edge_strength=0,
            long_veto_strength=1,
            short_veto_strength=0,
            short_edge_strength=0,
            extended_dte=False,
            reduced_size=True,
            reason="test",
        )
        self.assertEqual(get_size_multiplier(overlay), 0.5)
    
    def test_full_edge_boost(self):
        """Full edge returns 1.25."""
        overlay = OverlayResult(
            edge_strength=2,
            long_veto_strength=0,
            short_veto_strength=0,
            short_edge_strength=0,
            extended_dte=True,
            reduced_size=False,
            reason="test",
        )
        self.assertEqual(get_size_multiplier(overlay), 1.25)
    
    def test_neutral_normal_size(self):
        """No overlay returns 1.0."""
        overlay = OverlayResult(
            edge_strength=0,
            long_veto_strength=0,
            short_veto_strength=0,
            short_edge_strength=0,
            extended_dte=False,
            reduced_size=False,
            reason="test",
        )
        self.assertEqual(get_size_multiplier(overlay), 1.0)


class TestDteMultiplier(SimpleTestCase):
    """Test get_dte_multiplier."""
    
    def test_long_full_edge_extends(self):
        """Full long edge extends DTE by 50%."""
        overlay = OverlayResult(
            edge_strength=2,
            long_veto_strength=0,
            short_veto_strength=0,
            short_edge_strength=0,
            extended_dte=True,
            reduced_size=False,
            reason="test",
        )
        self.assertEqual(get_dte_multiplier(overlay), 1.5)
    
    def test_short_full_edge_extends(self):
        """Full short edge extends DTE by 15%."""
        overlay = OverlayResult(
            edge_strength=0,
            long_veto_strength=0,
            short_veto_strength=0,
            short_edge_strength=2,
            extended_dte=True,
            reduced_size=False,
            reason="test",
        )
        self.assertEqual(get_dte_multiplier(overlay), 1.15)
    
    def test_hard_veto_shortens(self):
        """Hard veto shortens DTE by 25%."""
        overlay = OverlayResult(
            edge_strength=0,
            long_veto_strength=2,
            short_veto_strength=0,
            short_edge_strength=0,
            extended_dte=False,
            reduced_size=True,
            reason="test",
        )
        self.assertEqual(get_dte_multiplier(overlay), 0.75)


class TestConfidenceAdjustment(SimpleTestCase):
    """Test adjust_confidence_with_overlay."""
    
    def test_hard_veto_forces_low(self):
        """Hard veto forces confidence to LOW."""
        overlay = OverlayResult(
            edge_strength=0,
            long_veto_strength=2,
            short_veto_strength=0,
            short_edge_strength=0,
            extended_dte=False,
            reduced_size=True,
            reason="test",
        )
        result = adjust_confidence_with_overlay(Confidence.HIGH, overlay)
        self.assertEqual(result, Confidence.LOW)
    
    def test_full_edge_bumps_up(self):
        """Full edge bumps confidence up one notch."""
        overlay = OverlayResult(
            edge_strength=2,
            long_veto_strength=0,
            short_veto_strength=0,
            short_edge_strength=0,
            extended_dte=True,
            reduced_size=False,
            reason="test",
        )
        result = adjust_confidence_with_overlay(Confidence.MEDIUM, overlay)
        self.assertEqual(result, Confidence.HIGH)
    
    def test_no_overlay_no_change(self):
        """No overlay leaves confidence unchanged."""
        overlay = OverlayResult(
            edge_strength=0,
            long_veto_strength=0,
            short_veto_strength=0,
            short_edge_strength=0,
            extended_dte=False,
            reduced_size=False,
            reason="test",
        )
        result = adjust_confidence_with_overlay(Confidence.MEDIUM, overlay)
        self.assertEqual(result, Confidence.MEDIUM)


# =============================================================================
# OPTION SIGNAL (TRADE DECISION) TESTS
# =============================================================================

class TestOptionSignalTradeDecision(SimpleTestCase):
    """Test _determine_trade_decision for option signal fallback logic."""
    
    def _make_fusion_result(self, state=MarketState.NO_TRADE, score=0, confidence=Confidence.LOW, short_source=None):
        return FusionResult(
            state=state,
            confidence=confidence,
            score=score,
            components={},
            short_source=short_source,
        )
    
    def _make_tactical_inactive(self):
        from signals.tactical_puts import TacticalPutResult, TacticalPutStrategy
        return TacticalPutResult(
            active=False, strength=0, strategy=TacticalPutStrategy.NONE,
            size_mult=0.0, dte_mult=1.0, reason="",
        )
    
    def _make_overlay_clean(self):
        return OverlayResult(
            edge_strength=0, long_veto_strength=0, short_veto_strength=0,
            short_edge_strength=0, extended_dte=False,
            reduced_size=False, reason="",
        )
    
    def _make_overlay_vetoed(self):
        return OverlayResult(
            edge_strength=0, long_veto_strength=2, short_veto_strength=0,
            short_edge_strength=0, extended_dte=False,
            reduced_size=True, reason="STRONG VETO",
        )
    
    def _call_decision(self, fusion_result=None, size_mult=1.0, dte_mult=1.0,
                        tactical_result=None, p_long=0.5, p_short=0.5,
                        signal_option_call=0, signal_option_put=0,
                        overlay=None, option_call_ok=False, option_put_ok=False,
                        row=None):
        from signals.services import SignalService
        svc = SignalService.__new__(SignalService)  # bypass __init__
        if row is None:
            row = make_row(distribution_pressure_score=0.90)
        return svc._determine_trade_decision(
            fusion_result=fusion_result or self._make_fusion_result(),
            size_mult=size_mult,
            dte_mult=dte_mult,
            tactical_result=tactical_result or self._make_tactical_inactive(),
            p_long=p_long,
            p_short=p_short,
            signal_option_call=signal_option_call,
            signal_option_put=signal_option_put,
            overlay=overlay or self._make_overlay_clean(),
            row=row,
            option_call_ok=option_call_ok,
            option_put_ok=option_put_ok,
        )
    
    # --- OPTION_CALL tests ---
    
    def test_option_call_fires_on_no_trade(self):
        """OPTION_CALL fires when fusion=NO_TRADE, signal=1, cooldown OK, overlay clean."""
        decision, notes, reasons, trace = self._call_decision(
            signal_option_call=1, option_call_ok=True,
        )
        self.assertEqual(decision, "OPTION_CALL")
        self.assertIn("MVRV cheap", notes)
    
    def test_option_put_fires_on_no_trade(self):
        """OPTION_PUT fires when fusion=NO_TRADE, signal=1, cooldown OK, overlay clean."""
        decision, notes, reasons, trace = self._call_decision(
            signal_option_put=1, option_put_ok=True,
        )
        self.assertEqual(decision, "OPTION_PUT")
        self.assertIn("MVRV overheated", notes)
    
    # --- Fusion priority tests ---
    
    def test_fusion_call_takes_priority_over_option_call(self):
        """When fusion has a directional view, it takes priority over option signals."""
        fusion = self._make_fusion_result(
            state=MarketState.EARLY_RECOVERY, score=4, confidence=Confidence.HIGH,
        )
        decision, notes, reasons, trace = self._call_decision(
            fusion_result=fusion,
            signal_option_call=1, option_call_ok=True,
        )
        self.assertEqual(decision, "CALL")
        self.assertIn("early_recovery", notes)
    
    def test_fusion_put_takes_priority_over_option_put(self):
        """When fusion says short, it takes priority over option put signals."""
        fusion = self._make_fusion_result(
            state=MarketState.BEAR_CONTINUATION, score=-4, confidence=Confidence.LOW,
            short_source="rule",
        )
        decision, notes, reasons, trace = self._call_decision(
            fusion_result=fusion,
            signal_option_put=1, option_put_ok=True,
        )
        self.assertEqual(decision, "PUT")
    
    def test_bull_probe_takes_priority_over_option(self):
        """Bull probe (fusion directional) beats option signal."""
        fusion = self._make_fusion_result(
            state=MarketState.BULL_PROBE, score=2, confidence=Confidence.MEDIUM,
        )
        decision, _, _, _ = self._call_decision(
            fusion_result=fusion,
            signal_option_call=1, option_call_ok=True,
        )
        self.assertEqual(decision, "CALL")  # From bull probe, not OPTION_CALL
    
    def test_fusion_call_beats_tactical_put(self):
        """When fusion has a directional view (e.g., EARLY_RECOVERY), it beats tactical put."""
        from signals.tactical_puts import TacticalPutResult, TacticalPutStrategy
        fusion = self._make_fusion_result(
            state=MarketState.EARLY_RECOVERY, score=4, confidence=Confidence.HIGH,
        )
        tactical_active = TacticalPutResult(
            active=True, strength=2, strategy=TacticalPutStrategy.PUT_SPREAD,
            size_mult=0.60, dte_mult=0.85,
            reason="FULL: Bull regime + MVRV-60d near-peak & rolling over",
        )
        decision, notes, _, trace = self._call_decision(
            fusion_result=fusion,
            tactical_result=tactical_active,
        )
        self.assertEqual(decision, "CALL")
        self.assertIn("early_recovery", notes)
        # Verify trace shows tactical was noted but fusion took priority
        self.assertTrue(any("fusion takes priority" in t for t in trace))
    
    # --- Overlay veto tests ---
    
    def test_option_call_vetoed_by_overlay(self):
        """OPTION_CALL is blocked when overlay vetoes (size_mult=0)."""
        decision, notes, reasons, trace = self._call_decision(
            signal_option_call=1, option_call_ok=True, size_mult=0,
        )
        self.assertEqual(decision, "NO_TRADE")
        self.assertIn("OPTION_CALL_OVERLAY_VETO", reasons)
    
    def test_option_put_vetoed_by_overlay(self):
        """OPTION_PUT is blocked when overlay vetoes (size_mult=0)."""
        decision, notes, reasons, trace = self._call_decision(
            signal_option_put=1, option_put_ok=True, size_mult=0,
        )
        self.assertEqual(decision, "NO_TRADE")
        self.assertIn("OPTION_PUT_OVERLAY_VETO", reasons)
    
    # --- Cooldown tests ---
    
    def test_option_call_blocked_by_cooldown(self):
        """Option call signal fires but cooldown not passed → NO_TRADE."""
        decision, _, reasons, trace = self._call_decision(
            signal_option_call=1, option_call_ok=False,  # cooldown active
        )
        self.assertEqual(decision, "NO_TRADE")
        self.assertTrue(any("cooldown_active" in t for t in trace))
    
    def test_option_put_blocked_by_cooldown(self):
        """Option put signal fires but cooldown not passed → NO_TRADE."""
        decision, _, reasons, trace = self._call_decision(
            signal_option_put=1, option_put_ok=False,  # cooldown active
        )
        self.assertEqual(decision, "NO_TRADE")
        self.assertTrue(any("cooldown_active" in t for t in trace))
    
    # --- Priority between CALL and PUT ---
    
    def test_option_call_priority_over_put(self):
        """When both option signals fire, CALL takes priority."""
        decision, notes, _, _ = self._call_decision(
            signal_option_call=1, signal_option_put=1,
            option_call_ok=True, option_put_ok=True,
        )
        self.assertEqual(decision, "OPTION_CALL")
    
    # --- Constants ---
    
    def test_option_signal_size_mult(self):
        """Option signals use 0.75x sizing (boosted — 81% hit rate)."""
        from signals.services import OPTION_SIGNAL_SIZE_MULT
        self.assertEqual(OPTION_SIGNAL_SIZE_MULT, 0.75)
    
    def test_option_signal_cooldown_days(self):
        """Option signal cooldown is 5 days (reduced to capture more winners)."""
        from signals.services import OPTION_SIGNAL_COOLDOWN_DAYS
        self.assertEqual(OPTION_SIGNAL_COOLDOWN_DAYS, 5)
    
    # --- NO_TRADE when no signals fire ---
    
    def test_no_trade_when_no_option_signals(self):
        """Default NO_TRADE when fusion is NO_TRADE and no option signals fire."""
        decision, _, reasons, _ = self._call_decision()
        self.assertEqual(decision, "NO_TRADE")
        self.assertIn("FUSION_STATE_NO_TRADE", reasons)


# =============================================================================
# OPTIONS STRATEGY & STOP-LOSS TESTS
# =============================================================================

class TestSpreadGuidance(SimpleTestCase):
    """Test SpreadGuidance dataclass and format_stop_loss_string."""
    
    def test_format_stop_loss_string_basic(self):
        """format_stop_loss_string produces expected format."""
        from signals.options import SpreadGuidance, format_stop_loss_string
        spread = SpreadGuidance(
            width_pct=0.10,
            take_profit_pct=0.70,
            max_hold_days=7,
            stop_loss_pct=0.04,
            scale_down_day=5,
        )
        result = format_stop_loss_string(spread)
        self.assertIn("4.0% stop", result)
        self.assertIn("scale to 25% on day 5", result)
        self.assertIn("hard cut day 7", result)
    
    def test_format_stop_loss_string_zero_stop(self):
        """format_stop_loss_string returns empty when stop_loss_pct <= 0."""
        from signals.options import SpreadGuidance, format_stop_loss_string
        spread = SpreadGuidance(
            width_pct=0.10,
            take_profit_pct=0.70,
            max_hold_days=7,
            stop_loss_pct=0.0,
            scale_down_day=5,
        )
        result = format_stop_loss_string(spread)
        self.assertEqual(result, "")
    
    def test_format_stop_loss_string_none_spread(self):
        """format_stop_loss_string returns empty when spread is None."""
        from signals.options import format_stop_loss_string
        result = format_stop_loss_string(None)
        self.assertEqual(result, "")


class TestDecisionStrategyMap(SimpleTestCase):
    """Test DECISION_STRATEGY_MAP has numeric execution fields."""
    
    def test_option_call_has_numeric_fields(self):
        """OPTION_CALL strategy has all numeric execution fields."""
        from signals.options import DECISION_STRATEGY_MAP
        strategy = DECISION_STRATEGY_MAP["OPTION_CALL"]
        self.assertIn("stop_loss_pct", strategy)
        self.assertIn("scale_down_day", strategy)
        self.assertIn("max_hold_days", strategy)
        self.assertIn("spread_width_pct", strategy)
        self.assertIn("take_profit_pct", strategy)
        self.assertEqual(strategy["stop_loss_pct"], 0.05)
        self.assertEqual(strategy["max_hold_days"], 7)
    
    def test_option_put_has_numeric_fields(self):
        """OPTION_PUT strategy has all numeric execution fields."""
        from signals.options import DECISION_STRATEGY_MAP
        strategy = DECISION_STRATEGY_MAP["OPTION_PUT"]
        self.assertEqual(strategy["stop_loss_pct"], 0.05)
        self.assertEqual(strategy["max_hold_days"], 6)
    
    def test_tactical_put_has_numeric_fields(self):
        """TACTICAL_PUT strategy has all numeric execution fields."""
        from signals.options import DECISION_STRATEGY_MAP
        strategy = DECISION_STRATEGY_MAP["TACTICAL_PUT"]
        self.assertEqual(strategy["stop_loss_pct"], 0.035)
        self.assertEqual(strategy["max_hold_days"], 6)
    
    def test_empty_strategy_has_none_numeric_fields(self):
        """_EMPTY_STRATEGY has None for all numeric fields."""
        from signals.options import _EMPTY_STRATEGY
        self.assertIsNone(_EMPTY_STRATEGY["stop_loss_pct"])
        self.assertIsNone(_EMPTY_STRATEGY["scale_down_day"])
        self.assertIsNone(_EMPTY_STRATEGY["max_hold_days"])


class TestGetStrategyWithPathRisk(SimpleTestCase):
    """Test get_strategy_with_path_risk adjustments."""
    
    def test_high_invalidation_shifts_strike_to_itm(self):
        """High invalidation rate (>=30%) shifts strike from SLIGHT_ITM to ITM."""
        from signals.options import get_strategy_with_path_risk, StrikeSelection
        from signals.fusion import MarketState
        
        # BEAR_PROBE has 55% invalidation rate in PATH_RISK_BY_STATE
        strategy = get_strategy_with_path_risk(
            state=MarketState.BEAR_PROBE,
            invalid_before_hit_rate=0.55,
            same_day_ambiguous_rate=0.0,
        )
        self.assertEqual(strategy.strike_guidance, StrikeSelection.ITM)
    
    def test_moderate_invalidation_no_shift(self):
        """Moderate invalidation rate (<30%) doesn't shift strike."""
        from signals.options import get_strategy_with_path_risk, StrikeSelection
        from signals.fusion import MarketState
        
        strategy = get_strategy_with_path_risk(
            state=MarketState.MOMENTUM_CONTINUATION,
            invalid_before_hit_rate=0.26,
            same_day_ambiguous_rate=0.0,
        )
        self.assertEqual(strategy.strike_guidance, StrikeSelection.SLIGHT_ITM)
    
    def test_combined_rate_triggers_shift(self):
        """Combined inv+amb rate >= 35% triggers shift even if inv alone < 30%."""
        from signals.options import get_strategy_with_path_risk, StrikeSelection
        from signals.fusion import MarketState
        
        strategy = get_strategy_with_path_risk(
            state=MarketState.BULL_PROBE,
            invalid_before_hit_rate=0.20,
            same_day_ambiguous_rate=0.16,  # combined = 36%
        )
        self.assertEqual(strategy.strike_guidance, StrikeSelection.ITM)
    
    def test_dte_floor_applied_on_high_risk(self):
        """High path risk enforces min DTE of 10 and optimal of 14."""
        from signals.options import get_strategy_with_path_risk
        from signals.fusion import MarketState
        
        strategy = get_strategy_with_path_risk(
            state=MarketState.BEAR_PROBE,
            invalid_before_hit_rate=0.55,
            same_day_ambiguous_rate=0.0,
        )
        self.assertGreaterEqual(strategy.dte.min_dte, 10)
        self.assertGreaterEqual(strategy.dte.optimal_dte, 14)


class TestStrategyMapSpreadGuidance(SimpleTestCase):
    """Test STRATEGY_MAP entries have valid SpreadGuidance."""
    
    def test_all_tradeable_states_have_spread(self):
        """All tradeable states (non-NO_TRADE) have SpreadGuidance."""
        from signals.options import STRATEGY_MAP
        from signals.fusion import MarketState
        
        for state, strategy in STRATEGY_MAP.items():
            if state in (MarketState.NO_TRADE, MarketState.TRANSITION_CHOP):
                self.assertIsNone(strategy.spread)
            else:
                self.assertIsNotNone(strategy.spread, f"{state} missing spread")
                self.assertGreater(strategy.spread.stop_loss_pct, 0, f"{state} has zero stop_loss_pct")
                self.assertGreater(strategy.spread.max_hold_days, 0, f"{state} has zero max_hold_days")
    
    def test_min_dte_exceeds_max_hold_days(self):
        """min_dte should be >= max_hold_days + 2 buffer for all states."""
        from signals.options import STRATEGY_MAP
        from signals.fusion import MarketState
        
        for state, strategy in STRATEGY_MAP.items():
            if state in (MarketState.NO_TRADE, MarketState.TRANSITION_CHOP):
                continue
            if strategy.spread is None:
                continue
            buffer = strategy.dte.min_dte - strategy.spread.max_hold_days
            self.assertGreaterEqual(
                buffer, 2,
                f"{state}: min_dte={strategy.dte.min_dte} should be >= max_hold_days={strategy.spread.max_hold_days} + 2"
            )


# =============================================================================
# COOLDOWN CONSTANT ALIGNMENT TESTS
# =============================================================================

class TestCooldownConstantAlignment(SimpleTestCase):
    """Test that cooldown constants are consistent across modules."""
    
    def test_services_exports_all_cooldown_constants(self):
        """services.py exports all cooldown constants for import by analysis commands."""
        from signals.services import (
            CORE_SIGNAL_COOLDOWN_DAYS,
            PROBE_COOLDOWN_DAYS,
            TACTICAL_PUT_COOLDOWN_DAYS,
            OPTION_SIGNAL_COOLDOWN_DAYS,
        )
        self.assertEqual(CORE_SIGNAL_COOLDOWN_DAYS, 7)
        self.assertEqual(PROBE_COOLDOWN_DAYS, 5)
        self.assertEqual(TACTICAL_PUT_COOLDOWN_DAYS, 7)
        self.assertEqual(OPTION_SIGNAL_COOLDOWN_DAYS, 5)


# =============================================================================
# SIGNAL RESULT NUMERIC FIELDS TESTS
# =============================================================================

class TestSignalResultNumericFields(SimpleTestCase):
    """Test SignalResult dataclass has numeric execution fields."""
    
    def test_signal_result_has_numeric_fields(self):
        """SignalResult dataclass includes all numeric execution fields."""
        from signals.services import SignalResult
        import inspect
        
        sig = inspect.signature(SignalResult)
        params = list(sig.parameters.keys())
        
        self.assertIn("stop_loss_pct", params)
        self.assertIn("scale_down_day", params)
        self.assertIn("max_hold_days", params)
        self.assertIn("spread_width_pct", params)
        self.assertIn("take_profit_pct", params)


class TestOptionSignalSoftScaling(SimpleTestCase):
    """Test that option signals apply soft overlay scaling."""
    
    def _make_fusion_no_trade(self):
        return FusionResult(
            state=MarketState.NO_TRADE,
            confidence=Confidence.LOW,
            score=0,
            components={},
        )
    
    def _make_tactical_inactive(self):
        from signals.tactical_puts import TacticalPutResult, TacticalPutStrategy
        return TacticalPutResult(
            active=False, strength=0, strategy=TacticalPutStrategy.NONE,
            size_mult=0.0, dte_mult=1.0, reason="",
        )
    
    def _make_overlay_soft_veto(self):
        """Overlay with soft veto (size_mult=0.5)."""
        return OverlayResult(
            edge_strength=0, long_veto_strength=1, short_veto_strength=0,
            short_edge_strength=0, extended_dte=False,
            reduced_size=True, reason="MODERATE VETO",
        )
    
    def test_option_call_applies_soft_scaling(self):
        """OPTION_CALL applies soft overlay scaling (0.75 * 0.5 = 0.375)."""
        from signals.services import SignalService, OPTION_SIGNAL_SIZE_MULT
        
        svc = SignalService.__new__(SignalService)
        row = make_row(distribution_pressure_score=0.90)
        
        decision, notes, reasons, trace = svc._determine_trade_decision(
            fusion_result=self._make_fusion_no_trade(),
            size_mult=0.5,  # soft veto
            dte_mult=1.0,
            tactical_result=self._make_tactical_inactive(),
            p_long=0.5,
            p_short=0.5,
            signal_option_call=1,
            signal_option_put=0,
            overlay=self._make_overlay_soft_veto(),
            row=row,
            option_call_ok=True,
            option_put_ok=False,
        )
        
        self.assertEqual(decision, "OPTION_CALL")
        # Check trace shows scaled size
        self.assertTrue(any("overlay=0.50" in t for t in trace))
        self.assertTrue(any("effective=0.38" in t or "effective=0.37" in t for t in trace))
    
    def test_option_put_applies_soft_scaling(self):
        """OPTION_PUT applies soft overlay scaling."""
        from signals.services import SignalService
        
        svc = SignalService.__new__(SignalService)
        row = make_row(distribution_pressure_score=0.90)
        
        decision, notes, reasons, trace = svc._determine_trade_decision(
            fusion_result=self._make_fusion_no_trade(),
            size_mult=0.5,  # soft veto
            dte_mult=1.0,
            tactical_result=self._make_tactical_inactive(),
            p_long=0.5,
            p_short=0.5,
            signal_option_call=0,
            signal_option_put=1,
            overlay=self._make_overlay_soft_veto(),
            row=row,
            option_call_ok=False,
            option_put_ok=True,
        )
        
        self.assertEqual(decision, "OPTION_PUT")
        # Check trace shows scaled size
        self.assertTrue(any("overlay=0.50" in t for t in trace))


# =============================================================================
# MVRV SHORT SIGNAL TESTS
# =============================================================================

class TestMvrvShortTradeDecision(SimpleTestCase):
    """Test _determine_trade_decision for MVRV short signal fallback logic."""
    
    def _make_fusion_result(self, state=MarketState.NO_TRADE, score=0, confidence=Confidence.LOW, short_source=None):
        return FusionResult(
            state=state,
            confidence=confidence,
            score=score,
            components={},
            short_source=short_source,
        )
    
    def _make_tactical_inactive(self):
        from signals.tactical_puts import TacticalPutResult, TacticalPutStrategy
        return TacticalPutResult(
            active=False, strength=0, strategy=TacticalPutStrategy.NONE,
            size_mult=0.0, dte_mult=1.0, reason="",
        )
    
    def _make_overlay_clean(self):
        return OverlayResult(
            edge_strength=0, long_veto_strength=0, short_veto_strength=0,
            short_edge_strength=0, extended_dte=False,
            reduced_size=False, reason="",
        )
    
    def _make_mvrv_short_active(self, mvrv_7d=1.05, mvrv_60d=1.02, btc_price=100000):
        from signals.mvrv_short import MvrvShortSignal
        return MvrvShortSignal(
            active=True,
            is_bear_mode=True,
            cycle_day=600,
            mvrv_7d=mvrv_7d,
            mvrv_60d=mvrv_60d,
            btc_price=btc_price,
            target_price=btc_price * 0.96,
            dca_trigger_price=btc_price * 1.04,
            reason=f"MVRV 7d={mvrv_7d:.4f} >= 1.02, MVRV 60d={mvrv_60d:.4f} >= 1.0",
        )
    
    def _make_mvrv_short_inactive(self, reason="Not in bear mode"):
        from signals.mvrv_short import MvrvShortSignal
        return MvrvShortSignal(
            active=False,
            is_bear_mode=False,
            cycle_day=300,
            mvrv_7d=1.05,
            mvrv_60d=1.02,
            btc_price=100000,
            target_price=96000,
            dca_trigger_price=104000,
            reason=reason,
        )
    
    def _call_decision(self, fusion_result=None, size_mult=1.0, dte_mult=1.0,
                        tactical_result=None, p_long=0.5, p_short=0.5,
                        signal_option_call=0, signal_option_put=0,
                        overlay=None, option_call_ok=False, option_put_ok=False,
                        mvrv_short_ok=False, mvrv_short_signal=None,
                        row=None):
        from signals.services import SignalService
        svc = SignalService.__new__(SignalService)  # bypass __init__
        if row is None:
            row = make_row(distribution_pressure_score=0.90)
        return svc._determine_trade_decision(
            fusion_result=fusion_result or self._make_fusion_result(),
            size_mult=size_mult,
            dte_mult=dte_mult,
            tactical_result=tactical_result or self._make_tactical_inactive(),
            p_long=p_long,
            p_short=p_short,
            signal_option_call=signal_option_call,
            signal_option_put=signal_option_put,
            overlay=overlay or self._make_overlay_clean(),
            row=row,
            option_call_ok=option_call_ok,
            option_put_ok=option_put_ok,
            mvrv_short_ok=mvrv_short_ok,
            mvrv_short_signal=mvrv_short_signal,
        )
    
    # --- MVRV_SHORT fires when conditions met ---
    
    def test_mvrv_short_fires_on_no_trade(self):
        """MVRV_SHORT fires when fusion=NO_TRADE, signal active, cooldown OK."""
        decision, notes, reasons, trace = self._call_decision(
            mvrv_short_ok=True,
            mvrv_short_signal=self._make_mvrv_short_active(),
        )
        self.assertEqual(decision, "MVRV_SHORT")
        self.assertIn("Bear mode", notes)
        self.assertIn("MVRV 7d >= 1.02", notes)
    
    def test_mvrv_short_includes_execution_params(self):
        """MVRV_SHORT notes include entry, DCA, and target prices."""
        decision, notes, reasons, trace = self._call_decision(
            mvrv_short_ok=True,
            mvrv_short_signal=self._make_mvrv_short_active(btc_price=100000),
        )
        self.assertEqual(decision, "MVRV_SHORT")
        self.assertIn("Entry: 33%", notes)
        self.assertIn("$100,000", notes)
        self.assertIn("DCA at $104,000", notes)
        self.assertIn("Target: $96,000", notes)
    
    # --- Fusion priority over MVRV_SHORT ---
    
    def test_fusion_call_takes_priority_over_mvrv_short(self):
        """When fusion has a directional view, it takes priority over MVRV short."""
        fusion = self._make_fusion_result(
            state=MarketState.EARLY_RECOVERY, score=4, confidence=Confidence.HIGH,
        )
        decision, notes, reasons, trace = self._call_decision(
            fusion_result=fusion,
            mvrv_short_ok=True,
            mvrv_short_signal=self._make_mvrv_short_active(),
        )
        self.assertEqual(decision, "CALL")
        self.assertIn("early_recovery", notes)
    
    def test_fusion_put_takes_priority_over_mvrv_short(self):
        """When fusion says short, it takes priority over MVRV short signal."""
        fusion = self._make_fusion_result(
            state=MarketState.BEAR_CONTINUATION, score=-4, confidence=Confidence.LOW,
            short_source="rule",
        )
        decision, notes, reasons, trace = self._call_decision(
            fusion_result=fusion,
            mvrv_short_ok=True,
            mvrv_short_signal=self._make_mvrv_short_active(),
        )
        self.assertEqual(decision, "PUT")
    
    # --- Option signals priority over MVRV_SHORT ---
    
    def test_option_call_takes_priority_over_mvrv_short(self):
        """OPTION_CALL fires before MVRV_SHORT (call > put > mvrv_short)."""
        decision, notes, reasons, trace = self._call_decision(
            signal_option_call=1,
            option_call_ok=True,
            mvrv_short_ok=True,
            mvrv_short_signal=self._make_mvrv_short_active(),
        )
        self.assertEqual(decision, "OPTION_CALL")
    
    def test_option_put_takes_priority_over_mvrv_short(self):
        """OPTION_PUT fires before MVRV_SHORT."""
        decision, notes, reasons, trace = self._call_decision(
            signal_option_put=1,
            option_put_ok=True,
            mvrv_short_ok=True,
            mvrv_short_signal=self._make_mvrv_short_active(),
        )
        self.assertEqual(decision, "OPTION_PUT")
    
    # --- MVRV_SHORT blocked by cooldown ---
    
    def test_mvrv_short_blocked_by_cooldown(self):
        """MVRV short signal active but cooldown not passed → NO_TRADE."""
        decision, _, reasons, trace = self._call_decision(
            mvrv_short_ok=False,  # cooldown active
            mvrv_short_signal=self._make_mvrv_short_active(),
        )
        self.assertEqual(decision, "NO_TRADE")
        self.assertTrue(any("cooldown_active" in t for t in trace))
    
    # --- MVRV_SHORT inactive conditions ---
    
    def test_mvrv_short_inactive_not_bear_mode(self):
        """MVRV short inactive when not in bear mode → NO_TRADE."""
        decision, _, reasons, trace = self._call_decision(
            mvrv_short_ok=True,
            mvrv_short_signal=self._make_mvrv_short_inactive("Not in bear mode (cycle day 300)"),
        )
        self.assertEqual(decision, "NO_TRADE")
        self.assertTrue(any("mvrv_short=inactive" in t for t in trace))
    
    def test_mvrv_short_inactive_mvrv_below_threshold(self):
        """MVRV short inactive when MVRV 7d below threshold → NO_TRADE."""
        from signals.mvrv_short import MvrvShortSignal
        inactive_signal = MvrvShortSignal(
            active=False,
            is_bear_mode=True,
            cycle_day=600,
            mvrv_7d=0.98,
            mvrv_60d=1.02,
            btc_price=100000,
            target_price=96000,
            dca_trigger_price=104000,
            reason="MVRV 7d below threshold (0.9800 < 1.02)",
        )
        decision, _, reasons, trace = self._call_decision(
            mvrv_short_ok=True,
            mvrv_short_signal=inactive_signal,
        )
        self.assertEqual(decision, "NO_TRADE")
    
    # --- Constants ---
    
    def test_mvrv_short_size_mult(self):
        """MVRV short uses 0.33x sizing (33% initial, 67% DCA)."""
        from signals.services import MVRV_SHORT_SIZE_MULT
        self.assertEqual(MVRV_SHORT_SIZE_MULT, 0.33)
    
    def test_mvrv_short_cooldown_days(self):
        """MVRV short cooldown is 5 days."""
        from signals.services import MVRV_SHORT_COOLDOWN_DAYS
        self.assertEqual(MVRV_SHORT_COOLDOWN_DAYS, 5)


class TestMvrvShortStrategyMap(SimpleTestCase):
    """Test MVRV_SHORT entry in DECISION_STRATEGY_MAP."""
    
    def test_mvrv_short_in_decision_strategy_map(self):
        """MVRV_SHORT has entry in DECISION_STRATEGY_MAP."""
        from signals.options import DECISION_STRATEGY_MAP
        self.assertIn("MVRV_SHORT", DECISION_STRATEGY_MAP)
    
    def test_mvrv_short_has_required_fields(self):
        """MVRV_SHORT strategy has all required fields."""
        from signals.options import DECISION_STRATEGY_MAP
        strategy = DECISION_STRATEGY_MAP["MVRV_SHORT"]
        self.assertIn("primary_structures", strategy)
        self.assertIn("strike_guidance", strategy)
        self.assertIn("dte_range", strategy)
        self.assertIn("rationale", strategy)
        self.assertIn("stop_loss", strategy)
        self.assertIn("stop_loss_pct", strategy)
        self.assertIn("max_hold_days", strategy)
        self.assertIn("take_profit_pct", strategy)
    
    def test_mvrv_short_execution_params(self):
        """MVRV_SHORT has correct execution parameters."""
        from signals.options import DECISION_STRATEGY_MAP
        strategy = DECISION_STRATEGY_MAP["MVRV_SHORT"]
        self.assertEqual(strategy["stop_loss_pct"], 0.04)  # DCA trigger
        self.assertEqual(strategy["max_hold_days"], 5)
        self.assertEqual(strategy["take_profit_pct"], 0.04)  # 4% target
    
    def test_mvrv_short_rationale_mentions_bear_mode(self):
        """MVRV_SHORT rationale mentions bear mode and MVRV thresholds."""
        from signals.options import DECISION_STRATEGY_MAP
        strategy = DECISION_STRATEGY_MAP["MVRV_SHORT"]
        self.assertIn("Bear mode", strategy["rationale"])
        self.assertIn("MVRV 7d", strategy["rationale"])
