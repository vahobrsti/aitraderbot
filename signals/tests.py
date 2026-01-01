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
    compute_confidence_score,
    score_to_confidence,
    classify_market_state,
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
        'mdia_regime_distribution': 0,
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
    
    def test_momentum_with_weak_uptrend(self):
        """MOMENTUM accepts weak_uptrend (looser MVRV condition)."""
        row = make_row(
            mdia_regime_inflow=1,
            whale_regime_mixed=1,
            mvrv_ls_weak_uptrend=1,
        )
        self.assertEqual(classify_market_state(row), MarketState.MOMENTUM_CONTINUATION)
    
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
            mdia_regime_distribution=1,
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
        """BEAR_PROBE: MDIA distrib + whale distrib + MVRV neutral."""
        row = make_row(
            mdia_regime_distribution=1,
            whale_regime_distribution=1,
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


class TestConfidenceScore(SimpleTestCase):
    """Test compute_confidence_score function."""
    
    def test_max_bullish_score(self):
        """Maximum bullish score: +6 (strong_inflow + broad_accum + early_recovery)."""
        row = make_row(
            mdia_regime_strong_inflow=1,
            whale_regime_broad_accum=1,
            mvrv_ls_regime_call_confirm_recovery=1,
        )
        score, components = compute_confidence_score(row)
        self.assertEqual(score, 6)
        self.assertEqual(components['mdia']['score'], 2)
        self.assertEqual(components['whale']['score'], 2)
        self.assertEqual(components['mvrv_ls']['score'], 2)
    
    def test_max_bearish_score(self):
        """Maximum bearish score: -5 (distrib + strong_distrib + put_confirm)."""
        row = make_row(
            mdia_regime_distribution=1,
            whale_regime_distribution_strong=1,
            mvrv_ls_regime_put_confirm=1,
        )
        score, components = compute_confidence_score(row)
        self.assertEqual(score, -5)
        self.assertEqual(components['mdia']['score'], -1)
        self.assertEqual(components['whale']['score'], -2)
        self.assertEqual(components['mvrv_ls']['score'], -2)
    
    def test_neutral_score(self):
        """Neutral: All components at 0."""
        row = make_row()
        score, components = compute_confidence_score(row)
        self.assertEqual(score, 0)
        self.assertEqual(components['mdia']['score'], 0)
        self.assertEqual(components['whale']['score'], 0)
        self.assertEqual(components['mvrv_ls']['score'], 0)
    
    def test_conflict_penalty(self):
        """Conflicts reduce score."""
        row = make_row(
            mdia_regime_strong_inflow=1,  # +2
            whale_regime_broad_accum=1,   # +2
            mvrv_ls_conflict=1,           # -1
            whale_mega_conflict=1,        # -1
        )
        score, components = compute_confidence_score(row)
        self.assertEqual(score, 2)  # 2+2-2 = 2
        self.assertEqual(components['conflicts']['score'], -2)
    
    def test_mvrv_recovery_before_trend(self):
        """Recovery is checked before trend (recovery is more specific)."""
        row = make_row(
            mvrv_ls_regime_call_confirm_recovery=1,
            mvrv_ls_regime_call_confirm_trend=1,  # Also set, but recovery wins
        )
        score, components = compute_confidence_score(row)
        self.assertEqual(components['mvrv_ls']['score'], 2)
        self.assertEqual(components['mvrv_ls']['label'], 'early_recovery')


class TestScoreToConfidence(SimpleTestCase):
    """Test score_to_confidence thresholds."""
    
    def test_high_confidence(self):
        """Score >= 4 is HIGH."""
        self.assertEqual(score_to_confidence(4), Confidence.HIGH)
        self.assertEqual(score_to_confidence(6), Confidence.HIGH)
    
    def test_medium_confidence(self):
        """Score 2-3 is MEDIUM."""
        self.assertEqual(score_to_confidence(2), Confidence.MEDIUM)
        self.assertEqual(score_to_confidence(3), Confidence.MEDIUM)
    
    def test_low_confidence(self):
        """Score < 2 is LOW."""
        self.assertEqual(score_to_confidence(1), Confidence.LOW)
        self.assertEqual(score_to_confidence(0), Confidence.LOW)
        self.assertEqual(score_to_confidence(-3), Confidence.LOW)


class TestFuseSignals(SimpleTestCase):
    """Test fuse_signals integration."""
    
    def test_fuse_strong_bullish(self):
        """Integration test for STRONG_BULLISH with HIGH confidence."""
        # Note: Use call_confirm, not recovery (recovery triggers EARLY_RECOVERY state)
        row = make_row(
            mdia_regime_strong_inflow=1,
            whale_regime_broad_accum=1,
            mvrv_ls_regime_call_confirm=1,
        )
        result = fuse_signals(row)
        self.assertEqual(result.state, MarketState.STRONG_BULLISH)
        self.assertEqual(result.confidence, Confidence.HIGH)
        self.assertGreaterEqual(result.score, 4)  # HIGH confidence threshold


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
        self.assertEqual(overlay.score_adjustment, 0)
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
        self.assertTrue(overlay.score_adjustment > 0)
    
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
        self.assertEqual(overlay.score_adjustment, -2)
    
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
            score_adjustment=-2,
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
            score_adjustment=-1,
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
            score_adjustment=2,
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
            score_adjustment=0,
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
            score_adjustment=2,
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
            score_adjustment=2,
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
            score_adjustment=-2,
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
            score_adjustment=-2,
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
            score_adjustment=2,
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
            score_adjustment=0,
            extended_dte=False,
            reduced_size=False,
            reason="test",
        )
        result = adjust_confidence_with_overlay(Confidence.MEDIUM, overlay)
        self.assertEqual(result, Confidence.MEDIUM)
