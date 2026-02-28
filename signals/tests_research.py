"""
Tests for the signals.research package.

Covers:
  - Bucket mapping (mutual exclusivity, all branches)
  - Research table schema
  - Stats aggregation correctness
  - Modeling I/O contract
  - Monotonicity / stability validation output
"""

import numpy as np
import pandas as pd
from django.test import SimpleTestCase

from signals.research.bucket_mapping import (
    add_bucket_columns,
    map_mdia_bucket,
    map_mvrv_ls_bucket,
    map_whale_bucket,
)
from signals.research.constants import (
    MDIA_BUCKETS,
    MVRV_LS_BUCKETS,
    WHALE_BUCKETS,
)
from signals.research.stats import (
    compute_bucket_stats,
    compute_combo_stats,
    compute_score_stats,
    compute_state_stats,
)
from signals.research.reporting import (
    validate_monotonicity,
    validate_state_stability,
)
from signals.tests import make_row


# =====================================================================
# Helpers
# =====================================================================
def _make_research_df(n: int = 50) -> pd.DataFrame:
    """Build a minimal synthetic research DataFrame for stats / reporting tests."""
    rng = np.random.RandomState(42)
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    mdia_buckets = rng.choice(MDIA_BUCKETS, n)
    whale_buckets = rng.choice(WHALE_BUCKETS, n)
    mvrv_buckets = rng.choice(MVRV_LS_BUCKETS, n)
    states = rng.choice(
        ["strong_bullish", "momentum", "no_trade", "distribution_risk"], n
    )
    scores = rng.randint(-3, 6, n)

    return pd.DataFrame(
        {
            "close": rng.uniform(30000, 70000, n),
            "mdia_bucket": mdia_buckets,
            "whale_bucket": whale_buckets,
            "mvrv_ls_bucket": mvrv_buckets,
            "fusion_state": states,
            "fusion_score": scores,
            "fusion_confidence": rng.choice(["high", "medium", "low"], n),
            "label_good_move_long": rng.randint(0, 2, n),
            "label_good_move_short": rng.randint(0, 2, n),
            "ret_7d": rng.uniform(-0.1, 0.1, n),
            "ret_14d": rng.uniform(-0.1, 0.1, n),
            "ret_21d": rng.uniform(-0.1, 0.1, n),
            "mfe_14d": rng.uniform(0, 0.15, n),
            "mae_14d": rng.uniform(0, 0.10, n),
            "year": 2022,
        },
        index=dates,
    )


# =====================================================================
# Bucket Mapping Tests
# =====================================================================
class TestMdiaBucket(SimpleTestCase):
    def test_strong_inflow(self):
        row = make_row(mdia_regime_strong_inflow=1, mdia_regime_inflow=1)
        self.assertEqual(map_mdia_bucket(row), "strong_inflow")

    def test_inflow(self):
        row = make_row(mdia_regime_inflow=1)
        self.assertEqual(map_mdia_bucket(row), "inflow")

    def test_aging(self):
        row = make_row(mdia_regime_aging=1)
        self.assertEqual(map_mdia_bucket(row), "aging")

    def test_neutral(self):
        row = make_row()
        self.assertEqual(map_mdia_bucket(row), "neutral")

    def test_priority_strong_over_inflow(self):
        """strong_inflow should win even when inflow is also set."""
        row = make_row(mdia_regime_strong_inflow=1, mdia_regime_inflow=1)
        self.assertEqual(map_mdia_bucket(row), "strong_inflow")

    def test_priority_inflow_over_aging(self):
        row = make_row(mdia_regime_inflow=1, mdia_regime_aging=1)
        self.assertEqual(map_mdia_bucket(row), "inflow")

    def test_returns_valid_bucket(self):
        for kwargs in [
            {},
            {"mdia_regime_strong_inflow": 1},
            {"mdia_regime_inflow": 1},
            {"mdia_regime_aging": 1},
        ]:
            row = make_row(**kwargs)
            self.assertIn(map_mdia_bucket(row), MDIA_BUCKETS)


class TestWhaleBucket(SimpleTestCase):
    def test_broad_accum(self):
        row = make_row(whale_regime_broad_accum=1)
        self.assertEqual(map_whale_bucket(row), "broad_accum")

    def test_strategic_accum(self):
        row = make_row(whale_regime_strategic_accum=1)
        self.assertEqual(map_whale_bucket(row), "strategic_accum")

    def test_distribution_strong(self):
        row = make_row(whale_regime_distribution_strong=1)
        self.assertEqual(map_whale_bucket(row), "distribution_strong")

    def test_distribution(self):
        row = make_row(whale_regime_distribution=1)
        self.assertEqual(map_whale_bucket(row), "distribution")

    def test_mixed(self):
        row = make_row(whale_regime_mixed=1)
        self.assertEqual(map_whale_bucket(row), "mixed")

    def test_neutral(self):
        row = make_row()
        self.assertEqual(map_whale_bucket(row), "neutral")

    def test_priority_broad_over_strategic(self):
        row = make_row(
            whale_regime_broad_accum=1, whale_regime_strategic_accum=1
        )
        self.assertEqual(map_whale_bucket(row), "broad_accum")

    def test_priority_distrib_strong_over_distrib(self):
        row = make_row(
            whale_regime_distribution_strong=1, whale_regime_distribution=1
        )
        self.assertEqual(map_whale_bucket(row), "distribution_strong")

    def test_returns_valid_bucket(self):
        for kwargs in [
            {},
            {"whale_regime_broad_accum": 1},
            {"whale_regime_strategic_accum": 1},
            {"whale_regime_distribution_strong": 1},
            {"whale_regime_distribution": 1},
            {"whale_regime_mixed": 1},
        ]:
            row = make_row(**kwargs)
            self.assertIn(map_whale_bucket(row), WHALE_BUCKETS)


class TestMvrvLsBucket(SimpleTestCase):
    def test_early_recovery(self):
        row = make_row(mvrv_ls_regime_call_confirm_recovery=1)
        self.assertEqual(map_mvrv_ls_bucket(row), "early_recovery")

    def test_trend_confirm(self):
        row = make_row(mvrv_ls_regime_call_confirm_trend=1)
        self.assertEqual(map_mvrv_ls_bucket(row), "trend_confirm")

    def test_call_confirm(self):
        row = make_row(mvrv_ls_regime_call_confirm=1)
        self.assertEqual(map_mvrv_ls_bucket(row), "call_confirm")

    def test_distribution_warning(self):
        row = make_row(mvrv_ls_regime_distribution_warning=1)
        self.assertEqual(map_mvrv_ls_bucket(row), "distribution_warning")

    def test_put_confirm(self):
        row = make_row(mvrv_ls_regime_put_confirm=1)
        self.assertEqual(map_mvrv_ls_bucket(row), "put_confirm")

    def test_neutral(self):
        row = make_row()
        self.assertEqual(map_mvrv_ls_bucket(row), "neutral")

    def test_priority_recovery_over_call_confirm(self):
        """Recovery is a subset of call_confirm; it must win."""
        row = make_row(
            mvrv_ls_regime_call_confirm_recovery=1,
            mvrv_ls_regime_call_confirm=1,
        )
        self.assertEqual(map_mvrv_ls_bucket(row), "early_recovery")

    def test_priority_trend_over_call_confirm(self):
        row = make_row(
            mvrv_ls_regime_call_confirm_trend=1,
            mvrv_ls_regime_call_confirm=1,
        )
        self.assertEqual(map_mvrv_ls_bucket(row), "trend_confirm")

    def test_returns_valid_bucket(self):
        for kwargs in [
            {},
            {"mvrv_ls_regime_call_confirm_recovery": 1},
            {"mvrv_ls_regime_call_confirm_trend": 1},
            {"mvrv_ls_regime_call_confirm": 1},
            {"mvrv_ls_regime_distribution_warning": 1},
            {"mvrv_ls_regime_put_confirm": 1},
        ]:
            row = make_row(**kwargs)
            self.assertIn(map_mvrv_ls_bucket(row), MVRV_LS_BUCKETS)


class TestAddBucketColumns(SimpleTestCase):
    def test_adds_three_columns(self):
        rows = [make_row(), make_row(mdia_regime_inflow=1)]
        df = pd.DataFrame(rows)
        result = add_bucket_columns(df)
        self.assertIn("mdia_bucket", result.columns)
        self.assertIn("whale_bucket", result.columns)
        self.assertIn("mvrv_ls_bucket", result.columns)

    def test_does_not_mutate_input(self):
        df = pd.DataFrame([make_row()])
        original_cols = set(df.columns)
        add_bucket_columns(df)
        self.assertEqual(set(df.columns), original_cols)


# =====================================================================
# Stats Tests
# =====================================================================
class TestBucketStats(SimpleTestCase):
    def test_returns_rows(self):
        rt = _make_research_df()
        stats = compute_bucket_stats(rt, "mdia_bucket")
        self.assertGreater(len(stats), 0)

    def test_has_required_columns(self):
        rt = _make_research_df()
        stats = compute_bucket_stats(rt, "mdia_bucket")
        for col in ("n", "long_hit_rate", "short_hit_rate", "ret_14d_mean", "_flagged"):
            self.assertIn(col, stats.columns, f"Missing column: {col}")

    def test_min_count_flagging(self):
        rt = _make_research_df(n=15)
        stats = compute_bucket_stats(rt, "mdia_bucket", min_count=20)
        # With only 15 rows total, all groups should be flagged
        self.assertTrue(stats["_flagged"].all())


class TestComboStats(SimpleTestCase):
    def test_pairwise_grouping(self):
        rt = _make_research_df()
        stats = compute_combo_stats(rt, ["mdia_bucket", "whale_bucket"])
        self.assertGreater(len(stats), 0)
        self.assertIn("mdia_bucket", stats.columns)
        self.assertIn("whale_bucket", stats.columns)


class TestStateStats(SimpleTestCase):
    def test_groups_by_state(self):
        rt = _make_research_df()
        stats = compute_state_stats(rt)
        self.assertIn("fusion_state", stats.columns)
        self.assertGreater(len(stats), 0)


class TestScoreStats(SimpleTestCase):
    def test_groups_by_score(self):
        rt = _make_research_df()
        stats = compute_score_stats(rt)
        self.assertIn("fusion_score", stats.columns)
        self.assertGreater(len(stats), 0)


# =====================================================================
# Reporting / Validation Tests
# =====================================================================
class TestMonotonicity(SimpleTestCase):
    def test_has_required_columns(self):
        rt = _make_research_df()
        result = validate_monotonicity(rt)
        for col in ("score", "n", "long_hit_rate", "ret_14d_median"):
            self.assertIn(col, result.columns, f"Missing column: {col}")

    def test_sorted_by_score(self):
        rt = _make_research_df()
        result = validate_monotonicity(rt)
        scores = result["score"].tolist()
        self.assertEqual(scores, sorted(scores))


class TestStateStability(SimpleTestCase):
    def test_has_required_columns(self):
        rt = _make_research_df()
        result = validate_state_stability(rt)
        for col in ("fusion_state", "year", "n"):
            self.assertIn(col, result.columns, f"Missing column: {col}")

    def test_returns_all_states(self):
        rt = _make_research_df()
        result = validate_state_stability(rt)
        result_states = set(result["fusion_state"].unique())
        expected_states = set(rt["fusion_state"].unique())
        self.assertEqual(result_states, expected_states)


# =====================================================================
# Modeling I/O Contract Tests
# =====================================================================
class TestModelComparison(SimpleTestCase):
    """Light test — only validates output structure, not accuracy."""

    def test_output_keys(self):
        from signals.research.modeling import run_model_comparison

        rt = _make_research_df(n=200)
        result = run_model_comparison(rt, target_col="label_good_move_long", n_splits=3)
        self.assertIn("model_results", result)
        self.assertIn("feature_importance", result)
        self.assertIn("summary", result)

    def test_seven_models(self):
        from signals.research.modeling import run_model_comparison

        rt = _make_research_df(n=200)
        result = run_model_comparison(rt, target_col="label_good_move_long", n_splits=3)
        self.assertEqual(len(result["model_results"]), 7)

    def test_auc_in_range(self):
        from signals.research.modeling import run_model_comparison

        rt = _make_research_df(n=200)
        result = run_model_comparison(rt, target_col="label_good_move_long", n_splits=3)
        for mr in result["model_results"]:
            if not np.isnan(mr["mean_auc"]):
                self.assertGreaterEqual(mr["mean_auc"], 0.0)
                self.assertLessEqual(mr["mean_auc"], 1.0)

    def test_summary_keys(self):
        from signals.research.modeling import run_model_comparison

        rt = _make_research_df(n=200)
        result = run_model_comparison(rt, target_col="label_good_move_long", n_splits=3)
        summary = result["summary"]
        self.assertIn("best_standalone", summary)
        self.assertIn("best_incremental", summary)
        self.assertIn("incremental_values", summary)
