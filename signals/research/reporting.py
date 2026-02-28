"""
Weighting proposal and validation reporting.

- generate_weighting_proposal: derive metric roles & score architecture
  from model comparison and bucket stats.
- validate_monotonicity: does a higher score ➞ better outcomes?
- validate_state_stability: per-state consistency across years.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from signals.research.constants import MIN_SAMPLE_THRESHOLD


# ────────────────────────────────────────────────────────────────────
# Weighting proposal
# ────────────────────────────────────────────────────────────────────
def generate_weighting_proposal(
    model_results: dict,
    bucket_stats: dict[str, pd.DataFrame] | None = None,
) -> dict:
    """Derive metric roles and a score architecture proposal.

    Parameters
    ----------
    model_results : dict
        Output of ``run_model_comparison``.
    bucket_stats : dict, optional
        Keyed by metric name (``mdia``, ``whales``, ``mvrv_ls``), each
        value a DataFrame from ``compute_bucket_stats``.

    Returns
    -------
    dict with keys:
        metric_roles, score_architecture, interaction_rules
    """
    summary = model_results.get("summary", {})
    importance = model_results.get("feature_importance", {})

    # ── Assign roles based on standalone + incremental results ─────
    best_standalone = summary.get("best_standalone")
    best_incremental = summary.get("best_incremental")
    confirmers = summary.get("confirmer_metrics", [])

    roles = {}
    for metric in ("mdia", "whales", "mvrv_ls"):
        if metric == best_standalone:
            roles[metric] = "anchor (strongest standalone)"
        elif metric == best_incremental:
            roles[metric] = "key incremental contributor"
        elif metric in confirmers:
            roles[metric] = "confirmer (low standalone, positive incremental)"
        else:
            roles[metric] = "secondary"

    # ── Score architecture ────────────────────────────────────────
    architecture = {
        "base_score": "Directional base from structural anchor metric",
        "confirmation_adjustment": "Upgrade / downgrade from confirmation metric",
        "timing_adjustment": "Timing-sensitive states accelerated by timing metric",
        "conflict_penalty": "Reduce confidence (don't flip direction) on conflicts",
    }

    # ── Interaction rules ─────────────────────────────────────────
    interaction_rules = [
        {
            "rule": "alignment_bonus",
            "description": "All three metrics agree on direction → +1 confidence",
        },
        {
            "rule": "confirmation_upgrade",
            "description": "Anchor bullish + confirmer bullish → upgrade conviction",
        },
        {
            "rule": "conflict_penalty",
            "description": "Anchor vs confirmer disagree → reduce confidence, do not flip",
        },
        {
            "rule": "timing_neutral",
            "description": "Timing metric neutral → no adjustment (don't penalise)",
        },
    ]

    return {
        "metric_roles": roles,
        "importance_scores": importance,
        "score_architecture": architecture,
        "interaction_rules": interaction_rules,
    }


# ────────────────────────────────────────────────────────────────────
# Monotonicity validation
# ────────────────────────────────────────────────────────────────────
def validate_monotonicity(
    research_df: pd.DataFrame,
    score_col: str = "fusion_score",
    min_count: int = MIN_SAMPLE_THRESHOLD,
) -> pd.DataFrame:
    """Check: does a higher score produce better outcomes?

    Returns one row per score bucket with:
        score, n, long_hit_rate, short_hit_rate,
        ret_14d_median, monotonic_long, monotonic_short
    """
    rows = []
    for score, grp in research_df.groupby(score_col):
        n = len(grp)
        row: dict = {"score": score, "n": n}
        row["_flagged"] = n < min_count

        if "label_good_move_long" in grp.columns:
            vals = grp["label_good_move_long"].dropna()
            row["long_hit_rate"] = vals.mean() if len(vals) else np.nan
        if "label_good_move_short" in grp.columns:
            vals = grp["label_good_move_short"].dropna()
            row["short_hit_rate"] = vals.mean() if len(vals) else np.nan
        if "ret_14d" in grp.columns:
            vals = grp["ret_14d"].dropna()
            row["ret_14d_median"] = vals.median() if len(vals) else np.nan

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values("score").reset_index(drop=True)

    # Check monotonicity (higher score → better long, worse short)
    if "long_hit_rate" in out.columns:
        lr = out["long_hit_rate"].values
        out["monotonic_long"] = all(
            lr[i] <= lr[i + 1]
            for i in range(len(lr) - 1)
            if not (np.isnan(lr[i]) or np.isnan(lr[i + 1]))
        )
    if "short_hit_rate" in out.columns:
        sr = out["short_hit_rate"].values
        out["monotonic_short"] = all(
            sr[i] >= sr[i + 1]
            for i in range(len(sr) - 1)
            if not (np.isnan(sr[i]) or np.isnan(sr[i + 1]))
        )

    return out


# ────────────────────────────────────────────────────────────────────
# State stability
# ────────────────────────────────────────────────────────────────────
def validate_state_stability(
    research_df: pd.DataFrame,
    min_count: int = MIN_SAMPLE_THRESHOLD,
) -> pd.DataFrame:
    """Per fusion_state × year: hit rate, median ret_14d, sample count.

    Useful for spotting states that only work in certain years.
    """
    rows = []
    for (state, year), grp in research_df.groupby(
        ["fusion_state", "year"], observed=True
    ):
        n = len(grp)
        row: dict = {"fusion_state": state, "year": year, "n": n}
        row["_flagged"] = n < min_count

        if "label_good_move_long" in grp.columns:
            vals = grp["label_good_move_long"].dropna()
            row["long_hit_rate"] = vals.mean() if len(vals) else np.nan
        if "label_good_move_short" in grp.columns:
            vals = grp["label_good_move_short"].dropna()
            row["short_hit_rate"] = vals.mean() if len(vals) else np.nan
        if "ret_14d" in grp.columns:
            vals = grp["ret_14d"].dropna()
            row["ret_14d_median"] = vals.median() if len(vals) else np.nan

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(["fusion_state", "year"])
        .reset_index(drop=True)
    )
