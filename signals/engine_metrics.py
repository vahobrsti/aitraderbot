# signals/engine_metrics.py
"""
Essential engine metrics extraction.

Collects the "at-a-glance" metrics that explain an engine decision for a single
feature row: all canonical buckets, exchange flow balance, sentiment,
mvrv_composite, mvrv_60d, and their z-scores.

Shared by the ``analyze_engine`` management command and the AnalyzeEngine API
endpoint so both surfaces stay consistent.
"""
from __future__ import annotations

import pandas as pd

from signals.fusion import fuse_signals, FusionResult
from signals.research.bucket_mapping import (
    map_mdia_bucket,
    map_whale_bucket,
    map_mvrv_ls_bucket,
    map_mvrv_60d_bucket,
)

# Mutually-exclusive flag buckets → active bucket name
_SENTIMENT_BUCKET_FLAGS = {
    "extreme_fear": "sent_bucket_extreme_fear",
    "fear": "sent_bucket_fear",
    "neutral": "sent_bucket_neutral",
    "greed": "sent_bucket_greed",
    "extreme_greed": "sent_bucket_extreme_greed",
}

_MVRV_COMPOSITE_BUCKET_FLAGS = {
    "deep_undervalued": "mvrv_bucket_deep_undervalued",
    "undervalued": "mvrv_bucket_undervalued",
    "fair": "mvrv_bucket_fair",
    "overvalued": "mvrv_bucket_overvalued",
    "extreme_overvalued": "mvrv_bucket_extreme_overvalued",
}


def _num(row: pd.Series, col: str):
    """Return a plain float for a column, or None if missing/NaN."""
    val = row.get(col, None)
    if val is None or pd.isna(val):
        return None
    return float(val)


def _active_flag_bucket(row: pd.Series, flag_map: dict) -> str:
    """Return the name of the single active (==1) flag bucket, else 'unknown'."""
    for name, col in flag_map.items():
        val = row.get(col, 0)
        if val is not None and not pd.isna(val) and int(val) == 1:
            return name
    return "unknown"


def collect_essential_metrics(row: pd.Series, fusion_result: FusionResult = None) -> dict:
    """
    Build the essential-metrics dict for one feature row.

    Sections: fusion, buckets, exchange_flow, sentiment, mvrv_composite, mvrv_60d.
    Missing columns resolve to None (numeric) or 'unknown' (bucket).
    """
    if fusion_result is None:
        fusion_result = fuse_signals(row)

    return {
        "fusion": {
            "state": fusion_result.state.value,
            "confidence": fusion_result.confidence.value,
            "score": fusion_result.score,
            "bear_mode": bool(fusion_result.components.get("bear_mode", False)),
            "cycle_day": fusion_result.components.get("cycle_day"),
        },
        "buckets": {
            "mdia": map_mdia_bucket(row),
            "whale": map_whale_bucket(row),
            "mvrv_ls": map_mvrv_ls_bucket(row),
            "mvrv_60d": map_mvrv_60d_bucket(row),
            "sentiment": _active_flag_bucket(row, _SENTIMENT_BUCKET_FLAGS),
            "mvrv_composite": _active_flag_bucket(row, _MVRV_COMPOSITE_BUCKET_FLAGS),
        },
        "exchange_flow": {
            "flow_raw": _num(row, "flow_raw"),
            "flow_sum_2": _num(row, "flow_sum_2"),
            "flow_sum_4": _num(row, "flow_sum_4"),
            "flow_sum_7": _num(row, "flow_sum_7"),
            "flow_sum_14": _num(row, "flow_sum_14"),
            "flow_sum_21": _num(row, "flow_sum_21"),
            "distribution_pressure_score": _num(row, "distribution_pressure_score"),
            "flow_pct_rank_180": _num(row, "flow_pct_rank_180"),
            "z_scores": {
                "flow_z_90": _num(row, "flow_z_90"),
                "flow_z_180": _num(row, "flow_z_180"),
            },
        },
        "sentiment": {
            "sentiment_norm": _num(row, "sentiment_norm"),
            "sentiment_roll_pct_180d": _num(row, "sentiment_roll_pct_180d"),
            "z_scores": {
                "sentiment_z_30d": _num(row, "sentiment_z_30d"),
                "sentiment_z_90d": _num(row, "sentiment_z_90d"),
                "sentiment_z_180d": _num(row, "sentiment_z_180d"),
                "sentiment_z_365d": _num(row, "sentiment_z_365d"),
            },
        },
        "mvrv_composite": {
            "mvrv_composite_pct": _num(row, "mvrv_composite_pct"),
            "z_scores": {
                "mvrv_comp_z_90d": _num(row, "mvrv_comp_z_90d"),
                "mvrv_comp_z_180d": _num(row, "mvrv_comp_z_180d"),
                "mvrv_comp_z_365d": _num(row, "mvrv_comp_z_365d"),
            },
        },
        "mvrv_60d": {
            "mvrv_60d": _num(row, "mvrv_60d"),
            "mvrv_60d_pct_rank": _num(row, "mvrv_60d_pct_rank"),
            "mvrv_60d_dist_from_max": _num(row, "mvrv_60d_dist_from_max"),
            "is_falling": int(row.get("mvrv_60d_is_falling", 0) or 0),
            "is_flattening": int(row.get("mvrv_60d_is_flattening", 0) or 0),
            "is_rising": int(row.get("mvrv_60d_is_rising", 0) or 0),
        },
    }
