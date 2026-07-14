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
    map_flow_bucket,
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

# Exchange-flow bucket → (direction, magnitude) trader interpretation.
# Inflow to exchanges = sell pressure; outflow = buy pressure.
_FLOW_PRESSURE = {
    "strong_inflow": ("sell", "strong"),
    "inflow": ("sell", "moderate"),
    "neutral": ("neutral", "none"),
    "outflow": ("buy", "moderate"),
    "strong_outflow": ("buy", "strong"),
    "unknown": ("unknown", "unknown"),
}


def _flow_pressure(bucket: str, flow_z_90) -> dict:
    """Translate the flow bucket into a buy/sell pressure reading.

    ``value`` is the signed z-score (positive = sell pressure, negative = buy).
    """
    direction, magnitude = _FLOW_PRESSURE.get(bucket, ("unknown", "unknown"))
    if direction == "neutral":
        label = "NEUTRAL"
    elif direction == "unknown":
        label = "UNKNOWN"
    else:
        label = f"{direction.upper()} pressure ({magnitude})"
    return {
        "direction": direction,
        "magnitude": magnitude,
        "label": label,
        "value": flow_z_90,
    }


# Normalized reading: (value, z_col) sources per metric. Window = 90d
# (chosen data-driven via analyze_window_sweep — stability-dominant, weak/unstable
# standalone predictive power at every window).
_NORMALIZED_SOURCES = {
    "mvrv_60d": ("mvrv_60d", "mvrv_60d_z_90d"),
    "sentiment": ("sentiment_norm", "sentiment_z_90d"),
    "exchange_flow": ("flow_sum_7", "flow_z_90"),
    "mvrv_composite": ("mvrv_composite_pct", "mvrv_comp_z_90d"),
}


def classify_z_position(z) -> str:
    """Map a z-score to a plain-language position vs its 90-day baseline.

    Thresholds mirror the mvrv_composite / flow buckets (+/-0.5, +/-1.5).
    """
    if z is None or pd.isna(z):
        return "unknown"
    if z > 1.5:
        return "stretched_high"
    if z > 0.5:
        return "high"
    if z >= -0.5:
        return "normal"
    if z >= -1.5:
        return "low"
    return "stretched_low"


def ensure_normalized_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``mvrv_60d_z_90d`` if the CSV doesn't already carry it.

    The other 90-day z columns (flow_z_90, sentiment_z_90d, mvrv_comp_z_90d) are
    produced by the feature pipeline; mvrv_60d has no 90d z, so derive it here
    from the full series. Returns a copy when it adds the column.
    """
    if "mvrv_60d_z_90d" in df.columns or "mvrv_60d" not in df.columns:
        return df
    s = df["mvrv_60d"]
    roll_mean = s.rolling(90, min_periods=30).mean()
    roll_std = s.rolling(90, min_periods=30).std()
    out = df.copy()
    out["mvrv_60d_z_90d"] = (s - roll_mean) / (roll_std + 1e-9)
    return out


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

    normalized = {}
    for name, (value_col, z_col) in _NORMALIZED_SOURCES.items():
        z = _num(row, z_col)
        normalized[name] = {
            "value": _num(row, value_col),
            "z_90": z,
            "position": classify_z_position(z),
        }

    return {
        "normalized": normalized,
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
            "exchange_flow": map_flow_bucket(row),
            "sentiment": _active_flag_bucket(row, _SENTIMENT_BUCKET_FLAGS),
            "mvrv_composite": _active_flag_bucket(row, _MVRV_COMPOSITE_BUCKET_FLAGS),
        },
        "exchange_flow": {
            "pressure": _flow_pressure(
                map_flow_bucket(row), _num(row, "flow_z_90")
            ),
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
