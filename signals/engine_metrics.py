# signals/engine_metrics.py
"""
Essential engine snapshot + confluence score for a single feature row.

Two families of inputs, all oriented so positive = bullish and combined into one
confluence score:

  1. Normalized metrics (mvrv_60d, sentiment, exchange_flow, mvrv_composite):
     raw value + 90-day z-score + position label.
  2. Fusion components (mdia, whale, mvrv_ls): per-horizon +1/0/-1 buckets summed.

For scoring, every input is mapped to a common +/-Z_CAP contribution scale, noise
gated, then combined via net * agreement so conflicting signals damp toward zero.

Shared by the ``analyze_engine`` command and the AnalyzeEngine API endpoint.
"""
from __future__ import annotations

import pandas as pd

from signals.fusion import fuse_signals, FusionResult

# ── Normalized metrics: (value_col, z_col). Window = 90d (analyze_window_sweep). ──
_NORMALIZED_SOURCES = {
    "mvrv_60d": ("mvrv_60d", "mvrv_60d_z_90d"),
    "sentiment": ("sentiment_norm", "sentiment_z_90d"),
    "exchange_flow": ("flow_sum_7", "flow_z_90"),
    "mvrv_composite": ("mvrv_composite_pct", "mvrv_comp_z_90d"),
}

# Bullish orientation of each normalized metric's z (+1: high z bullish).
DIRECTION_SIGNS = {
    "mvrv_60d": -1,        # mean-reversion: recent buyers deep in profit = bearish
    "exchange_flow": -1,   # inflow to exchanges = sell pressure = bearish
    "sentiment": -1,       # contrarian: greed = bearish, fear = bullish
    "mvrv_composite": -1,  # mean-reversion: overvalued = bearish
}

# ── Fusion components: per-horizon bucket columns summed. ─────────────────────
_MDIA_COLS = ["mdia_bucket_1d", "mdia_bucket_2d", "mdia_bucket_4d", "mdia_bucket_7d"]
_WHALE_MEGA_COLS = [
    "whale_mega_bucket_1d", "whale_mega_bucket_2d", "whale_mega_bucket_4d",
    "whale_mega_bucket_7d", "whale_mega_bucket_14d",
]
_WHALE_SMALL_COLS = [
    "whale_small_bucket_1d", "whale_small_bucket_2d",
    "whale_small_bucket_4d", "whale_small_bucket_7d",
]
_MVRV_LS_COLS = [
    "mvrv_ls_trend_2d", "mvrv_ls_trend_4d", "mvrv_ls_trend_7d", "mvrv_ls_trend_14d",
]
NEUTRAL_BAND = 0.5       # |oriented z| below this = 0 vote (matches "normal" label)
DIRECTION_THRESHOLD = 2  # |score| below this (on the -7..+7 scale) = neutral


def classify_z_position(z) -> str:
    """Map a z-score to a plain-language position vs its 90-day baseline."""
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
    """Add ``mvrv_60d_z_90d`` if the CSV doesn't already carry it."""
    if "mvrv_60d_z_90d" in df.columns or "mvrv_60d" not in df.columns:
        return df
    s = df["mvrv_60d"]
    roll_mean = s.rolling(90, min_periods=30).mean()
    roll_std = s.rolling(90, min_periods=30).std()
    out = df.copy()
    out["mvrv_60d_z_90d"] = (s - roll_mean) / (roll_std + 1e-9)
    return out


def _num(row: pd.Series, col: str):
    val = row.get(col, None)
    if val is None or pd.isna(val):
        return None
    return float(val)


def _collapse_horizons(row: pd.Series, cols: list[str], prefix: str, flip: bool = False):
    """Collapse each present horizon bucket to a directional -1/0/+1 vote.

    Magnitude is intentionally discarded here: a strong ``mdia`` bucket (+/-2)
    counts as one directional vote, same as +/-1, so no single horizon can carry
    a whole component. Returns (horizons_dict, sum_of_votes, count).
    """
    horizons = {}
    total = 0
    count = 0
    for c in cols:
        v = row.get(c, None)
        if v is None or pd.isna(v):
            continue
        vote = 0 if v == 0 else (1 if v > 0 else -1)
        if flip:
            vote = -vote
        horizons[c.replace(prefix, "")] = vote
        total += vote
        count += 1
    return horizons, total, count


def compute_fusion_components(row: pd.Series) -> dict:
    """Per-horizon fusion buckets collapsed to directional votes, oriented so
    positive = bullish.

    mdia is sign-flipped (its raw buckets are negative-when-bullish: inflow < 0).
    whale combines mega + small; mvrv_ls uses its trend horizons.
    """
    mdia_h, mdia_sum, mdia_n = _collapse_horizons(row, _MDIA_COLS, "mdia_bucket_", flip=True)
    mega_h, mega_sum, mega_n = _collapse_horizons(row, _WHALE_MEGA_COLS, "whale_mega_bucket_")
    small_h, small_sum, small_n = _collapse_horizons(row, _WHALE_SMALL_COLS, "whale_small_bucket_")
    mvrv_h, mvrv_sum, mvrv_n = _collapse_horizons(row, _MVRV_LS_COLS, "mvrv_ls_trend_")

    return {
        "mdia": {"sum": mdia_sum, "n": mdia_n, "horizons": mdia_h},
        "whale": {
            "sum": mega_sum + small_sum,
            "n": mega_n + small_n,
            "mega_sum": mega_sum,
            "small_sum": small_sum,
        },
        "mvrv_ls": {"sum": mvrv_sum, "n": mvrv_n, "horizons": mvrv_h},
    }


def _z_vote(z, sign: int) -> int:
    """Collapse an oriented z-score to a -1 / 0 / +1 vote (0.5 neutral band)."""
    if z is None or pd.isna(z):
        return 0
    o = sign * z
    if o > NEUTRAL_BAND:
        return 1
    if o < -NEUTRAL_BAND:
        return -1
    return 0


def _fusion_vote(oriented_sum: float, n: int) -> int:
    """Collapse a fusion component to a -1 / 0 / +1 vote via the mean of its
    horizons, rounded on the same 0.5 band as the metrics.

    e.g. mvrv_ls sum -3 over 4 horizons = -0.75 -> -1; mdia +1 over 4 = +0.25 -> 0.
    """
    if n <= 0:
        return 0
    mean = oriented_sum / n
    if mean >= NEUTRAL_BAND:
        return 1
    if mean <= -NEUTRAL_BAND:
        return -1
    return 0


def compute_engine_score(normalized: dict, fusion_components: dict) -> dict:
    """Discrete confluence score: each of the 7 inputs votes -1/0/+1, summed.

    Range -7..+7. Higher = more bullish. Direction (vote magnitude only, not
    calibrated alpha) is oriented per DIRECTION_SIGNS for the normalized metrics
    and already baked into the oriented sum for the fusion components.
    """
    votes = {}
    for name, sign in DIRECTION_SIGNS.items():
        votes[name] = _z_vote(normalized.get(name, {}).get("z_90"), sign)
    for name, fc in fusion_components.items():
        votes[name] = _fusion_vote(fc["sum"], fc.get("n", 0))

    value = sum(votes.values())
    active = sum(1 for v in votes.values() if v != 0)

    if value >= DIRECTION_THRESHOLD:
        direction = "bullish"
    elif value <= -DIRECTION_THRESHOLD:
        direction = "bearish"
    else:
        direction = "neutral"

    return {
        "value": value,
        "direction": direction,
        "active": active,
        "votes": votes,
    }


def collect_essential_metrics(row: pd.Series, fusion_result: FusionResult = None) -> dict:
    """Reduced engine snapshot: fusion context, per-metric value/z/label,
    fusion components, and the combined confluence score.
    """
    if fusion_result is None:
        fusion_result = fuse_signals(row)

    metrics = {}
    for name, (value_col, z_col) in _NORMALIZED_SOURCES.items():
        z = _num(row, z_col)
        metrics[name] = {
            "value": _num(row, value_col),
            "z_90": z,
            "label": classify_z_position(z),
        }

    fusion_components = compute_fusion_components(row)

    return {
        "fusion": {
            "state": fusion_result.state.value,
            "confidence": fusion_result.confidence.value,
            "score": fusion_result.score,
            "bear_mode": bool(fusion_result.components.get("bear_mode", False)),
            "cycle_day": fusion_result.components.get("cycle_day"),
        },
        "metrics": metrics,
        "fusion_components": fusion_components,
        "score": compute_engine_score(metrics, fusion_components),
    }
