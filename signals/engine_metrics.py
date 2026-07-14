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
    "mvrv_60d": +1,        # recent-buyer valuation rising vs norm = bullish
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
# Max attainable |sum| per component (used to normalise onto the +/-Z_CAP scale).
# mdia: 4 horizons x |2|; whale: 9 horizons x |1|; mvrv_ls: 4 horizons x |1|.
_FUSION_MAX_MAG = {"mdia": 8.0, "whale": 9.0, "mvrv_ls": 4.0}

NEUTRAL_BAND = 0.5   # |contribution| below this = noise, contributes 0
Z_CAP = 2.0          # common contribution scale; no input dominates
DIRECTION_THRESHOLD = 20  # |score| below this = neutral / no edge


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


def _sum_cols(row: pd.Series, cols: list[str]) -> float:
    """Sum present, non-NaN horizon buckets (missing horizons count as 0)."""
    total = 0.0
    for c in cols:
        v = row.get(c, None)
        if v is not None and not pd.isna(v):
            total += float(v)
    return total


def compute_fusion_components(row: pd.Series) -> dict:
    """Sum per-horizon fusion buckets, oriented so positive = bullish.

    mdia is sign-flipped (its raw buckets are negative-when-bullish: inflow < 0).
    whale combines mega + small; mvrv_ls sums its trend horizons.
    """
    mdia_raw = _sum_cols(row, _MDIA_COLS)
    mega = _sum_cols(row, _WHALE_MEGA_COLS)
    small = _sum_cols(row, _WHALE_SMALL_COLS)
    mvrv_ls_raw = _sum_cols(row, _MVRV_LS_COLS)

    def _oriented_horizons(cols, prefix, flip=False):
        out = {}
        for c in cols:
            v = row.get(c, None)
            iv = 0 if v is None or pd.isna(v) else int(v)
            out[c.replace(prefix, "")] = -iv if flip else iv
        return out

    return {
        "mdia": {
            "sum": -mdia_raw,
            "max_mag": _FUSION_MAX_MAG["mdia"],
            "horizons": _oriented_horizons(_MDIA_COLS, "mdia_bucket_", flip=True),
        },
        "whale": {
            "sum": mega + small,
            "max_mag": _FUSION_MAX_MAG["whale"],
            "mega_sum": mega,
            "small_sum": small,
        },
        "mvrv_ls": {
            "sum": mvrv_ls_raw,
            "max_mag": _FUSION_MAX_MAG["mvrv_ls"],
            "horizons": _oriented_horizons(_MVRV_LS_COLS, "mvrv_ls_trend_"),
        },
    }


def _gate_clip(x: float) -> float:
    """Clip to +/-Z_CAP, then zero out anything inside the neutral band."""
    x = max(-Z_CAP, min(Z_CAP, x))
    return 0.0 if abs(x) < NEUTRAL_BAND else float(x)


def compute_engine_score(normalized: dict, fusion_components: dict) -> dict:
    """Confluence score across the 4 normalized metrics + 3 fusion components.

    Heuristic, not calibrated alpha (metrics are weak/unstable standalone). Each
    input is put on a common +/-Z_CAP scale, noise-gated, then combined:
    conviction = net * agreement, so conflicting inputs damp toward zero.
    """
    contributions = {}

    # Normalized metrics: z already ~standardised, cap + gate directly.
    for name, sign in DIRECTION_SIGNS.items():
        z = normalized.get(name, {}).get("z_90")
        contributions[name] = 0.0 if z is None or pd.isna(z) else _gate_clip(sign * z)

    # Fusion components: normalise sum onto the same +/-Z_CAP scale, then gate.
    for name, fc in fusion_components.items():
        max_mag = fc.get("max_mag", 0.0)
        contributions[name] = (
            0.0 if max_mag <= 0 else _gate_clip(fc["sum"] / max_mag * Z_CAP)
        )

    active = sum(1 for c in contributions.values() if c != 0)
    net = sum(contributions.values())
    if active > 0 and net != 0:
        agree = sum(
            1 for c in contributions.values() if c != 0 and (c > 0) == (net > 0)
        )
        agreement = agree / active
    else:
        agreement = 0.0

    conviction = net * agreement
    max_possible = Z_CAP * len(contributions)
    value = round(conviction / max_possible * 100)

    if value >= DIRECTION_THRESHOLD:
        direction = "bullish"
    elif value <= -DIRECTION_THRESHOLD:
        direction = "bearish"
    else:
        direction = "neutral"

    return {
        "value": value,
        "direction": direction,
        "net": round(net, 3),
        "agreement": round(agreement, 3),
        "active": active,
        "contributions": {k: round(v, 3) for k, v in contributions.items()},
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
