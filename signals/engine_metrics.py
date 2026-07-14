# signals/engine_metrics.py
"""
Essential engine metrics + confluence score for a single feature row.

For each of four orthogonal metrics (mvrv_60d, sentiment, exchange_flow,
mvrv_composite) we expose only what matters: the raw value, its 90-day z-score,
and a human-readable position label. On top of those we compute a single
confluence score that rewards agreement across metrics and gates out noise.

Shared by the ``analyze_engine`` command and the AnalyzeEngine API endpoint.
"""
from __future__ import annotations

import pandas as pd

from signals.fusion import fuse_signals, FusionResult

# Normalized reading: (value_col, z_col) per metric. Window = 90d, chosen
# data-driven via analyze_window_sweep (stability-dominant; standalone predictive
# power is weak/unstable at every window, so metrics are combined via confluence).
_NORMALIZED_SOURCES = {
    "mvrv_60d": ("mvrv_60d", "mvrv_60d_z_90d"),
    "sentiment": ("sentiment_norm", "sentiment_z_90d"),
    "exchange_flow": ("flow_sum_7", "flow_z_90"),
    "mvrv_composite": ("mvrv_composite_pct", "mvrv_comp_z_90d"),
}

# Bullish orientation of each metric's z-score (+1: high z = bullish;
# -1: high z = bearish). See compute_engine_score docstring for rationale.
DIRECTION_SIGNS = {
    "mvrv_60d": +1,        # recent-buyer valuation rising vs its norm = bullish
    "exchange_flow": -1,   # inflow to exchanges = distribution/sell pressure = bearish
    "sentiment": -1,       # contrarian: greed = bearish, fear = bullish
    "mvrv_composite": -1,  # mean-reversion: overvalued = bearish, undervalued = bullish
}

NEUTRAL_BAND = 0.5   # |z| below this contributes nothing (noise gate)
Z_CAP = 2.0          # clip extreme z so one metric can't dominate the score
DIRECTION_THRESHOLD = 20  # |score| below this = neutral / no edge


def classify_z_position(z) -> str:
    """Map a z-score to a plain-language position vs its 90-day baseline.

    Thresholds mirror the codebase convention (+/-0.5, +/-1.5).
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


def compute_engine_score(normalized: dict) -> dict:
    """Confluence score from the four normalized metric z-scores.

    Method (heuristic, not calibrated alpha — the metrics are weak/unstable
    standalone predictors, so the score's job is to summarise agreement, not to
    claim edge):
      1. Orient each z bullish via DIRECTION_SIGNS.
      2. Gate noise: |z| < NEUTRAL_BAND contributes 0.
      3. Clip to +/-Z_CAP so no single metric dominates.
      4. net = sum of contributions; agreement = share of *active* metrics that
         match net's sign. conviction = net * agreement, so conflicting signals
         are damped toward zero ("chop / no real move").
      5. Scale to -100..+100 and label direction.
    """
    contributions = {}
    active = 0
    for name, sign in DIRECTION_SIGNS.items():
        z = normalized.get(name, {}).get("z_90")
        if z is None or pd.isna(z) or abs(z) < NEUTRAL_BAND:
            contributions[name] = 0.0
            continue
        d = sign * z
        contributions[name] = float(max(-Z_CAP, min(Z_CAP, d)))
        active += 1

    net = sum(contributions.values())
    if active > 0 and net != 0:
        agree = sum(
            1 for c in contributions.values() if c != 0 and (c > 0) == (net > 0)
        )
        agreement = agree / active
    else:
        agreement = 0.0

    conviction = net * agreement
    max_possible = Z_CAP * len(DIRECTION_SIGNS)
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
    """Build the reduced engine snapshot: fusion context, per-metric
    value/z/label, and the confluence score.
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

    return {
        "fusion": {
            "state": fusion_result.state.value,
            "confidence": fusion_result.confidence.value,
            "score": fusion_result.score,
            "bear_mode": bool(fusion_result.components.get("bear_mode", False)),
            "cycle_day": fusion_result.components.get("cycle_day"),
        },
        "metrics": metrics,
        "score": compute_engine_score(metrics),
    }
