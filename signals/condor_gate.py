"""
Iron Condor Range Gate
======================

Decides whether an iron condor entry is safe by scoring the current
regime and applying hard vetoes.

Evaluated AFTER the main fusion cascade reaches NO_TRADE / TRANSITION_CHOP.
Never overrides directional signals.

Label definition (frozen):
    ±10% band, 7-day forward horizon.
    "Did BTC stay within ±10% of today's close over the next 7 calendar days?"

Calibration:
    Base rate: 58% of days stay in range.
    Gate at score ≥ 75 + no vetoes: 67% precision, ~6% pass rate.
    Optimised for capital preservation (fewer trades, higher win rate).

Score components (additive, max 100):
    +25  fusion state is NO_TRADE or TRANSITION_CHOP
    +20  MVRV LS level is neutral
    +15  no directional option signals firing  [diagnostic only — also a hard veto]
    +15  sentiment bucket is neutral
    +10  sentiment is flattening
    +10  distribution pressure below expanding median (no lookahead)
    + 5  MVRV-60d flat 7d+ AND mvrv_usd_60d in 0.88–1.02 (provisional, pending WF)
    + 5  whale regime is mixed (not directional)

Score penalties (subtractive):
    -15  MVRV strong uptrend or strong downtrend  [diagnostic — also a hard veto]
    -10  MDIA strong inflow                       [diagnostic — also a hard veto]
    -10  sentiment extreme (fear or greed bucket)

    NOTE on redundancy: option signals, MVRV trend, and MDIA strong inflow
    appear as both score penalties AND hard vetoes. The penalties exist so the
    score reflects regime quality for diagnostics/logging even when the veto
    would block execution anyway. At decision time, hard vetoes are the
    binding constraint — the score is informational.

Hard vetoes (instant block, checked independently of score):
    - OPTION_CALL or OPTION_PUT signal active
    - MVRV strong uptrend or strong downtrend
    - MDIA strong inflow (directional impulse)
    - Extreme sentiment persisting 5d+
    - Strong whale distribution
    - Realized volatility spike (ATR-based, see below)
    - Large gap day (|open-prev_close|/prev_close > 4%)

Precedence: Veto overrides score. Penalties are diagnostic and
threshold-shaping only. At execution time, hard vetoes are the
binding constraint.

Monitoring / recalibration triggers:
    - Precision drops below 62% over the last 30 condor trades
    - False-positive breakout rate exceeds 45% over last 30 trades
    - 5+ consecutive losing condor trades
    - Walk-forward threshold std exceeds 25 across recent 8 folds
    When any trigger fires, re-run chop_analysis + condor_walkforward
    and re-evaluate score weights and threshold.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration (frozen after calibration)
# ---------------------------------------------------------------------------

# Label: ±10% band, 7-day forward horizon
CONDOR_BAND_PCT = 0.10
CONDOR_HORIZON_DAYS = 7

# Gate threshold — score ≥ 75 required
# 67% precision, ~6% pass rate, ~1-2 trades/month
DEFAULT_SCORE_THRESHOLD = 75

# Cooldown between condor entries (days)
CONDOR_COOLDOWN_DAYS = 5

# Volatility veto thresholds
ATR_EXPANSION_THRESHOLD = 1.5   # 7d ATR / 30d ATR — veto if expanding
GAP_VETO_THRESHOLD = 0.04       # |open - prev_close| / prev_close


@dataclass
class CondorGateResult:
    """Result of the iron condor range gate evaluation."""
    score: float                    # 0-100 range score
    eligible: bool                  # True if score >= threshold and no hard vetoes
    veto_reasons: list[str]         # Hard veto reasons (if any)
    score_components: dict          # Breakdown for explainability
    threshold: float                # Threshold used


def compute_range_score(
    row: pd.Series,
    fusion_state: Optional[str] = None,
    dp_expanding_median: Optional[float] = None,
) -> tuple[float, dict]:
    """
    Compute range score from a single feature row.

    Args:
        row: Feature row from build_features_and_labels_from_raw.
        fusion_state: Optional fusion state string (e.g. "no_trade").
            If provided, used directly instead of looking for fusion_state_* columns.
        dp_expanding_median: Expanding median of distribution_pressure_score
            computed up to (and including) this row. Avoids lookahead bias.
            If None, falls back to 0.5 (neutral).

    Returns (score, components_dict).
    """
    components = {}
    score = 0.0

    # --- Positive contributors ---

    # Fusion chop state (+25)
    if fusion_state is not None:
        is_chop_state = fusion_state in ("no_trade", "transition_chop")
    else:
        is_no_trade = row.get("fusion_state_no_trade", 0) == 1
        is_chop = row.get("fusion_state_transition_chop", 0) == 1
        is_chop_state = is_no_trade or is_chop
    chop_contrib = 25.0 if is_chop_state else 0.0
    score += chop_contrib
    components["chop_state"] = chop_contrib

    # MVRV LS neutral (+20)
    mvrv_neutral = float(row.get("mvrv_ls_level_neutral", 0))
    mvrv_contrib = 20.0 * mvrv_neutral
    score += mvrv_contrib
    components["mvrv_neutral"] = mvrv_contrib

    # No option signals (+15) — diagnostic; also a hard veto
    opt_call = float(row.get("signal_option_call", 0))
    opt_put = float(row.get("signal_option_put", 0))
    no_opt = 1.0 if (opt_call == 0 and opt_put == 0) else 0.0
    no_opt_contrib = 15.0 * no_opt
    score += no_opt_contrib
    components["no_option_signal"] = no_opt_contrib

    # Sentiment neutral (+15)
    sent_neutral = float(row.get("sent_bucket_neutral", 0))
    sent_contrib = 15.0 * sent_neutral
    score += sent_contrib
    components["sent_neutral"] = sent_contrib

    # Sentiment flattening (+10)
    sent_flat = float(row.get("sent_is_flattening", 0))
    flat_contrib = 10.0 * sent_flat
    score += flat_contrib
    components["sent_flat"] = flat_contrib

    # Low distribution pressure (+10) — uses expanding median, no lookahead
    dp = float(row.get("distribution_pressure_score", 0.5))
    median_ref = dp_expanding_median if dp_expanding_median is not None else 0.5
    low_dp = 1.0 if dp <= median_ref else 0.0
    dp_contrib = 10.0 * low_dp
    score += dp_contrib
    components["low_dist_pressure"] = dp_contrib
    components["_dp_median_ref"] = round(median_ref, 4)

    # Whale mixed (+5)
    whale_mixed = float(row.get("whale_regime_mixed", 0))
    whale_contrib = 5.0 * whale_mixed
    score += whale_contrib
    components["whale_mixed"] = whale_contrib

    # MVRV-60d flat in undervalued zone (+5, provisional)
    # When MVRV-60d z-score is flat for 7+ consecutive days AND mvrv_usd_60d
    # is in the 0.9–1.0 range, historical analysis shows 78% safe rate
    # (vs 58% base) with tight 9.2% median range.
    #
    # Weight is +5 (provisional) pending walk-forward validation against all
    # other contributors on the same sample. Will promote to +10 if
    # incremental lift is confirmed out-of-sample.
    #
    # Uses mvrv_60d_flat_streak_7d (7+ consecutive flat days) instead of
    # single-day mvrv_60d_is_flattening to avoid flip-flop entries near
    # the ±0.5 z-score threshold boundary.
    #
    # Band hysteresis: entry requires [0.90, 1.00] inclusive.
    # Once active, stays active while [0.88, 1.02] to reduce churn.
    # The gate is stateless (row-level), so hysteresis is approximated:
    # we check the relaxed band and require the streak flag (which itself
    # implies the strict band was met at streak start).
    #
    # Source: features/management/commands/analyze_flat_z_range.py
    MVRV_60D_ENTRY_LO, MVRV_60D_ENTRY_HI = 0.90, 1.00
    MVRV_60D_STAY_LO, MVRV_60D_STAY_HI = 0.88, 1.02

    mvrv_60d_flat_streak = float(row.get("mvrv_60d_flat_streak_7d", 0))
    mvrv_60d_raw = float(row.get("mvrv_60d", row.get("mvrv_usd_60d", 0.0)))

    # Streak flag implies strict band was met at streak start;
    # current value only needs to be within the relaxed stay band.
    in_stay_band = MVRV_60D_STAY_LO <= mvrv_60d_raw <= MVRV_60D_STAY_HI
    is_flat_underval = 1.0 if (mvrv_60d_flat_streak == 1 and in_stay_band) else 0.0
    flat_underval_contrib = 5.0 * is_flat_underval
    score += flat_underval_contrib
    components["mvrv_60d_flat_underval"] = flat_underval_contrib

    # --- Penalties (diagnostic — overlap with hard vetoes is intentional) ---

    # MVRV strong trend (-15)
    mvrv_up = float(row.get("mvrv_ls_strong_uptrend", 0))
    mvrv_down = float(row.get("mvrv_ls_strong_downtrend", 0))
    trend_penalty = -15.0 if (mvrv_up == 1 or mvrv_down == 1) else 0.0
    score += trend_penalty
    components["mvrv_trend_penalty"] = trend_penalty

    # MDIA strong inflow (-10)
    mdia_strong = float(row.get("mdia_regime_strong_inflow", 0))
    mdia_penalty = -10.0 * mdia_strong
    score += mdia_penalty
    components["mdia_strong_penalty"] = mdia_penalty

    # Sentiment extreme (-10)
    ext_fear = float(row.get("sent_bucket_extreme_fear", 0))
    ext_greed = float(row.get("sent_bucket_extreme_greed", 0))
    sent_penalty = -10.0 if (ext_fear == 1 or ext_greed == 1) else 0.0
    score += sent_penalty
    components["sent_extreme_penalty"] = sent_penalty

    score = max(0.0, min(100.0, score))
    return score, components


def check_hard_vetoes(
    row: pd.Series,
    atr_ratio: Optional[float] = None,
    gap_pct: Optional[float] = None,
) -> list[str]:
    """
    Check hard veto conditions that block condor entry regardless of score.

    Args:
        row: Feature row.
        atr_ratio: ATR(7) / ATR(30) ratio. Veto if > ATR_EXPANSION_THRESHOLD.
            Computed by caller from OHLC to avoid lookahead.
        gap_pct: |open - prev_close| / prev_close. Veto if > GAP_VETO_THRESHOLD.

    Returns list of veto reason strings (empty = no vetoes).
    """
    vetoes = []

    # --- Directional signal vetoes ---
    if row.get("signal_option_call", 0) == 1:
        vetoes.append("OPTION_CALL_ACTIVE")
    if row.get("signal_option_put", 0) == 1:
        vetoes.append("OPTION_PUT_ACTIVE")

    # --- Trend/momentum vetoes ---
    if row.get("mvrv_ls_strong_uptrend", 0) == 1:
        vetoes.append("MVRV_STRONG_UPTREND")
    if row.get("mvrv_ls_strong_downtrend", 0) == 1:
        vetoes.append("MVRV_STRONG_DOWNTREND")
    if row.get("mdia_regime_strong_inflow", 0) == 1:
        vetoes.append("MDIA_STRONG_INFLOW")

    # --- Sentiment persistence vetoes ---
    if row.get("sent_extreme_greed_persist_5d", 0) == 1:
        vetoes.append("EXTREME_GREED_PERSIST_5D")
    if row.get("sent_extreme_fear_persist_5d", 0) == 1:
        vetoes.append("EXTREME_FEAR_PERSIST_5D")

    # --- Smart money directional veto ---
    if row.get("whale_regime_distribution_strong", 0) == 1:
        vetoes.append("WHALE_DISTRIB_STRONG")

    # --- Volatility / crash vetoes (from OHLC) ---
    if atr_ratio is not None and atr_ratio > ATR_EXPANSION_THRESHOLD:
        vetoes.append(f"ATR_EXPANSION({atr_ratio:.2f}>{ATR_EXPANSION_THRESHOLD})")
    if gap_pct is not None and gap_pct > GAP_VETO_THRESHOLD:
        vetoes.append(f"LARGE_GAP({gap_pct:.2%}>{GAP_VETO_THRESHOLD:.0%})")

    return vetoes


def compute_vol_metrics(ohlc_df: pd.DataFrame) -> tuple[Optional[float], Optional[float]]:
    """
    Compute volatility metrics from OHLC DataFrame for the latest row.

    Args:
        ohlc_df: DataFrame with columns btc_open, btc_high, btc_low, btc_close,
                 sorted by date ascending. Needs at least 30 rows.

    Returns:
        (atr_ratio, gap_pct) — either can be None if insufficient data.
        atr_ratio: ATR(7) / ATR(30). Values > 1.5 indicate expanding vol.
        gap_pct: |today_open - yesterday_close| / yesterday_close.
    """
    if len(ohlc_df) < 31:
        return None, None

    high = ohlc_df["btc_high"]
    low = ohlc_df["btc_low"]
    close = ohlc_df["btc_close"]
    opn = ohlc_df["btc_open"]

    # True Range
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr_7 = tr.rolling(7).mean().iloc[-1]
    atr_30 = tr.rolling(30).mean().iloc[-1]

    atr_ratio = float(atr_7 / atr_30) if atr_30 > 0 else None

    # Gap
    today_open = float(opn.iloc[-1])
    yesterday_close = float(close.iloc[-2])
    gap_pct = abs(today_open - yesterday_close) / yesterday_close if yesterday_close > 0 else None

    return atr_ratio, gap_pct


def evaluate_condor_gate(
    row: pd.Series,
    threshold: float = DEFAULT_SCORE_THRESHOLD,
    fusion_state: Optional[str] = None,
    dp_expanding_median: Optional[float] = None,
    atr_ratio: Optional[float] = None,
    gap_pct: Optional[float] = None,
) -> CondorGateResult:
    """
    Full condor gate evaluation: score + hard vetoes.

    Args:
        row: Feature row (from build_features_and_labels_from_raw).
        threshold: Minimum score to pass gate (default 75).
        fusion_state: Fusion state string for live scoring.
        dp_expanding_median: Expanding median of distribution_pressure_score
            up to this row. Prevents lookahead bias.
        atr_ratio: ATR(7)/ATR(30) for vol expansion veto.
        gap_pct: Today's gap size for crash veto.

    Returns:
        CondorGateResult with eligibility decision.
    """
    score, components = compute_range_score(
        row,
        fusion_state=fusion_state,
        dp_expanding_median=dp_expanding_median,
    )
    vetoes = check_hard_vetoes(row, atr_ratio=atr_ratio, gap_pct=gap_pct)

    eligible = (score >= threshold) and (len(vetoes) == 0)

    return CondorGateResult(
        score=score,
        eligible=eligible,
        veto_reasons=vetoes,
        score_components=components,
        threshold=threshold,
    )
