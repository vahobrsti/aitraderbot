"""
Chop / Range Analysis for Iron Condor Entry Filtering
======================================================

Joins the feature matrix with 7-day range labels and fusion states to answer:
  1. What features are common on RANGE_7D days?
  2. Which conditions best separate range vs breakout days?
  3. How often do chop states (NO_TRADE, TRANSITION_CHOP) align with actual range?
  4. Builds a practical "range score" gate for iron condor entries.

Run via:  python manage.py chop_analysis --out reports/chop_report.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from datafeed.models import RawDailyData
from features.feature_builder import build_features_and_labels_from_raw
from features.metrics.range_label import calculate as calc_range_labels
from signals.fusion import add_fusion_features, MarketState


# ---------------------------------------------------------------------------
# Feature groups relevant to chop detection
# ---------------------------------------------------------------------------
CHOP_FEATURE_GROUPS = {
    "fusion_state": [
        "fusion_state_no_trade",
        "fusion_state_transition_chop",
    ],
    "mdia_neutral": [
        "mdia_regime_aging",
        "mdia_regime_neutral",
    ],
    "mvrv_neutral": [
        "mvrv_ls_level_neutral",
        "mvrv_ls_mixed",
        "mvrv_ls_conflict",
    ],
    "whale_neutral": [
        "whale_regime_mixed",
    ],
    "sentiment_flat": [
        "sent_bucket_neutral",
        "sent_is_flattening",
    ],
    "exchange_flow_calm": [
        "distribution_pressure_score",  # continuous — will be binned
    ],
    "option_signals_absent": [
        "signal_option_call",
        "signal_option_put",
    ],
    "volatility_proxy": [
        "range_7d_max_up_pct",   # from range_label (only on historical)
        "range_7d_max_down_pct",
    ],
}


def build_analysis_dataframe(threshold: float = 0.10) -> pd.DataFrame:
    """
    Pull raw data from DB, build features + fusion + range labels.
    Returns a single DataFrame ready for analysis.
    """
    qs = RawDailyData.objects.order_by("date").values()
    raw = pd.DataFrame.from_records(qs)
    if raw.empty:
        raise ValueError("No RawDailyData rows found in database.")

    # Build features (14d horizon / 5% target — standard)
    feats = build_features_and_labels_from_raw(raw, horizon_days=14, target_return=0.05)

    # Add fusion state columns
    feats = add_fusion_features(feats)

    # Build range labels from the raw OHLC (need original raw df with date index)
    raw_indexed = raw.copy().sort_values("date").set_index("date")
    range_df = calc_range_labels(raw_indexed, horizon=7, threshold=threshold)

    # Join range labels onto feature matrix (inner join on date index)
    combined = feats.join(range_df, how="inner")

    return combined


# ---------------------------------------------------------------------------
# Analysis 1: Feature prevalence on RANGE_7D vs breakout days
# ---------------------------------------------------------------------------
def feature_prevalence(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each binary feature, compute:
      - prevalence on RANGE_7D days
      - prevalence on breakout days
      - lift = range_rate / breakout_rate
    """
    has_label = df["range_7d_label"].notna()
    labeled = df[has_label].copy()

    is_range = labeled["range_7d_binary"] == 1
    range_df = labeled[is_range]
    breakout_df = labeled[~is_range]

    # Collect all binary (0/1) columns
    binary_cols = [
        c for c in labeled.columns
        if labeled[c].dropna().isin([0, 1]).all()
        and c not in ("range_7d_binary",)
        and labeled[c].dtype in ("int64", "float64", "Int64")
    ]

    rows = []
    for col in binary_cols:
        range_rate = range_df[col].mean() if len(range_df) else 0
        breakout_rate = breakout_df[col].mean() if len(breakout_df) else 0
        lift = (range_rate / breakout_rate) if breakout_rate > 0 else np.nan
        rows.append({
            "feature": col,
            "range_rate": round(range_rate, 4),
            "breakout_rate": round(breakout_rate, 4),
            "lift": round(lift, 3) if not np.isnan(lift) else np.nan,
            "range_n": int(range_df[col].sum()),
            "breakout_n": int(breakout_df[col].sum()),
        })

    result = pd.DataFrame(rows).sort_values("lift", ascending=False)
    return result


# ---------------------------------------------------------------------------
# Analysis 2: Fusion state alignment with actual range outcomes
# ---------------------------------------------------------------------------
def fusion_state_alignment(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each fusion_market_state, compute:
      - total days
      - RANGE_7D count and rate
      - breakout count and rate
      - precision as a chop predictor
    """
    has_label = df["range_7d_label"].notna()
    labeled = df[has_label].copy()

    rows = []
    for state in labeled["fusion_market_state"].unique():
        mask = labeled["fusion_market_state"] == state
        subset = labeled[mask]
        n = len(subset)
        range_n = int((subset["range_7d_binary"] == 1).sum())
        breakout_n = n - range_n

        # Breakout type breakdown
        bo_up = int((subset["range_7d_label"] == "BREAKOUT_UP").sum())
        bo_down = int((subset["range_7d_label"] == "BREAKOUT_DOWN").sum())
        bo_both = int((subset["range_7d_label"] == "BREAKOUT_BOTH").sum())

        rows.append({
            "fusion_state": state,
            "total_days": n,
            "range_7d_n": range_n,
            "range_7d_rate": round(range_n / n, 3) if n else 0,
            "breakout_n": breakout_n,
            "breakout_up": bo_up,
            "breakout_down": bo_down,
            "breakout_both": bo_both,
        })

    return pd.DataFrame(rows).sort_values("range_7d_rate", ascending=False)


# ---------------------------------------------------------------------------
# Analysis 3: Composite chop conditions vs range outcome
# ---------------------------------------------------------------------------
def chop_condition_combos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Test combinations of chop-indicative conditions and measure
    their precision at predicting RANGE_7D.
    """
    has_label = df["range_7d_label"].notna()
    labeled = df[has_label].copy()

    # Build boolean chop indicators
    is_chop_state = (
        labeled.get("fusion_state_no_trade", 0).astype(bool)
        | labeled.get("fusion_state_transition_chop", 0).astype(bool)
    )
    mvrv_neutral = labeled.get("mvrv_ls_level_neutral", 0).astype(bool)
    mdia_not_inflow = ~(
        labeled.get("mdia_regime_inflow", 0).astype(bool)
        | labeled.get("mdia_regime_strong_inflow", 0).astype(bool)
    )
    whale_not_directional = ~(
        labeled.get("whale_regime_broad_accum", 0).astype(bool)
        | labeled.get("whale_regime_strategic_accum", 0).astype(bool)
        | labeled.get("whale_regime_distribution", 0).astype(bool)
        | labeled.get("whale_regime_distribution_strong", 0).astype(bool)
    )
    sent_neutral = labeled.get("sent_bucket_neutral", 0).astype(bool)
    sent_flat = labeled.get("sent_is_flattening", 0).astype(bool)
    no_option_signal = (
        ~labeled.get("signal_option_call", 0).astype(bool)
        & ~labeled.get("signal_option_put", 0).astype(bool)
    )

    # Low distribution pressure (below median)
    dp = labeled.get("distribution_pressure_score", pd.Series(0.5, index=labeled.index))
    dp_median = dp.median()
    low_dp = dp <= dp_median

    conditions = {
        "chop_state": is_chop_state,
        "mvrv_neutral": mvrv_neutral,
        "mdia_not_inflow": mdia_not_inflow,
        "whale_not_directional": whale_not_directional,
        "sent_neutral": sent_neutral,
        "sent_flat": sent_flat,
        "no_option_signal": no_option_signal,
        "low_dist_pressure": low_dp,
    }

    is_range = labeled["range_7d_binary"] == 1
    base_range_rate = is_range.mean()

    rows = []

    # Individual conditions
    for name, mask in conditions.items():
        n = mask.sum()
        if n == 0:
            continue
        range_rate = is_range[mask].mean()
        rows.append({
            "condition": name,
            "n_days": int(n),
            "range_7d_rate": round(range_rate, 3),
            "lift_vs_base": round(range_rate / base_range_rate, 2) if base_range_rate > 0 else np.nan,
        })

    # Key combinations
    combos = {
        "chop_state + mvrv_neutral": is_chop_state & mvrv_neutral,
        "chop_state + sent_neutral": is_chop_state & sent_neutral,
        "chop_state + no_option_signal": is_chop_state & no_option_signal,
        "chop_state + mvrv_neutral + sent_neutral": is_chop_state & mvrv_neutral & sent_neutral,
        "chop_state + mvrv_neutral + no_option_signal": is_chop_state & mvrv_neutral & no_option_signal,
        "chop_state + whale_not_dir + sent_flat": is_chop_state & whale_not_directional & sent_flat,
        "full_chop_gate": (
            is_chop_state & mvrv_neutral & no_option_signal
            & sent_neutral & low_dp
        ),
        "relaxed_chop_gate": (
            is_chop_state & no_option_signal & low_dp
        ),
    }

    for name, mask in combos.items():
        n = mask.sum()
        if n == 0:
            continue
        range_rate = is_range[mask].mean()
        rows.append({
            "condition": name,
            "n_days": int(n),
            "range_7d_rate": round(range_rate, 3),
            "lift_vs_base": round(range_rate / base_range_rate, 2) if base_range_rate > 0 else np.nan,
        })

    return pd.DataFrame(rows).sort_values("range_7d_rate", ascending=False)


# ---------------------------------------------------------------------------
# Analysis 4: Range score construction
# ---------------------------------------------------------------------------
def compute_range_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a composite range_score (0-100) from chop-indicative features.
    Higher score = more likely to stay in range = safer for iron condor.

    Components (additive, each 0 or weight):
      +25  fusion state is NO_TRADE or TRANSITION_CHOP
      +20  MVRV LS level is neutral
      +15  no directional option signals firing
      +15  sentiment bucket is neutral
      +10  sentiment is flattening
      +10  distribution pressure below median
      + 5  whale regime is mixed (not directional)

    Penalty (subtractive):
      -20  any OPTION_CALL or OPTION_PUT signal active
      -15  MVRV strong uptrend or strong downtrend
      -10  MDIA strong inflow (directional impulse)
      -10  sentiment extreme (fear or greed bucket)
    """
    score = pd.Series(0.0, index=df.index)

    # Positive contributors
    score += 25 * (
        df.get("fusion_state_no_trade", 0).astype(float)
        + df.get("fusion_state_transition_chop", 0).astype(float)
    ).clip(upper=1)

    score += 20 * df.get("mvrv_ls_level_neutral", 0).astype(float)

    no_opt = (
        (1 - df.get("signal_option_call", 0).astype(float))
        * (1 - df.get("signal_option_put", 0).astype(float))
    )
    score += 15 * no_opt

    score += 15 * df.get("sent_bucket_neutral", 0).astype(float)
    score += 10 * df.get("sent_is_flattening", 0).astype(float)

    dp = df.get("distribution_pressure_score", pd.Series(0.5, index=df.index))
    dp_median = dp.median()
    score += 10 * (dp <= dp_median).astype(float)

    score += 5 * df.get("whale_regime_mixed", 0).astype(float)

    # Penalties
    score -= 20 * (
        df.get("signal_option_call", 0).astype(float)
        + df.get("signal_option_put", 0).astype(float)
    ).clip(upper=1)

    score -= 15 * (
        df.get("mvrv_ls_strong_uptrend", 0).astype(float)
        + df.get("mvrv_ls_strong_downtrend", 0).astype(float)
    ).clip(upper=1)

    score -= 10 * df.get("mdia_regime_strong_inflow", 0).astype(float)

    score -= 10 * (
        df.get("sent_bucket_extreme_fear", 0).astype(float)
        + df.get("sent_bucket_extreme_greed", 0).astype(float)
    ).clip(upper=1)

    # Clamp to 0-100
    score = score.clip(0, 100)

    df = df.copy()
    df["range_score"] = score
    return df


# ---------------------------------------------------------------------------
# Analysis 5: Range score calibration — how well does it predict RANGE_7D?
# ---------------------------------------------------------------------------
def range_score_calibration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin range_score into deciles and measure actual RANGE_7D rate per bin.
    """
    has_label = df["range_7d_label"].notna()
    labeled = df[has_label].copy()

    if "range_score" not in labeled.columns:
        labeled = compute_range_score(labeled)

    # Create score bins
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    labeled["score_bin"] = pd.cut(labeled["range_score"], bins=bins, include_lowest=True)

    rows = []
    for bin_label, group in labeled.groupby("score_bin", observed=True):
        n = len(group)
        range_n = int((group["range_7d_binary"] == 1).sum())
        rows.append({
            "score_bin": str(bin_label),
            "n_days": n,
            "range_7d_n": range_n,
            "range_7d_rate": round(range_n / n, 3) if n else 0,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Do-not-trade conditions (breakout risk)
# ---------------------------------------------------------------------------
def breakout_risk_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify conditions that strongly predict breakout (anti-chop).
    These become "do-not-trade" filters for iron condor.
    """
    has_label = df["range_7d_label"].notna()
    labeled = df[has_label].copy()
    is_range = labeled["range_7d_binary"] == 1
    base_breakout_rate = (~is_range).mean()

    conditions = {
        "strong_bullish": labeled.get("fusion_state_strong_bullish", 0).astype(bool),
        "early_recovery": labeled.get("fusion_state_early_recovery", 0).astype(bool),
        "bear_continuation": labeled.get("fusion_state_bear_continuation", 0).astype(bool),
        "momentum": labeled.get("fusion_state_momentum", 0).astype(bool),
        "option_call_active": labeled.get("signal_option_call", 0).astype(bool),
        "option_put_active": labeled.get("signal_option_put", 0).astype(bool),
        "mvrv_strong_uptrend": labeled.get("mvrv_ls_strong_uptrend", 0).astype(bool),
        "mvrv_strong_downtrend": labeled.get("mvrv_ls_strong_downtrend", 0).astype(bool),
        "mdia_strong_inflow": labeled.get("mdia_regime_strong_inflow", 0).astype(bool),
        "extreme_greed": labeled.get("sent_bucket_extreme_greed", 0).astype(bool),
        "extreme_fear": labeled.get("sent_bucket_extreme_fear", 0).astype(bool),
        "whale_broad_accum": labeled.get("whale_regime_broad_accum", 0).astype(bool),
        "whale_distrib_strong": labeled.get("whale_regime_distribution_strong", 0).astype(bool),
    }

    rows = []
    for name, mask in conditions.items():
        n = int(mask.sum())
        if n == 0:
            continue
        breakout_rate = (~is_range[mask]).mean()
        rows.append({
            "condition": name,
            "n_days": n,
            "breakout_rate": round(breakout_rate, 3),
            "lift_vs_base": round(breakout_rate / base_breakout_rate, 2) if base_breakout_rate > 0 else np.nan,
        })

    return pd.DataFrame(rows).sort_values("breakout_rate", ascending=False)


# ---------------------------------------------------------------------------
# Master report
# ---------------------------------------------------------------------------
def run_full_analysis(out_dir: Optional[str] = None) -> dict:
    """
    Run all analyses and optionally save CSVs.
    Returns dict of DataFrames.
    """
    print("Building analysis dataframe from DB...")
    df = build_analysis_dataframe()
    df = compute_range_score(df)

    print(f"  Total rows: {len(df)}")
    has_label = df["range_7d_label"].notna()
    labeled = df[has_label]
    print(f"  Rows with 7d label: {len(labeled)}")

    range_n = (labeled["range_7d_binary"] == 1).sum()
    print(f"  RANGE_7D: {range_n} ({range_n/len(labeled)*100:.1f}%)")
    print(f"  Breakout: {len(labeled)-range_n} ({(len(labeled)-range_n)/len(labeled)*100:.1f}%)")
    print()

    print("1. Feature prevalence (range vs breakout)...")
    prevalence = feature_prevalence(df)

    print("2. Fusion state alignment...")
    alignment = fusion_state_alignment(df)

    print("3. Chop condition combos...")
    combos = chop_condition_combos(df)

    print("4. Range score calibration...")
    calibration = range_score_calibration(df)

    print("5. Breakout risk conditions (do-not-trade)...")
    breakout_risk = breakout_risk_conditions(df)

    results = {
        "prevalence": prevalence,
        "fusion_alignment": alignment,
        "chop_combos": combos,
        "range_score_calibration": calibration,
        "breakout_risk": breakout_risk,
        "full_data": df,
    }

    if out_dir:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        prevalence.to_csv(out_path / "feature_prevalence.csv", index=False)
        alignment.to_csv(out_path / "fusion_state_alignment.csv", index=False)
        combos.to_csv(out_path / "chop_condition_combos.csv", index=False)
        calibration.to_csv(out_path / "range_score_calibration.csv", index=False)
        breakout_risk.to_csv(out_path / "breakout_risk_conditions.csv", index=False)

        # Export full labeled dataset for further exploration
        export_cols = [c for c in df.columns if c != "range_7d_label" or True]
        df[export_cols].to_csv(out_path / "chop_analysis_full.csv")

        print(f"\nAll reports saved to {out_path}/")

    # Print summary tables
    print("\n" + "=" * 70)
    print("FUSION STATE ALIGNMENT WITH 5-DAY RANGE")
    print("=" * 70)
    print(alignment.to_string(index=False))

    print("\n" + "=" * 70)
    print("CHOP CONDITION COMBOS — RANGE_7D PRECISION")
    print("=" * 70)
    print(combos.to_string(index=False))

    print("\n" + "=" * 70)
    print("RANGE SCORE CALIBRATION")
    print("=" * 70)
    print(calibration.to_string(index=False))

    print("\n" + "=" * 70)
    print("BREAKOUT RISK CONDITIONS (DO-NOT-TRADE)")
    print("=" * 70)
    print(breakout_risk.to_string(index=False))

    print("\n" + "=" * 70)
    print("TOP 20 FEATURES BY LIFT (range vs breakout)")
    print("=" * 70)
    print(prevalence.head(20).to_string(index=False))

    return results
