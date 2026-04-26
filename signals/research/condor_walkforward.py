"""
Walk-Forward Validation for Iron Condor Range Gate
===================================================

Strict walk-forward: train period → future test period only.
No look-ahead bias — threshold is calibrated on train, evaluated on test.

Tracks:
  - Precision/recall for RANGE_7D
  - Tail-loss on false positives (breakout days where gate said "go")
  - Gate pass rate (how often the gate fires)
  - Breakout type breakdown on false positives

Usage:
    python manage.py condor_walkforward --out reports/condor_wf
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from datafeed.models import RawDailyData
from features.feature_builder import build_features_and_labels_from_raw
from features.metrics.range_label import calculate as calc_range_labels
from signals.fusion import add_fusion_features
from signals.condor_gate import compute_range_score, check_hard_vetoes


@dataclass
class WalkForwardFold:
    """Results for a single walk-forward fold."""
    fold_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    threshold: float
    # Test-period metrics
    test_n: int
    gate_pass_n: int
    gate_pass_rate: float
    true_range_n: int       # gate passed AND was RANGE_7D
    false_positive_n: int   # gate passed BUT was breakout
    precision: float        # true_range / gate_pass
    recall: float           # true_range / total_range_in_test
    # False positive analysis
    fp_breakout_up: int
    fp_breakout_down: int
    fp_breakout_both: int
    fp_avg_max_move: float  # avg of max(up_pct, down_pct) on FP days
    fp_worst_move: float    # worst single move on FP days


def _build_full_dataset() -> pd.DataFrame:
    """Build features + fusion + range labels from DB. No lookahead bias."""
    qs = RawDailyData.objects.order_by("date").values()
    raw = pd.DataFrame.from_records(qs)
    if raw.empty:
        raise ValueError("No RawDailyData rows found.")

    feats = build_features_and_labels_from_raw(raw, horizon_days=14, target_return=0.05)
    feats = add_fusion_features(feats)

    raw_indexed = raw.copy().sort_values("date").set_index("date")
    range_df = calc_range_labels(raw_indexed, horizon=7, threshold=0.10)
    combined = feats.join(range_df, how="inner")

    # Expanding median of distribution_pressure_score (no lookahead)
    if "distribution_pressure_score" in combined.columns:
        dp_expanding = combined["distribution_pressure_score"].expanding().median()
    else:
        dp_expanding = pd.Series(0.5, index=combined.index)

    # Precompute ATR ratio and gap pct from raw OHLC (no lookahead)
    ohlc = raw_indexed[["btc_open", "btc_high", "btc_low", "btc_close"]].copy()
    prev_close = ohlc["btc_close"].shift(1)
    tr = pd.concat([
        ohlc["btc_high"] - ohlc["btc_low"],
        (ohlc["btc_high"] - prev_close).abs(),
        (ohlc["btc_low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_7 = tr.rolling(7).mean()
    atr_30 = tr.rolling(30).mean()
    atr_ratio_series = (atr_7 / atr_30).reindex(combined.index)
    gap_pct_series = ((ohlc["btc_open"] - prev_close).abs() / prev_close).reindex(combined.index)

    # Add range_score column (with expanding median, no lookahead)
    scores = []
    for i in range(len(combined)):
        row = combined.iloc[i]
        dp_med = float(dp_expanding.iloc[i]) if not pd.isna(dp_expanding.iloc[i]) else 0.5
        s, _ = compute_range_score(row, dp_expanding_median=dp_med)
        scores.append(s)
    combined["range_score"] = scores

    # Add hard veto column (with vol metrics)
    veto_flags = []
    for i in range(len(combined)):
        row = combined.iloc[i]
        atr_r = float(atr_ratio_series.iloc[i]) if not pd.isna(atr_ratio_series.iloc[i]) else None
        gap_p = float(gap_pct_series.iloc[i]) if not pd.isna(gap_pct_series.iloc[i]) else None
        vetoes = check_hard_vetoes(row, atr_ratio=atr_r, gap_pct=gap_p)
        veto_flags.append(len(vetoes) > 0)
    combined["has_hard_veto"] = veto_flags

    return combined


def _calibrate_threshold(
    train: pd.DataFrame,
    min_precision: float = 0.65,
    min_pass_rate: float = 0.05,
) -> float:
    """
    Find the lowest threshold that achieves min_precision on train data.
    Also requires at least min_pass_rate gate passes to avoid degenerate thresholds.
    """
    has_label = train["range_7d_label"].notna()
    labeled = train[has_label]
    if len(labeled) == 0:
        return 60.0  # fallback

    is_range = labeled["range_7d_binary"] == 1
    n_total = len(labeled)

    best_threshold = 100.0  # start high (most restrictive)

    for thresh in range(100, 9, -5):  # 100, 95, 90, ... 10
        no_veto = ~labeled["has_hard_veto"]
        gate_pass = (labeled["range_score"] >= thresh) & no_veto
        n_pass = gate_pass.sum()
        pass_rate = n_pass / n_total

        if pass_rate < min_pass_rate:
            continue

        if n_pass == 0:
            continue

        precision = is_range[gate_pass].mean()
        if precision >= min_precision:
            best_threshold = float(thresh)
            # Don't break — keep going lower to find the least restrictive
            # threshold that still meets precision target

    return best_threshold


def run_walkforward(
    train_months: int = 12,
    test_months: int = 3,
    step_months: int = 3,
    min_precision: float = 0.65,
    out_dir: Optional[str] = None,
) -> list[WalkForwardFold]:
    """
    Run strict walk-forward validation.

    Args:
        train_months: Training window size in months.
        test_months: Test window size in months.
        step_months: Step size between folds in months.
        min_precision: Minimum precision target for threshold calibration.
        out_dir: Optional directory to save CSV results.

    Returns:
        List of WalkForwardFold results.
    """
    print("Building full dataset from DB...")
    df = _build_full_dataset()

    # Only use rows with valid range labels
    has_label = df["range_7d_label"].notna()
    df_labeled = df[has_label].copy()
    dates = df_labeled.index.sort_values()

    if len(dates) == 0:
        raise ValueError("No rows with valid range labels.")

    # Ensure dates are Timestamps for DateOffset arithmetic
    if not isinstance(dates, pd.DatetimeIndex):
        dates = pd.DatetimeIndex(dates)
        df_labeled.index = dates

    min_date = dates.min()
    max_date = dates.max()

    print(f"  Date range: {min_date} to {max_date} ({len(dates)} rows)")

    # Generate fold boundaries
    folds = []
    fold_id = 0
    train_start = min_date

    while True:
        train_end = train_start + pd.DateOffset(months=train_months)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)

        if test_start >= max_date:
            break

        # Clip test_end to available data
        test_end = min(test_end, max_date + pd.Timedelta(days=1))

        train_mask = (dates >= train_start) & (dates < train_end)
        test_mask = (dates >= test_start) & (dates < test_end)

        train_df = df_labeled[train_mask]
        test_df = df_labeled[test_mask]

        if len(train_df) < 60 or len(test_df) < 10:
            train_start += pd.DateOffset(months=step_months)
            continue

        # Calibrate threshold on train
        threshold = _calibrate_threshold(train_df, min_precision=min_precision)

        # Evaluate on test
        no_veto = ~test_df["has_hard_veto"]
        gate_pass = (test_df["range_score"] >= threshold) & no_veto
        is_range = test_df["range_7d_binary"] == 1

        n_test = len(test_df)
        n_pass = int(gate_pass.sum())
        n_true_range = int((gate_pass & is_range).sum())
        n_fp = n_pass - n_true_range
        total_range_in_test = int(is_range.sum())

        precision = n_true_range / n_pass if n_pass > 0 else 0.0
        recall = n_true_range / total_range_in_test if total_range_in_test > 0 else 0.0

        # False positive breakdown
        fp_mask = gate_pass & ~is_range
        fp_df = test_df[fp_mask]
        fp_up = int((fp_df["range_7d_label"] == "BREAKOUT_UP").sum()) if len(fp_df) else 0
        fp_down = int((fp_df["range_7d_label"] == "BREAKOUT_DOWN").sum()) if len(fp_df) else 0
        fp_both = int((fp_df["range_7d_label"] == "BREAKOUT_BOTH").sum()) if len(fp_df) else 0

        # Tail-loss: max adverse move on FP days
        if len(fp_df) > 0:
            fp_max_move = fp_df[["range_7d_max_up_pct", "range_7d_max_down_pct"]].max(axis=1)
            fp_avg_max = float(fp_max_move.mean())
            fp_worst = float(fp_max_move.max())
        else:
            fp_avg_max = 0.0
            fp_worst = 0.0

        fold = WalkForwardFold(
            fold_id=fold_id,
            train_start=str(train_start.date()) if hasattr(train_start, 'date') else str(train_start),
            train_end=str(train_end.date()) if hasattr(train_end, 'date') else str(train_end),
            test_start=str(test_start.date()) if hasattr(test_start, 'date') else str(test_start),
            test_end=str(test_end.date()) if hasattr(test_end, 'date') else str(test_end),
            threshold=threshold,
            test_n=n_test,
            gate_pass_n=n_pass,
            gate_pass_rate=round(n_pass / n_test, 3) if n_test else 0,
            true_range_n=n_true_range,
            false_positive_n=n_fp,
            precision=round(precision, 3),
            recall=round(recall, 3),
            fp_breakout_up=fp_up,
            fp_breakout_down=fp_down,
            fp_breakout_both=fp_both,
            fp_avg_max_move=round(fp_avg_max, 4),
            fp_worst_move=round(fp_worst, 4),
        )
        folds.append(fold)

        print(f"  Fold {fold_id}: train {fold.train_start}→{fold.train_end}, "
              f"test {fold.test_start}→{fold.test_end} | "
              f"thresh={threshold:.0f}, pass={n_pass}/{n_test} ({fold.gate_pass_rate:.1%}), "
              f"prec={precision:.1%}, recall={recall:.1%}, FP={n_fp}")

        fold_id += 1
        train_start += pd.DateOffset(months=step_months)

    if not folds:
        print("No valid folds generated. Need more data.")
        return []

    # Aggregate summary
    fold_df = pd.DataFrame([f.__dict__ for f in folds])

    print("\n" + "=" * 70)
    print("WALK-FORWARD SUMMARY")
    print("=" * 70)
    print(f"  Folds: {len(folds)}")
    print(f"  Avg threshold: {fold_df['threshold'].mean():.0f}")
    print(f"  Avg precision: {fold_df['precision'].mean():.1%}")
    print(f"  Avg recall:    {fold_df['recall'].mean():.1%}")
    print(f"  Avg pass rate: {fold_df['gate_pass_rate'].mean():.1%}")
    print(f"  Avg FP/fold:   {fold_df['false_positive_n'].mean():.1f}")
    print(f"  Avg FP move:   {fold_df['fp_avg_max_move'].mean():.2%}")
    print(f"  Worst FP move: {fold_df['fp_worst_move'].max():.2%}")

    # Threshold stability
    print(f"\n  Threshold range: {fold_df['threshold'].min():.0f} – {fold_df['threshold'].max():.0f}")
    print(f"  Threshold std:   {fold_df['threshold'].std():.1f}")

    if out_dir:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        fold_df.to_csv(out_path / "walkforward_folds.csv", index=False)
        print(f"\n  Results saved to {out_path}/walkforward_folds.csv")

    return folds
