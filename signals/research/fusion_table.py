"""
Research table builder.

Produces one row per date with:
  - canonical bucket columns  (mdia_bucket, whale_bucket, mvrv_ls_bucket)
  - pairwise / triple combination labels
  - fusion state, score, confidence
  - forward returns  (ret_1d … ret_21d)
  - MFE / MAE  (mfe_14d, mae_14d)
  - labels  (label_good_move_long, label_good_move_short)
  - year tag
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from signals.fusion import fuse_signals
from signals.research.bucket_mapping import add_bucket_columns
from signals.research.constants import RETURN_HORIZONS, CANONICAL_HORIZON


# ────────────────────────────────────────────────────────────────────
# Price loader (same pattern as analyze_path_stats._load_prices)
# ────────────────────────────────────────────────────────────────────
def _load_prices_from_db() -> pd.DataFrame:
    """Load btc_close / btc_high / btc_low from RawDailyData."""
    from datafeed.models import RawDailyData

    qs = RawDailyData.objects.order_by("date").values(
        "date", "btc_close", "btc_high", "btc_low"
    )
    px = pd.DataFrame.from_records(qs)
    if px.empty:
        return px
    px["date"] = pd.to_datetime(px["date"])
    px = px.set_index("date").sort_index()
    px.index = px.index.normalize()
    return px


# ────────────────────────────────────────────────────────────────────
# Forward return helpers
# ────────────────────────────────────────────────────────────────────
def _add_forward_returns(
    df: pd.DataFrame,
    close: pd.Series,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Add ret_Nd columns (simple forward return)."""
    if horizons is None:
        horizons = RETURN_HORIZONS
    for h in horizons:
        col = f"ret_{h}d"
        df[col] = close.shift(-h) / close - 1.0
    return df


def _add_mfe_mae(
    df: pd.DataFrame,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    horizon: int = CANONICAL_HORIZON,
) -> pd.DataFrame:
    """Add mfe_{h}d and mae_{h}d (long-biased, unsigned)."""
    # MFE: max upside within window
    fwd_max = high.rolling(horizon).max().shift(-horizon)
    df[f"mfe_{horizon}d"] = fwd_max / close - 1.0

    # MAE: max downside within window (positive = drawdown magnitude)
    fwd_min = low.rolling(horizon).min().shift(-horizon)
    df[f"mae_{horizon}d"] = 1.0 - fwd_min / close
    return df


# ────────────────────────────────────────────────────────────────────
# Combination columns
# ────────────────────────────────────────────────────────────────────
def _add_combo_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add pairwise and triple combination string columns."""
    df["combo_2_mdia_whale"] = df["mdia_bucket"] + "|" + df["whale_bucket"]
    df["combo_2_mdia_mvrv"] = df["mdia_bucket"] + "|" + df["mvrv_ls_bucket"]
    df["combo_2_whale_mvrv"] = df["whale_bucket"] + "|" + df["mvrv_ls_bucket"]
    df["combo_3_all"] = (
        df["mdia_bucket"] + "|" + df["whale_bucket"] + "|" + df["mvrv_ls_bucket"]
    )
    return df


# ────────────────────────────────────────────────────────────────────
# Fusion columns
# ────────────────────────────────────────────────────────────────────
def _add_fusion_columns(df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    """Add fusion_state, fusion_score, fusion_confidence from the fusion engine."""
    states, scores, confidences = [], [], []
    for idx in df.index:
        if idx in features_df.index:
            row = features_df.loc[idx]
        else:
            # Fallback: try positional
            row = features_df.iloc[0]
        result = fuse_signals(row)
        states.append(result.state.value)
        scores.append(result.score)
        confidences.append(result.confidence.value)

    df["fusion_state"] = states
    df["fusion_score"] = scores
    df["fusion_confidence"] = confidences
    return df


# ────────────────────────────────────────────────────────────────────
# Main builder
# ────────────────────────────────────────────────────────────────────
def build_research_table(
    features_df: pd.DataFrame,
    price_df: Optional[pd.DataFrame] = None,
    horizon: int = CANONICAL_HORIZON,
) -> pd.DataFrame:
    """Build the research table.

    Parameters
    ----------
    features_df : DataFrame
        Feature CSV loaded with ``pd.read_csv(..., index_col=0,
        parse_dates=True)``.  Must contain all regime columns.
    price_df : DataFrame, optional
        Must have columns ``btc_close``, ``btc_high``, ``btc_low`` indexed
        by normalised date.  If *None*, prices are loaded from
        ``RawDailyData``.
    horizon : int
        Forward-return horizon in days (default 14).

    Returns
    -------
    DataFrame with one row per date.
    """
    # ── 1. Load prices ──────────────────────────────────────────────
    if price_df is None:
        price_df = _load_prices_from_db()

    if price_df.empty:
        raise ValueError(
            "No price data available.  Pass price_df or ensure "
            "RawDailyData is populated."
        )

    # Normalise feature index
    feat_idx = pd.to_datetime(features_df.index).normalize()
    features_df = features_df.copy()
    features_df.index = feat_idx

    # Align dates (inner join — only dates present in both)
    common = feat_idx.intersection(price_df.index)
    if common.empty:
        raise ValueError("No overlapping dates between features and prices.")

    # ── 2. Start research dataframe ─────────────────────────────────
    rt = pd.DataFrame(index=common)
    rt.index.name = "date"
    rt["close"] = price_df.loc[common, "btc_close"]

    # ── 3. Bucket columns ───────────────────────────────────────────
    aligned_feats = features_df.loc[common]
    rt_with_buckets = add_bucket_columns(aligned_feats)
    rt["mdia_bucket"] = rt_with_buckets["mdia_bucket"].values
    rt["whale_bucket"] = rt_with_buckets["whale_bucket"].values
    rt["mvrv_ls_bucket"] = rt_with_buckets["mvrv_ls_bucket"].values

    # ── 4. Combination columns ──────────────────────────────────────
    _add_combo_columns(rt)

    # ── 5. Fusion columns ───────────────────────────────────────────
    _add_fusion_columns(rt, aligned_feats)

    # ── 6. Labels (carry from features CSV) ─────────────────────────
    for lbl in ("label_good_move_long", "label_good_move_short"):
        if lbl in aligned_feats.columns:
            rt[lbl] = aligned_feats[lbl].values.astype(int)
        else:
            rt[lbl] = np.nan

    # ── 7. Forward returns ──────────────────────────────────────────
    close_aligned = price_df.loc[common, "btc_close"]
    _add_forward_returns(rt, close_aligned, RETURN_HORIZONS)

    # ── 8. MFE / MAE ───────────────────────────────────────────────
    high_aligned = price_df.loc[common, "btc_high"]
    low_aligned = price_df.loc[common, "btc_low"]
    _add_mfe_mae(rt, high_aligned, low_aligned, close_aligned, horizon)

    # ── 9. Year tag ─────────────────────────────────────────────────
    rt["year"] = rt.index.year

    return rt
