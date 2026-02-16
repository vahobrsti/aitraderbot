import pandas as pd
import numpy as np


def calculate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Exchange Flow Balance Feature Engineering

    Raw input: exchange_flow_balance (inflows - outflows, daily)
    Positive = BTC moving into exchanges (distribution pressure)
    Negative = BTC leaving exchanges (supply tightening)

    Produces:
      A) Raw structural: flow_raw, flow_sum_{2,4,7,14,21}
      B) Momentum/slope: flow_slope_{2,4,7,14}
      C) Extremeness: flow_z_{90,180}, flow_pct_rank_180
      D) Composite: distribution_pressure_score, high_inflow_flag
    """
    feats = {}

    # Source column (aliased in feature_builder.py)
    raw = df.get("exchange_flow_raw")
    if raw is None:
        raw = df.get("exchange_flow_balance")
    if raw is None:
        # Return empty DataFrame with correct index if no data
        return pd.DataFrame(index=df.index)

    # ========== A) RAW STRUCTURAL FEATURES ==========
    feats["flow_raw"] = raw

    sum_windows = [2, 4, 7, 14, 21]
    for w in sum_windows:
        feats[f"flow_sum_{w}"] = raw.rolling(w, min_periods=1).sum()

    # ========== B) MOMENTUM / ACCELERATION ==========
    # slope = S_t - S_{t-w} (change in rolling sum over a full window)
    # Measures true momentum of supply build-up / withdrawal
    slope_windows = [2, 4, 7, 14]
    for w in slope_windows:
        rolling_sum = feats[f"flow_sum_{w}"]
        feats[f"flow_slope_{w}"] = rolling_sum - rolling_sum.shift(w)

    # ========== C) EXTREMENESS ==========
    # Z-scores of 7-day rolling sum against longer baselines
    flow_sum_7 = feats["flow_sum_7"]

    # 90-day z-score
    mean_90 = flow_sum_7.rolling(90, min_periods=30).mean()
    std_90 = flow_sum_7.rolling(90, min_periods=30).std()
    feats["flow_z_90"] = (flow_sum_7 - mean_90) / (std_90 + 1e-9)

    # 180-day z-score
    mean_180 = flow_sum_7.rolling(180, min_periods=60).mean()
    std_180 = flow_sum_7.rolling(180, min_periods=60).std()
    feats["flow_z_180"] = (flow_sum_7 - mean_180) / (std_180 + 1e-9)

    # 180-day percentile rank (robust to fat tails)
    feats["flow_pct_rank_180"] = flow_sum_7.rolling(180, min_periods=60).apply(
        lambda x: (x <= x.iloc[-1]).mean(),
        raw=False,
    )

    # ========== D) COMPOSITE ==========
    # distribution_pressure_score: blended from normalised flow_sum_7 + slope_7 + z_90
    # Each component normalised to roughly 0-1 using 90-day percentile rank
    norm_sum_7 = flow_sum_7.rolling(90, min_periods=30).apply(
        lambda x: (x <= x.iloc[-1]).mean(),
        raw=False,
    )
    norm_slope_7 = feats["flow_slope_7"].rolling(90, min_periods=30).apply(
        lambda x: (x <= x.iloc[-1]).mean(),
        raw=False,
    )
    z_90 = feats["flow_z_90"]
    # Clip z-score to [0,1] range: map [-2, +2] -> [0, 1]
    norm_z_90 = ((z_90.clip(-2, 2) + 2) / 4)

    feats["distribution_pressure_score"] = (
        0.60 * norm_sum_7.fillna(0.5)
        + 0.25 * norm_slope_7.fillna(0.5)
        + 0.15 * norm_z_90.fillna(0.5)
    )

    return pd.DataFrame(feats, index=df.index)
