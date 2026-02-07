import pandas as pd

def calculate(df: pd.DataFrame) -> pd.DataFrame:
    """
    MVRV composite extremes and 60d enhancements
    """
    feats = {}
    
    # Composite MVRV calculation
    mvrv_cols = [
        'mvrv_usd_1d',
        'mvrv_usd_7d',
        'mvrv_usd_30d',
        'mvrv_usd_60d',
        'mvrv_usd_90d',
        'mvrv_usd_180d',
        'mvrv_usd_365d',
    ]
    # Ensure cols exist
    existing_cols = [c for c in mvrv_cols if c in df.columns]
    mvrv_composite = df[existing_cols].mean(axis=1)
    # feats['mvrv_composite'] = mvrv_composite
    
    # mvrv_composite_pct
    mvrv = (mvrv_composite - 1.0) * 100.0
    feats["mvrv_composite_pct"] = mvrv

    # (a) Rolling z-scores
    for win in [90, 180, 365]:
        min_p = max(30, win // 3)
        roll_mean = mvrv.rolling(win, min_periods=min_p).mean()
        roll_std = mvrv.rolling(win, min_periods=min_p).std()
        z = (mvrv - roll_mean) / (roll_std + 1e-9)
        
        feats[f"mvrv_comp_z_{win}d"] = z
        feats[f"mvrv_comp_undervalued_{win}d"] = (z < -1.0).astype(int)
        feats[f"mvrv_comp_overheated_{win}d"] = (z > 1.0).astype(int)

    # (b) Min / max and distance
    for win in [90, 180, 365]:
        min_p = max(30, win // 3)
        roll_min = mvrv.rolling(win, min_periods=min_p).min()
        roll_max = mvrv.rolling(win, min_periods=min_p).max()
        roll_range = roll_max - roll_min

        feats[f"mvrv_comp_min_{win}d"] = roll_min
        feats[f"mvrv_comp_max_{win}d"] = roll_max

        # Relative distance
        safe_range = roll_range.clip(lower=1e-9)
        
        dist_from_min = mvrv - roll_min
        dist_from_max = roll_max - mvrv
        
        rel_dist_min = dist_from_min / safe_range
        rel_dist_max = dist_from_max / safe_range

        feats[f"mvrv_comp_dist_from_min_{win}d"] = dist_from_min
        feats[f"mvrv_comp_dist_from_max_{win}d"] = dist_from_max
        feats[f"mvrv_comp_rel_dist_min_{win}d"] = rel_dist_min
        feats[f"mvrv_comp_rel_dist_max_{win}d"] = rel_dist_max

        # New lows / highs
        prev_min = mvrv.shift(1).rolling(win, min_periods=min_p).min()
        prev_max = mvrv.shift(1).rolling(win, min_periods=min_p).max()

        feats[f"mvrv_comp_new_low_{win}d"] = (mvrv < prev_min).astype(int)
        feats[f"mvrv_comp_new_high_{win}d"] = (mvrv > prev_max).astype(int)

        # "Near bottom" and "near top" flags
        MVRV_REL_THRESH = 0.05
        feats[f"mvrv_comp_near_bottom_{win}d"] = (rel_dist_min < MVRV_REL_THRESH).astype(int)
        feats[f"mvrv_comp_near_top_{win}d"] = (rel_dist_max < MVRV_REL_THRESH).astype(int)

    # (b.1) Consolidated "Near Bottom/Top Any"
    feats["mvrv_comp_near_bottom_any"] = (
        (feats["mvrv_comp_near_bottom_90d"] == 1) |
        (feats["mvrv_comp_near_bottom_180d"] == 1) |
        (feats["mvrv_comp_near_bottom_365d"] == 1)
    ).astype(int)

    feats["mvrv_comp_near_top_any"] = (
        (feats["mvrv_comp_near_top_90d"] == 1) |
        (feats["mvrv_comp_near_top_180d"] == 1) |
        (feats["mvrv_comp_near_top_365d"] == 1)
    ).astype(int)

    # (d) Aggregate flags
    feats["mvrv_undervalued_extreme"] = (
        (feats["mvrv_comp_near_bottom_any"] == 1) |
        ((feats["mvrv_comp_new_low_180d"] == 1) | (feats["mvrv_comp_new_low_365d"] == 1))
    ).astype(int)

    feats["mvrv_overheated_extreme"] = (
        (feats["mvrv_comp_near_top_any"] == 1) |
        ((feats["mvrv_comp_new_high_180d"] == 1) | (feats["mvrv_comp_new_high_365d"] == 1))
    ).astype(int)
    
    # (d.1) MVRV-60d short signal enhancement features
    mvrv_60d = df["mvrv_usd_60d"]
    feats["mvrv_60d"] = mvrv_60d
    
    # Feature 1: mvrv_usd_60d percentile over last 60 days
    feats["mvrv_60d_pct_rank"] = mvrv_60d.rolling(60, min_periods=20).apply(
        lambda x: (x.iloc[-1] >= x).sum() / len(x) if len(x) > 0 else 0.5,
        raw=False
    )
    
    # Feature 2: Distance from rolling 60-day max
    roll_min_60 = mvrv_60d.rolling(60, min_periods=20).min()
    roll_max_60 = mvrv_60d.rolling(60, min_periods=20).max()
    range_60 = roll_max_60 - roll_min_60
    feats["mvrv_60d_dist_from_max"] = (roll_max_60 - mvrv_60d) / (range_60 + 1e-9)
    
    # Feature 3-5: MVRV-60d trend detection
    mvrv_60d_delta_7d = mvrv_60d.diff(7)
    mvrv_60d_std = mvrv_60d.rolling(30, min_periods=10).std()
    mvrv_60d_delta_z = mvrv_60d_delta_7d / (mvrv_60d_std + 1e-9)
    
    feats["mvrv_60d_is_falling"] = (mvrv_60d_delta_z < -0.5).astype(int)
    feats["mvrv_60d_is_flattening"] = ((mvrv_60d_delta_z >= -0.5) & (mvrv_60d_delta_z <= 0.5)).astype(int)
    feats["mvrv_60d_is_rising"] = (mvrv_60d_delta_z > 0.5).astype(int)

    # (e) Composite MVRV (Valuation Backbone) - Z-Score Buckets & Relaive Regimes
    z_long = feats["mvrv_comp_z_365d"]

    # Buckets
    feats['mvrv_bucket_deep_undervalued'] = (z_long < -1.5).astype(int)
    feats['mvrv_bucket_undervalued']      = ((z_long >= -1.5) & (z_long < -0.5)).astype(int)
    feats['mvrv_bucket_fair']             = ((z_long >= -0.5) & (z_long <= 0.5)).astype(int)
    feats['mvrv_bucket_overvalued']       = ((z_long > 0.5) & (z_long <= 1.5)).astype(int)
    feats['mvrv_bucket_extreme_overvalued'] = (z_long > 1.5).astype(int)

    # Direction Overlay (Z-score Space)
    z_trend_7d = z_long.diff(7)
    feats['mvrv_z_trend_7d']  = z_trend_7d
    feats['mvrv_is_rising']     = (z_trend_7d > 0).astype(int)
    feats['mvrv_is_falling']    = (z_trend_7d < 0).astype(int)
    feats['mvrv_is_flattening'] = (z_trend_7d.abs() < 0.1).astype(int)

    # Trading Interpretation Regimes
    is_undervalued_zone = (
        (feats['mvrv_bucket_deep_undervalued'] == 1) | 
        (feats['mvrv_bucket_undervalued'] == 1)
    )
    feats['regime_mvrv_call_backbone'] = (
         is_undervalued_zone & (feats['mvrv_is_rising'] == 1)
    ).astype(int)

    feats['regime_mvrv_reduce_longs'] = (
        (feats['mvrv_bucket_overvalued'] == 1) & 
        (feats['mvrv_is_flattening'] == 1)
    ).astype(int)

    feats['regime_mvrv_put_supportive'] = (
        (feats['mvrv_bucket_extreme_overvalued'] == 1) & 
        (feats['mvrv_is_falling'] == 1)
    ).astype(int)

    feats['regime_mvrv_call_accumulate'] = (
        (feats['mvrv_comp_new_low_180d'] == 1) | 
        (feats['mvrv_comp_new_low_365d'] == 1)
    ).astype(int)

    return pd.DataFrame(feats, index=df.index)
