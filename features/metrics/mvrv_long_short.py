import pandas as pd

def calculate(df: pd.DataFrame) -> pd.DataFrame:
    """
    MVRV long–short (Robust Bucket System)
    """
    feats = {}
    
    # Needs mvrv_long_short_diff_usd in df (aliased or raw)
    # The original code did: df['mvrv_long_short'] = df['mvrv_long_short_diff_usd']
    # We assume 'mvrv_long_short' is available or we use 'mvrv_long_short_diff_usd' directly if passed.
    # To be safe, let's look for likely candidates or rely on caller to prep.
    # For now, we assume the caller passes a DF with 'mvrv_long_short_diff_usd'.
    
    ls = df['mvrv_long_short_diff_usd']
    feats['mvrv_ls_value'] = ls
    feats['mvrv_long_short'] = ls

    # ========== A) LEVEL BUCKETS (Rolling Percentile - 365d) ==========
    ls_roll_pct = ls.rolling(365, min_periods=90).apply(
        lambda x: x.rank(pct=True).iloc[-1],
        raw=False
    )
    feats['mvrv_ls_roll_pct_365d'] = ls_roll_pct
    
    # Level bucket assignment
    ls_level = pd.Series(0, index=df.index)  # Default neutral
    ls_level.loc[ls_roll_pct < 0.10] = -2
    ls_level.loc[(ls_roll_pct >= 0.10) & (ls_roll_pct < 0.30)] = -1
    ls_level.loc[(ls_roll_pct >= 0.30) & (ls_roll_pct < 0.70)] = 0
    ls_level.loc[(ls_roll_pct >= 0.70) & (ls_roll_pct < 0.90)] = 1
    ls_level.loc[ls_roll_pct >= 0.90] = 2
    feats['mvrv_ls_level'] = ls_level.astype(int)
    
    # Expose individual level flags
    feats['mvrv_ls_level_extreme_neg'] = (ls_level == -2).astype(int)
    feats['mvrv_ls_level_neg'] = (ls_level == -1).astype(int)
    feats['mvrv_ls_level_neutral'] = (ls_level == 0).astype(int)
    feats['mvrv_ls_level_pos'] = (ls_level == 1).astype(int)
    feats['mvrv_ls_level_extreme_pos'] = (ls_level == 2).astype(int)

    # ========== B) DIRECTION / TREND BUCKETS (Primary) ==========
    trend_horizons = [2, 4, 7, 14]
    horizon_thresholds = {2: 0.8, 4: 1.0, 7: 1.0, 14: 1.2}
    trend_buckets = {}
    
    for h in trend_horizons:
        delta_h = ls - ls.shift(h)
        feats[f'mvrv_ls_delta_{h}d'] = delta_h
        
        # Rolling mean and std of the delta (90d window)
        delta_mean = delta_h.rolling(90, min_periods=30).mean().fillna(0)
        delta_std = delta_h.rolling(90, min_periods=30).std().fillna(1e-9)
        
        # Proper Z-score
        delta_z = (delta_h - delta_mean) / (delta_std + 1e-9)
        feats[f'mvrv_ls_delta_z_{h}d'] = delta_z
        
        # Horizon-specific threshold
        thresh = horizon_thresholds[h]
        trend_bucket = pd.Series(0, index=df.index)  # Default flat
        trend_bucket.loc[delta_z > thresh] = 1    # Rising
        trend_bucket.loc[delta_z < -thresh] = -1  # Falling
        
        feats[f'mvrv_ls_trend_{h}d'] = trend_bucket.astype(int)
        trend_buckets[h] = trend_bucket

    # ========== C) MULTI-HORIZON CONFIRMATION LOGIC ==========
    rising_count = (
        (trend_buckets[2] == 1).astype(int) +
        (trend_buckets[4] == 1).astype(int) +
        (trend_buckets[7] == 1).astype(int) +
        (trend_buckets[14] == 1).astype(int)
    )
    falling_count = (
        (trend_buckets[2] == -1).astype(int) +
        (trend_buckets[4] == -1).astype(int) +
        (trend_buckets[7] == -1).astype(int) +
        (trend_buckets[14] == -1).astype(int)
    )
    
    feats['mvrv_ls_rising_count'] = rising_count
    feats['mvrv_ls_falling_count'] = falling_count
    
    # Strong Uptrend
    strong_uptrend = (rising_count >= 3) & (falling_count == 0)
    feats['mvrv_ls_strong_uptrend'] = strong_uptrend.astype(int)
    
    # Weak Uptrend
    weak_uptrend = (rising_count >= 2) & (falling_count == 0) & ~strong_uptrend
    feats['mvrv_ls_weak_uptrend'] = weak_uptrend.astype(int)
    
    # Strong Downtrend
    strong_downtrend = (falling_count >= 3) & (rising_count == 0)
    feats['mvrv_ls_strong_downtrend'] = strong_downtrend.astype(int)
    
    # Weak Downtrend
    weak_downtrend = (falling_count >= 2) & (rising_count == 0) & ~strong_downtrend
    feats['mvrv_ls_weak_downtrend'] = weak_downtrend.astype(int)
    
    # Early Rollover Warning (permissive)
    early_rollover_any = (trend_buckets[2] == -1) | (trend_buckets[4] == -1)
    feats['mvrv_ls_early_rollover_any'] = early_rollover_any.astype(int)
    
    # Early Rollover Regime (strict)
    early_rollover = (
        (trend_buckets[2] == -1) & 
        (trend_buckets[4] == -1) & 
        ~strong_uptrend &
        ~weak_uptrend &
        ~strong_downtrend
    )
    feats['mvrv_ls_early_rollover'] = early_rollover.astype(int)
    
    # Mixed / Transition/ Conflict
    feats['mvrv_ls_mixed'] = (~strong_uptrend & ~strong_downtrend & ~weak_uptrend & ~weak_downtrend).astype(int)
    conflict = (rising_count > 0) & (falling_count > 0)
    feats['mvrv_ls_conflict'] = conflict.astype(int)
    
    # Level Ready
    feats['mvrv_ls_level_ready'] = ls_roll_pct.notna().astype(int)

    # ========== D) FINAL REGIMES (Outputs) ==========
    
    # Regime: Call Confirm (Strong Uptrend)
    feats['mvrv_ls_regime_call_confirm'] = strong_uptrend.astype(int)
    
    feats['mvrv_ls_regime_call_confirm_recovery'] = (
        strong_uptrend & (ls_level < 0)
    ).astype(int)
    feats['mvrv_ls_regime_call_confirm_trend'] = (
        strong_uptrend & (ls_level >= 0)
    ).astype(int)
    
    # Regime: Put Confirm (Strong Downtrend)
    feats['mvrv_ls_regime_put_confirm'] = strong_downtrend.astype(int)
    
    # Regime: Put Early Warning
    feats['mvrv_ls_regime_put_early'] = weak_downtrend.astype(int)
    
    # Regime: Distribution Warning
    feats['mvrv_ls_regime_distribution_warning_early'] = (
        (ls_level >= 1) & early_rollover_any & ~strong_downtrend
    ).astype(int)
    feats['mvrv_ls_regime_distribution_warning'] = (
        (ls_level >= 1) & early_rollover
    ).astype(int)
    
    # Regime: Bear Continuation
    feats['mvrv_ls_regime_bear_continuation'] = (
        (ls_level <= -1) & strong_downtrend
    ).astype(int)
    
    # Regime: None
    feats['mvrv_ls_regime_none'] = (feats['mvrv_ls_mixed'] & ~conflict).astype(int)

    return pd.DataFrame(feats, index=df.index)
