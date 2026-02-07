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
    
    # Needs mvrv_long_short_diff_usd in df
    raw_ls = df['mvrv_long_short_diff_usd']
    feats['mvrv_ls_value'] = raw_ls
    feats['mvrv_long_short'] = raw_ls

    # ========== STEP 1: SMOOTHING & NORMALIZATION ==========
    # User feedback: "Noise is your enemy... Use an EMA"
    # We use span=14 as suggested
    ls_ema = raw_ls.ewm(span=14, adjust=False).mean()
    feats['mvrv_ls_ema'] = ls_ema

    # User feedback: "z-score of the level... cleaner terrain signal"
    # Using 365d window as anchor for the z-score
    ls_roll_mean = ls_ema.rolling(365, min_periods=90).mean()
    ls_roll_std = ls_ema.rolling(365, min_periods=90).std()
    
    # Avoid division by zero
    ls_z = (ls_ema - ls_roll_mean) / (ls_roll_std + 1e-9)
    feats['mvrv_ls_z_score_365d'] = ls_z

    # Percentile (Retained for legacy/debugging, but logic moves to Z-score)
    ls_roll_pct = raw_ls.rolling(365, min_periods=90).apply(
        lambda x: x.rank(pct=True).iloc[-1],
        raw=False
    )
    feats['mvrv_ls_roll_pct_365d'] = ls_roll_pct

    # ========== STEP 2: LEVEL BUCKETS (Z-Score Based) ==========
    # Intuition:
    # Captitulation: <= -1.5
    # Recov/Early Bull: -1.5 < z <= -0.5 (or < 0)
    # Neutral/Trans: -0.5 < z < 0.5
    # Bull/Expansion: 0.5 <= z < 1.5
    # Overheated/Dist: >= 1.5
    
    ls_level = pd.Series(0, index=df.index)
    ls_level.loc[ls_z <= -1.5] = -2                # Capitulation / Deep Value
    ls_level.loc[(ls_z > -1.5) & (ls_z <= -0.5)] = -1 # Late Bear / Early Recovery
    ls_level.loc[(ls_z > -0.5) & (ls_z < 0.5)] = 0  # Neutral
    ls_level.loc[(ls_z >= 0.5) & (ls_z < 1.5)] = 1  # Bull / Expansion
    ls_level.loc[ls_z >= 1.5] = 2                  # Overheated / Distribution Risk
    
    feats['mvrv_ls_level'] = ls_level.astype(int)

    # Expose individual level flags
    feats['mvrv_ls_level_extreme_neg'] = (ls_level == -2).astype(int)
    feats['mvrv_ls_level_neg'] = (ls_level == -1).astype(int)
    feats['mvrv_ls_level_neutral'] = (ls_level == 0).astype(int)
    feats['mvrv_ls_level_pos'] = (ls_level == 1).astype(int)
    feats['mvrv_ls_level_extreme_pos'] = (ls_level == 2).astype(int)

    # ========== STEP 3: DIRECTION / TREND BUCKETS (Smoothed) ==========
    trend_horizons = [2, 4, 7, 14]
    horizon_thresholds = {2: 0.8, 4: 1.0, 7: 1.0, 14: 1.2}
    trend_buckets = {}

    for h in trend_horizons:
        # Calculate delta on smoothed EMA
        delta_h = ls_ema - ls_ema.shift(h)
        feats[f'mvrv_ls_delta_{h}d'] = delta_h
        
        # Z-score of the delta
        d_mean = delta_h.rolling(90, min_periods=30).mean().fillna(0)
        d_std = delta_h.rolling(90, min_periods=30).std().fillna(1e-9)
        delta_z = (delta_h - d_mean) / (d_std + 1e-9)
        
        # Smooth the Z-score itself (EMA 3)
        delta_z_smooth = delta_z.ewm(span=3, adjust=False).mean()
        feats[f'mvrv_ls_delta_z_{h}d'] = delta_z_smooth
        
        # Thresholds
        thresh = horizon_thresholds[h]
        trend_bucket = pd.Series(0, index=df.index)
        trend_bucket.loc[delta_z_smooth > thresh] = 1
        trend_bucket.loc[delta_z_smooth < -thresh] = -1
        
        # Hysteresis: "must persist 2 of last 3 days"
        # Rolling sum of booleans
        is_rising = (trend_bucket == 1).rolling(3).sum() >= 2
        is_falling = (trend_bucket == -1).rolling(3).sum() >= 2
        
        final_trend = pd.Series(0, index=df.index)
        final_trend.loc[is_rising] = 1
        final_trend.loc[is_falling] = -1
        
        feats[f'mvrv_ls_trend_{h}d'] = final_trend.astype(int)
        trend_buckets[h] = final_trend

    # ========== STEP 4: MULTI-HORIZON AGGREGATION ==========
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

    # Strong Uptrend (3+ rising)
    strong_uptrend = (rising_count >= 3) & (falling_count == 0)
    feats['mvrv_ls_strong_uptrend'] = strong_uptrend.astype(int)

    # Weak Uptrend (2+ rising)
    weak_uptrend = (rising_count >= 2) & (falling_count == 0) & ~strong_uptrend
    feats['mvrv_ls_weak_uptrend'] = weak_uptrend.astype(int)

    # Strong Downtrend (3+ falling)
    strong_downtrend = (falling_count >= 3) & (rising_count == 0)
    feats['mvrv_ls_strong_downtrend'] = strong_downtrend.astype(int)

    # Weak Downtrend (2+ falling)
    weak_downtrend = (falling_count >= 2) & (rising_count == 0) & ~strong_downtrend
    feats['mvrv_ls_weak_downtrend'] = weak_downtrend.astype(int)

    # Rollovers
    early_rollover_any = (trend_buckets[2] == -1) | (trend_buckets[4] == -1)
    feats['mvrv_ls_early_rollover_any'] = early_rollover_any.astype(int)
    
    early_rollover = (
        (trend_buckets[2] == -1) & 
        (trend_buckets[4] == -1) & 
        ~strong_uptrend & 
        ~weak_uptrend & 
        ~strong_downtrend
    )
    feats['mvrv_ls_early_rollover'] = early_rollover.astype(int)
    
    # Conflict
    conflict = (rising_count > 0) & (falling_count > 0)
    feats['mvrv_ls_conflict'] = conflict.astype(int)
    
    # Level Ready
    feats['mvrv_ls_level_ready'] = ls_z.notna().astype(int)

    # ========== STEP 5: FINAL REGIMES (User Refinements) ==========

    # 1. Early Recovery (User: "ls_level == -2 AND rising_count >= 2")
    # This maps to 'mvrv_ls_regime_call_confirm_recovery' for Fusion.
    recovery_condition = (ls_level <= -1) & (rising_count >= 2) & (falling_count == 0)
    feats['mvrv_ls_regime_call_confirm_recovery'] = recovery_condition.astype(int)

    # 2. Call Confirm (Standard fallback)
    feats['mvrv_ls_regime_call_confirm'] = strong_uptrend.astype(int)

    # 3. Call Confirm Trend (Positive Terrain + Strong Uptrend)
    feats['mvrv_ls_regime_call_confirm_trend'] = (
        strong_uptrend & (ls_level >= 0)
    ).astype(int)

    # 4. Distribution Warning (User: "High terrain + turning down")
    distrib_warning = (ls_level >= 2) & (falling_count >= 2)
    feats['mvrv_ls_regime_distribution_warning'] = distrib_warning.astype(int)
    
    # Keep legacy/early warning field too, but align logic? 
    # Let's keep it simple or just mirror the main warning for now to avoid confusion
    feats['mvrv_ls_regime_distribution_warning_early'] = distrib_warning.astype(int)

    # 5. Put Confirm (Strong Downtrend)
    feats['mvrv_ls_regime_put_confirm'] = strong_downtrend.astype(int)
    
    # 6. Put Early Warning
    feats['mvrv_ls_regime_put_early'] = weak_downtrend.astype(int)

    # 7. Bear Continuation
    feats['mvrv_ls_regime_bear_continuation'] = (
        (ls_level <= -1) & strong_downtrend
    ).astype(int)
    
    # 8. None/Mixed
    feats['mvrv_ls_mixed'] = (~strong_uptrend & ~strong_downtrend).astype(int)
    feats['mvrv_ls_regime_none'] = (feats['mvrv_ls_mixed'] & ~conflict).astype(int)

    return pd.DataFrame(feats, index=df.index)
