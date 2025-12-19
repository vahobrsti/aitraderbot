# features/feature_builder.py

import pandas as pd


def build_features_and_labels_from_raw(
    df: pd.DataFrame,
    horizon_days: int = 14,
    target_return: float = 0.05,
) -> pd.DataFrame:
    """
    df: DataFrame from RawDailyData (datafeed.models.RawDailyData.objects.values()).
    Returns: DataFrame with engineered features + a binary label.
    """

    # Ensure sorted by date and use date as index
    df = df.sort_values('date').set_index('date')

    # ---------- Derived raw series ----------

    # Composite MVRV = simple average of your different horizons
    mvrv_cols = [
        'mvrv_usd_1d',
        'mvrv_usd_7d',
        'mvrv_usd_30d',
        'mvrv_usd_60d',
        'mvrv_usd_90d',
        'mvrv_usd_180d',
        'mvrv_usd_365d',
    ]
    df['mvrv_composite'] = df[mvrv_cols].mean(axis=1)
    # convert to a PnL-style percentage: -100..+ (roughly) 100+
    # > 0 = in profit, < 0 = in loss, 0 = breakeven
    df['mvrv_composite_pct'] = (df['mvrv_composite'] - 1.0) * 100.0


    # Whale buckets: small vs big
    df['btc_holders_1_100'] = df['btc_holders_1_10'] + df['btc_holders_10_100']
    df['btc_holders_100_10k'] = df['btc_holders_100_1k'] + df['btc_holders_1k_10k']

    # Convenience aliases
    df['mvrv_long_short'] = df['mvrv_long_short_diff_usd']
    # normalize the sentiment
    df['sentiment_raw'] = df['sentiment_weighted_total']
    s = df["sentiment_raw"]
    # Use expanding window to avoid look-ahead bias
    mu = s.expanding().mean()
    sigma = s.expanding().std()
    df["sentiment_norm"] = (s - mu) / sigma

    df['mean_price'] = df['btc_price_mean']

    feats = {}

    # ---------- 1) MDIA slope & Regimes (Robust) ----------
    # Intuition:
    # - MDIA falling = Capital Inflow (Bullish)
    # - We want broad participation (falling across horizons) AND acceleration
    
    # 1. Calculate Raw Slopes
    mdia_horizons = [1, 2, 4, 7]
    for win in mdia_horizons:
        slope_col = f'mdia_slope_{win}d'
        # Raw delta
        feats[slope_col] = df['mdia'] - df['mdia'].shift(win)
        
        # 2. Normalize (Rolling Z-Score of the Slope)
        #    This makes "steepness" relative to the last 90 days
        #    Use 90d window to capture medium-term volatility of flows
        z_win = 90
        slope_series = feats[slope_col]
        roll_mean = slope_series.rolling(z_win, min_periods=30).mean()
        roll_std = slope_series.rolling(z_win, min_periods=30).std()
        
        # Avoid div by zero
        z_score = (slope_series - roll_mean) / (roll_std + 1e-9)
        feats[f'mdia_slope_z_{win}d'] = z_score

        # 3. Ordinal Buckets
        #    +1 = Rising (Distribution) -> Z > 0.5
        #     0 = Neutral / Noise       -> -0.5 <= Z <= 0.5
        #    -1 = Inflow                -> -1.5 <= Z < -0.5
        #    -2 = Strong Inflow         -> Z < -1.5
        
        # Vectorized bucket assignment
        bucket_col = f'mdia_bucket_{win}d'
        buckets = pd.Series(0, index=df.index) # Default 0
        
        buckets.loc[z_score > 0.5] = 1
        buckets.loc[z_score < -0.5] = -1
        buckets.loc[z_score < -1.5] = -2
        
        feats[bucket_col] = buckets.astype(int)

    # 4. Breadth Regime: "All horizons showing inflow"
    #    Check if 1d, 2d, 4d, 7d are all <= -1 (Inflow or Strong Inflow)
    feats['mdia_breadth_inflow'] = (
        (feats['mdia_bucket_1d'] <= -1) &
        (feats['mdia_bucket_2d'] <= -1) &
        (feats['mdia_bucket_4d'] <= -1) &
        (feats['mdia_bucket_7d'] <= -1)
    ).astype(int)

    # 4b. Moderate Breadth: "3 out of 4 horizons showing inflow"
    inflow_count = (
        (feats['mdia_bucket_1d'] <= -1).astype(int) +
        (feats['mdia_bucket_2d'] <= -1).astype(int) +
        (feats['mdia_bucket_4d'] <= -1).astype(int) +
        (feats['mdia_bucket_7d'] <= -1).astype(int)
    )
    feats['mdia_breadth_inflow_3of4'] = (inflow_count >= 3).astype(int)
    # Expose count as a feature for ML/Debugging
    feats['mdia_inflow_count'] = inflow_count

    # 5. Acceleration Regime: "Short term is dropping FASTER (more negative) than long term"
    #    Compare buckets: is 2d bucket MORE NEGATIVE (lower) than 4d bucket?
    #    e.g. 2d is -2 (Strong), 4d is -1 (Normal) -> Acceleration!
    feats['mdia_accel_2d_vs_4d'] = (
        feats['mdia_bucket_2d'] < feats['mdia_bucket_4d']
    ).astype(int)
    
    # 5b. Acceleration Regime (Strict): "2d dropping faster than 1d" (Immediate acceleration)
    feats['mdia_accel_2d_vs_1d'] = (
        feats['mdia_bucket_2d'] < feats['mdia_bucket_1d']
    ).astype(int)
    
    # Combined Acceleration: Either 2d vs 4d OR 2d vs 1d (Flexible)
    feats['mdia_is_accelerating'] = (
        (feats['mdia_accel_2d_vs_4d'] == 1) | 
        (feats['mdia_accel_2d_vs_1d'] == 1)
    ).astype(int)

    # 6. Final Regimes

    # (A) Strong Inflow: Broad participation AND Acceleration
    feats['mdia_regime_strong_inflow'] = (
        (feats['mdia_breadth_inflow'] == 1) &
        (feats['mdia_is_accelerating'] == 1)
    ).astype(int)

    # (B) Inflow (Moderate): Broad participation (3/4 horizons) but not Strong
    feats['mdia_regime_inflow'] = (
        (feats['mdia_breadth_inflow_3of4'] == 1) &
        (feats['mdia_regime_strong_inflow'] == 0) # Exclusive
    ).astype(int)

    # (C) Distribution / Risk: Short term horizons are rising (Positive Buckets) 
    # AND we are NOT in a broad inflow regime (to avoid mixed signals)
    feats['mdia_regime_distribution'] = (
        (feats['mdia_bucket_1d'] >= 1) &
        (feats['mdia_bucket_2d'] >= 1) &
        (feats['mdia_breadth_inflow'] == 0)
    ).astype(int)

    # ---------- 2) MVRV long–short (Robust Bucket System) ----------
    # Two separate things to bucketize:
    # A) Level (secondary / confidence weighting) - is current value extreme relative to history?
    # B) Direction / trend (primary / regime-defining) - is the metric improving or deteriorating?

    ls = df['mvrv_long_short']
    feats['mvrv_ls_value'] = ls

    # ========== A) LEVEL BUCKETS (Rolling Percentile - 365d) ==========
    # Level -2: extreme negative (bottom 10%)
    # Level -1: negative (10-30%)
    # Level  0: neutral (30-70%)
    # Level +1: positive (70-90%)
    # Level +2: extreme positive (top 10%)
    
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
    # For each horizon h ∈ {2,4,7,14}:
    # Trend +1: meaningfully rising
    # Trend  0: flat / noise
    # Trend -1: meaningfully falling
    # Use threshold relative to rolling std of Δh to avoid "+0.0001 counts as rising"
    
    trend_horizons = [2, 4, 7, 14]
    trend_buckets = {}
    
    for h in trend_horizons:
        delta_h = ls - ls.shift(h)
        feats[f'mvrv_ls_delta_{h}d'] = delta_h
        
        # Rolling mean and std of the delta (90d window for stability)
        delta_mean = delta_h.rolling(90, min_periods=30).mean().fillna(0)
        delta_std = delta_h.rolling(90, min_periods=30).std().fillna(1e-9)
        
        # Proper Z-score: mean-centered to reduce false signals during drift
        delta_z = (delta_h - delta_mean) / (delta_std + 1e-9)
        feats[f'mvrv_ls_delta_z_{h}d'] = delta_z
        
        # Bucket: use 0.5 std as threshold for "meaningful"
        trend_bucket = pd.Series(0, index=df.index)  # Default flat
        trend_bucket.loc[delta_z > 0.5] = 1    # Rising
        trend_bucket.loc[delta_z < -0.5] = -1  # Falling
        
        feats[f'mvrv_ls_trend_{h}d'] = trend_bucket.astype(int)
        trend_buckets[h] = trend_bucket

    # ========== C) MULTI-HORIZON CONFIRMATION LOGIC ==========
    # From the four directional buckets (2/4/7/14), derive trend breadth
    
    # Count how many horizons are +1 (rising) and -1 (falling)
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
    
    # Strong Uptrend: 3+ of 4 horizons are +1, and none are -1 (high conviction)
    strong_uptrend = (rising_count >= 3) & (falling_count == 0)
    feats['mvrv_ls_strong_uptrend'] = strong_uptrend.astype(int)
    
    # Weak Uptrend: 2+ horizons rising, none falling (early confirmation)
    weak_uptrend = (rising_count >= 2) & (falling_count == 0) & ~strong_uptrend
    feats['mvrv_ls_weak_uptrend'] = weak_uptrend.astype(int)
    
    # Strong Downtrend: 3+ of 4 horizons are -1, and none are +1 (high conviction)
    strong_downtrend = (falling_count >= 3) & (rising_count == 0)
    feats['mvrv_ls_strong_downtrend'] = strong_downtrend.astype(int)
    
    # Weak Downtrend: 2+ horizons falling, none rising (early warning)
    weak_downtrend = (falling_count >= 2) & (rising_count == 0) & ~strong_downtrend
    feats['mvrv_ls_weak_downtrend'] = weak_downtrend.astype(int)
    
    # Early Rollover: short-term (2d/4d) turning down first
    early_rollover = (trend_buckets[2] == -1) | (trend_buckets[4] == -1)
    feats['mvrv_ls_early_rollover'] = early_rollover.astype(int)
    
    # Mixed / Transition: anything else
    feats['mvrv_ls_mixed'] = (~strong_uptrend & ~strong_downtrend & ~weak_uptrend & ~weak_downtrend).astype(int)

    # ========== D) FINAL REGIMES (Outputs) ==========
    
    # Regime: Call Confirm (Strong Uptrend)
    # Stronger if level >= 0 (not deeply negative)
    feats['mvrv_ls_regime_call_confirm'] = strong_uptrend.astype(int)
    feats['mvrv_ls_regime_call_confirm_strong'] = (
        strong_uptrend & (ls_level >= 0)
    ).astype(int)
    feats['mvrv_ls_regime_early_recovery'] = (
        strong_uptrend & (ls_level < 0)
    ).astype(int)
    
    # Regime: Put Confirm (Strong Downtrend - high conviction)
    feats['mvrv_ls_regime_put_confirm'] = strong_downtrend.astype(int)
    
    # Regime: Put Early Warning (Weak Downtrend - early confirmation)
    feats['mvrv_ls_regime_put_early'] = weak_downtrend.astype(int)
    
    # Regime: Distribution Warning (level high AND early rollover)
    # Triggers earlier than put_confirm - when 2d/4d flip down first at high levels
    feats['mvrv_ls_regime_distribution_warning'] = (
        (ls_level >= 1) & (early_rollover | weak_downtrend | strong_downtrend)
    ).astype(int)
    
    # Regime: Bear Continuation (level very negative AND downtrend continues)
    feats['mvrv_ls_regime_bear_continuation'] = (
        (ls_level <= -1) & strong_downtrend
    ).astype(int)
    
    # Regime: None (mixed - should not gate trades)
    feats['mvrv_ls_regime_none'] = feats['mvrv_ls_mixed']
    # ---------- 3) MVRV composite extremes [start] ----------
    mvrv = df["mvrv_composite_pct"]

    # (a) Rolling z-scores (generic ML features)
    for win in [90, 180, 365]:
        min_p = max(30, win // 3)
        roll_mean = mvrv.rolling(win, min_periods=min_p).mean()
        roll_std = mvrv.rolling(win, min_periods=min_p).std()
        z = (mvrv - roll_mean) / (roll_std + 1e-9)
        
        if (z > 2.0).any():
            # (Optional debug print removed for performance)
            pass
        feats[f"mvrv_comp_z_{win}d"] = z
        feats[f"mvrv_comp_undervalued_{win}d"] = (z < -1.0).astype(int)
        feats[f"mvrv_comp_overheated_{win}d"] = (z > 1.0).astype(int)

    # (b) Min / max and distance from them over 3m, 6m, 1y
    for win in [90, 180, 365]:
        min_p = max(30, win // 3)
        roll_min = mvrv.rolling(win, min_periods=min_p).min()
        roll_max = mvrv.rolling(win, min_periods=min_p).max()
        roll_range = roll_max - roll_min

        feats[f"mvrv_comp_min_{win}d"] = roll_min
        feats[f"mvrv_comp_max_{win}d"] = roll_max

        # Relative distance (scale by range)
        # Avoid division by zero by clipping
        safe_range = roll_range.clip(lower=1e-9)
        
        dist_from_min = mvrv - roll_min
        dist_from_max = roll_max - mvrv
        
        rel_dist_min = dist_from_min / safe_range
        rel_dist_max = dist_from_max / safe_range

        feats[f"mvrv_comp_dist_from_min_{win}d"] = dist_from_min
        feats[f"mvrv_comp_dist_from_max_{win}d"] = dist_from_max
        
        # Save relative distances as features
        feats[f"mvrv_comp_rel_dist_min_{win}d"] = rel_dist_min
        feats[f"mvrv_comp_rel_dist_max_{win}d"] = rel_dist_max

        # new lows / highs versus previous window (exclude today)
        prev_min = mvrv.shift(1).rolling(win, min_periods=min_p).min()
        prev_max = mvrv.shift(1).rolling(win, min_periods=min_p).max()

        feats[f"mvrv_comp_new_low_{win}d"] = (mvrv < prev_min).astype(int)
        feats[f"mvrv_comp_new_high_{win}d"] = (mvrv > prev_max).astype(int)

        # "Near bottom" and "near top" flags (relative)
        # Threshold: within bottom 5% of the range
        MVRV_REL_THRESH = 0.05
        
        feats[f"mvrv_comp_near_bottom_{win}d"] = (rel_dist_min < MVRV_REL_THRESH).astype(int)
        feats[f"mvrv_comp_near_top_{win}d"] = (rel_dist_max < MVRV_REL_THRESH).astype(int)

    # (b.1) Consolidated "Near Bottom/Top Any"
    # We check if it is near bottom in ANY of the horizons 
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

    # (d) Aggregate flags that reflect your intuition:

    # Undervalued: at/near multi-month lows OR fresh multi-month low
    feats["mvrv_undervalued_extreme"] = (
        (feats["mvrv_comp_near_bottom_any"] == 1)
        |
        (
            (feats["mvrv_comp_new_low_180d"] == 1)
            | (feats["mvrv_comp_new_low_365d"] == 1)
        )
    ).astype(int)

    # Overheated: at/near multi-month highs OR fresh multi-month high
    feats["mvrv_overheated_extreme"] = (
        (feats["mvrv_comp_near_top_any"] == 1)
        |
        (
            (feats["mvrv_comp_new_high_180d"] == 1)
            | (feats["mvrv_comp_new_high_365d"] == 1)
        )
        
    ).astype(int)

    # Keep raw composite pct as well
    feats["mvrv_composite_pct"] = mvrv

    # (e) Composite MVRV (Valuation Backbone) - Z-Score Buckets & Relaive Regimes
    # Normalization: Rolling Z-score (long window: 365 days / 1 year)
    
    z_long = feats["mvrv_comp_z_365d"]

    # Buckets
    feats['mvrv_bucket_deep_undervalued'] = (z_long < -1.5).astype(int)
    feats['mvrv_bucket_undervalued']      = ((z_long >= -1.5) & (z_long < -0.5)).astype(int) # explicit check
    feats['mvrv_bucket_fair']             = ((z_long >= -0.5) & (z_long <= 0.5)).astype(int)
    feats['mvrv_bucket_overvalued']       = ((z_long > 0.5) & (z_long <= 1.5)).astype(int)
    feats['mvrv_bucket_extreme_overvalued'] = (z_long > 1.5).astype(int)

    # Direction Overlay (Z-score Space)
    # Use change in Z-score over 7 days for stable momentum
    z_trend_7d = z_long.diff(7)
    
    feats['mvrv_z_trend_7d']  = z_trend_7d
    feats['mvrv_is_rising']     = (z_trend_7d > 0).astype(int)
    feats['mvrv_is_falling']    = (z_trend_7d < 0).astype(int)
    feats['mvrv_is_flattening'] = (z_trend_7d.abs() < 0.1).astype(int) # < 0.1 std dev change over 7d

    # Trading Interpretation Regimes
    
    # 1. Undervalued + Rising -> CALL Backbone
    is_undervalued_zone = (
        (feats['mvrv_bucket_deep_undervalued'] == 1) | 
        (feats['mvrv_bucket_undervalued'] == 1)
    )
    feats['regime_mvrv_call_backbone'] = (
         is_undervalued_zone & (feats['mvrv_is_rising'] == 1)
    ).astype(int)

    # 3. Overvalued + Flattening -> Reduce Longs
    feats['regime_mvrv_reduce_longs'] = (
        (feats['mvrv_bucket_overvalued'] == 1) & 
        (feats['mvrv_is_flattening'] == 1)
    ).astype(int)

    # 4. Extreme Overvaluation + Rolling Over (Falling) -> PUT-supportive
    feats['regime_mvrv_put_supportive'] = (
        (feats['mvrv_bucket_extreme_overvalued'] == 1) & 
        (feats['mvrv_is_falling'] == 1)
    ).astype(int)

    # 5. New Low for 6m or 1y -> Very Good Call Supportive (Accumulate)
    feats['regime_mvrv_call_accumulate'] = (
        (feats['mvrv_comp_new_low_180d'] == 1) | 
        (feats['mvrv_comp_new_low_365d'] == 1)
    ).astype(int)



    # ---------- 4) Whale accumulation ----------
    # Buckets of interest: 1-100 (Retail/Small) and 100-10k (Whale/Smart Money)
    cols_map = {
        '1_100': 'btc_holders_1_100',
        '100_10k': 'btc_holders_100_10k'
    }

    # 1. Base Changes over 1d, 2d, 4d, 7d, 30d
    windows = [1, 2, 4, 7, 30]
    for bucket, col in cols_map.items():
        for win in windows:
            # Change in raw balance
            feats[f'whale_{bucket}_change_{win}d'] = df[col] - df[col].shift(win)

    # 2. "Good" Signal: 100-10k increasing (Strong smart money signal)
    # Intuition: "if the balance held by this group is increasing over 1day, 2day, 4day and 1week"
    # We create a "Consistent Accumulation" flag if ALL those windows are positive
    feats['whale_smart_accum_consistent'] = (
        (feats['whale_100_10k_change_1d'] > 0) &
        (feats['whale_100_10k_change_2d'] > 0) &
        (feats['whale_100_10k_change_4d'] > 0) &
        (feats['whale_100_10k_change_7d'] > 0)
    ).astype(int)

    # 3. "Very Good": Both 1-100 AND 100-10k are increasing
    # We check the 7d trend for this broad signal
    feats['whale_broad_accum_7d'] = (
        (feats['whale_100_10k_change_7d'] > 0) & 
        (feats['whale_1_100_change_7d'] > 0)
    ).astype(int)

    # 4. Divergence Check: "if 1-100 increasing but 100-10k decreasing... check total"
    
    # Calculate Total Holdings (Small + Medium/Large)
    df['total_holders_1_to_10k'] = df['btc_holders_1_100'] + df['btc_holders_100_10k']
    
    # Check if total is flat or up
    feats['whale_total_change_7d'] = df['total_holders_1_to_10k'] - df['total_holders_1_to_10k'].shift(7)
    
    # Condition: Retail Buying (1-100 UP) AND Whales Selling (100-10k DOWN)
    retail_buy_whale_sell = (feats['whale_1_100_change_7d'] > 0) & (feats['whale_100_10k_change_7d'] < 0)
    
    # Signal: Divergence is "Okay" if Total Supply held is still Flat/Up
    feats['whale_retail_absorption_positive'] = (
        retail_buy_whale_sell & (feats['whale_total_change_7d'] >= 0)
    ).astype(int)
   
    # ---------- 5) Sentiment ----------
    # Work with normalized sentiment
    sent = df["sentiment_norm"]

    # (a) 7-day trend (for bonus points)
    feats["sentiment_trend_7d"] = sent - sent.shift(7)
    feats["sentiment_downtrend_7d"] = (feats["sentiment_trend_7d"] < 0).astype(int)
    feats["sentiment_uptrend_7d"] = (feats["sentiment_trend_7d"] > 0).astype(int)

    # (b) Rolling stats over 1m, 3m, 6m, 1y
    # User requested: 1 month, 3 months, 6 months (and we keep 1y for long-term context)
    windows = [30, 90, 180, 365]
    BOTTOM_THRESH = 0.5
    TOP_THRESH = 0.5 

    for win in windows:
        # Min / Max
        roll_min = sent.rolling(win).min()
        roll_max = sent.rolling(win).max()
        roll_range = roll_max - roll_min
        
        feats[f"sentiment_min_{win}d"] = roll_min
        feats[f"sentiment_max_{win}d"] = roll_max
        
        
        # Relative Distance (Range Scaled)
        # Avoid division by zero by clipping
        safe_range = roll_range.clip(lower=1e-9)
        
        dist_from_min = sent - roll_min
        dist_from_max = roll_max - sent
        
        rel_dist_min = dist_from_min / safe_range
        rel_dist_max = dist_from_max / safe_range
        
        # Near Bottom / Top flags (Relative)
        # Threshold: within bottom 10% of range (Legacy diagnostics)
        LEGACY_REL_THRESH = 0.10
        
        feats[f"sentiment_near_bottom_{win}d"] = (
            rel_dist_min < LEGACY_REL_THRESH
        ).astype(int)
        
        feats[f"sentiment_near_top_{win}d"] = (
            (rel_dist_max < LEGACY_REL_THRESH)
        ).astype(int)

        # New Lows / Highs (strict: lower/higher than previous window's min/max)
        prev_min = sent.shift(1).rolling(win).min()
        prev_max = sent.shift(1).rolling(win).max()
        
        feats[f"sentiment_new_low_{win}d"] = (sent < prev_min).astype(int)
        feats[f"sentiment_new_high_{win}d"] = (sent > prev_max).astype(int)

        # Z-scores (optional but useful)
        roll_mean = sent.rolling(win).mean()
        roll_std = sent.rolling(win).std()
        feats[f"sentiment_z_{win}d"] = (sent - roll_mean) / (roll_std + 1e-9)


    # ---------- 5.5) Relative Sentiment (Crowd Psychology) - Canonical ----------

    sent = df["sentiment_norm"]

    # (1) Rolling percentile level (180d): percentile of TODAY within last 180d
    sent_roll_pct_180d = sent.rolling(180, min_periods=60).apply(
        lambda x: x.rank(pct=True).iloc[-1],
        raw=False
    )
    feats["sentiment_roll_pct_180d"] = sent_roll_pct_180d

    # (2) Buckets
    feats["sent_bucket_extreme_fear"]  = (sent_roll_pct_180d < 0.10).astype(int)
    feats["sent_bucket_fear"]          = ((sent_roll_pct_180d >= 0.10) & (sent_roll_pct_180d < 0.30)).astype(int)
    feats["sent_bucket_neutral"]       = ((sent_roll_pct_180d >= 0.30) & (sent_roll_pct_180d < 0.70)).astype(int)
    feats["sent_bucket_greed"]         = ((sent_roll_pct_180d >= 0.70) & (sent_roll_pct_180d < 0.90)).astype(int)
    feats["sent_bucket_extreme_greed"] = (sent_roll_pct_180d >= 0.90).astype(int)

    # (3) Momentum overlays (Robust)
    # Require "rising" to mean positive change > threshold
    MOMENTUM_THRESH = 0.05
    sent_change_3d = sent.diff(3)
    sent_change_5d = sent.diff(5)

    feats["sent_is_rising"]  = ((sent_change_3d > MOMENTUM_THRESH) & (sent_change_5d > MOMENTUM_THRESH)).astype(int)
    feats["sent_is_falling"] = ((sent_change_3d < -MOMENTUM_THRESH) & (sent_change_5d < -MOMENTUM_THRESH)).astype(int)

    # Flattening: small raw change relative to recent volatility
    # Threshold: < 0.25 * rolling_std_90d
    # Use min_periods and conservative fill
    roll_std_90 = sent.rolling(90, min_periods=30).std().fillna(1.0) 
    feats["sent_is_flattening"] = (sent_change_3d.abs() < (0.25 * roll_std_90)).astype(int)

    # (4) Persistence
    # Extreme Greed
    feats["sent_extreme_greed_persist_5d"] = (
        feats["sent_bucket_extreme_greed"]
        .rolling(5, min_periods=5)
        .min()
        .fillna(0)
        .astype(int)
    )
    # Extreme Fear (New)
    feats["sent_extreme_fear_persist_5d"] = (
        feats["sent_bucket_extreme_fear"]
        .rolling(5, min_periods=5)
        .min()
        .fillna(0)
        .astype(int)
    )
    # Greed Persistence (7d) - Not just extreme
    feats["sent_greed_persist_7d"] = (
        feats["sent_bucket_greed"]
        .rolling(7, min_periods=7)
        .min()
        .fillna(0)
        .astype(int)
    )
    # Fear Persistence (7d) - Not just extreme (symmetric with greed)
    feats["sent_fear_persist_7d"] = (
        feats["sent_bucket_fear"]
        .rolling(7, min_periods=7)
        .min()
        .fillna(0)
        .astype(int)
    )

    # (5) Trading regimes (Renamed to sent_regime_...)

    # Fear (incl extreme fear) + rising -> CALL-supportive
    feats["sent_regime_call_supportive"] = (
        (feats["sent_bucket_fear"] | feats["sent_bucket_extreme_fear"]) &
        feats["sent_is_rising"]
    ).astype(int)

    # Extreme fear + flattening -> mean-reversion call
    feats["sent_regime_call_mean_reversion"] = (
        feats["sent_bucket_extreme_fear"] &
        feats["sent_is_flattening"]
    ).astype(int)

    # Greed (incl extreme greed) + rising -> PUT-supportive (contrarian)
    feats["sent_regime_put_supportive"] = (
        (feats["sent_bucket_greed"] | feats["sent_bucket_extreme_greed"]) &
        feats["sent_is_rising"]
    ).astype(int)

    # Avoid longs: sustained extreme greed
    feats["sent_regime_avoid_longs"] = feats["sent_extreme_greed_persist_5d"]

    # ---------- 6) Interactions / "Option Signals" ----------
    # Intuition:
    # Call Option (Buy Exposure):
    #   - MVRV is Low (Undervalued OR New Low OR Near Historical Bottom) -> Price is cheap
    #   - Sentiment is Negative -> Crowd is fearful
    
    mvrv_is_cheap = (
        (feats['mvrv_comp_undervalued_90d'] == 1) |
        (feats['mvrv_comp_new_low_180d'] == 1) |
        (feats['mvrv_comp_near_bottom_any'] == 1)
    )
    sent_is_fear = (df['sentiment_norm'] < -1.0)  # "Very negative"

    feats['signal_option_call'] = (mvrv_is_cheap & sent_is_fear).astype(int)

    # Put Option (Sell/Hedge Exposure):
    #   - MVRV is High (Overheated OR New High OR Near Historical Top) -> Price is expensive
    #   - Sentiment is Positive -> Crowd is euphoric

    mvrv_is_expensive = (
        (feats['mvrv_comp_overheated_90d'] == 1) |
        (feats['mvrv_comp_new_high_180d'] == 1) |
        (feats['mvrv_comp_near_top_any'] == 1)
    )
    sent_is_greed = (df['sentiment_norm'] > 1.0) # "Very positive"

    feats['signal_option_put'] = (
        mvrv_is_expensive & 
        sent_is_greed & 
        (feats['mdia_slope_7d'] > 0) &
        (feats['whale_100_10k_change_30d'] < 0)
    ).astype(int)

    # ---------- 7) Labels: Long & Short Opportunities ----------
    # We create two distinct labels so the model can learn both directions.
    window = horizon_days
    
    # LONG Label: Did price go UP by target_return?
    fwd_max = df['mean_price'].rolling(window).max().shift(-window)
    target_long = df['mean_price'] * (1 + target_return)
    feats['label_good_move_long'] = (fwd_max >= target_long).astype(int)

    # SHORT Label: Did price DROP by target_return?
    fwd_min = df['mean_price'].rolling(window).min().shift(-window)
    target_short = df['mean_price'] * (1 - target_return)
    feats['label_good_move_short'] = (fwd_min <= target_short).astype(int)

    # Convert dict to DataFrame once at the end to avoid fragmentation
    feats = pd.DataFrame(feats, index=df.index)
    feats = feats.dropna().copy()
    return feats
