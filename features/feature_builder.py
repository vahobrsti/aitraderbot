# features/feature_builder.py

import pandas as pd


def build_features_and_labels_from_raw(
    df: pd.DataFrame,
    horizon_days: int = 21,
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
    df['mvrv_composite_pct'] = (df['mvrv_composite_ratio'] - 1.0) * 100.0


    # Whale buckets: small vs big
    df['btc_holders_1_100'] = df['btc_holders_1_10'] + df['btc_holders_10_100']
    df['btc_holders_100_10k'] = df['btc_holders_100_1k'] + df['btc_holders_1k_10k']

    # Convenience aliases
    df['mvrv_long_short'] = df['mvrv_long_short_diff_usd']
    # normalize the sentiment
    df['sentiment_raw'] = df['sentiment_weighted_total']
    s = df["sentiment_raw"]
    mu = s.mean()
    sigma = s.std()  # sample std dev (perfectly fine)
    df["sentiment_norm"] = (s - mu) / sigma

    df['close'] = df['btc_close']

    feats = pd.DataFrame(index=df.index)

    # ---------- 1) MDIA slope ----------
    # Intuition: 
    # - Negative slope = Good (Capital injecting). The more negative, the better.
    # - Positive slope = Bad (Stagnant). Check Sentiment:
    #    -> MDIA(+) & Sentiment(+) => BAD (Price likely down)
    #    -> MDIA(+) & Sentiment(-) => GOOD (Recovery mid-term)

    # 1. Calculate Slopes (Deltas) AND Gradients (Per-Day Rate)
    # We need gradients to fairly compare 1d vs 2d vs 4d vs 7d
    for win in [1, 2, 4, 7, 14]:
        slope_col = f'mdia_slope_{win}d'
        feats[slope_col] = df['mdia'] - df['mdia'].shift(win)
        # Gradient = Delta / Days
        feats[f'mdia_grad_{win}d'] = feats[slope_col] / win

    # 1.1 Measure "Injection Acceleration" (The Term Structure of Slopes)
    # "Compare 1d to 2d, 2d to 4d, 4d to 7d"
    # If 1d is *more negative* than 2d, we are accelerating downwards (New Capital Entering).
    # We calculate the difference: (ShortTerm - LongTerm). 
    # Negative Value = Accelerating Downwards (Good).
    
    feats['mdia_accel_1d_vs_2d'] = feats['mdia_grad_1d'] - feats['mdia_grad_2d']
    feats['mdia_accel_2d_vs_4d'] = feats['mdia_grad_2d'] - feats['mdia_grad_4d']
    feats['mdia_accel_4d_vs_7d'] = feats['mdia_grad_4d'] - feats['mdia_grad_7d']

    # 1.2 "More positive = Higher Downside Risk" (Stagnation Severity)
    # We look at the 7d trend for this general risk gauge
    feats['mdia_slope_stagnation_severity'] = feats['mdia_slope_7d'].clip(lower=0)

    # 2. Interactions with Sentiment
    # (We use df['sentiment_norm'] which was calculated above)
    mdia_is_pos_7d = feats['mdia_slope_7d'] > 0
    sent_is_pos = df['sentiment_norm'] > 0
    sent_is_neg = df['sentiment_norm'] < 0

    # Case 1: MDIA(+) [Bad] AND Sentiment(+) [Bad] -> Very Bearish
    feats['signal_mdia_stagnant_overheated'] = (
        mdia_is_pos_7d & sent_is_pos
    ).astype(int)

    # Case 2: MDIA(+) [Bad] BUT Sentiment(-) [Good] -> Recovery Context
    feats['signal_mdia_stagnant_fear'] = (
        mdia_is_pos_7d & sent_is_neg
    ).astype(int)

    # Case 3: MDIA(-) [Good] -> Pure Capital Injection Signal
    # We just use the raw slope features (mdia_slope_Xd) for this, 
    # as "the more negative the better" is naturally captured by the raw value.
    feats['signal_mdia_injection'] = (
        feats['mdia_slope_7d'] < 0
    ).astype(int)

    # ---------- 2) MVRV long–short ----------
    # Intuition:
    # 1. Value: Positive (+) is Good, Negative (-) is Bad.
    # 2. Trend: Downtrend is VERY BEARISH (regardless of value). Uptrend is good.
    # 3. Action: Calculate slopes over 1d, 2d, 4d, 8d.

    feats['mvrv_ls_value'] = df['mvrv_long_short']

    # (a) Calculate Slopes/Trends over user-specified windows
    for win in [1, 2, 4, 8]:
        feats[f'mvrv_ls_trend_{win}d'] = df['mvrv_long_short'] - df['mvrv_long_short'].shift(win)

    # (b) Define Trend Regimes
    # "See if it is trending up or down" -> We use the 4d and 8d trends for robustness
    is_downtrending = (feats['mvrv_ls_trend_4d'] < 0) & (feats['mvrv_ls_trend_8d'] < 0)
    is_uptrending = (feats['mvrv_ls_trend_4d'] > 0) & (feats['mvrv_ls_trend_8d'] > 0)

    # (c) Interaction / Signals
    
    # "if + , good; if -, bad" (Base Condition)
    feats['mvrv_ls_is_positive'] = (feats['mvrv_ls_value'] > 0).astype(int)
    
    # "if it's in downtrend regardless of current value is very bearish"
    feats['mvrv_ls_bearish_flush'] = is_downtrending.astype(int)
    
    # "if it is in an uptrend, then good"
    feats['mvrv_ls_bullish_recovery'] = is_uptrending.astype(int)

    # Composite Score (Optional but helpful):
    # +1 if uptrend, -1 if downtrend, 0 neutral
    feats['mvrv_ls_trend_score'] = 0
    feats.loc[is_uptrending, 'mvrv_ls_trend_score'] = 1
    feats.loc[is_downtrending, 'mvrv_ls_trend_score'] = -1
    feats['mvrv_ls_very_bearish_regime'] = ( (feats['mvrv_ls_value'] < 0) & is_downtrending).astype(int)
    # ---------- 3) MVRV composite extremes [start] ----------
    mvrv = df["mvrv_composite_pct"]

    # (a) Rolling z-scores (generic ML features)
    for win in [90, 180, 365]:
        roll_mean = mvrv.rolling(win).mean()
        roll_std = mvrv.rolling(win).std()
        z = (mvrv - roll_mean) / (roll_std + 1e-9)

        feats[f"mvrv_comp_z_{win}d"] = z
        feats[f"mvrv_comp_undervalued_{win}d"] = (z < -1.0).astype(int)
        feats[f"mvrv_comp_overheated_{win}d"] = (z > 1.0).astype(int)

    # (b) Min / max and distance from them over 3m, 6m, 1y
    for win in [90, 180, 365]:
        roll_min = mvrv.rolling(win).min()
        roll_max = mvrv.rolling(win).max()

        feats[f"mvrv_comp_min_{win}d"] = roll_min
        feats[f"mvrv_comp_max_{win}d"] = roll_max

        feats[f"mvrv_comp_dist_from_min_{win}d"] = mvrv - roll_min
        feats[f"mvrv_comp_dist_from_max_{win}d"] = roll_max - mvrv

        # new lows / highs versus previous window (exclude today)
        prev_min = mvrv.shift(1).rolling(win).min()
        prev_max = mvrv.shift(1).rolling(win).max()

        feats[f"mvrv_comp_new_low_{win}d"] = (mvrv < prev_min).astype(int)
        feats[f"mvrv_comp_new_high_{win}d"] = (mvrv > prev_max).astype(int)

    # (b.1) closest distance to *any* horizon low / high
    feats["mvrv_comp_dist_from_min_any"] = feats[
        [
            "mvrv_comp_dist_from_min_90d",
            "mvrv_comp_dist_from_min_180d",
            "mvrv_comp_dist_from_min_365d",
        ]
    ].min(axis=1)

    feats["mvrv_comp_dist_from_max_any"] = feats[
        [
            "mvrv_comp_dist_from_max_90d",
            "mvrv_comp_dist_from_max_180d",
            "mvrv_comp_dist_from_max_365d",
        ]
    ].min(axis=1)

    # (c) "Near bottom" and "near top" flags (tunable in percentage points)
    MVRV_BOTTOM_THRESH = 5.0  # within 5 %-points of a local minimum
    MVRV_TOP_THRESH = 5.0     # within 5 %-points of a local maximum

    feats["mvrv_comp_near_bottom_any"] = (
        feats["mvrv_comp_dist_from_min_any"] < MVRV_BOTTOM_THRESH
    ).astype(int)

    feats["mvrv_comp_near_top_any"] = (
        feats["mvrv_comp_dist_from_max_any"] < MVRV_TOP_THRESH
    ).astype(int)

    for win in [90, 180, 365]:
        feats[f"mvrv_comp_near_bottom_{win}d"] = (
            feats[f"mvrv_comp_dist_from_min_{win}d"] < MVRV_BOTTOM_THRESH
        ).astype(int)

        feats[f"mvrv_comp_near_top_{win}d"] = (
            feats[f"mvrv_comp_dist_from_max_{win}d"] < MVRV_TOP_THRESH
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


    # ---------- 4) Whale accumulation ----------
    # Buckets of interest: 1-100 (Retail/Small) and 100-10k (Whale/Smart Money)
    cols_map = {
        '1_100': 'btc_holders_1_100',
        '100_10k': 'btc_holders_100_10k'
    }

    # 1. Base Changes over 1d, 2d, 4d, 7d
    windows = [1, 2, 4, 7]
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
        
        feats[f"sentiment_min_{win}d"] = roll_min
        feats[f"sentiment_max_{win}d"] = roll_max
        
        # Distance from extremes
        feats[f"sentiment_dist_from_min_{win}d"] = sent - roll_min
        feats[f"sentiment_dist_from_max_{win}d"] = roll_max - sent  # positive distance
        
        # Near Bottom / Top flags
        feats[f"sentiment_near_bottom_{win}d"] = (
            feats[f"sentiment_dist_from_min_{win}d"] < BOTTOM_THRESH
        ).astype(int)
        
        feats[f"sentiment_near_top_{win}d"] = (
            feats[f"sentiment_dist_from_max_{win}d"] < TOP_THRESH
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

    # (c) Aggregate Signals

    # Common conditions across 1m, 3m, 6m
    # "check historical values over 1 month, 3month and 6 months..."
    near_bottom_multi = (
        (feats["sentiment_near_bottom_30d"] == 1) |
        (feats["sentiment_near_bottom_90d"] == 1) |
        (feats["sentiment_near_bottom_180d"] == 1)
    )

    near_top_multi = (
        (feats["sentiment_near_top_30d"] == 1) |
        (feats["sentiment_near_top_90d"] == 1) |
        (feats["sentiment_near_top_180d"] == 1)
    )

    # BUY SIGNAL
    # 1. Current value is negative
    # 2. Close to min on 1m, 3m, 6m
    # 3. Bonus: Downtrend
    # OR: New Low (Strong signal) on longer timeframes (e.g. 6m or 1y)
    feats["sentiment_contrarian_buy"] = (
        (
            (sent < 0) & 
            near_bottom_multi & 
            (feats["sentiment_downtrend_7d"] == 1)
        )
        |
        (
            (feats["sentiment_new_low_180d"] == 1) | (feats["sentiment_new_low_365d"] == 1)
        )
    ).astype(int)

    # SELL SIGNAL
    # 1. Current value is positive
    # 2. Close to max on 1m, 3m, 6m
    # 3. Bonus: Uptrend
    # OR: New High (Strong signal)
    feats["sentiment_contrarian_sell"] = (
        (
            (sent > 0) & 
            near_top_multi & 
            (feats["sentiment_uptrend_7d"] == 1)
        )
        |
        (
            (feats["sentiment_new_high_180d"] == 1) | (feats["sentiment_new_high_365d"] == 1)
        )
    ).astype(int)

    # ---------- 6) Interactions ----------
    # (Note: Specific MDIA & Sentiment interactions were handled in Section 1)
    
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

    feats['signal_option_put'] = (mvrv_is_expensive & sent_is_greed).astype(int)

    # ---------- 7) Label: "good move within horizon_days?" ----------
    window = horizon_days
    fwd_max = df['close'].rolling(window).max().shift(-window + 1)
    target_level = df['close'] * (1 + target_return)
    label = (fwd_max >= target_level).astype(int)

    feats['label_good_move'] = label

    # Drop rows where we don't have enough history/forward data
    feats = feats.dropna().copy()
    return feats
