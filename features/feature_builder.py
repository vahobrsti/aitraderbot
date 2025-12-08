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
    feats['mdia_slope_7d'] = df['mdia'] - df['mdia'].shift(7)
    feats['mdia_slope_30d'] = df['mdia'] - df['mdia'].shift(30)
    feats['mdia_slope_pos'] = (feats['mdia_slope_7d'] > 0).astype(int)
    feats['mdia_slope_neg'] = (feats['mdia_slope_7d'] < 0).astype(int)

    # ---------- 2) MVRV long–short ----------
    feats['mvrv_ls_value'] = df['mvrv_long_short']
    feats['mvrv_ls_trend_7d'] = df['mvrv_long_short'] - df['mvrv_long_short'].shift(7)
    feats['mvrv_ls_trend_30d'] = df['mvrv_long_short'] - df['mvrv_long_short'].shift(30)
    feats['mvrv_ls_positive'] = (feats['mvrv_ls_value'] > 0).astype(int)
    feats['mvrv_ls_negative'] = (feats['mvrv_ls_value'] < 0).astype(int)
    feats['mvrv_ls_uptrend'] = (feats['mvrv_ls_trend_30d'] > 0).astype(int)
    feats['mvrv_ls_downtrend'] = (feats['mvrv_ls_trend_30d'] < 0).astype(int)

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
    for bucket, col in [
        ('1_100', 'btc_holders_1_100'),
        ('100_10k', 'btc_holders_100_10k'),
    ]:
        for win in [1, 7, 30, 180]:
            feats[f'whale_{bucket}_change_{win}d'] = df[col] - df[col].shift(win)

        feats[f'whale_{bucket}_accum_7d'] = (
            feats[f'whale_{bucket}_change_7d'] > 0
        ).astype(int)
        feats[f'whale_{bucket}_accum_30d'] = (
            feats[f'whale_{bucket}_change_30d'] > 0
        ).astype(int)

    feats['broad_accum_flag'] = (
        (feats['whale_1_100_accum_30d'] == 1)
        & (feats['whale_100_10k_accum_30d'] == 1)
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
    feats['mdia_pos_and_sent_pos'] = (
        (feats['mdia_slope_7d'] > 0)
        & (feats['sentiment_z_30d'] > 0)
    ).astype(int)

    feats['mdia_pos_and_sent_neg'] = (
        (feats['mdia_slope_7d'] > 0)
        & (feats['sentiment_z_30d'] < 0)
    ).astype(int)

    feats['strong_value_env'] = (
        (feats['mvrv_comp_undervalued_90d'] == 1)
        & (feats['whale_100_10k_accum_30d'] == 1)
    ).astype(int)

    # ---------- 7) Label: "good move within horizon_days?" ----------
    window = horizon_days
    fwd_max = df['close'].rolling(window).max().shift(-window + 1)
    target_level = df['close'] * (1 + target_return)
    label = (fwd_max >= target_level).astype(int)

    feats['label_good_move'] = label

    # Drop rows where we don't have enough history/forward data
    feats = feats.dropna().copy()
    return feats
