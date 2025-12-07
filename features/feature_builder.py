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
    df['sentiment_raw'] = df['sentiment_weighted_total']
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

    # ---------- 3) MVRV composite extremes ----------
    for win in [90, 180, 365]:
        roll_mean = df['mvrv_composite_pct'].rolling(win).mean()
        roll_std = df['mvrv_composite_pct'].rolling(win).std()
        z = (df['mvrv_composite_pct'] - roll_mean) / (roll_std + 1e-9)
        feats[f'mvrv_comp_z_{win}d'] = z
        feats[f'mvrv_comp_undervalued_{win}d'] = (z < -1.0).astype(int)
        feats[f'mvrv_comp_overheated_{win}d'] = (z > 1.0).astype(int)
        feats['mvrv_composite_pct'] = df['mvrv_composite_pct']

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
    for win in [30, 180, 365]:
        mean = df['sentiment_raw'].rolling(win).mean()
        std = df['sentiment_raw'].rolling(win).std()
        feats[f'sentiment_z_{win}d'] = (
            df['sentiment_raw'] - mean
        ) / (std + 1e-9)

    feats['sentiment_trend_7d'] = df['sentiment_raw'] - df['sentiment_raw'].shift(7)
    feats['sentiment_very_negative'] = (
        feats['sentiment_z_180d'] < -1.0
    ).astype(int)
    feats['sentiment_downtrend_7d'] = (
        feats['sentiment_trend_7d'] < 0
    ).astype(int)

    feats['contrarian_buy_flag'] = (
        (feats['sentiment_very_negative'] == 1)
        & (feats['sentiment_downtrend_7d'] == 1)
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
