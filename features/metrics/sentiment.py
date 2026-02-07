import pandas as pd

def calculate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sentiment normalized features and trends
    """
    feats = {}
    
    # Pre-calculate sentiment_norm if not present
    # The original code did:
    # df['sentiment_raw'] = df['sentiment_weighted_total']
    # s = df["sentiment_raw"]
    # mu = s.expanding().mean()
    # sigma = s.expanding().std()
    # df["sentiment_norm"] = (s - mu) / sigma
    
    # We should probably do this here if it's not in df.
    # Safe to re-compute or check.
    if 'sentiment_norm' not in df.columns:
        if 'sentiment_weighted_total' in df.columns:
            s = df['sentiment_weighted_total']
            feats['sentiment_raw'] = s
            mu = s.expanding().mean()
            sigma = s.expanding().std()
            sent = (s - mu) / sigma
            feats['sentiment_norm'] = sent
        else:
             # Cannot calculate
             return pd.DataFrame()
    else:
        sent = df['sentiment_norm']
        feats['sentiment_norm'] = sent # Pass through if it was strictly required to be part of output

    # (a) 7-day trend
    feats["sentiment_trend_7d"] = sent - sent.shift(7)
    feats["sentiment_downtrend_7d"] = (feats["sentiment_trend_7d"] < 0).astype(int)
    feats["sentiment_uptrend_7d"] = (feats["sentiment_trend_7d"] > 0).astype(int)

    # (b) Rolling stats
    windows = [30, 90, 180, 365]
    for win in windows:
        # Min / Max
        roll_min = sent.rolling(win).min()
        roll_max = sent.rolling(win).max()
        roll_range = roll_max - roll_min
        
        feats[f"sentiment_min_{win}d"] = roll_min
        feats[f"sentiment_max_{win}d"] = roll_max
        
        # Relative Distance
        safe_range = roll_range.clip(lower=1e-9)
        dist_from_min = sent - roll_min
        dist_from_max = roll_max - sent
        
        rel_dist_min = dist_from_min / safe_range
        rel_dist_max = dist_from_max / safe_range
        
        LEGACY_REL_THRESH = 0.10
        feats[f"sentiment_near_bottom_{win}d"] = (rel_dist_min < LEGACY_REL_THRESH).astype(int)
        feats[f"sentiment_near_top_{win}d"] = (rel_dist_max < LEGACY_REL_THRESH).astype(int)

        # New Lows / Highs
        prev_min = sent.shift(1).rolling(win).min()
        prev_max = sent.shift(1).rolling(win).max()
        
        feats[f"sentiment_new_low_{win}d"] = (sent < prev_min).astype(int)
        feats[f"sentiment_new_high_{win}d"] = (sent > prev_max).astype(int)

        # Z-scores
        roll_mean = sent.rolling(win).mean()
        roll_std = sent.rolling(win).std()
        feats[f"sentiment_z_{win}d"] = (sent - roll_mean) / (roll_std + 1e-9)

    # (c) Relative Sentiment (Crowd Psychology)
    sent_roll_pct_180d = sent.rolling(180, min_periods=60).apply(
        lambda x: x.rank(pct=True).iloc[-1],
        raw=False
    )
    feats["sentiment_roll_pct_180d"] = sent_roll_pct_180d

    # Buckets
    feats["sent_bucket_extreme_fear"]  = (sent_roll_pct_180d < 0.10).astype(int)
    feats["sent_bucket_fear"]          = ((sent_roll_pct_180d >= 0.10) & (sent_roll_pct_180d < 0.30)).astype(int)
    feats["sent_bucket_neutral"]       = ((sent_roll_pct_180d >= 0.30) & (sent_roll_pct_180d < 0.70)).astype(int)
    feats["sent_bucket_greed"]         = ((sent_roll_pct_180d >= 0.70) & (sent_roll_pct_180d < 0.90)).astype(int)
    feats["sent_bucket_extreme_greed"] = (sent_roll_pct_180d >= 0.90).astype(int)

    # Momentum overlays
    MOMENTUM_THRESH = 0.05
    sent_change_3d = sent.diff(3)
    sent_change_5d = sent.diff(5)

    feats["sent_is_rising"]  = ((sent_change_3d > MOMENTUM_THRESH) & (sent_change_5d > MOMENTUM_THRESH)).astype(int)
    feats["sent_is_falling"] = ((sent_change_3d < -MOMENTUM_THRESH) & (sent_change_5d < -MOMENTUM_THRESH)).astype(int)

    # Flattening
    roll_std_90 = sent.rolling(90, min_periods=30).std().fillna(1.0) 
    feats["sent_is_flattening"] = (sent_change_3d.abs() < (0.25 * roll_std_90)).astype(int)

    # Persistence
    feats["sent_extreme_greed_persist_5d"] = (
        feats["sent_bucket_extreme_greed"]
        .rolling(5, min_periods=5).min().fillna(0).astype(int)
    )
    feats["sent_extreme_fear_persist_5d"] = (
        feats["sent_bucket_extreme_fear"]
        .rolling(5, min_periods=5).min().fillna(0).astype(int)
    )
    feats["sent_greed_persist_7d"] = (
        feats["sent_bucket_greed"]
        .rolling(7, min_periods=7).min().fillna(0).astype(int)
    )
    feats["sent_fear_persist_7d"] = (
        feats["sent_bucket_fear"]
        .rolling(7, min_periods=7).min().fillna(0).astype(int)
    )

    # Trading regimes
    feats["sent_regime_call_supportive"] = (
        (feats["sent_bucket_fear"] | feats["sent_bucket_extreme_fear"]) &
        feats["sent_is_rising"]
    ).astype(int)

    feats["sent_regime_call_mean_reversion"] = (
        feats["sent_bucket_extreme_fear"] &
        feats["sent_is_flattening"]
    ).astype(int)

    feats["sent_regime_put_supportive"] = (
        (feats["sent_bucket_greed"] | feats["sent_bucket_extreme_greed"]) &
        feats["sent_is_rising"]
    ).astype(int)

    feats["sent_regime_avoid_longs"] = feats["sent_extreme_greed_persist_5d"]

    return pd.DataFrame(feats, index=df.index)
