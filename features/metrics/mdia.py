import pandas as pd

def calculate(df: pd.DataFrame) -> pd.DataFrame:
    """
    MDIA slope & Regimes (Robust)
    - MDIA falling = Capital Inflow (Bullish)
    - Broad participation + acceleration
    """
    feats = {}
    
    # 1. Calculate Raw Slopes
    mdia_horizons = [1, 2, 4, 7]
    for win in mdia_horizons:
        slope_col = f'mdia_slope_{win}d'
        # Raw delta
        feats[slope_col] = df['mdia'] - df['mdia'].shift(win)
        
        # 2. Normalize (Rolling Z-Score of the Slope)
        z_win = 90
        slope_series = feats[slope_col]
        roll_mean = slope_series.rolling(z_win, min_periods=30).mean()
        roll_std = slope_series.rolling(z_win, min_periods=30).std()
        
        # Avoid div by zero
        z_score = (slope_series - roll_mean) / (roll_std + 1e-9)
        feats[f'mdia_slope_z_{win}d'] = z_score

        # 3. Ordinal Buckets
        bucket_col = f'mdia_bucket_{win}d'
        buckets = pd.Series(0, index=df.index) # Default 0
        
        buckets.loc[z_score > 0.5] = 1
        buckets.loc[z_score < -0.5] = -1
        buckets.loc[z_score < -1.5] = -2
        
        feats[bucket_col] = buckets.astype(int)

    # Convert distinct series to a DataFrame for easier vectorized ops
    temp_df = pd.DataFrame(feats, index=df.index)

    # 4. Breadth Regime: "All horizons showing inflow"
    features_out = temp_df.copy()
    
    features_out['mdia_breadth_inflow'] = (
        (temp_df['mdia_bucket_1d'] <= -1) &
        (temp_df['mdia_bucket_2d'] <= -1) &
        (temp_df['mdia_bucket_4d'] <= -1) &
        (temp_df['mdia_bucket_7d'] <= -1)
    ).astype(int)

    # 4b. Moderate Breadth: "3 out of 4 horizons showing inflow"
    inflow_count = (
        (temp_df['mdia_bucket_1d'] <= -1).astype(int) +
        (temp_df['mdia_bucket_2d'] <= -1).astype(int) +
        (temp_df['mdia_bucket_4d'] <= -1).astype(int) +
        (temp_df['mdia_bucket_7d'] <= -1).astype(int)
    )
    features_out['mdia_breadth_inflow_3of4'] = (inflow_count >= 3).astype(int)
    features_out['mdia_inflow_count'] = inflow_count

    # 5. Acceleration Regime
    # Compare buckets: is 2d bucket MORE NEGATIVE (lower) than 4d bucket?
    features_out['mdia_accel_2d_vs_4d'] = (
        temp_df['mdia_bucket_2d'] < temp_df['mdia_bucket_4d']
    ).astype(int)
    
    features_out['mdia_accel_2d_vs_1d'] = (
        temp_df['mdia_bucket_2d'] < temp_df['mdia_bucket_1d']
    ).astype(int)
    
    features_out['mdia_is_accelerating'] = (
        (features_out['mdia_accel_2d_vs_4d'] == 1) | 
        (features_out['mdia_accel_2d_vs_1d'] == 1)
    ).astype(int)

    # 6. Final Regimes

    # (A) Strong Inflow
    features_out['mdia_regime_strong_inflow'] = (
        (features_out['mdia_breadth_inflow'] == 1) &
        (features_out['mdia_is_accelerating'] == 1)
    ).astype(int)

    # (B) Inflow (Moderate)
    features_out['mdia_regime_inflow'] = (
        (features_out['mdia_breadth_inflow_3of4'] == 1) &
        (features_out['mdia_regime_strong_inflow'] == 0) # Exclusive
    ).astype(int)

    # (C) Distribution / Risk
    features_out['mdia_regime_distribution'] = (
        (temp_df['mdia_bucket_1d'] >= 1) &
        (temp_df['mdia_bucket_2d'] >= 1) &
        (features_out['mdia_breadth_inflow'] == 0)
    ).astype(int)

    return features_out
