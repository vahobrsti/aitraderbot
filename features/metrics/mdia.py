import pandas as pd

def calculate(df: pd.DataFrame) -> pd.DataFrame:
    """
    MDIA slope & Regimes (Data-Aligned + Persistence)
    
    Empirical constraints from data analysis:
    - 1d lead corr ≈ -0.016 (noise)
    - 2d lead corr ≈ -0.053
    - 4d lead corr ≈ -0.110
    - 7d lead corr ≈ -0.152
    
    Design principles:
    - 7d/4d are the backbone (highest predictive power)
    - 2d is supportive/confirmer only
    - 1d is ignored for regime decisions
    - 3-day EMA smoothing on slopes before z-scoring (reduce noise)
    - 2-day persistence for regime flags (avoid churn)
    - MDIA falling = Capital Inflow (Bullish)
    - MDIA rising = Aging/trend-unfriendly (NOT bearish by itself)
    """
    feats = {}
    
    # 1. Calculate Raw Slopes with optional smoothing
    mdia_horizons = [1, 2, 4, 7]
    for win in mdia_horizons:
        slope_col = f'mdia_slope_{win}d'
        # Raw delta
        raw_slope = df['mdia'] - df['mdia'].shift(win)
        
        # Apply 3-day EMA smoothing for 2d/4d/7d (reduce high-freq noise)
        # Keep 1d unsmoothed for diagnostics
        if win >= 2:
            slope_series = raw_slope.ewm(span=3, adjust=False).mean()
        else:
            slope_series = raw_slope
            
        feats[slope_col] = slope_series
        feats[f'mdia_slope_raw_{win}d'] = raw_slope  # Keep raw for debugging
        
        # 2. Normalize (Rolling Z-Score of the Smoothed Slope)
        z_win = 56
        roll_mean = slope_series.rolling(z_win, min_periods=30).mean()
        roll_std = slope_series.rolling(z_win, min_periods=30).std()
        
        # Avoid div by zero
        z_score = (slope_series - roll_mean) / (roll_std + 1e-9)
        feats[f'mdia_slope_z_{win}d'] = z_score

        # 3. Ordinal Buckets (data-aligned thresholds)
        bucket_col = f'mdia_bucket_{win}d'
        buckets = pd.Series(0, index=df.index)  # Default 0 = neutral
        
        # Positive thresholds (rising MDIA = aging)
        buckets.loc[z_score > 0.5] = 1
        buckets.loc[z_score > 1.0] = 2  # Strong aging
        
        # Negative thresholds (falling MDIA = inflow)
        buckets.loc[z_score < -0.25] = -1  # Inflow
        buckets.loc[z_score < -1.0] = -2   # Strong inflow
        
        feats[bucket_col] = buckets.astype(int)

    # Convert to DataFrame for vectorized ops
    temp_df = pd.DataFrame(feats, index=df.index)
    features_out = temp_df.copy()

    # =========================================================================
    # 4. SIGN + MAGNITUDE CONDITIONS (data-aligned: 4d/7d primary)
    # =========================================================================
    z_7d = temp_df['mdia_slope_z_7d']
    z_4d = temp_df['mdia_slope_z_4d']
    z_2d = temp_df['mdia_slope_z_2d']
    
    # Primary inflow signals (based on predictive horizons)
    z_7d_inflow = (z_7d < -0.25)
    z_4d_inflow = (z_4d < -0.25)
    z_2d_inflow = (z_2d < -0.25)
    z_2d_neutral_or_inflow = (z_2d <= 0.25)  # Not fighting
    
    z_7d_strong = (z_7d < -1.0)
    z_4d_strong = (z_4d < -1.0)
    
    # Aging signals (rising MDIA)
    z_7d_positive = (z_7d > 0.25)
    z_4d_positive = (z_4d > 0.25)

    # =========================================================================
    # 5. BREADTH REGIMES (with 2-day persistence)
    # =========================================================================
    
    # Strong Inflow: (4d strong OR 7d strong) AND both negative
    raw_strong_inflow = (
        (z_7d_strong | z_4d_strong) &
        (z_7d < 0) &
        (z_4d < 0) &
        z_2d_neutral_or_inflow
    )
    # Apply 2-day persistence: require signal on both today and yesterday
    features_out['mdia_breadth_strong'] = (
        raw_strong_inflow & raw_strong_inflow.shift(1).astype(bool).fillna(False)
    ).astype(int)
    
    # Moderate Inflow: (7d inflow) OR (4d + 2d both inflow)
    raw_moderate_inflow = (
        z_7d_inflow |
        (z_4d_inflow & z_2d_inflow)
    )
    features_out['mdia_breadth_moderate'] = (
        raw_moderate_inflow & raw_moderate_inflow.shift(1).astype(bool).fillna(False)
    ).astype(int)
    
    # Legacy compatibility
    features_out['mdia_breadth_inflow'] = features_out['mdia_breadth_strong']
    
    # Diagnostic: count of predictive horizons in inflow (4d, 7d only)
    features_out['mdia_inflow_count'] = (
        z_4d_inflow.astype(int) + z_7d_inflow.astype(int)
    )

    # =========================================================================
    # 6. ACCELERATION / FRESHENING (momentum building)
    # =========================================================================
    # "Freshening": 7d negative, 4d negative, 2d just joined negative
    z_2d_prev = z_2d.shift(1)
    
    features_out['mdia_freshening'] = (
        (z_7d < 0) &
        (z_4d < 0) &
        (z_2d < 0) &
        (z_2d_prev >= 0)  # 2d just turned negative
    ).astype(int)
    
    # Magnitude-based: 2d more extreme than 4d, both negative
    features_out['mdia_accel_magnitude'] = (
        (z_2d.abs() > z_4d.abs()) &
        (z_2d < 0) &
        (z_4d < 0)
    ).astype(int)
    
    # Combined acceleration
    features_out['mdia_is_accelerating'] = (
        (features_out['mdia_freshening'] == 1) |
        (features_out['mdia_accel_magnitude'] == 1)
    ).astype(int)
    
    # Legacy
    features_out['mdia_accel_2d_vs_4d'] = features_out['mdia_accel_magnitude']

    # =========================================================================
    # 7. FINAL REGIMES (with persistence)
    # =========================================================================

    # (A) Strong Inflow: Strong breadth + accelerating
    raw_regime_strong = (
        (features_out['mdia_breadth_strong'] == 1) &
        (features_out['mdia_is_accelerating'] == 1)
    )
    features_out['mdia_regime_strong_inflow'] = (
        raw_regime_strong & raw_regime_strong.shift(1).astype(bool).fillna(False)
    ).astype(int)

    # (B) Inflow (Moderate): Breadth without strong (exclusive)
    raw_regime_inflow = (
        (features_out['mdia_breadth_moderate'] == 1) &
        (features_out['mdia_regime_strong_inflow'] == 0)
    )
    features_out['mdia_regime_inflow'] = (
        raw_regime_inflow & raw_regime_inflow.shift(1).astype(bool).fillna(False)
    ).astype(int)

    # (C) Aging / No Inflow: 7d AND 4d both positive
    # This means "trend-unfriendly" NOT "bearish" - whales/MVRV decide bearishness
    raw_regime_aging = (
        z_7d_positive & z_4d_positive
    )
    features_out['mdia_regime_aging'] = (
        raw_regime_aging & raw_regime_aging.shift(1).astype(bool).fillna(False)
    ).astype(int)
    
    # Legacy: distribution points to aging (but fusion.py should NOT use for shorts)
    features_out['mdia_regime_distribution'] = features_out['mdia_regime_aging']

    return features_out
