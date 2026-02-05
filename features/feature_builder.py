import pandas as pd
from features.metrics import (
    cycle,
    mdia,
    mvrv_long_short,
    mvrv_composite,
    whales,
    sentiment,
    interactions,
    labels,
)

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

    # ---------- Derived raw series (needed by multiple modules) ----------
    
    # 1. Whale raw buckets (for whales.py)
    if 'btc_holders_1_10' in df.columns:
        df['btc_holders_1_100'] = df['btc_holders_1_10'] + df['btc_holders_10_100']
        df['btc_holders_100_10k'] = df['btc_holders_100_1k'] + df['btc_holders_1k_10k']

    # 2. Aliases (for mvrv_long_short.py, labels.py)
    if 'mvrv_long_short_diff_usd' in df.columns:
        df['mvrv_long_short'] = df['mvrv_long_short_diff_usd']
    
    if 'btc_price_mean' in df.columns:
        df['mean_price'] = df['btc_price_mean']

    # 3. Sentiment Raw (for sentiment.py)
    if 'sentiment_weighted_total' in df.columns:
        df['sentiment_raw'] = df['sentiment_weighted_total']

    # Execute modules in dependency order
    feature_dfs = []

    # 1. Cycle (Independent)
    df_cycle = cycle.calculate(df)
    feature_dfs.append(df_cycle)
    df = pd.concat([df, df_cycle], axis=1)

    # 2. MDIA (Independent)
    df_mdia = mdia.calculate(df)
    feature_dfs.append(df_mdia)
    df = pd.concat([df, df_mdia], axis=1)

    # 3. MVRV Long/Short (Uses mvrv_long_short)
    df_mvrv_ls = mvrv_long_short.calculate(df)
    feature_dfs.append(df_mvrv_ls)
    df = pd.concat([df, df_mvrv_ls], axis=1)

    # 4. MVRV Composite (Uses raw mvrv cols + mvrv_usd_60d)
    df_mvrv_comp = mvrv_composite.calculate(df)
    feature_dfs.append(df_mvrv_comp)
    df = pd.concat([df, df_mvrv_comp], axis=1)

    # 5. Whales (Uses btc_holders_... and mvrv_usd_60d)
    df_whales = whales.calculate(df)
    feature_dfs.append(df_whales)
    df = pd.concat([df, df_whales], axis=1)

    # 6. Sentiment (Uses sentiment_raw)
    df_sent = sentiment.calculate(df)
    feature_dfs.append(df_sent)
    df = pd.concat([df, df_sent], axis=1)

    # 7. Interactions (Uses MVRV Comp features, Sentiment, Whales)
    df_inter = interactions.calculate(df)
    feature_dfs.append(df_inter)
    df = pd.concat([df, df_inter], axis=1)

    # 8. Labels (Uses mean_price)
    df_labels = labels.calculate(df, horizon_days=horizon_days, target_return=target_return)
    feature_dfs.append(df_labels)
    # No need to update df for labels as it's the last step
    
    # Combine all engineered features
    feats = pd.concat(feature_dfs, axis=1)
    
    # Remove duplicates if any
    feats = feats.loc[:, ~feats.columns.duplicated()]

    # Original cleanup
    feats = feats.dropna().copy()
    
    return feats
