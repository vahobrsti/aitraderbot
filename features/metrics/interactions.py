import pandas as pd

def calculate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interactions / "Option Signals"
    """
    feats = {}
    
    # 1. Call Option (Buy Exposure)
    #   - MVRV is Low (Undervalued OR New Low OR Near Historical Bottom)
    #   - Sentiment is Negative
    
    # Assuming MVRV Composite module ran first
    mvrv_is_cheap = (
        (df.get('mvrv_comp_undervalued_90d', 0) == 1) |
        (df.get('mvrv_comp_new_low_180d', 0) == 1) |
        (df.get('mvrv_comp_near_bottom_any', 0) == 1)
    )
    
    # Assuming Sentiment module ran first
    # Using 'sentiment_norm' which should be present
    sent_is_fear = (df['sentiment_norm'] < -1.0) 

    feats['signal_option_call'] = (mvrv_is_cheap & sent_is_fear).astype(int)

    # 2. Put Option (Sell/Hedge Exposure)
    #   - MVRV-60d is near peak
    #   - Sentiment is Positive
    #   - Whales distributing

    # From MVRV Composite
    mvrv_60d_near_peak = (
        (df.get('mvrv_60d_pct_rank', 0) >= 0.80) |
        (df.get('mvrv_60d_dist_from_max', 1.0) <= 0.20)
    )
    
    sent_is_greed = (df['sentiment_norm'] > 1.0)
    
    # From Whales
    whale_distrib = (df.get('whale_regime_distribution', 0) == 1)

    feats['signal_option_put'] = (
        mvrv_60d_near_peak & 
        sent_is_greed & 
        whale_distrib
    ).astype(int)

    return pd.DataFrame(feats, index=df.index)
