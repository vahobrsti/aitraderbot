import pandas as pd

def calculate(df: pd.DataFrame, horizon_days: int = 14, target_return: float = 0.05) -> pd.DataFrame:
    """
    Labels: Long & Short Opportunities
    """
    feats = {}
    window = horizon_days
    
    # Assuming 'mean_price' exists (from feature_builder prep)
    price = df['mean_price']
    
    # LONG Label: Did price go UP by target_return?
    fwd_max = price.rolling(window).max().shift(-window)
    target_long = price * (1 + target_return)
    feats['label_good_move_long'] = (fwd_max >= target_long).astype(int)

    # SHORT Label: Did price DROP by target_return?
    fwd_min = price.rolling(window).min().shift(-window)
    target_short = price * (1 - target_return)
    feats['label_good_move_short'] = (fwd_min <= target_short).astype(int)

    return pd.DataFrame(feats, index=df.index)
