import pandas as pd

def calculate(df: pd.DataFrame, horizon_days: int = 14, target_return: float = 0.05) -> pd.DataFrame:
    """
    Labels: Long & Short Opportunities

    Signal fires at EOD of date T → entry is at the open of T+1.
    Entry price = btc_open shifted forward by 1 day (next row's open).
    Forward window = T+1 through T+horizon_days (excludes signal day).
    """
    feats = {}
    window = horizon_days

    # Entry price is next day's open (signal fires EOD, enter at next open)
    entry_price = df['btc_open'].shift(-1)

    # Forward window starts from T+1 (shift by window+1 to exclude signal day)
    # rolling(window) over btc_high/btc_low, then shift(-(window+1)) aligns
    # the window to [T+1 .. T+window] instead of [T .. T+window-1]
    fwd_max = df['btc_high'].rolling(window).max().shift(-(window + 1))
    fwd_min = df['btc_low'].rolling(window).min().shift(-(window + 1))

    # LONG Label: Did price go UP by target_return from next-day open?
    target_long = entry_price * (1 + target_return)
    feats['label_good_move_long'] = (fwd_max >= target_long).astype(int)

    # SHORT Label: Did price DROP by target_return from next-day open?
    target_short = entry_price * (1 - target_return)
    feats['label_good_move_short'] = (fwd_min <= target_short).astype(int)

    return pd.DataFrame(feats, index=df.index)
