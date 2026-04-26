"""
Range Label: Forward outcome classification from DB OHLC.

For each date, looks forward N observations (rows) and classifies:
  - RANGE_{N}D:      price stayed inside +/-threshold of close
  - BREAKOUT_UP:     high exceeded +threshold but low stayed above -threshold
  - BREAKOUT_DOWN:   low breached -threshold but high stayed below +threshold
  - BREAKOUT_BOTH:   both +threshold and -threshold were breached (whipsaw)

Uses btc_high / btc_low from RawDailyData (not CSV).

NOTE: "horizon" means next N rows (observations), not calendar days.
If the daily index has gaps, the actual calendar span may exceed N days.
The RawDailyData table has continuous daily rows, so in practice N rows ≈ N days.
"""

import pandas as pd


# Column name prefix — frozen to match condor gate label definition
_PREFIX = "range_7d"


def calculate(df: pd.DataFrame, horizon: int = 7, threshold: float = 0.10) -> pd.DataFrame:
    """
    Args:
        df: DataFrame with DatetimeIndex, must contain btc_close, btc_high, btc_low.
        horizon: forward-looking window in rows/observations (default 7).
        threshold: percentage band (default 0.10 = +/-10%).

    Returns:
        DataFrame with columns:
          - range_7d_label: categorical string (RANGE_7D, BREAKOUT_UP, BREAKOUT_DOWN, BREAKOUT_BOTH)
          - range_7d_binary: 1 if RANGE_7D, 0 otherwise
          - range_7d_max_up_pct: max upside % in window
          - range_7d_max_down_pct: max downside % in window (positive = down)
    """
    feats = {}

    close = df["btc_close"]

    # Rolling max high and min low over the NEXT `horizon` observations
    # shift(-horizon) aligns the future window to the current row
    fwd_max_high = df["btc_high"].rolling(horizon).max().shift(-horizon)
    fwd_min_low = df["btc_low"].rolling(horizon).min().shift(-horizon)

    max_up_pct = (fwd_max_high - close) / close
    max_down_pct = (close - fwd_min_low) / close

    feats[f"{_PREFIX}_max_up_pct"] = max_up_pct
    feats[f"{_PREFIX}_max_down_pct"] = max_down_pct

    broke_up = max_up_pct >= threshold
    broke_down = max_down_pct >= threshold

    range_tag = f"RANGE_7D"
    label = pd.Series(range_tag, index=df.index)
    label[broke_up & ~broke_down] = "BREAKOUT_UP"
    label[~broke_up & broke_down] = "BREAKOUT_DOWN"
    label[broke_up & broke_down] = "BREAKOUT_BOTH"
    # Rows where forward data is missing get NaN
    label[fwd_max_high.isna()] = pd.NA

    feats[f"{_PREFIX}_label"] = label
    feats[f"{_PREFIX}_binary"] = (label == range_tag).astype("Int64")

    return pd.DataFrame(feats, index=df.index)
