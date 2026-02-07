import numpy as np
import pandas as pd

def calculate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bitcoin Halving Cycle Features
    Bitcoin halvings occur ~every 4 years (~1460 days). The market behaves
    differently at different phases of the cycle.
    """
    feats = {}

    # Historical halving dates
    HALVING_DATES = [
        pd.Timestamp('2012-11-28'),  # 50 -> 25 BTC
        pd.Timestamp('2016-07-09'),  # 25 -> 12.5 BTC
        pd.Timestamp('2020-05-11'),  # 12.5 -> 6.25 BTC
        pd.Timestamp('2024-04-19'),  # 6.25 -> 3.125 BTC
        pd.Timestamp('2028-04-01'),  # Estimated next halving
    ]
    CYCLE_LENGTH_DAYS = 1460  # ~4 years between halvings

    def get_cycle_info(date):
        """Get cycle info for a single date"""
        date = pd.Timestamp(date)
        # Find the most recent halving before or on this date
        past_halvings = [h for h in HALVING_DATES if h <= date]
        if not past_halvings:
            # Before first halving - use first halving as reference
            last_halving = HALVING_DATES[0]
            days_since = (date - last_halving).days  # Will be negative
        else:
            last_halving = max(past_halvings)
            days_since = (date - last_halving).days
        
        # Find next halving
        future_halvings = [h for h in HALVING_DATES if h > date]
        if future_halvings:
            next_halving = min(future_halvings)
            days_to_next = (next_halving - date).days
        else:
            # After last known halving - estimate
            days_to_next = CYCLE_LENGTH_DAYS - days_since
        
        # Cycle progress (0 = just halved, 1 = about to halve)
        cycle_progress = min(1.0, max(0.0, days_since / CYCLE_LENGTH_DAYS))
        
        return days_since, days_to_next, cycle_progress

    # Compute cycle features for each day
    cycle_data = [get_cycle_info(d) for d in df.index]
    days_since_halving = pd.Series([c[0] for c in cycle_data], index=df.index)
    days_to_next_halving = pd.Series([c[1] for c in cycle_data], index=df.index)
    cycle_progress = pd.Series([c[2] for c in cycle_data], index=df.index)

    # Core cycle features
    feats['cycle_days_since_halving'] = days_since_halving
    feats['cycle_days_to_next_halving'] = days_to_next_halving
    feats['cycle_progress'] = cycle_progress

    # Phase buckets (one-hot)
    feats['cycle_phase_0_6m'] = ((days_since_halving >= 0) & (days_since_halving < 180)).astype(int)
    feats['cycle_phase_6_12m'] = ((days_since_halving >= 180) & (days_since_halving < 365)).astype(int)
    feats['cycle_phase_12_18m'] = ((days_since_halving >= 365) & (days_since_halving < 548)).astype(int)
    feats['cycle_phase_18_30m'] = ((days_since_halving >= 548) & (days_since_halving < 912)).astype(int)
    feats['cycle_phase_30_48m'] = (days_since_halving >= 912).astype(int)

    # Smooth seasonality
    feats['cycle_sin'] = np.sin(2 * np.pi * cycle_progress)
    feats['cycle_cos'] = np.cos(2 * np.pi * cycle_progress)

    return pd.DataFrame(feats, index=df.index)
