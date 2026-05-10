# MVRV Short Signal (Bear Market Tactical)

A tactical short signal designed specifically for bear market conditions (cycle days 540-900 post-halving).

## Philosophy

The signal exploits a behavioral pattern in bear markets: when short-term holders (7-day MVRV) are profitable while medium-term holders (60-day MVRV) are at or above breakeven, the market is temporarily overextended. In bear regimes, these relief rallies tend to fade quickly as holders who bought the dip take profits, creating predictable mean-reversion opportunities.

The key insight is the **overlay combination**:
- `mvrv_7d >= 1.02`: Recent buyers are profitable (short-term greed)
- `mvrv_60d >= 1.0`: Medium-term holders at breakeven (no capitulation cushion)

When both conditions align in a bear cycle, price tends to drop 4% before rising 4% with a **1.8:1 ratio** (58% drop-first vs 31% rise-first). This edge was validated out-of-sample across three independent bear markets (2018, 2022, 2025-2026).

## Why DCA Works Here

The $30/$60 split (33%/67%) exploits the asymmetry:
- If drop hits first (58%): Small win on small position
- If rise hits first, then drops (23%): Big win on averaged-down position  
- If rise hits first, never drops (19%): Small loss on small position

This creates positive expected value (+1.99% per trade) while limiting downside exposure.

## Performance

**Hit Rate:** 68.2% (15/22 trades)

### Backtest Results by Period

| Period | Role | Signals | Drop First | Rise First | Ratio |
|--------|------|---------|------------|------------|-------|
| 2018 | Train | 38 | 55% | 37% | 1.5:1 |
| 2022 | Test | 32 | 56% | 31% | 1.8:1 |
| 2025-2026 | Test | 9 | 67% | 11% | 6.0:1 |

## DCA Execution Strategy

- **Entry 1:** 33% of position at signal price
- **DCA Trigger:** +4% price rise → add remaining 67%
- **Target:** -4% from original entry (not averaged entry)
- **Max Hold:** 5 days
- **Expected Value:** +1.99% per trade ($1.79 per $90 risked)

## Strategy Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mvrv7d` | 1.02 | MVRV 7d threshold (short-term profit level) |
| `--mvrv60d` | 1.0 | MVRV 60d overlay (medium-term breakeven) |
| `--target` | 4.0 | Target drop % |
| `--window` | 5 | Window days to hit target |
| `--bear-start` | 540 | Bear mode start (cycle days post-halving) |
| `--bear-end` | 900 | Bear mode end (cycle days post-halving) |

## Commands

```bash
# Run backtest with default parameters
python manage.py analyze_mvrv_short

# Adjust thresholds
python manage.py analyze_mvrv_short --mvrv7d 1.05 --target 5

# Walk-forward validation (train/test split by bear market)
python manage.py analyze_mvrv_short --walkforward

# Check today's signal status
python manage.py analyze_mvrv_short --today
```

## Integration with Trade Setup

When MVRV_SHORT fires, the Trade Setup Builder constructs a bear put spread:

| Parameter | Value | Source |
|-----------|-------|--------|
| DTE | 12-18d | TTH p75 (10d) + buffer |
| Spread Width | 9% | MFE p75 (13.9%) × 0.65 |
| Stop Loss | 7.0% | MAE(W) p75 (7.19%) |
| Delta Target | -0.60 | Shakeout-heavy signal |
| Take Profit | 70% | Standard spread TP |
| Scale Down | Day 7 | Reduce position if not profitable |

## Signal Flow

```
Bear Mode Active (days 540-900)
    ↓
MVRV 7d >= 1.02 (short-term greed)
    ↓
MVRV 60d >= 1.0 (no capitulation cushion)
    ↓
Cooldown Check (5 days)
    ↓
MVRV_SHORT Signal Fires
    ↓
Trade Setup Builder → Bear Put Spread
    ↓
Validation (11 checks)
    ↓
Telegram Notification + API
```

## Key Insights

1. **Bear-only signal**: Only fires during cycle days 540-900 post-halving
2. **DCA is essential**: The 33%/67% split exploits the asymmetric payoff
3. **Shakeout-heavy (57%)**: Most winners experience price going against before hitting target
4. **High invalidation (43%)**: Many winners get invalidated before hit - need wide stops
5. **Patience required**: TTH p75 is 10 days (slowest of all signals)
6. **Wide stops needed**: MAE(W) p75 is 7.19% (second highest after OPTION_CALL)

## Path Profile

The Trade Setup Builder automatically detects MVRV_SHORT as a shakeout-heavy signal and recommends DCA entry:

```json
{
  "path_profile": {
    "shakeout_pct": 0.57,
    "invalidation_pct": 0.43,
    "mae_p75": 0.0719,
    "clean_win_pct": 0.571,
    "is_shakeout_heavy": true,
    "is_invalidation_heavy": true,
    "entry_strategy": "dca",
    "entry_note": "33% initial, 67% DCA at +7% (shakeout-heavy)"
  }
}
```

## Related Documentation

- [Entry Policy Design](entry_policy_design.md) - Full policy calibration
- [Iron Condor Spec](iron_condor_spec.md) - Range-bound strategy
- [Production Setup](PRODUCTION_SETUP.md) - Deployment guide
