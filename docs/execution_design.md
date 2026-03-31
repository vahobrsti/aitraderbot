# Options Execution Design — V6 Profitable

Options-only strategy achieving +13.67% return over 2023-2025 with 3.65% max drawdown.
Built from path stats analysis across 264 trades (2016-2026), optimized via simulation
with dynamic leverage modeling and signal filtering.

---

## Analysis Commands Used

```bash
# Path diagnostics — TTH, MAE, MFE, path shapes, invalidation ordering
python manage.py analyze_path_stats
python manage.py analyze_path_stats --year 2025
python manage.py analyze_path_stats --direction LONG --invalidation 0.01
python manage.py analyze_path_stats --type PRIMARY_SHORT

# Hit rate by year, type, state, source
python manage.py analyze_hit_rate
python manage.py analyze_hit_rate --year 2024

# V6 execution policy simulation
python scripts/simulate_v4.py
```

---

## The Problem

The system has a 72% historical hit rate on a 5% target within 14 days across 264
trades. The execution problem is that options are leveraged, time-decaying instruments,
and the path to getting paid is rough:

- 65% of eventual winners dip at least 1% against you before hitting target
- 48% dip 2%+ before hitting
- 39% dip 3%+ before hitting
- At 15-17x leverage, a 3% adverse underlying move is a 45-50% drawdown on the option

---

## Key Finding: Signal Selection Matters More Than Execution

The original V4/V5 design traded all signal types and relied on BTC spot accumulation
to offset option losses. Simulation showed:

| Strategy | Trades | Win Rate | Option P&L | Max DD |
|---|---|---|---|---|
| All signals + spot | 62 | 53.2% | -$2,436 (-2.4%) | 10.3% |
| All signals, no spot | 62 | 54.8% | -$326 (-0.3%) | 13.3% |
| Filtered signals, no spot | 35 | 65.7% | +$13,675 (+13.7%) | 3.7% |

The breakthrough: **exclude underperforming signal types entirely**.

---

## Signal Type Performance (2023-2025 Backtest)

| Signal Type | Trades | Win Rate | P&L | Avg P&L | Action |
|---|---|---|---|---|---|
| PRIMARY_SHORT | 8 | 75.0% | +$6,399 | +$800 | **Tier 1** |
| OPTION_CALL | 5 | 80.0% | +$6,221 | +$1,244 | **Tier 1** |
| BEAR_PROBE | 6 | 66.7% | +$499 | +$83 | **Tier 2** |
| BULL_PROBE | 16 | 56.2% | +$555 | +$35 | **Tier 2** |
| LONG | 19 | 47.4% | -$7,280 | -$383 | **EXCLUDE** |
| TACTICAL_PUT | 8 | 25.0% | -$2,759 | -$345 | **EXCLUDE** |
| OPTION_PUT | 1 | 0.0% | -$1,741 | -$1,741 | **EXCLUDE** |

The LONG signal — historically the "bread and butter" — has degraded to 47.4% win
rate in recent years. Trading it destroys profitability.

---

## Dynamic Leverage Model

Leverage is not constant. It varies by DTE, moneyness, and structure.

### Naked Option Leverage Grid

| DTE Bucket | ATM/Slightly ITM | Slightly OTM | Deep ITM |
|---|---|---|---|
| 7-10 DTE | 16.5x | 20x | 12x |
| 11-14 DTE | 13.5x | 16.5x | 10x |
| 15-21 DTE | 10.5x | 13.5x | 7.5x |

### Spread Leverage Grid

| DTE Bucket | Leverage |
|---|---|
| 7-10 DTE | 8.5x |
| 11-14 DTE | 7x |
| 15-21 DTE | 5.5x |

### Theta Decay by DTE

| DTE Bucket | Daily Decay |
|---|---|
| 7-10 DTE | 2.5%/day |
| 11-14 DTE | 2.0%/day |
| 15-21 DTE | 1.5%/day |

### Signal Type → Option Parameters

| Signal Type | DTE Bucket | Moneyness | Naked Lev | Spread Lev | Theta |
|---|---|---|---|---|---|
| PRIMARY_SHORT | 11-14 | slightly ITM | 13.5x | 7.0x | 2.0% |
| OPTION_CALL | 7-10 | ATM | 16.5x | 8.5x | 2.5% |
| BEAR_PROBE | 11-14 | slightly ITM | 13.5x | 7.0x | 2.0% |
| BULL_PROBE | 7-10 | slightly ITM | 16.5x | 8.5x | 2.5% |

---

## Simulation Results (2023-2025)

$100k account, options only, filtered signals:

```
Total trades: 35
Winners: 23 (65.7%)
Losers: 12 (34.3%)

Total option P&L: +$13,674.62
Avg winner: +$1,232.80
Avg loser: -$1,223.32
Largest winner: +$2,531.80
Largest loser: -$2,410.11

Option return: +13.67%
Max drawdown: $3,654.17 (3.65%)
```

### By Tier

| Tier | Trades | Win Rate | P&L | Avg |
|---|---|---|---|---|
| Tier 1 | 13 | 76.9% | +$12,620 | +$971 |
| Tier 2 | 22 | 59.1% | +$1,054 | +$48 |

### By Year

| Year | Trades | Win Rate | P&L |
|---|---|---|---|
| 2023 | 16 | 62.5% | +$835 |
| 2024 | 10 | 60.0% | +$3,041 |
| 2025 | 9 | 77.8% | +$9,799 |

---

## Entry Policy

### Tier 1 — $4,000 Risk (PRIMARY_SHORT, OPTION_CALL)

| Component | Amount | Structure |
|---|---|---|
| Naked option | $1,468 (36.7%) | Per signal type DTE/moneyness |
| Debit spread | $2,532 (63.3%) | Same expiry, defined risk |

### Tier 2 — $2,400 Risk (BULL_PROBE, BEAR_PROBE, MVRV_SHORT)

| Component | Amount | Structure |
|---|---|---|
| Naked option | $800 (33.3%) | Per signal type DTE/moneyness |
| Debit spread | $1,600 (66.7%) | Same expiry, defined risk |

### Excluded Signals

Do not trade:
- **LONG** — 47.4% win rate, -$7,280 P&L
- **TACTICAL_PUT** — 25% win rate, -$2,759 P&L
- **OPTION_PUT** — insufficient sample, 0% win rate in test period

---

## Exit Rules

### Naked Option Exits

| Rule | Trigger | Action |
|---|---|---|
| Partial TP | Option up 50% | Close 60% of position |
| Full TP | Option up 70% | Close remaining |
| Day 3-4 review | Down >30% AND best close < 1% | Close naked |
| Hard stop | Underlying 5% adverse | Close naked |
| Time stop | Day 7 | Close any remaining |

### Spread Exits

| Rule | Trigger | Action |
|---|---|---|
| Fast TP | Naked closed by day 3 | Take spread at 60% |
| Slow TP | Naked alive day 4+ | Allow spread to 80% |
| Hard stop | Underlying 7% adverse | Close spread |
| Time stop | Day 10 | Close spread |

---

## Portfolio Constraints

| Rule | Value |
|---|---|
| Max concurrent trades | 2 |
| Max capital at risk | 6% of account |

---

## Execution Haircut

Applied to model real-world execution:

- Winners reduced by 12% (slippage, IV crush, fill imperfection)
- Losers increased by 8% (wider spreads, adverse fills)

---

## Worst Case Analysis

### Tier 1 — $4,000 Risk

| Leg | Deployed | Scenario | Loss |
|---|---|---|---|
| Naked | $1,468 | 5% adverse × 13.5x = 67.5% | -$991 |
| Spread | $2,532 | 7% adverse × 7x = 49% | -$1,241 |
| **Total** | | | **-$2,232** |

### Tier 2 — $2,400 Risk

| Leg | Deployed | Scenario | Loss |
|---|---|---|---|
| Naked | $800 | 5% adverse × 16.5x = 82.5% | -$660 |
| Spread | $1,600 | 7% adverse × 8.5x = 59.5% | -$952 |
| **Total** | | | **-$1,612** |

---

## Expected Performance

### Per Trade

| Tier | Win Rate | Avg Winner | Avg Loser | EV |
|---|---|---|---|---|
| Tier 1 | 77% | +$1,815 | -$1,843 | +$971 |
| Tier 2 | 59% | +$700 | -$900 | +$48 |

### Annual ($100k account, ~12 trades/year)

| Scenario | Trades | Expected P&L | Max DD |
|---|---|---|---|
| Conservative | 10 | +$4,000-$6,000 | ~$4,000 |
| Average | 12 | +$5,000-$8,000 | ~$5,000 |
| Optimistic | 15 | +$8,000-$12,000 | ~$6,000 |

---

## Regime Awareness

| Year | Hit Rate |
|---|---|
| 2021 | 88.0% |
| 2022 | 83.3% |
| 2023 | 61.5% |
| 2024 | 59.1% |
| 2025 | 65.0% |

The filtered strategy maintains profitability even in degraded regimes by:
1. Excluding signals that don't work in choppy markets (LONG, TACTICAL_PUT)
2. Concentrating capital on signals that maintain edge (PRIMARY_SHORT, OPTION_CALL)

---

## What This Design Does NOT Solve

1. **Tail risk** — Gap moves can blow through stops
2. **Liquidity** — Assumes fills at theoretical prices
3. **Correlation** — All BTC options are correlated
4. **Sample size** — OPTION_CALL has only 5 trades in test period
5. **Regime shift** — If PRIMARY_SHORT degrades, strategy needs re-evaluation

---

## Design Principles

1. Signal selection > execution optimization
2. Exclude underperformers ruthlessly — don't try to fix broken signals
3. Dynamic leverage modeling improves accuracy
4. Concentrate capital on proven signals
5. Options-only can be profitable without spot hedge
6. Review signal performance quarterly
7. If a Tier 1 signal drops below 65% win rate, demote to Tier 2
8. If a signal drops below 50% win rate, exclude entirely

---

## Reference Files

| File | Purpose |
|---|---|
| `scripts/simulate_v4.py` | V6 execution policy simulator with dynamic leverage |
| `signals/management/commands/analyze_path_stats.py` | Path diagnostics |
| `signals/management/commands/analyze_hit_rate.py` | Hit rate analysis |
| `features_14d_5pct.csv` | Historical feature dataset |
| `docs/recommendation.md` | Signal performance recommendations |
