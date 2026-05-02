# Options Execution Design — V7 Path-Stable

Options-only strategy prioritizing path stability over raw returns. V7 addresses the
core mismatch between theoretical backtests and real-world execution: leverage spikes
near ATM cause -40% drawdowns that are psychologically unsurvivable.

**V7 Philosophy:** Your edge already exists. Optimize for path stability, not return maximization.

---

## V6 → V7 Changes Summary

| Parameter | V6 | V7 | Rationale |
|---|---|---|---|
| Naked exposure | 33-37% | 20-25% | Reduce gamma/path damage |
| Spread width | ~6% | 8-12% | Lower effective leverage |
| Primary DTE | 7-14 | 14-21 | More time, less theta burn |
| Leverage model | Optimistic (13-17x) | Realistic (6-10x base, 12-18x stress) | Match real-world observations |
| Strike selection | ATM/slightly ITM | Deeper ITM | Avoid gamma spikes |

---

## Analysis Commands

```bash
# Path diagnostics
python manage.py analyze_path_stats
python manage.py analyze_path_stats --year 2025
python manage.py analyze_path_stats --direction LONG

# Hit rate analysis
python manage.py analyze_hit_rate
python manage.py analyze_hit_rate --year 2024

# V7 execution policy simulation
python manage.py simulate                    # Base case
python manage.py simulate --stress           # Stress test (upper-bound leverage)
python manage.py simulate --years 2023 2024 2025
```

---

## The Problem

The system has edge (65-72% hit rate), but options execution is brutal:

- 65% of eventual winners dip at least 1% against you before hitting target
- 39% dip 3%+ before hitting
- V6 assumed leverage was controlled, but real data shows leverage is state-dependent
  and spikes to 15-20x near ATM strikes

**Result:** -40% drawdowns, emotional stress, premature exits.

---

## V7 Simulation Results (2021-2025)

$100k account, options only:

> Note: These figures are a historical snapshot. The current simulator excludes
> `LONG`, `TACTICAL_PUT`, `OPTION_PUT`, and `MVRV_SHORT` via `TIER_MAP` in
> `execution/management/commands/simulate.py`, so reruns on current code can
> produce different totals/trade mix.

### Base Case (midpoint leverage)

```
Total trades: 68
Winners: 35 (51.5%)
Losers: 33 (48.5%)

Total option P&L: +$3,072
Avg winner: +$834
Avg loser: -$791
Largest winner: +$2,098
Largest loser: -$1,642

Option return: +3.07%
Max drawdown: $4,830 (4.83%)
```

### Stress Test (upper-bound leverage)

```
Total trades: 68
Winners: 39 (57.4%)
Losers: 29 (42.6%)

Total option P&L: +$2,554
Avg winner: +$841
Avg loser: -$1,043
Largest winner: +$2,098
Largest loser: -$1,970

Option return: +2.55%
Max drawdown: $6,991 (6.99%)
```

### By Year

| Year | Trades | Win Rate | P&L (Base) |
|---|---|---|---|
| 2021 | 14 | 50.0% | +$1,558 |
| 2022 | 19 | 42.1% | -$2,180 |
| 2023 | 16 | 50.0% | +$580 |
| 2024 | 10 | 50.0% | +$1,179 |
| 2025 | 9 | 77.8% | +$1,934 |

### By Signal Type

| Signal Type | Trades | Win Rate | P&L | Avg | Status |
|---|---|---|---|---|---|
| BULL_PROBE | 20 | 60.0% | +$4,878 | +$244 | **Tier 2** |
| BEAR_PROBE | 7 | 57.1% | +$921 | +$132 | **Tier 2** |
| PRIMARY_SHORT | 18 | 50.0% | -$66 | -$4 | **Tier 1** |
| OPTION_CALL | 11 | 54.5% | -$1,671 | -$152 | **Tier 1** |
| MVRV_SHORT | 12 | 33.3% | -$989 | -$82 | **EXCLUDE (historical; now skipped in simulator)** |

---

## Dynamic Leverage Model (V7)

V7 uses realistic leverage ranges based on real-world observations.

### Naked Option Leverage Grid

| DTE Bucket | ATM | Slightly ITM | Slightly OTM | Deep ITM |
|---|---|---|---|---|
| 7-10 DTE | 12-18x (stress) | 10-15x | 14-20x (stress) | 6-10x |
| 11-14 DTE | 10-15x (stress) | 8-12x | 12-16x | 6-9x |
| 15-21 DTE | 7-11x | 6-10x | 9-13x | 4-7x |

**Rule:** If distance to strike < 3%, avoid naked options entirely.

### Spread Leverage Grid (Wider Spreads)

| DTE Bucket | Leverage | Width |
|---|---|---|
| 7-10 DTE | 5-8x | 8-12% |
| 11-14 DTE | 4-6x | 8-12% |
| 15-21 DTE | 3-5x | 8-12% |

### Theta Decay by DTE

| DTE Bucket | Daily Decay |
|---|---|
| 7-10 DTE | 2.5%/day |
| 11-14 DTE | 2.0%/day |
| 15-21 DTE | 1.5%/day |

### Signal Type → Option Parameters (V7)

| Signal Type | DTE Bucket | Moneyness | Naked Lev | Spread Lev | Theta |
|---|---|---|---|---|---|
| PRIMARY_SHORT | 15-21 | slightly ITM | 8.0x | 4.0x | 1.5% |
| OPTION_CALL | 11-14 | slightly ITM | 10.0x | 5.0x | 2.0% |
| BEAR_PROBE | 15-21 | slightly ITM | 8.0x | 4.0x | 1.5% |
| BULL_PROBE | 11-14 | slightly ITM | 10.0x | 5.0x | 2.0% |
| MVRV_SHORT | 15-21 | deep ITM | 5.5x | 4.0x | 1.5% |

---

## Entry Policy (V7)

### Tier 1 — $4,000 Risk (PRIMARY_SHORT, OPTION_CALL)

| Component | Amount | Structure |
|---|---|---|
| Naked option | $800 (20%) | Per signal type DTE/moneyness |
| Debit spread | $3,200 (80%) | 8-12% width, same expiry |

### Tier 2 — $2,400 Risk (BULL_PROBE, BEAR_PROBE)

| Component | Amount | Structure |
|---|---|---|
| Naked option | $600 (25%) | Per signal type DTE/moneyness |
| Debit spread | $1,800 (75%) | 8-12% width, same expiry |

### Excluded Signals

Do not trade:
- **LONG** — Degraded to 47% win rate in recent years
- **TACTICAL_PUT** — 25% win rate
- **OPTION_PUT** — Insufficient sample
- **MVRV_SHORT** — 33% win rate, negative P&L

---

## Iron Condor Execution (Premium Selling)

IRON_CONDOR is a fundamentally different trade type from directional options. It sells premium on both sides and profits from time decay in range-bound markets.

### Entry Policy

| Component | Value |
|---|---|
| Risk budget | $2,000 per condor (Tier 2 equivalent) |
| Structure | 4-leg iron condor (sell put spread + sell call spread) |
| Wing width | $2,000 per side (nearest available strikes) |
| Max concurrent condors | 1 |

### Strike Selection (MVRV Drift-Based)

Strikes are computed by `signals/options.py::compute_condor_strikes()`:

```
cost_basis = spot / mvrv_60d
drift = max(trailing_7d_mvrv) - min(trailing_7d_mvrv)

short_call = max(spot × 1.10, cost_basis × (mvrv + 1.5 × drift))
short_put  = min(spot × 0.90, cost_basis × (mvrv - 1.5 × drift))
```

Map to nearest available exchange strikes. Long wings = next strike beyond short strikes.

The signal stores computed levels in `condor_short_call`, `condor_short_put`, and `condor_strike_meta` (includes drift, sources, distances).

### Expiry Selection

| Parameter | Value | Rationale |
|---|---|---|
| Target DTE | 13–14 days | Matches 7d hold + theta buffer |
| Min DTE | 10 days | Enough time value for premium |
| Max DTE | 21 days | Beyond this, premium is too thin for OTM wings |

Prefer the nearest monthly/weekly expiry in the 10–21d window.

### Exit Rules

| Rule | Trigger | Action |
|---|---|---|
| Take profit | 50% of max credit collected | Close all 4 legs |
| Scale down | Day 5 of hold | Reduce to 25% position |
| Hard time stop | Day 7 of hold | Close all remaining |
| Stop loss | Underlying moves 6% from entry | Close all 4 legs |

### R:R Profile

Based on real option data (May 1 2026 expiry, Apr 18 entry):

| Metric | Value |
|---|---|
| Typical credit (13d DTE, $2k wings) | $300–$500 |
| Max risk | $1,500–$1,700 |
| R:R | ~1:3.9 |
| Win rate needed for breakeven | ~80% |
| Backtested win rate (drift method) | 76.3% overall, **85% in MVRV 1.00–1.04** |

### Key Differences from Directional Trades

| Aspect | Directional (CALL/PUT) | Iron Condor |
|---|---|---|
| Edge source | Underlying direction | Time decay + range |
| Leverage concern | Gamma spikes near ATM | Both wings tested simultaneously |
| Exit trigger | Underlying adverse move | Underlying moves to either wing |
| Worst case | One-sided loss | One spread goes max loss |
| Sizing | Tier 1/2 based on signal | Fixed $2k risk budget |

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

## Worst Case Analysis (V7)

### Tier 1 — $4,000 Risk

| Leg | Deployed | Scenario | Loss |
|---|---|---|---|
| Naked | $800 | 5% adverse × 10x = 50% | -$400 |
| Spread | $3,200 | 7% adverse × 5x = 35% | -$1,120 |
| **Total** | | | **-$1,520** |

### Tier 2 — $2,400 Risk

| Leg | Deployed | Scenario | Loss |
|---|---|---|---|
| Naked | $600 | 5% adverse × 10x = 50% | -$300 |
| Spread | $1,800 | 7% adverse × 5x = 35% | -$630 |
| **Total** | | | **-$930** |

**V6 vs V7 worst case:** Tier 1 loss reduced from -$2,232 to -$1,520 (32% improvement).

---

## V6 vs V7 Comparison

| Metric | V6 | V7 Base | V7 Stress |
|---|---|---|---|
| Return | +10.95% | +3.07% | +2.55% |
| Max Drawdown | 7.53% | 4.83% | 6.99% |
| Win Rate | 61.8% | 51.5% | 57.4% |
| Largest Loss | -$2,935 | -$1,642 | -$1,970 |
| Survivable? | Questionable | Yes | Yes |

**Trade-off:** V7 sacrifices ~8% return for ~35% reduction in max drawdown and ~45% reduction in largest single loss.

---

## Expected Performance (V7)

### Per Trade

| Tier | Win Rate | Avg Winner | Avg Loser | EV |
|---|---|---|---|---|
| Tier 1 | 52% | +$1,000 | -$1,400 | -$60 |
| Tier 2 | 59% | +$800 | -$700 | +$180 |

### Annual ($100k account, ~12 trades/year)

| Scenario | Trades | Expected P&L | Max DD |
|---|---|---|---|
| Conservative | 10 | +$1,000-$2,000 | ~$3,500 |
| Average | 12 | +$2,000-$4,000 | ~$5,000 |
| Optimistic | 15 | +$3,000-$5,000 | ~$6,000 |

---

## Design Principles (V7)

1. **Path stability > return maximization** — You can't capture edge you can't hold
2. **Realistic leverage modeling** — Use ranges, simulate both base and stress
3. **Reduce gamma exposure** — Wider spreads, deeper ITM, less naked
4. **Shift DTE higher** — 14-21 DTE primary, not 7-14
5. **Exclude ruthlessly** — If win rate < 50%, don't trade it
6. **Survivable drawdowns** — Target max DD < 7% even under stress
7. **Review quarterly** — Demote signals that degrade

---

## What V7 Does NOT Solve

1. **Tail risk** — Gap moves can still blow through stops
2. **Liquidity** — Assumes fills at theoretical prices
3. **Correlation** — All BTC options are correlated
4. **Lower returns** — V7 trades return for stability
5. **Regime shift** — If BULL_PROBE degrades, strategy needs re-evaluation

---

## Reference Files

| File | Purpose |
|---|---|
| `execution/management/commands/simulate.py` | V7 execution policy simulator |
| `signals/management/commands/analyze_path_stats.py` | Path diagnostics |
| `signals/management/commands/analyze_hit_rate.py` | Hit rate analysis |
| `docs/options_data_leverage_plan.md` | Current options data collection and leverage modeling plan |
| `features_14d_5pct.csv` | Historical feature dataset |
| `docs/recommendation.md` | Signal performance recommendations |
| `docs/iron_condor_spec.md` | Full iron condor specification (gate, strikes, tail risk) |
| `signals/options.py` | `compute_condor_strikes()` — MVRV drift strike logic |
