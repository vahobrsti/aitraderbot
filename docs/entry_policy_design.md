# Entry Policy Design from Path Analysis

**Data Source:** `analyze_path_stats` (14d horizon, 5% target, 4% invalidation)  
**Total Trades:** 344 | **Overall Win Rate:** 68.9%  
**Generated:** 2026-05-03 (Updated with corrected IRON_CONDOR)

---

## Executive Summary

Path analysis reveals distinct profiles per signal type. The key insight: **TTH (time-to-hit) drives DTE**, **winner MAE drives stop loss**, and **MFE drives spread width**. 

| Signal | n | Win% | TTH med | TTH p75 | MAE(W) p75 | MFE p75 | Clean Win% |
|--------|---|------|---------|---------|------------|---------|------------|
| CALL (LONG) | 85 | 69.4% | 4d | 7d | 4.71% | 15.03% | 72.9% |
| PUT (PRIMARY_SHORT) | 37 | 75.7% | 4d | 6d | 4.41% | 14.28% | 71.4% |
| BULL_PROBE | 45 | 71.1% | 3.5d | 5d | 3.84% | 28.24% | 78.1% |
| BEAR_PROBE | 17 | 58.8% | 4d | 8.5d | 6.53% | 12.13% | 60.0% |
| TACTICAL_PUT | 25 | 48.0% | 4d | 5.5d | 3.08% | 10.40% | 75.0% |
| OPTION_CALL | 26 | 92.3% | 3d | 5d | 8.48% | 15.26% | 45.8% |
| OPTION_PUT | 20 | 80.0% | 2d | 3d | 6.82% | 23.08% | 56.2% |
| MVRV_SHORT | 21 | 66.7% | 4d | 10d | 7.19% | 13.90% | 57.1% |
| IRON_CONDOR | 84 | 63.1% | N/A | N/A | 6.76% | N/A | N/A |

---

## Signal-by-Signal Entry Policy

### 1. CALL (Core Long)
**Profile:** Reliable workhorse with moderate speed  
**n=85, Win Rate=69.4%**

| Metric | Value | Implication |
|--------|-------|-------------|
| TTH median | 4d | Most winners resolve within a week |
| TTH p75 | 7d | 75% of winners hit by day 7 |
| MAE(W) p75 | 4.71% | Winners survive up to 4.7% drawdown |
| MFE p75 | 15.03% | Strong overshoot potential |
| Clean winners | 72.9% | Low invalidation rate |

**Entry Policy:**
```
DTE: min=9, max=14, optimal=11
  └─ Rationale: TTH p75 (7d) + 2d buffer = 9d minimum
Spread Width: 10%
  └─ Rationale: MFE p75 (15%) × 0.67 = 10% (capture 2/3 of move)
Stop Loss: 4.5%
  └─ Rationale: MAE(W) p75 (4.71%) rounded
Delta Target: 0.55-0.65 (slight ITM)
  └─ Rationale: Survive 4.7% shakeout, still capture move
```

---

### 2. PUT (Core Short)
**Profile:** Highest conviction short signal  
**n=37, Win Rate=75.7%**

| Metric | Value | Implication |
|--------|-------|-------------|
| TTH median | 4d | Fast resolution |
| TTH p75 | 6d | Faster than CALL |
| MAE(W) p75 | 4.41% | Tight drawdowns on winners |
| MFE p75 | 14.28% | Strong downside capture |
| Clean winners | 71.4% | Solid clean rate |

**Entry Policy:**
```
DTE: min=8, max=12, optimal=10
  └─ Rationale: TTH p75 (6d) + 2d buffer = 8d minimum
Spread Width: 9%
  └─ Rationale: MFE p75 (14.28%) × 0.65 = 9.3%
Stop Loss: 4.5%
  └─ Rationale: MAE(W) p75 (4.41%) + small buffer
Delta Target: -0.55 to -0.65 (slight ITM)
  └─ Rationale: Capture fast drops
```

---

### 3. BULL_PROBE (Exploratory Long)
**Profile:** Fast-moving, high MFE potential  
**n=45, Win Rate=71.1%**

| Metric | Value | Implication |
|--------|-------|-------------|
| TTH median | 3.5d | Fastest long signal |
| TTH p75 | 5d | Quick resolution |
| MAE(W) p75 | 3.84% | Tight stops work |
| MFE p75 | 28.24% | **Highest MFE of all signals** |
| Clean winners | 78.1% | Very clean paths |

**Entry Policy:**
```
DTE: min=7, max=11, optimal=9
  └─ Rationale: TTH p75 (5d) + 2d buffer = 7d minimum
Spread Width: 12%
  └─ Rationale: MFE p75 (28%) is huge, but cap at 12% for risk
Stop Loss: 4.0%
  └─ Rationale: MAE(W) p75 (3.84%) rounded up
Delta Target: 0.50-0.60 (ATM to slight ITM)
  └─ Rationale: Fast moves favor ATM
Position Size: 50% of core
  └─ Rationale: Exploratory, smaller size
```

---

### 4. BEAR_PROBE (Exploratory Short)
**Profile:** Messy, high invalidation, needs wider stops  
**n=17, Win Rate=58.8%**

| Metric | Value | Implication |
|--------|-------|-------------|
| TTH median | 4d | Moderate speed |
| TTH p75 | 8.5d | Slow tail |
| MAE(W) p75 | 6.53% | **Widest MAE of probes** |
| MFE p75 | 12.13% | Moderate upside |
| Clean winners | 60.0% | **40% invalidated before hit** |

**Entry Policy:**
```
DTE: min=11, max=16, optimal=13
  └─ Rationale: TTH p75 (8.5d) + 2.5d buffer = 11d minimum
Spread Width: 8%
  └─ Rationale: MFE p75 (12%) × 0.67 = 8%
Stop Loss: 6.5%
  └─ Rationale: MAE(W) p75 (6.53%) - need wider stop
Delta Target: -0.50 to -0.60 (ATM to slight ITM)
  └─ Rationale: Wider stop needs more delta cushion
Position Size: 40% of core
  └─ Rationale: Low win rate, high invalidation
```

**⚠️ Warning:** 40% of winners get invalidated before hitting target. Consider:
- Wider initial stop (6.5% vs 4%)
- Smaller position size
- ITM strikes for more cushion

---

### 5. TACTICAL_PUT (Hedge Inside Bull)
**Profile:** Low win rate but tight MAE on winners  
**n=25, Win Rate=48.0%**

| Metric | Value | Implication |
|--------|-------|-------------|
| TTH median | 4d | Moderate |
| TTH p75 | 5.5d | Relatively fast |
| MAE(W) p75 | 3.08% | **Tightest MAE of all** |
| MFE p75 | 10.40% | Moderate |
| Clean winners | 75.0% | Clean when it works |

**Entry Policy:**
```
DTE: min=8, max=12, optimal=10
  └─ Rationale: TTH p75 (5.5d) + 2.5d buffer = 8d minimum
Spread Width: 7%
  └─ Rationale: MFE p75 (10.4%) × 0.67 = 7%
Stop Loss: 3.5%
  └─ Rationale: MAE(W) p75 (3.08%) + small buffer
Delta Target: -0.35 to -0.45 (slight OTM)
  └─ Rationale: Hedge, not directional bet
Position Size: 30% of core
  └─ Rationale: Low win rate (48%), hedge purpose
```

**Note:** 48% win rate is acceptable for a hedge. The tight MAE (3.08%) means winners are clean. Losers get stopped out quickly.

---

### 6. OPTION_CALL (Rule-Based Long)
**Profile:** Highest win rate, but messy paths  
**n=26, Win Rate=92.3%**

| Metric | Value | Implication |
|--------|-------|-------------|
| TTH median | 3d | Fast |
| TTH p75 | 5d | Quick resolution |
| MAE(W) p75 | 8.48% | **Wide drawdowns** |
| MFE p75 | 15.26% | Strong upside |
| Clean winners | 45.8% | **54% invalidated or ambiguous** |

**Entry Policy:**
```
DTE: min=7, max=12, optimal=9
  └─ Rationale: TTH p75 (5d) + 2d buffer = 7d minimum
Spread Width: 10%
  └─ Rationale: MFE p75 (15.26%) × 0.67 = 10%
Stop Loss: 8.5%
  └─ Rationale: MAE(W) p75 (8.48%) - MUST be wide
Delta Target: 0.60-0.70 (ITM)
  └─ Rationale: Wide MAE needs ITM cushion
Position Size: 75% of core
  └─ Rationale: 92% win rate justifies size
```

**⚠️ Critical:** 54% of winners get invalidated before hitting target. This signal has the highest win rate but the messiest paths. **Use ITM strikes and wide stops.**

---

### 7. OPTION_PUT (Rule-Based Short)
**Profile:** Fast, high win rate, volatile paths  
**n=20, Win Rate=80.0%**

| Metric | Value | Implication |
|--------|-------|-------------|
| TTH median | 2d | **Fastest signal** |
| TTH p75 | 3d | Very quick |
| MAE(W) p75 | 6.82% | Wide drawdowns |
| MFE p75 | 23.08% | **Second highest MFE** |
| Clean winners | 56.2% | 44% messy |

**Entry Policy:**
```
DTE: min=5, max=10, optimal=7
  └─ Rationale: TTH p75 (3d) + 2d buffer = 5d minimum
Spread Width: 12%
  └─ Rationale: MFE p75 (23%) × 0.52 = 12% (cap for risk)
Stop Loss: 7.0%
  └─ Rationale: MAE(W) p75 (6.82%) rounded up
Delta Target: -0.60 to -0.70 (ITM)
  └─ Rationale: Fast moves + wide MAE = need ITM
Position Size: 75% of core
  └─ Rationale: 80% win rate
```

**Note:** Fastest signal (median TTH 2d). Can use shorter DTE but need ITM strikes to survive the 6.8% drawdowns.

---

### 8. MVRV_SHORT (Bear Market Tactical)
**Profile:** Slow, shakeout-heavy, needs patience  
**n=21, Win Rate=66.7%**

| Metric | Value | Implication |
|--------|-------|-------------|
| TTH median | 4d | Moderate |
| TTH p75 | 10d | **Slowest signal** |
| MAE(W) p75 | 7.19% | Wide drawdowns |
| MFE p75 | 13.90% | Moderate |
| Clean winners | 57.1% | 43% invalidated |
| Path shape | 57% shakeout | **Most shakeout-heavy** |

**Entry Policy:**
```
DTE: min=12, max=18, optimal=14
  └─ Rationale: TTH p75 (10d) + 2d buffer = 12d minimum
Spread Width: 9%
  └─ Rationale: MFE p75 (13.9%) × 0.65 = 9%
Stop Loss: 7.0%
  └─ Rationale: MAE(W) p75 (7.19%) - need wide stop
Delta Target: -0.55 to -0.65 (slight ITM)
  └─ Rationale: Shakeout paths need cushion
Position Size: 33% initial, 67% DCA at +4%
  └─ Rationale: 57% shakeout rate = DCA strategy
```

**⚠️ Key Insight:** 57% of winners have "shakeout then expansion" path shape. This is the highest of any signal. **DCA strategy is correct** - initial entry gets shaken out, DCA catches the move.

---

### 9. IRON_CONDOR (Range-Bound)
**Profile:** Range-bound premium selling, 7-day horizon, ±10% range  
**n=84, Win Rate=63.1%**

| Metric | Value | Implication |
|--------|-------|-------------|
| Horizon | 7d | Fixed 7-day hold period |
| Range | ±10% | Price must stay within ±10% of entry |
| MAE(W) p75 | 6.76% | Max move on winners |
| MAE(L) avg | 18.56% | Losers breach significantly |

**Entry Policy:**
```
DTE: min=9, max=13, optimal=9
  └─ Rationale: 7d horizon + 2d buffer = 9d minimum
Wing Width: 10%
  └─ Rationale: Standard condor width matching range threshold
Stop Loss: 6.8% underlying move
  └─ Rationale: MAE(W) p75 (6.76%) - exit if approaching range boundary
Take Profit: 50% of max credit
  └─ Rationale: Take profit early, don't hold to expiry
Position Size: 50% of core
  └─ Rationale: Premium selling strategy
```

**Note:** IRON_CONDOR uses a different hit definition than directional trades:
- **Hit** = price stays within ±10% of entry over 7 days
- **Miss** = price breaches ±10% boundary at any point
- No TTH/MFE metrics (range-bound, not directional)

---

## Calibrated Policy Summary

```python
CALIBRATED_POLICY = {
    "CALL": {
        "dte": {"min": 9, "max": 14, "optimal": 11},
        "spread_width_pct": 0.10,
        "stop_loss_pct": 0.045,
        "take_profit_pct": 0.70,
        "max_hold_days": 9,
        "scale_down_day": 6,
        "delta_target": 0.60,
        "expected_edge": 0.069,  # 69.4% × 10%
    },
    "PUT": {
        "dte": {"min": 8, "max": 12, "optimal": 10},
        "spread_width_pct": 0.09,
        "stop_loss_pct": 0.045,
        "take_profit_pct": 0.70,
        "max_hold_days": 8,
        "scale_down_day": 5,
        "delta_target": -0.60,
        "expected_edge": 0.068,  # 75.7% × 9%
    },
    "BULL_PROBE": {
        "dte": {"min": 7, "max": 11, "optimal": 9},
        "spread_width_pct": 0.12,
        "stop_loss_pct": 0.040,
        "take_profit_pct": 0.70,
        "max_hold_days": 7,
        "scale_down_day": 5,
        "delta_target": 0.55,
        "expected_edge": 0.085,  # 71.1% × 12%
    },
    "BEAR_PROBE": {
        "dte": {"min": 11, "max": 16, "optimal": 13},
        "spread_width_pct": 0.08,
        "stop_loss_pct": 0.065,
        "take_profit_pct": 0.70,
        "max_hold_days": 11,
        "scale_down_day": 7,
        "delta_target": -0.55,
        "expected_edge": 0.047,  # 58.8% × 8%
    },
    "TACTICAL_PUT": {
        "dte": {"min": 8, "max": 12, "optimal": 10},
        "spread_width_pct": 0.07,
        "stop_loss_pct": 0.035,
        "take_profit_pct": 0.70,
        "max_hold_days": 8,
        "scale_down_day": 5,
        "delta_target": -0.40,
        "expected_edge": 0.034,  # 48% × 7%
    },
    "OPTION_CALL": {
        "dte": {"min": 7, "max": 12, "optimal": 9},
        "spread_width_pct": 0.10,
        "stop_loss_pct": 0.085,
        "take_profit_pct": 0.70,
        "max_hold_days": 7,
        "scale_down_day": 4,
        "delta_target": 0.65,
        "expected_edge": 0.092,  # 92.3% × 10%
    },
    "OPTION_PUT": {
        "dte": {"min": 5, "max": 10, "optimal": 7},
        "spread_width_pct": 0.12,
        "stop_loss_pct": 0.070,
        "take_profit_pct": 0.70,
        "max_hold_days": 5,
        "scale_down_day": 3,
        "delta_target": -0.65,
        "expected_edge": 0.096,  # 80% × 12%
    },
    "MVRV_SHORT": {
        "dte": {"min": 12, "max": 18, "optimal": 14},
        "spread_width_pct": 0.09,
        "stop_loss_pct": 0.070,
        "take_profit_pct": 0.70,
        "max_hold_days": 12,
        "scale_down_day": 7,
        "delta_target": -0.60,
        "expected_edge": 0.060,  # 66.7% × 9%
    },
    "IRON_CONDOR": {
        "dte": {"min": 9, "max": 13, "optimal": 9},
        "spread_width_pct": 0.10,
        "stop_loss_pct": 0.068,
        "take_profit_pct": 0.50,
        "max_hold_days": 9,
        "scale_down_day": 8,
        "delta_target": 0.20,  # OTM wings
        "expected_edge": 0.063,  # 63.1% × 10%
    },
}
```

---

## Key Insights

### 1. DTE Formula
```
min_dte = TTH_p75 + 2 days buffer
max_dte = min_dte + 4-6 days
optimal_dte = TTH_median + 2-3 days
```

### 2. Stop Loss Formula
```
stop_loss = MAE(winners)_p75 + 0.5% buffer
```
Exception: OPTION_CALL and OPTION_PUT need wider stops (use MAE_p75 directly).

### 3. Spread Width Formula
```
spread_width = MFE_p75 × 0.65-0.70
```
Cap at 12% for risk management.

### 4. Expected Edge Formula
```
expected_edge = win_rate × spread_width
```
Must exceed execution costs (~1.5% for 2-leg spread).

### 5. Path Shape Implications
- **Clean expansion (A):** Use ATM strikes, tight stops
- **Shakeout then expansion (B):** Use ITM strikes, wider stops, consider DCA
- **Slow grind (C):** Use longer DTE, patient exits
- **Overshoot then mean reversion (D):** Take profit at 70%, don't hold for max

### 6. Invalidation Rate Implications
- **< 25% invalidation:** Standard stops work
- **25-40% invalidation:** Widen stops by 1-2%
- **> 40% invalidation:** Use ITM strikes, DCA strategy, or reduce size

---

## Implementation Checklist

- [ ] Update `policy.py` with calibrated DTE targets
- [ ] Update `policy.py` with calibrated spread widths
- [ ] Update `policy.py` with calibrated stop losses
- [ ] Update `policy.py` with calibrated expected edge
- [ ] Add delta targets to policy (new field)
- [ ] Update `deribit_entry.py` to use delta targets from policy
- [ ] Run `calibrate_policy` to generate JSON overlay
- [ ] Verify executor uses calibrated values
