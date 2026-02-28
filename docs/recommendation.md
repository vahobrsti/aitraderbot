# Signal Performance & Trading Recommendations

Based on backtest hit rates 2017–2025 and current production logic (`services.py` v2025-12-28).

---

## Trade Types & Priority

The system evaluates trades in this order — first match wins:

| Priority | Type | Direction | Sizing | Source | Strategy |
|----------|------|-----------|--------|--------|----------|
| 1 | CALL | 🟢 Long | overlay × base | Fusion (STRONG_BULLISH, EARLY_RECOVERY, MOMENTUM) | Long call / call spread |
| 1 | PUT | 🔴 Short | overlay × base | Fusion (DISTRIBUTION_RISK, BEAR_CONTINUATION, BEAR_PROBE) | Put spread / put diagonal |
| 1 | CALL | 🟢 Long | overlay × base | Fusion (BULL_PROBE) | Call spread (defined risk) |
| 2 | TACTICAL_PUT | 🔴 Put | 0.40–0.60x | Tactical | Hedge inside bull regime (fusion = NO_TRADE only) |
| 3 | OPTION_CALL | 🟢 Long | 0.75x | Rule | MVRV cheap (2+ flags) + Sentiment fear |
| 3 | OPTION_PUT | 🔴 Short | 0.75x | Rule | MVRV hot + Sentiment greed + Whale distribution |
| 4 | NO_TRADE | — | 0 | — | Stay flat |

> **Key rule**: Fusion state always beats tactical and option signals. OPTION_CALL takes priority over OPTION_PUT when both fire.

---

## Signal Tiers (Hit Rate)

### Tier 1: High Confidence (75%+)

| Signal | Hit Rate | Action |
|--------|----------|--------|
| OPTION_CALL | **81%** | Size up (0.75x) — best standalone signal |
| EARLY_RECOVERY | 83% | Full size |
| BEAR_CONTINUATION | 79% | Full size |

### Tier 2: Good Edge (60–72%)

| Signal | Hit Rate | Action |
|--------|----------|--------|
| Core LONG (STRONG_BULLISH, MOMENTUM) | **72.6%** | Full size |
| PRIMARY_SHORT (DISTRIBUTION_RISK) | 62.5% | Full size |
| BULL_PROBE | 60.8% | 0.35–0.60x (defined risk) |

### Tier 3: Marginal

| Signal | Hit Rate | Action |
|--------|----------|--------|
| BEAR_PROBE | 54.8% | 0.35–0.60x, requires stricter overlay |
| OPTION_PUT | 44% | EFB veto filters worst setups |
| TACTICAL_PUT | ~40% | Hedge only, never override fusion |

---

## Overlay Sizing Reference

### Long Overlays (Sentiment + MVRV Composite)

| Overlay | Trigger | Size | DTE |
|---------|---------|------|-----|
| Full Edge (+2) | Fear stabilizing + MVRV undervalued | 1.25x | ×1.50 |
| Partial Edge (+1) | Sentiment OR MVRV favorable | 1.10x | ×1.00 |
| Moderate Veto (−1) | Euphoria persisting | 0.50x | ×0.85 |
| Strong Veto (−2) | Euphoria + MVRV overvalued rollover | **0.0x** (no trade) | — |

### Short Overlays (MVRV-60d Near-Peak Score)

| Score | Overlay | Size | DTE |
|-------|---------|------|-----|
| ≥ 0.85 | Full Edge | 1.15x | ×1.15 |
| ≥ 0.75 | Partial Edge | 1.05x | ×1.05 |
| ≤ 0.35 (DIST_RISK) | Soft Veto | 0.50x | ×0.85 |
| ≤ 0.25 | Hard Veto | **0.0x** | — |

**BEAR_PROBE stricter thresholds**: soft veto at ≤ 0.50, hard veto at ≤ 0.40.

**Absolute guard**: MVRV-60d < 1.0 → hard veto all shorts (holders underwater).

### EFB Veto (OPTION_PUT only)

`distribution_pressure_score < 0.40` → soft veto. BTC leaving exchanges = supply tightening = shorts unreliable.

---

## Stricter Short Logic

Short setups historically suffer from false positives during choppy structures. The new hierarchy applies a strict `mdia_inflow` gate—shorts cannot fire if the market is experiencing active near-term capital inflow.

Key rules:
- **BEAR_CONTINUATION**: Requires definitive whale distribution + MVRV macro bearishness.
- **DISTRIBUTION_RISK**: Fires when whales and MVRV align bearishly, even if trailing momentum hasn't fully collapsed. Very strict.
- **BEAR_PROBE**: Weakest short state. Triggers on strong distribution alone, but is heavily vetted by overlays and size penalized.

---

## Option Signal Rules

| Signal | Feature | Conditions |
|--------|---------|------------|
| OPTION_CALL | `signal_option_call` | MVRV cheap (2+ of: undervalued_90d, new_low_180d, near_bottom) **AND** sentiment_norm < −1.0 |
| OPTION_PUT | `signal_option_put` | MVRV 60d near-peak (pct_rank ≥ 0.80 OR dist_from_max ≤ 0.20) **AND** sentiment_norm > 1.0 **AND** whale distributing |

- **5-day cooldown** between same-type option signals
- **0.75x sizing** (up from 0.50x — OPTION_CALL hits 81%)
- Subject to overlay veto + EFB veto (OPTION_PUT only)
- Only promoted when fusion = NO_TRADE

---

## DTE Quick Reference

| State | DTE Range | Optimal |
|-------|-----------|---------|
| STRONG_BULLISH | 7–14d | 11d |
| EARLY_RECOVERY | 14–30d | 21d |
| MOMENTUM | 7–14d | 11d |
| DISTRIBUTION_RISK | 7–14d | 12d |
| BEAR_CONTINUATION | 7–14d | 12d |
| BULL_PROBE | 7–12d | 9d |
| BEAR_PROBE | 7–12d | 9d |

> Overlays adjust DTE via multiplier (0.75–1.50).

---

## Stop Loss Strategy

Data-driven exit system calibrated from `analyze_path_stats` (7 states × 6 invalidation levels, 14d horizon, 5% target). Three-layer exit:

### Per-State Parameters

| State | Stop | Scale-Down Day | Hard Cut | Spread Width | Stop/Width |
|-------|------|----------------|----------|-------------|------------|
| STRONG_BULLISH | 4.0% | Day 5 | Day 6 | 9% | 44% |
| EARLY_RECOVERY | 4.0% | Day 6 | Day 8 | 11% | 36% |
| MOMENTUM | 4.0% | Day 5 | Day 6 | 9% | 44% |
| DISTRIBUTION_RISK | 4.0% | Day 5 | Day 6 | 9% | 44% |
| BEAR_CONTINUATION | 3.5% | Day 4 | Day 6 | 10% | 35% |
| BULL_PROBE | 3.5% | Day 4 | Day 5 | 7% | 50% |
| BEAR_PROBE | 4.0% | Day 6 | Day 7 | 7% | 57% |

### Exit Timeline

1. **Fixed price stop**: If underlying moves stop_loss_pct against you → close all
2. **Scale-down**: At scale_down_day (≈ p75 TTH) → reduce to 25% position
3. **Hard time stop**: At max_hold_days → close everything remaining

### Key Data Points

- **Sweet spot 3.5–4.0%**: Below 3% → 40%+ false stops. Above 5% → starts missing losers
- **~20% of winners hit after day 7** → keeping 25% position matches conditional probability
- **Winner MAE 2–4% vs Loser MAE 9–22%** → stop sits cleanly between winner noise and loser trajectory
- On a spread, 4% underlying stop ≈ 35–50% of premium lost (not 100%), because time value remains
- Verified stable across 2017–2025 (early crypto 2017–2018 was noisier but still within range)

## Cooldown Settings

| Trade Type | Cooldown |
|------------|----------|
| LONG / PRIMARY_SHORT | 7 days |
| TACTICAL_PUT | 7 days |
| BULL_PROBE / BEAR_PROBE | 5 days |
| OPTION_CALL / OPTION_PUT | 5 days |

---

---

## Statistical Models (Time-to-Hit)

Based on 114 successful trades (hits) from 2017–2025:

| Metric | Value | Insight |
|--------|-------|---------|
| **Median Time** | **4.0 days** | Winners validate quickly. |
| **Mean Time** | 5.2 days | Skewed by a few slower trades (max 14d). |
| **Fastest Hits** | OPTION_PUT (2.0d) | Downside moves are sharper. |
| **Slowest Hits** | LONG (5.0d) | Organic growth takes slightly longer. |

**Distribution of Days-to-Hit:**
- **1–3 days**: 40% (Quick profit taking)
- **4–7 days**: 35% (Standard swing)
- **8–14 days**: 25% (Grinding moves)

> **Strategic Implication**: If a trade hasn't hit its target by **Day 7**, the probability of a "clean win" drops significantly. Consider tightening stops or taking partial profits if available.

---

## Actionable Recommendations

1. **Trust EARLY_RECOVERY** — 83% hit rate. Your best fusion signal. Size up.

2. **OPTION_CALL is your best fallback** — 81% hit rate at 0.75x sizing. When fusion is NO_TRADE and MVRV is cheap + fear, this fires.

3. **Fusion beats everything** — Never override a fusion directional view with tactical or option signals. The system enforces this.

4. **Short signals are solid** — BEAR_CONTINUATION (79%) and DISTRIBUTION_RISK (62.5%) both work organically via the strict empirical ruleset.

5. **Watch BEAR_PROBE overlays** — Stricter thresholds (0.50/0.40 vs standard 0.35/0.25) filter best here.

6. **OPTION_PUT needs EFB confirmation** — 44% raw HR, but EFB veto filters the worst setups (69% veto accuracy).

7. **TACTICAL_PUT is insurance** — ~40% hit rate. Only fires when fusion = NO_TRADE. Don't size up.

---

## Diagnostics

```bash
# Quick fusion check (via helper script on VPS)
bash scripts/analyze_fusion.sh
bash scripts/analyze_fusion.sh --date 2026-02-17

# Full detail
python manage.py analyze_fusion --explain --date 2026-02-17

# Backtest hit rates
python manage.py analyze_hit_rate --year 2025

# Score a CSV dataset
python manage.py score_dataset



# Diagnose NO_TRADE days
python manage.py diagnose_notrade --year 2025
```
