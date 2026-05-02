# On-Chain BTC Options Signal System

A Bitcoin options trading signal system that fuses on-chain metrics (MDIA, MVRV, whale behavior) with sentiment analysis to classify market regimes and generate actionable trade signals with automated execution on Bybit and Deribit.

## Overview

This system generates trading signals by fusing three orthogonal market indicators with ML predictions:

| Indicator | Role | Measures |
|-----------|------|----------|
| **MDIA** (Mean Dollar Invested Age) | Timing/Impulse | Is fresh capital entering NOW? |
| **Whales** | Intent/Sponsorship | Is smart money backing the move? |
| **MVRV-LS** (Long/Short) | Macro Confirmation | Is market structurally ready? |

Think of it as: **MDIA = ignition, Whales = fuel, MVRV-LS = terrain**

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                               │
│  datafeed/ → Raw on-chain data ingestion (Google Sheets/APIs)   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       FEATURE LAYER                              │
│  features/feature_builder.py → 70+ regime features               │
│  - MDIA slopes, buckets, regimes                                 │
│  - MVRV composite, LS, 60d percentiles                           │
│  - Whale accumulation/distribution patterns                      │
│  - Sentiment z-scores, buckets                                   │
│  - Exchange flow balance (EFB) distribution pressure             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       SIGNAL LAYER                               │
│  signals/                                                        │
│  ├── fusion.py      → Market state classification (8 states)    │
│  ├── overlays.py    → Edge/veto modifiers for execution         │
│  ├── tactical_puts.py → Hedging logic inside bull regimes       │
│  ├── options.py     → Option strategy, strikes, DTE & spreads   │
│  ├── services.py    → SignalService for scoring + persistence   │
│  └── models.py      → DailySignal model for DB storage          │
└─────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                         ML LAYER                                 │
│  ml/training.py → Model training with walk-forward validation   │
│  ml/predict.py → Inference for daily scoring                    │
│  models/ → Probabilities (future use: automated sizing/risk)    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       API LAYER                                  │
│  api/views.py → REST API endpoints with token authentication    │
│  api/serializers.py → DRF serializers for signal data           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    EXECUTION LAYER (NEW)                         │
│  execution/                                                      │
│  ├── exchanges/     → Bybit & Deribit adapters                  │
│  ├── services/      → Orchestrator + Risk management            │
│  └── models.py      → ExchangeAccount, Intent, Order, Position  │
│  Signal → Intent → Risk Check → Exchange → Audit Log            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Market Regimes & States

The fusion engine (`signals/fusion.py`) operates in two distinct modes based on the current cycle phase.

### Standard Mode (8 Canonical States)

Active outside of the bear market window.

The fusion engine (`signals/fusion.py`) classifies each day into one of 8 states using hierarchical rules:

| State | Description | Direction | Sizing |
|-------|-------------|-----------|--------|
| `STRONG_BULLISH` | All indicators aligned bullish | 🟢 Long | 1.0x |
| `EARLY_RECOVERY` | Smart money leading, structure turning | 🟢 Long | 1.0x |
| `MOMENTUM_CONTINUATION` | Trend continuing without strong sponsorship | 🟢 Long | 1.0x |
| `BULL_PROBE` | Timing + sponsorship, macro neutral | 🟢 Long | 0.35-0.60x |
| `DISTRIBUTION_RISK` | Smart money exiting, structure cracking | 🔴 Short | 1.0x |
| `BEAR_CONTINUATION` | No buyers, sellers in control | 🔴 Short | 1.0x |
| `BEAR_PROBE` | Selling + distribution, macro neutral | 🔴 Short | 0.35-0.60x |
| `NO_TRADE` | Conflicting signals, stay flat | ⚪ None | 0x |

### Bear Mode (5 Specialized States)

Active during the expected bear market window (**Days 540–900 after a halving event**). Classification anchors on valuation pain and capital velocity.

| State | Description | Direction | Sizing |
|-------|-------------|-----------|--------|
| `BEAR_EXHAUSTION_LONG` | Holder capitulation verified by capital inflow | 🟢 Long | Full Sizing |
| `BEAR_RALLY_LONG` | Underwater holders getting temporary relief | 🟢 Long | Standard Sizing |
| `BEAR_CONTINUATION_SHORT`| Profitable holders selling into weakness/aging flows | 🔴 Short | Full Sizing |
| `LATE_DISTRIBUTION_SHORT`| Breakeven/profitable holders losing conviction | 🔴 Short | Standard Sizing |
| `TRANSITION_CHOP` | Conflicting valuation and flow data, stand down | ⚪ None | 0x |

*Note: Bear mode can still return `NO_TRADE` in some edge cases (e.g., absent/unmatched MVRV data) outside of these 5 states.*

### Classification Hierarchy (Standard Mode)

```
🚀 STRONG_BULLISH
   └─ MDIA strong_inflow + Whale sponsored + MVRV macro bullish

📈 EARLY_RECOVERY  
   └─ MDIA inflow + Whale sponsored + MVRV recovery

🐻 BEAR_CONTINUATION
   └─ NOT MDIA inflow + Whale distrib + (MVRV put OR bear)

⚠️ DISTRIBUTION_RISK
   └─ NOT MDIA inflow + Whale distrib + NOT MVRV macro bullish

🔥 MOMENTUM_CONTINUATION
   └─ MDIA inflow + Whale sponsored/mixed + MVRV macro bullish

🎯 BULL_PROBE (0.5x sizing)
   └─ MDIA inflow + Whale sponsored + MVRV macro neutral

🔴 BEAR_PROBE (0.5x sizing)
   └─ NOT MDIA inflow + Whale strong_distribution + MVRV macro neutral

🟡 NO_TRADE
   └─ No alignment (fallback)

📗 OPTION_CALL (0.75x sizing, rule-based fallback)
   └─ MVRV cheap (2+ flags) + Sentiment fear — promoted when fusion=NO_TRADE

📕 OPTION_PUT (0.75x sizing, rule-based fallback)
   └─ MVRV overheated + Sentiment greed + Whale distrib — promoted when fusion=NO_TRADE
```

### Classification Hierarchy (Bear Mode: Days 540-900 Post-Halving)

During the bear market window (days 540-900 since last halving), the engine uses valuation pain (`mvrv_60d` bucket) and capital velocity (`mdia`) as primary anchors.

```
🚀 BEAR_EXHAUSTION_LONG
   └─ MVRV-60d deep underwater + MDIA inflow + NOT MVRV macro bearish

📈 BEAR_RALLY_LONG  
   └─ MVRV-60d underwater/deep + MDIA inflow + (MVRV macro neutral OR bullish)

🐻 BEAR_CONTINUATION_SHORT
   └─ MVRV-60d profitable + NOT MDIA inflow + (MDIA aging OR MVRV macro bearish)

⚠️ LATE_DISTRIBUTION_SHORT
   └─ MVRV-60d breakeven/profitable + NOT MDIA inflow + (MVRV macro bearish OR neutral)

🟡 TRANSITION_CHOP
   └─ Conflicting valuation and flow signals (fallback)
```

### Stricter Short Logic

Short setups historically suffer from false positives during choppy structures. The new hierarchy applies a strict `mdia_inflow` gate—shorts cannot fire if the market is experiencing active near-term capital inflow.

Key rules:
- **BEAR_CONTINUATION**: Requires definitive whale distribution + (MVRV put OR bear).
- **DISTRIBUTION_RISK**: Fires when whales are distributing and MVRV is not macro bullish, provided there is no active MDIA inflow.
- **BEAR_PROBE**: Weakest short state. Triggers on strong distribution alone, but is heavily vetted by overlays and size penalized to 0.35-0.60x.

### Option Signal Fallback

When fusion returns `NO_TRADE`, **rule-based option signals** can fire as fallback trades:

| Signal | Direction | Conditions | Sizing |
|--------|-----------|------------|--------|
| `OPTION_CALL` | 🟢 Long | MVRV cheap (2+ of: undervalued_90d, new_low_180d, near_bottom) + Sentiment fear (sent_norm < -1.0) | 0.75x |
| `OPTION_PUT` | 🔴 Short | MVRV near-peak (60d_pct ≥ 0.80 OR 60d_dist_from_max ≤ 0.20) + Sentiment greed (sent_norm > 1.0) + Whale distribution | 0.75x |

Key design decisions:
- **Feature-level independent of fusion**: Option signals are computed from `signal_option_call` / `signal_option_put` features from `interactions.py`
- **5-day cooldown**: Prevents rapid consecutive option signals (reduced from 7d — OPTION_CALL hits 81%)
- **Overlay filtered**: Subject to the same overlay veto logic (size_mult == 0 blocks the trade)
- **In production** (`services.py`): Only promoted to actual trade when fusion = NO_TRADE (fusion takes priority)
- **In analysis** (`analyze_hit_rate`): Tracked independently with their own cooldown/overlay gates

Use `analyze_fusion --explain --date YYYY-MM-DD` to see both fusion state and option signal status.

---

## Static Confidence Mapping

Instead of dynamic linear scoring, states have static, empirically validated properties mapped directly to execution parameters (`fusion.STATE_PROPERTIES`):

| State | Static Confidence | Static Score | Impact |
|-------|-------------------|--------------|--------|
| `STRONG_BULLISH` | HIGH | +5 | Full Sizing |
| `EARLY_RECOVERY` | HIGH | +4 | Full Sizing |
| `BEAR_CONTINUATION` | HIGH | -5 | Full Sizing |
| `MOMENTUM_CONTINUATION` | MEDIUM | +3 | Standard Sizing |
| `DISTRIBUTION_RISK` | MEDIUM | -3 | Standard Sizing |
| `BULL_PROBE` | LOW | +1 | Gated 0.5x sizing |
| `BEAR_PROBE` | LOW | -2 | Gated 0.5x sizing |
| `NO_TRADE` | LOW | 0 | No Trade |
| `BEAR_EXHAUSTION_LONG` | HIGH | +4 | Full Sizing |
| `BEAR_RALLY_LONG` | MEDIUM | +2 | Standard Sizing |
| `BEAR_CONTINUATION_SHORT`| HIGH | -4 | Full Sizing |
| `LATE_DISTRIBUTION_SHORT`| MEDIUM | -2 | Standard Sizing |
| `TRANSITION_CHOP` | LOW | 0 | No Trade |

---

## ML Training Pipeline

The ML layer (`ml/training.py`) provides two training approaches:

### Holdout Training
- **Train**: Up to 2023-12-31
- **Validation**: 2024-01-01 to 2024-12-31
- **Test**: 2025-01-01 onwards

### Walk-Forward Validation
Rolling validation with 6-month windows for stability assessment:
- Trains on expanding window
- Validates on next 6 months
- Reports Top 5% precision and AUC per fold
- Final model trained on all pre-2025 data

### Feature Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `PURE` | Raw numeric features only | Baseline ML |
| `HYBRID` | Raw + handcrafted regimes | Production (default) |

### Model Artifacts

Saved to `models/` directory:
- `long_model.joblib` - Long position classifier
- `short_model.joblib` - Short position classifier

---

## Overlays (Execution Modifiers)

Overlays (`signals/overlays.py`) modify **how hard** you press a trade. They never flip direction—only amplify or reduce conviction.

### Long Overlays (Sentiment + MVRV Composite)

| Overlay | Trigger | Effect |
|---------|---------|--------|
| **Full Edge** (+2) | Fear stabilizing + MVRV undervalued | Size +25%, DTE +50% |
| **Partial Edge** (+1) | Sentiment OR MVRV favorable | Size +10% |
| **Moderate Veto** (-1) | Euphoria persisting | Size -50% |
| **Strong Veto** (-2) | Euphoria + MVRV overvalued rollover | **No trade** |

### Short Overlays (MVRV-60d + EFB Distribution Pressure)

**Layer 1: MVRV-60d Near-Peak Score**

Blends `mvrv_60d_pct_rank` and `mvrv_60d_dist_from_max`:

| Score | Overlay | Effect |
|-------|---------|--------|
| ≥ 0.85 | Full Edge | Size +15%, DTE extended |
| ≥ 0.75 | Partial Edge | Size +5% |
| ≤ 0.35 | Soft Veto | Size -50% |
| ≤ 0.25 | Hard Veto | **No trade** |

**Layer 2: EFB Distribution Pressure** (`compute_efb_veto`)

Vetoes `OPTION_PUT` trades when `distribution_pressure_score < 0.40` (BTC leaving exchanges = supply tightening = shorts unreliable). Applied in `services._determine_trade_decision()` after overlay check. Tuned from historical miss analysis: OPTION_PUT HR improved from 47% → 57% (69% veto accuracy, +5 net correct vetoes).

The `distribution_pressure_score` is a composite of `flow_sum_7` (60%), `flow_slope_7` (25%), and `flow_z_90` (15%) from the exchange flow feature module (`features/metrics/exchange_flow.py`).

### Veto Dominance Rules

1. **STRONG/HARD veto always wins** (overrides any edge)
2. **Moderate/soft veto beats partial edge**
3. **Full edge can override moderate veto**
4. **EFB veto only applies to OPTION_PUT** (soft veto, never downgrades existing decisions)

---

## Trade Types

Hit rates from 5% target, 14-day horizon, 266 trades (with overlays and cooldowns active):

> **Note:** Hit rate measures whether the underlying moves 5% in the expected direction within 14 days — it does not account for option premium decay, IV crush, or actual P&L. A "hit" doesn't guarantee profit on the option trade.

| Type | Direction | Sizing | Source | Hit Rate (5%) |
|------|-----------|--------|--------|---------------|
| OPTION_CALL | 🟢 Long | 0.75x | Rule | **92.3%** |
| OPTION_PUT | 🔴 Short | 0.75x | Rule | **80.0%** |
| BULL_PROBE | 🟢 Long | 0.35-0.60x | Fusion | **75.7%** |
| PRIMARY_SHORT | 🔴 Short | 1.0x | Fusion | **74.4%** |
| LONG | 🟢 Long | 1.0x | Fusion | **72.2%** |
| MVRV_SHORT | 🔴 Short | 0.75x | Rule | **70.0%** |
| BEAR_PROBE | 🔴 Short | 0.35-0.60x | Fusion | **55.6%** |
| TACTICAL_PUT | 🔴 Put | 0.4-0.6x | Tactical | **43.5%** |
| IRON_CONDOR | 🟡 Neutral | 0.50x | Condor Gate | **60.0%** (±10%/7d range) |

### Option Strategy Selection — Fusion States (`STRATEGY_MAP` in `options.py`)

Strategies tuned from `analyze_path_stats` (14d horizon, 5% target).

| State | Primary Structure | Strike | DTE | Spread Width | Take-Profit | Max Hold |
|-------|------------------|--------|-----|-------------|-------------|----------|
| STRONG_BULLISH | Call spread, long call | SLIGHT_ITM | 9–14d (opt 11) | 9% | 70% | 7d |
| EARLY_RECOVERY | Call spread, long call | SLIGHT_ITM | 14–30d (opt 21) | 11% | 70% | 7d |
| MOMENTUM | Call spread, long call | SLIGHT_ITM | 12–21d (opt 14) | 10% | 70% | 10d |
| DISTRIBUTION_RISK | Put spread | SLIGHT_ITM | 8–14d (opt 12) | 8% | 70% | 6d |
| BEAR_CONTINUATION | Put spread | SLIGHT_ITM | 8–14d (opt 12) | 10% | 70% | 6d |
| BULL_PROBE | Call spread | SLIGHT_ITM | 10–14d (opt 12) | 7% | 70% | 8d |
| BEAR_PROBE | Put spread | SLIGHT_ITM | 12–16d (opt 14) | 7% | 70% | 10d |
| BEAR_EXHAUSTION_LONG | Call spread, long call | SLIGHT_ITM | 8–14d (opt 11) | 9% | 70% | 6d |
| BEAR_RALLY_LONG | Call spread | SLIGHT_ITM | 10–14d (opt 12) | 8% | 70% | 8d |
| BEAR_CONTINUATION_SHORT | Put spread | SLIGHT_ITM | 8–14d (opt 12) | 10% | 70% | 6d |
| LATE_DISTRIBUTION_SHORT | Put spread | SLIGHT_ITM | 10–14d (opt 12) | 8% | 70% | 8d |

### Option Strategy Selection — Decision Overrides (`DECISION_STRATEGY_MAP` in `options.py`)

Trade decisions that don't map 1:1 to a fusion MarketState get their own strategy guidance:

| Decision | Structures | Strike | DTE | Rationale |
|----------|-----------|--------|-----|----------|
| OPTION_CALL | long_call, call_spread | slight_itm | 7–14d | MVRV cheap + Sentiment fear. Exploratory long probe. |
| OPTION_PUT | long_put, put_spread | slight_itm | 7–14d | MVRV overheated + Sentiment greed. Defined-risk short. |
| TACTICAL_PUT | put_spread | slight_otm | 7–12d | Hedge inside bull: MVRV-60d near-peak & rolling over. |
| IRON_CONDOR | iron_condor | otm | 7–14d | Range gate: chop state + neutral metrics. Sell premium, wide wings. |

### Per-State Path-Risk Adjustment (`get_strategy_with_path_risk`)

When a state's invalidation-before-hit rate ≥ 30% (or combined inv + ambiguous ≥ 35%), strikes shift one level deeper ITM and DTE floors are raised. Uses **per-state constants** (not flat aggregates) from the 5% target analysis:

| State | Inv Rate | Triggers? | Notes |
|-------|----------|-----------|-------|
| EARLY_RECOVERY | 6.7% | ❌ | n=17, very clean |
| BEAR_CONTINUATION | 16.7% | ❌ | n=7, clean |
| DISTRIBUTION_RISK | 17.4% | ❌ | n=35, clean |
| BULL_PROBE | 19.0% | ❌ | n=83, moderate |
| BEAR_RALLY_LONG | 21.2% | ❌ | n=66 |
| BEAR_CONTINUATION_SHORT | 25.0% | ❌ | n=4 |
| MOMENTUM_CONTINUATION | 26.3% | ❌ | n=191, dropped below 30% |
| STRONG_BULLISH | 28.6% | ❌ | n=11 |
| LATE_DISTRIBUTION_SHORT | 28.6% | ❌ | n=59 |
| **BEAR_EXHAUSTION_LONG** | **50.0%** | ✅ | n=5, runtime shifts strike -> ITM, DTE -> 10-14d |
| **BEAR_PROBE** | **55.0%** | ✅ | n=36, messiest state, triggers path-risk |

**Runtime Structure Gating** (`generate_trade_signal`): Advanced structures are conditionally added by state:
- **Backspreads**: Only for high-confidence structural continuations (STRONG_BULLISH → call backspread, BEAR_CONTINUATION → put backspread)
- **Credit Spreads**: Scaled by IV percentile policy.

### Stop Loss Strategy

Data-driven three-layer exit system calibrated from `analyze_path_stats` (7 states × 6 invalidation levels, 14d horizon, 5% target). Integrated into `SpreadGuidance` in `options.py`.

**Per-State Parameters:**

| State | Fixed Stop | Scale-Down Day | Hard Cut | Stop/Width |
|-------|-----------|----------------|----------|------------|
| STRONG_BULLISH | 4.0% | Day 5 (→25%) | Day 7 | 44% |
| EARLY_RECOVERY | 4.0% | Day 4 (→25%) | Day 7 | 36% |
| MOMENTUM | 4.5% | Day 6 (→25%) | Day 10 | 45% |
| DISTRIBUTION_RISK | 3.5% | Day 4 (→25%) | Day 6 | 44% |
| BEAR_CONTINUATION | 3.5% | Day 4 (→25%) | Day 6 | 35% |
| BULL_PROBE | 3.5% | Day 4 (→25%) | Day 8 | 50% |
| BEAR_PROBE | 4.0% | Day 7 (→25%) | Day 10 | 57% |
| BEAR_EXHAUSTION_LONG | 4.0% | Day 5 (→25%) | Day 6 | 44% |
| BEAR_RALLY_LONG | 3.5% | Day 4 (→25%) | Day 8 | 44% |
| BEAR_CONTINUATION_SHORT | 3.5% | Day 4 (→25%) | Day 6 | 35% |
| LATE_DISTRIBUTION_SHORT | 4.0% | Day 5 (→25%) | Day 8 | 50% |

**Exit timeline:**
1. **Fixed price stop**: Underlying moves `stop_loss_pct` against you → close all
2. **Scale-down**: At `scale_down_day` (≈ p75 TTH) → reduce to 25% position
3. **Hard time stop**: At `max_hold_days` → close everything remaining

**Why these numbers:**
- **3.5–4.0% sweet spot**: Validated by sweeping 2–5% across all states. Below 3% → 40%+ false stops (killing winners). Above 5% → starts missing losers with diminishing returns.
- **Stop ≈ 40–50% of spread width**: Natural ratio — at 4% underlying stop on a 9% spread, actual capital loss is ~35–50% of premium (spread still has time value).
- **~20% of winners hit after day 7**: Scaling to 25% at the p75 TTH day matches position size to conditional probability of success.
- **Winner MAE 2–4% vs Loser MAE 9–22%**: The 4% stop sits cleanly between winner noise and loser trajectory — the populations separate well.
- **Regime-stable**: Year-by-year robustness check (2017–2025) confirms 3.5–4% works across market eras. Modern regime (2020+) shows even lower false stops (~15–20%).

---

## REST API

### Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/v1/health/` | GET | ❌ No | Health check |
| `/api/v1/signals/` | GET | ✅ Token | List signals (paginated, summary view) |
| `/api/v1/signals/latest/` | GET | ✅ Token | Latest signal (full detail) |
| `/api/v1/signals/<date>/` | GET | ✅ Token | Signal by date (YYYY-MM-DD format) |
| `/api/v1/fusion/explain/` | GET | ✅ Token | Explain fusion logic. Requires `?date=YYYY-MM-DD`. Optional: `?year=YYYY` |
| `/api/v1/fusion/analysis/metric-stats/` | GET | ✅ Token | Metric distribution. Requires `?metric=<mdia_bucket\|whale_bucket\|mvrv_ls_bucket>`. Optional: `?min_count=10&year=YYYY` |
| `/api/v1/fusion/analysis/combo-stats/` | GET | ✅ Token | Group-by combos. Requires `?group_by=<comma_separated_columns>` (e.g. `mdia_bucket,whale_bucket`). Optional: `?min_count=10&year=YYYY` |
| `/api/v1/fusion/analysis/state-stats/` | GET | ✅ Token | State hit-rates. Optional: `?min_count=10&year=YYYY` |
| `/api/v1/fusion/analysis/score-validation/` | GET | ✅ Token | Score validation. Optional: `?type=<monotonicity\|stability>&min_count=10&year=YYYY` |

### Authentication

All authenticated endpoints require Bearer token in the `Authorization` header:

```bash
# Health check (no auth)
curl http://localhost:8000/api/v1/health/

# Get latest signal (auth required)
curl -H "Authorization: Token YOUR_API_TOKEN" \
     http://localhost:8000/api/v1/signals/latest/

# Get signals list
curl -H "Authorization: Token YOUR_API_TOKEN" \
     http://localhost:8000/api/v1/signals/

# Get signal for specific date
curl -H "Authorization: Token YOUR_API_TOKEN" \
     http://localhost:8000/api/v1/signals/2024-11-06/
```

### API Response Fields

**Full signal response** (`/latest/` and `/<date>/`):

| Field | Type | Description |
|-------|------|-------------|
| `date` | string | Signal date (YYYY-MM-DD) |
| `p_long` | float | ML probability for long position |
| `p_short` | float | ML probability for short position |
| `signal_option_call` | int | Call signal (0/1) |
| `signal_option_put` | int | Put signal (0/1) |
| `fusion_state` | string | Core regime identifier (e.g. `strong_bullish`) |
| `fusion_confidence` | string | Sizing conviction (`high`, `medium`, `low`) |
| `fusion_score` | int | Static integer score assigned to the state |
| `short_source` | string | Source of short signal (`rule`, `tactical`, etc.) |
| `score_components` | dict | Dictionary showing which boolean traits contributed |
| `overlay_reason` | string | Explanation from overlay logic |
| `size_multiplier` | float | Position size multiplier |
| `dte_multiplier` | float | DTE adjustment multiplier |
| `tactical_put_active` | bool | Whether tactical put is triggered |
| `tactical_put_strategy` | string | Strategy type if tactical put active |
| `tactical_put_size` | float | Tactical put sizing |
| `trade_decision` | string | Final decision (CALL/PUT/TACTICAL_PUT/OPTION_CALL/OPTION_PUT/MVRV_SHORT/IRON_CONDOR/NO_TRADE) |
| `trade_notes` | string | Additional notes |
| `no_trade_reasons` | list | Reasons why trade was blocked (if NO_TRADE) |
| `decision_trace` | list | Step-by-step decision trace for debugging |
| `effective_size` | float | Final position size after all adjustments |
| `decision_version` | string | Version of decision logic |
| `model_versions` | dict | ML model filenames used |
| `option_structures` | string | Recommended structures (e.g., `call_spread`) |
| `strike_guidance` | string | Strike selection (e.g., `slight_itm`, `itm`) |
| `dte_range` | string | DTE range (e.g., `7-14d`) |
| `strategy_rationale` | string | Strategy explanation with spread guidance (width, take-profit, max-hold) |
| `stop_loss` | string | Stop loss guidance (e.g., `4.0% stop \| scale to 25% on day 5 \| hard cut day 7`) |
| `stop_loss_pct` | float | Numeric stop loss percentage for execution |
| `scale_down_day` | int | Day to scale down position |
| `max_hold_days` | int | Maximum days to hold position |
| `spread_width_pct` | float | Spread width percentage |
| `take_profit_pct` | float | Take profit percentage |
| `condor_score` | float | Iron condor range score (0-100) |
| `condor_eligible` | bool | Whether condor gate passed |
| `condor_veto_reasons` | list | Hard veto reasons blocking condor |
| `condor_score_components` | dict | Breakdown of range score components |
| `created_at` | datetime | Record creation timestamp |
| `updated_at` | datetime | Record update timestamp |

**Summary response** (`/signals/` list):

| Field | Type | Description |
|-------|------|-------------|
| `date` | string | Signal date |
| `p_long` | float | ML probability for long |
| `p_short` | float | ML probability for short |
| `fusion_state` | string | Market state |
| `fusion_score` | int | Fusion score |
| `short_source` | string | Source of short signal |
| `trade_decision` | string | Final trade decision |
| `condor_score` | float | Iron condor range score |
| `condor_eligible` | bool | Whether condor gate passed |

### Creating API Tokens

```bash
# Create a new API token for a user
python manage.py create_api_token --username telegram_bot

# Token is printed to console - save it securely
```

---

## Execution Layer (Status: Data Collection Phase)

The `execution` app contains infrastructure for automated BTC options execution on Bybit and Deribit. However, **live execution is currently paused** while we collect real options data to build empirical leverage profiles.

### Current Status

The system has theoretical edge (65-72% hit rate on underlying moves), but options execution requires understanding real-world leverage behavior:

- Options leverage is state-dependent and spikes to 15-20x near ATM strikes
- Without empirical leverage data, position sizing and stop losses are unreliable
- We're collecting options snapshots via `datafeed/ingestion/` to build leverage surfaces

See [docs/options_data_leverage_plan.md](docs/options_data_leverage_plan.md) for the data collection roadmap.

### What Exists (Not Yet Production-Ready)

The execution infrastructure is built but untested with real capital:

```
execution/
├── exchanges/           → Bybit V5 & Deribit adapters
├── services/            → Orchestrator, risk checks, position management
├── models.py            → ExchangeAccount, Intent, Order, Position
└── management/commands/ → execute_signal, sync_positions, manage_exits
```

### V7 Execution Design

When we resume execution, the V7 design prioritizes path stability over returns:

- **Wider spreads** (8-12% width) to reduce effective leverage
- **Deeper ITM strikes** to avoid gamma spikes
- **Longer DTE** (14-21d primary) for more time value
- **Realistic leverage model** (6-10x base, 12-18x stress)

See [execution/docs/execution_design.md](execution/docs/execution_design.md) for full V7 specification.

### Data Collection Commands

```bash
# Collect options snapshots from exchanges
python manage.py collect_options --exchange bybit --dte-min 7 --dte-max 21 --moneyness 0.15
python manage.py collect_options --exchange deribit --dte-min 7 --dte-max 21 --moneyness 0.15

# Export collected data for analysis
python manage.py export_options --format csv
```

### Next Steps

1. Collect 2-4 weeks of options snapshots across market conditions
2. Build empirical leverage profiles by DTE/moneyness/IV regime
3. Validate V7 assumptions against real data
4. Paper trade on testnet before live execution

---

## Commands

### Daily Operations

```bash
# Build feature CSV from raw data
python manage.py build_features

# Generate and persist today's signal (with verbose output)
python manage.py generate_signal --verbose

# Generate signal without persistence (dry run)
python manage.py generate_signal --verbose --no-persist
```

### Training

```bash
# Train ML models (holdout validation)
python manage.py train_models

# Analyze historical path performance and exits
python manage.py analyze_path_stats --target 0.05 --horizon 14

# Diagnose why trades didn't fire
python manage.py diagnose_notrade --year 2025
```

### Data Sync

```bash
# Sync data from Google Sheets
python manage.py sync_sheets
```

### API Server

```bash
# Start API server
python manage.py runserver

# Create API token
python manage.py create_api_token --username telegram_bot
```

### MVRV Short Signal (Bear Market Tactical)

A tactical short signal designed specifically for bear market conditions (cycle days 540-900 post-halving).

**Philosophy:**

The signal exploits a behavioral pattern in bear markets: when short-term holders (7-day MVRV) are profitable while medium-term holders (60-day MVRV) are at or above breakeven, the market is temporarily overextended. In bear regimes, these relief rallies tend to fade quickly as holders who bought the dip take profits, creating predictable mean-reversion opportunities.

The key insight is the **overlay combination**:
- `mvrv_7d >= 1.02`: Recent buyers are profitable (short-term greed)
- `mvrv_60d >= 1.0`: Medium-term holders at breakeven (no capitulation cushion)

When both conditions align in a bear cycle, price tends to drop 4% before rising 4% with a **1.8:1 ratio** (58% drop-first vs 31% rise-first). This edge was validated out-of-sample across three independent bear markets (2018, 2022, 2025-2026).

**Why DCA works here:**

The $30/$60 split (33%/67%) exploits the asymmetry:
- If drop hits first (58%): Small win on small position
- If rise hits first, then drops (23%): Big win on averaged-down position  
- If rise hits first, never drops (19%): Small loss on small position

This creates positive expected value (+1.99% per trade) while limiting downside exposure.

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

**Strategy Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mvrv7d` | 1.02 | MVRV 7d threshold (short-term profit level) |
| `--mvrv60d` | 1.0 | MVRV 60d overlay (medium-term breakeven) |
| `--target` | 4.0 | Target drop % |
| `--window` | 5 | Window days to hit target |
| `--bear-start` | 540 | Bear mode start (cycle days post-halving) |
| `--bear-end` | 900 | Bear mode end (cycle days post-halving) |

**Backtest Results (Walk-Forward Validated):**
| Period | Role | Signals | Drop First | Rise First | Ratio |
|--------|------|---------|------------|------------|-------|
| 2018 | Train | 38 | 55% | 37% | 1.5:1 |
| 2022 | Test | 32 | 56% | 31% | 1.8:1 |
| 2025-2026 | Test | 9 | 67% | 11% | 6.0:1 |

**DCA Execution Strategy:**
- Entry 1: 33% of position at signal price
- DCA Trigger: +4% price rise → add remaining 67%
- Target: -4% from original entry (not averaged entry)
- Max Hold: 5 days
- Expected Value: +1.99% per trade ($1.79 per $90 risked)

---

## Iron Condor Range Gate

Monetizes NO_TRADE / TRANSITION_CHOP days by selling iron condors when BTC is likely to stay range-bound.

**How it works:** Scores each day 0-100 based on existing features (chop fusion state, MVRV neutral, flat sentiment, no directional signals, calm exchange flows). Score ≥ 75 + no hard vetoes → `IRON_CONDOR` fires. Sits at priority 6 in the decision cascade — only evaluates after all directional signals are exhausted.

**Label:** ±10% band, 7-observation forward horizon. Base rate: 58% of days stay in range.

**Performance:** 60% hit rate overall (89 trades, 2017–2026). 80.6% since 2023 (25/31 trades).

**Strategy:** 10% OTM wings, 7–14d DTE, 50% take-profit, 6% stop-loss, scale down day 5, hard cut day 7.

**Hard vetoes:** Option signals active, MVRV strong trend, MDIA strong inflow, extreme sentiment persisting 5d+, strong whale distribution, ATR expansion (7d/30d > 1.5), large gap (> 4%).

```bash
# Research commands (run ad-hoc, not daily)
python manage.py chop_analysis --out reports/chop
python manage.py condor_walkforward --out reports/condor_wf

# Hit rate analysis (includes IRON_CONDOR alongside all other types)
python manage.py analyze_hit_rate --type IRON_CONDOR
```

**Report CSVs** (`reports/chop/`): Research artifacts for calibration, not daily outputs. Re-run when adding features, changing fusion rules, or quarterly for recalibration.

| File | Purpose |
|------|---------|
| `feature_prevalence.csv` | Which features predict range vs breakout (lift ratios) |
| `fusion_state_alignment.csv` | How each fusion state correlates with 7d range outcomes |
| `chop_condition_combos.csv` | Precision of combined chop conditions |
| `range_score_calibration.csv` | Actual range rate per score decile |
| `breakout_risk_conditions.csv` | Conditions that predict breakout (hard veto source) |
| `chop_analysis_full.csv` | Complete labeled dataset for ad-hoc analysis |

See [docs/iron_condor_spec.md](docs/iron_condor_spec.md) for full specification including tail risk profile, stop vs label band rationale, and monitoring triggers.

**TODOs (Condor Gate Hardening):**
- [ ] Validate `mvrv_60d_flat_underval` (+5 provisional) incremental lift conditional on existing score components — standalone 78% safe rate may collapse once trend/vol filters are active
- [ ] Run `condor_walkforward` with +5/+7/+10 weight variants; promote weight only if out-of-sample precision improves without inflating pass rate into bad regimes
- [ ] Add IV floor filter (min IV rank at entry) — 58.9% hit rate on underlying ≠ profitable condors without sufficient premium collected
- [ ] Add liquidity gate (max bid-ask spread on 4 legs) — thin OTM BTC options can erase theoretical edge
- [ ] Backtest stop-loss slippage on 4-leg close during fast moves — the -25.9% worst-case flat-z period shows price can gap through the 6% stop
- [ ] Integrate `OptionSnapshot` data to measure actual condor P&L (not just range-staying rate)

---

## Configuration

### Cooldown Settings (Anti-Clustering)

| Trade Type | Cooldown | Constant |
|------------|----------|----------|
| CALL (LONG) | 7 days | `CORE_SIGNAL_COOLDOWN_DAYS` |
| PUT (SHORT) | 7 days | `CORE_SIGNAL_COOLDOWN_DAYS` |
| BULL_PROBE / BEAR_PROBE | 5 days | `PROBE_COOLDOWN_DAYS` |
| TACTICAL_PUT | 7 days | `TACTICAL_PUT_COOLDOWN_DAYS` |
| OPTION_CALL | 5 days | `OPTION_SIGNAL_COOLDOWN_DAYS` |
| OPTION_PUT | 5 days | `OPTION_SIGNAL_COOLDOWN_DAYS` |
| MVRV_SHORT | 5 days | `MVRV_SHORT_COOLDOWN_DAYS` |
| IRON_CONDOR | 5 days | `CONDOR_COOLDOWN_DAYS` |

---

## Key Files

| File | Purpose |
|------|---------|
| `features/feature_builder.py` | Feature engineering (MDIA, MVRV, Whales, Sentiment) |
| `signals/fusion.py` | Market state classification engine |
| `signals/overlays.py` | Edge amplifiers and veto gates |
| `signals/tactical_puts.py` | Hedge puts inside bull regimes |
| `signals/options.py` | Option strategy, strikes, DTE, spread guidance, path-risk adjustment |
| `signals/services.py` | SignalService for scoring + persistence |
| `signals/models.py` | DailySignal Django model |
| `features/metrics/interactions.py` | Option signal rules (MVRV cheap/hot + sentiment) |
| `ml/training.py` | ML training pipeline with walk-forward validation |
| `ml/predict.py` | Model inference for daily scoring |
| `api/views.py` | REST API endpoints |
| `api/serializers.py` | DRF serializers |
| `api/urls.py` | API URL routing |
| `execution/exchanges/base.py` | Provider-agnostic exchange interface |
| `signals/condor_gate.py` | Iron condor range gate (score, vetoes, evaluation) |
| `docs/iron_condor_spec.md` | Full iron condor specification |
| `execution/exchanges/bybit.py` | Bybit V5 API adapter |
| `execution/exchanges/deribit.py` | Deribit API adapter |
| `execution/services/orchestrator.py` | Signal → Intent → Risk → Exchange flow |
| `execution/services/risk.py` | Risk management (limits, duplicates) |
| `execution/models.py` | ExchangeAccount, ExecutionIntent, Order, Position |

---

---

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up database
python manage.py migrate

# Sync data from Google Sheets
python manage.py sync_sheets

# Build features
python manage.py build_features

# Train models
python manage.py train_models

# Start API server
python manage.py runserver
```

---

---

## Key Trading Insights (2025 Analysis - Retrained Models)

Based on truly out-of-sample analysis on 2025 data using retrained models:

### 1. Trust the Fusion Signal First
The Fusion engine alone achieved an **86% hit rate (12/14)** in 2025.
- **Fusion is the primary alpha generator.**
- ML probabilities are useful for sizing/risk management but should not gate trades too aggressively.

### 2. ML Probability Thresholds (Current Testing Phase)
The system is currently in a testing phase. The signal flow completely trusts the rule-based **Fusion Engine** to make trade decisions (LONG, SHORT, NO_TRADE).

**The ML probabilities (`p_long` and `p_short`) are recorded for informational purposes and are NOT currently used to gate or filter trades.**

In the next phase (automated trading), these empirical thresholds will be used to automate **position sizing and risk management**:

| Direction | Threshold (Future Sizing Target) | Observation |
|-----------|----------------------------------|-------------|
| **LONG** | p_long ≥ 0.70 | High confidence required. All 2025 winners had p ≥ 0.70. |
| **SHORT** | p_short ≥ 0.38 | **Conservative Model.** Winners appear as low as 0.38. Do not gate with 0.50. |

**Action**: For shorts, if Fusion says `BEAR_PROBE`/`SHORT` and `p_short` is even moderately active (>0.38), this will inform future automated sizing logic.

### 3. Per-Trade Cooldown is Critical
A **7-day core cooldown** on `CALL` and `PUT` prevents clustering while preserving directional priority.

Tactical and option trades use their own cooldowns (`TACTICAL_PUT`: 7d, `OPTION_*`: 5d), so fallback setups are still allowed when core trades are blocked.

### 4. What to Monitor (Testing Phase)
1. **Short Signals with Low ML**: Validate if `BEAR_PROBE` continues performing when `p_short` is 0.38-0.45.
2. **Fusion vs ML Divergence**: Monitor the outcomes when Fusion says TRADE but ML is very low (e.g., < 0.30). Currently, the system executes these trades based on Fusion state alone; tracking their performance will inform the rules for future automated gating and sizing.
3. **NO_TRADE days**: Remain noisy. High ML on NO_TRADE days is still a coin flip (50% hit rate). Stick to Fusion states.

---

## Philosophy

1. **Terrain over timing**: MVRV-LS is structural, not a trade timer
2. **Whale sponsorship required**: No trade without smart money alignment
3. **Probes are smaller**: Macro-neutral trades use defined risk at 0.5x
4. **Overlays never override fusion**: They amplify or reduce, not flip
5. **Fusion beats tactical**: When fusion has a directional view, it takes priority over tactical puts
6. **Clustering prevention**: Per-trade cooldowns reduce repeated entries without forcing a global no-trade lockout
7. **ML + Rules hybrid**: ML for probability, rules for regime classification
8. **Fallback signals**: Tactical puts can fire when bullish core CALL is cooldown-blocked; option signals fire as fallback when fusion is NO_TRADE
9. **Data-driven DTE**: All DTEs compressed to match actual TTH (median 3 days, 75th pct 6 days at 5% target). Don't pay for 30–60 days of theta when moves resolve in under a week
10. **Survive the shakeout**: SLIGHT_ITM strikes default across all states (75th MAE ~5% for winners). Per-state path-risk adjustment pushes to ITM only for genuinely messy states (momentum 31.8%, distribution_risk 50% invalidation)
11. **Defined risk first**: Call/put spreads as primary structure for all states. Advanced structures (backspreads, credit spreads) gated by confidence + IV conditions
12. **Per-state, not blanket**: Path-risk constants, invalidation rates, and DTE guidance are calibrated per market state from 5% target analysis—not applied as flat averages
13. **Three-layer exit**: Fixed stop + scale-down + hard time stop. Stop at 40–50% of spread width catches losers early while giving winners room to breathe through shakeouts. Scale-down to 25% at p75 TTH matches position to conditional win probability

---

*On-chain regime classification for BTC options trading.*
