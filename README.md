# AI Trader Bot

A Bitcoin options trading signal system using on-chain metrics, whale behavior, sentiment analysis, and machine learning.

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
│                      EXECUTION LAYER                             │
│  analyze_fusion command → Deep dive into fusion engine behavior  │
│  generate_signal → Daily automated signal persistence            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Market Regimes & States

### 8 Canonical Market States

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

### Classification Hierarchy

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

Hit rates from 5% target, 14-day horizon, 213 trades (with overlays and cooldowns active):

| Type | Direction | Sizing | Source | Hit Rate (3.5%) | Hit Rate (5%) |
|------|-----------|--------|--------|-----------------|---------------|
| OPTION_CALL | 🟢 Long | 0.75x | Rule | **94.1%** | **94.1%** |
| PRIMARY_SHORT | 🔴 Short | 1.0x | Fusion | **87.5%** | **87.5%** |
| LONG | 🟢 Long | 1.0x | Fusion | 79.0% | **72.6%** |
| OPTION_PUT | 🔴 Short | 0.75x | Rule | 75.0% | **68.8%** |
| TACTICAL_PUT | 🔴 Put | 0.4-0.6x | Tactical | 70.6% | **64.7%** |
| BULL_PROBE | 🟢 Long | 0.35-0.60x | Fusion | 74.5% | **64.7%** |
| BEAR_PROBE | 🔴 Short | 0.35-0.60x | Fusion | 66.7% | **57.1%** |

### Option Strategy Selection — Fusion States (`STRATEGY_MAP` in `options.py`)

Strategies tuned from `analyze_path_stats` (14d horizon, 5% target, 213 trades).

**Key data**: median TTH 3 days, 69% hit rate, 49% overshoot→mean-revert path, 91.8% of winners exceed 6%.

| State | Primary Structure | Strike | DTE | Spread Width | Take-Profit | Max Hold |
|-------|------------------|--------|-----|-------------|-------------|----------|
| STRONG_BULLISH | Call spread, long call | SLIGHT_ITM | 7–14d (opt 11) | 9% | 70% | 6d |
| EARLY_RECOVERY | Call spread, long call | SLIGHT_ITM | 14–30d (opt 21) | 11% | 70% | 8d |
| MOMENTUM | Call spread, long call | SLIGHT_ITM | 7–14d (opt 11) | 9% | 70% | 6d |
| DISTRIBUTION_RISK | Put spread | SLIGHT_ITM | 7–14d (opt 12) | 9% | 70% | 6d |
| BEAR_CONTINUATION | Put spread | SLIGHT_ITM | 7–14d (opt 12) | 10% | 70% | 6d |
| BULL_PROBE | Call spread | SLIGHT_ITM | 7–12d (opt 9) | 7% | 70% | 5d |
| BEAR_PROBE | Put spread | SLIGHT_ITM | 7–12d (opt 9) | 7% | 70% | 7d |

### Option Strategy Selection — Decision Overrides (`DECISION_STRATEGY_MAP` in `options.py`)

Trade decisions that don't map 1:1 to a fusion MarketState get their own strategy guidance:

| Decision | Structures | Strike | DTE | Rationale |
|----------|-----------|--------|-----|----------|
| OPTION_CALL | long_call, call_spread | slight_itm | 7–14d | MVRV cheap + Sentiment fear. Exploratory long probe. |
| OPTION_PUT | long_put, put_spread | slight_itm | 7–14d | MVRV overheated + Sentiment greed. Defined-risk short. |
| TACTICAL_PUT | put_spread | slight_otm | 7–12d | Hedge inside bull: MVRV-60d near-peak & rolling over. |

### Per-State Path-Risk Adjustment (`get_strategy_with_path_risk`)

When a state's invalidation-before-hit rate ≥ 30% (or combined inv + ambiguous ≥ 35%), strikes shift one level deeper ITM and DTE floors are raised. Uses **per-state constants** (not flat aggregates) from the 5% target analysis:

| State | Inv Rate | Triggers? | Notes |
|-------|----------|-----------|-------|
| STRONG_BULLISH | 0.0% | ❌ | n=1, clean |
| EARLY_RECOVERY | 9.1% | ❌ | Very clean paths at 5% |
| BULL_PROBE | 11.4% | ❌ | Wide stop absorbs shakeouts |
| BEAR_CONTINUATION | 25.0% | ❌ | Small sample (n=4) |
| BEAR_PROBE | 26.9% | ❌ | Moderate |
| **MOMENTUM** | **31.8%** | ✅ | Messiest long state |
| **DISTRIBUTION_RISK** | **50.0%** | ✅ | Conservative (small sample) |

**Runtime Structure Gating** (`generate_trade_signal`): Advanced structures are conditionally added by state:
- **Backspreads**: Only for high-confidence structural continuations (STRONG_BULLISH → call backspread, BEAR_CONTINUATION → put backspread)
- **Credit Spreads**: Scaled by IV percentile policy.

### Stop Loss Strategy

Data-driven three-layer exit system calibrated from `analyze_path_stats` (7 states × 6 invalidation levels, 14d horizon, 5% target). Integrated into `SpreadGuidance` in `options.py`.

**Per-State Parameters:**

| State | Fixed Stop | Scale-Down Day | Hard Cut | Stop/Width |
|-------|-----------|----------------|----------|------------|
| STRONG_BULLISH | 4.0% | Day 5 (→25%) | Day 6 | 44% |
| EARLY_RECOVERY | 4.0% | Day 6 (→25%) | Day 8 | 36% |
| MOMENTUM | 4.0% | Day 5 (→25%) | Day 6 | 44% |
| DISTRIBUTION_RISK | 4.0% | Day 5 (→25%) | Day 6 | 44% |
| BEAR_CONTINUATION | 3.5% | Day 4 (→25%) | Day 6 | 35% |
| BULL_PROBE | 3.5% | Day 4 (→25%) | Day 5 | 50% |
| BEAR_PROBE | 4.0% | Day 6 (→25%) | Day 7 | 57% |

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
| `score_components` | dict | Dictionary showing which boolean traits contributed |
| `overlay_reason` | string | Explanation from overlay logic |
| `size_multiplier` | float | Position size multiplier |
| `dte_multiplier` | float | DTE adjustment multiplier |
| `tactical_put_active` | bool | Whether tactical put is triggered |
| `tactical_put_strategy` | string | Strategy type if tactical put active |
| `tactical_put_size` | float | Tactical put sizing |
| `trade_decision` | string | Final decision (CALL/PUT/TACTICAL_PUT/OPTION_CALL/OPTION_PUT/NO_TRADE) |
| `trade_notes` | string | Additional notes |
| `option_structures` | string | Recommended structures (e.g., `call_spread`) |
| `strike_guidance` | string | Strike selection (e.g., `slight_itm`, `itm`) |
| `dte_range` | string | DTE range (e.g., `7-14d`) |
| `strategy_rationale` | string | Strategy explanation with spread guidance (width, take-profit, max-hold) |
| `stop_loss` | string | Stop loss guidance (e.g., `4.0% stop \| scale to 25% on day 5 \| hard cut day 6`) |

**Summary response** (`/signals/` list):

| Field | Type | Description |
|-------|------|-------------|
| `date` | string | Signal date |
| `p_long` | float | ML probability for long |
| `p_short` | float | ML probability for short |
| `fusion_state` | string | Market state |
| `fusion_score` | int | Fusion score |
| `trade_decision` | string | Final trade decision |

### Creating API Tokens

```bash
# Create a new API token for a user
python manage.py create_api_token --username telegram_bot

# Token is printed to console - save it securely
```

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

### Diagnostics

```bash
# Analyze why days are NO_TRADE
python manage.py diagnose_notrade --year 2024

# Deep dive into fusion engine behavior (includes option signal stats)
python manage.py analyze_fusion
python manage.py analyze_fusion --direction all --year 2025  # shows OPTION_CALL/PUT in setups

# Explain a specific date (fusion + option signals)
python manage.py analyze_fusion --explain --date 2025-05-05

# Backtest hit rates (includes OPTION_CALL/OPTION_PUT)
python manage.py analyze_hit_rate --year 2025

# Sanity check option signals with MVRV flags
python manage.py sanity_check --year 2025 --cooldown 7

# Analyze MVRV-LS neutral terrain
python manage.py analyze_neutral
```

---

## Configuration

### Cooldown Settings (Anti-Clustering)

| Trade Type | Cooldown | Constant |
|------------|----------|----------|
| CALL (LONG / BULL_PROBE) | 7 days | `CORE_SIGNAL_COOLDOWN_DAYS` |
| PUT (SHORT / BEAR_PROBE) | 7 days | `CORE_SIGNAL_COOLDOWN_DAYS` |
| TACTICAL_PUT | 7 days | `TACTICAL_PUT_COOLDOWN_DAYS` |
| OPTION_CALL | 5 days | `OPTION_SIGNAL_COOLDOWN_DAYS` |
| OPTION_PUT | 5 days | `OPTION_SIGNAL_COOLDOWN_DAYS` |

### Environment Variables

Create a `.env` file with:

```bash
# Database
DATABASE_URL=sqlite:///db.sqlite3

# Google Sheets credentials path
GOOGLE_SHEETS_CREDENTIALS=/path/to/service-account.json

# Django settings
SECRET_KEY=your-secret-key
DEBUG=False
```

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

---

## Sample Output

```
============================================================
SIGNAL ANALYSIS: 2024-11-06
============================================================

--- ML MODEL SCORES ---
p_long  = 0.847
p_short = 0.123
signal_option_call = 1
signal_option_put  = 0

--- FUSION STATE ---
🟢 State: strong_bullish
   Confidence: high
   Score: +5

--- OVERLAY ---
   LONG EDGE (FULL): Fear + MVRV undervalued rising
   Size Multiplier: 1.25
   DTE Multiplier: 1.50

--- OPTION STRATEGY ---
   Structures: call_spread, long_call
   Strike: slight_itm
   DTE: 7-14d
   Rationale: Fresh capital + smart money + exhaustion resolved.
            [spread width=9%, take-profit=70%, max-hold=6d]
   Stop Loss: 4.0% stop | scale to 25% on day 5 | hard cut day 6

============================================================
TRADE DECISIONS
============================================================

✅ CALL
   Reason: Fusion: strong_bullish
   Confidence: high
   Size: 1.25
   Structures: LONG_CALL
```

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

## Project Structure

```
aibot/
├── aitrader/           # Django project settings
├── api/                # REST API app
│   ├── views.py        # API endpoints
│   ├── serializers.py  # DRF serializers
│   └── urls.py         # URL routing
├── datafeed/           # Data ingestion
│   └── ingestion/      # Google Sheets sync
├── features/           # Feature engineering
│   └── feature_builder.py
├── ml/                 # Machine learning
│   ├── training.py     # Model training
│   └── predict.py      # Inference
├── signals/            # Signal generation
│   ├── fusion.py       # Market state classifier
│   ├── overlays.py     # Edge/veto logic
│   ├── tactical_puts.py
│   ├── options.py
│   ├── services.py     # SignalService
│   └── models.py       # DailySignal model
├── models/             # Trained model artifacts
├── credentials/        # Service account credentials
└── manage.py
```

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

*Built for BTC options trading with on-chain metrics.*
