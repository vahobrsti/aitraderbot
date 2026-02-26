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
│  list_trades command → Final trade opportunities with sizing     │
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
   └─ MDIA strong_inflow + Whale sponsored + MVRV call_confirm

📈 EARLY_RECOVERY  
   └─ MDIA inflow + Whale sponsored + MVRV recovery

🐻 BEAR_CONTINUATION
   └─ NOT MDIA inflow + Whale distrib + (MVRV put OR bear)

⚠️ DISTRIBUTION_RISK
   └─ Whale distrib + not MDIA strong + (MVRV rollover/weak_down/warning)

🔥 MOMENTUM_CONTINUATION
   └─ MDIA inflow + Whale mixed/neutral + MVRV improving

🎯 BULL_PROBE (0.5x sizing)
   └─ MDIA inflow + Whale sponsored + MVRV neutral

🔴 BEAR_PROBE (0.5x sizing)
   └─ Whale strong_distribution + MVRV neutral (no MDIA requirement)

🟡 NO_TRADE
   └─ No alignment (fallback)

📗 OPTION_CALL (0.75x sizing, rule-based fallback)
   └─ MVRV cheap (2+ flags) + Sentiment fear — promoted when fusion=NO_TRADE

📕 OPTION_PUT (0.75x sizing, rule-based fallback)
   └─ MVRV overheated + Sentiment greed + Whale distrib — promoted when fusion=NO_TRADE
```

### Hybrid Classification for Shorts

Short setups use a **hybrid approach** to catch distribution tops that rule-based logic misses:

1. **Rule-based shorts**: Traditional logic (BEAR_CONTINUATION, DISTRIBUTION_RISK, BEAR_PROBE)
2. **Score-based shorts**: When rules return NO_TRADE, check weighted short score

**Score-based Short Detection** uses `MEGA_WHALE_WEIGHT = 1.5` to amplify mega whale distribution signals:

| Short Score | State | Notes |
|-------------|-------|-------|
| ≤ -3.5 | BEAR_CONTINUATION | Strong distribution |
| -3.5 to -2.5 | DISTRIBUTION_RISK | Moderate distribution |
| -2.5 to -2.0 | BEAR_PROBE | Weak but tradeable |

The `short_source` field tracks origin: `'rule'` or `'score'`. Use `analyze_fusion --explain` to see breakdown.

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

## Scoring System

### Confidence Score Calculation

Each indicator contributes to a numeric score (-6 to +6 range):

| Indicator | Condition | Score |
|-----------|-----------|-------|
| **MDIA** | `strong_inflow` | +2 |
| **MDIA** | `inflow` (moderate) | +1 |
| **MDIA** | `aging` (rising) | 0 (Neutral) |
| **Whales** | `broad_accum` | +2 |
| **Whales** | `strategic_accum` | +1 |
| **Whales** | `strong_distribution` | -2 |
| **Whales** | `distribution` | -1 |
| **MVRV-LS** | `call_confirm_recovery` | +2 |
| **MVRV-LS** | `trend_confirm` | +1 |
| **MVRV-LS** | `put_confirm` | -2 |
| **MVRV-LS** | `distribution_warning` | -1 |
| **Conflicts** | Mega whale or MVRV conflict | -1 each |

> **Note**: Small whale conflicts (`whale_small_conflict`) are **not penalized**. 
> Mega whales (100-10k BTC) have significantly more market impact than small whales (1-100 BTC).
> When small whales show mixed signals but mega whales have clear direction, trust the mega whales.

### Score → Confidence Mapping

| Score | Confidence | Position Sizing |
|-------|------------|-----------------|
| ≥ +4 | HIGH | Full size (1.0x) |
| +2 to +3 | MEDIUM | Normal size |
| < +2 | LOW | Reduced or no trade |

### Score Thresholds for Trade Entry

| State | Min Score | Effect |
|-------|-----------|--------|
| BULL_PROBE | ≥ +2 | Gates weak probes |
| BEAR_PROBE | ≤ -2 | Gates weak shorts |
| HIGH confidence | ≥ +4 | Full sizing |
| MEDIUM confidence | +2 to +3 | Normal sizing |

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
| BEAR_PROBE | Put spread | SLIGHT_ITM | 7–12d (opt 9) | 7% | 70% | 5d |

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

**Runtime Structure Gating** (`generate_trade_signal`): Advanced structures are conditionally added:
- **Backspreads**: Only for HIGH confidence + score ≥ 4 (MOMENTUM → call backspread, BEAR_CONTINUATION → put backspread)
- **Short call spreads**: Only when IV percentile ≥ 85% in DISTRIBUTION_RISK or BEAR_CONTINUATION

---

## REST API

### Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/v1/health/` | GET | ❌ No | Health check |
| `/api/v1/signals/` | GET | ✅ Token | List signals (paginated, summary view) |
| `/api/v1/signals/latest/` | GET | ✅ Token | Latest signal (full detail) |
| `/api/v1/signals/<date>/` | GET | ✅ Token | Signal by date (YYYY-MM-DD format) |

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
| `fusion_state` | string | Market state (e.g., `strong_bullish`) |
| `fusion_confidence` | string | Confidence level (HIGH/MEDIUM/LOW) |
| `fusion_score` | int | Numeric fusion score (-6 to +6) |
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

# List all trade opportunities
python manage.py list_trades --year 2024
```

### Training

```bash
# Train ML models (holdout validation)
python manage.py train_models

# Train with walk-forward validation
python manage.py train_walk_forward
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

---

*Built for BTC options trading with on-chain metrics.*
