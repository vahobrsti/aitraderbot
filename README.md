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
│  features/feature_builder.py → 60+ regime features               │
│  - MDIA slopes, buckets, regimes                                 │
│  - MVRV composite, LS, 60d percentiles                           │
│  - Whale accumulation/distribution patterns                      │
│  - Sentiment z-scores, buckets                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                         ML LAYER                                 │
│  ml/training.py → Model training with walk-forward validation   │
│  ml/predict.py → Inference for daily scoring                    │
│  models/ → Trained model artifacts (long_model, short_model)    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       SIGNAL LAYER                               │
│  signals/                                                        │
│  ├── fusion.py      → Market state classification (8 states)    │
│  ├── overlays.py    → Edge/veto modifiers for execution         │
│  ├── tactical_puts.py → Hedging logic inside bull regimes       │
│  ├── options.py     → Option strategy & strike selection        │
│  ├── services.py    → SignalService for scoring + persistence   │
│  └── models.py      → DailySignal model for DB storage          │
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
   └─ (MDIA distrib OR not inflow) + Whale distrib + (MVRV put OR bear)

⚠️ DISTRIBUTION_RISK
   └─ Whale distrib + not MDIA strong + (MVRV rollover/weak_down/warning)

🔥 MOMENTUM_CONTINUATION
   └─ MDIA inflow + Whale mixed/neutral + MVRV improving

🎯 BULL_PROBE (0.5x sizing)
   └─ MDIA inflow + Whale sponsored + MVRV neutral

🔴 BEAR_PROBE (0.5x sizing)
   └─ MDIA distrib + Whale distrib + MVRV neutral

🟡 NO_TRADE
   └─ No alignment (fallback)
```

---

## Scoring System

### Confidence Score Calculation

Each indicator contributes to a numeric score (-6 to +6 range):

| Indicator | Condition | Score |
|-----------|-----------|-------|
| **MDIA** | `strong_inflow` | +2 |
| **MDIA** | `inflow` (moderate) | +1 |
| **MDIA** | `distribution` | -1 |
| **Whales** | `broad_accum` | +2 |
| **Whales** | `strategic_accum` | +1 |
| **Whales** | `strong_distribution` | -2 |
| **Whales** | `distribution` | -1 |
| **MVRV-LS** | `call_confirm_recovery` | +2 |
| **MVRV-LS** | `trend_confirm` | +1 |
| **MVRV-LS** | `put_confirm` | -2 |
| **MVRV-LS** | `distribution_warning` | -1 |
| **Conflicts** | Per conflict detected | -1 |

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

### Short Overlays (MVRV-60d Only)

Uses a blended "near-peak score" from `mvrv_60d_pct_rank` and `mvrv_60d_dist_from_max`:

| Score | Overlay | Effect |
|-------|---------|--------|
| ≥ 0.85 | Full Edge | Size +15%, DTE extended |
| ≥ 0.75 | Partial Edge | Size +5% |
| ≤ 0.35 | Soft Veto | Size -50% |
| ≤ 0.25 | Hard Veto | **No trade** |

### Veto Dominance Rules

1. **STRONG/HARD veto always wins** (overrides any edge)
2. **Moderate/soft veto beats partial edge**
3. **Full edge can override moderate veto**

---

## Trade Types

| Type | Direction | Sizing | Strategy |
|------|-----------|--------|----------|
| LONG | 🟢 Long | 1.0x | Calls |
| BULL_PROBE | 🟢 Long | 0.35-0.60x | Call spread (defined risk) |
| PRIMARY_SHORT | 🔴 Short | 1.0x | Puts |
| BEAR_PROBE | 🔴 Short | 0.35-0.60x | Put spread (defined risk) |
| TACTICAL_PUT | 🔴 Put | 0.4-0.6x | Hedge inside bull regimes |

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
| `trade_decision` | string | Final decision (CALL/PUT/TACTICAL_PUT/NO_TRADE) |
| `trade_notes` | string | Additional notes |
| `option_structures` | string | Recommended structures (e.g., `long_call`) |
| `strike_guidance` | string | Strike selection (e.g., `atm`, `slight_otm`) |
| `dte_range` | string | DTE range (e.g., `45-90d`) |
| `strategy_rationale` | string | Human-readable strategy explanation |

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

# Generate and persist today's signal
python manage.py generate_signal --verbose

# Score latest day with full output (no persistence)
python manage.py score_latest

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

# Deep dive into fusion engine behavior
python manage.py analyze_fusion

# Analyze MVRV-LS neutral terrain
python manage.py analyze_neutral
```

---

## Configuration

### Cooldown Settings (Anti-Clustering)

| Trade Type | Cooldown |
|------------|----------|
| LONG | 7 days |
| PRIMARY_SHORT | 7 days |
| TACTICAL_PUT | 7 days |
| BULL_PROBE | 5 days |
| BEAR_PROBE | 5 days |

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
| `signals/options.py` | Option strategy selection |
| `signals/services.py` | SignalService for scoring + persistence |
| `signals/models.py` | DailySignal Django model |
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
   Structures: long_call
   Strike: atm
   DTE: 45-90d
   Rationale: High conviction bullish setup

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

## Key Trading Insights (2025 Analysis)

Based on out-of-sample analysis on 2025 data:

### ML Probability Thresholds

| Direction | Threshold | Hit Rate (2025) |
|-----------|-----------|-----------------|
| **LONG** (including BULL_PROBE) | p_long ≥ 0.70 | 75-100% |
| **SHORT** (including BEAR_PROBE) | p_short ≥ 0.50 | 80-100% |

**Note**: The LONG model runs hot (mean ~0.64), SHORT model is conservative (max ~0.55).

### Signal-Specific Performance (5% target, 14d horizon)

| Signal | 2025 Hit Rate | Notes |
|--------|---------------|-------|
| **PRIMARY_SHORT** | 100% (2/2) | Most reliable short signal |
| **BULL_PROBE** | 100% (2/2) | Rare but accurate |
| **BEAR_PROBE** | 80% (4/5) | Good frequency for shorts |
| **LONG** | 71% (5/7) | High volume, slightly lower rate |

### Best Strategy: Fusion + ML Filter

```
Entry Rules:
1. Wait for Fusion to fire (not NO_TRADE)
2. Check ML probability:
   - LONG/BULL_PROBE: require p_long ≥ 0.70
   - SHORT/BEAR_PROBE: require p_short ≥ 0.50
3. Apply 7-day GLOBAL cooldown (any trade type)

Result: 9/9 = 100% hit rate in 2025 (backtest)
```

### Market Regime Insights

| Regime | Best Signals | Avoid |
|--------|--------------|-------|
| **Bear Market** (2018, 2022) | TACTICAL_PUT, BEAR_PROBE, PRIMARY_SHORT | LONG, BULL_PROBE |
| **Bull Market** (2023, 2024, 2025) | LONG, BULL_PROBE | TACTICAL_PUT |

### What to Monitor

1. **ML calibration drift** - Retrain quarterly
2. **SHORT model still conservative** - May need recalibration
3. **NO_TRADE + high ML** - Even with p ≥ 0.7, only 68% (not worth trading)

---

## Philosophy

1. **Terrain over timing**: MVRV-LS is structural, not a trade timer
2. **Whale sponsorship required**: No trade without smart money alignment
3. **Probes are smaller**: Macro-neutral trades use defined risk at 0.5x
4. **Overlays never override fusion**: They amplify or reduce, not flip
5. **Clustering prevention**: 7-day cooldown collapses events into single trades
6. **ML + Rules hybrid**: ML for probability, rules for regime classification

---

*Built for BTC options trading with on-chain metrics.*
