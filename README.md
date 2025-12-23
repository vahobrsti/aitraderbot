# AI Trader Bot

A Bitcoin options trading signal system using on-chain metrics, whale behavior, and sentiment analysis.

## Overview

This system generates trading signals by fusing three orthogonal market indicators:

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
│                       SIGNAL LAYER                               │
│  features/signals/                                               │
│  ├── fusion.py      → Market state classification (8 states)    │
│  ├── overlays.py    → Edge/veto modifiers for execution         │
│  ├── tactical_puts.py → Hedging logic inside bull regimes       │
│  └── options.py     → Option strategy & strike selection        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       API LAYER                                  │
│  api/ → REST API with token authentication                      │
│  features/services.py → SignalService for scoring + persistence │
│  features/models.py → DailySignal model for DB storage          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      EXECUTION LAYER                             │
│  list_trades command → Final trade opportunities with sizing     │
│  generate_signal → Daily automated signal persistence            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Fusion Engine (How Signals Work)

The fusion engine (`features/signals/fusion.py`) is the core decision-making system. It combines regime signals from three orthogonal indicators to classify market state and compute confidence scores.

### Step 1: Confidence Score Calculation

Each indicator contributes to a numeric score:

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

**Score Range**: -6 to +6 (theoretical)

### Step 2: Score → Confidence Mapping

| Score | Confidence | Position Sizing |
|-------|------------|-----------------|
| ≥ +4 | HIGH | Full size (1.0x) |
| +2 to +3 | MEDIUM | Normal size |
| < +2 | LOW | Reduced or no trade |

### Step 3: Market State Classification

The fusion engine classifies each day into one of 8 states using hierarchical rules:

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLASSIFICATION HIERARCHY                      │
│  (Most specific rules evaluated first)                          │
└─────────────────────────────────────────────────────────────────┘

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

### State Summary

| State | Description | Trade |
|-------|-------------|-------|
| `STRONG_BULLISH` | All indicators aligned bullish | 🟢 Long 1.0x |
| `EARLY_RECOVERY` | Smart money leading, structure turning | 🟢 Long 1.0x |
| `MOMENTUM_CONTINUATION` | Trend continuing without strong sponsorship | 🟢 Long 1.0x |
| `BULL_PROBE` | Timing + sponsorship, macro neutral | 🟢 Long 0.35-0.60x |
| `DISTRIBUTION_RISK` | Smart money exiting, structure cracking | 🔴 Short 1.0x |
| `BEAR_CONTINUATION` | No buyers, sellers in control | 🔴 Short 1.0x |
| `BEAR_PROBE` | Selling + distribution, macro neutral | 🔴 Short 0.35-0.60x |
| `NO_TRADE` | Conflicting signals, stay flat | ⚪ No trade |

---

## Overlays (Execution Modifiers)

Overlays (`features/signals/overlays.py`) modify **how hard** you press a trade. They never flip direction—only amplify or reduce conviction.

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

## Commands

### Core Commands

```bash
# Build feature CSV from raw data
python manage.py build_features

# Generate and persist today's signal
python manage.py generate_signal --verbose

# Score latest day with full output
python manage.py score_latest

# List all trade opportunities
python manage.py list_trades --year 2024
```

### API Commands

```bash
# Create API token for Telegram bot
python manage.py create_api_token --username telegram_bot

# Start API server
python manage.py runserver
```

### Training Commands

```bash
# Train ML models
python manage.py train_models
```

### Diagnostic Commands

```bash
# Analyze why days are NO_TRADE
python manage.py diagnose_notrade --year 2024

# Deep dive into fusion engine behavior
python manage.py analyze_fusion

# Analyze MVRV-LS neutral terrain
python manage.py analyze_neutral
```

---

## REST API

| Endpoint | Auth | Description |
|----------|------|-------------|
| `GET /api/v1/health/` | ❌ | Health check |
| `GET /api/v1/signals/` | ✅ Token | List signals (paginated) |
| `GET /api/v1/signals/latest/` | ✅ Token | Latest signal |
| `GET /api/v1/signals/<date>/` | ✅ Token | Signal by date |

**Authentication:**
```bash
curl -H "Authorization: Token YOUR_TOKEN" http://localhost:8000/api/v1/signals/latest/
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

### Score Thresholds

| State | Min Score | Effect |
|-------|-----------|--------|
| BULL_PROBE | ≥ +2 | Gates weak probes |
| BEAR_PROBE | ≤ -2 | Gates weak shorts |
| HIGH confidence | ≥ +4 | Full sizing |
| MEDIUM confidence | +2 to +3 | Normal sizing |

---

## Key Files

| File | Purpose |
|------|---------|
| `features/feature_builder.py` | Feature engineering (MDIA, MVRV, Whales, Sentiment) |
| `features/signals/fusion.py` | Market state classification engine |
| `features/signals/overlays.py` | Edge amplifiers and veto gates |
| `features/signals/tactical_puts.py` | Hedge puts inside bull regimes |
| `features/signals/options.py` | Option strategy selection |
| `features/services.py` | SignalService for scoring + persistence |
| `features/models.py` | DailySignal model |
| `api/views.py` | REST API endpoints |

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

# Build features
python manage.py build_features

# Train models
python manage.py train_models
```

---

## Philosophy

1. **Terrain over timing**: MVRV-LS is structural, not a trade timer
2. **Whale sponsorship required**: No trade without smart money alignment
3. **Probes are smaller**: Macro-neutral trades use defined risk at 0.5x
4. **Overlays never override fusion**: They amplify or reduce, not flip
5. **Clustering prevention**: 7-day cooldown collapses events into single trades

---

*Built for BTC options trading with on-chain metrics.*
