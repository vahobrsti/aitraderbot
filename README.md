# On-Chain BTC Options Signal System

A Bitcoin options trading signal system that fuses on-chain metrics (MDIA, MVRV, whale behavior) with sentiment analysis to classify market regimes and generate actionable trade signals with automated execution on Deribit.

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
│  datafeed/ingestion/ → Options data from Deribit/Bybit          │
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
│  ├── condor_gate.py → Iron condor range gate (chop monetization)│
│  ├── income_gate.py → Bull put / bear call spread income gates  │
│  ├── services.py    → SignalService for scoring + persistence   │
│  └── models.py      → DailySignal model for DB storage          │
└─────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                         ML LAYER                                 │
│  ml/training.py → Model training with walk-forward validation   │
│  ml/predict.py → Inference for daily scoring                    │
│  models/ → Trained models for signal scoring and option pricing │
│  execution/services/option_pricer.py → Learned leverage model   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       API LAYER                                  │
│  api/views.py → REST API endpoints with token authentication    │
│  api/serializers.py → DRF serializers for signal data           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    EXECUTION LAYER                               │
│  execution/                                                      │
│  ├── services/policy.py      → Data-driven policy engine        │
│  ├── services/trade_setup.py → Automated trade construction     │
│  ├── services/trade_validator.py → 11 pre-flight checks         │
│  ├── exchanges/deribit.py    → Deribit API adapter              │
│  └── models.py               → Intent, Order, Position          │
│  Signal → Trade Setup → Validation → Exchange → Audit Log       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   NOTIFICATIONS LAYER                            │
│  notifications/notifier.py → Telegram alerts with trade setups  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

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

# Generate today's signal
python manage.py generate_signal --verbose

# Start API server
python manage.py runserver
```

---

## Execution Layer

The execution layer transforms signals into executable trade setups with full validation.

### Signal → Trade Setup Flow

```
Signal (MVRV_SHORT)
    ↓
Policy Engine (DTE, delta, width from path analysis)
    ↓
Option Selection (best long leg by delta target)
    ↓
Short Leg Selection (weighted scoring: R:R 40%, width 30%, budget 20%, liquidity 10%)
    ↓
Trade Validator (11 pre-flight checks)
    ↓
Trade Setup (complete executable specification)
    ↓
Telegram Notification + API Response
```

### Trade Setup Builder

Automatically constructs complete trade setups from signals:

```python
from execution.services.trade_setup import TradeSetupBuilder
from datetime import date

builder = TradeSetupBuilder()
setup = builder.build_setup(signal_date=date(2026, 5, 2))

# For Telegram notification
message = setup.to_telegram_message()

# For API response
data = setup.to_dict()
```

### Policy Engine

Data-driven policy calibrated from path analysis (`analyze_path_stats`):

| Signal | Win% | Trades | DTE | Spread Width | Stop Loss | Delta | Shakeout% |
|--------|------|--------|-----|--------------|-----------|-------|-----------|
| OPTION_CALL | 91.7% | 24 | 7-12d | 10% | 8.5% | 0.65 | 35% |
| OPTION_PUT | 77.8% | 18 | 5-10d | 12% | 7.0% | -0.65 | 31% |
| PRIMARY_SHORT | 75.7% | 37 | 8-12d | 9% | 4.5% | -0.60 | 18% |
| BULL_PROBE | 71.4% | 42 | 7-11d | 12% | 4.0% | 0.55 | 19% |
| LONG | 70.9% | 79 | 9-14d | 10% | 4.5% | 0.60 | 21% |
| MVRV_SHORT | 68.2% | 22 | 12-18d | 9% | 7.0% | -0.60 | **57%** |
| IRON_CONDOR | 63.1% | 84 | 9-13d | 10% | 6.8% | 0.20 | 0% |
| BEAR_PROBE | 53.3% | 15 | 11-16d | 8% | 6.5% | -0.55 | **40%** |
| TACTICAL_PUT | 50.0% | 24 | 8-12d | 7% | 3.5% | -0.40 | 15% |

**Key calibration formulas:**
- **DTE** = TTH p75 + 2 days buffer
- **Spread Width** = MFE p75 × 0.65-0.70
- **Stop Loss** = MAE(winners) p75 + buffer

### Path-Aware Entry Strategy

Signals with high shakeout rates (≥40%) use DCA entry:

| Signal | Shakeout% | Invalidation% | Entry Strategy |
|--------|-----------|---------------|----------------|
| MVRV_SHORT | 57% | 43% | **DCA**: 33% initial, 67% on +7% rise |
| BEAR_PROBE | 40% | 40% | **DCA**: 33% initial, 67% on +6.5% rise |
| OPTION_CALL | 35% | 54% | **Scaled**: 50% initial, 50% on confirmation |
| OPTION_PUT | 31% | 44% | **Scaled**: 50% initial, 50% on confirmation |
| Others | <30% | <35% | **Single**: Full position at entry |

**Why DCA for shakeout-heavy signals:**
- 57% of MVRV_SHORT winners experience price going against before hitting target
- Initial small position survives the shakeout
- DCA entry catches the move at better price
- Reduces average entry cost and improves R:R

### Trade Validator (11 Checks)

| Code | Threshold | Severity | Action |
|------|-----------|----------|--------|
| SCALE_DOWN_NOT_EXECUTABLE | contracts=1 & scale_down | WARNING | Fallback: close full |
| TAKE_PROFIT_TOO_CONSERVATIVE | TP<30% & R:R>1 | WARNING | Suggest 50-70% TP |
| STOP_LOSS_BASIS_MISMATCH | Always | INFO | Add option-value stop |
| WIDTH_DEVIATION_FROM_POLICY | >30% deviation | WARNING | Flag as budget override |
| HIGH_EXECUTION_COST_IMPACT | costs >10% of max profit | WARNING | Suggest limit orders |
| WIDE_BID_ASK_* | spread > 15% | WARNING | Review liquidity |
| LOW_OPEN_INTEREST_* | OI < 5 | WARNING | Review liquidity |
| POOR_RISK_REWARD | R:R < 0.5:1 | BLOCKING | Reject trade |
| POSITION_EXCEEDS_BUDGET | risk > budget × 1.1 | BLOCKING | Reject trade |
| INSUFFICIENT_NET_EDGE | net_edge < 5% | BLOCKING | Reject trade |
| LIQUIDITY_INSUFFICIENT_SIZE | OI < contracts × 10 | WARNING | Reduce size |

### Example Trade Setup Output

```
🔴 *MVRV_SHORT TRADE SETUP*
📅 2026-05-02

*Market:* BTC @ `$78,653`
*Expiry:* 2026-05-15 (12 DTE)

━━━━━━━━━━━━━━━━━━━━━━
*TRADE: Bear Put Spread*
━━━━━━━━━━━━━━━━━━━━━━

*BUY:* `BTC-15MAY26-80000-P`
  Strike: `$80,000` | Δ: `-0.581` | IV: `35.3%`
  Ask: `$2,757.48`

*SELL:* `BTC-15MAY26-76000-P`
  Strike: `$76,000` | Δ: `-0.292` | IV: `37.9%`
  Bid: `$1,024.21`

━━━━━━━━━━━━━━━━━━━━━━
*METRICS*
━━━━━━━━━━━━━━━━━━━━━━
Width: `$4,000` (5.1%)
Net Debit: `$1,733.27`
Max Profit: `$2,266.73`
R:R: `1:1.31`
Net Edge: `72.3%`

━━━━━━━━━━━━━━━━━━━━━━
*EXIT RULES*
━━━━━━━━━━━━━━━━━━━━━━
🛑 Stop (Spot): BTC > `$83,474` (+6.1%)
🛑 Stop (Value): Spread < `$693` (60% lost)
✅ Take Profit: `60%` = `$1,360`/contract
⏰ Max Hold: 11d (until 2026-05-13)
📉 Scale Down: Day 5 → CLOSE FULL
```

---

## Multi-Signal Days & Hourly Re-evaluation

The signal system supports multiple trade types coexisting on the same day (e.g., `MVRV_SHORT` + `IRON_CONDOR`). Signals are evaluated hourly and new signal types can fire at any time throughout the day.

### How It Works

1. **Hourly cron** runs `generate_signal` throughout the day
2. **Independent gates**: Each signal type (CALL, PUT, OPTION_CALL, MVRV_SHORT, IRON_CONDOR) is evaluated independently
3. **Additive**: New signal types can fire at any hour (e.g., IRON_CONDOR at 2pm even if CALL fired at 9am)
4. **Idempotent**: If the same signal type already exists, it's updated only if meaningfully changed (no duplicate notifications)
5. **NO_TRADE ignored**: Fusion transitioning to NO_TRADE mid-day doesn't affect existing tradeable signals

### Database Schema

Each `(date, trade_decision)` pair is unique. Multiple rows can exist for the same date:

```
2026-05-23 | CALL         | is_active=True  (fired at 9am)
2026-05-23 | IRON_CONDOR  | is_active=True  (fired at 2pm)
```

### Operator Deactivation

The `is_active` field allows operators to manually deactivate signals:

```python
# Deactivate a signal (prevents execution/API from seeing it)
signal.is_active = False
signal.save()
```

Deactivated signals:
- Are excluded from execution and API responses
- Still count toward cooldowns (prevents gaming)
- **Stay deactivated** even if they re-qualify on subsequent hourly runs (operator decision is respected)

### Veto Handling

When overlays block a directional trade, the system emits `OVERLAY_VETO` in the `no_trade_reasons` field. However:

- `NO_TRADE` rows are **not persisted** — only tradeable signals are stored
- If a tradeable signal fires later in the day, any stale `NO_TRADE` rows are deactivated
- Execution commands and API endpoints check for active tradeable signals, not veto rows

```bash
# Execute the highest-priority signal for today
python manage.py execute_deribit --latest

# Execute a specific signal type on multi-signal days
python manage.py execute_deribit --latest --type IRON_CONDOR
```

---

## Market Regimes & States

The fusion engine (`signals/fusion.py`) operates in two distinct modes based on the current cycle phase.

### Standard Mode (8 Canonical States)

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

### Bear Mode (Days 540–900 Post-Halving)

| State | Description | Direction | Sizing |
|-------|-------------|-----------|--------|
| `BEAR_EXHAUSTION_LONG` | Holder capitulation + capital inflow | 🟢 Long | Full |
| `BEAR_RALLY_LONG` | Underwater holders getting relief | 🟢 Long | Standard |
| `BEAR_CONTINUATION_SHORT` | Profitable holders selling | 🔴 Short | Full |
| `LATE_DISTRIBUTION_SHORT` | Breakeven holders losing conviction | 🔴 Short | Standard |
| `TRANSITION_CHOP` | Conflicting signals | ⚪ None | 0x |

### Option Signal Fallback

When fusion returns `NO_TRADE`, rule-based option signals can fire:

| Signal | Direction | Conditions | Sizing |
|--------|-----------|------------|--------|
| `OPTION_CALL` | 🟢 Long | MVRV cheap + Sentiment fear | 0.75x |
| `OPTION_PUT` | 🔴 Short | MVRV overheated + Sentiment greed + Whale distrib | 0.75x |

---

## Trade Types & Hit Rates

Hit rates from 5% target, 14-day horizon, 345 trades (with overlays and cooldowns active):

| Type | Direction | Sizing | Trades | Hit Rate |
|------|-----------|--------|--------|----------|
| OPTION_CALL | 🟢 Long | 0.75x | 24 | **91.7%** |
| OPTION_PUT | 🔴 Short | 0.75x | 18 | **77.8%** |
| PRIMARY_SHORT | 🔴 Short | 1.0x | 37 | **75.7%** |
| BULL_PROBE | 🟢 Long | 0.35-0.60x | 42 | **71.4%** |
| LONG | 🟢 Long | 1.0x | 79 | **70.9%** |
| MVRV_SHORT | 🔴 Short | 0.75x | 22 | **68.2%** |
| IRON_CONDOR | 🟡 Neutral | 0.50x | 84 | **63.1%** |
| BULL_PUT_SPREAD | 🟢 Long | 0.50x | — | *pending* |
| BEAR_CALL_SPREAD | 🔴 Short | 0.50x | — | *pending* |
| BEAR_PROBE | 🔴 Short | 0.35-0.60x | 15 | **53.3%** |
| TACTICAL_PUT | 🔴 Put | 0.4-0.6x | 24 | **50.0%** |

> **Note:** Hit rate measures whether the underlying moves 5% in the expected direction within 14 days — it does not account for option premium decay, IV crush, or actual P&L.

---

## REST API

### Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/v1/health/` | GET | ❌ | Health check |
| `/api/v1/signals/` | GET | ✅ | List signals (paginated) |
| `/api/v1/signals/latest/` | GET | ✅ | Latest signal (full detail) |
| `/api/v1/signals/<date>/` | GET | ✅ | Signal by date (single object; `?all=true` for list) |
| `/api/v1/signals/<date>/setup/` | GET | ✅ | **Trade setup for date** (supports `?type=` override) |
| `/api/v1/signals/latest/setup/` | GET | ✅ | **Trade setup for latest tradeable signal** |
| `/api/v1/options/predict/` | POST | ✅ | **Predict option price under BTC scenarios** |
| `/api/v1/fusion/explain/` | GET | ✅ | Explain fusion logic |

### Authentication

```bash
# Health check (no auth)
curl http://localhost:8000/api/v1/health/

# Get latest signal
curl -H "Authorization: Token YOUR_TOKEN" \
     http://localhost:8000/api/v1/signals/latest/

# Get trade setup for latest tradeable signal
curl -H "Authorization: Token YOUR_TOKEN" \
     http://localhost:8000/api/v1/signals/latest/setup/

# Get trade setup for specific date
curl -H "Authorization: Token YOUR_TOKEN" \
     http://localhost:8000/api/v1/signals/2026-05-02/setup/

# Predict option price under different BTC scenarios
curl -X POST -H "Authorization: Token YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"current_spot": 80800, "strike": 79000, "option_type": "put", "dte": 8, "current_premium": 720, "scenarios": [77000, 75000, 83000]}' \
     http://localhost:8000/api/v1/options/predict/
```

### Trade Setup Response

```json
{
  "signal_date": "2026-05-02",
  "signal_type": "MVRV_SHORT",
  "direction": "SHORT",
  "spot_price": 78652.94,
  "expiry": "2026-05-15",
  "dte": 12,
  "legs": {
    "long": {
      "symbol": "BTC-15MAY26-80000-P",
      "action": "BUY",
      "strike": 80000.0,
      "delta": -0.5814,
      "iv": 0.3531,
      "price": 2757.48,
      "open_interest": 257
    },
    "short": {
      "symbol": "BTC-15MAY26-76000-P",
      "action": "SELL",
      "strike": 76000.0,
      "delta": -0.2917,
      "iv": 0.3789,
      "price": 1024.21,
      "open_interest": 541
    }
  },
  "metrics": {
    "spread_width": 4000.0,
    "net_debit": 1733.27,
    "max_profit": 2266.73,
    "max_loss": 1733.27,
    "risk_reward": 1.31,
    "net_edge_pct": 0.72
  },
  "exit_rules": {
    "stop_loss_spot": 83474.36,
    "stop_loss_value": 693.31,
    "take_profit_pct": 0.6,
    "max_hold_days": 11,
    "scale_down_day": 5,
    "scale_down_action": "close_full_position"
  },
  "validation": {
    "passed": true,
    "warnings": ["..."],
    "blocking": []
  },
  "policy_version": "2026-05-03.3"
}
```

### Option Price Prediction

Predict how option prices will move under different BTC scenarios using a learned model trained on 85k+ historical option snapshots.

**Request:**
```bash
curl -X POST -H "Authorization: Token YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
       "current_spot": 80800,
       "strike": 79000,
       "option_type": "put",
       "dte": 8,
       "current_premium": 720,
       "scenarios": [77000, 75000, 73000, 83000, 85000],
       "iv": 0.45
     }' \
     http://localhost:8000/api/v1/options/predict/
```

**Response:**
```json
{
  "current": {
    "spot": 80800,
    "strike": 79000,
    "option_type": "put",
    "dte": 8,
    "moneyness_pct": -2.23,
    "moneyness_label": "slightly OTM",
    "intrinsic": 0
  },
  "scenarios": [
    {
      "btc_price": 77000,
      "btc_change_pct": -4.7,
      "moneyness_label": "ITM",
      "predicted_return": {
        "p10": -41.0,
        "p50": -4.4,
        "p90": 35.8
      },
      "estimated_premium": {
        "conservative": 425,
        "base": 689,
        "optimistic": 978
      },
      "intrinsic_value": 2000,
      "black_scholes": {
        "iv_40": 2853,
        "iv_50": 3251,
        "iv_60": 3659
      }
    }
  ],
  "model_info": {
    "model_loaded": true,
    "buckets_available": 847
  }
}
```

**Use case:** Paper trading entry optimization. Before entering a position, simulate adverse scenarios to understand downside risk and find better entry points.

---

## Commands

### Daily Operations

```bash
# Build feature CSV from raw data
python manage.py build_features

# Generate today's signal with Telegram notification
python manage.py generate_signal --notify

# Generate signal without trade setup in notification
python manage.py generate_signal --notify --no-setup

# Generate signal without persistence (dry run)
python manage.py generate_signal --verbose --no-persist

# Generate signal for specific trade type (multi-signal days)
python manage.py generate_signal --type MVRV_SHORT --notify
```

### Execution

```bash
# Execute today's signal (ALWAYS dry-run first!)
python manage.py execute_deribit --latest --dry-run
python manage.py execute_deribit --latest

# Execute specific signal type on multi-signal days
python manage.py execute_deribit --latest --type MVRV_SHORT --dry-run

# Check position status
python manage.py sync_positions --all

# Check for unprotected positions
python manage.py check_protection

# Manually trigger exit check
python manage.py manage_exits --dry-run
python manage.py manage_exits
```

### Analysis

```bash
# Analyze historical path performance
python manage.py analyze_path_stats --target 0.05 --horizon 14

# Analyze hit rates by signal type
python manage.py analyze_hit_rate --type MVRV_SHORT

# Calibrate policy from path analysis
python manage.py calibrate_policy

# Diagnose why trades didn't fire
python manage.py diagnose_notrade --year 2025

# Recovery policy analysis (FLIP/HOLD/CUT recommendations)
python manage.py analyze_loser_recovery
python manage.py analyze_loser_recovery --type MVRV_SHORT
python manage.py analyze_loser_recovery --simulate-policy --sensitivity-analysis
```

### Manual Trade Setup (Income Strategies)

Generate trade setups for any signal type on demand — useful for income strategies and manual overrides:

```bash
# List all available signal types with policy parameters
python manage.py manual_setup --list

# Generate setup for a specific signal (uses real option chain)
python manage.py manual_setup --signal CALL
python manage.py manual_setup --signal IRON_CONDOR
python manage.py manual_setup --signal MVRV_SHORT

# Generate all setups at once
python manage.py manual_setup --all

# Use specific date
python manage.py manual_setup --signal PUT --date 2026-05-09

# Output as JSON (for programmatic use)
python manage.py manual_setup --signal CALL --json

# Show theoretical setup only (no option chain lookup)
python manage.py manual_setup --signal CALL --theoretical

# Show full option chain candidates
python manage.py manual_setup --signal CALL --show-chain

# Override spot price
python manage.py manual_setup --signal CALL --spot 105000
```

### Data Collection

```bash
# Collect options snapshots from Deribit
python manage.py collect_options --exchange deribit --dte-min 7 --dte-max 21

# Export collected data for analysis
python manage.py export_options --format csv

# Train option response model from snapshots
python manage.py train_option_response --horizon-days 1 --walk-forward-splits 5
```

### Training

```bash
# Train ML models (holdout validation)
python manage.py train_models

# Train for production
python manage.py train_models --production --mode hybrid --lag 1
```

---

## Overlays (Execution Modifiers)

Overlays modify **how hard** you press a trade. They never flip direction—only amplify or reduce conviction.

### Long Overlays

| Overlay | Trigger | Effect |
|---------|---------|--------|
| **Full Edge** (+2) | Fear stabilizing + MVRV undervalued | Size +25%, DTE +50% |
| **Partial Edge** (+1) | Sentiment OR MVRV favorable | Size +10% |
| **Moderate Veto** (-1) | Euphoria persisting | Size -50% |
| **Strong Veto** (-2) | Euphoria + MVRV overvalued | **No trade** |

### Short Overlays

| Score | Overlay | Effect |
|-------|---------|--------|
| ≥ 0.85 | Full Edge | Size +15%, DTE extended |
| ≥ 0.75 | Partial Edge | Size +5% |
| ≤ 0.35 | Soft Veto | Size -50% |
| ≤ 0.25 | Hard Veto | **No trade** |

---

## Iron Condor Range Gate

Monetizes NO_TRADE / TRANSITION_CHOP days by selling iron condors when BTC is likely to stay range-bound.

**How it works:** Scores each day 0-100 based on existing features (chop fusion state, MVRV neutral, flat sentiment, no directional signals, calm exchange flows). Score ≥ 75 + no hard vetoes → `IRON_CONDOR` fires.

**Label:** ±10% band, 7-day forward horizon. Base rate: 58% of days stay in range.

**Performance:** 63.1% hit rate (84 trades). 80.6% since 2023.

**Strategy:** 10% OTM wings, 7–14d DTE, 50% take-profit, 6.8% stop-loss.

### Strike Selection: MVRV Drift-Based

```
cost_basis = spot / mvrv_60d
drift = max(trailing_7d_mvrv) - min(trailing_7d_mvrv)

short_call = max(spot × 1.10, cost_basis × (mvrv + 1.5 × drift))
short_put  = min(spot × 0.90, cost_basis × (mvrv - 1.5 × drift))
```

---

## Income Spread Gates (Bull Put / Bear Call)

Directional credit spreads that monetize regime conviction by selling premium on the side the market is unlikely to move toward. They sit below the iron condor in the signal hierarchy.

**How it works:** Each gate scores on-chain conditions 0–100, applies hard vetoes, then filters the live option chain for tradable spreads. Two-layer evaluation: regime eligibility (on-chain) is independent of chain quality (market microstructure).

**Key principle:** These are NOT directional bets. They're income strategies that profit when price does NOT move against you. Bull put = "price won't fall below X." Bear call = "price won't rally above Y."

### Strike Boundaries (MVRV-Derived)

| Strategy | MVRV Used | Boundary | Rationale |
|----------|-----------|----------|-----------|
| Bull Put Spread | MVRV-60D | `floor = spot / mvrv_60d` | Cost basis of recent buyers = sell exhaustion level |
| Bear Call Spread | MVRV Composite | `ceiling = (spot / mvrv_composite) × P90_180d` | Broad profit-taking pressure caps upside |

### Scoring Components

**Bull Put Spread (sells puts below support in bullish regime):**

| Component | Condition | Points |
|-----------|-----------|--------|
| MDIA inflow | Fresh capital entering | +25 |
| Whale sponsored | Strategic accum (100-10K BTC) | +20 |
| MVRV macro bullish | Call confirm / recovery / trend | +20 |
| No option put signal | No conflicting bearish signal | +15 |
| Sentiment safe | Not extreme greed | +10 |
| Whale not distributing | No active selling | +10 |

**Bear Call Spread (sells calls above resistance in bearish regime):**

| Component | Condition | Points |
|-----------|-----------|--------|
| No MDIA inflow / aging | No fresh capital | +25 |
| Whale distribution | Smart money selling | +20 |
| MVRV macro bearish | Put confirm / bear continuation | +20 |
| No option call signal | No conflicting bullish signal | +15 |
| Sentiment greed/flat | Crowd complacent | +10 |
| Whale not accumulating | No buying pressure | +10 |

**Threshold:** 70/100 + no hard vetoes → regime eligible.

### Option Chain Filters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Delta | 0.15–0.30 | Short leg probability range |
| DTE | 9–21 (tactical) / 21–45 (income) | Time windows |
| Bid/ask spread | ≤ 15% | Liquidity filter |
| Credit/width | ≥ 15%* | Minimum premium collected |
| Spread width | 3–8% of spot | Position sizing |

*Default in code is 25%, but live BTC chains typically offer 14–18% at target deltas. Recommend 15% for production.

### Priority & Conflicts

```
Directional (CALL/PUT/TACTICAL_PUT/MVRV_SHORT) > IRON_CONDOR > Income Spreads
```

Income gates only fire when fusion = NO_TRADE / TRANSITION_CHOP, no directional signals, and condor did NOT pass.

### Usage

```bash
# Analyze hit rates for income strategies
python manage.py analyze_hit_rate --type BULL_PUT_SPREAD
python manage.py analyze_hit_rate --type BEAR_CALL_SPREAD

# Path diagnostics
python manage.py analyze_path_stats --type BULL_PUT_SPREAD
python manage.py analyze_path_stats --type BEAR_CALL_SPREAD
```

---

## MVRV Short Signal

A tactical short signal for bear market conditions (cycle days 540-900 post-halving). See [docs/mvrv_short_signal.md](docs/mvrv_short_signal.md) for full documentation.

**Quick Stats:**
- **Hit Rate:** 68.2% (22 trades)
- **Trigger:** MVRV 7d ≥ 1.02 + MVRV 60d ≥ 1.0
- **Strategy:** DCA (33% initial, 67% on +4% rise)
- **Target:** -4% within 5 days

```bash
# Check today's signal status
python manage.py analyze_mvrv_short --today
```

---

## Configuration

### Cooldown Settings

| Trade Type | Cooldown |
|------------|----------|
| CALL / PUT | 7 days |
| BULL_PROBE / BEAR_PROBE | 5 days |
| TACTICAL_PUT | 7 days |
| OPTION_CALL / OPTION_PUT | 5 days |
| MVRV_SHORT | 5 days |
| IRON_CONDOR | 5 days |
| BULL_PUT_SPREAD | 5 days |
| BEAR_CALL_SPREAD | 5 days |

### Environment Variables

```bash
# Database
DATABASE_URL=postgres://user:pass@host:5432/dbname

# Google Sheets
GSPREAD_SHEET_ID=your_sheet_id

# Telegram
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Deribit (primary exchange)
DERIBIT_API_KEY=your_client_id
DERIBIT_API_SECRET=your_client_secret

# Bybit (optional)
BYBIT_API_KEY=your_api_key
BYBIT_API_SECRET=your_api_secret
```

---

## Key Files

| File | Purpose |
|------|---------|
| `features/feature_builder.py` | Feature engineering (MDIA, MVRV, Whales, Sentiment) |
| `signals/fusion.py` | Market state classification engine |
| `signals/overlays.py` | Edge amplifiers and veto gates |
| `signals/options.py` | Option strategy, strikes, DTE, spread guidance |
| `signals/services.py` | SignalService for scoring + persistence |
| `signals/income_gate.py` | **Bull put spread & bear call spread income gates** |
| `signals/management/commands/manual_setup.py` | **Manual trade setup for any signal type** |
| `execution/services/policy.py` | Data-driven policy engine |
| `execution/services/trade_setup.py` | Automated trade construction |
| `execution/services/trade_validator.py` | 11 pre-flight validation checks |
| `execution/services/option_pricer.py` | **Learned option pricing model (85k+ snapshots)** |
| `execution/services/recovery.py` | **Recovery decision engine (FLIP/HOLD/CUT)** |
| `execution/exchanges/deribit.py` | Deribit API adapter |
| `notifications/notifier.py` | Telegram notifications with trade setups |
| `api/views.py` | REST API endpoints |

---

## Testing

```bash
# Run all tests
python manage.py test

# Run specific test modules
python manage.py test execution.tests_trade_setup
python manage.py test execution.tests
python manage.py test execution.tests_recovery
python manage.py test signals.tests
python manage.py test signals.tests_recovery_candidate
python manage.py test signals.tests_recovery_mfe_analysis
python manage.py test signals.tests_recovery_policy_validation
python manage.py test signals.tests_recovery_performance_regression

# Run with verbosity
python manage.py test --verbosity=2
```

---

## Deployment

See [deploy/PRODUCTION_SETUP.md](deploy/PRODUCTION_SETUP.md) for full VPS deployment guide.

### Quick Deploy

```bash
# On VPS
cd /var/www/app
git pull origin main
source venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py test --verbosity=1
sudo systemctl restart gunicorn
```

### Cron Jobs

```cron
# Hourly signal pipeline (UTC) - re-evaluates until tradeable signals fire
5 * * * * python manage.py refresh_sheet
10 * * * * python manage.py sync_sheets
13 * * * * python manage.py build_features
16 * * * * python manage.py generate_signal --notify

# Execution layer
* * * * * python manage.py check_protection
*/5 * * * * python manage.py manage_exits
*/5 * * * * python manage.py sync_positions --all
0 * * * * python manage.py reconcile --all
```

> **Note:** Hourly signal generation allows the system to catch signals that qualify later in the day. New signal types can fire at any hour — the system is additive, not locked after first fire.

---

## Philosophy

1. **Terrain over timing**: MVRV-LS is structural, not a trade timer
2. **Whale sponsorship required**: No trade without smart money alignment
3. **Probes are smaller**: Macro-neutral trades use defined risk at 0.5x
4. **Overlays never override fusion**: They amplify or reduce, not flip
5. **Data-driven policy**: DTE, width, stops calibrated from path analysis
6. **Validation before execution**: 11 pre-flight checks catch silent drift
7. **Survive the shakeout**: SLIGHT_ITM strikes default across all states
8. **Three-layer exit**: Fixed stop + scale-down + hard time stop
9. **Net edge gate**: Trades must have ≥5% edge after costs
10. **Liquidity stress testing**: OI must support position size

---

## Documentation

| Document | Purpose |
|----------|---------|
| [docs/entry_policy_design.md](docs/entry_policy_design.md) | Policy calibration from path analysis |
| [docs/mvrv_short_signal.md](docs/mvrv_short_signal.md) | MVRV Short signal specification |
| [docs/iron_condor_spec.md](docs/iron_condor_spec.md) | Iron condor specification |
| [docs/options_data_leverage_plan.md](docs/options_data_leverage_plan.md) | Options data collection roadmap |
| [docs/recovery_decision_engine.md](docs/recovery_decision_engine.md) | Recovery decision engine (FLIP/HOLD/CUT) |
| [deploy/PRODUCTION_SETUP.md](deploy/PRODUCTION_SETUP.md) | VPS deployment guide |
| [execution/docs/execution_design.md](execution/docs/execution_design.md) | Execution layer design |

---

*On-chain regime classification for BTC options trading.*
