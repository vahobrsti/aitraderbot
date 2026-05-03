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

| Signal | Win% | Trades | DTE | Spread Width | Stop Loss | Delta |
|--------|------|--------|-----|--------------|-----------|-------|
| OPTION_CALL | 91.7% | 24 | 7-12d | 10% | 8.5% | 0.65 |
| OPTION_PUT | 77.8% | 18 | 5-10d | 12% | 7.0% | -0.65 |
| PRIMARY_SHORT | 75.7% | 37 | 8-12d | 9% | 4.5% | -0.60 |
| BULL_PROBE | 71.4% | 42 | 7-11d | 12% | 4.0% | 0.55 |
| LONG | 70.9% | 79 | 9-14d | 10% | 4.5% | 0.60 |
| MVRV_SHORT | 68.2% | 22 | 12-18d | 9% | 7.0% | -0.60 |
| IRON_CONDOR | 63.1% | 84 | 9-13d | 10% | 6.8% | 0.20 |
| BEAR_PROBE | 53.3% | 15 | 11-16d | 8% | 6.5% | -0.55 |
| TACTICAL_PUT | 50.0% | 24 | 8-12d | 7% | 3.5% | -0.40 |

**Key calibration formulas:**
- **DTE** = TTH p75 + 2 days buffer
- **Spread Width** = MFE p75 × 0.65-0.70
- **Stop Loss** = MAE(winners) p75 + buffer

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
| `/api/v1/signals/<date>/` | GET | ✅ | Signal by date |
| `/api/v1/signals/<date>/setup/` | GET | ✅ | **Trade setup for date** |
| `/api/v1/signals/latest/setup/` | GET | ✅ | **Trade setup for latest tradeable signal** |
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
```

### Execution

```bash
# Execute today's signal (ALWAYS dry-run first!)
python manage.py execute_deribit --latest --dry-run
python manage.py execute_deribit --latest

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
```

### Data Collection

```bash
# Collect options snapshots from Deribit
python manage.py collect_options --exchange deribit --dte-min 7 --dte-max 21

# Export collected data for analysis
python manage.py export_options --format csv
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
| `execution/services/policy.py` | Data-driven policy engine |
| `execution/services/trade_setup.py` | Automated trade construction |
| `execution/services/trade_validator.py` | 11 pre-flight validation checks |
| `execution/exchanges/deribit.py` | Deribit API adapter |
| `notifications/notifier.py` | Telegram notifications with trade setups |
| `api/views.py` | REST API endpoints |

---

## Testing

```bash
# Run all tests (370 tests)
python manage.py test

# Run specific test modules
python manage.py test execution.tests_trade_setup
python manage.py test execution.tests
python manage.py test signals.tests

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
# Daily signal pipeline (UTC)
5 0 * * * python manage.py refresh_sheet
10 0 * * * python manage.py sync_sheets
13 0 * * * python manage.py build_features
16 0 * * * python manage.py generate_signal --notify

# Execution layer
* * * * * python manage.py check_protection
*/5 * * * * python manage.py manage_exits
*/5 * * * * python manage.py sync_positions --all
0 * * * * python manage.py reconcile --all
```

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
| [deploy/PRODUCTION_SETUP.md](deploy/PRODUCTION_SETUP.md) | VPS deployment guide |
| [execution/docs/execution_design.md](execution/docs/execution_design.md) | Execution layer design |

---

*On-chain regime classification for BTC options trading.*
