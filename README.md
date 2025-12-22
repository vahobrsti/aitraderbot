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
│  datafeed/ → Raw on-chain data APIs        │
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
│                      EXECUTION LAYER                             │
│  list_trades command → Final trade opportunities with sizing     │
└─────────────────────────────────────────────────────────────────┘
```

## Market States

The fusion engine classifies each day into one of 8 states:

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

## Trade Types

| Type | Direction | Sizing | Strategy |
|------|-----------|--------|----------|
| LONG | 🟢 Long | 1.0x | Calls |
| BULL_PROBE | 🟢 Long | 0.35-0.60x | Call spread (defined risk) |
| PRIMARY_SHORT | 🔴 Short | 1.0x | Puts |
| BEAR_PROBE | 🔴 Short | 0.35-0.60x | Put spread (defined risk) |
| TACTICAL_PUT | 🔴 Put | 0.4-0.6x | Hedge inside bull regimes |

## Commands

### Core Commands

```bash
# Build feature CSV from raw data
python manage.py build_features

# List all trade opportunities
python manage.py list_trades --year 2024

# Score today's market state
python manage.py score_latest
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

### Training Commands

```bash
# Train ML models
python manage.py train_models
```

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

## Key Files

| File | Purpose |
|------|---------|
| `features/feature_builder.py` | Feature engineering (MDIA, MVRV, Whales, Sentiment) |
| `features/signals/fusion.py` | Market state classification engine |
| `features/signals/overlays.py` | Edge amplifiers and veto gates |
| `features/signals/tactical_puts.py` | Hedge puts inside bull regimes |
| `features/signals/options.py` | Option strategy selection |
| `features/management/commands/list_trades.py` | Trade opportunity listing |

## Sample Output

```
==========================================================================================
ALL TRADE OPPORTUNITIES (2024)
==========================================================================================

LONG:                  8
BULL_PROBE (0.5x):     5
PRIMARY_SHORT:         1
BEAR_PROBE (0.5x):     2
TACTICAL_PUT:          1
─────────────────────────
TOTAL:                17

------------------------------------------------------------------------------------------
Date         | Type           | Dir      | State                | Size  | Notes
------------------------------------------------------------------------------------------
2024-01-04   | BULL_PROBE     | 🟢 LONG  | bull_probe           | 0.35  | score=+2
2024-02-15   | LONG           | 🟢 LONG  | strong_bullish       | 1.00  | score=+4
2024-03-15   | PRIMARY_SHORT  | 🔴 SHORT | distribution_risk    | 1.00  | score=-1
2024-11-06   | LONG           | 🟢 LONG  | strong_bullish       | 1.00  | score=+5
...
```

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
```

## Philosophy

1. **Terrain over timing**: MVRV-LS is structural, not a trade timer
2. **Whale sponsorship required**: No trade without smart money alignment
3. **Probes are smaller**: Macro-neutral trades use defined risk at 0.5x
4. **Overlays never override fusion**: They amplify or reduce, not flip
5. **Clustering prevention**: 7-day cooldown collapses events into single trades

---

*Built for BTC options trading with on-chain metrics.*
