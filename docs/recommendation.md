# Signal Performance Recommendations

Based on backtest hit rates from 2017-2025 (350 trades, 67% overall).

## Signal Tiers

### Tier 1: High Confidence (75%+)
| Signal | Hit Rate | Action |
|--------|----------|--------|
| EARLY_RECOVERY | 83% | Full size |
| BEAR_CONTINUATION | 79% | Full size |

### Tier 2: Good Edge (60-70%)
| Signal | Hit Rate | Action |
|--------|----------|--------|
| MOMENTUM | 69% | Normal size |
| STRONG_BULLISH | 62% | Normal size |
| DISTRIBUTION_RISK | 61% | Normal size |
| BULL_PROBE | 61% | Normal size (score ≥+3) |
| BEAR_PROBE | 61% | Normal size (score ≤-3) |

### Tier 3: Marginal / Hedge Only
| Signal | Hit Rate | Action |
|--------|----------|--------|
| TACTICAL_PUT | 43% | Hedge only, small size |
| Weak probes (±2) | 40-52% | Skip or 0.35x size |

## Recommendations

1. **Trust EARLY_RECOVERY** — Your best signal at 83%. Size up on these.

2. **MOMENTUM degraded recently** — Was 69% overall but only 50% since 2021. Consider adding confirmation or reducing size in choppy markets.

3. **Probes need score filtering:**
   - Score ±3 or ±4: 65-100% hit rate → trade normally
   - Score ±2: ~45% hit rate → reduce size or skip

4. **Short signals are solid** — Both BEAR_CONTINUATION (79%) and DISTRIBUTION_RISK (61%) work. Don't be afraid to short.

5. **TACTICAL_PUT remains a hedge** — 43% hit rate doesn't justify as alpha signal, but useful as portfolio insurance.

## Position Sizing Guide

| Confidence | Signal Type | Size |
|------------|-------------|------|
| HIGH | EARLY_RECOVERY, BEAR_CONTINUATION | 1.0x |
| MEDIUM | MOMENTUM, core states, strong probes | 0.75x |
| LOW | Weak probes (±2) | 0.35x |
| HEDGE | TACTICAL_PUT | 0.40-0.60x |

## Key Insight

The fusion engine produces **real edge** — 67% hit rate over 8 years is strong. Focus sizing on conviction level, not just direction.
