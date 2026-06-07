# Metric Definitions & Price Action Correlation

## On-Chain Supply Metrics

### Strategic Whale Balance

**Definition:** The amount of Bitcoin held by addresses containing 100–10,000 BTC. This cohort represents large, sophisticated market participants ("Bitcoin millionaires").

**Role in Fusion:** Intent/sponsorship — determines whether smart money is backing a move.

**Price Action Correlation:**
- **Positive slope (rising):** Whales are accumulating — historically precedes or accompanies bullish price action. Large holders tend to buy during weakness and hold through strength.
- **Negative slope (falling):** Whales are distributing — often a leading indicator of price tops or sustained selling pressure.

Strategic whale behavior carries the highest weight among supply metrics. Their accumulation or distribution is a primary input for determining whether a move has "sponsorship."

---

### Small Whale Balance

**Definition:** The amount of Bitcoin held by addresses containing 1–100 BTC. Represents advanced retail and smaller high-conviction investors.

**Price Action Correlation:**
- **Positive slope (rising):** Smaller holders are accumulating — however, this cohort tends to buy rallies and sell dips. Their accumulation often coincides with sell-the-rally conditions rather than confirming bullish continuation.
- **Negative slope (falling):** Smaller holders are reducing exposure — can signal capitulation or waning retail conviction.

**Weight:** Lower than Strategic Whales. Small whale behavior is confirmatory, not primary. Their accumulation alone does not validate a bullish thesis — it needs strategic whale alignment.

---

### Exchange Flow Balance

**Definition:** Net difference between Bitcoin flowing into exchanges and Bitcoin flowing out.

**Price Action Correlation:**
- **Positive value:** More BTC entering exchanges than leaving — increases liquid supply available for selling. Bearish pressure.
- **Negative value:** More BTC leaving exchanges than entering — supply being removed from markets into self-custody. Bullish (supply squeeze).
- **Positive slope (becoming more positive):** Selling pressure is accelerating — bearish.
- **Negative slope (becoming more negative):** Withdrawal trend is strengthening — bullish, supply tightening.

---

## Valuation / Profitability Metrics

### MVRV

**Definition:** Market Value divided by Realized Value. Measured across multiple timeframes: 1d, 7d, 30d, 60d, 90d, 180d, 365d. It is a bounded metric — it oscillates within historical ranges rather than trending indefinitely.

**Price Action Correlation:**
- **Extreme high:** Does NOT necessarily mean cycle top — but it signals elevated sell pressure. Holders in significant profit are more likely to take profits, creating headwinds.
- **Extreme low:** Does NOT necessarily mean cycle bottom — but it signals sell exhaustion. Holders at or below cost basis have less incentive to sell.
- **Positive slope:** Unrealized profits are growing — price is rising faster than the aggregate cost basis. Bullish momentum, but watch for overextension.
- **Negative slope:** Unrealized profits are shrinking — price is falling toward or below cost basis. Bearish momentum.

**Key nuance:** MVRV is a pressure gauge, not a binary top/bottom signal. Extremes indicate where pressure builds, not where reversals are guaranteed.

---

### MVRV 60D USD

**Definition:** MVRV calculated only for coins that last moved within the previous ~60 days. Measures profitability of recent buyers ("new money").

**This is the best single metric for correlating with local tops and bottoms.** It is the strongest indicator of short-term price action.

**Price Action Correlation:**
- **High positive value:** Recent buyers are in significant profit — local top risk. Short-term selling pressure likely.
- **Negative value:** Recent buyers are underwater — local bottom signal. Capitulation or selling exhaustion.
- **Positive slope:** New entrants are becoming more profitable — confirms short-term bullish continuation.
- **Negative slope:** New entrants are losing money — short-term trend weakening, potential local reversal.

**Role in Fusion (Bear Mode):** MVRV 60D is the anchor for long entries in bear markets. The system maps it into buckets: `deep_underwater`, `underwater`, `breakeven`, `profitable` — and uses these to determine whether conditions favor bear exhaustion longs or continuation shorts.

---

### MVRV Composite

**Definition:** The average of multiple MVRV horizons (1d, 7d, 30d, 60d, 90d, 180d, 365d) combined into a single metric.

**Price Action Correlation:**
- **Negative value:** Most holder cohorts are at a loss — historically associated with recovery phases and strong buying opportunities.
- **Highly positive value:** Most cohorts are in profit — elevated profit-taking risk, often precedes corrections.
- **Positive slope:** Broad-based profitability is improving — bullish trend confirmed across timeframes.
- **Negative slope:** Profitability is deteriorating across cohorts — bearish, broad weakness.

---

### MVRV Long/Short Divergence

**Definition:** Difference between 365-day MVRV (long-term holders) and 60-day MVRV (recent buyers). Computed as a z-score over a 365-day rolling window.

**Role in Fusion:** Macro confirmation — determines whether the market is structurally ready for a move.

**Price Action Correlation:**
- **Positive value:** Long-term holders are more profitable than recent buyers — typical bull market state.
- **Negative value:** Recent buyers are more profitable than long-term holders — typical bear/sideways state.
- **Extreme low (z-score ≤ -1.5):** Capitulation signal. Long-term holders are deeply underwater relative to short-term participants. Historically marks major bottoms and recovery zones.
- **Positive slope:** The gap is widening in favor of long-term holders — trend is intact and accelerating. Bullish continuation.
- **Negative slope:** The gap is narrowing or reversing — trend maturation or early distribution.

**Regime Buckets (from code):**
| Z-Score | Level | Interpretation |
|---------|-------|----------------|
| ≤ -1.5 | -2 | Capitulation / Deep Value |
| -1.5 to -0.5 | -1 | Late Bear / Early Recovery |
| -0.5 to 0.5 | 0 | Neutral |
| 0.5 to 1.5 | 1 | Bull / Expansion |
| ≥ 1.5 | 2 | Overheated / Distribution Risk |

---

## Sentiment & Activity Metrics

### Weighted Sentiment

**Definition:** Social sentiment metric combining positive/negative commentary with discussion volume.

**Price Action Correlation (contrarian indicator):**
- **Highly positive value:** Market is euphoric — historically precedes pullbacks. Crowd is overly bullish.
- **Highly negative value:** Market is fearful — historically precedes recoveries. Crowd is overly bearish.

**Divergence Rules (sentiment vs. price action):**
- **Price drops + sentiment starts rising:** Bearish. The crowd is buying the dip prematurely — more downside likely. Rising optimism during a decline signals denial, not recovery.
- **Price rises + sentiment stays flat/neutral:** Bullish. The rally lacks crowd participation — no euphoria to unwind. Wall of worry = continuation.
- **Price rises + sentiment spikes positive:** Caution. Crowd is chasing — local top risk.
- **Price drops + sentiment stays flat/neutral:** Neutral to bearish. No panic = no capitulation yet.

**Key nuance:** Sentiment works best as a contrarian signal at extremes AND as a divergence signal against price. Moderate readings in isolation have less predictive value.

---

### Mean Dollar Invested Age (MDIA)

**Definition:** The average age of all invested dollars within the Bitcoin network. Measures how long capital has remained dormant.

**Role in Fusion:** Timing/impulse — determines whether fresh capital is entering NOW.

**Core Principle:** MDIA falling = capital inflow (bullish). MDIA rising = aging (trend-unfriendly).

**Price Action Correlation:**
- **Rising MDIA (positive slope):** Coins are aging — no fresh capital entering the network. Trend-unfriendly. Rallies without capital inflow lack fuel and tend to stall or reverse. The steeper the aging slope, the worse for price continuation.
- **Falling MDIA (negative slope):** Old coins are moving, fresh capital is entering — bullish impulse. Dormant capital re-entering circulation signals renewed activity and historically accompanies strong price moves.

**Regime Classification (from code):**

The system uses 4d and 7d slope z-scores as the primary decision horizons (highest predictive correlation). 2d is supportive only. 1d is ignored for regime decisions.

| Regime | Condition | Meaning |
|--------|-----------|---------|
| Strong Inflow | 4d or 7d z < -1.0, both negative, accelerating | High-conviction bullish impulse |
| Inflow | 7d z < -0.25 OR (4d + 2d both < -0.25) | Capital entering, supportive of price |
| Aging | 7d AND 4d z > 0.25 (persisted 2 days) | No fresh capital, trend-unfriendly |

**Important:** Aging does NOT mean bearish by itself. It means "trend-unfriendly." Whether aging translates to actual bearish price action depends on whale behavior and MVRV structure. MDIA provides the timing signal; whales and MVRV provide the directional conviction.

---

## Fusion Logic Summary

The system combines these metrics hierarchically:

| Metric | Role | Weight |
|--------|------|--------|
| MDIA | Timing/impulse (is fresh capital entering?) | Gate condition |
| Strategic Whales | Intent/sponsorship (is smart money backing it?) | Primary directional |
| MVRV Long/Short | Macro confirmation (is market structurally ready?) | Confirmer |
| MVRV 60D | Local valuation (bear mode anchor for longs) | Bear mode primary |
| Small Whales | Retail conviction (lower weight, often contrarian) | Secondary |
| Sentiment | Contrarian/divergence signal | Contextual |

**Key Fusion States:**
- **Strong Bullish:** Strong MDIA inflow + whale sponsorship + MVRV macro bullish
- **Early Recovery:** MDIA inflow + whale sponsorship + MVRV recovery (from capitulation)
- **Momentum Continuation:** MDIA inflow + whale support + MVRV macro bullish
- **Distribution Risk:** No MDIA inflow + whale distribution + MVRV not bullish
- **Bear Continuation:** No MDIA inflow + whale distribution + MVRV put/bear confirmed
