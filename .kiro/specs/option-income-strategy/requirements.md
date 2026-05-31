# Requirements Document: Option Income Strategy Module

## Introduction

The Option Income Strategy module extends the existing BTC options trading assistant to support income-generating option strategies for users who are long-term BTC holders. The module has **two primary functions**:

1. **Signal Generation**: Generate income strategy signals (CSP, credit spreads) using existing metrics (MVRV, IV rank, fusion state) to expand signal variety beyond directional-only signals
2. **Safe Execution**: Validate, execute atomically, and track income strategy positions with strict safety constraints

The module integrates with the existing signal generation infrastructure (`signals/` module) and reuses proven metrics while adding income-specific signal logic.

## Module Purpose and Integration

### Problem Statement

The current signal system generates **directional signals only** (OPTION_CALL, OPTION_PUT, MVRV_SHORT, IRON_CONDOR). This creates gaps:
- Months can pass without actionable signals in range-bound or low-conviction markets
- No signals for **premium collection** (selling puts/spreads for income)
- No signals for **BTC accumulation** (selling puts at favorable prices to synthetically accumulate)

### Solution

This module adds **income strategy signals** that fire when:
- Directional conviction is low but premium is attractive (high IV rank)
- Market conditions favor selling premium (range-bound, mean-reverting)
- Accumulation opportunities exist (undervalued MVRV + high IV = sell puts)

### Integration with Existing System

The module integrates with the existing signal infrastructure:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Existing Signal System                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  signals/fusion.py      → MarketState classification (13 states)            │
│  signals/options.py     → Option structure recommendations                   │
│  signals/condor_gate.py → Iron condor eligibility (ALREADY EXISTS)          │
│  signals/mvrv_short.py  → MVRV-based short signal                           │
│  signals/services.py    → SignalService orchestration                        │
│  signals/models.py      → DailySignal persistence                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     NEW: Income Strategy Module                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  Signal Generation (NEW):                                                    │
│    - CSP_ACCUMULATION signal (sell puts to accumulate BTC)                  │
│    - BULL_PUT_SPREAD signal (sell put spreads for income)                   │
│    - BEAR_CALL_SPREAD signal (sell call spreads for income)                 │
│                                                                              │
│  Safe Execution (NEW):                                                       │
│    - Strategy validation (protective legs, collateral)                       │
│    - Atomic/sequential-safe execution                                        │
│    - Position tracking and exit management                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### What This Module Does NOT Do

- **Does NOT replace iron condor signals**: `condor_gate.py` already handles iron condor eligibility. This module may extend execution/tracking but does not duplicate signal logic.
- **Does NOT replace directional signals**: OPTION_CALL, OPTION_PUT, MVRV_SHORT remain in existing modules.
- **Does NOT create new metrics**: Uses existing MVRV, IV rank, fusion state, sentiment metrics.

## Scope and Exchange Constraints

### Phase 1: Deribit Linear USDC Options Only

The initial implementation is restricted to **Deribit Linear USDC Options** only. This constraint exists because:

1. **European-style, cash-settled**: Deribit options are European (exercise at expiry only) and cash-settled in USDC. There is NO physical delivery of BTC, NO early assignment, and NO "receiving BTC" or "BTC called away" events.

2. **USDC settlement eliminates inverse collateral risk**: Linear USDC options are quoted, margined, and settled in USDC. This avoids the dangerous scenario where BTC-collateralized positions face mark-to-market pressure as BTC falls (collateral value drops while liability rises).

3. **Simpler margin model**: USDC-settled options have predictable collateral requirements without the convexity risk of inverse BTC options.

4. **Mature API and combo orders**: Deribit has well-documented APIs and supports atomic combo orders for multi-leg strategies.

| Property | Deribit Linear USDC |
|----------|---------------------|
| Exercise Style | European |
| Settlement Type | Cash-settled |
| Settlement Currency | USDC |
| Contract Size | 1 BTC |
| Physical Delivery | NO |
| Early Assignment | NO |
| Margin Model | Isolated/Portfolio |
| Collateral Types | USDC only |
| API | REST + WebSocket at `www.deribit.com` |

### Explicitly Out of Scope (Phase 1)

- **Deribit Inverse BTC Options**: Require separate collateral/liquidation modeling due to BTC-denominated margin
- **Thalex Options**: Deferred to Phase 2
- **Physical delivery options**: Not supported
- **Any exchange with American-style options**: Early assignment not modeled

### Phase 2: Thalex Integration

Phase 2 will add **Thalex USD Options** support. Thalex shares the same fundamental settlement model as Deribit Linear USDC (European, cash-settled), making integration straightforward:

| Property | Thalex |
|----------|--------|
| Exercise Style | European |
| Settlement Type | Cash-settled |
| Settlement Currency | USD |
| Contract Size | 1 BTC |
| Physical Delivery | NO |
| Early Assignment | NO |
| Margin Model | Portfolio (default) |
| Collateral Types | Multi-asset (USDT, BTC, ETH) |
| API | REST + WebSocket at `thalex.com/api/v2` |

**Thalex-specific considerations for Phase 2:**
- Multi-asset collateral with `collateral_factor` applied to non-USD assets
- Portfolio margin by default (more capital efficient)
- Combination orders support up to 4 legs
- Testnet available at `testnet.thalex.com`

### Future Phases

- Phase 2: Thalex integration (requires exchange adapter implementation)
- Phase 3: Deribit Inverse BTC Options (requires BTC collateral stress modeling)

## Glossary

- **Income_Strategy**: An option strategy designed to generate premium income while maintaining defined risk parameters
- **Accumulation_Strategy**: A strategy designed to synthetically accumulate BTC exposure at favorable prices via cash-settled options
- **Cash_Secured_Put**: A put option sold with full USDC collateral equal to max loss at expiry
- **Bear_Call_Spread**: A credit spread with a short call protected by a long call at a higher strike, providing defined-risk income generation (replaces naked "covered calls" which have undefined loss)
- **Credit_Spread**: A spread that receives net premium, with defined max loss equal to spread width minus credit
- **Debit_Spread**: A spread that pays net premium, with defined max profit equal to spread width minus debit
- **Iron_Condor**: A four-leg strategy combining a put spread and call spread, with defined max loss
- **Collateral_Requirement**: The USDC capital that must be locked to secure a position per exchange margin rules
- **Settlement_Risk**: The risk at expiry when cash settlement occurs (European options have no early exercise)
- **Capital_Efficiency**: The ratio of potential return to capital deployed
- **Market_Regime**: Current market conditions (bullish, bearish, range-bound, high IV, low IV)
- **Defined_Risk**: A position where maximum loss is known and capped at entry AND collateral is sufficient to cover that loss without liquidation
- **Naked_Option**: An uncovered short option with theoretically unlimited risk
- **Blow_Up_Risk**: Risk of catastrophic loss exceeding account value or forced liquidation
- **European_Option**: Option that can only be exercised at expiry (all Deribit options)
- **Cash_Settlement**: Settlement in currency (USDC) rather than physical delivery of underlying
- **Atomic_Execution**: Multi-leg order execution where all legs fill or none fill
- **Protective_Leg**: A long option that caps the loss on a corresponding short option

## Safety Constraints (Non-Negotiable)

### Constraint S1: No Naked Short Options
THE system SHALL NOT recommend, construct, or validate any strategy containing naked (uncovered) short calls or naked short puts without sufficient collateral or protective legs.

### Constraint S2: No Undefined Loss Exposure
THE system SHALL NOT recommend any strategy where maximum loss cannot be calculated at entry time.

### Constraint S3: No Leveraged Short Structures
THE system SHALL NOT recommend any strategy that uses margin to amplify short option exposure beyond available collateral.

### Constraint S4: Full Collateralization Required
THE system SHALL require collateral sufficient to cover maximum loss without liquidation:
- Cash-secured put: `collateral >= max_loss = (strike - 0) * contract_size * quantity` (worst case BTC → 0)
- Credit spread: `collateral >= spread_width * contract_size * quantity`

Note: Single-leg short calls ("synthetic covered calls") are NOT supported because max loss is undefined (unlimited upside).

### Constraint S5: Protective Legs Must Be Complete and Verified
THE system SHALL require every short option leg in a spread to have a corresponding protective long leg that meets ALL of the following:
- Same underlying asset (BTC)
- Same expiry date
- Same settlement currency (USDC)
- Same option type (call protects call, put protects put)
- Correct strike ordering (long put strike ≤ short put strike; long call strike ≥ short call strike)
- Equal or greater quantity on the long leg
- No uncovered ratio tail
- Long leg must be filled/owned BEFORE short leg order is placed (atomic execution or sequential with verification)

### Constraint S6: Explicit Rejection with Reason
THE system SHALL reject unsafe strategies with a clear, specific reason code and human-readable explanation.

### Constraint S7: Atomic or Sequential-Safe Execution
THE system SHALL ensure multi-leg strategies are executed atomically (combo order) OR sequentially with long-leg-first verification:
- If atomic combo orders are available: use them
- If not: buy protective long leg first, verify fill, then sell short leg up to filled protected quantity only
- On partial short overfill: immediately cancel remainder and buy-to-close excess
- On long leg rejection: abort entire strategy, do not place short leg

### Constraint S8: Real Account State Validation
THE system SHALL validate collateral and holdings against authenticated exchange account state, NOT user-supplied parameters. User-supplied values may be used for simulation/preview only, clearly marked as "unverified simulation."

### Constraint S9: No Physical Delivery Assumptions
THE system SHALL NOT assume physical BTC delivery, early assignment, or "receiving BTC" events. All Deribit options are European and cash-settled.

## Requirements

### Requirement 1: Strategy Classification Engine

**User Story:** As a trader, I want the system to classify option strategies into income vs accumulation categories so that I can select strategies aligned with my goals.

#### Acceptance Criteria

1. THE system SHALL classify strategies into two primary categories:
   - **Income Strategies**: Credit spreads (bull put, bear call), iron condors
   - **Accumulation Strategies**: Cash-secured puts (synthetic BTC accumulation via premium)
2. THE system SHALL identify the following strategy types (Deribit Linear USDC):
   - `CASH_SECURED_PUT`: Short put with full USDC collateral for max loss
   - `BULL_PUT_SPREAD`: Credit spread, bullish bias, defined risk
   - `BEAR_CALL_SPREAD`: Credit spread, bearish bias, defined risk (replaces naked covered calls)
   - `IRON_CONDOR`: Neutral credit strategy with defined wings
   - `PUT_DEBIT_SPREAD`: Defined-risk bearish play
   - `CALL_DEBIT_SPREAD`: Defined-risk bullish play
   
   **Explicitly NOT supported (undefined loss):**
   - ❌ `SYNTHETIC_COVERED_CALL`: Single-leg short call has unlimited upside loss
   - ❌ `CASH_SETTLED_WHEEL_CALL_PHASE`: Depends on naked short calls
3. THE system SHALL tag each strategy with attributes:
   - `is_income`: Boolean indicating premium-generating intent
   - `is_accumulation`: Boolean indicating synthetic BTC accumulation intent
   - `is_defined_risk`: Boolean (must be True AND verified for all strategies)
   - `is_fully_collateralized`: Boolean indicating collateral covers max loss without liquidation
   - `requires_usdc_collateral`: Boolean (True for all Phase 1 strategies)
   - `settlement_currency`: "USDC"
   - `exercise_style`: "european"
   - `settlement_type`: "cash" (no physical delivery)

### Requirement 2: Strategy Eligibility Validation

**User Story:** As a trader, I want the system to validate that a strategy meets all safety constraints before presenting it as an option.

#### Acceptance Criteria

1. THE system SHALL validate each strategy against the allowed universe:
   - ✅ Cash-secured puts (fully collateralized in USDC)
   - ✅ Defined-risk credit spreads (bull put, bear call) with verified protective legs
   - ✅ Debit spreads (put debit, call debit)
   - ✅ Iron condors (all short legs protected by verified long legs)
2. THE system SHALL reject strategies in the excluded universe:
   - ❌ Naked calls (including "synthetic covered calls") → Rejection code: `NAKED_CALL_PROHIBITED`
   - ❌ Naked puts without sufficient collateral → Rejection code: `NAKED_PUT_INSUFFICIENT_COLLATERAL`
   - ❌ Short straddles → Rejection code: `SHORT_STRADDLE_PROHIBITED`
   - ❌ Short strangles → Rejection code: `SHORT_STRANGLE_PROHIBITED`
   - ❌ Ratio spreads with uncovered tails → Rejection code: `UNCOVERED_RATIO_PROHIBITED`
   - ❌ Any leveraged short structure → Rejection code: `LEVERAGED_SHORT_PROHIBITED`
   - ❌ Inverse BTC options → Rejection code: `INVERSE_OPTIONS_NOT_SUPPORTED`
   - ❌ Non-USDC settlement → Rejection code: `NON_USDC_SETTLEMENT_NOT_SUPPORTED`
   - ❌ Cash-settled wheel (depends on naked calls) → Rejection code: `NAKED_CALL_PROHIBITED`
3. THE system SHALL verify protective leg completeness for spreads:
   - Same underlying: `long_leg.underlying == short_leg.underlying == "BTC"`
   - Same expiry: `long_leg.expiry == short_leg.expiry`
   - Same settlement: `long_leg.settlement_currency == short_leg.settlement_currency == "USDC"`
   - Same option type: `long_leg.option_type == short_leg.option_type`
   - Correct strike ordering:
     - Put spread: `long_leg.strike <= short_leg.strike`
     - Call spread: `long_leg.strike >= short_leg.strike`
   - Quantity coverage: `long_leg.quantity >= short_leg.quantity`
   - No ratio tail: `short_leg.quantity <= long_leg.quantity` (no excess short)
   - Rejection if any check fails: `PROTECTIVE_LEG_INVALID` with specific sub-code
4. THE system SHALL verify collateral sufficiency against AUTHENTICATED exchange state:
   - Cash-secured put: `exchange_usdc_balance >= strike * contract_size * quantity`
   - Credit spread: `exchange_usdc_balance >= spread_width * contract_size * quantity`
   - Rejection if insufficient: `INSUFFICIENT_COLLATERAL_VERIFIED`
5. WHEN a strategy fails validation, THE system SHALL return:
   - `is_valid`: False
   - `rejection_code`: Specific code from rejection list
   - `rejection_reason`: Human-readable explanation
   - `suggested_alternative`: Safe alternative if available
   - `validation_source`: "exchange_verified" or "simulation_unverified"

### Requirement 3: Payoff Calculator

**User Story:** As a trader, I want to see complete payoff metrics for any income strategy so that I can make informed decisions.

#### Acceptance Criteria

1. THE system SHALL calculate the following metrics for each strategy (all values in USDC):
   - `max_profit_usdc`: Maximum possible profit in USDC
   - `max_loss_usdc`: Maximum possible loss in USDC (must be defined and finite)
   - `breakeven_price`: BTC price(s) at which P&L = 0
   - `breakeven_prices`: List for multi-leg strategies
   - `risk_reward_ratio`: max_profit / max_loss
   - `probability_of_profit`: Estimated PoP based on delta
   - `expected_value_usdc`: Probability-weighted expected return in USDC
2. THE system SHALL calculate collateral requirements in USDC:
   - `collateral_required_usdc`: Total USDC capital locked per exchange margin rules
   - `buying_power_reduction_usdc`: Impact on available margin
   - `capital_efficiency`: max_profit / collateral_required (annualized)
   - `liquidation_buffer_pct`: Distance to liquidation price as % of spot
3. THE system SHALL calculate settlement risk metrics (European options, no early assignment):
   - `itm_probability_at_expiry`: Based on delta and time to expiry
   - `settlement_risk`: HIGH/MEDIUM/LOW based on moneyness and DTE
   - `max_settlement_payout_usdc`: Maximum cash settlement amount
4. THE system SHALL use explicit USDC denomination:
   - All premium values in USDC (converted from BTC-denominated quotes via spot price)
   - All collateral values in USDC
   - All P&L values in USDC
   - Conversion formula: `usdc_value = btc_value * spot_price`
5. THE system SHALL support payoff calculation for:
   - Single-leg strategies (cash-secured puts only)
   - Two-leg spreads (vertical spreads)
   - Four-leg strategies (iron condors)

### Requirement 4: Risk Engine

**User Story:** As a trader, I want the system to enforce risk limits and prevent any strategy that could cause account blow-up.

#### Acceptance Criteria

1. THE system SHALL enforce position-level risk limits:
   - `max_single_position_risk_pct`: Maximum % of account at risk per position (default: 5%)
   - `max_total_income_exposure_pct`: Maximum % of account in income strategies (default: 30%)
   - `max_settlement_exposure_pct`: Maximum % of account exposed to ITM settlement (default: 50%)
2. THE system SHALL enforce strategy-level constraints:
   - Credit spreads: `spread_width <= max_spread_width_usd` (default: $5,000)
   - Iron condors: Both wings must have protective legs
   - Cash-secured puts: Cannot exceed available USDC
3. THE system SHALL calculate and display:
   - `account_risk_pct`: Position risk as % of account
   - `margin_utilization_pct`: Collateral as % of available margin
   - `concentration_risk`: Exposure to single expiry/strike
4. THE system SHALL block execution with `RISK_LIMIT_EXCEEDED` when:
   - Position risk exceeds `max_single_position_risk_pct`
   - Total income exposure exceeds `max_total_income_exposure_pct`
   - Settlement exposure exceeds `max_settlement_exposure_pct`
5. THE system SHALL provide risk warnings (non-blocking) for:
   - High IV environment (IV rank > 80): `HIGH_IV_PREMIUM_WARNING`
   - Low IV environment (IV rank < 20): `LOW_IV_PREMIUM_WARNING`
   - Near-expiry positions (DTE < 7): `GAMMA_RISK_WARNING`
   - High delta positions (|delta| > 0.40): `DIRECTIONAL_RISK_WARNING`

### Requirement 5: Market Regime Recommendation Engine

**User Story:** As a trader, I want strategy recommendations based on current market conditions so that I can optimize my income generation.

#### Acceptance Criteria

1. THE system SHALL classify market regime into:
   - `BULLISH`: Uptrend, positive momentum
   - `BEARISH`: Downtrend, negative momentum
   - `RANGE_BOUND`: Sideways, low directional conviction
   - `HIGH_IV`: IV rank > 60, elevated premiums
   - `LOW_IV`: IV rank < 30, compressed premiums
2. THE system SHALL recommend strategies by regime:

   | Regime | Primary Recommendations | Secondary | Avoid |
   |--------|------------------------|-----------|-------|
   | BULLISH | Cash-secured puts, Bull put spreads | - | Bear call spreads |
   | BEARISH | Bear call spreads | Put debit spreads | Cash-secured puts |
   | RANGE_BOUND | Iron condors, Credit spreads | - | Directional plays |
   | HIGH_IV | Credit spreads, Iron condors | Cash-secured puts | Debit spreads |
   | LOW_IV | Debit spreads | - | Credit spreads |

3. THE system SHALL provide regime-specific guidance:
   - Strike selection (OTM distance based on regime)
   - DTE targeting (shorter in high IV, longer in low IV)
   - Position sizing adjustments
   - Exit timing recommendations
4. THE system SHALL integrate with existing fusion engine signals:
   - Use `fusion_state` from DailySignal for regime classification
   - Use `fusion_confidence` to adjust recommendation strength
   - Cross-reference with existing MVRV, overlay signals

### Requirement 6: Income Strategy Signal Generation

**User Story:** As a trader, I want the system to generate income strategy signals based on market regime classification so that I can sell premium in favorable conditions and avoid selling into breakouts.

#### Background

The strategy itself matters less than the **regime detection**. Premium selling performance is dominated by:
- Volatility regime (IV richness, IV vs RV spread)
- Trend vs chop classification
- Expansion vs compression detection
- Macro valuation state

The existing signal system generates directional signals (OPTION_CALL, OPTION_PUT, MVRV_SHORT) and iron condor signals (via `condor_gate.py`). This requirement adds **regime-driven income strategy signals** that fire when conditions favor selling premium.

#### 6.1 Regime Classification (5-Axis Model)

THE system SHALL classify market regime using 5 axes:

| Axis | States | Why It Matters |
|------|--------|----------------|
| **Direction** | Strong bullish, Mild bullish, Neutral, Mild bearish, Strong bearish | Income strategies hate violent trends |
| **IV Level** | Low (<30), Moderate (30-50), Elevated (50-70), Extreme (>70) | Premium richness |
| **IV vs RV Spread** | IV < RV (danger), IV ≈ RV (neutral), IV > RV (favorable) | Rich premium only helps if realized movement stays lower |
| **Trend vs Chop** | Strong trend, Mild trend, Chop/Mean-reversion, Compression, Expansion | Most important for condors and spreads |
| **Macro Valuation** | Deep value (<0.90), Undervalued (0.90-0.98), Fair (0.98-1.05), Overvalued (1.05-1.15), Extreme (>1.15) | Accumulation desirability |

#### 6.2 IV vs Realized Volatility (CRITICAL)

THE system SHALL compute IV vs RV spread:
- `iv_rv_spread = iv_current - realized_vol_7d`
- `iv_rv_ratio = iv_current / realized_vol_7d`

| Situation | IV vs RV | Good for Selling? |
|-----------|----------|-------------------|
| IV 60%, RV 35% | IV >> RV | ✅ Excellent |
| IV 60%, RV 55% | IV ≈ RV | ⚠️ Marginal |
| IV 60%, RV 80% | IV < RV | ❌ Dangerous |

THE system SHALL require `iv_rv_ratio > 1.2` for premium selling signals (IV at least 20% above realized).

#### 6.3 Trend vs Chop Detection

THE system SHALL classify trend regime:

| Regime | Characteristics | Premium Selling |
|--------|-----------------|-----------------|
| **Strong Trend** | Momentum persistent, directional | ❌ AVOID all premium selling |
| **Mild Trend** | Drift with pullbacks | ⚠️ Directional spreads only |
| **Chop/Mean-Reversion** | Range behavior, support/resistance | ✅ Ideal for condors |
| **Compression** | Narrowing range, low RV | ⚠️ Expansion risk coming |
| **Expansion** | Breakout, volatility spike | ❌ AVOID all premium selling |

THE system SHALL use existing metrics to detect trend regime:
- `fusion_state` for directional bias
- `atr_ratio` (7d ATR / 30d ATR) for expansion/compression
- `mvrv_ls_level_neutral` for chop detection
- `sent_is_flattening` for compression signals

#### 6.4 Regime Confidence Scores

THE system SHALL output confidence scores (0.0-1.0) for each regime classification, not binary states:

```
{
  "direction": {"state": "mild_bullish", "confidence": 0.72},
  "iv_regime": {"state": "elevated", "confidence": 0.85},
  "iv_rv_favorable": {"state": true, "confidence": 0.78},
  "trend_regime": {"state": "chop", "confidence": 0.65},
  "valuation": {"state": "undervalued", "confidence": 0.91}
}
```

THE system SHALL require minimum confidence thresholds for signal generation:
- Direction confidence ≥ 0.60
- Trend regime confidence ≥ 0.70 (most important)
- IV regime confidence ≥ 0.65

#### 6.5 Strategy Selection Logic

THE system SHALL select strategies based on regime combination:

| Regime Combination | Strategy | Rationale |
|--------------------|----------|-----------|
| Undervalued + Panic IV + Capitulation exhaustion | CSP_ACCUMULATION | Buy fear, accumulate cheap |
| Mild bullish + Rich IV + IV > RV + Support intact | BULL_PUT_SPREAD | Drift up, collect premium |
| Mild bearish + Rich IV + IV > RV + Resistance confirmed | BEAR_CALL_SPREAD | Drift down, collect premium |
| Chop + Rich IV + Low RV + No expansion signal | IRON_CONDOR | Range-bound, sell both sides |
| **Strong trend (either direction)** | **NO PREMIUM SELLING** | Trend will run you over |
| **Expansion regime** | **NO PREMIUM SELLING** | Breakout will destroy spreads |

#### 6.6 CSP_ACCUMULATION Signal Conditions

THE system SHALL generate `CSP_ACCUMULATION` when ALL conditions are met:

**Required:**
- Macro valuation: `mvrv_60d < 0.95` (undervalued)
- IV elevated: `iv_rank > 0.50`
- IV > RV: `iv_rv_ratio > 1.2`
- Direction: Neutral to mild bullish OR capitulation exhaustion
- Trend regime: NOT strong trend down, NOT expansion

**Preferred (boost confidence):**
- Panic IV spike (IV rank > 0.80)
- Local capitulation (sentiment extreme fear)
- Oversold conditions
- Long-term accumulation zone (MVRV < 0.90)

**Avoid (reduce confidence or veto):**
- Strong momentum breakdown
- Volatility expansion downward
- Cascading deleveraging signals
- Whale distribution active

#### 6.7 BULL_PUT_SPREAD Signal Conditions

THE system SHALL generate `BULL_PUT_SPREAD` when ALL conditions are met:

**Required:**
- Direction: Mild bullish or neutral bullish (NOT strong bullish)
- IV elevated: `iv_rank > 0.40`
- IV > RV: `iv_rv_ratio > 1.2`
- Trend regime: Chop or mild trend up (NOT expansion)
- Support structure intact

**Preferred (boost confidence):**
- Bullish drift pattern
- Support holding on pullbacks
- IV significantly overpriced vs RV

**Avoid (reduce confidence or veto):**
- Trend breakdown
- Macro uncertainty events
- Expansion phase
- Strong bearish momentum

#### 6.8 BEAR_CALL_SPREAD Signal Conditions

THE system SHALL generate `BEAR_CALL_SPREAD` when ALL conditions are met:

**Required:**
- Direction: Mild bearish or neutral bearish (NOT strong bearish)
- IV elevated: `iv_rank > 0.40`
- IV > RV: `iv_rv_ratio > 1.2`
- Trend regime: Chop or mild trend down (NOT expansion)
- Resistance confirmed

**Preferred (boost confidence):**
- Resistance rejection
- Weak momentum / failed breakout
- Distribution phase
- Declining trend

**Avoid (reduce confidence or veto):**
- Short squeeze environment
- Strong momentum uptrend
- Breakout expansion
- Bullish whale accumulation

**WARNING:** BTC upside squeezes are brutal. Bear call spreads require higher confidence threshold (0.75) than bull put spreads (0.65).

#### 6.9 IRON_CONDOR Signal Conditions

Iron condors need the **strictest filtering** because they lose on both breakout directions.

THE system SHALL generate `IRON_CONDOR` when ALL conditions are met:

**Required:**
- Trend regime: Strong chop (confidence ≥ 0.80)
- IV elevated: `iv_rank > 0.50`
- IV > RV: `iv_rv_ratio > 1.3` (stricter than spreads)
- Realized volatility: Low (below 30-day median)
- Price inside stable range
- No imminent expansion signal

**Hard Vetoes (instant block):**
- Expansion regime detected
- Strong trend (either direction)
- Post-compression release pattern
- Macro catalyst nearby (halving, ETF decision, etc.)
- ATR expansion (7d ATR / 30d ATR > 1.3)

**Note:** Iron condor logic already exists in `condor_gate.py`. This requirement extends it with IV vs RV checks.

#### 6.10 Global Vetoes (Apply to ALL Income Signals)

THE system SHALL NOT generate any income signal when:
- **Strong trend detected** (direction confidence > 0.80 for bullish or bearish)
- **Expansion regime** (ATR ratio > 1.5 or gap > 4%)
- **IV < RV** (realized vol exceeding implied = danger)
- **Existing directional signal active** (OPTION_CALL, OPTION_PUT, MVRV_SHORT)
- **IV rank < 0.30** (premium too cheap to justify risk)

#### 6.11 Strike and DTE Guidance

THE system SHALL provide regime-adjusted strike and DTE guidance:

| Signal | Strike Guidance | DTE Range | Spread Width |
|--------|-----------------|-----------|--------------|
| CSP_ACCUMULATION | 10-15% OTM (accumulation price) | 14-30d | N/A |
| BULL_PUT_SPREAD | Short: 5-10% OTM, Long: 15-20% OTM | 14-21d | 10% of spot |
| BEAR_CALL_SPREAD | Short: 7-12% OTM, Long: 17-22% OTM | 14-21d | 10% of spot |

**Note:** Bear call spread strikes are wider (more OTM) due to BTC's upside squeeze risk.

#### 6.12 Signal Performance Tracking

THE system SHALL track signal performance by regime:
- Win rate per signal type
- Win rate per regime combination
- Average premium collected
- Average settlement paid (for CSP)
- Regime accuracy (did predicted regime hold?)

### Requirement 7: Cash-Secured Put Income Tracking

**User Story:** As a long-term BTC holder, I want to track my cash-secured put income strategy performance over time.

#### Acceptance Criteria

1. THE system SHALL track cash-secured put strategy metrics:
   - `total_premium_collected_usdc`: Lifetime premium collected in USDC
   - `total_settlement_paid_usdc`: Total cash settlement paid when puts expire ITM
   - `net_income_usdc`: Premium collected minus settlement paid
   - `annualized_yield_pct`: Net income / capital deployed (annualized)
2. THE system SHALL handle cash settlement events correctly:
   - Short put expires ITM: **Pay** cash settlement = `(strike - spot) * size` (this is a LOSS)
   - Short put expires OTM: Keep full premium (this is a GAIN)
3. THE system SHALL clearly distinguish:
   - "Cash settlement paid" vs "BTC assigned" (the latter does NOT happen on Deribit)
   - Premium income vs settlement losses
4. THE system SHALL provide position history:
   - Track each CSP position opened/closed
   - Track premium received per position
   - Track settlement outcome per position
   - Calculate running P&L

**Note:** The traditional "wheel strategy" (rotating between puts and covered calls) is NOT supported because the call phase requires naked short calls which have undefined loss. Users wanting call-side income should use bear call spreads instead.

### Requirement 8: Integration with Existing Trade Setup

**User Story:** As a developer, I want the income strategy module to integrate seamlessly with the existing trade setup infrastructure.

#### Acceptance Criteria

1. THE system SHALL extend `TradeSetup` dataclass to support income strategies:
   - Add `strategy_category`: "income" | "accumulation" | "directional"
   - Add `collateral_type`: "usdc"
   - Add `settlement_currency`: "USDC"
   - Add `exercise_style`: "european"
   - Add `settlement_handling`: Instructions for cash settlement scenarios
2. THE system SHALL integrate with existing `TradeSetupBuilder`:
   - Add `build_income_setup()` method
   - Reuse existing option snapshot selection logic
   - Reuse existing validation pipeline
   - Add income-specific validation (protective leg verification, collateral check)
3. THE system SHALL add new API endpoints:
   - `GET /api/v1/income/strategies/` - List available income strategies
   - `GET /api/v1/income/recommend/` - Get regime-based recommendations (simulation mode)
   - `POST /api/v1/income/setup/` - Build income strategy setup
   - `GET /api/v1/income/csp/status/` - Get CSP income tracking status
   - `POST /api/v1/income/validate/` - Validate strategy against AUTHENTICATED exchange state
4. THE system SHALL clearly separate simulation vs verified modes:
   - Simulation endpoints accept user-supplied balances, marked as "unverified"
   - Execution endpoints MUST use authenticated exchange state
   - API responses include `validation_mode`: "simulation" | "exchange_verified"
5. THE system SHALL persist income strategy data:
   - Extend `ExecutionIntent` for income strategy tracking
   - Add `IncomePosition` model for active income positions
   - Add `CSPIncomeTracker` model for CSP income tracking

### Requirement 9: Rejection Reason System

**User Story:** As a trader, I want clear explanations when a strategy is rejected so that I can understand the risk and find alternatives.

#### Acceptance Criteria

1. THE system SHALL define rejection codes:

   | Code | Severity | Description |
   |------|----------|-------------|
   | `NAKED_CALL_PROHIBITED` | BLOCKING | Naked calls have unlimited loss potential |
   | `NAKED_PUT_INSUFFICIENT_COLLATERAL` | BLOCKING | Put requires more collateral than available |
   | `SHORT_STRADDLE_PROHIBITED` | BLOCKING | Undefined risk on both sides |
   | `SHORT_STRANGLE_PROHIBITED` | BLOCKING | Undefined risk on both sides |
   | `UNCOVERED_RATIO_PROHIBITED` | BLOCKING | Uncovered tails have unlimited risk |
   | `LEVERAGED_SHORT_PROHIBITED` | BLOCKING | Leverage amplifies blow-up risk |
   | `INSUFFICIENT_COLLATERAL_VERIFIED` | BLOCKING | Exchange-verified insufficient USDC |
   | `EXCHANGE_VERIFICATION_REQUIRED` | BLOCKING | Execution requires authenticated exchange state |
   | `PROTECTIVE_LEG_INVALID` | BLOCKING | Protective leg fails validation |
   | `PROTECTIVE_LEG_EXPIRY_MISMATCH` | BLOCKING | Long and short leg expiries differ |
   | `PROTECTIVE_LEG_TYPE_MISMATCH` | BLOCKING | Long and short leg option types differ |
   | `PROTECTIVE_LEG_SETTLEMENT_MISMATCH` | BLOCKING | Settlement currencies differ |
   | `PROTECTIVE_LEG_QUANTITY_INSUFFICIENT` | BLOCKING | Long leg quantity < short leg quantity |
   | `PROTECTIVE_LEG_STRIKE_INVALID` | BLOCKING | Strike ordering incorrect for spread type |
   | `INVERSE_OPTIONS_NOT_SUPPORTED` | BLOCKING | Inverse BTC options not supported |
   | `NON_USDC_SETTLEMENT_NOT_SUPPORTED` | BLOCKING | Only USDC settlement supported |
   | `PARTIAL_FILL_NAKED_EXPOSURE` | BLOCKING | Partial fill would create naked position |
   | `LONG_LEG_NOT_FILLED` | BLOCKING | Cannot place short leg without filled long leg |
   | `RISK_LIMIT_EXCEEDED` | BLOCKING | Position exceeds risk parameters |
   | `SPREAD_WIDTH_EXCEEDED` | BLOCKING | Spread wider than max allowed |
   | `HIGH_SETTLEMENT_RISK` | WARNING | ITM position approaching expiry |
   | `LOW_PREMIUM_EFFICIENCY` | WARNING | Premium too low for risk taken |
   | `SIMULATION_UNVERIFIED` | WARNING | Balances not verified against exchange (simulation only) |

2. THE system SHALL provide for each rejection:
   - `code`: Machine-readable rejection code
   - `severity`: BLOCKING or WARNING
   - `message`: Human-readable explanation
   - `details`: Specific values that triggered rejection
   - `alternative`: Suggested safe alternative (if available)
   - `validation_source`: "exchange_verified" or "simulation"

### Requirement 10: Position Sizing for Income Strategies

**User Story:** As a trader, I want appropriate position sizing that balances income generation with risk management.

#### Acceptance Criteria

1. THE system SHALL calculate position size based on:
   - Available USDC collateral
   - Risk budget per position
   - Maximum settlement exposure
   - Concentration limits
2. THE system SHALL enforce sizing constraints:
   - Single position: max 5% of account at risk
   - Total income positions: max 30% of account
   - Single expiry concentration: max 20% of income exposure
3. THE system SHALL provide sizing recommendations:
   - `recommended_contracts`: Optimal number of contracts
   - `max_contracts`: Maximum allowed by risk limits
   - `min_contracts`: Minimum for meaningful premium
   - `sizing_rationale`: Explanation of sizing decision

### Requirement 11: Exit Management for Income Strategies

**User Story:** As a trader, I want clear exit rules for income strategies including profit targets, stop losses, and roll decisions.

#### Acceptance Criteria

1. THE system SHALL define exit rules per strategy type:
   - Credit spreads: Close at 50% profit or 200% loss
   - Iron condors: Close at 50% profit, manage at 21 DTE
   - Cash-secured puts: Accept settlement or roll
2. THE system SHALL provide roll recommendations:
   - When to roll (DTE threshold, profit threshold)
   - Roll direction (same strike, different strike)
   - Roll timing (before/after expiry)
3. THE system SHALL treat every roll as a fully revalidated close-and-open sequence:
   - Close existing position (full validation)
   - Open new position (full validation including collateral, protective legs)
   - No "roll" shortcut that bypasses safety validation
4. THE system SHALL track exit metrics:
   - `days_held`: Time in position
   - `current_pnl_usd`: Mark-to-market P&L in USD
   - `pnl_pct_of_max`: Current P&L as % of max profit
   - `exit_recommendation`: HOLD, CLOSE, ROLL

### Requirement 12: Atomic Execution for Multi-Leg Strategies

**User Story:** As a trader, I want multi-leg strategies to be executed atomically to prevent partial fills that create naked exposure.

#### Acceptance Criteria

1. THE system SHALL prefer atomic combo orders when available:
   - Use Deribit combo/spread order types if supported
   - All legs fill or none fill
   - No partial naked exposure possible
2. WHEN atomic orders are not available, THE system SHALL use sequential-safe execution:
   - Step 1: Place buy order for protective long leg
   - Step 2: Wait for long leg fill confirmation
   - Step 3: IF long leg filled: place sell order for short leg, quantity ≤ long leg filled quantity
   - Step 4: IF long leg rejected/unfilled: abort, do not place short leg
3. THE system SHALL handle partial fill scenarios:
   - Short leg overfill beyond protected quantity: immediately cancel remainder, buy-to-close excess
   - Long leg partial fill: only sell short leg up to filled long quantity
   - Any naked exposure detected: emergency close with market order
4. THE system SHALL track leg fill state:
   - `long_leg_filled_qty`: Verified filled quantity of protective leg
   - `short_leg_max_qty`: Cannot exceed long_leg_filled_qty
   - `naked_exposure_detected`: Boolean, triggers emergency close if True
5. THE system SHALL implement `DeribitExecutor`:
   - Extend `_execute_legs_atomic()` for Deribit combo orders
   - MUST enforce long-leg-first for income strategies
   - MUST implement `_verify_protective_coverage()` before placing short leg
   - MUST implement `_emergency_close_naked_exposure()` for partial fill recovery
