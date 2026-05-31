# Implementation Plan: Option Income Strategy Module

## Overview

This implementation plan covers the development of the Option Income Strategy module for **Deribit Linear USDC Options only** (Phase 1). The module has **two primary functions**:

1. **Signal Generation**: Generate income strategy signals (CSP_ACCUMULATION, BULL_PUT_SPREAD, BEAR_CALL_SPREAD) using existing metrics
2. **Safe Execution**: Validate, execute atomically, and track income strategy positions with strict safety constraints

### Critical Design Constraints (from Review)

1. **Deribit Linear USDC Only**: All options are European-style, cash-settled in USDC. No physical BTC delivery, no early assignment.
2. **Atomic or Sequential-Safe Execution**: Multi-leg strategies require atomic combo orders OR long-leg-first sequential execution with verification.
3. **Exchange-Verified Collateral**: Execution requires authenticated exchange state, not user-supplied values.
4. **Complete Protective Leg Validation**: Same underlying, expiry, settlement, option type, correct strike ordering, equal-or-greater quantity.
5. **Integration with Existing Signals**: Reuses existing metrics (MVRV, IV rank, fusion state) and signal infrastructure.

## Phase 0: Income Signal Generator (NEW)

### Task 0: Regime Classification Engine

- [ ] 0.1 Create regime classification data structures
  - Create `signals/income_signals.py`
  - Define `DirectionState` enum (strong_bullish, mild_bullish, neutral, mild_bearish, strong_bearish)
  - Define `IVRegime` enum (low, moderate, elevated, extreme)
  - Define `TrendRegime` enum (strong_trend, mild_trend, chop, compression, expansion)
  - Define `ValuationState` enum (deep_value, undervalued, fair, overvalued, extreme)
  - Define `RegimeAxis` dataclass (state, confidence)
  - Define `RegimeClassification` dataclass (5 axes + computed metrics)
  - _Requirements: 6.1, 6.4_

- [ ] 0.2 Implement IV vs Realized Volatility computation (CRITICAL)
  - Implement `compute_iv_rv_metrics()` function
  - Compute `iv_rv_spread = iv_current - realized_vol_7d`
  - Compute `iv_rv_ratio = iv_current / realized_vol_7d`
  - Return `is_favorable = iv_rv_ratio > 1.2`
  - Add unit tests for edge cases (RV=0, IV<RV, IV>>RV)
  - _Requirements: 6.2_

- [ ] 0.3 Implement realized volatility calculation
  - Add `compute_realized_vol_7d()` function
  - Use existing OHLC data from RawDailyData
  - Calculate 7-day rolling standard deviation of returns
  - Annualize to match IV scale
  - Add unit tests
  - _Requirements: 6.2_

- [ ] 0.4 Implement trend vs chop detection
  - Implement `classify_trend_regime()` function
  - Use `atr_ratio` (7d ATR / 30d ATR) for expansion/compression
  - Use `fusion_state` for directional bias
  - Use `mvrv_ls_level_neutral` for chop detection
  - Use `sent_is_flattening` for compression signals
  - Use `gap_pct` for breakout detection
  - Return `RegimeAxis` with state and confidence
  - Add unit tests for each regime
  - _Requirements: 6.3_

- [ ] 0.5 Implement direction classification
  - Implement `classify_direction()` function
  - Map fusion_state to 5-level direction (strong/mild bullish/bearish, neutral)
  - Include confidence based on fusion_confidence
  - Add unit tests
  - _Requirements: 6.1_

- [ ] 0.6 Implement valuation classification
  - Implement `classify_valuation()` function
  - Map MVRV-60d to valuation states (deep_value, undervalued, fair, overvalued, extreme)
  - Include confidence based on MVRV stability
  - Add unit tests
  - _Requirements: 6.1_

- [ ] 0.7 Implement global veto checks
  - Implement `check_global_vetoes()` function
  - Veto: Strong trend detected (confidence > 0.80)
  - Veto: Expansion regime (ATR ratio > 1.5 or gap > 4%)
  - Veto: IV < RV (realized vol exceeding implied)
  - Veto: Existing directional signal active
  - Veto: IV rank < 0.30
  - Return list of veto reasons
  - Add unit tests for each veto
  - _Requirements: 6.10_

### Task 1: Income Signal Logic

- [ ] 1.1 Implement CSP_ACCUMULATION signal logic
  - Implement `_evaluate_csp_accumulation()` method
  - Required: MVRV undervalued (< 0.95)
  - Required: IV elevated (> 0.50)
  - Required: IV > RV (ratio > 1.2)
  - Required: Direction neutral to mild bullish OR capitulation exhaustion
  - Required: Trend regime NOT strong trend down, NOT expansion
  - Preferred: Panic IV spike, local capitulation, oversold, accumulation zone
  - Avoid: Momentum breakdown, volatility expansion, deleveraging, whale distribution
  - Return strike guidance: 10-15% OTM, DTE 14-30d
  - Add unit tests for each condition
  - _Requirements: 6.6_

- [ ] 1.2 Implement BULL_PUT_SPREAD signal logic
  - Implement `_evaluate_bull_put_spread()` method
  - Required: Direction mild bullish or neutral (NOT strong bullish)
  - Required: IV elevated (> 0.40)
  - Required: IV > RV (ratio > 1.2)
  - Required: Trend regime chop or mild trend up (NOT expansion)
  - Required: Support structure intact
  - Preferred: Bullish drift, support holding, IV significantly overpriced
  - Avoid: Trend breakdown, macro uncertainty, expansion, strong bearish momentum
  - Return strike guidance: short 5-10% OTM, long 15-20% OTM, DTE 14-21d
  - Add unit tests for each condition
  - _Requirements: 6.7_

- [ ] 1.3 Implement BEAR_CALL_SPREAD signal logic
  - Implement `_evaluate_bear_call_spread()` method
  - Required: Direction mild bearish or neutral (NOT strong bearish)
  - Required: IV elevated (> 0.40)
  - Required: IV > RV (ratio > 1.2)
  - Required: Trend regime chop or mild trend down (NOT expansion)
  - Required: Resistance confirmed
  - Preferred: Resistance rejection, weak momentum, distribution phase
  - Avoid: Short squeeze, strong uptrend, breakout expansion, whale accumulation
  - **Higher confidence threshold (0.75)** due to BTC upside squeeze risk
  - Return strike guidance: short 7-12% OTM, long 17-22% OTM, DTE 14-21d
  - Add unit tests for each condition
  - _Requirements: 6.8_

- [ ] 1.4 Extend condor_gate.py with IV vs RV check
  - Add IV vs RV ratio check to `evaluate_condor_gate()`
  - Require `iv_rv_ratio > 1.3` (stricter than spreads)
  - Add to hard vetoes if IV < RV
  - Add unit tests
  - _Requirements: 6.9_

### Task 2: Integration and Persistence

- [ ] 2.1 Integrate with SignalService
  - Add `evaluate_income_signals()` method to SignalService
  - Call regime classification first
  - Check global vetoes
  - Evaluate each signal type in priority order
  - Return `IncomeSignalResult` with regime details
  - Add integration tests
  - _Requirements: 6.5_

- [ ] 2.2 Extend DailySignal model for income signals
  - Add `income_signal_type` field (CharField)
  - Add `income_signal_active` field (BooleanField)
  - Add `income_signal_rationale` field (TextField)
  - Add `income_regime_direction` field (CharField)
  - Add `income_regime_iv` field (CharField)
  - Add `income_regime_trend` field (CharField)
  - Add `income_iv_rv_ratio` field (FloatField)
  - Add `income_regime_confidence` field (JSONField)
  - Create migration
  - _Requirements: 6.4, 6.12_

- [ ] 2.3 Add income signal performance tracking
  - Track win rate per signal type
  - Track win rate per regime combination
  - Track average premium collected
  - Track average settlement paid (for CSP)
  - Track regime accuracy (did predicted regime hold?)
  - Add dashboard/API endpoint for signal performance
  - _Requirements: 6.12_

## Phase 1: Core Infrastructure (Foundation)

### Task 1: Strategy Classification System

- [ ] 1.1 Create strategy type enums and data structures
  - Create `execution/services/income/__init__.py`
  - Create `execution/services/income/classifier.py`
  - Define `StrategyType` enum (USDC-settled strategies only)
  - Define `StrategyCategory` enum (INCOME, ACCUMULATION, DIRECTIONAL)
  - Define `CollateralType` enum (USDC only for Phase 1)
  - Define `SettlementType` enum (CASH_USDC only for Phase 1)
  - Define `ExerciseStyle` enum (EUROPEAN only - all Deribit options)
  - _Requirements: 1.1, 1.2_

- [ ] 1.2 Implement StrategyProfile dataclass
  - Define all strategy attributes (is_income, is_accumulation, is_defined_risk, is_fully_collateralized)
  - Define USDC collateral requirements per strategy
  - Define max_loss_formula and max_profit_formula (USDC-denominated)
  - Add settlement_type and exercise_style fields
  - Remove early_assignment_risk (always False for European options)
  - _Requirements: 1.3_

- [ ] 1.3 Implement StrategyClassifier class
  - Create PROFILES registry with all strategy profiles (USDC-settled only)
  - Implement `classify(strategy_type)` method
  - Implement `validate_exchange_support(strategy_type, exchange)` method
  - Add unit tests for classifier
  - _Requirements: 1.1, 1.2, 1.3_

### Task 2: Safety Validator

- [ ] 2.1 Create rejection code system
  - Create `execution/services/income/validator.py`
  - Define `RejectionSeverity` enum (BLOCKING, WARNING)
  - Define `RejectionCode` enum with all rejection codes including:
    - Protective leg validation codes (expiry, type, settlement, quantity, strike mismatch)
    - Execution safety codes (partial fill, long leg not filled)
    - Exchange/product codes (inverse not supported, non-USDC not supported)
  - Define `RejectionDetail` dataclass
  - Define `ValidationResult` dataclass
  - _Requirements: 8.1_

- [ ] 2.2 Implement prohibited pattern detection
  - Implement `_check_prohibited_patterns()` method
  - Detect naked calls → NAKED_CALL_PROHIBITED
  - Detect naked puts without sufficient collateral → NAKED_PUT_INSUFFICIENT_COLLATERAL
  - Detect short straddles → SHORT_STRADDLE_PROHIBITED
  - Detect short strangles → SHORT_STRANGLE_PROHIBITED
  - Detect ratio spreads with uncovered tails → UNCOVERED_RATIO_PROHIBITED
  - Detect leveraged short structures → LEVERAGED_SHORT_PROHIBITED
  - Add unit tests for each prohibited pattern
  - _Requirements: 2.2, Safety Constraints S1-S3_

- [ ] 2.3 Implement product support validation (Phase 1)
  - Implement `_check_product_support()` method
  - Reject non-USDC settlement → NON_USDC_SETTLEMENT_NOT_SUPPORTED
  - Reject non-Deribit exchanges → EXCHANGE_NOT_SUPPORTED
  - Reject inverse BTC options → INVERSE_OPTIONS_NOT_SUPPORTED
  - Add unit tests
  - _Requirements: Scope constraints_

- [ ] 2.4 Implement COMPLETE protective leg validation
  - Implement `_check_protective_legs_complete()` method
  - Check 1: Same underlying (BTC) → PROTECTIVE_LEG_UNDERLYING_MISMATCH
  - Check 2: Same expiry → PROTECTIVE_LEG_EXPIRY_MISMATCH
  - Check 3: Same settlement currency (USDC) → PROTECTIVE_LEG_SETTLEMENT_MISMATCH
  - Check 4: Same option type (call/put) → PROTECTIVE_LEG_TYPE_MISMATCH
  - Check 5: Correct strike ordering → PROTECTIVE_LEG_STRIKE_INVALID
    - Put spread: long strike ≤ short strike
    - Call spread: long strike ≥ short strike
  - Check 6: Quantity coverage (long qty ≥ short qty) → PROTECTIVE_LEG_QUANTITY_INSUFFICIENT
  - Add unit tests for EACH check individually
  - Add integration test for combined validation
  - _Requirements: 2.3, Safety Constraint S5_

- [ ] 2.5 Implement USDC collateral validation
  - Implement `_check_collateral_usdc()` method
  - Validate against AUTHENTICATED exchange state (not user-supplied)
  - Cash-secured put: collateral >= strike * size * qty
  - Credit spread: collateral >= width * size * qty
  - Return INSUFFICIENT_COLLATERAL_VERIFIED if insufficient
  - Add `is_exchange_verified` flag to AccountState
  - Add unit tests for collateral checks
  - _Requirements: 2.4, Safety Constraint S4, S8_

- [ ] 2.6 Implement liquidity gates
  - Implement `_check_liquidity()` method
  - Check bid-ask spread vs max_spread_pct threshold
  - Check open interest vs min_open_interest threshold
  - Check quote staleness
  - Return LIQUIDITY_WARNING for violations
  - Add unit tests
  - _Requirements: Should-fix liquidity gates_

- [ ] 2.7 Implement SafetyValidator class
  - Combine all validation checks
  - Implement `validate(strategy, account, require_exchange_verification)` method
  - Add SIMULATION_UNVERIFIED warning when account not exchange-verified
  - Return ValidationResult with pass/fail and details
  - Include suggested alternatives for rejections
  - Add integration tests
  - _Requirements: 2.5, Safety Constraint S6, S8_


### Task 3: Payoff Calculator

- [ ] 3.1 Create payoff data structures
  - Create `execution/services/income/payoff.py`
  - Define `PayoffMetrics` dataclass with all metrics (USDC-denominated)
  - Define `OptionLeg` dataclass for leg representation
  - Add settlement_currency field (USDC for Phase 1)
  - Remove early_assignment_risk (European options only)
  - Add settlement_risk field instead
  - _Requirements: 3.1_

- [ ] 3.2 Implement premium conversion (BTC to USDC)
  - Implement `_convert_premium_to_usdc()` method
  - Use spot price for conversion: usdc_value = btc_value * spot_price
  - Match existing pattern in deribit_executor.py line 671
  - Add unit tests for conversion
  - _Requirements: 3.4 (explicit USDC denomination)_

- [ ] 3.3 Implement single-leg payoff calculations (USDC)
  - Implement `_calc_cash_secured_put()` method
    - max_profit_usdc = premium_usdc
    - max_loss_usdc = (strike * size * qty) - premium_usdc
    - breakeven = strike - (premium_usdc / (size * qty))
    - collateral_usdc = strike * size * qty
  - Note: Single-leg short calls NOT supported (undefined loss)
  - Add unit tests for single-leg calculations
  - _Requirements: 3.1, 3.2_

- [ ] 3.4 Implement spread payoff calculations (USDC)
  - Implement `_calc_credit_spread()` method
    - net_credit_usdc = short_premium_usdc - long_premium_usdc
    - max_profit_usdc = net_credit_usdc
    - max_loss_usdc = (width * size * qty) - net_credit_usdc
    - breakeven calculation (direction-dependent)
  - Implement `_calc_debit_spread()` method
    - net_debit_usdc = long_premium_usdc - short_premium_usdc
    - max_profit_usdc = (width * size * qty) - net_debit_usdc
    - max_loss_usdc = net_debit_usdc
  - Add unit tests for spread calculations
  - _Requirements: 3.1, 3.5_

- [ ] 3.5 Implement iron condor payoff calculation (USDC)
  - Implement `_calc_iron_condor()` method
  - Calculate combined put spread + call spread metrics
  - max_profit_usdc = put_credit_usdc + call_credit_usdc
  - max_loss_usdc = max(put_spread_loss, call_spread_loss)
  - Two breakeven prices
  - Collateral = max of either spread
  - Add unit tests
  - _Requirements: 3.1, 3.5_

- [ ] 3.6 Implement settlement risk assessment (European options)
  - Implement `_assess_settlement_risk()` method (NOT early assignment)
  - Calculate ITM probability at expiry from delta
  - Classify risk as HIGH/MEDIUM/LOW based on moneyness and DTE
  - Calculate max settlement payout in USDC
  - Add unit tests
  - _Requirements: 3.3_

- [ ] 3.7 Implement capital efficiency calculation
  - Calculate annualized return on collateral (USDC)
  - Factor in DTE for annualization
  - Calculate probability of profit from delta
  - Calculate expected value in USDC
  - Add liquidation_buffer_pct calculation
  - Add unit tests
  - _Requirements: 3.2_

## Phase 2: Risk Management

### Task 4: Risk Engine

- [ ] 4.1 Create risk configuration
  - Create `execution/services/income/risk.py`
  - Define `RiskLimits` dataclass with all limits
  - Define `RiskAssessment` dataclass
  - Add default limits to policy configuration
  - _Requirements: 4.1_

- [ ] 4.2 Implement position-level risk checks
  - Implement `_check_position_risk()` method
  - Check max_loss / account_value <= max_single_position_risk_pct
  - Return RISK_LIMIT_EXCEEDED if exceeded
  - Add unit tests
  - _Requirements: 4.1, 4.4_

- [ ] 4.3 Implement portfolio-level risk checks
  - Implement `_check_portfolio_risk()` method
  - Check total income exposure <= max_total_income_exposure_pct
  - Check settlement exposure <= max_settlement_exposure_pct
  - Check single expiry concentration <= max_single_expiry_concentration_pct
  - Add unit tests
  - _Requirements: 4.1, 4.2_

- [ ] 4.4 Implement risk warnings
  - Implement `_check_warnings()` method
  - HIGH_IV_PREMIUM_WARNING when IV rank > 80
  - LOW_IV_PREMIUM_WARNING when IV rank < 20
  - GAMMA_RISK_WARNING when DTE < 7
  - DIRECTIONAL_RISK_WARNING when |delta| > 0.40
  - Add unit tests
  - _Requirements: 4.3_

- [ ] 4.5 Implement RiskEngine class
  - Combine all risk checks
  - Implement `assess(strategy, payoff, account)` method
  - Return RiskAssessment with approval status and details
  - Add integration tests
  - _Requirements: 4.1, 4.2, 4.3, 4.4_


## Phase 3: Atomic Execution (CRITICAL SAFETY)

### Task 5: Atomic Execution Controller

- [ ] 5.1 Create atomic execution data structures
  - Create `execution/services/income/atomic_executor.py`
  - Define `LegFillState` dataclass (symbol, filled_qty, verified)
  - Define `AtomicExecutionResult` dataclass
  - Define `NakedExposureAlert` dataclass
  - _Requirements: 11.1, 11.4, Safety Constraint S7_

- [ ] 5.2 Implement long-leg-first execution
  - Implement `_execute_long_leg_first()` method
  - Step 1: Place buy order for protective long leg
  - Step 2: Wait for long leg fill confirmation
  - Step 3: Verify filled quantity
  - Step 4: Only proceed if long leg filled
  - Add unit tests
  - _Requirements: 11.2, Safety Constraint S7_

- [ ] 5.3 Implement short leg quantity limiting
  - Implement `_calculate_safe_short_qty()` method
  - Short leg quantity MUST NOT exceed long leg filled quantity
  - Track `long_leg_filled_qty` and `short_leg_max_qty`
  - Add unit tests
  - _Requirements: 11.2, 11.4_

- [ ] 5.4 Implement partial fill recovery
  - Implement `_handle_partial_fill_naked_exposure()` method
  - Detect short leg overfill beyond protected quantity
  - Immediately cancel remainder
  - Buy-to-close any naked excess
  - Set `naked_exposure_detected = True` and trigger emergency close
  - Add unit tests
  - _Requirements: 11.3_

- [ ] 5.5 Implement emergency close for naked exposure
  - Implement `_emergency_close_naked_exposure()` method
  - Use market order to close naked position immediately
  - Log emergency event with full details
  - Add unit tests
  - _Requirements: 11.3_

- [ ] 5.6 Extend DeribitExecutor for income strategies
  - Modify `_execute_legs_atomic()` to enforce long-leg-first for income spreads
  - Add `_verify_protective_coverage()` check before placing short leg
  - Integrate with existing `_handle_partial_fill()` method
  - Add integration tests
  - _Requirements: 11.5_

- [ ] 5.7 Add atomic combo order support (if available)
  - Check if Deribit supports combo/spread orders
  - Implement `_execute_atomic_combo()` if supported
  - Fall back to sequential-safe execution if not
  - Add unit tests
  - _Requirements: 11.1_

## Phase 4: Recommendation System

### Task 6: Market Regime Classification

- [ ] 6.1 Create recommendation data structures
  - Create `execution/services/income/recommendation.py`
  - Define `MarketRegime` enum (BULLISH, BEARISH, RANGE_BOUND, HIGH_IV, LOW_IV)
  - Define `StrategyRecommendation` dataclass
  - _Requirements: 5.1_

- [ ] 6.2 Implement regime classification
  - Implement `classify_regime()` method
  - Integrate with existing fusion_state from DailySignal
  - Map fusion states to regimes (STRONG_BULLISH → BULLISH, etc.)
  - Override with IV-based regime when IV rank extreme
  - Add unit tests
  - _Requirements: 5.1_

### Task 7: Recommendation Engine

- [ ] 7.1 Define strategy fit matrix
  - Create REGIME_STRATEGY_FIT mapping
  - Define fit scores (0-1) for each strategy in each regime
  - Document rationale for each fit score
  - _Requirements: 5.2_

- [ ] 7.2 Define strike and DTE guidance
  - Create STRIKE_GUIDANCE mapping by regime and strategy
  - Create DTE_GUIDANCE mapping by regime
  - Include regime-specific adjustments
  - _Requirements: 5.3_

- [ ] 7.3 Implement recommendation generation
  - Implement `recommend()` method
  - Filter strategies by account constraints (USDC available)
  - Sort by fit score
  - Generate rationale for each recommendation
  - Include warnings for regime-specific risks
  - Add unit tests
  - _Requirements: 5.2, 5.3_

- [ ] 7.4 Integrate with fusion engine
  - Read fusion_state and fusion_confidence from DailySignal
  - Cross-reference with MVRV and overlay signals
  - Adjust recommendation strength based on confidence
  - Add integration tests
  - _Requirements: 5.4_

## Phase 5: Cash-Secured Put Income Tracking

### Task 8: CSP Income Tracking

- [ ] 8.1 Create CSP income tracking data models
  - Create `CSPIncomeTracker` Django model
  - Add fields for USDC tracking, premium collected, settlement paid
  - Add migration for new model
  - _Requirements: 6.1_

- [ ] 8.2 Implement CSPIncomeManager class
  - Create `execution/services/income/csp_tracker.py`
  - Implement `get_metrics()` method
  - Implement `record_position()` method
  - Add unit tests
  - _Requirements: 6.1_

- [ ] 8.3 Implement CSP metrics calculation (USDC)
  - Calculate total_premium_collected_usdc
  - Calculate total_settlement_paid_usdc (when puts expire ITM)
  - Calculate net_income_usdc (premium - settlement)
  - Calculate annualized_yield_pct
  - Add unit tests
  - _Requirements: 6.2_

- [ ] 8.4 Implement CASH SETTLEMENT handling (NOT assignment)
  - Implement `handle_cash_settlement()` method
  - Short put settles ITM: **PAY** cash = (strike - spot) * size (this is a LOSS)
  - Short put settles OTM: keep premium (this is a GAIN)
  - Add clear documentation: NO physical BTC delivery on Deribit
  - Add unit tests
  - _Requirements: 6.3, Safety Constraint S9_

- [ ] 8.5 Implement position history tracking
  - Track each CSP position opened/closed
  - Track premium received per position
  - Track settlement outcome per position
  - Calculate running P&L
  - Add unit tests
  - _Requirements: 6.4_

**Note:** Traditional wheel strategy tasks removed - the call phase requires naked short calls which have undefined loss. Users wanting call-side income should use bear call spreads instead.


## Phase 6: Integration

### Task 9: Data Model Integration

- [ ] 9.1 Create IncomePosition model
  - Add `IncomePosition` model to `execution/models.py`
  - Include strategy_type, category, legs (JSON), metrics
  - Include status tracking (open, closed, settled, rolled, expired)
  - Add settlement_currency field (USDC for Phase 1)
  - Add migration
  - _Requirements: 7.4_

- [ ] 9.2 Extend ExecutionIntent for income strategies
  - Add income-specific fields to ExecutionIntent
  - Add strategy_category field
  - Add collateral_type field (USDC for Phase 1)
  - Add settlement_handling field (cash settlement instructions)
  - Remove assignment_handling (no assignment on Deribit)
  - Add migration
  - _Requirements: 7.1_

- [ ] 9.3 Add income strategy policy configuration
  - Extend `policy.py` with `IncomeStrategyConfig`
  - Add risk limits configuration
  - Add exit rules configuration
  - Add CSP parameters configuration (replaces wheel - wheel required naked calls)
  - Add `get_income_policy()` function
  - _Requirements: 4.1, 6.2, 10.1_

### Task 10: TradeSetupBuilder Integration

- [ ] 10.1 Extend TradeSetupBuilder for income strategies
  - Add `build_income_setup()` method to TradeSetupBuilder
  - Reuse existing option snapshot selection logic
  - Integrate with SafetyValidator (with exchange verification)
  - Integrate with PayoffCalculator (USDC-denominated)
  - Integrate with RiskEngine
  - _Requirements: 7.2_

- [ ] 10.2 Implement strategy-specific builders
  - Implement `_build_csp_setup()` for cash-secured puts
  - Implement `_build_bull_put_spread()` for bull put spreads
  - Implement `_build_bear_call_spread()` for bear call spreads (defined-risk call income)
  - Implement `_build_iron_condor()` for iron condors
  - Note: Single-leg short calls NOT supported (undefined loss)
  - Add unit tests for each builder
  - _Requirements: 7.2_

- [ ] 10.3 Create IncomeStrategySetup dataclass
  - Define complete setup structure
  - Include legs, metrics, validation, risk assessment
  - Include exit rules and roll recommendations
  - Add settlement_currency and exercise_style fields
  - Implement `to_dict()` for API serialization
  - _Requirements: 7.1, 7.2_

### Task 11: API Endpoints

- [ ] 11.1 Create income strategy API views
  - Create `api/income_views.py`
  - Implement `IncomeStrategyListView` (GET /api/v1/income/strategies/)
  - Include settlement_type and exercise_style in response
  - Add authentication and permissions
  - Add unit tests
  - _Requirements: 7.3_

- [ ] 11.2 Implement recommendation endpoint (SIMULATION mode)
  - Implement `IncomeRecommendView` (GET /api/v1/income/recommend/)
  - Accept iv_rank, usdc_available params (SIMULATION ONLY)
  - Mark response as "validation_mode": "simulation"
  - Include warning about unverified balances
  - Return regime classification and ranked recommendations
  - Add unit tests
  - _Requirements: 7.3, Safety Constraint S8_

- [ ] 11.3 Implement validation endpoint (EXCHANGE-VERIFIED mode)
  - Implement `IncomeValidateView` (POST /api/v1/income/validate/)
  - REQUIRE account_id for authenticated exchange state
  - Fetch REAL balances from Deribit API
  - Mark response as "validation_mode": "exchange_verified"
  - This is the ONLY endpoint that can approve for execution
  - Add unit tests
  - _Requirements: 7.3, Safety Constraint S8_

- [ ] 11.4 Implement setup endpoint (dual mode)
  - Implement `IncomeSetupView` (POST /api/v1/income/setup/)
  - Support mode="simulation" (user-supplied balances, unverified)
  - Support mode="execution" (requires account_id, exchange-verified)
  - Clearly mark validation_mode in response
  - Add warning for simulation mode
  - Add unit tests
  - _Requirements: 7.3, Safety Constraint S8_

- [ ] 11.5 Implement CSP income status endpoint
  - Implement `CSPIncomeStatusView` (GET /api/v1/income/csp/status/)
  - Return CSP income tracking metrics
  - Include note about cash settlement (NOT assignment)
  - Support listing all positions or specific position by ID
  - Add unit tests
  - _Requirements: 7.3_

- [ ] 11.6 Register API routes
  - Add income endpoints to `api/urls.py`
  - Add serializers in `api/income_serializers.py`
  - Add integration tests for all endpoints
  - _Requirements: 7.3_


## Phase 7: Exit Management

### Task 12: Exit Rules Engine

- [ ] 12.1 Define exit rule configuration
  - Add exit rules to IncomeStrategyConfig
  - Define profit targets per strategy type
  - Define loss limits per strategy type
  - Define management DTE thresholds
  - _Requirements: 10.1_

- [ ] 12.2 Implement exit recommendation logic
  - Create exit recommendation method
  - Calculate current P&L as % of max profit (USDC)
  - Recommend CLOSE when profit target reached
  - Recommend CLOSE when loss limit reached
  - Recommend ROLL when approaching expiry
  - _Requirements: 10.2_

- [ ] 12.3 Implement roll as close-and-open sequence
  - Implement roll as TWO separate operations: close + open
  - Each operation MUST pass full safety validation
  - No "roll" shortcut that bypasses validation
  - Add unit tests
  - _Requirements: 10.3_

- [ ] 12.4 Implement roll recommendations
  - Determine when to roll (DTE threshold, profit threshold)
  - Calculate roll direction (same strike, different strike)
  - Calculate roll timing (before/after expiry)
  - Add unit tests
  - _Requirements: 10.2_

## Phase 8: Testing & Validation (CRITICAL)

### Task 13: Unit Tests

- [ ] 13.1 Strategy classifier tests
  - Test all strategy type classifications (USDC-settled only)
  - Test profile attribute correctness
  - Test exchange support validation (Deribit only)
  - Test edge cases (unknown types, unsupported exchanges)
  - _Requirements: 1.1, 1.2, 1.3_

- [ ] 13.2 Safety validator tests - Prohibited patterns
  - Test naked call rejection
  - Test naked put insufficient collateral rejection
  - Test short straddle rejection
  - Test short strangle rejection
  - Test ratio spread with uncovered tail rejection
  - Test leveraged short rejection
  - Verify rejection messages are clear and actionable
  - _Requirements: 2.2, Safety Constraints S1-S3_

- [ ] 13.3 Safety validator tests - Protective leg validation (COMPLETE)
  - Test expiry mismatch rejection
  - Test option type mismatch rejection
  - Test settlement currency mismatch rejection
  - Test underlying mismatch rejection
  - Test strike ordering validation (put spread: long ≤ short)
  - Test strike ordering validation (call spread: long ≥ short)
  - Test quantity coverage (long qty ≥ short qty)
  - Test ratio tail detection
  - **Test iron condor: BOTH wings must have protective legs**
  - **Test iron condor: rejection if put wing missing protective leg**
  - **Test iron condor: rejection if call wing missing protective leg**
  - **Test iron condor: each short leg validated independently**
  - _Requirements: 2.4, Safety Constraint S5_

- [ ] 13.4 Safety validator tests - Exchange/product constraints
  - Test inverse BTC options rejection
  - Test non-USDC settlement rejection
  - Test non-Deribit exchange rejection
  - _Requirements: Scope constraints_

- [ ] 13.5 Safety validator tests - Collateral validation
  - Test exchange-verified collateral check
  - Test simulation mode warning
  - Test insufficient collateral rejection
  - _Requirements: 2.5, Safety Constraint S4, S8_

- [ ] 13.6 Payoff calculator tests (USDC-denominated)
  - Test CSP payoff calculations in USDC
  - Test credit spread payoff calculations (bull put, bear call)
  - Test iron condor payoff calculations
  - Test premium BTC-to-USDC conversion
  - Test settlement risk assessment (NOT early assignment)
  - Verify max_loss is always defined and finite
  - Test that single-leg short calls are REJECTED with NAKED_CALL_PROHIBITED
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [ ] 13.7 Risk engine tests
  - Test position-level risk limit enforcement
  - Test portfolio-level risk limit enforcement
  - Test warning generation
  - Test risk assessment completeness
  - Test zero account value handling
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [ ] 13.8 Recommendation engine tests
  - Test regime classification from fusion states
  - Test IV-based regime override
  - Test strategy fit scoring
  - Test recommendation filtering by USDC constraints
  - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [ ] 13.9 CSP income tracker tests
  - Test metrics calculations (USDC-denominated)
  - Test CASH SETTLEMENT handling (short put ITM = LOSS, pay settlement)
  - Test position history tracking
  - Verify NO physical BTC delivery language
  - _Requirements: 6.1, 6.2, 6.3, 6.4, Safety Constraint S9_

### Task 14: Atomic Execution Tests (CRITICAL)

- [ ] 14.1 Long-leg-first execution tests
  - Test long leg placed before short leg
  - Test short leg blocked if long leg not filled
  - Test short leg quantity limited to long leg filled quantity
  - _Requirements: 11.2, Safety Constraint S7_

- [ ] 14.2 Partial fill recovery tests
  - Test short leg overfill detection
  - Test immediate cancel of overfill remainder
  - Test buy-to-close of naked excess
  - Test naked_exposure_detected flag
  - _Requirements: 11.3_

- [ ] 14.3 Emergency close tests
  - Test emergency close triggered on naked exposure
  - Test market order used for emergency close
  - Test event logging for emergency close
  - _Requirements: 11.3_

### Task 15: Integration Tests

- [ ] 15.1 End-to-end strategy setup tests
  - Test complete CSP setup flow (USDC)
  - Test complete credit spread setup flow (bull put, bear call)
  - Test complete iron condor setup flow
  - Verify all safety constraints enforced
  - Note: Single-leg short calls NOT tested (not supported)
  - _Requirements: 7.1, 7.2_

- [ ] 15.2 API endpoint tests - Simulation vs Verified modes
  - Test recommendation endpoint (simulation mode, unverified warning)
  - Test validation endpoint (exchange-verified mode)
  - Test setup endpoint simulation mode
  - Test setup endpoint execution mode (requires account_id)
  - Test rejection of execution without account_id
  - _Requirements: 7.3, Safety Constraint S8_

- [ ] 15.3 API endpoint tests - CSP Income
  - Test CSP income status endpoint
  - Test cash settlement language (NOT assignment)
  - _Requirements: 7.3_

### Task 16: Edge Case Tests (CRITICAL)

- [ ] 16.1 Boundary condition tests
  - Test at exact risk limit boundaries
  - Test with zero USDC collateral
  - Test with extreme IV values
  - Test with very short DTE (gamma risk)
  - Test with zero account value (should not divide by zero)
  - _Requirements: 4.1, 4.4_

- [ ] 16.2 Spoofed balance tests
  - Test that execution mode REQUIRES account_id
  - Test that user-supplied balances are marked "simulation"
  - Test that simulation mode cannot approve for execution
  - _Requirements: Safety Constraint S8_

- [ ] 16.3 Deribit-specific tests
  - Test European option behavior (no early assignment)
  - Test cash settlement behavior (no physical delivery)
  - Test USDC settlement only
  - _Requirements: Safety Constraint S9, Scope constraints_

- [ ] 16.4 Error handling tests
  - Test with missing option data
  - Test with invalid strategy types
  - Test with malformed requests
  - Test database error handling
  - Test exchange API error handling
  - _Requirements: 7.3, 8.1_

## Phase 9: Documentation

### Task 17: Documentation

- [ ] 17.1 API documentation
  - Document all new endpoints
  - Document simulation vs execution modes
  - Include request/response examples
  - Document error codes and messages
  - _Requirements: 7.3_

- [ ] 17.2 Strategy guide documentation
  - Document each supported strategy (USDC-settled)
  - Include payoff diagrams
  - Include risk/reward characteristics
  - Include regime suitability
  - Document cash settlement (NOT assignment)
  - _Requirements: 1.1, 5.2_

- [ ] 17.3 Safety constraints documentation
  - Document all prohibited strategies
  - Explain why each is prohibited
  - Document rejection codes and meanings
  - Document Deribit-specific constraints (European, cash-settled)
  - _Requirements: Safety Constraints, 8.1_

- [ ] 17.4 Exchange-specific documentation
  - Document Deribit Linear USDC options scope
  - Document why inverse BTC options are excluded (Phase 1)
  - Document cash settlement vs physical delivery
  - Document European vs American exercise
  - _Requirements: Scope constraints_

## Notes

- **Safety First**: All tasks involving strategy validation must be completed before any execution-related tasks
- **Deribit USDC Only**: Phase 1 is restricted to Deribit Linear USDC options
- **No Physical Delivery**: All Deribit options are European and cash-settled
- **Exchange-Verified Execution**: Execution requires authenticated exchange state
- **Atomic Execution**: Multi-leg strategies require long-leg-first or atomic combo orders
- **Testing Priority**: Safety validator and atomic execution tests are highest priority

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2", "1.3"] },
    { "id": 1, "tasks": ["2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7"] },
    { "id": 2, "tasks": ["3.1", "3.2", "3.3", "3.4", "3.5", "3.6", "3.7"] },
    { "id": 3, "tasks": ["4.1", "4.2", "4.3", "4.4", "4.5"] },
    { "id": 4, "tasks": ["5.1", "5.2", "5.3", "5.4", "5.5", "5.6", "5.7"] },
    { "id": 5, "tasks": ["6.1", "6.2", "7.1", "7.2", "7.3", "7.4"] },
    { "id": 6, "tasks": ["8.1", "8.2", "8.3", "8.4", "8.5", "8.6"] },
    { "id": 7, "tasks": ["9.1", "9.2", "9.3", "10.1", "10.2", "10.3"] },
    { "id": 8, "tasks": ["11.1", "11.2", "11.3", "11.4", "11.5", "11.6"] },
    { "id": 9, "tasks": ["12.1", "12.2", "12.3", "12.4"] },
    { "id": 10, "tasks": ["13.1", "13.2", "13.3", "13.4", "13.5", "13.6", "13.7", "13.8", "13.9"] },
    { "id": 11, "tasks": ["14.1", "14.2", "14.3", "15.1", "15.2", "15.3"] },
    { "id": 12, "tasks": ["16.1", "16.2", "16.3", "16.4"] },
    { "id": 13, "tasks": ["17.1", "17.2", "17.3", "17.4"] }
  ]
}
```

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Naked option exposure | SafetyValidator blocks all prohibited patterns; atomic execution enforces long-leg-first |
| Partial fill naked exposure | AtomicExecutionController detects and emergency-closes naked positions |
| Undefined max loss | PayoffCalculator always returns finite max_loss; validation rejects undefined |
| Collateral insufficiency | Collateral check against EXCHANGE-VERIFIED state is blocking |
| Spoofed balances | Execution mode REQUIRES account_id; simulation mode clearly marked |
| Inverse BTC collateral risk | Phase 1 restricted to USDC-settled options only |
| Assignment confusion | All documentation uses "cash settlement" language; no "receive BTC" or "BTC called away" |
| Protective leg gaps | Complete validation: same underlying, expiry, settlement, type, strike ordering, quantity |
| Risk limit breach | RiskEngine blocks execution when limits exceeded |
