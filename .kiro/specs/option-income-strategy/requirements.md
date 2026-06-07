# Requirements Document

## Introduction

Add BULL_PUT_SPREAD and BEAR_CALL_SPREAD income gates to the existing option signal pipeline. These gates operate alongside the existing IRON_CONDOR gate, using the same scoring pattern (0–100 additive score + hard vetoes) and returning a structured result dataclass. Each gate evaluates on-chain regime conditions, applies directional filters, and selects option chain contracts matching delta, credit, DTE, and spread constraints. Scope is limited to these two spread types — no WHEEL, CSP, or COVERED_CALL strategies.

## Glossary

- **Income_Gate**: A scoring module that evaluates whether market conditions favor opening a credit spread, returning a score (0–100), eligibility boolean, veto reasons, score components, and threshold.
- **Bull_Put_Spread_Gate**: The Income_Gate instance for SHORT_PUT_SPREAD entries — sells a put spread below support in bullish-leaning regimes to collect premium.
- **Bear_Call_Spread_Gate**: The Income_Gate instance for SHORT_CALL_SPREAD entries — sells a call spread above resistance in bearish-leaning regimes to collect premium.
- **IncomeGateResult**: The dataclass returned by each Income_Gate evaluation, following the CondorGateResult pattern (score, eligible, veto_reasons, score_components, threshold, structure, spread_guidance).
- **Fusion_Engine**: The existing signal fusion module (signals/fusion.py) that classifies market state and exposes regime component flags.
- **Option_Chain**: The set of available BTC option contracts fetched from Deribit or Bybit, including delta, bid, ask, implied volatility, DTE, moneyness, and spread percentage.
- **Short_Leg**: The option contract sold (written) in a credit spread, defining the maximum profit strike.
- **Long_Leg**: The option contract bought in a credit spread, defining the maximum loss boundary.
- **Credit**: The net premium received when opening a credit spread (short leg premium minus long leg premium).
- **Spread_Width**: The distance in price between the short leg strike and the long leg strike.
- **DTE**: Days to expiration of an option contract.

## Requirements

### Requirement 1: Bull Put Spread Gate Scoring

**User Story:** As the trading system, I want a Bull Put Spread gate that scores bullish income conditions, so that SHORT_PUT_SPREAD entries only occur when on-chain regime signals align.

#### Acceptance Criteria

1. WHEN the Bull_Put_Spread_Gate evaluates a feature row, THE Income_Gate SHALL compute an additive score from 0 to 100 using the following positive components: mdia_inflow active (+25), whale_sponsored active (+20), mvrv_macro_bullish or mvrv_recovery confirmed (+20), no signal_option_put firing (+15), sentiment not in extreme greed (+10), whale regime mixed or neutral (+10).
2. WHEN the computed score is below the configured threshold, THE Bull_Put_Spread_Gate SHALL set eligible to false in the IncomeGateResult.
3. WHEN the computed score meets or exceeds the configured threshold and no hard vetoes are active, THE Bull_Put_Spread_Gate SHALL set eligible to true in the IncomeGateResult.
4. THE Bull_Put_Spread_Gate SHALL return an IncomeGateResult dataclass containing score, eligible, veto_reasons, score_components, threshold, and the SHORT_PUT_SPREAD structure identifier.

### Requirement 2: Bull Put Spread Hard Vetoes

**User Story:** As the trading system, I want hard vetoes that block Bull Put Spread entries regardless of score, so that the gate never fires in unsafe conditions.

#### Acceptance Criteria

1. WHEN signal_option_put is active, THE Bull_Put_Spread_Gate SHALL add a veto reason and set eligible to false.
2. WHEN sentiment is in extreme greed persisting 5 days or more, THE Bull_Put_Spread_Gate SHALL add a veto reason and set eligible to false.
3. WHEN mvrv_macro_bearish is confirmed (mvrv_ls_regime_put_confirm, mvrv_ls_regime_bear_continuation, mvrv_ls_early_rollover, mvrv_ls_weak_downtrend, or mvrv_ls_regime_distribution_warning), THE Bull_Put_Spread_Gate SHALL add a veto reason and set eligible to false.
4. WHEN whale_regime_distribution_strong is active, THE Bull_Put_Spread_Gate SHALL add a veto reason and set eligible to false.
5. WHEN a higher-priority directional signal (OPTION_CALL or TACTICAL_PUT) is active for the same expiration window, THE Bull_Put_Spread_Gate SHALL add a veto reason and set eligible to false.
6. WHEN the ATR expansion ratio exceeds 1.5, THE Bull_Put_Spread_Gate SHALL add a veto reason and set eligible to false.

### Requirement 3: Bear Call Spread Gate Scoring

**User Story:** As the trading system, I want a Bear Call Spread gate that scores bearish income conditions, so that SHORT_CALL_SPREAD entries only occur when on-chain regime signals align.

#### Acceptance Criteria

1. WHEN the Bear_Call_Spread_Gate evaluates a feature row, THE Income_Gate SHALL compute an additive score from 0 to 100 using the following positive components: no mdia_inflow or mdia_aging active (+25), whale_distribution active (+20), mvrv_macro_bearish or mvrv_distribution_warning confirmed (+20), no signal_option_call firing (+15), sentiment in greed or flattening (+10), whale regime distribution strong (+10).
2. WHEN the computed score is below the configured threshold, THE Bear_Call_Spread_Gate SHALL set eligible to false in the IncomeGateResult.
3. WHEN the computed score meets or exceeds the configured threshold and no hard vetoes are active, THE Bear_Call_Spread_Gate SHALL set eligible to true in the IncomeGateResult.
4. THE Bear_Call_Spread_Gate SHALL return an IncomeGateResult dataclass containing score, eligible, veto_reasons, score_components, threshold, and the SHORT_CALL_SPREAD structure identifier.

### Requirement 4: Bear Call Spread Hard Vetoes

**User Story:** As the trading system, I want hard vetoes that block Bear Call Spread entries regardless of score, so that the gate never fires in unsafe conditions.

#### Acceptance Criteria

1. WHEN signal_option_call is active, THE Bear_Call_Spread_Gate SHALL add a veto reason and set eligible to false.
2. WHEN mdia_regime_strong_inflow is active, THE Bear_Call_Spread_Gate SHALL add a veto reason and set eligible to false.
3. WHEN mvrv_macro_bullish is confirmed (mvrv_ls_regime_call_confirm, mvrv_ls_regime_call_confirm_recovery, or mvrv_ls_regime_call_confirm_trend), THE Bear_Call_Spread_Gate SHALL add a veto reason and set eligible to false.
4. WHEN whale_sponsored is active (whale_regime_broad_accum or whale_regime_strategic_accum), THE Bear_Call_Spread_Gate SHALL add a veto reason and set eligible to false.
5. WHEN a higher-priority directional signal (OPTION_PUT or MVRV_SHORT) is active for the same expiration window, THE Bear_Call_Spread_Gate SHALL add a veto reason and set eligible to false.
6. WHEN the ATR expansion ratio exceeds 1.5, THE Bear_Call_Spread_Gate SHALL add a veto reason and set eligible to false.

### Requirement 5: Option Chain Filtering for Short Leg Selection

**User Story:** As the trading system, I want option chain filters that select appropriate short leg contracts, so that credit spreads have favorable risk/reward characteristics.

#### Acceptance Criteria

1. WHEN selecting a short leg for either spread type, THE Income_Gate SHALL filter for contracts with absolute delta between 0.15 and 0.30 inclusive.
2. WHEN selecting a short leg, THE Income_Gate SHALL require the bid/ask spread percentage to be at or below a configurable maximum (default 15%).
3. WHEN the gate type is Bull_Put_Spread, THE Income_Gate SHALL select put contracts with strikes below the current spot price (out-of-the-money puts).
4. WHEN the gate type is Bear_Call_Spread, THE Income_Gate SHALL select call contracts with strikes above the current spot price (out-of-the-money calls).
5. THE Income_Gate SHALL support two DTE windows: tactical (9–21 days) and slower income (21–45 days), selectable via configuration.

### Requirement 6: Credit and Spread Width Validation

**User Story:** As the trading system, I want credit and spread width validation, so that each trade meets minimum profitability and maximum loss constraints.

#### Acceptance Criteria

1. WHEN a short leg is selected, THE Income_Gate SHALL compute the credit as the net premium received (short leg bid minus long leg ask).
2. WHEN the credit is below the configured minimum percentage of spread width (default 25%), THE Income_Gate SHALL reject the spread candidate.
3. THE Income_Gate SHALL compute maximum loss as spread width minus credit received.
4. WHEN the maximum loss exceeds the configured cap (expressed as a percentage of portfolio allocation), THE Income_Gate SHALL reject the spread candidate.
5. THE Income_Gate SHALL select the long leg strike such that spread width falls within the configured range (default 3–8% of spot price).

### Requirement 7: Signal Conflict Prevention

**User Story:** As the trading system, I want income gates to respect the existing signal hierarchy, so that credit spreads never conflict with higher-priority directional trades.

#### Acceptance Criteria

1. THE Income_Gate SHALL evaluate after the Fusion_Engine and existing directional gates (OPTION_CALL, OPTION_PUT, TACTICAL_PUT, MVRV_SHORT) have completed.
2. WHEN the Fusion_Engine classifies the market as STRONG_BULLISH, EARLY_RECOVERY, or MOMENTUM_CONTINUATION, THE Bear_Call_Spread_Gate SHALL be ineligible.
3. WHEN the Fusion_Engine classifies the market as BEAR_CONTINUATION or DISTRIBUTION_RISK, THE Bull_Put_Spread_Gate SHALL be ineligible.
4. WHEN an IRON_CONDOR gate is eligible for the same evaluation period, THE Income_Gate SHALL defer to the IRON_CONDOR (condor takes precedence over single-leg credit spreads).
5. THE Income_Gate SHALL verify that no existing open position overlaps with the proposed spread strikes before declaring eligibility.

### Requirement 8: IncomeGateResult Dataclass

**User Story:** As a developer, I want a structured result dataclass for income gates, so that downstream consumers have a consistent interface matching the existing CondorGateResult pattern.

#### Acceptance Criteria

1. THE IncomeGateResult SHALL contain the following fields: score (float, 0–100), eligible (bool), veto_reasons (list of strings), score_components (dict), threshold (float), structure (OptionStructure enum value), and spread_guidance (SpreadGuidance instance or None).
2. WHEN eligible is true, THE IncomeGateResult SHALL include a SpreadGuidance instance with width_pct, take_profit_pct, max_hold_days, stop_loss_pct, and scale_down_day populated.
3. THE IncomeGateResult SHALL be importable from the same signals package as CondorGateResult.

### Requirement 9: Configuration and Thresholds

**User Story:** As a developer, I want configurable thresholds for income gates, so that parameters can be tuned without code changes.

#### Acceptance Criteria

1. THE Income_Gate SHALL expose a configurable score threshold with a default value of 70.
2. THE Income_Gate SHALL expose configurable option chain filter parameters: min_delta (default 0.15), max_delta (default 0.30), max_bid_ask_spread_pct (default 0.15), min_credit_pct (default 0.25), min_spread_width_pct (default 0.03), max_spread_width_pct (default 0.08).
3. THE Income_Gate SHALL expose configurable DTE windows: tactical_min_dte (default 9), tactical_max_dte (default 21), income_min_dte (default 21), income_max_dte (default 45).
4. THE Income_Gate SHALL expose a configurable cooldown period between same-type spread entries (default 5 days).
5. THE Income_Gate SHALL expose a configurable maximum concurrent positions per spread type (default 1).

### Requirement 10: Integration with Existing OptionStructure Enum

**User Story:** As a developer, I want income gates to use the existing SHORT_PUT_SPREAD and SHORT_CALL_SPREAD enum values, so that no new enum entries are needed.

#### Acceptance Criteria

1. THE Bull_Put_Spread_Gate SHALL reference OptionStructure.SHORT_PUT_SPREAD for its structure field.
2. THE Bear_Call_Spread_Gate SHALL reference OptionStructure.SHORT_CALL_SPREAD for its structure field.
3. THE Income_Gate SHALL add entries to DECISION_STRATEGY_MAP for "BULL_PUT_SPREAD" and "BEAR_CALL_SPREAD" keys with appropriate SpreadGuidance, strike_guidance, dte_range, and rationale values.
