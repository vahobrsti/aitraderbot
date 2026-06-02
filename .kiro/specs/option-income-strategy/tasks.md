# Implementation Tasks

## Task 1: Create dataclasses and chain normalization
- [x] Create `signals/income_gate.py` with `IncomeGateResult` dataclass (score, regime_eligible, eligible, veto_reasons, score_components, threshold, structure, chain_valid, chain_rejection_reason, spread_guidance, short_strike, long_strike, credit, dte, max_loss)
- [x] Add `IncomeGateConfig` dataclass with all configurable parameters and defaults (score_threshold=70, min_delta=0.15, max_delta=0.30, max_bid_ask_spread_pct=0.15, min_credit_pct=0.25, min_spread_width_pct=0.03, max_spread_width_pct=0.08, tactical DTE 9-21, income DTE 21-45, cooldown_days=5, max_concurrent=1, atr_expansion_threshold=1.5)
- [x] Add `SpreadCandidate` internal dataclass (short_strike, long_strike, credit, spread_width, dte, max_loss, short_delta)
- [x] Implement `normalize_chain_columns(chain_df)` that maps Deribit/Bybit column names to canonical schema (strike, side, delta, bid, ask, dte, spread_pct), converts signed delta to absolute, and returns None-safe result
- [x] Add necessary imports (dataclasses, Optional, pd, OptionStructure, SpreadGuidance from signals.options)

**Requirements:** Req 8, Req 9

## Task 2: Implement Bull Put Spread scoring and vetoes
- [x] Implement `compute_bull_put_score(row: pd.Series) -> tuple[float, dict]` with additive components: mdia_inflow (+25), whale_sponsored (+20), mvrv_macro_bullish (+20), no_option_put (+15), sentiment_safe (+10), whale_not_distributing (+10)
- [x] Implement `check_bull_put_vetoes(row: pd.Series, atr_ratio: Optional[float], fusion_state: Optional[str], higher_priority_active: bool, condor_eligible: bool) -> list[str]` with vetoes: OPTION_PUT_ACTIVE, EXTREME_GREED_PERSIST_5D, MVRV_MACRO_BEARISH, WHALE_DISTRIB_STRONG, HIGHER_PRIORITY_SIGNAL, ATR_EXPANSION, FUSION_STATE_BEARISH, CONDOR_PRECEDENCE

**Requirements:** Req 1, Req 2, Req 7

## Task 3: Implement Bear Call Spread scoring and vetoes
- [x] Implement `compute_bear_call_score(row: pd.Series) -> tuple[float, dict]` with additive components: no_mdia_inflow_or_aging (+25), whale_distribution (+20), mvrv_macro_bearish (+20), no_option_call (+15), sentiment_greed_or_flat (+10), whale_not_accumulating (+10)
- [x] Implement `check_bear_call_vetoes(row: pd.Series, atr_ratio: Optional[float], fusion_state: Optional[str], higher_priority_active: bool, condor_eligible: bool) -> list[str]` with vetoes: OPTION_CALL_ACTIVE, MDIA_STRONG_INFLOW, MVRV_MACRO_BULLISH, WHALE_SPONSORED, HIGHER_PRIORITY_SIGNAL, ATR_EXPANSION, FUSION_STATE_BULLISH, CONDOR_PRECEDENCE

**Requirements:** Req 3, Req 4, Req 7

## Task 4: Implement MVRV-derived strike boundaries and option chain filtering
- [x] Implement `compute_strike_boundaries(spot_price, mvrv_60d, mvrv_composite, mvrv_composite_p90_180d) -> tuple[float, float]` — computes floor (spot/mvrv_60d) for bull put and ceiling ((spot/mvrv_composite)*p90) for bear call
- [x] Implement `filter_option_chain(chain_df: pd.DataFrame, side: str, spot_price: float, config: IncomeGateConfig, dte_mode: str, strike_boundary: Optional[float]) -> pd.DataFrame` — filters by side, OTM, MVRV boundary, delta range, DTE window, bid/ask spread; sorts by abs(delta) descending. Falls back to delta-only if boundary eliminates all candidates.
- [x] Implement `select_spread(filtered_chain: pd.DataFrame, full_chain: pd.DataFrame, side: str, spot_price: float, config: IncomeGateConfig) -> Optional[SpreadCandidate]` — iterates short leg candidates, finds matching long leg from full chain at correct width, validates credit/spread_width ratio, returns SpreadCandidate or None

**Requirements:** Req 5, Req 6

## Task 5: Implement evaluate functions and wire integration
- [x] Implement `evaluate_bull_put_gate(row, chain_df, spot_price, config, atr_ratio, fusion_state, higher_priority_active, condor_eligible) -> IncomeGateResult` — two-layer evaluation: regime scoring/vetoes first, then computes floor via mvrv_60d from row, passes to chain filtering/spread selection
- [x] Implement `evaluate_bear_call_gate(row, chain_df, spot_price, config, atr_ratio, fusion_state, higher_priority_active, condor_eligible) -> IncomeGateResult` — same two-layer pattern, computes ceiling via mvrv_composite and mvrv_composite_p90_180d from row
- [x] Add DECISION_STRATEGY_MAP entries in `signals/options.py` for "BULL_PUT_SPREAD" and "BEAR_CALL_SPREAD" with SpreadGuidance defaults (stop_loss_pct=0.02 in spot-move units)
- [x] Export new public functions and classes from `signals/__init__.py`

**Requirements:** Req 1, Req 3, Req 8, Req 10
