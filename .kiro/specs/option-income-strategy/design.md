# Design Document

## Overview

Add two directional credit spread income gates (`Bull_Put_Spread_Gate` and `Bear_Call_Spread_Gate`) to the existing signal pipeline. Both follow the established `condor_gate.py` pattern: additive scoring (0–100) + hard vetoes → structured result dataclass.

Key design decision: the gate produces **two layers of truth**:
1. **Regime eligibility** — on-chain scoring + vetoes pass (independent of option chain)
2. **Spread selection** — option chain has a tradable contract meeting credit/delta/DTE constraints

This separation means "regime passed but no acceptable credit" is a distinct, diagnosable state from "regime not eligible." Backtesting and live diagnostics benefit from seeing both layers independently.

## Architecture

### New File: `signals/income_gate.py`

Single module containing all income gate logic. Mirrors `signals/condor_gate.py` in structure.

```
signals/income_gate.py
├── IncomeGateResult (dataclass)
├── IncomeGateConfig (dataclass — thresholds & chain filter params)
├── normalize_chain_columns(chain_df) → DataFrame
├── compute_strike_boundaries(spot, mvrv_60d, mvrv_composite, mvrv_composite_p90) → (floor, ceiling)
├── compute_bull_put_score(row) → (float, dict)
├── check_bull_put_vetoes(row, atr_ratio, fusion_state, higher_priority_active, condor_eligible) → list[str]
├── compute_bear_call_score(row) → (float, dict)
├── check_bear_call_vetoes(row, atr_ratio, fusion_state, higher_priority_active, condor_eligible) → list[str]
├── filter_option_chain(chain_df, side, spot_price, config, dte_mode, strike_boundary) → DataFrame
├── select_spread(filtered_chain, side, spot_price, config) → Optional[SpreadCandidate]
├── evaluate_bull_put_gate(row, ...) → IncomeGateResult
├── evaluate_bear_call_gate(row, ...) → IncomeGateResult
```

### Data Flow

```
Feature Row (pd.Series)
        │
        ▼
┌─────────────────────┐
│   Fusion Engine     │  ← already runs first (signals/fusion.py)
│   (MarketState)     │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐     ┌─────────────────────┐
│  Condor Gate        │     │  Income Gates        │
│  (existing)         │     │  (new)               │
└────────┬────────────┘     └────────┬────────────┘
         │                           │
         ▼                           ▼
┌─────────────────────────────────────────────────┐
│  Signal Conflict Resolution                      │
│  Priority: Directional > Condor > Income Spread  │
└──────────────────────────────────────────────────┘
```

### Priority Hierarchy

1. Directional signals (OPTION_CALL, OPTION_PUT, TACTICAL_PUT, MVRV_SHORT)
2. IRON_CONDOR (existing condor gate)
3. Income spreads (BULL_PUT_SPREAD, BEAR_CALL_SPREAD) — new

**Rationale for condor precedence:** Condor fires in chop/range states; income spreads fire directionally. Overlap should be rare. When it occurs, condor is preferred because range premium is structurally safer (both sides hedged) and directional credit spreads carry more directional risk. This is a conservative default — high-quality directional setups may be skipped, but that's acceptable for an income-first system.

## Components and Interfaces

### Public API

```python
def evaluate_bull_put_gate(
    row: pd.Series,
    chain_df: Optional[pd.DataFrame],
    spot_price: float,
    config: IncomeGateConfig = IncomeGateConfig(),
    atr_ratio: Optional[float] = None,
    fusion_state: Optional[str] = None,
    higher_priority_active: bool = False,
    condor_eligible: bool = False,
) -> IncomeGateResult:

def evaluate_bear_call_gate(
    row: pd.Series,
    chain_df: Optional[pd.DataFrame],
    spot_price: float,
    config: IncomeGateConfig = IncomeGateConfig(),
    atr_ratio: Optional[float] = None,
    fusion_state: Optional[str] = None,
    higher_priority_active: bool = False,
    condor_eligible: bool = False,
) -> IncomeGateResult:
```

**Why explicit `higher_priority_active` and `condor_eligible` params:** TACTICAL_PUT, MVRV_SHORT, and open-position overlap are not reliably available as feature-row columns. The caller (pipeline orchestrator) already knows these states and passes them in. This avoids `income_gate.py` needing to infer pipeline state from incomplete row data.

### Internal Functions

```python
def normalize_chain_columns(chain_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names from Deribit/Bybit schemas to canonical names.
    Handles: option_type/side, days_to_expiry/dte, signed vs absolute delta.
    Returns DataFrame with columns: strike, side, delta, bid, ask, dte, spread_pct
    """

def compute_strike_boundaries(
    spot_price: float,
    mvrv_60d: float,
    mvrv_composite: float,
    mvrv_composite_p90_180d: float,
) -> tuple[float, float]:
    """Compute MVRV-derived floor (bull put) and ceiling (bear call) prices."""

def compute_bull_put_score(row: pd.Series) -> tuple[float, dict]:
def check_bull_put_vetoes(row, atr_ratio, fusion_state, higher_priority_active, condor_eligible) -> list[str]:

def compute_bear_call_score(row: pd.Series) -> tuple[float, dict]:
def check_bear_call_vetoes(row, atr_ratio, fusion_state, higher_priority_active, condor_eligible) -> list[str]:

def filter_option_chain(chain_df, side, spot_price, config, dte_mode, strike_boundary) -> pd.DataFrame:
def select_spread(filtered_chain, side, spot_price, config) -> Optional[SpreadCandidate]:
```

### Import Strategy

`income_gate.py` imports `OptionStructure` and `SpreadGuidance` from `signals/options.py`. This is a one-way dependency (income_gate → options). If `options.py` ever needs to reference income gate results, shared types should be extracted to `signals/types.py` to avoid circular imports. For now, the one-way dependency is safe.

## Data Models

### IncomeGateResult

```python
@dataclass
class IncomeGateResult:
    """Result of an income spread gate evaluation."""
    # --- Regime layer (always populated) ---
    score: float                          # 0-100 additive score
    regime_eligible: bool                 # True if score >= threshold and no vetoes
    eligible: bool                        # True if regime_eligible AND chain_valid
    veto_reasons: list[str]               # Hard veto reasons
    score_components: dict                # Breakdown for explainability
    threshold: float                      # Threshold used
    structure: OptionStructure            # SHORT_PUT_SPREAD or SHORT_CALL_SPREAD

    # --- Chain layer (populated only when regime_eligible and chain provided) ---
    chain_valid: bool                     # True if a tradable spread was found
    chain_rejection_reason: Optional[str] # Why chain failed (if regime passed but chain didn't)
    spread_guidance: Optional[SpreadGuidance]  # Populated when eligible
    short_strike: Optional[float]         # Selected short leg strike
    long_strike: Optional[float]          # Selected long leg strike
    credit: Optional[float]              # Net credit received (short_bid - long_ask)
    dte: Optional[int]                   # Days to expiration of selected spread
    max_loss: Optional[float]            # Spread width - credit
```

The two-layer design means:
- `regime_eligible=True, chain_valid=False` → "Setup was good, chain was garbage"
- `regime_eligible=False` → "On-chain conditions don't support this trade"
- `eligible=True` → "Both layers passed, trade is actionable"

### IncomeGateConfig

```python
@dataclass
class IncomeGateConfig:
    """Configurable parameters for income gates."""
    # Scoring
    score_threshold: float = 70.0

    # Option chain filters
    min_delta: float = 0.15
    max_delta: float = 0.30
    max_bid_ask_spread_pct: float = 0.15
    min_credit_pct: float = 0.25          # credit / spread_width minimum
    min_spread_width_pct: float = 0.03    # spread_width / spot minimum
    max_spread_width_pct: float = 0.08    # spread_width / spot maximum

    # DTE windows
    tactical_min_dte: int = 9
    tactical_max_dte: int = 21
    income_min_dte: int = 21
    income_max_dte: int = 45

    # Position management
    cooldown_days: int = 5
    max_concurrent: int = 1

    # Volatility
    atr_expansion_threshold: float = 1.5
```

### SpreadCandidate (internal, not exported)

```python
@dataclass
class SpreadCandidate:
    """Internal: a validated spread pair from the option chain."""
    short_strike: float
    long_strike: float
    credit: float
    spread_width: float
    dte: int
    max_loss: float
    short_delta: float
```

## Detailed Design

### 1. Bull Put Spread Scoring

| Component | Condition | Points |
|-----------|-----------|--------|
| mdia_inflow | `mdia_regime_inflow == 1` OR `mdia_regime_strong_inflow == 1` | +25 |
| whale_sponsored | `whale_regime_broad_accum == 1` OR `whale_regime_strategic_accum == 1` | +20 |
| mvrv_macro_bullish | Any of: `mvrv_ls_regime_call_confirm`, `mvrv_ls_regime_call_confirm_recovery`, `mvrv_ls_regime_call_confirm_trend` | +20 |
| no_option_put | `signal_option_put == 0` | +15 |
| sentiment_safe | `sent_bucket_extreme_greed == 0` | +10 |
| whale_not_distributing | `whale_regime_distribution == 0` AND `whale_regime_distribution_strong == 0` | +10 |

**Practical maximum: 100.** Note: `whale_sponsored` (+20) and `whale_not_distributing` (+10) are compatible — a whale can be accumulating (sponsored) and simultaneously not distributing. The old `whale_regime_mixed` bonus was removed because `mixed` and `sponsored` are mutually exclusive in the whale regime classifier, which would have capped the real max at 90.

### 2. Bull Put Spread Hard Vetoes

| Veto | Condition | Reason String |
|------|-----------|---------------|
| Option put active | `signal_option_put == 1` | `OPTION_PUT_ACTIVE` |
| Extreme greed persist | `sent_extreme_greed_persist_5d == 1` | `EXTREME_GREED_PERSIST_5D` |
| MVRV macro bearish | Any bearish MVRV flag active | `MVRV_MACRO_BEARISH` |
| Whale distribution strong | `whale_regime_distribution_strong == 1` | `WHALE_DISTRIB_STRONG` |
| Higher-priority signal | `higher_priority_active == True` (passed by caller) | `HIGHER_PRIORITY_SIGNAL` |
| ATR expansion | `atr_ratio > config.atr_expansion_threshold` | `ATR_EXPANSION(x.xx>1.5)` |
| Fusion state conflict | `fusion_state` in (`bear_continuation`, `distribution_risk`) | `FUSION_STATE_BEARISH` |
| Condor precedence | `condor_eligible == True` (passed by caller) | `CONDOR_PRECEDENCE` |

### 3. Bear Call Spread Scoring

| Component | Condition | Points |
|-----------|-----------|--------|
| no_mdia_inflow_or_aging | (`mdia_regime_inflow == 0` AND `mdia_regime_strong_inflow == 0`) OR `mdia_regime_aging == 1` | +25 |
| whale_distribution | `whale_regime_distribution == 1` OR `whale_regime_distribution_strong == 1` | +20 |
| mvrv_macro_bearish | Any of: `mvrv_ls_regime_put_confirm`, `mvrv_ls_regime_bear_continuation`, `mvrv_ls_early_rollover`, `mvrv_ls_weak_downtrend`, `mvrv_ls_regime_distribution_warning` | +20 |
| no_option_call | `signal_option_call == 0` | +15 |
| sentiment_greed_or_flat | `sent_bucket_greed == 1` OR `sent_is_flattening == 1` | +10 |
| whale_not_accumulating | `whale_regime_broad_accum == 0` AND `whale_regime_strategic_accum == 0` | +10 |

**Practical maximum: 100.** Same fix applied: `whale_distribution` (+20) and `whale_not_accumulating` (+10) are compatible — distributing whales are by definition not accumulating.

### 4. Bear Call Spread Hard Vetoes

| Veto | Condition | Reason String |
|------|-----------|---------------|
| Option call active | `signal_option_call == 1` | `OPTION_CALL_ACTIVE` |
| MDIA strong inflow | `mdia_regime_strong_inflow == 1` | `MDIA_STRONG_INFLOW` |
| MVRV macro bullish | Any bullish MVRV flag active | `MVRV_MACRO_BULLISH` |
| Whale sponsored | `whale_regime_broad_accum == 1` OR `whale_regime_strategic_accum == 1` | `WHALE_SPONSORED` |
| Higher-priority signal | `higher_priority_active == True` (passed by caller) | `HIGHER_PRIORITY_SIGNAL` |
| ATR expansion | `atr_ratio > config.atr_expansion_threshold` | `ATR_EXPANSION(x.xx>1.5)` |
| Fusion state conflict | `fusion_state` in (`strong_bullish`, `early_recovery`, `momentum`) | `FUSION_STATE_BULLISH` |
| Condor precedence | `condor_eligible == True` (passed by caller) | `CONDOR_PRECEDENCE` |

### 5. Option Chain Column Normalization

Before filtering, `normalize_chain_columns()` maps exchange-specific schemas to canonical columns:

| Canonical Column | Deribit Source | Bybit Source | Notes |
|-----------------|---------------|--------------|-------|
| `strike` | `strike` | `strike` | — |
| `side` | `option_type` (lowered) | `option_type` or `side` | Normalized to "put"/"call" |
| `delta` | `delta` (signed) | `delta` (signed) | Stored as absolute value internally |
| `bid` | `bid` | `bid_price` | — |
| `ask` | `ask` | `ask_price` | — |
| `dte` | `days_to_expiry` or `dte` | `dte` or `days_to_expiry` | — |
| `spread_pct` | `spread_pct` or computed `(ask-bid)/mid` | `spread_pct` or computed | — |

If a required column is missing after normalization, the chain is treated as empty (chain_valid=False, chain_rejection_reason="MISSING_COLUMNS").

### 6. MVRV-Derived Strike Boundaries

The income gates use MVRV metrics to compute on-chain price boundaries that anchor strike selection to structural valuation levels rather than relying solely on delta.

#### Which MVRV for Which Strategy

| Strategy | Primary MVRV | Why | Boundary Computation |
|----------|-------------|-----|---------------------|
| **Bull Put Spread** | **MVRV-60D** | Measures profitability of recent buyers (new money). Best single metric for local bottoms. When negative, recent buyers are underwater → sell exhaustion → price floor. Short-term horizon matches the 9–45 DTE of the spread. | `floor = spot / mvrv_60d` |
| **Bear Call Spread** | **MVRV Composite** | Average profitability across ALL holder cohorts (1d–365d). When highly positive, broad profit-taking pressure builds across every timeframe simultaneously → price ceiling. The composite captures the "everyone is in profit" state that precedes corrections, which is more relevant for capping upside than any single cohort. | `ceiling = cost_basis * mvrv_composite_p90_180d` |

#### Rationale for the Split

- **Bull put (floor):** You need to know where *recent* buyers break even, because that's where short-term selling exhaustion occurs. MVRV-60D isolates the cohort most likely to panic-sell on a dip. If they're already underwater, further selling pressure is limited → safe to sell puts below.

- **Bear call (ceiling):** You need to know where *broad* profit-taking kicks in. A single cohort being profitable doesn't cap price — but when the composite is elevated, it means holders across 7d, 30d, 60d, 90d, 180d, 365d are all in profit simultaneously. That's when coordinated selling pressure builds from multiple timeframes → safe to sell calls above.

- **Why not MVRV-60D for both?** MVRV-60D is excellent for floors (local bottoms) but less reliable for ceilings. Recent buyers being profitable doesn't cap price — they might just hold. The ceiling needs the composite because it captures the "everyone wants to take profit" state.

- **Why not MVRV Composite for both?** The composite is too slow for floors. It averages across 365d holders who don't react to short-term dips. MVRV-60D reacts faster and correlates better with local bottoms.

#### Boundary Computation

```python
def compute_strike_boundaries(
    spot_price: float,
    mvrv_60d: float,
    mvrv_composite: float,
    mvrv_composite_p90_180d: float,
) -> tuple[float, float]:
    """
    Compute MVRV-derived price boundaries for strike selection.
    
    Returns:
        (floor_price, ceiling_price)
        - floor_price: short put strikes must be at or below this
        - ceiling_price: short call strikes must be at or above this
    """
    # Bull put floor: cost basis of 60-day cohort
    cost_basis_60d = spot_price / mvrv_60d
    floor_price = cost_basis_60d
    
    # Bear call ceiling: cost basis scaled by composite P90
    # cost_basis_composite = spot / mvrv_composite (current broad cost basis)
    # ceiling = cost_basis_composite * mvrv_composite_p90_180d
    # Simplified: ceiling = (spot / mvrv_composite) * mvrv_composite_p90_180d
    cost_basis_composite = spot_price / mvrv_composite
    ceiling_price = cost_basis_composite * mvrv_composite_p90_180d
    
    return floor_price, ceiling_price
```

#### Example

```
spot = 80,000
mvrv_60d = 1.09
mvrv_composite = 1.05
mvrv_composite_p90_180d = 1.12

floor = 80,000 / 1.09 = 73,394  (bull put short strike must be ≤ this)
ceiling = (80,000 / 1.05) * 1.12 = 85,333  (bear call short strike must be ≥ this)
```

#### Integration with Chain Filtering

The MVRV boundary is applied as an additional filter step between OTM filter and delta filter:

1. Side filter
2. OTM filter
3. **MVRV boundary filter** ← NEW
   - Bull put: keep only strikes ≤ `floor_price`
   - Bear call: keep only strikes ≥ `ceiling_price`
4. Delta filter
5. DTE filter
6. Bid/ask filter

If the MVRV boundary eliminates all candidates (no strikes exist at or beyond the boundary), the gate falls back to delta-only selection but flags `chain_rejection_reason="MVRV_BOUNDARY_TOO_TIGHT"` as a diagnostic. This prevents the boundary from making the gate permanently inactive when option chains don't have strikes at extreme levels.

#### P90 Computation

`mvrv_composite_p90_180d` is the 90th percentile of MVRV Composite over the trailing 180 days. Using P90 instead of the raw max avoids anchoring to single-day spikes that aren't representative of sustainable ceilings. This value should be pre-computed in the feature pipeline (same as other rolling statistics) and available in the feature row.

### 7. Option Chain Filtering

```python
def filter_option_chain(
    chain_df: pd.DataFrame,
    side: str,  # "put" or "call"
    spot_price: float,
    config: IncomeGateConfig,
    dte_mode: str = "tactical",  # "tactical" or "income"
    strike_boundary: Optional[float] = None,  # MVRV-derived floor or ceiling
) -> pd.DataFrame:
```

Filter pipeline (applied sequentially):
1. **Side filter**: Keep only puts (bull put) or calls (bear call)
2. **OTM filter**: Puts with strike < spot; Calls with strike > spot
3. **MVRV boundary filter**: If `strike_boundary` provided — puts: keep strike ≤ boundary; calls: keep strike ≥ boundary
4. **Delta filter**: `config.min_delta <= abs(delta) <= config.max_delta`
5. **DTE filter**: Based on `dte_mode` — tactical (9–21) or income (21–45)
6. **Bid/ask filter**: `spread_pct <= config.max_bid_ask_spread_pct`
7. **Sort**: By absolute delta descending (prefer higher delta within range for more credit)

### 8. Spread Selection

```python
def select_spread(
    filtered_chain: pd.DataFrame,
    side: str,
    spot_price: float,
    config: IncomeGateConfig,
) -> Optional[SpreadCandidate]:
```

Logic:
1. Iterate short leg candidates from filtered chain (sorted by delta desc)
2. For each short leg, find the long leg: same expiry, same side, strike offset within `config.min_spread_width_pct` to `config.max_spread_width_pct` of spot
   - Bull put: long strike = short strike - width (further OTM)
   - Bear call: long strike = short strike + width (further OTM)
3. Compute credit = short_bid - long_ask
4. Compute spread_width = abs(short_strike - long_strike)
5. Validate: `credit / spread_width >= config.min_credit_pct`
6. If valid, return SpreadCandidate; else try next short leg candidate
7. If no candidate passes, return None

### 9. Evaluate Function Flow

```
evaluate_bull_put_gate(row, chain_df, spot_price, ...):
    1. score, components = compute_bull_put_score(row)
    2. vetoes = check_bull_put_vetoes(row, atr_ratio, fusion_state, higher_priority_active, condor_eligible)
    3. regime_eligible = (score >= threshold) and (len(vetoes) == 0)
    4. if not regime_eligible:
         return IncomeGateResult(regime_eligible=False, eligible=False, chain_valid=False, ...)
    5. if chain_df is None or chain_df.empty:
         return IncomeGateResult(regime_eligible=True, eligible=False, chain_valid=False,
                                 chain_rejection_reason="NO_CHAIN_DATA", ...)
    6. normalized = normalize_chain_columns(chain_df)
    7. mvrv_60d = row.get("mvrv_60d", row.get("mvrv_usd_60d"))
       floor_price = spot_price / mvrv_60d if mvrv_60d and mvrv_60d > 0 else None
    8. filtered = filter_option_chain(normalized, "put", spot_price, config, dte_mode,
                                      strike_boundary=floor_price)
    9. candidate = select_spread(filtered, "put", spot_price, config)
   10. if candidate is None:
         return IncomeGateResult(regime_eligible=True, eligible=False, chain_valid=False,
                                 chain_rejection_reason="NO_ACCEPTABLE_SPREAD", ...)
   11. return IncomeGateResult(regime_eligible=True, eligible=True, chain_valid=True,
                               spread_guidance=..., short_strike=..., ...)

evaluate_bear_call_gate(row, chain_df, spot_price, ...):
    1-4. Same regime scoring/veto pattern
    5-6. Same chain_df and normalization checks
    7. mvrv_composite = row.get("mvrv_composite")
       mvrv_composite_p90 = row.get("mvrv_composite_p90_180d")
       if mvrv_composite and mvrv_composite > 0 and mvrv_composite_p90:
           ceiling_price = (spot_price / mvrv_composite) * mvrv_composite_p90
       else:
           ceiling_price = None
    8. filtered = filter_option_chain(normalized, "call", spot_price, config, dte_mode,
                                      strike_boundary=ceiling_price)
    9-11. Same spread selection and result construction
```

### 10. Integration Points

#### In `signals/options.py` — DECISION_STRATEGY_MAP additions:

```python
"BULL_PUT_SPREAD": {
    "structure": OptionStructure.SHORT_PUT_SPREAD,
    "spread_guidance": SpreadGuidance(
        width_pct=0.05, take_profit_pct=0.50,
        max_hold_days=18, stop_loss_pct=0.02, scale_down_day=12
    ),
    "strike_guidance": "Short put at 0.20 delta OTM, long put 3-8% below",
    "dte_range": "9-21 days (tactical) or 21-45 days (income)",
    "rationale": "Collect premium below support in bullish regime"
},
"BEAR_CALL_SPREAD": {
    "structure": OptionStructure.SHORT_CALL_SPREAD,
    "spread_guidance": SpreadGuidance(
        width_pct=0.05, take_profit_pct=0.50,
        max_hold_days=18, stop_loss_pct=0.02, scale_down_day=12
    ),
    "strike_guidance": "Short call at 0.20 delta OTM, long call 3-8% above",
    "dte_range": "9-21 days (tactical) or 21-45 days (income)",
    "rationale": "Collect premium above resistance in bearish regime"
}
```

**Note on `stop_loss_pct`:** This uses the same unit as existing SpreadGuidance consumers — percentage move in the underlying (spot). `0.02` means "exit if spot moves 2% toward the short strike." This is consistent with how `condor_gate` and downstream position management interpret the field. The earlier draft used `0.40` meaning "40% of spread width" which is a different unit and would break downstream consumers.

#### Caller integration (wherever condor gate is called):

The income gates are evaluated after the condor gate. The caller passes `condor_eligible` from the condor gate result. If condor is eligible, income gates are vetoed via `CONDOR_PRECEDENCE`. Otherwise, evaluate both income gates and return the one with the higher score (if any is eligible).

### 11. SpreadGuidance Defaults

| Parameter | Bull Put Spread | Bear Call Spread | Unit |
|-----------|----------------|-----------------|------|
| width_pct | 0.05 (5% of spot) | 0.05 (5% of spot) | fraction of spot |
| take_profit_pct | 0.50 (close at 50% of max profit) | 0.50 | fraction of max credit |
| max_hold_days | 18 | 18 | calendar days |
| stop_loss_pct | 0.02 (exit if spot moves 2% toward short strike) | 0.02 | fraction of spot |
| scale_down_day | 12 (reduce to 25% at day 12) | 12 | calendar days |

## Error Handling

| Scenario | Behavior |
|----------|----------|
| `chain_df` is None or empty | `regime_eligible` still computed; `chain_valid=False`, `chain_rejection_reason="NO_CHAIN_DATA"` |
| Chain columns missing after normalization | `chain_valid=False`, `chain_rejection_reason="MISSING_COLUMNS"` |
| MVRV boundary eliminates all candidates | Falls back to delta-only; `chain_rejection_reason="MVRV_BOUNDARY_TOO_TIGHT"` if no candidates remain after fallback either |
| `mvrv_60d` or `mvrv_composite` missing/zero | Boundary skipped for that side; delta-only selection used |
| No contracts pass delta/DTE/bid-ask filters | `chain_valid=False`, `chain_rejection_reason="NO_CONTRACTS_PASS_FILTERS"` |
| No spread pair meets credit threshold | `chain_valid=False`, `chain_rejection_reason="NO_ACCEPTABLE_SPREAD"` |
| `atr_ratio` is None | ATR veto is skipped (same as condor gate behavior) |
| `fusion_state` is None | Fusion state veto is skipped; scoring still uses row-level flags |

## Correctness Properties

1. **Regime scoring is deterministic and stateless** — given the same feature row, the same score and vetoes are produced regardless of chain data.
2. **Chain filtering never modifies the input DataFrame** — all operations return new DataFrames.
3. **Condor gate is never modified** — income gates only read the condor result via `condor_eligible` param.
4. **Mutual exclusivity of directional gates** — bull put vetoes on bearish fusion states; bear call vetoes on bullish fusion states. Both cannot be eligible simultaneously for the same row (by construction of fusion states).
5. **Credit is always positive when eligible** — `select_spread` only returns candidates where `short_bid > long_ask` and `credit / spread_width >= min_credit_pct`.
6. **MVRV boundary is a soft constraint** — if the boundary eliminates all chain candidates, the gate falls back to delta-only selection rather than permanently blocking. This ensures the gate remains functional when option chains lack extreme strikes.
7. **Strike boundaries are structurally sound** — bull put short strike ≤ floor (cost basis of 60d cohort); bear call short strike ≥ ceiling (composite P90 scaled cost basis). Violations are impossible when eligible=True.

## Testing Strategy

- Unit tests for scoring functions with synthetic feature rows
- Unit tests for veto logic (each veto condition independently)
- Unit tests for `normalize_chain_columns` with Deribit and Bybit sample schemas
- Unit tests for option chain filtering with mock chain DataFrames
- Unit tests for `select_spread` with edge cases (no long leg available, credit too low)
- Integration test: full evaluate flow with realistic feature row + chain data
- Two-layer test: verify `regime_eligible=True, chain_valid=False` state is reachable
- Regression: ensure condor gate behavior is unchanged (no imports from income_gate)

## File Changes Summary

| File | Change |
|------|--------|
| `signals/income_gate.py` | **New** — all income gate logic |
| `signals/options.py` | Add DECISION_STRATEGY_MAP entries for BULL_PUT_SPREAD and BEAR_CALL_SPREAD |
| `signals/__init__.py` | Export IncomeGateResult, evaluate_bull_put_gate, evaluate_bear_call_gate |
