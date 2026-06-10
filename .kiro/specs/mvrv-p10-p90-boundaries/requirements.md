# Requirements Document

## Introduction

Replace the P5/P95 percentile boundaries in the MVRV-60D strike selection logic with P10/P90. The core insight: if 90% of the time over the last 6 months, MVRV-60D has been within this range, and the current MVRV is already *outside* that range, there is no meaningful support/resistance anchor — the signal should be dropped entirely (hard veto) rather than falling back to weaker heuristics.

This change affects three files:
1. `features/metrics/mvrv_composite.py` — produce `mvrv_60d_p10_180d` and `mvrv_60d_p90_180d`
2. `signals/income_gate.py` — consume P10/P90 for strike boundaries, add hard vetoes when current MVRV is outside the P10–P90 band, remove soft fallback
3. `features/metrics/credit_spread_label.py` — align labeling logic with the new boundaries

## Glossary

- **P10/P90**: The 10th and 90th percentile of MVRV-60D over a rolling 180-day window. Represents the band within which MVRV has spent 80% of the last 6 months.
- **Floor (Bull Put)**: The MVRV-derived price below which the short put strike should be placed. Computed as `cost_basis = spot / mvrv_60d`, adjusted by P10 when buyers are underwater.
- **Ceiling (Bear Call)**: The MVRV-derived price above which the short call strike should be placed. Computed as `cost_basis × P90`.
- **Hard Veto**: A condition that blocks the trade entirely, regardless of score.
- **Soft Fallback**: Current behavior where the MVRV boundary filter is ignored if it eliminates all candidates. Being removed.

## Requirements

### Requirement 1: Compute P10/P90 in mvrv_composite.py

**User Story:** As the income gate, I need P10 and P90 rolling percentiles of MVRV-60D over 180 days, so that strike boundaries use the 90% historical range rather than the extreme 5%/95% tails.

#### Acceptance Criteria

1. WHEN `mvrv_composite.py` calculates features, IT SHALL produce `mvrv_60d_p10_180d` as the 10th percentile of `mvrv_60d` over a rolling 180-day window (min_periods=60).
2. WHEN `mvrv_composite.py` calculates features, IT SHALL produce `mvrv_60d_p90_180d` as the 90th percentile of `mvrv_60d` over a rolling 180-day window (min_periods=60).
3. THE existing `mvrv_60d_p5_180d` and `mvrv_60d_p95_180d` fields SHALL be retained for backward compatibility but are no longer used by the income gate strike boundary logic.

### Requirement 2: Bull Put Floor Uses P10 and Vetoes When Outside Band

**User Story:** As the trading system, I want the bull put floor to use P10 instead of P5, and to hard-veto when current MVRV-60D is already below P10 (meaning we're in extreme territory with no valid floor anchor).

#### Acceptance Criteria

1. WHEN computing the bull put floor for strike selection, THE system SHALL use `mvrv_60d_p10_180d` instead of `mvrv_60d_p5_180d` as the fallback multiplier when buyers are underwater (mvrv_60d < 1).
2. WHEN `mvrv_60d < mvrv_60d_p10_180d` (current MVRV is below the 10th percentile of the last 6 months), THE bull put gate SHALL add a hard veto `MVRV_BELOW_P10_BAND` and reject the signal.
3. THE `compute_strike_boundaries` utility function SHALL accept `mvrv_60d_p10_180d` instead of `mvrv_60d_p5_180d` as the floor parameter.
4. THE soft fallback in `filter_option_chain` SHALL be removed for the MVRV boundary filter — if no strikes pass the boundary, the chain selection returns empty (rejected).

### Requirement 3: Bear Call Ceiling Uses P90 and Vetoes When Outside Band

**User Story:** As the trading system, I want the bear call ceiling to use P90 instead of P95, and to hard-veto when current MVRV-60D is already above P90 (meaning we're at extreme highs with no valid ceiling anchor).

#### Acceptance Criteria

1. WHEN computing the bear call ceiling for strike selection, THE system SHALL use `mvrv_60d_p90_180d` instead of `mvrv_60d_p95_180d` for the ceiling calculation (`ceiling = cost_basis × P90`).
2. WHEN `mvrv_60d > mvrv_60d_p90_180d` (current MVRV is above the 90th percentile of the last 6 months), THE bear call gate SHALL add a hard veto `MVRV_ABOVE_P90_BAND` and reject the signal.
3. THE `compute_strike_boundaries` utility function SHALL accept `mvrv_60d_p90_180d` instead of `mvrv_60d_p95_180d` as the ceiling parameter.

### Requirement 4: Credit Spread Label Alignment

**User Story:** As the backtesting system, I need the credit spread labeling logic to use the same P10/P90 boundaries so that labels match production gate behavior.

#### Acceptance Criteria

1. WHEN computing `label_bull_put_safe_14d`, THE labeling module SHALL use `mvrv_60d_p10_180d` instead of `mvrv_60d_p5_180d` for the underwater floor calculation.
2. WHEN computing `label_bear_call_safe_14d`, THE labeling module SHALL use `mvrv_60d_p90_180d` instead of `mvrv_60d_p95_180d` for the ceiling calculation.
3. WHEN `mvrv_60d_p10_180d` or `mvrv_60d_p90_180d` columns are missing from the DataFrame, THE labeling module SHALL fall back to the fixed 4% OTM distance (existing behavior).

### Requirement 5: Remove Soft Boundary Fallback

**User Story:** As the trading system, I want MVRV boundaries to be hard constraints so that we never enter a spread without a valid on-chain anchor.

#### Acceptance Criteria

1. WHEN the MVRV boundary filter in `filter_option_chain` eliminates all candidates for puts, THE function SHALL return an empty DataFrame (no soft fallback to unfiltered set).
2. WHEN the MVRV boundary filter in `filter_option_chain` eliminates all candidates for calls, THE function SHALL return an empty DataFrame (no soft fallback to unfiltered set).
3. THE `chain_rejection_reason` in this case SHALL be `NO_CONTRACTS_PASS_MVRV_BOUNDARY` (distinct from the existing `NO_CONTRACTS_PASS_FILTERS`).
