# Implementation Tasks

## Task 1: Add P10/P90 computation to mvrv_composite.py

- [ ] Add `mvrv_60d_p10_180d` = rolling 180d quantile(0.10) with min_periods=60
- [ ] Add `mvrv_60d_p90_180d` = rolling 180d quantile(0.90) with min_periods=60
- [ ] Keep existing P5/P95 fields unchanged
- [ ] Update the section comment to reflect P10/P90 are the primary boundary anchors

## Task 2: Add P10/P90 band vetoes to income_gate.py

- [ ] In `check_bull_put_vetoes`: add `MVRV_BELOW_P10_BAND` veto when `mvrv_60d < mvrv_60d_p10_180d`
- [ ] In `check_bear_call_vetoes`: add `MVRV_ABOVE_P90_BAND` veto when `mvrv_60d > mvrv_60d_p90_180d`

## Task 3: Replace P5/P95 with P10/P90 in income_gate.py strike boundary computation

- [ ] In `evaluate_bull_put_gate`: change `mvrv_60d_p5_180d` → `mvrv_60d_p10_180d` for underwater floor
- [ ] In `evaluate_bear_call_gate`: change `mvrv_60d_p95_180d` → `mvrv_60d_p90_180d` for ceiling
- [ ] Update `compute_strike_boundaries` function signature: `mvrv_60d_p5_180d` → `mvrv_60d_p10_180d`, `mvrv_60d_p95_180d` → `mvrv_60d_p90_180d`
- [ ] Update docstrings to reflect P10/P90 semantics

## Task 4: Remove soft fallback in filter_option_chain

- [ ] Remove the "Only apply if it doesn't eliminate everything" soft fallback logic
- [ ] Make MVRV boundary a hard filter (if no strikes pass, return empty DataFrame)

## Task 5: Update credit_spread_label.py

- [ ] Replace `mvrv_60d_p5_180d` → `mvrv_60d_p10_180d` for bull put floor calculation
- [ ] Replace `mvrv_60d_p95_180d` → `mvrv_60d_p90_180d` for bear call ceiling calculation
- [ ] Keep fallback to 4% OTM when new columns are missing
- [ ] Update docstring/comments

## Task 6: Check manual_setup.py for P5/P95 references

- [ ] Review `signals/management/commands/manual_setup.py` for `mvrv_60d_p5_180d` / `mvrv_60d_p95_180d` usage
- [ ] Update to use P10/P90 if applicable, or add P10/P90 as additional context in display output
