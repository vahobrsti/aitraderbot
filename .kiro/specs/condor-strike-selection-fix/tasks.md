# Implementation Plan

## Overview

Replace fixed ±10% spot-band condor strike selection with a shared,
credit-filtered, delta-ranked selector. Credit adequacy (`min_credit_pct`) is the
hard eligibility constraint; the policy Δ0.20 is the ranking target. The selector
is shared by the setup builder and the live-execution path so they cannot
diverge, and a sub-minimum credit becomes a blocking validation failure.

## Tasks

- [x] 1. Add condor delta band to policy config
  - In `execution/services/policy.py`, add `min_delta: float = 0.12` and
    `max_delta: float = 0.35` to `CondorConfig` and set them in the `POLICY_V1`
    `condor=CondorConfig(...)` instance.
  - Retain `spot_call_band` / `spot_put_band` only as an explicit fallback.
  - _Requirements: 2.1_

- [x] 2. Build the shared credit-filtered, delta-ranked selector
  - Add `select_condor_structure(options, spot, condor_cfg, delta_target, mvrv_strikes)`
    (module-level in `execution/services/trade_setup.py`, or a shared helper both
    it and `deribit_entry.py` import).
  - Group options by expiry within the resolved DTE band; build short-call and
    short-put candidates in `[min_delta, max_delta]` (exclude null delta), each
    paired with its wing at `wing_offset_usd`.
  - Form complete four-leg candidates, compute `net_credit` and
    `credit_pct = net_credit / wing_width`.
  - _Requirements: 2.1, 2.2, 3.4_

- [x] 3. Apply credit eligibility filter and deterministic ranking
  - Drop candidates with `credit_pct < min_credit_pct`.
  - Rank survivors by: combined short-delta closeness to `delta_target`, then
    MVRV alignment bonus, then liquidity/bid-ask, then a stable
    `(expiry, short_call_strike, short_put_strike)` tiebreak.
  - Return: best qualified candidate; else a "constructible-but-unqualified"
    result; else `None` when nothing is constructible.
  - _Requirements: 2.2, 2.3, 2.5_

- [x] 4. Wire selector into `_build_condor_setup` with blocking + None policy
  - Replace band-based short-strike selection with `select_condor_structure`.
  - Qualified → build setup, `validation_passed = True`.
  - Constructible-but-unqualified → build best-effort setup and append credit
    shortfall to `blocking` (not `warnings`) so `validation_passed = False`.
  - Nothing constructible → `return None`.
  - Preserve the existing four-leg construction, metrics, exit rules, and the
    `>8% OTM` distance warnings.
  - _Requirements: 2.4, 2.5, 3.1, 3.2, 3.3_

- [x] 5. Align live-execution and scan paths to the shared selector
  - Update `_plan_condor` in `execution/services/deribit_entry.py` to select
    strikes via `select_condor_structure` and to treat sub-minimum credit as a
    hard reject rather than a warning.
  - Update condor target logic in `execution/management/commands/scan_entries.py`
    to read from the shared routine (or clearly mark its output as indicative).
  - _Requirements: 2.1, 2.2, 2.4_

- [x] 6. Add focused unit tests
  - Cover: nearest-target-delta ranking; credit eligibility overriding a
    sub-minimum nearest-Δ0.20 structure; null-delta exclusion; shared-expiry and
    deterministic tie resolution; MVRV never selecting a sub-minimum structure;
    sub-minimum → blocking; no in-band candidates → `None`; non-condor build
    unchanged.
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 3.5, 3.6_

- [x] 7. Verify fix and non-regression end-to-end
  - Run `python manage.py dev_signal_cycle --date 2026-07-28`; confirm short
    strikes near Δ0.20 (SC ~67000 / SP ~61000), credit ≥ 15% of width, improved
    R:R, `validation_passed = True`.
  - Confirm a directional `PUT`/`CALL` date builds identically to pre-fix output.
  - Confirm setup and execution paths produce the same structure for the date.
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

## Task Dependency Graph

```
Task 1 (config delta band)
      │
      ▼
Task 2 (candidate builder) ──▶ Task 3 (credit filter + ranking)
                                     │
                          ┌──────────┴───────────┐
                          ▼                       ▼
              Task 4 (wire into setup)   Task 5 (align execution/scan)
                          │                       │
                          └───────────┬───────────┘
                                      ▼
                              Task 6 (unit tests)
                                      ▼
                              Task 7 (e2e verify)
```

- Task 1: no prerequisites.
- Task 2 depends on Task 1 (delta-band config).
- Task 3 depends on Task 2 (needs candidates to filter/rank).
- Tasks 4 and 5 depend on Task 3 (both consume the selector); they can proceed in
  parallel.
- Task 6 depends on Tasks 4 and 5.
- Task 7 depends on Task 6.

```json
{
  "waves": [
    { "wave": 1, "tasks": ["1"] },
    { "wave": 2, "tasks": ["2"] },
    { "wave": 3, "tasks": ["3"] },
    { "wave": 4, "tasks": ["4", "5"] },
    { "wave": 5, "tasks": ["6"] },
    { "wave": 6, "tasks": ["7"] }
  ]
}
```

## Notes

- The selector mirrors the income gate's existing enumerate-and-filter approach
  (`filter_option_chain` / `_find_all_valid_spreads`); reuse patterns where
  practical instead of inventing new ones.
- `trade_setup._build_condor_setup` is the authoritative structure source;
  execution consumes the same selection.
- MVRV is a ranking preference only — it must never select a sub-minimum-credit
  structure over a qualified one.
