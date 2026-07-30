# Bugfix Requirements Document

## Introduction

The iron condor trade setup builder selects short strikes using fixed ±10% spot
bands that ignore volatility and days-to-expiry (DTE). At short expiries with
elevated implied volatility, a 10% move is well beyond one standard deviation,
so the selected short strikes are far out-of-the-money and nearly worthless.
The resulting condor collects a net credit far below the policy minimum, but the
system only records a warning instead of rejecting the trade — allowing
structurally poor risk/reward trades to pass through.

Observed on signal date 2026-07-28 (BTC spot $63,705, expiry 2026-08-07, ~9 DTE):
the generated condor collected a net credit of $127.38 against a $3,000 wing
width (4.2% of width) versus the policy minimum of 15% (`min_credit_pct = 0.15`),
producing a 1:0.04 risk/reward (max profit $127.38, max loss $2,872.62). The
selector landed on short strikes at roughly Δ0.057 (70000-C) and Δ0.088
(58000-P). The system emitted the warning "Credit 4.2% below minimum 15.0%" but
did not block the trade.

This is confirmed to be a strike-selection defect, not a DTE availability issue:
at the same 9-DTE expiry, selecting Δ~0.20 short strikes (short call 67000 /
short put 61000) yields 17–18% credit-of-width and clears the gate. Chain
analysis found 454 valid OTM condor combinations on this date meeting the 15%
threshold. The policy already declares a Δ0.20 target for `IRON_CONDOR` via
`get_signal_delta`, but the condor strike selector ignores it.

Affected paths:
- Band-based path: `_build_condor_setup` in `execution/services/trade_setup.py`
  uses `CondorConfig.spot_call_band` / `spot_put_band` (both 0.10) from
  `execution/services/policy.py`.
- MVRV-based path: when the signal carries `condor_short_call` /
  `condor_short_put` (from `compute_condor_strikes` in `signals/options.py`,
  populated via `signals/services.py`), those values override the band defaults.
  This path takes the *wider* of the drift-based and spot-based levels, pushing
  strikes even further OTM and reducing credit further.

## Bug Analysis

### Current Behavior (Defect)

When a condor setup is built, short strikes are chosen by volatility- and
DTE-agnostic rules, and sub-threshold credit is only warned about rather than
blocked.

1.1 WHEN a condor setup is built with no signal-provided condor short strikes THEN the system selects short strikes using fixed ±10% spot bands (`spot_call_band` / `spot_put_band`), independent of volatility and DTE
1.2 WHEN the ±10% spot bands are applied at short DTE with elevated implied volatility THEN the system selects short strikes far out-of-the-money (e.g. Δ~0.057 call, Δ~0.088 put) that collect negligible premium
1.3 WHEN the signal provides MVRV-drift-based condor short strikes THEN the system uses the wider of the drift-based and spot-band levels, pushing short strikes even further OTM and further reducing credit
1.4 WHEN the resulting net credit is below `min_credit_pct` (15% of wing width) THEN the system records a warning ("Credit X% below minimum 15.0%") but leaves `validation_blocking` empty and sets `validation_passed = True`, allowing the trade to pass
1.5 WHEN the selected strikes produce a credit far below the minimum (e.g. 4.2% of width, 1:0.04 risk/reward) THEN the system returns the structurally poor condor setup as a valid trade

### Expected Behavior (Correct)

Short strikes should be selected so the resulting credit meets the policy
minimum, and any condor that still falls below the minimum must be handled
consistently rather than silently warned.

2.1 WHEN a condor setup is built THEN the system SHALL select short strikes using a volatility- and DTE-aware target (e.g. the policy's Δ~0.20 target from `get_signal_delta`, or volatility-scaled bands) rather than fixed ±10% spot bands
2.2 WHEN volatility- and DTE-aware short strikes are selected on a date where qualifying strikes exist THEN the system SHALL produce a condor whose net credit meets or exceeds `min_credit_pct` (15% of wing width)
2.3 WHEN the signal provides MVRV-drift-based condor short strikes THEN the system SHALL reconcile them with the credit-adequacy target so the selected strikes do not fall below the credit minimum
2.4 WHEN a condor's net credit is below `min_credit_pct` after strike selection THEN the system SHALL handle it consistently — either not producing the setup or rejecting it (blocking) rather than returning it as a valid trade with only a warning
2.5 WHEN no strike combination on the signal date can meet the credit minimum THEN the system SHALL NOT return a passing condor setup for that date

### Unchanged Behavior (Regression Prevention)

Existing condor construction, metrics, and non-condor behavior must be preserved.

3.1 WHEN a condor setup produces a credit at or above `min_credit_pct` THEN the system SHALL CONTINUE TO return it as a valid trade setup
3.2 WHEN a condor setup is built THEN the system SHALL CONTINUE TO construct the 4-leg structure (short call, long call wing, short put, long put wing) with wings offset by `wing_offset_usd` and all legs sharing the same expiry
3.3 WHEN a condor setup is built THEN the system SHALL CONTINUE TO compute net credit, wing width, max profit, max loss, risk/reward, breakevens, contracts, and exit rules using the existing formulas
3.4 WHEN a condor setup is built THEN the system SHALL CONTINUE TO resolve the expiry via the existing DTE band so calls and puts share the same expiry
3.5 WHEN the signal type is not `IRON_CONDOR` (e.g. directional spreads) THEN the system SHALL CONTINUE TO select strikes and build setups exactly as before
3.6 WHEN required option data (calls, puts, or a valid DTE band) is unavailable on the signal date THEN the system SHALL CONTINUE TO return no setup as it does today

## Bug Condition and Properties

### Bug Condition

```pascal
FUNCTION isBugCondition(X)
  INPUT: X of type CondorSetupInput   // signal_date, signal_type = IRON_CONDOR, spot_price, option chain
  OUTPUT: boolean

  // The bug manifests when a condor is buildable but the selected short strikes
  // yield credit below the policy minimum, yet a qualifying combination exists.
  setup ← buildCondor(X)               // current strike-selection behavior
  RETURN setup ≠ NULL
     AND (setup.net_credit / setup.wing_width) < condor_cfg.min_credit_pct
     AND existsQualifyingCombo(X, condor_cfg.min_credit_pct)
END FUNCTION
```

### Property: Fix Checking

```pascal
// For every input that currently triggers the bug, the fixed builder must
// either produce a credit-adequate condor or refuse to return a passing one.
FOR ALL X WHERE isBugCondition(X) DO
  setup ← buildCondor'(X)
  ASSERT setup = NULL
      OR setup.validation_passed = FALSE
      OR (setup.net_credit / setup.wing_width) >= condor_cfg.min_credit_pct
END FOR
```

### Property: Preservation Checking

Because the fix changes how *all* condor strikes are selected (credit-filtered,
delta-ranked), a passing condor's strikes/metrics may legitimately change. So
preservation is expressed as invariants, not byte-identical output:

```pascal
// For every non-buggy input, the fixed builder preserves these invariants.
FOR ALL X WHERE NOT isBugCondition(X) DO
  // 1. Non-condor and no-data behavior is identical.
  IF X.signal_type ≠ IRON_CONDOR OR NOT hasOptionData(X) THEN
    ASSERT buildCondor'(X) = buildCondor(X)

  // 2. The four-leg structure and metric formulas are unchanged in form.
  ASSERT structureAndMetricFormulas(buildCondor'(X)) = structureAndMetricFormulas(buildCondor(X))

  // 3. A previously passing condor stays passing ONLY IF its newly selected
  //    structure still satisfies the credit gate (strikes may differ).
  IF buildCondor(X).validation_passed
     AND (buildCondor'(X).net_credit / buildCondor'(X).wing_width) >= min_credit_pct THEN
    ASSERT buildCondor'(X).validation_passed = TRUE
END FOR
```

Where **F** = `_build_condor_setup` before the fix and **F'** = `_build_condor_setup`
after the fix.
