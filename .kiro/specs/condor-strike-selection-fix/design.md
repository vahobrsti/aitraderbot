# Bugfix Design Document

## Overview

The iron condor builder selects short strikes from fixed ±10% spot bands that
ignore volatility and DTE, landing on near-worthless deep-OTM strikes (Δ~0.06)
whose credit falls far below the policy minimum, yet only records a warning.

This fix replaces the band-based selection with a **credit-filtered,
delta-ranked** condor selector: it enumerates complete four-leg candidates in a
sellable delta band, keeps only those whose credit meets `min_credit_pct`, and
ranks the survivors by closeness to the policy Δ0.20 target (with MVRV alignment
and liquidity as tiebreakers). **Credit adequacy is the hard eligibility
constraint; delta is only the ranking target.** The sub-minimum credit check is
promoted from a warning to a blocking validation failure, and the same selection
routine is shared with the live-execution path so setup and execution cannot
diverge.

## Glossary

- **Short strike**: the option leg the condor sells (short call above spot, short
  put below spot). Distance from spot drives premium collected.
- **Delta target**: absolute option delta the ranker aims for. The policy already
  declares `0.20` for `IRON_CONDOR` via `signal_delta_targets`.
- **Delta band**: `[min_delta, max_delta]` sellable range (`0.12–0.35`, matching
  the income gate) bounding acceptable short strikes.
- **Wing**: protective long leg bought `wing_offset_usd` beyond each short
  strike. Wing width drives max loss.
- **Credit %**: `net_credit / wing_width`; the eligibility constraint is
  `credit_pct >= min_credit_pct` (0.15).
- **Candidate condor**: a complete four-leg structure (short call + call wing +
  short put + put wing) at a single shared expiry.
- **MVRV strikes**: `signal.condor_short_call` / `condor_short_put`, produced by
  `compute_condor_strikes` (drift-based) in `signals/options.py`.

## Bug Details

`_build_condor_setup` (`execution/services/trade_setup.py`) chooses condor short
strikes from fixed ±10% spot bands and only warns when credit is too small:

```python
target_short_call = signal.condor_short_call or spot_price * (1 + condor_cfg.spot_call_band)  # +10%
target_short_put  = signal.condor_short_put  or spot_price * (1 - condor_cfg.spot_put_band)   # -10%
short_call = min([o for o in call_options if strike >= spot],
                 key=lambda x: abs(strike - target_short_call))
...
if credit_pct < condor_cfg.min_credit_pct:
    warnings.append(f"Credit {credit_pct:.1%} below minimum {condor_cfg.min_credit_pct:.1%}")
```

Observed on 2026-07-28 (spot $63,705, ~9 DTE): short strikes landed at Δ0.057
(70000-C) / Δ0.088 (58000-P), credit $127.38 on a $3,000 wing = 4.2% of width vs
the 15% minimum, R:R 1:0.04. `validation_blocking` stayed empty so
`validation_passed` was `True` and the trade passed. The MVRV path makes it
worse: `compute_condor_strikes` takes the *wider* of drift-based and spot-band
levels, so `signal.condor_short_call/put` can only push strikes further OTM.

The live-execution path `_plan_condor` in `execution/services/deribit_entry.py`
(and target logic in `scan_entries.py`) carries the **same two defects**
independently.

## Expected Behavior

- Condor short strikes are chosen from complete four-leg candidates that satisfy
  `min_credit_pct`, ranked toward the policy Δ0.20. On 2026-07-28 this yields
  SC ~67000 (Δ0.19) / SP ~61000 (Δ0.23) → 17–18% credit. (Req 2.1, 2.2)
- MVRV strikes influence *ranking* among already credit-qualified candidates;
  they never move a strike in a way that drops the condor below the credit gate.
  (Req 2.3)
- A constructible condor whose best candidate is still below `min_credit_pct` is
  rejected via blocking validation, not a silent warning; when no candidate on
  the date can meet the minimum, no passing setup is returned. (Req 2.4, 2.5)
- The setup path and the live-execution path use the same selection routine, so
  an approved structure is the one executed. (Req 2.1–2.5 operationally)
- Non-condor behavior, the four-leg construction, metric formulas, and no-data
  behavior are preserved. (Req 3.1–3.6)

## Hypothesized Root Cause

The ±10% `spot_call_band` / `spot_put_band` are volatility- and DTE-agnostic. At
~9 DTE / ~40% IV a 10% move is ~1.55σ, so the nearest-strike snap lands deep OTM
(Δ~0.06) where premium is negligible. The policy's Δ0.20 target
(`get_signal_delta`) exists but is only consulted for directional spreads. And
because credit is never a selection input — only a post-hoc warning — a poor
structure is returned even when credit-adequate structures exist on the same
chain (454 qualifying combinations existed on 2026-07-28).

## Correctness Properties

### Property 1: Fix

For every input where the current builder returns a condor with
`credit_pct < min_credit_pct` while at least one credit-qualified four-leg
candidate exists on the chain, the fixed builder returns a setup with
`credit_pct >= min_credit_pct` and `validation_passed = True`.

**Validates: Requirements 2.1, 2.2, 2.3**

### Property 2: Credit gate is authoritative

For every constructible condor input, the returned result satisfies exactly one
of: (a) a setup with `credit_pct >= min_credit_pct` and
`validation_passed = True`; (b) a setup with `validation_passed = False` when no
candidate meets the minimum; or (c) `None` when no four-leg candidate can be
constructed/evaluated from available option data.

**Validates: Requirements 2.4, 2.5**

### Property 3: Preservation (invariants)

The fix preserves these invariants rather than byte-identical output:

- Non-`IRON_CONDOR` signal types produce identical results to before.
- No-data behavior is unchanged (missing calls/puts/DTE band → `None`).
- The four-leg structure and all metric formulas (net credit, wing width, max
  profit/loss, R:R, breakevens, contracts, exit rules) are unchanged.
- A previously valid condor remains a passing setup **only if** its newly
  selected structure still satisfies the credit gate; its specific strikes and
  metrics may change because selection now optimizes credit/delta.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**

## Fix Implementation

### 1. Shared credit-filtered, delta-ranked selector (Req 2.1, 2.2)

Introduce one selection routine (e.g. `select_condor_structure`) used by both the
setup builder and live execution. Given the option chain for a signal date, spot,
`condor_cfg`, and the Δ0.20 target:

1. **Group by expiry** within the resolved DTE band so each candidate is a single
   shared-expiry structure (no cross-expiry legs).
2. For each expiry:
   - Build **short-call candidates**: OTM calls (`strike >= spot`) with
     `min_delta <= |delta| <= max_delta`; null-delta rows excluded. Pair each with
     its **call wing** (nearest strike `>= short.strike + wing_offset_usd`).
   - Build **short-put candidates**: OTM puts (`strike <= spot`) in the same delta
     band; pair each with its **put wing** (nearest strike
     `<= short.strike - wing_offset_usd`).
   - Form complete **four-leg condors** from the call-spread × put-spread
     candidates; compute `net_credit` and `credit_pct = net_credit / wing_width`.
3. **Eligibility filter**: drop any candidate with
   `credit_pct < min_credit_pct`. (Hard constraint — this is what guarantees
   Req 2.2.)
4. **Rank** the survivors deterministically by:
   1. combined closeness of both short deltas to `delta_target`
      (`|dc − 0.20| + |dp − 0.20|`),
   2. MVRV alignment bonus (Section 2),
   3. liquidity/bid-ask spread,
   4. a final stable tiebreak on `(expiry, short_call_strike, short_put_strike)`
      so results never depend on DB row order.
5. Return the top-ranked candidate, or a sentinel indicating "constructible but
   none credit-qualified", or `None` if no candidate could be built at all.

Delta is the *ranking target*; the 15% credit rule is the *eligibility
constraint*. This closes the gap where nearest-Δ0.20 strikes do not by themselves
guarantee adequate credit.

### 2. MVRV as a ranking preference (Req 2.3)

MVRV strikes (`signal.condor_short_call` / `condor_short_put`) are applied as a
**ranking bonus among already credit-qualified candidates** — e.g. prefer
candidates whose short strikes sit at or beyond the MVRV cushion levels. They are
never used to move a chosen strike after the fact, so they cannot push a passing
condor below the credit gate. When the signal has no MVRV strikes, the bonus is
zero and ranking falls back to delta/liquidity.

### 3. Explicit, deterministic expiry selection (Req 3.4)

Because several expiries in the DTE band may carry near-Δ0.20 options, expiry is
chosen by evaluating each expiry's best credit-qualified condor and ranking those
structures (step 4 above), not by globally picking the nearest-delta call and
forcing puts to match. The final `(expiry, strikes)` tiebreak makes ties
deterministic.

### 4. Credit gate as blocking + return-None policy (Req 2.4, 2.5)

- If the selector returns a credit-qualified candidate → build the `TradeSetup`;
  `credit_pct >= min_credit_pct` holds by construction and `validation_passed`
  stays `True`.
- If the selector returns "constructible but none qualified" → build the
  best-effort `TradeSetup` for diagnostics and append the credit shortfall to
  **`blocking`** so `validation_passed = False`.
- If the selector returns `None` (no four-leg candidate constructible from
  available data) → `return None`, matching existing no-data behavior.

**None vs failed setup (explicit):**
- `None` = insufficient option data to construct/evaluate any policy-compliant
  candidate (missing calls/puts, no DTE band, or no in-band strikes at all).
- Failed `TradeSetup` (`validation_passed = False`) = a condor is constructible
  but its best candidate's economics fail the credit gate.

### 5. Shared routine across setup and execution (Req operational)

`_build_condor_setup` (`trade_setup.py`) and `_plan_condor`
(`deribit_entry.py`) both call `select_condor_structure`. `trade_setup` remains
the authoritative structure definition; execution consumes the same selection so
it cannot reconstruct a different, poorer structure. `scan_entries.py` target
display is updated to read from the shared routine (or clearly labeled as
indicative only). `spot_call_band` / `spot_put_band` are retained only as an
explicit last-resort fallback and no longer drive normal selection.

### 6. Config (Req 2.1)

Add to `CondorConfig` in `execution/services/policy.py` and the `POLICY_V1`
instance:

```python
min_delta: float = 0.12
max_delta: float = 0.35
```

The Δ0.20 target is read from the existing `signal_delta_targets["IRON_CONDOR"]`;
no new delta constant is introduced.

## Testing Strategy

Automated unit tests (mocked option chains) are the primary verification; the
historical command run is corroborating evidence.

- **Nearest-target-delta ranking**: among credit-qualified candidates, the one
  with shorts closest to Δ0.20 is chosen.
- **Credit eligibility**: a chain where the nearest-Δ0.20 structure is
  sub-minimum but a slightly different credit-qualified structure exists → the
  qualified structure is selected (guards Req 2.2).
- **Null-delta exclusion**: options with null delta are never selected.
- **Shared-expiry / deterministic expiry**: legs share one expiry; tie inputs
  across expiries resolve deterministically regardless of row order.
- **MVRV never breaks the gate**: an MVRV preference does not select a
  sub-minimum structure over a qualified one.
- **Blocking on sub-minimum**: a chain where no candidate qualifies → returned
  setup has `validation_passed = False` with a blocking message.
- **None on no candidates**: missing/short chain → `None`.
- **Non-condor unchanged**: a `PUT`/`CALL` build is byte-identical to pre-fix.
- **Command check**: `python manage.py dev_signal_cycle --date 2026-07-28` →
  SC ~67000 / SP ~61000, credit ≥ 15%, `validation_passed = True`.

## Risks / Downstream

- **Operational consistency (elevated):** `deribit_entry._plan_condor` and
  `scan_entries.py` share the old defects. Because a divergent live structure
  would defeat the safety fix, aligning them via the shared selector is **in
  scope**, not a follow-up. `trade_setup` is the authoritative structure source.
- **Delta availability:** selection requires `OptionSnapshot.delta`; null-delta
  rows are skipped, and a date with no usable deltas yields `None`.
- **Combinatorics:** enumerating call-spread × put-spread candidates per expiry
  is bounded by listed strikes (tens per side) and mirrors the income gate's
  existing approach, so cost is modest; ranking is deterministic.
- **Intended behavior change:** some low-IV/short-DTE dates that previously
  produced a bad passing condor will now FAIL or produce no setup. This reduces
  condor frequency in exactly those regimes and is the intended correction.

## Requirements Traceability

| Requirement | Addressed by |
|-------------|--------------|
| 2.1 volatility/DTE-aware target | Fix §1, §6 |
| 2.2 credit ≥ min when qualifying strikes exist | Fix §1 (eligibility filter) |
| 2.3 reconcile MVRV strikes | Fix §2 (ranking preference) |
| 2.4 reject sub-minimum (blocking) | Fix §4 |
| 2.5 no passing setup when unmeetable | Fix §1, §4 |
| 3.1–3.6 preservation invariants | Property 3 + Testing Strategy |
| operational parity | Fix §5 |
