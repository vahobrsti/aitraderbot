# Design Document

## Overview

This change tightens the MVRV-derived strike boundaries from P5/P95 to P10/P90, and converts the boundary from a "soft filter" (falls back if no candidates) to a "hard gate" (vetoes the signal if we're in extreme territory, rejects if no candidates pass the boundary).

The philosophy: if MVRV-60D has spent 90% of the last 6 months within a band and we're currently *outside* that band, there's no reliable on-chain anchor for a credit spread. Better to skip than to place a spread with a weak or nonexistent support/resistance reference.

## Changes by File

### 1. `features/metrics/mvrv_composite.py`

**Add P10/P90 computation** alongside existing P5/P95:

```python
# Existing (kept for backward compatibility):
feats["mvrv_60d_p5_180d"] = mvrv_60d.rolling(180, min_periods=60).quantile(0.05)
feats["mvrv_60d_p95_180d"] = mvrv_60d.rolling(180, min_periods=60).quantile(0.95)

# New:
feats["mvrv_60d_p10_180d"] = mvrv_60d.rolling(180, min_periods=60).quantile(0.10)
feats["mvrv_60d_p90_180d"] = mvrv_60d.rolling(180, min_periods=60).quantile(0.90)
```

No other changes to this file. P5/P95 remain for any other consumers.

### 2. `signals/income_gate.py`

#### a) New vetoes in `check_bull_put_vetoes`:

```python
# After existing MVRV_FLOOR_INVALID veto:
mvrv_60d_p10 = row.get("mvrv_60d_p10_180d")
if mvrv_60d_valid and mvrv_60d_p10 is not None:
    try:
        p10_val = float(mvrv_60d_p10)
        if p10_val == p10_val and float(mvrv_60d) < p10_val:
            vetoes.append("MVRV_BELOW_P10_BAND")
    except (ValueError, TypeError):
        pass
```

#### b) New veto in `check_bear_call_vetoes`:

```python
# New veto: current MVRV-60D above P90 = no ceiling anchor
mvrv_60d_raw = row.get("mvrv_60d", row.get("mvrv_usd_60d"))
mvrv_60d_p90 = row.get("mvrv_60d_p90_180d")
if mvrv_60d_raw is not None and mvrv_60d_p90 is not None:
    try:
        mvrv_val = float(mvrv_60d_raw)
        p90_val = float(mvrv_60d_p90)
        if mvrv_val == mvrv_val and p90_val == p90_val and mvrv_val > p90_val:
            vetoes.append("MVRV_ABOVE_P90_BAND")
    except (ValueError, TypeError):
        pass
```

#### c) Floor computation in `evaluate_bull_put_gate` — P5 → P10:

```python
# Change from:
mvrv_60d_p5 = row.get("mvrv_60d_p5_180d")
# To:
mvrv_60d_p10 = row.get("mvrv_60d_p10_180d")
# And use p10 in the underwater fallback:
floor_price = cost_basis * float(mvrv_60d_p10)
```

#### d) Ceiling computation in `evaluate_bear_call_gate` — P95 → P90:

```python
# Change from:
mvrv_60d_p95 = row.get("mvrv_60d_p95_180d")
# To:
mvrv_60d_p90 = row.get("mvrv_60d_p90_180d")
# And:
ceiling_price = cost_basis * float(mvrv_60d_p90)
```

#### e) Remove soft fallback in `filter_option_chain`:

```python
# BEFORE (soft):
if strike_boundary is not None and not df.empty:
    if side == "put":
        boundary_filtered = df[df["strike"] <= strike_boundary]
    else:
        boundary_filtered = df[df["strike"] >= strike_boundary]
    if not boundary_filtered.empty:
        df = boundary_filtered

# AFTER (hard):
if strike_boundary is not None and not df.empty:
    if side == "put":
        df = df[df["strike"] <= strike_boundary]
    else:
        df = df[df["strike"] >= strike_boundary]
```

No fallback. If empty, `filter_option_chain` returns empty → `evaluate_*_gate` returns with `chain_rejection_reason="NO_CONTRACTS_PASS_FILTERS"`.

#### f) Update `compute_strike_boundaries` signature:

```python
def compute_strike_boundaries(
    spot_price: float,
    mvrv_60d: Optional[float],
    mvrv_60d_p10_180d: Optional[float] = None,  # was p5
    mvrv_60d_p90_180d: Optional[float] = None,   # was p95
) -> tuple[Optional[float], Optional[float]]:
```

### 3. `features/metrics/credit_spread_label.py`

Replace `mvrv_60d_p5_180d` → `mvrv_60d_p10_180d` and `mvrv_60d_p95_180d` → `mvrv_60d_p90_180d`:

```python
mvrv_60d_p10 = (
    df["mvrv_60d_p10_180d"].astype(float)
    if "mvrv_60d_p10_180d" in df.columns
    else pd.Series(np.nan, index=df.index)
)
mvrv_60d_p90 = (
    df["mvrv_60d_p90_180d"].astype(float)
    if "mvrv_60d_p90_180d" in df.columns
    else pd.Series(np.nan, index=df.index)
)
```

Fallback to 4% OTM remains when columns are missing.

## Risk Assessment

- **Tighter boundaries = fewer signals.** P10/P90 is narrower than P5/P95, so some previously valid setups will now be vetoed. This is intentional — quality over quantity.
- **Hard boundary removes last-resort trades.** Previously, if no strikes passed the MVRV filter, the system would still try to find *something*. Now it won't. If the chain doesn't support the boundary, we don't trade.
- **Backward compatibility:** P5/P95 fields remain in `mvrv_composite.py` output. No downstream breakage for any module still referencing them.
- **`compute_strike_boundaries` callers:** This utility is used in `income_gate.py` (internal to evaluate functions) and possibly in `manual_setup.py`. The parameter rename is backward-compatible via keyword args, but `manual_setup.py` should be checked.

## Out of Scope

- Changing the `_compute_required_credit_pct` adaptive threshold logic (separate concern).
- Modifying condor gate boundaries (different strategy, different constraints).
- Adding new DTE windows or delta bands.
