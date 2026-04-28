# Iron Condor Range Gate — Specification

## Label Definition (Frozen)

- **Band:** ±10% from close
- **Horizon:** 7 observations (rows in RawDailyData)
- **Note:** "7 observations" means the next 7 rows in the daily table. RawDailyData has continuous daily entries, so 7 rows ≈ 7 calendar days. If gaps exist, the actual calendar span may exceed 7 days.
- **Outcome classes:** RANGE_7D, BREAKOUT_UP, BREAKOUT_DOWN, BREAKOUT_BOTH
- **Base rate:** 58% of historical days (2017–2026) are RANGE_7D

## Strategy Parameters (Frozen)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Wings | 10% OTM each side | Matches label band |
| DTE | 7–14 days | 7d aligns with label horizon; up to 14d for liquidity |
| Take profit | 50% of max credit | Standard short-vol exit |
| Stop loss | 6% underlying move | See "Stop vs Label Band" below |
| Scale down | Day 5, reduce to 25% | Time-based risk reduction |
| Hard cut | Day 7 | Aligns with label horizon |
| Position size | 50% of base | Conservative for premium selling |

### Stop vs Label Band

The stop (6%) is tighter than the label band (10%) by design. The label asks "did price stay inside ±10% over 7 days?" but the stop protects against intraday/intra-period adverse moves that could widen beyond recovery.

A 6% move against the position means:
- The short strike is being tested
- Delta exposure is accelerating
- Gamma risk is increasing rapidly

Waiting for a full 10% move before stopping would mean the short strike is deep ITM and the position is likely at max loss. The 6% stop exits while the position still has recovery value.

## Score & Gate

- **Threshold:** ≥ 75 (score range 0–100)
- **Precision at threshold:** 67% (vs 58% base)
- **Pass rate:** ~6% of days (~1–2 trades/month)
- **Cooldown:** 5 days between entries

### Score Components

| Component | Points | Condition |
|-----------|--------|-----------|
| Fusion chop state | +25 | NO_TRADE or TRANSITION_CHOP |
| MVRV LS neutral | +20 | mvrv_ls_level_neutral == 1 |
| No option signals | +15 | Neither CALL nor PUT active |
| Sentiment neutral | +15 | sent_bucket_neutral == 1 |
| Sentiment flattening | +10 | sent_is_flattening == 1 |
| Low distribution pressure | +10 | dp_score ≤ expanding median |
| MVRV-60d flat + undervalued | +5 (provisional) | mvrv_60d_flat_streak_7d == 1 AND mvrv_usd_60d in stay band [0.88, 1.02] |
| Whale mixed | +5 | whale_regime_mixed == 1 |
| MVRV strong trend | -15 | Strong uptrend or downtrend |
| MDIA strong inflow | -10 | mdia_regime_strong_inflow == 1 |
| Sentiment extreme | -10 | Extreme fear or greed bucket |

The MVRV-60d flat + undervalued component was added based on analysis showing 78% safe rate (vs 58% base) when the MVRV-60d z-score is flat and the raw value is in the 0.9–1.0 range. See `features/management/commands/analyze_flat_z_range.py` for the research command.

Design choices:
- **Weight +5 (provisional):** Reduced from initial +10 pending walk-forward calibration against all other contributors on the same sample. Will promote to +10 if incremental lift is confirmed out-of-sample.
- **Streak stability:** Uses `mvrv_60d_flat_streak_7d` (7+ consecutive flat days) instead of single-day `mvrv_60d_is_flattening` to avoid flip-flop entries near the ±0.5 z-score threshold boundary.
- **Band hysteresis:** Entry requires mvrv_usd_60d in [0.90, 1.00]. Once the 7-day streak is established, the gate uses a relaxed stay band [0.88, 1.02] to reduce churn from minor oscillations at band edges.

### Precedence Rule

Veto overrides score. Penalties are diagnostic and threshold-shaping only. At execution time, hard vetoes are the binding constraint.

## Tail Risk Profile

Based on 89 backtested trades (2017–2026, with 5-day cooldown):

| Metric | Value |
|--------|-------|
| Overall hit rate | 59.6% (53/89) |
| Recent hit rate (2023–2026) | ~80% (25/31) |
| Total false positives | 36 |
| Avg FP max move | 18.4% |
| Median FP max move | 16.9% |
| P90 FP max move | 28.0% |
| P95 FP max move | 29.9% |
| P99 FP max move | 36.7% |
| Worst FP max move | 40.2% (2018-01-30, BTC crash) |
| Max consecutive losers | 8 (2017-04 to 2017-09, early volatile era) |
| Max consecutive losers (2023+) | 2 |

### Worst 5 Trades

| Date | Score | Max Move | Outcome |
|------|-------|----------|---------|
| 2018-01-30 | 85 | -40.2% | BREAKOUT_DOWN |
| 2017-09-08 | 75 | -30.3% | BREAKOUT_DOWN |
| 2017-09-14 | 75 | +29.8% | BREAKOUT_UP |
| 2018-12-16 | 75 | +29.1% | BREAKOUT_UP |
| 2018-02-04 | 75 | -26.9% | BREAKOUT_BOTH |

All worst trades are from 2017–2019 when the fusion system was immature. The ATR expansion and gap vetoes (added later) would have caught the 2018 crash.

## Execution Realism

The walk-forward validation does NOT include:
- Trading fees (exchange + network)
- Bid-ask spread / slippage on option legs
- Liquidity constraints (BTC options can be thin at far OTM strikes)
- Fill probability (limit orders may not fill in fast markets)
- IV crush / expansion effects on P&L

The hit rate measures whether the underlying stayed in range — not whether the condor was profitable. Actual P&L depends on:
- Premium collected at entry (function of IV)
- Spread width between bid/ask on 4 legs
- Whether 50% take-profit was achievable before expiry

These require real option snapshot data from the `OptionSnapshot` collection pipeline. Until that data is available, treat the hit rate as an upper bound on win rate.

## Operational Validation TODOs

The statistical edge (range-staying rate) is necessary but not sufficient for profitability. The following must be validated before treating condor signals as fully tradeable:

- [ ] **IV floor filter** — Define minimum IV rank (or IV percentile) at entry to ensure collected premium justifies tail risk. Requires `OptionSnapshot` pipeline data. Without sufficient premium, even a 78% win rate can be net negative.
- [ ] **Liquidity gate** — Validate bid-ask spread on all 4 legs is within acceptable bounds (e.g., spread < X% of credit). Thin OTM BTC options can erode the theoretical edge entirely. Requires real-time option book data.
- [ ] **Stop-loss execution realism** — Backtest actual slippage on 4-leg close during fast moves. The 6% stop assumes clean fills; the -25.9% worst-case flat-z period shows price can gap through stops. Quantify expected slippage under stress.
- [ ] **Combined filter backtest** — Re-run `condor_walkforward` with the new `mvrv_60d_flat_underval` score component active. Verify precision improvement holds out-of-sample and doesn't just overfit to the 27 qualifying periods.
- [ ] **Conditional lift validation** — Measure the incremental lift of `mvrv_60d_flat_underval` conditional on existing score components (chop state, MVRV neutral, sentiment neutral, etc.), not standalone. A strong standalone feature can add zero incremental edge once trend/vol filters are already active. If conditional lift < 2pp, remove the component.
- [ ] **Score weight calibration** — Current +5 is provisional. Run `condor_walkforward` with +5, +7, +10 variants and compare precision/pass-rate tradeoff. Promote weight only if out-of-sample precision improves without inflating pass rate into bad regimes.
- [ ] **OptionSnapshot integration** — Once the option snapshot pipeline is live, overlay IV/spread/volume data onto flat-z qualifying periods to measure actual P&L (not just range-staying).

## Monitoring & Recalibration Triggers

Re-run `chop_analysis` + `condor_walkforward` when ANY of these fire:

| Trigger | Threshold | Check Frequency |
|---------|-----------|-----------------|
| Precision drop | < 62% over last 30 condor trades | After every 10 trades |
| FP breakout rate | > 45% over last 30 trades | After every 10 trades |
| Consecutive losers | ≥ 5 in a row | After every trade |
| Walk-forward instability | Threshold std > 25 across last 8 folds | Quarterly |
| Feature pipeline change | Any new metric module added | On change |
| Fusion engine change | Any rule modification | On change |

## Decision Cascade Position

```
Priority 1: Fusion directional (CALL/PUT)
Priority 2: Tactical Put
Priority 3: OPTION_CALL fallback
Priority 4: OPTION_PUT fallback
Priority 5: MVRV_SHORT
Priority 6: IRON_CONDOR          ← this system
Priority 7: NO_TRADE
```

## Files

| File | Role |
|------|------|
| `signals/condor_gate.py` | Score, vetoes, gate evaluation |
| `features/metrics/range_label.py` | Forward range label from OHLC |
| `signals/research/chop_analysis.py` | Feature prevalence & calibration research |
| `signals/research/condor_walkforward.py` | Walk-forward validation |
| `signals/management/commands/chop_analysis.py` | Research command |
| `signals/management/commands/condor_walkforward.py` | Validation command |
| `signals/management/commands/analyze_hit_rate.py` | Hit rate (includes IRON_CONDOR) |
| `signals/migrations/0008_add_condor_gate_fields.py` | DB migration |
| `signals/options.py` | IRON_CONDOR strategy mapping |
| `signals/services.py` | Live pipeline integration |
| `signals/models.py` | DailySignal condor fields |
| `features/management/commands/analyze_flat_z_range.py` | Flat-z price range research |

## Report CSVs (`reports/chop/`, `reports/condor_wf/`)

Research artifacts for calibration. Not daily outputs — regenerate when adding features, changing fusion rules, or quarterly as a health check.

Generated by:
```bash
python manage.py chop_analysis --out reports/chop
python manage.py condor_walkforward --out reports/condor_wf
```

### `reports/chop/`

| File | Purpose |
|------|---------|
| `feature_prevalence.csv` | For each of the 155 binary features, shows how often it's active on range days vs breakout days with a lift ratio. Features with high lift became positive score components; features with low lift became penalties/vetoes. |
| `fusion_state_alignment.csv` | Each fusion state's actual 7-day range rate. Confirms NO_TRADE and TRANSITION_CHOP correlate with flat markets. |
| `chop_condition_combos.csv` | Tests combinations (chop state + MVRV neutral + sentiment neutral, etc.) and measures precision. Validates the gate's composite logic. |
| `range_score_calibration.csv` | Score bins (0–10, 10–20, ..., 90–100) with actual range rate per bin. This is how the threshold of 75 was chosen. |
| `breakout_risk_conditions.csv` | Ranks conditions by breakout prediction strength. Source of the hard vetoes. |
| `chop_analysis_full.csv` | Complete labeled dataset with all features, fusion states, range labels, and scores. For ad-hoc notebook exploration. |

### `reports/condor_wf/`

| File | Purpose |
|------|---------|
| `walkforward_folds.csv` | Per-fold results from strict walk-forward validation: threshold, precision, recall, pass rate, false positive breakdown, tail-loss metrics. |

These CSVs are gitignored. The `reports/` directories are kept via `.gitkeep`.
