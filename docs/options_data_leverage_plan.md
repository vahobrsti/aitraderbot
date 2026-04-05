# Options Data & Leverage Modeling Plan

Current priority is to build an empirical understanding of option leverage behavior from real snapshots, then feed that into execution/simulation design.

## Goal

Replace fixed leverage assumptions with observed option response profiles:

- `% option move / % underlying move`
- conditioned on DTE, moneyness, IV regime, and liquidity regime

## Workflow

1. Collect options snapshots continuously
2. Open paper/hypothetical trades tied to snapshots
3. Track path metrics during hold period (MAE/MFE)
4. Close trades and compute attribution
5. Aggregate into leverage/risk surfaces
6. Use those surfaces to update execution policy and simulator logic

## Phase 1: Data Collection

Use the collection commands to gather Bybit and Deribit option snapshots:

```bash
python manage.py collect_options --exchange bybit --dte-min 7 --dte-max 21 --moneyness 0.15
python manage.py collect_options --exchange deribit --dte-min 7 --dte-max 21 --moneyness 0.15
```

Store per snapshot:

- contract identity (symbol, strike, expiry, option type, exchange)
- underlying context (spot/index)
- prices (bid/ask/mid/mark/last)
- greeks and IV
- liquidity (size, volume, open interest)

Primary models:

- `datafeed.models.OptionSnapshot`
- `datafeed.models.OptionTrade`

## Phase 2: Paper Trade Path Tracking

Use `datafeed.services.TradeTracker` to create and close hypothetical trades against snapshot history:

- open trade at entry condition
- update excursions while open
- close on TP/SL/time/manual rule
- compute:
  - realized PnL / PnL%
  - MAE / MFE
  - IV change
  - spot change %

## Phase 3: Leverage Profiling

Build empirical profiles by bucket:

- DTE bucket (e.g., 7-10, 11-14, 15-21)
- moneyness bucket (ATM / slight ITM / deep ITM / slight OTM)
- IV percentile regime
- liquidity/spread regime

Outputs:

- expected option response per 1% underlying move
- tail-loss probabilities per bucket
- path-damage statistics (early adverse move risk)

## Phase 4: Integrate Into Execution Design

After enough sample size:

- revise leverage assumptions in execution policy docs
- rework simulator to use empirical response curves
- keep synthetic leverage mode as fallback/stress harness only

## Notes

- `TradeTracker` is currently research/paper tracking, not exchange execution.
- Live order placement remains in `execution/` services and exchange adapters.
