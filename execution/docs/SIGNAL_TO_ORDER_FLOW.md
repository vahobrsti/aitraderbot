# Signal to Order Flow

Detailed internal flow from signal to exchange order placement.

## Overview

```
DailySignal → ExecutionIntent → Risk Check → Instrument Selection → Order → Exchange API
```

## Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. ENTRY POINT: execute_signal command                                     │
│     python manage.py execute_signal --latest --account bybit-main           │
│     File: execution/management/commands/execute_signal.py                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2. LOAD SIGNAL: DailySignal from signals app                               │
│     - trade_decision: OPTION_CALL, OPTION_PUT, TACTICAL_PUT, CALL, PUT      │
│     - size_multiplier, stop_loss_pct, take_profit_pct, max_hold_days        │
│     File: signals/models.py                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  3. CREATE INTENT: orchestrator.create_intent_from_signal()                 │
│     File: execution/services/orchestrator.py                                │
│                                                                             │
│     Maps decision → (direction, option_type):                               │
│       CALL / OPTION_CALL  → (long, call)                                    │
│       PUT / OPTION_PUT    → (short, put)                                    │
│       TACTICAL_PUT        → (long, put)                                     │
│                                                                             │
│     Calculates: target_notional = account.max_position_usd * 0.5 * size_mult│
│     Creates: ExecutionIntent with status='pending'                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  4. PROCESS INTENT: orchestrator.process_intent()                           │
│     File: execution/services/orchestrator.py                                │
│                                                                             │
│     Step 1: Risk Check → risk_manager.check_intent()                        │
│             File: execution/services/risk.py                                │
│             - May reject or adjust notional                                 │
│                                                                             │
│     Step 2: Select Instrument → _select_instrument()                        │
│             File: execution/services/orchestrator.py                        │
│             - Finds suitable option contract (symbol) by option type + DTE  │
│                                                                             │
│     Step 3: Calculate Qty → _calculate_qty()                                │
│             - Converts notional USD → contract qty                          │
│                                                                             │
│     Step 4: Place Entry Order → _place_entry_order()                        │
│             - Creates Order model record                                    │
│             - Builds OrderRequest(symbol, side, 'market', qty)              │
│             - Calls adapter.place_order(request)                            │
│                                                                             │
│     Step 5: Sync Order Status → _sync_order_status()                        │
│             - Polls exchange for fill status                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  5. EXCHANGE ADAPTER: adapter.place_order(request)                          │
│     File: execution/exchanges/bybit.py (or deribit.py)                      │
│                                                                             │
│     Bybit place_order():                                                    │
│     - Maps to Bybit V5 params: category, symbol, side, orderType, qty, etc. │
│     - Calls: session.place_order(**params)  ← HTTP POST to Bybit API        │
│     - Returns: OrderResponse with exchange_order_id or error                │
│                                                                             │
│     Bybit API endpoint: POST /v5/order/create                               │
│     Required params: category, symbol, side, orderType, qty                 │
│     Optional: price, triggerPrice, reduceOnly, orderLinkId, timeInForce     │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Files

| File | Purpose |
|------|---------|
| `execution/management/commands/execute_signal.py` | CLI entry point |
| `execution/services/orchestrator.py` | Main execution flow (including current instrument selection) |
| `execution/services/risk.py` | Risk checks (limits, duplicates) |
| `execution/services/instrument_selector.py` | Reusable selector utilities (currently not wired into orchestrator path) |
| `execution/exchanges/bybit.py` | Bybit V5 API adapter |
| `execution/exchanges/deribit.py` | Deribit API adapter |
| `execution/models.py` | ExecutionIntent, Order, Position |

## Decision Mapping

```python
decision_map = {
    'CALL': ('long', 'call'),
    'OPTION_CALL': ('long', 'call'),
    'PUT': ('short', 'put'),
    'OPTION_PUT': ('short', 'put'),
    'TACTICAL_PUT': ('long', 'put'),
}
```

## Bybit Order Parameters

The `place_order` method in `bybit.py` maps to Bybit V5 API:

| Our Param | Bybit Param | Notes |
|-----------|-------------|-------|
| `symbol` | `symbol` | e.g., BTC-26DEC25-100000-C |
| `side` | `side` | Buy / Sell |
| `order_type` | `orderType` | Market / Limit |
| `qty` | `qty` | Contract quantity |
| `price` | `price` | For limit orders |
| `trigger_price` | `triggerPrice` | For conditional orders |
| `time_in_force` | `timeInForce` | GTC / IOC / FOK / PostOnly |
| `reduce_only` | `reduceOnly` | Close-only flag |
| `client_order_id` | `orderLinkId` | Our internal reference |

### Missing Parameters (Future)

These Bybit params are not yet implemented:

| Parameter | Purpose |
|-----------|---------|
| `positionIdx` | Required for hedge-mode |
| `takeProfit` / `stopLoss` | Native TP/SL (not used for options) |
| `triggerBy` | Price type for triggers (LastPrice/MarkPrice/IndexPrice) |
| `closeOnTrigger` | Ensures stop closes position |
| `isLeverage` | For spot margin |

## V1 Limitations

- **Options only**: `instrument_type` hardcoded to `'option'`
- **Single-leg only**: No spreads
- **Market orders only**: No limit entry
- **Polling exits**: Options don't support native SL/TP

## Example Execution

```bash
# Dry run
python manage.py execute_signal --latest --account bybit-prod --dry-run

# Execute
python manage.py execute_signal --latest --account bybit-prod

# Force re-execute (if intent exists)
python manage.py execute_signal --date 2024-01-15 --account bybit-prod --force
```
