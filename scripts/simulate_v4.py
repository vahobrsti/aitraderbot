#!/usr/bin/env python
"""
V4 Execution Policy Simulator

Simulates the V4 options execution design across historical trades using
actual forward price paths from RawDailyData.

Assumptions:
- Dynamic leverage based on DTE bucket, moneyness, and structure
- Theta decay varies by DTE bucket
- No IV expansion/contraction modeled (conservative)
- No slippage or fees
- Spot BTC: bought at entry, never sold

Usage:
    python scripts/simulate_v4.py
"""
import os
import sys
import django

# Setup Django
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "aitrader.settings")
django.setup()

import numpy as np
import pandas as pd
from pathlib import Path

from datafeed.models import RawDailyData

# ── Dynamic Leverage Configuration ───────────────────────────────────────────

# DTE buckets
DTE_BUCKETS = {
    "7_10": (7, 10),
    "11_14": (11, 14),
    "15_21": (15, 21),
}

# Moneyness buckets
MONEYNESS_BUCKETS = ["atm", "slightly_itm", "deep_itm", "slightly_otm"]

# Naked option leverage grid: (dte_bucket, moneyness) -> (min_leverage, max_leverage)
# We use midpoint for simulation
NAKED_LEVERAGE_GRID = {
    # 7-10 DTE
    ("7_10", "atm"): (15, 18),
    ("7_10", "slightly_itm"): (15, 18),
    ("7_10", "slightly_otm"): (18, 22),
    ("7_10", "deep_itm"): (10, 14),
    # 11-14 DTE
    ("11_14", "atm"): (12, 15),
    ("11_14", "slightly_itm"): (12, 15),
    ("11_14", "slightly_otm"): (15, 18),
    ("11_14", "deep_itm"): (8, 12),
    # 15-21 DTE
    ("15_21", "atm"): (9, 12),
    ("15_21", "slightly_itm"): (9, 12),
    ("15_21", "slightly_otm"): (12, 15),
    ("15_21", "deep_itm"): (6, 9),
}

# Spread leverage grid: dte_bucket -> (min_leverage, max_leverage)
SPREAD_LEVERAGE_GRID = {
    "7_10": (7, 10),
    "11_14": (6, 8),
    "15_21": (4, 7),
}

# Theta decay by DTE bucket (% per day)
# Shorter DTE = faster decay
THETA_DECAY_GRID = {
    "7_10": 0.025,   # 2.5% per day
    "11_14": 0.020,  # 2.0% per day
    "15_21": 0.015,  # 1.5% per day
}

# Default trade parameters (can be overridden per signal type)
DEFAULT_DTE_BUCKET = "11_14"
DEFAULT_MONEYNESS = "slightly_itm"

# Signal type to option parameters mapping
# Format: signal_type -> (dte_bucket, moneyness)
SIGNAL_OPTION_PARAMS = {
    # Tier 1 signals - standard parameters
    "LONG": ("11_14", "slightly_itm"),
    "PRIMARY_SHORT": ("11_14", "slightly_itm"),
    "BULL_PROBE": ("7_10", "slightly_itm"),  # faster signals, shorter DTE
    # Tier 2 signals
    "OPTION_CALL": ("7_10", "atm"),  # high conviction, ATM for max leverage
    "OPTION_PUT": ("7_10", "atm"),
    "MVRV_SHORT": ("15_21", "slightly_itm"),  # slower signal, longer DTE
    # Tier 3 signals
    "BEAR_PROBE": ("11_14", "slightly_itm"),
    "TACTICAL_PUT": ("7_10", "slightly_otm"),  # cheap hedges
}


def get_leverage_params(signal_type):
    """
    Get dynamic leverage and theta for a signal type.
    Returns: (naked_leverage, spread_leverage, theta_decay)
    """
    dte_bucket, moneyness = SIGNAL_OPTION_PARAMS.get(
        signal_type, (DEFAULT_DTE_BUCKET, DEFAULT_MONEYNESS)
    )

    # Naked leverage: use midpoint of range
    naked_range = NAKED_LEVERAGE_GRID.get(
        (dte_bucket, moneyness),
        NAKED_LEVERAGE_GRID.get((DEFAULT_DTE_BUCKET, DEFAULT_MONEYNESS), (12, 15))
    )
    naked_leverage = (naked_range[0] + naked_range[1]) / 2

    # Spread leverage: use midpoint of range
    spread_range = SPREAD_LEVERAGE_GRID.get(dte_bucket, (6, 8))
    spread_leverage = (spread_range[0] + spread_range[1]) / 2

    # Theta decay
    theta_decay = THETA_DECAY_GRID.get(dte_bucket, 0.02)

    return naked_leverage, spread_leverage, theta_decay, dte_bucket, moneyness

# Tier definitions: 100% options, no spot
TIERS = {
    1: {"risk": 4000, "option_pct": 1.0, "spot_pct": 0.0, "naked_pct": 0.367, "spread_pct": 0.633},
    2: {"risk": 2400, "option_pct": 1.0, "spot_pct": 0.0, "naked_pct": 0.333, "spread_pct": 0.667},
    3: {"risk": 1200, "option_pct": 1.0, "spot_pct": 0.0, "naked_pct": 0.0,   "spread_pct": 1.0},
}

# Signal type to tier mapping - BALANCED PROFITABLE
# Based on 2023-2025 backtest performance
TIER_MAP = {
    "LONG": None,           # EXCLUDE: 47.4% win rate, -$7,280 P&L
    "PRIMARY_SHORT": 1,     # T1: 75% win rate, +$6,399 P&L, best performer
    "BULL_PROBE": 2,        # T2: 62.5% win rate, +$500 P&L, decent volume
    "OPTION_CALL": 1,       # T1: 80% win rate, +$6,221 P&L, excellent
    "OPTION_PUT": None,     # EXCLUDE: insufficient sample, 0% win rate
    "MVRV_SHORT": 2,        # T2: keep for diversification
    "BEAR_PROBE": 2,        # T2: 66.7% win rate, +$499 P&L
    "TACTICAL_PUT": None,   # EXCLUDE: 25% win rate, -$2,759 P&L
}

# Exit rules — calibrated to 5% underlying target (5% × 15x = 75% option gain)
UNDERLYING_TARGET = 0.05    # 5% underlying move target

# Execution haircut — real-world adjustment for slippage, IV, fill imperfection
WINNER_HAIRCUT = 0.88   # reduce winners by 12%
LOSER_HAIRCUT = 1.08    # increase losers by 8%

# Portfolio constraint
MAX_CONCURRENT_TRADES = 2
MAX_CAPITAL_AT_RISK_PCT = 0.06  # 6% of account

# Conditional spread TP
SPREAD_TP_FAST = 0.60   # fast move (naked hit TP by day 3) → take spread at 60%
SPREAD_TP_SLOW = 0.80   # slow/grinding move → allow spread to run to 80%
NAKED_STOP_PCT = 0.05       # 5% underlying adverse
NAKED_TP_PARTIAL = 0.50     # 50% option gain -> close 60%
NAKED_TP_FULL = 0.70        # 70% option gain -> close rest
NAKED_DAY34_THRESHOLD = -0.30  # if option down >30% by day 3-4, close
NAKED_TIME_STOP = 7         # day 7 hard exit

SPREAD_STOP_PCT = 0.07      # 7% underlying adverse
SPREAD_TP_PCT = 0.60        # 60% of max spread value
SPREAD_TIME_STOP = 10       # day 10 hard exit


def build_trades(csv_path, years):
    """
    Enumerate trades by delegating to analyze_path_stats._build_trades_df().

    This ensures the simulator always uses the exact same trade enumeration
    logic as the analysis command — no duplication, no drift.
    """
    from signals.management.commands.analyze_path_stats import Command as PathStatsCommand

    cmd = PathStatsCommand()
    options = {
        "csv": str(csv_path),
        "long_model": "models/long_model.joblib",
        "short_model": "models/short_model.joblib",
    }

    trades_df = cmd._build_trades_df(
        csv_path=Path(csv_path),
        options=options,
        year_filter=None,
        no_overlay=False,
        no_cooldown=False,
    )

    if trades_df.empty:
        return trades_df

    # Filter to requested years
    trades_df = trades_df[trades_df["year"].isin(years)].copy()
    return trades_df


def load_prices():
    """Load BTC daily prices from DB."""
    qs = RawDailyData.objects.order_by("date").values("date", "btc_close", "btc_high", "btc_low")
    px = pd.DataFrame.from_records(qs)
    if px.empty:
        return px
    px["date"] = pd.to_datetime(px["date"])
    px = px.set_index("date").sort_index()
    px.index = px.index.normalize()
    return px


def simulate_trade(trade, price_df, horizon=14):
    """
    Simulate V4 execution policy on a single trade.
    Returns dict with P&L breakdown.
    """
    dt = pd.Timestamp(trade["date"])
    direction = trade["direction"]
    trade_type = trade["type"]
    tier_num = TIER_MAP.get(trade_type)
    if tier_num is None:
        return None  # Skip excluded signal types
    tier = TIERS[tier_num]

    if dt not in price_df.index:
        return None

    entry_price = float(price_df.loc[dt, "btc_close"])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None

    path = price_df.loc[dt:].iloc[1:horizon + 1]
    if len(path) < horizon:
        return None

    # ── Get dynamic leverage parameters for this signal type ──
    naked_leverage, spread_leverage, theta_decay, dte_bucket, moneyness = get_leverage_params(trade_type)

    # ── Allocations ──
    total_risk = tier["risk"]
    spot_amount = total_risk * tier["spot_pct"]
    option_budget = total_risk * tier["option_pct"]
    naked_amount = option_budget * tier["naked_pct"]
    spread_amount = option_budget * tier["spread_pct"]

    # ── Spot: buy BTC at entry, track value ──
    btc_bought = spot_amount / entry_price
    # We'll track spot value at trade end for reporting, but spot is never sold

    # ── Simulate day by day ──
    naked_alive = naked_amount > 0
    naked_initial = naked_amount  # preserve original for reporting
    naked_partial_taken = False
    naked_closed_day = None
    naked_pnl = 0.0
    naked_remaining = naked_amount  # tracks current position size after partial TP

    spread_alive = spread_amount > 0
    spread_value = spread_amount
    spread_closed_day = None
    spread_pnl = 0.0

    is_long = direction == "LONG"

    # Track best favorable close seen so far (for "no progress" check)
    best_favorable_close = 0.0

    for day_idx in range(len(path)):
        day_num = day_idx + 1  # day 1, 2, ...
        high = float(path.iloc[day_idx]["btc_high"])
        low = float(path.iloc[day_idx]["btc_low"])
        close = float(path.iloc[day_idx]["btc_close"])

        if is_long:
            # Favorable = price going up
            best_move = (high / entry_price) - 1.0
            worst_move = 1.0 - (low / entry_price)  # positive = adverse
            close_move = (close / entry_price) - 1.0
        else:
            # Favorable = price going down
            best_move = 1.0 - (low / entry_price)
            worst_move = (high / entry_price) - 1.0  # positive = adverse
            close_move = 1.0 - (close / entry_price)

        best_favorable_close = max(best_favorable_close, close_move)

        # ── NAKED LEG ──
        if naked_alive:
            # Check hard stop first using intraday adverse (5%)
            if worst_move >= NAKED_STOP_PCT:
                loss_pct = min(NAKED_STOP_PCT * naked_leverage, 1.0)
                naked_pnl += -naked_remaining * loss_pct
                naked_alive = False
                naked_closed_day = day_num
                # Continue to spread leg below
            else:
                # Check TP using intraday favorable move (not just close)
                best_option_gain_pct = best_move * naked_leverage - (theta_decay * day_num)
                # Also compute close-based value for time/review exits
                close_option_pnl_pct = close_move * naked_leverage - (theta_decay * day_num)
                current_naked_value = naked_remaining * max(1.0 + close_option_pnl_pct, 0)

                # Partial TP: intraday high reached 50% option gain
                if not naked_partial_taken and best_option_gain_pct >= NAKED_TP_PARTIAL:
                    partial_size = naked_remaining * 0.60
                    naked_pnl += partial_size * NAKED_TP_PARTIAL
                    naked_remaining *= 0.40
                    naked_partial_taken = True
                    # Recompute current value for remaining position
                    current_naked_value = naked_remaining * max(1.0 + close_option_pnl_pct, 0)

                # Full TP: intraday high reached 70% option gain
                if naked_alive and best_option_gain_pct >= NAKED_TP_FULL:
                    naked_pnl += naked_remaining * NAKED_TP_FULL
                    naked_remaining = 0
                    naked_alive = False
                    naked_closed_day = day_num

                # Day 3-4 review: down >30% AND no meaningful progress
                # "No progress" = best favorable close so far < 1% underlying move
                if naked_alive and day_num in (3, 4):
                    if close_option_pnl_pct < NAKED_DAY34_THRESHOLD and best_favorable_close < 0.01:
                        naked_pnl += current_naked_value - naked_remaining
                        naked_remaining = 0
                        naked_alive = False
                        naked_closed_day = day_num

                # Day 7 time stop
                if naked_alive and day_num >= NAKED_TIME_STOP:
                    naked_pnl += current_naked_value - naked_remaining
                    naked_remaining = 0
                    naked_alive = False
                    naked_closed_day = day_num

        # ── SPREAD LEG ──
        if spread_alive:
            # Check hard stop using intraday adverse (7%)
            if worst_move >= SPREAD_STOP_PCT:
                loss_pct = min(SPREAD_STOP_PCT * spread_leverage, 1.0)
                spread_pnl = -spread_amount * loss_pct
                spread_alive = False
                spread_closed_day = day_num
            else:
                # Spread P&L: lower leverage, less theta impact (30% of naked theta)
                best_spread_pnl_pct = best_move * spread_leverage - (theta_decay * 0.3 * day_num)
                # Close-based value for time exits
                close_spread_pnl_pct = close_move * spread_leverage - (theta_decay * 0.3 * day_num)
                # Spread has capped upside: max gain = spread_amount (100% of debit)
                current_spread_value = spread_amount * (1.0 + close_spread_pnl_pct)
                current_spread_value = max(current_spread_value, 0)
                current_spread_value = min(current_spread_value, spread_amount * 2.0)  # max ~2x

                spread_gain_pct = (current_spread_value / spread_amount) - 1.0 if spread_amount > 0 else 0

                # Conditional spread TP:
                # fast move (naked closed by day 3) → 60% TP
                # slow/grinding move → allow 80%
                spread_tp = SPREAD_TP_FAST if (naked_closed_day is not None and naked_closed_day <= 3) else SPREAD_TP_SLOW
                if best_spread_pnl_pct >= spread_tp:
                    spread_pnl = spread_amount * spread_tp
                    spread_alive = False
                    spread_closed_day = day_num

                # Day 10 time stop
                if spread_alive and day_num >= SPREAD_TIME_STOP:
                    spread_pnl = current_spread_value - spread_amount
                    spread_alive = False
                    spread_closed_day = day_num

    # If still alive at end of horizon (shouldn't happen with time stops, but safety)
    if naked_alive:
        final_close = float(path.iloc[-1]["btc_close"])
        if is_long:
            final_move = (final_close / entry_price) - 1.0
        else:
            final_move = 1.0 - (final_close / entry_price)
        final_pnl_pct = final_move * naked_leverage - (theta_decay * horizon)
        naked_pnl += naked_remaining * max(final_pnl_pct, -1.0)
        naked_remaining = 0

    if spread_alive:
        final_close = float(path.iloc[-1]["btc_close"])
        if is_long:
            final_move = (final_close / entry_price) - 1.0
        else:
            final_move = 1.0 - (final_close / entry_price)
        final_pnl_pct = final_move * spread_leverage - (theta_decay * 0.3 * horizon)
        spread_pnl = spread_amount * max(min(final_pnl_pct, 1.0), -1.0)

    # Spot value at end of trade window
    final_close = float(path.iloc[-1]["btc_close"])
    spot_end_value = btc_bought * final_close
    spot_unrealized_pnl = spot_end_value - spot_amount

    total_option_pnl = naked_pnl + spread_pnl

    # Apply real-world execution haircut
    if total_option_pnl > 0:
        total_option_pnl *= WINNER_HAIRCUT
    else:
        total_option_pnl *= LOSER_HAIRCUT

    return {
        "date": dt,
        "type": trade_type,
        "direction": direction,
        "tier": tier_num,
        "dte_bucket": dte_bucket,
        "moneyness": moneyness,
        "naked_leverage": naked_leverage,
        "spread_leverage": spread_leverage,
        "theta_decay": theta_decay,
        "entry_price": entry_price,
        "total_risk": total_risk,
        "naked_amount": naked_initial,
        "spread_amount": spread_amount,
        "spot_amount": spot_amount,
        "naked_pnl": round(naked_pnl, 2),
        "spread_pnl": round(spread_pnl, 2),
        "option_pnl": round(total_option_pnl, 2),
        "spot_btc": btc_bought,
        "spot_unrealized": round(spot_unrealized_pnl, 2),
        "naked_closed_day": naked_closed_day,
        "spread_closed_day": spread_closed_day,
        "win": 1 if total_option_pnl > 0 else 0,
    }


def main():
    INITIAL_ACCOUNT = 100_000
    YEARS = [2021, 2022, 2023, 2024, 2025]

    print(f"\n{'=' * 90}")
    print(f"V5 EXECUTION POLICY SIMULATION (Dynamic Leverage)")
    print(f"{'=' * 90}")
    print(f"Account: ${INITIAL_ACCOUNT:,} | Years: {YEARS} | Target: {UNDERLYING_TARGET*100:.0f}%")
    print(f"\nDynamic Leverage Grid:")
    print(f"  Naked: 7-10 DTE ATM/ITM: 16.5x | 11-14 DTE: 13.5x | 15-21 DTE: 10.5x")
    print(f"  Spread: 7-10 DTE: 8.5x | 11-14 DTE: 7x | 15-21 DTE: 5.5x")
    print(f"  Theta: 7-10 DTE: 2.5%/day | 11-14 DTE: 2%/day | 15-21 DTE: 1.5%/day")
    print(f"\nNaked stop: {NAKED_STOP_PCT*100:.0f}% | Spread stop: {SPREAD_STOP_PCT*100:.0f}%")
    print(f"Naked TP: partial@{NAKED_TP_PARTIAL*100:.0f}% / full@{NAKED_TP_FULL*100:.0f}% | Spread TP: fast@{SPREAD_TP_FAST*100:.0f}% / slow@{SPREAD_TP_SLOW*100:.0f}%")
    print(f"Haircut: winners×{WINNER_HAIRCUT:.0%} / losers×{LOSER_HAIRCUT:.0%} | Max concurrent: {MAX_CONCURRENT_TRADES} | Max capital at risk: {MAX_CAPITAL_AT_RISK_PCT:.0%}")
    print()

    # Build trades
    trades_df = build_trades("features_14d_5pct.csv", YEARS)
    if trades_df.empty:
        print("No trades found.")
        return

    print(f"Trades enumerated: {len(trades_df)}")

    # Load prices
    price_df = load_prices()
    if price_df.empty:
        print("No price data.")
        return

    # Simulate each trade with portfolio constraint
    results = []
    open_trades = []  # list of (entry_date, expected_close_day)

    for _, trade in trades_df.iterrows():
        trade_date = pd.Timestamp(trade["date"])
        tier_num = TIER_MAP.get(trade["type"])
        if tier_num is None:
            continue  # Skip excluded signal types
        trade_risk = TIERS[tier_num]["risk"]

        # Remove trades that have closed (assume max hold = 10 days)
        open_trades = [(d, c) for d, c in open_trades if (trade_date - d).days < c]

        # Portfolio constraint checks
        capital_at_risk = sum(TIERS[TIER_MAP.get(t["type"], 3)]["risk"]
                              for t in [trade] if True) * len(open_trades)
        concurrent_count = len(open_trades)

        if concurrent_count >= MAX_CONCURRENT_TRADES:
            continue  # skip — too many open positions

        r = simulate_trade(trade, price_df)
        if r:
            # Track this trade as open for its expected hold duration
            hold_days = r.get("spread_closed_day") or SPREAD_TIME_STOP
            open_trades.append((trade_date, hold_days))
            results.append(r)

    if not results:
        print("No trades with sufficient forward data.")
        return

    res_df = pd.DataFrame(results)

    # ── TRADE-BY-TRADE LOG ──
    print(f"\n{'=' * 90}")
    print("TRADE-BY-TRADE RESULTS")
    print(f"{'=' * 90}")
    print(f"{'Date':12s} | {'Type':16s} | {'Dir':5s} | {'DTE':5s} | {'NkLev':5s} | {'Naked P&L':>10s} | {'Spread P&L':>10s} | {'Option P&L':>10s} | {'Win':3s}")
    print("-" * 105)

    cumulative_option_pnl = 0
    cumulative_btc = 0
    equity = INITIAL_ACCOUNT
    peak_equity = equity
    max_drawdown = 0
    equity_curve = []

    for _, r in res_df.iterrows():
        cumulative_option_pnl += r["option_pnl"]
        cumulative_btc += r["spot_btc"]
        equity += r["option_pnl"]
        peak_equity = max(peak_equity, equity)
        drawdown = peak_equity - equity
        max_drawdown = max(max_drawdown, drawdown)

        equity_curve.append({
            "date": r["date"],
            "equity": round(equity, 2),
            "drawdown": round(drawdown, 2),
            "cumulative_btc": cumulative_btc,
        })

        win_str = "✅" if r["win"] else "❌"
        print(
            f"{r['date'].strftime('%Y-%m-%d'):12s} | {r['type']:16s} | {r['direction']:5s} | {r['dte_bucket']:5s} | "
            f"{r['naked_leverage']:5.1f} | ${r['naked_pnl']:>+9.2f} | ${r['spread_pnl']:>+9.2f} | ${r['option_pnl']:>+9.2f} | {win_str}"
        )

    # ── EQUITY CURVE ──
    print(f"\n{'=' * 90}")
    print("EQUITY CURVE")
    print(f"{'=' * 90}")
    print(f"{'Date':12s} | {'Equity':>12s} | {'Drawdown':>10s} | {'BTC Held':>10s}")
    print("-" * 55)
    for e in equity_curve:
        print(
            f"{e['date'].strftime('%Y-%m-%d'):12s} | ${e['equity']:>11,.2f} | ${e['drawdown']:>9,.2f} | "
            f"{e['cumulative_btc']:.5f}"
        )

    # ── SUMMARY STATS ──
    print(f"\n{'=' * 90}")
    print("SUMMARY")
    print(f"{'=' * 90}")

    total_trades = len(res_df)
    winners = res_df[res_df["win"] == 1]
    losers = res_df[res_df["win"] == 0]

    print(f"Total trades: {total_trades}")
    print(f"Winners: {len(winners)} ({len(winners)/total_trades*100:.1f}%)")
    print(f"Losers: {len(losers)} ({len(losers)/total_trades*100:.1f}%)")
    print()

    total_option_pnl = res_df["option_pnl"].sum()
    total_btc_accumulated = res_df["spot_btc"].sum()
    total_spot_deployed = res_df["spot_amount"].sum()

    print(f"Total option P&L: ${total_option_pnl:>+,.2f}")
    print(f"Avg winner: ${winners['option_pnl'].mean():>+,.2f}" if len(winners) else "Avg winner: N/A")
    print(f"Avg loser: ${losers['option_pnl'].mean():>+,.2f}" if len(losers) else "Avg loser: N/A")
    print(f"Largest winner: ${res_df['option_pnl'].max():>+,.2f}")
    print(f"Largest loser: ${res_df['option_pnl'].min():>+,.2f}")
    print()

    print(f"Final equity (options only): ${equity:>,.2f}")
    print(f"Option return: {(equity - INITIAL_ACCOUNT) / INITIAL_ACCOUNT * 100:>+.2f}%")
    print(f"Max drawdown: ${max_drawdown:>,.2f} ({max_drawdown/INITIAL_ACCOUNT*100:.2f}%)")
    print()

    # Get latest BTC price for spot valuation
    latest_btc = float(price_df.iloc[-1]["btc_close"])
    btc_value = total_btc_accumulated * latest_btc

    print(f"BTC accumulated: {total_btc_accumulated:.6f} BTC")
    print(f"Total spot deployed: ${total_spot_deployed:>,.2f}")
    print(f"BTC value (at ${latest_btc:,.0f}): ${btc_value:>,.2f}")
    print(f"Spot unrealized P&L: ${btc_value - total_spot_deployed:>+,.2f}")
    print()

    combined_return = (equity - INITIAL_ACCOUNT) + (btc_value - total_spot_deployed)
    print(f"Combined return (options realized + spot unrealized): ${combined_return:>+,.2f}")
    print(f"Combined return %: {combined_return / INITIAL_ACCOUNT * 100:>+.2f}%")

    # ── BY TIER ──
    print(f"\n{'=' * 90}")
    print("BY TIER")
    print(f"{'=' * 90}")
    for tier_num in [1, 2, 3]:
        tier_df = res_df[res_df["tier"] == tier_num]
        if tier_df.empty:
            continue
        t_wins = tier_df[tier_df["win"] == 1]
        print(
            f"Tier {tier_num}: {len(tier_df)} trades | "
            f"Win rate: {len(t_wins)/len(tier_df)*100:.1f}% | "
            f"Option P&L: ${tier_df['option_pnl'].sum():>+,.2f} | "
            f"Avg: ${tier_df['option_pnl'].mean():>+,.2f}"
        )

    # ── BY YEAR ──
    print(f"\n{'=' * 90}")
    print("BY YEAR")
    print(f"{'=' * 90}")
    for year in YEARS:
        year_df = res_df[res_df["date"].dt.year == year]
        if year_df.empty:
            continue
        y_wins = year_df[year_df["win"] == 1]
        print(
            f"{year}: {len(year_df)} trades | "
            f"Win rate: {len(y_wins)/len(year_df)*100:.1f}% | "
            f"Option P&L: ${year_df['option_pnl'].sum():>+,.2f} | "
            f"BTC accumulated: {year_df['spot_btc'].sum():.6f}"
        )

    # ── BY TYPE ──
    print(f"\n{'=' * 90}")
    print("BY TYPE")
    print(f"{'=' * 90}")
    for ttype in res_df["type"].unique():
        type_df = res_df[res_df["type"] == ttype]
        tw = type_df[type_df["win"] == 1]
        print(
            f"{ttype:16s}: {len(type_df):2d} trades | "
            f"Win: {len(tw)/len(type_df)*100:5.1f}% | "
            f"P&L: ${type_df['option_pnl'].sum():>+8,.2f} | "
            f"Avg: ${type_df['option_pnl'].mean():>+7,.2f}"
        )

    # ── BY LEVERAGE PROFILE ──
    print(f"\n{'=' * 90}")
    print("BY LEVERAGE PROFILE")
    print(f"{'=' * 90}")
    for dte in ["7_10", "11_14", "15_21"]:
        dte_df = res_df[res_df["dte_bucket"] == dte]
        if dte_df.empty:
            continue
        dw = dte_df[dte_df["win"] == 1]
        avg_naked_lev = dte_df["naked_leverage"].mean()
        avg_spread_lev = dte_df["spread_leverage"].mean()
        print(
            f"DTE {dte:5s}: {len(dte_df):2d} trades | "
            f"Win: {len(dw)/len(dte_df)*100:5.1f}% | "
            f"Naked Lev: {avg_naked_lev:5.1f}x | "
            f"Spread Lev: {avg_spread_lev:4.1f}x | "
            f"P&L: ${dte_df['option_pnl'].sum():>+8,.2f}"
        )

    # ── SIGNAL TYPE LEVERAGE MAPPING ──
    print(f"\n{'=' * 90}")
    print("SIGNAL TYPE → LEVERAGE MAPPING")
    print(f"{'=' * 90}")
    print(f"{'Type':16s} | {'DTE':5s} | {'Moneyness':12s} | {'Naked Lev':>9s} | {'Spread Lev':>10s} | {'Theta':>6s}")
    print("-" * 75)
    for sig_type in sorted(SIGNAL_OPTION_PARAMS.keys()):
        dte, money = SIGNAL_OPTION_PARAMS[sig_type]
        nk_lev, sp_lev, theta, _, _ = get_leverage_params(sig_type)
        print(
            f"{sig_type:16s} | {dte:5s} | {money:12s} | {nk_lev:8.1f}x | {sp_lev:9.1f}x | {theta*100:5.1f}%"
        )


if __name__ == "__main__":
    main()
