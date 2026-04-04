"""
V7 Execution Policy Simulator

Simulates the V7 options execution design across historical trades using
actual forward price paths from RawDailyData.

Assumptions:
- Dynamic leverage based on DTE bucket, moneyness, and structure
- Theta decay varies by DTE bucket
- No IV expansion/contraction modeled (conservative)
- No slippage or fees
- Spot BTC: bought at entry, never sold

Usage:
    python manage.py simulate
    python manage.py simulate --years 2023 2024
    python manage.py simulate --csv features_14d_5pct.csv --account 50000
"""
import numpy as np
import pandas as pd
from pathlib import Path

from django.core.management.base import BaseCommand

from datafeed.models import RawDailyData


# ── Dynamic Leverage Configuration ───────────────────────────────────────────

DTE_BUCKETS = {
    "7_10": (7, 10),
    "11_14": (11, 14),
    "15_21": (15, 21),
}

MONEYNESS_BUCKETS = ["atm", "slightly_itm", "deep_itm", "slightly_otm"]

NAKED_LEVERAGE_GRID = {
    # V7: More realistic leverage ranges (base 6-10x, stress 12-18x)
    # Reduced from optimistic V6 values to match real-world observations
    ("7_10", "atm"): (12, 18),        # stress zone - avoid naked here
    ("7_10", "slightly_itm"): (10, 15),
    ("7_10", "slightly_otm"): (14, 20),  # stress zone
    ("7_10", "deep_itm"): (6, 10),
    ("11_14", "atm"): (10, 15),       # stress zone
    ("11_14", "slightly_itm"): (8, 12),  # was 12-15, now realistic
    ("11_14", "slightly_otm"): (12, 16),
    ("11_14", "deep_itm"): (6, 9),
    ("15_21", "atm"): (7, 11),
    ("15_21", "slightly_itm"): (6, 10),  # was 9-12, now base range
    ("15_21", "slightly_otm"): (9, 13),
    ("15_21", "deep_itm"): (4, 7),
}

SPREAD_LEVERAGE_GRID = {
    # V7: Wider spreads = lower effective leverage
    # Models 8-12% width instead of ~6% width
    "7_10": (5, 8),    # was 7-10
    "11_14": (4, 6),   # was 6-8
    "15_21": (3, 5),   # was 4-7
}

THETA_DECAY_GRID = {
    "7_10": 0.025,
    "11_14": 0.020,
    "15_21": 0.015,
}

DEFAULT_DTE_BUCKET = "11_14"
DEFAULT_MONEYNESS = "slightly_itm"

SIGNAL_OPTION_PARAMS = {
    # V7: Shifted DTE higher (14-21 primary), deeper ITM to reduce gamma
    "LONG": ("15_21", "deep_itm"),
    "PRIMARY_SHORT": ("15_21", "slightly_itm"),  # was 11_14
    "BULL_PROBE": ("11_14", "slightly_itm"),     # was 7_10
    "OPTION_CALL": ("11_14", "slightly_itm"),    # was 7_10 ATM (high gamma)
    "OPTION_PUT": ("11_14", "slightly_itm"),     # was 7_10 ATM
    "MVRV_SHORT": ("15_21", "deep_itm"),         # deeper ITM
    "BEAR_PROBE": ("15_21", "slightly_itm"),     # was 11_14
    "TACTICAL_PUT": ("11_14", "slightly_otm"),   # was 7_10
}

TIERS = {
    # V7: Reduced naked exposure (20-25% max) to minimize gamma/path damage
    1: {"risk": 4000, "option_pct": 1.0, "spot_pct": 0.0, "naked_pct": 0.20, "spread_pct": 0.80},
    2: {"risk": 2400, "option_pct": 1.0, "spot_pct": 0.0, "naked_pct": 0.25, "spread_pct": 0.75},
    3: {"risk": 1200, "option_pct": 1.0, "spot_pct": 0.0, "naked_pct": 0.0, "spread_pct": 1.0},
}

TIER_MAP = {
    "LONG": None,
    "PRIMARY_SHORT": 1,
    "BULL_PROBE": 2,
    "OPTION_CALL": 1,
    "OPTION_PUT": None,
    "MVRV_SHORT": None,
    "BEAR_PROBE": 2,
    "TACTICAL_PUT": None,
}

UNDERLYING_TARGET = 0.05
WINNER_HAIRCUT = 0.88
LOSER_HAIRCUT = 1.08
MAX_CONCURRENT_TRADES = 2
MAX_CAPITAL_AT_RISK_PCT = 0.06

SPREAD_TP_FAST = 0.60
SPREAD_TP_SLOW = 0.80
NAKED_STOP_PCT = 0.05
NAKED_TP_PARTIAL = 0.50
NAKED_TP_FULL = 0.70
NAKED_DAY34_THRESHOLD = -0.30
NAKED_TIME_STOP = 7
SPREAD_STOP_PCT = 0.07
SPREAD_TP_PCT = 0.60
SPREAD_TIME_STOP = 10


def get_leverage_params(signal_type, stress_mode=False):
    """Get dynamic leverage and theta for a signal type.
    
    V7: Added stress_mode to simulate leverage spikes near ATM.
    stress_mode=True uses upper bound of leverage ranges.
    """
    dte_bucket, moneyness = SIGNAL_OPTION_PARAMS.get(
        signal_type, (DEFAULT_DTE_BUCKET, DEFAULT_MONEYNESS)
    )
    naked_range = NAKED_LEVERAGE_GRID.get(
        (dte_bucket, moneyness),
        NAKED_LEVERAGE_GRID.get((DEFAULT_DTE_BUCKET, DEFAULT_MONEYNESS), (8, 12))
    )
    # V7: Use midpoint for base, upper bound for stress
    if stress_mode:
        naked_leverage = naked_range[1]  # worst case
    else:
        naked_leverage = (naked_range[0] + naked_range[1]) / 2
    
    spread_range = SPREAD_LEVERAGE_GRID.get(dte_bucket, (4, 6))
    if stress_mode:
        spread_leverage = spread_range[1]
    else:
        spread_leverage = (spread_range[0] + spread_range[1]) / 2
    
    theta_decay = THETA_DECAY_GRID.get(dte_bucket, 0.02)
    return naked_leverage, spread_leverage, theta_decay, dte_bucket, moneyness


def build_trades(csv_path, years):
    """Enumerate trades using analyze_path_stats logic."""
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


def simulate_trade(trade, price_df, horizon=14, stress_mode=False):
    """Simulate V7 execution policy on a single trade.
    
    V7: Added stress_mode for worst-case leverage simulation.
    """
    dt = pd.Timestamp(trade["date"])
    direction = trade["direction"]
    trade_type = trade["type"]
    tier_num = TIER_MAP.get(trade_type)
    if tier_num is None:
        return None
    tier = TIERS[tier_num]

    if dt not in price_df.index:
        return None

    entry_price = float(price_df.loc[dt, "btc_close"])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None

    path = price_df.loc[dt:].iloc[1:horizon + 1]
    if len(path) < horizon:
        return None

    naked_leverage, spread_leverage, theta_decay, dte_bucket, moneyness = get_leverage_params(trade_type, stress_mode)

    total_risk = tier["risk"]
    spot_amount = total_risk * tier["spot_pct"]
    option_budget = total_risk * tier["option_pct"]
    naked_amount = option_budget * tier["naked_pct"]
    spread_amount = option_budget * tier["spread_pct"]

    btc_bought = spot_amount / entry_price

    naked_alive = naked_amount > 0
    naked_initial = naked_amount
    naked_partial_taken = False
    naked_closed_day = None
    naked_pnl = 0.0
    naked_remaining = naked_amount

    spread_alive = spread_amount > 0
    spread_value = spread_amount
    spread_closed_day = None
    spread_pnl = 0.0

    is_long = direction == "LONG"
    best_favorable_close = 0.0

    for day_idx in range(len(path)):
        day_num = day_idx + 1
        high = float(path.iloc[day_idx]["btc_high"])
        low = float(path.iloc[day_idx]["btc_low"])
        close = float(path.iloc[day_idx]["btc_close"])

        if is_long:
            best_move = (high / entry_price) - 1.0
            worst_move = 1.0 - (low / entry_price)
            close_move = (close / entry_price) - 1.0
        else:
            best_move = 1.0 - (low / entry_price)
            worst_move = (high / entry_price) - 1.0
            close_move = 1.0 - (close / entry_price)

        best_favorable_close = max(best_favorable_close, close_move)

        if naked_alive:
            if worst_move >= NAKED_STOP_PCT:
                loss_pct = min(NAKED_STOP_PCT * naked_leverage, 1.0)
                naked_pnl += -naked_remaining * loss_pct
                naked_alive = False
                naked_closed_day = day_num
            else:
                best_option_gain_pct = best_move * naked_leverage - (theta_decay * day_num)
                close_option_pnl_pct = close_move * naked_leverage - (theta_decay * day_num)
                current_naked_value = naked_remaining * max(1.0 + close_option_pnl_pct, 0)

                if not naked_partial_taken and best_option_gain_pct >= NAKED_TP_PARTIAL:
                    partial_size = naked_remaining * 0.60
                    naked_pnl += partial_size * NAKED_TP_PARTIAL
                    naked_remaining *= 0.40
                    naked_partial_taken = True
                    current_naked_value = naked_remaining * max(1.0 + close_option_pnl_pct, 0)

                if naked_alive and best_option_gain_pct >= NAKED_TP_FULL:
                    naked_pnl += naked_remaining * NAKED_TP_FULL
                    naked_remaining = 0
                    naked_alive = False
                    naked_closed_day = day_num

                if naked_alive and day_num in (3, 4):
                    if close_option_pnl_pct < NAKED_DAY34_THRESHOLD and best_favorable_close < 0.01:
                        naked_pnl += current_naked_value - naked_remaining
                        naked_remaining = 0
                        naked_alive = False
                        naked_closed_day = day_num

                if naked_alive and day_num >= NAKED_TIME_STOP:
                    naked_pnl += current_naked_value - naked_remaining
                    naked_remaining = 0
                    naked_alive = False
                    naked_closed_day = day_num

        if spread_alive:
            if worst_move >= SPREAD_STOP_PCT:
                loss_pct = min(SPREAD_STOP_PCT * spread_leverage, 1.0)
                spread_pnl = -spread_amount * loss_pct
                spread_alive = False
                spread_closed_day = day_num
            else:
                best_spread_pnl_pct = best_move * spread_leverage - (theta_decay * 0.3 * day_num)
                close_spread_pnl_pct = close_move * spread_leverage - (theta_decay * 0.3 * day_num)
                current_spread_value = spread_amount * (1.0 + close_spread_pnl_pct)
                current_spread_value = max(current_spread_value, 0)
                current_spread_value = min(current_spread_value, spread_amount * 2.0)

                spread_tp = SPREAD_TP_FAST if (naked_closed_day is not None and naked_closed_day <= 3) else SPREAD_TP_SLOW
                if best_spread_pnl_pct >= spread_tp:
                    spread_pnl = spread_amount * spread_tp
                    spread_alive = False
                    spread_closed_day = day_num

                if spread_alive and day_num >= SPREAD_TIME_STOP:
                    spread_pnl = current_spread_value - spread_amount
                    spread_alive = False
                    spread_closed_day = day_num

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

    final_close = float(path.iloc[-1]["btc_close"])
    spot_end_value = btc_bought * final_close
    spot_unrealized_pnl = spot_end_value - spot_amount

    total_option_pnl = naked_pnl + spread_pnl
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


class Command(BaseCommand):
    help = "Simulate V7 execution policy across historical trades"

    def add_arguments(self, parser):
        parser.add_argument(
            "--csv",
            default="features_14d_5pct.csv",
            help="Path to features CSV file",
        )
        parser.add_argument(
            "--years",
            nargs="+",
            type=int,
            default=[2021, 2022, 2023, 2024, 2025],
            help="Years to simulate",
        )
        parser.add_argument(
            "--account",
            type=int,
            default=100_000,
            help="Initial account size",
        )
        parser.add_argument(
            "--stress",
            action="store_true",
            help="Run stress test with upper-bound leverage (simulates gamma spikes)",
        )

    def handle(self, *args, **options):
        csv_path = options["csv"]
        years = options["years"]
        initial_account = options["account"]
        stress_mode = options.get("stress", False)

        mode_label = "STRESS TEST (upper-bound leverage)" if stress_mode else "BASE CASE"
        self.stdout.write(f"\n{'=' * 90}")
        self.stdout.write(f"V7 EXECUTION POLICY SIMULATION — {mode_label}")
        self.stdout.write(f"{'=' * 90}")
        self.stdout.write(f"Account: ${initial_account:,} | Years: {years} | Target: {UNDERLYING_TARGET*100:.0f}%")
        self.stdout.write(f"\nV7 Changes:")
        self.stdout.write(f"  • Naked exposure: 20-25% (was 33-37%)")
        self.stdout.write(f"  • Spread width: 8-12% (was ~6%)")
        self.stdout.write(f"  • DTE shifted: 14-21 primary (was 7-14)")
        self.stdout.write(f"  • Leverage: realistic ranges (base 6-10x, stress 12-18x)")
        self.stdout.write(f"\nDynamic Leverage Grid (V7):")
        self.stdout.write(f"  Naked: 11-14 DTE ITM: 8-12x | 15-21 DTE ITM: 6-10x")
        self.stdout.write(f"  Spread: 11-14 DTE: 4-6x | 15-21 DTE: 3-5x")
        self.stdout.write(f"  Theta: 11-14 DTE: 2%/day | 15-21 DTE: 1.5%/day")
        self.stdout.write(f"\nNaked stop: {NAKED_STOP_PCT*100:.0f}% | Spread stop: {SPREAD_STOP_PCT*100:.0f}%")
        self.stdout.write(f"Naked TP: partial@{NAKED_TP_PARTIAL*100:.0f}% / full@{NAKED_TP_FULL*100:.0f}% | Spread TP: fast@{SPREAD_TP_FAST*100:.0f}% / slow@{SPREAD_TP_SLOW*100:.0f}%")
        self.stdout.write(f"Haircut: winners×{WINNER_HAIRCUT:.0%} / losers×{LOSER_HAIRCUT:.0%} | Max concurrent: {MAX_CONCURRENT_TRADES} | Max capital at risk: {MAX_CAPITAL_AT_RISK_PCT:.0%}")
        self.stdout.write("")

        trades_df = build_trades(csv_path, years)
        if trades_df.empty:
            self.stdout.write("No trades found.")
            return

        self.stdout.write(f"Trades enumerated: {len(trades_df)}")

        price_df = load_prices()
        if price_df.empty:
            self.stdout.write("No price data.")
            return

        results = []
        open_trades = []

        for _, trade in trades_df.iterrows():
            trade_date = pd.Timestamp(trade["date"])
            tier_num = TIER_MAP.get(trade["type"])
            if tier_num is None:
                continue

            open_trades = [
                (d, c, r) for d, c, r in open_trades if (trade_date - d).days < c
            ]
            concurrent_count = len(open_trades)
            capital_at_risk = sum(r for _, _, r in open_trades)
            max_capital_at_risk = initial_account * MAX_CAPITAL_AT_RISK_PCT

            if concurrent_count >= MAX_CONCURRENT_TRADES:
                continue
            trade_risk = TIERS[tier_num]["risk"]
            if capital_at_risk + trade_risk > max_capital_at_risk:
                continue

            r = simulate_trade(trade, price_df, stress_mode=stress_mode)
            if r:
                hold_days = r.get("spread_closed_day") or SPREAD_TIME_STOP
                open_trades.append((trade_date, hold_days, r["total_risk"]))
                results.append(r)

        if not results:
            self.stdout.write("No trades with sufficient forward data.")
            return

        res_df = pd.DataFrame(results)
        self._print_results(res_df, price_df, initial_account, years)

    def _print_results(self, res_df, price_df, initial_account, years):
        self.stdout.write(f"\n{'=' * 90}")
        self.stdout.write("TRADE-BY-TRADE RESULTS")
        self.stdout.write(f"{'=' * 90}")
        self.stdout.write(f"{'Date':12s} | {'Type':16s} | {'Dir':5s} | {'DTE':5s} | {'NkLev':5s} | {'Naked P&L':>10s} | {'Spread P&L':>10s} | {'Option P&L':>10s} | {'Win':3s}")
        self.stdout.write("-" * 105)

        cumulative_btc = 0
        equity = initial_account
        peak_equity = equity
        max_drawdown = 0
        equity_curve = []

        for _, r in res_df.iterrows():
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
            self.stdout.write(
                f"{r['date'].strftime('%Y-%m-%d'):12s} | {r['type']:16s} | {r['direction']:5s} | {r['dte_bucket']:5s} | "
                f"{r['naked_leverage']:5.1f} | ${r['naked_pnl']:>+9.2f} | ${r['spread_pnl']:>+9.2f} | ${r['option_pnl']:>+9.2f} | {win_str}"
            )

        self.stdout.write(f"\n{'=' * 90}")
        self.stdout.write("EQUITY CURVE")
        self.stdout.write(f"{'=' * 90}")
        self.stdout.write(f"{'Date':12s} | {'Equity':>12s} | {'Drawdown':>10s} | {'BTC Held':>10s}")
        self.stdout.write("-" * 55)
        for e in equity_curve:
            self.stdout.write(
                f"{e['date'].strftime('%Y-%m-%d'):12s} | ${e['equity']:>11,.2f} | ${e['drawdown']:>9,.2f} | "
                f"{e['cumulative_btc']:.5f}"
            )

        self._print_summary(res_df, price_df, initial_account, equity, max_drawdown, years)

    def _print_summary(self, res_df, price_df, initial_account, equity, max_drawdown, years):
        self.stdout.write(f"\n{'=' * 90}")
        self.stdout.write("SUMMARY")
        self.stdout.write(f"{'=' * 90}")

        total_trades = len(res_df)
        winners = res_df[res_df["win"] == 1]
        losers = res_df[res_df["win"] == 0]

        self.stdout.write(f"Total trades: {total_trades}")
        self.stdout.write(f"Winners: {len(winners)} ({len(winners)/total_trades*100:.1f}%)")
        self.stdout.write(f"Losers: {len(losers)} ({len(losers)/total_trades*100:.1f}%)")
        self.stdout.write("")

        total_option_pnl = res_df["option_pnl"].sum()
        total_btc_accumulated = res_df["spot_btc"].sum()
        total_spot_deployed = res_df["spot_amount"].sum()

        self.stdout.write(f"Total option P&L: ${total_option_pnl:>+,.2f}")
        self.stdout.write(f"Avg winner: ${winners['option_pnl'].mean():>+,.2f}" if len(winners) else "Avg winner: N/A")
        self.stdout.write(f"Avg loser: ${losers['option_pnl'].mean():>+,.2f}" if len(losers) else "Avg loser: N/A")
        self.stdout.write(f"Largest winner: ${res_df['option_pnl'].max():>+,.2f}")
        self.stdout.write(f"Largest loser: ${res_df['option_pnl'].min():>+,.2f}")
        self.stdout.write("")

        self.stdout.write(f"Final equity (options only): ${equity:>,.2f}")
        self.stdout.write(f"Option return: {(equity - initial_account) / initial_account * 100:>+.2f}%")
        self.stdout.write(f"Max drawdown: ${max_drawdown:>,.2f} ({max_drawdown/initial_account*100:.2f}%)")
        self.stdout.write("")

        latest_btc = float(price_df.iloc[-1]["btc_close"])
        btc_value = total_btc_accumulated * latest_btc

        self.stdout.write(f"BTC accumulated: {total_btc_accumulated:.6f} BTC")
        self.stdout.write(f"Total spot deployed: ${total_spot_deployed:>,.2f}")
        self.stdout.write(f"BTC value (at ${latest_btc:,.0f}): ${btc_value:>,.2f}")
        self.stdout.write(f"Spot unrealized P&L: ${btc_value - total_spot_deployed:>+,.2f}")
        self.stdout.write("")

        combined_return = (equity - initial_account) + (btc_value - total_spot_deployed)
        self.stdout.write(f"Combined return (options realized + spot unrealized): ${combined_return:>+,.2f}")
        self.stdout.write(f"Combined return %: {combined_return / initial_account * 100:>+.2f}%")

        self._print_breakdowns(res_df, years)

    def _print_breakdowns(self, res_df, years):
        self.stdout.write(f"\n{'=' * 90}")
        self.stdout.write("BY TIER")
        self.stdout.write(f"{'=' * 90}")
        for tier_num in [1, 2, 3]:
            tier_df = res_df[res_df["tier"] == tier_num]
            if tier_df.empty:
                continue
            t_wins = tier_df[tier_df["win"] == 1]
            self.stdout.write(
                f"Tier {tier_num}: {len(tier_df)} trades | "
                f"Win rate: {len(t_wins)/len(tier_df)*100:.1f}% | "
                f"Option P&L: ${tier_df['option_pnl'].sum():>+,.2f} | "
                f"Avg: ${tier_df['option_pnl'].mean():>+,.2f}"
            )

        self.stdout.write(f"\n{'=' * 90}")
        self.stdout.write("BY YEAR")
        self.stdout.write(f"{'=' * 90}")
        for year in years:
            year_df = res_df[res_df["date"].dt.year == year]
            if year_df.empty:
                continue
            y_wins = year_df[year_df["win"] == 1]
            self.stdout.write(
                f"{year}: {len(year_df)} trades | "
                f"Win rate: {len(y_wins)/len(year_df)*100:.1f}% | "
                f"Option P&L: ${year_df['option_pnl'].sum():>+,.2f} | "
                f"BTC accumulated: {year_df['spot_btc'].sum():.6f}"
            )

        self.stdout.write(f"\n{'=' * 90}")
        self.stdout.write("BY TYPE")
        self.stdout.write(f"{'=' * 90}")
        for ttype in res_df["type"].unique():
            type_df = res_df[res_df["type"] == ttype]
            tw = type_df[type_df["win"] == 1]
            self.stdout.write(
                f"{ttype:16s}: {len(type_df):2d} trades | "
                f"Win: {len(tw)/len(type_df)*100:5.1f}% | "
                f"P&L: ${type_df['option_pnl'].sum():>+8,.2f} | "
                f"Avg: ${type_df['option_pnl'].mean():>+7,.2f}"
            )

        self.stdout.write(f"\n{'=' * 90}")
        self.stdout.write("BY LEVERAGE PROFILE")
        self.stdout.write(f"{'=' * 90}")
        for dte in ["7_10", "11_14", "15_21"]:
            dte_df = res_df[res_df["dte_bucket"] == dte]
            if dte_df.empty:
                continue
            dw = dte_df[dte_df["win"] == 1]
            avg_naked_lev = dte_df["naked_leverage"].mean()
            avg_spread_lev = dte_df["spread_leverage"].mean()
            self.stdout.write(
                f"DTE {dte:5s}: {len(dte_df):2d} trades | "
                f"Win: {len(dw)/len(dte_df)*100:5.1f}% | "
                f"Naked Lev: {avg_naked_lev:5.1f}x | "
                f"Spread Lev: {avg_spread_lev:4.1f}x | "
                f"P&L: ${dte_df['option_pnl'].sum():>+8,.2f}"
            )

        self.stdout.write(f"\n{'=' * 90}")
        self.stdout.write("SIGNAL TYPE → LEVERAGE MAPPING")
        self.stdout.write(f"{'=' * 90}")
        self.stdout.write(f"{'Type':16s} | {'DTE':5s} | {'Moneyness':12s} | {'Naked Lev':>9s} | {'Spread Lev':>10s} | {'Theta':>6s}")
        self.stdout.write("-" * 75)
        for sig_type in sorted(SIGNAL_OPTION_PARAMS.keys()):
            dte, money = SIGNAL_OPTION_PARAMS[sig_type]
            nk_lev, sp_lev, theta, _, _ = get_leverage_params(sig_type)
            self.stdout.write(
                f"{sig_type:16s} | {dte:5s} | {money:12s} | {nk_lev:8.1f}x | {sp_lev:9.1f}x | {theta*100:5.1f}%"
            )
