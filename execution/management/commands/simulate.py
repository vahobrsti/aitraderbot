"""
V8 Execution Policy Simulator

Simulates options execution design across historical trades using
actual forward price paths from RawDailyData.

Pricing Modes:
- linear: Legacy fixed leverage model (backward compatible)
- payoff_only: Path-exact payoff geometry (bounded P&L from structure)
- synthetic_iv: Black-Scholes with regime-mapped synthetic IV
- learned: ML-based leverage prediction from OptionSnapshot data

Same-Bar Ambiguity Handling:
- stop_first: Assume stop hit first (conservative, default)
- tp_first: Assume TP hit first (optimistic)
- probabilistic: Weight by distance to each level
- close_only: Only use close price, ignore intrabar

Scenario Outputs:
- base: Mid fills, stable IV
- conservative: Worst spread, mild IV crush on winners
- stressed: Adverse IV shift, worst fills + slippage

Usage:
    python manage.py simulate
    python manage.py simulate --pricing-mode payoff_only
    python manage.py simulate --pricing-mode learned --predictor-model models/option_response_predictor.json
    python manage.py simulate --pricing-mode synthetic_iv --same-bar-policy stop_first
    python manage.py simulate --years 2023 2024 --scenarios
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from django.core.management.base import BaseCommand

from datafeed.models import RawDailyData
from execution.services.option_pricer import (
    PricingMode,
    OptionPricer,
    LeveragePredictor,
    ScenarioRunner,
    SameBarPolicy,
    resolve_same_bar_ambiguity,
    BlackScholes,
)


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
    # Tier 1: $4000 risk, 20% naked, 80% spread
    "PRIMARY_SHORT": 1,
    "OPTION_CALL": 1,
    "CALL": 1,
    "PUT": 1,
    # Tier 2: $2400 risk, 25% naked, 75% spread
    "BULL_PROBE": 2,
    "BEAR_PROBE": 2,
    "TACTICAL_PUT": 2,
    "OPTION_PUT": 2,
    "MVRV_SHORT": 2,
    # IRON_CONDOR excluded - different structure, handled separately
    # LONG excluded - not in policy
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

# Default realized volatility for synthetic IV (annualized)
DEFAULT_REALIZED_VOL = 0.60


@dataclass
class SimulationConfig:
    """Configuration for simulation run."""
    pricing_mode: PricingMode = PricingMode.PAYOFF_ONLY
    same_bar_policy: SameBarPolicy = SameBarPolicy.STOP_FIRST
    stress_mode: bool = False
    run_scenarios: bool = False
    predictor_model_path: Optional[Path] = None
    realized_vol: float = DEFAULT_REALIZED_VOL
    
    # Spread structure parameters
    spread_width_pct: float = 0.10  # 10% spread width


@dataclass
class ScenarioResult:
    """Results for a single scenario."""
    pnl: float
    pnl_pct: float
    label: str
    iv_assumption: str


@dataclass
class TradeResult:
    """Complete trade result with optional scenario breakdown."""
    date: pd.Timestamp
    trade_type: str
    direction: str
    tier: int
    dte_bucket: str
    moneyness: str
    entry_price: float
    total_risk: float
    
    # P&L by component
    naked_pnl: float
    spread_pnl: float
    option_pnl: float
    spot_btc: float
    spot_unrealized: float
    
    # Leverage used
    naked_leverage: float
    spread_leverage: float
    theta_decay: float
    
    # Exit info
    naked_closed_day: Optional[int]
    spread_closed_day: Optional[int]
    exit_reason: str  # 'tp', 'stop', 'time_stop', 'same_bar_ambiguity'
    
    # Win/loss
    win: int
    
    # Scenario results (if run_scenarios=True)
    scenarios: Optional[Dict[str, ScenarioResult]] = None
    
    # Pricing mode used
    pricing_mode: str = "linear"


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
    """Simulate execution policy on a single trade (legacy linear mode).
    
    Added stress_mode for worst-case leverage simulation.
    """
    return simulate_trade_v8(
        trade, price_df, horizon=horizon,
        config=SimulationConfig(
            pricing_mode=PricingMode.LINEAR,
            stress_mode=stress_mode,
        )
    )


def simulate_trade_v8(
    trade,
    price_df: pd.DataFrame,
    horizon: int = 14,
    config: Optional[SimulationConfig] = None,
    pricer: Optional[OptionPricer] = None,
    predictor: Optional[LeveragePredictor] = None,
) -> Optional[Dict[str, Any]]:
    """
    Simulate trade with configurable pricing mode.
    
    Supports:
    - linear: Legacy fixed leverage model
    - payoff_only: Path-exact payoff geometry
    - synthetic_iv: Black-Scholes with regime-mapped IV
    - learned: ML-based leverage prediction
    """
    config = config or SimulationConfig()
    
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

    # Get base leverage params
    naked_leverage, spread_leverage, theta_decay, dte_bucket, moneyness = get_leverage_params(
        trade_type, config.stress_mode
    )
    
    is_long = direction == "LONG"
    option_type = "call" if is_long else "put"
    
    # Calculate strike based on moneyness bucket
    moneyness_pct = {"atm": 0.0, "slightly_itm": -0.03, "slightly_otm": 0.03, 
                    "deep_itm": -0.08, "deep_otm": 0.08}.get(moneyness, 0.0)
    strike = entry_price * (1 + moneyness_pct)
    dte = {"7_10": 8, "11_14": 12, "15_21": 18}.get(dte_bucket, 12)
    
    # Initialize pricer if needed
    if config.pricing_mode != PricingMode.LINEAR and pricer is None:
        pricer = OptionPricer(mode=config.pricing_mode)
    
    # Calculate entry option price for non-linear modes
    entry_option_price = None
    entry_iv = config.realized_vol
    
    if config.pricing_mode in (PricingMode.SYNTHETIC_IV, PricingMode.PAYOFF_ONLY):
        entry_result = pricer.price_at_entry(
            spot=entry_price, strike=strike, dte=dte,
            option_type=option_type, realized_vol=config.realized_vol, regime="neutral",
        )
        entry_option_price = entry_result.get("ask") or entry_result.get("mid")
        entry_iv = entry_result.get("iv", config.realized_vol)
    
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
    exit_reason = "time_stop"

    spread_alive = spread_amount > 0
    spread_closed_day = None
    spread_pnl = 0.0
    best_favorable_close = 0.0

    for day_idx in range(len(path)):
        day_num = day_idx + 1
        high = float(path.iloc[day_idx]["btc_high"])
        low = float(path.iloc[day_idx]["btc_low"])
        close = float(path.iloc[day_idx]["btc_close"])
        current_dte = max(dte - day_num, 0)

        if is_long:
            best_move = (high / entry_price) - 1.0
            worst_move = 1.0 - (low / entry_price)
            close_move = (close / entry_price) - 1.0
        else:
            best_move = 1.0 - (low / entry_price)
            worst_move = (high / entry_price) - 1.0
            close_move = 1.0 - (close / entry_price)

        best_favorable_close = max(best_favorable_close, close_move)
        
        # Calculate option P&L based on pricing mode
        if config.pricing_mode == PricingMode.LINEAR:
            option_pnl_pct = close_move * naked_leverage - (theta_decay * day_num)
            best_option_gain_pct = best_move * naked_leverage - (theta_decay * day_num)
        elif config.pricing_mode == PricingMode.PAYOFF_ONLY:
            option_pnl_pct = _payoff_pnl(entry_price, close, strike, option_type, theta_decay, day_num)
            best_option_gain_pct = _payoff_pnl(entry_price, high if is_long else low, strike, option_type, theta_decay, day_num)
        elif config.pricing_mode == PricingMode.SYNTHETIC_IV and pricer:
            exit_result = pricer.price_at_exit(
                spot=close, strike=strike, dte=current_dte, option_type=option_type,
                entry_price=entry_option_price, realized_vol=config.realized_vol, regime="neutral",
            )
            option_pnl_pct = exit_result.get("pnl_pct", 0) or 0
            best_result = pricer.price_at_exit(
                spot=high if is_long else low, strike=strike, dte=current_dte, option_type=option_type,
                entry_price=entry_option_price, realized_vol=config.realized_vol, regime="neutral",
            )
            best_option_gain_pct = best_result.get("pnl_pct", 0) or 0
        elif config.pricing_mode == PricingMode.LEARNED and predictor:
            prediction = predictor.predict(
                dte=dte, moneyness=moneyness_pct, option_type=option_type,
                btc_change_pct=close_move if is_long else -close_move, iv=entry_iv, days_held=day_num,
            )
            option_pnl_pct = prediction.p50
            best_prediction = predictor.predict(
                dte=dte, moneyness=moneyness_pct, option_type=option_type,
                btc_change_pct=best_move if is_long else -best_move, iv=entry_iv, days_held=day_num,
            )
            best_option_gain_pct = best_prediction.p50
        else:
            option_pnl_pct = close_move * naked_leverage - (theta_decay * day_num)
            best_option_gain_pct = best_move * naked_leverage - (theta_decay * day_num)
        
        # Same-bar ambiguity check
        tp_level = entry_price * (1 + NAKED_TP_PARTIAL / naked_leverage) if is_long else entry_price * (1 - NAKED_TP_PARTIAL / naked_leverage)
        sl_level = entry_price * (1 - NAKED_STOP_PCT) if is_long else entry_price * (1 + NAKED_STOP_PCT)
        exit_type, _ = resolve_same_bar_ambiguity(
            high=high, low=low, close=close, entry_price=entry_price,
            tp_level=tp_level, sl_level=sl_level, is_long=is_long, policy=config.same_bar_policy,
        )

        if naked_alive:
            if worst_move >= NAKED_STOP_PCT:
                loss_pct = min(NAKED_STOP_PCT * naked_leverage, 1.0)
                naked_pnl += -naked_remaining * loss_pct
                naked_alive = False
                naked_closed_day = day_num
                exit_reason = "stop"
            else:
                current_naked_value = naked_remaining * max(1.0 + option_pnl_pct, 0)
                if not naked_partial_taken and best_option_gain_pct >= NAKED_TP_PARTIAL:
                    partial_size = naked_remaining * 0.60
                    naked_pnl += partial_size * NAKED_TP_PARTIAL
                    naked_remaining *= 0.40
                    naked_partial_taken = True
                    current_naked_value = naked_remaining * max(1.0 + option_pnl_pct, 0)
                    exit_reason = "tp_partial"
                if naked_alive and best_option_gain_pct >= NAKED_TP_FULL:
                    naked_pnl += naked_remaining * NAKED_TP_FULL
                    naked_remaining = 0
                    naked_alive = False
                    naked_closed_day = day_num
                    exit_reason = "tp"
                if naked_alive and day_num in (3, 4):
                    if option_pnl_pct < NAKED_DAY34_THRESHOLD and best_favorable_close < 0.01:
                        naked_pnl += current_naked_value - naked_remaining
                        naked_remaining = 0
                        naked_alive = False
                        naked_closed_day = day_num
                        exit_reason = "day34_cut"
                if naked_alive and day_num >= NAKED_TIME_STOP:
                    naked_pnl += current_naked_value - naked_remaining
                    naked_remaining = 0
                    naked_alive = False
                    naked_closed_day = day_num
                    exit_reason = "time_stop"

        if spread_alive:
            spread_pnl_pct = option_pnl_pct * 0.6 if config.pricing_mode != PricingMode.LINEAR else close_move * spread_leverage - (theta_decay * 0.3 * day_num)
            best_spread_pnl_pct = best_option_gain_pct * 0.6 if config.pricing_mode != PricingMode.LINEAR else best_move * spread_leverage - (theta_decay * 0.3 * day_num)
            
            if worst_move >= SPREAD_STOP_PCT:
                loss_pct = min(SPREAD_STOP_PCT * spread_leverage, 1.0)
                spread_pnl = -spread_amount * loss_pct
                spread_alive = False
                spread_closed_day = day_num
                if exit_reason == "time_stop":
                    exit_reason = "stop"
            else:
                current_spread_value = spread_amount * (1.0 + spread_pnl_pct)
                current_spread_value = max(min(current_spread_value, spread_amount * 2.0), 0)
                spread_tp = SPREAD_TP_FAST if (naked_closed_day and naked_closed_day <= 3) else SPREAD_TP_SLOW
                if best_spread_pnl_pct >= spread_tp:
                    spread_pnl = spread_amount * spread_tp
                    spread_alive = False
                    spread_closed_day = day_num
                    if exit_reason == "time_stop":
                        exit_reason = "tp"
                if spread_alive and day_num >= SPREAD_TIME_STOP:
                    spread_pnl = current_spread_value - spread_amount
                    spread_alive = False
                    spread_closed_day = day_num

    # Handle positions still open at horizon end
    if naked_alive:
        final_close = float(path.iloc[-1]["btc_close"])
        final_move = (final_close / entry_price) - 1.0 if is_long else 1.0 - (final_close / entry_price)
        final_pnl_pct = final_move * naked_leverage - (theta_decay * horizon)
        naked_pnl += naked_remaining * max(final_pnl_pct, -1.0)

    if spread_alive:
        final_close = float(path.iloc[-1]["btc_close"])
        final_move = (final_close / entry_price) - 1.0 if is_long else 1.0 - (final_close / entry_price)
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

    result = {
        "date": dt, "type": trade_type, "direction": direction, "tier": tier_num,
        "dte_bucket": dte_bucket, "moneyness": moneyness,
        "naked_leverage": naked_leverage, "spread_leverage": spread_leverage, "theta_decay": theta_decay,
        "entry_price": entry_price, "total_risk": total_risk,
        "naked_amount": naked_initial, "spread_amount": spread_amount, "spot_amount": spot_amount,
        "naked_pnl": round(naked_pnl, 2), "spread_pnl": round(spread_pnl, 2), "option_pnl": round(total_option_pnl, 2),
        "spot_btc": btc_bought, "spot_unrealized": round(spot_unrealized_pnl, 2),
        "naked_closed_day": naked_closed_day, "spread_closed_day": spread_closed_day,
        "exit_reason": exit_reason, "win": 1 if total_option_pnl > 0 else 0,
        "pricing_mode": config.pricing_mode.value,
    }
    
    # Run scenarios if requested
    if config.run_scenarios and pricer:
        runner = ScenarioRunner(pricer)
        scenarios = runner.run_scenarios(
            spot_entry=entry_price, spot_exit=final_close, strike=strike,
            dte_entry=dte, dte_exit=max(dte - (naked_closed_day or spread_closed_day or horizon), 0),
            option_type=option_type, entry_iv=entry_iv, realized_vol=config.realized_vol,
        )
        result["scenarios"] = {
            k: {"pnl": round((naked_initial + spread_amount) * (v.get("pnl_pct") or 0), 2),
                "pnl_pct": round((v.get("pnl_pct") or 0) * 100, 2),
                "iv_assumption": v.get("iv_assumption", "")}
            for k, v in scenarios.items()
        }
    
    return result


def _payoff_pnl(entry_spot, current_spot, strike, option_type, theta_decay, day_num):
    """Calculate P&L using payoff-only mode."""
    if option_type == "call":
        intrinsic = max(current_spot - strike, 0)
        entry_intrinsic = max(entry_spot - strike, 0)
    else:
        intrinsic = max(strike - current_spot, 0)
        entry_intrinsic = max(strike - entry_spot, 0)
    time_value_entry = entry_spot * 0.03
    entry_premium = entry_intrinsic + time_value_entry
    time_value_current = time_value_entry * max(1 - theta_decay * day_num, 0)
    current_value = intrinsic + time_value_current
    return (current_value - entry_premium) / entry_premium if entry_premium > 0 else 0.0


class Command(BaseCommand):
    help = "Simulate execution policy across historical trades with configurable pricing modes"

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
        parser.add_argument(
            "--pricing-mode",
            choices=["linear", "payoff_only", "synthetic_iv", "learned"],
            default="linear",
            help="Pricing mode: linear (legacy), payoff_only (bounded), synthetic_iv (Black-Scholes), learned (ML)",
        )
        parser.add_argument(
            "--same-bar-policy",
            choices=["stop_first", "tp_first", "probabilistic", "close_only"],
            default="stop_first",
            help="Policy for same-bar TP/SL ambiguity",
        )
        parser.add_argument(
            "--scenarios",
            action="store_true",
            help="Run base/conservative/stressed scenarios for each trade",
        )
        parser.add_argument(
            "--predictor-model",
            default="models/option_response_predictor.json",
            help="Path to trained LeveragePredictor model (for learned mode)",
        )
        parser.add_argument(
            "--realized-vol",
            type=float,
            default=0.60,
            help="Realized volatility for synthetic IV (annualized, e.g., 0.60 = 60%%)",
        )

    def handle(self, *args, **options):
        csv_path = options["csv"]
        years = options["years"]
        initial_account = options["account"]
        stress_mode = options.get("stress", False)
        
        # Parse pricing mode
        pricing_mode_str = options.get("pricing_mode", "linear")
        pricing_mode = {
            "linear": PricingMode.LINEAR,
            "payoff_only": PricingMode.PAYOFF_ONLY,
            "synthetic_iv": PricingMode.SYNTHETIC_IV,
            "learned": PricingMode.LEARNED,
        }[pricing_mode_str]
        
        # Parse same-bar policy
        same_bar_str = options.get("same_bar_policy", "stop_first")
        same_bar_policy = {
            "stop_first": SameBarPolicy.STOP_FIRST,
            "tp_first": SameBarPolicy.TP_FIRST,
            "probabilistic": SameBarPolicy.PROBABILISTIC,
            "close_only": SameBarPolicy.CLOSE_ONLY,
        }[same_bar_str]
        
        run_scenarios = options.get("scenarios", False)
        predictor_model_path = Path(options.get("predictor_model", "models/option_response_predictor.json"))
        realized_vol = options.get("realized_vol", 0.60)
        
        # Build config
        config = SimulationConfig(
            pricing_mode=pricing_mode,
            same_bar_policy=same_bar_policy,
            stress_mode=stress_mode,
            run_scenarios=run_scenarios,
            predictor_model_path=predictor_model_path,
            realized_vol=realized_vol,
        )
        
        # Load predictor if using learned mode
        predictor = None
        if pricing_mode == PricingMode.LEARNED:
            if predictor_model_path.exists():
                predictor = LeveragePredictor(model_path=predictor_model_path)
                self.stdout.write(f"Loaded predictor from {predictor_model_path}")
            else:
                self.stdout.write(self.style.WARNING(
                    f"Predictor model not found at {predictor_model_path}, falling back to synthetic estimates"
                ))
                predictor = LeveragePredictor()
        
        # Initialize pricer for non-linear modes
        pricer = None
        if pricing_mode != PricingMode.LINEAR:
            pricer = OptionPricer(mode=pricing_mode)

        mode_label = f"{pricing_mode_str.upper()}"
        if stress_mode:
            mode_label += " + STRESS"
        if run_scenarios:
            mode_label += " + SCENARIOS"
            
        self.stdout.write(f"\n{'=' * 90}")
        self.stdout.write(f"EXECUTION POLICY SIMULATION — {mode_label}")
        self.stdout.write(f"{'=' * 90}")
        self.stdout.write(f"Account: ${initial_account:,} | Years: {years} | Target: {UNDERLYING_TARGET*100:.0f}%")
        self.stdout.write(f"Pricing mode: {pricing_mode_str} | Same-bar policy: {same_bar_str}")
        if pricing_mode == PricingMode.SYNTHETIC_IV:
            self.stdout.write(f"Realized vol: {realized_vol:.0%}")
        self.stdout.write(f"\nDynamic Leverage Grid:")
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

            r = simulate_trade_v8(trade, price_df, config=config, pricer=pricer, predictor=predictor)
            if r:
                hold_days = r.get("spread_closed_day") or SPREAD_TIME_STOP
                open_trades.append((trade_date, hold_days, r["total_risk"]))
                results.append(r)

        if not results:
            self.stdout.write("No trades with sufficient forward data.")
            return

        res_df = pd.DataFrame(results)
        self._print_results(res_df, price_df, initial_account, years, config)

    def _print_results(self, res_df, price_df, initial_account, years, config: SimulationConfig):
        self.stdout.write(f"\n{'=' * 90}")
        self.stdout.write("TRADE-BY-TRADE RESULTS")
        self.stdout.write(f"{'=' * 90}")
        
        if config.run_scenarios:
            self.stdout.write(f"{'Date':12s} | {'Type':16s} | {'Dir':5s} | {'Exit':10s} | {'Option P&L':>10s} | {'Base':>8s} | {'Cons':>8s} | {'Stress':>8s} | {'Win':3s}")
        else:
            self.stdout.write(f"{'Date':12s} | {'Type':16s} | {'Dir':5s} | {'DTE':5s} | {'NkLev':5s} | {'Naked P&L':>10s} | {'Spread P&L':>10s} | {'Option P&L':>10s} | {'Win':3s}")
        self.stdout.write("-" * 115)

        cumulative_btc = 0
        equity = initial_account
        peak_equity = equity
        max_drawdown = 0
        equity_curve = []
        
        # Scenario tracking
        scenario_equity = {"base": initial_account, "conservative": initial_account, "stressed": initial_account}

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
            
            if config.run_scenarios and "scenarios" in r and r["scenarios"]:
                scenarios = r["scenarios"]
                base_pnl = scenarios.get("base", {}).get("pnl", 0)
                cons_pnl = scenarios.get("conservative", {}).get("pnl", 0)
                stress_pnl = scenarios.get("stressed", {}).get("pnl", 0)
                
                scenario_equity["base"] += base_pnl
                scenario_equity["conservative"] += cons_pnl
                scenario_equity["stressed"] += stress_pnl
                
                exit_reason = r.get("exit_reason", "")[:10]
                self.stdout.write(
                    f"{r['date'].strftime('%Y-%m-%d'):12s} | {r['type']:16s} | {r['direction']:5s} | {exit_reason:10s} | "
                    f"${r['option_pnl']:>+9.2f} | ${base_pnl:>+7.0f} | ${cons_pnl:>+7.0f} | ${stress_pnl:>+7.0f} | {win_str}"
                )
            else:
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

        self._print_summary(res_df, price_df, initial_account, equity, max_drawdown, years, config, scenario_equity)

    def _print_summary(self, res_df, price_df, initial_account, equity, max_drawdown, years, config: SimulationConfig, scenario_equity: Dict[str, float]):
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
        
        # Scenario comparison
        if config.run_scenarios:
            self.stdout.write(f"\n{'=' * 90}")
            self.stdout.write("SCENARIO COMPARISON")
            self.stdout.write(f"{'=' * 90}")
            self.stdout.write(f"{'Scenario':<15s} | {'Final Equity':>14s} | {'Return':>10s} | {'vs Base':>10s}")
            self.stdout.write("-" * 60)
            
            base_eq = scenario_equity.get("base", initial_account)
            for scenario_name in ["base", "conservative", "stressed"]:
                eq = scenario_equity.get(scenario_name, initial_account)
                ret = (eq - initial_account) / initial_account * 100
                vs_base = (eq - base_eq) / initial_account * 100 if scenario_name != "base" else 0
                self.stdout.write(
                    f"{scenario_name.capitalize():<15s} | ${eq:>13,.2f} | {ret:>+9.2f}% | {vs_base:>+9.2f}%"
                )
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
