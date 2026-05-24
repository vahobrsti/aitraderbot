"""
Trade Setup Builder - Generates complete trade setups from signals.

Combines signal data with option snapshots and policy to produce
executable trade setups with full validation.

Usage:
    from execution.services.trade_setup import TradeSetupBuilder
    
    builder = TradeSetupBuilder()
    setup = builder.build_setup(signal_date=date(2026, 5, 2))
    
    # For Telegram
    message = setup.to_telegram_message()
    
    # For API
    data = setup.to_dict()
"""
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from typing import Optional
import math

from django.db.models import Max

from datafeed.models import OptionSnapshot, RawDailyData
from signals.models import DailySignal
from execution.services.policy import get_policy, PolicyVersion
from execution.services.trade_validator import (
    TradeValidator, 
    SpreadPlan, 
    ValidationResult,
    IssueSeverity,
)


@dataclass
class LegSetup:
    """Single option leg details."""
    symbol: str
    action: str  # "BUY" or "SELL"
    strike: float
    delta: float
    iv: float
    price: float  # Ask for buy, Bid for sell
    open_interest: int
    bid_ask_spread_pct: float


@dataclass
class ExitRules:
    """
    Exit rule configuration with path-aware enhancements.
    
    Standard exits:
    - stop_loss_spot: BTC price trigger
    - stop_loss_value: Spread value trigger
    - take_profit: % of max profit
    - max_hold: Days until forced exit
    - scale_down: Day to reduce position
    
    Path-aware exits:
    - profit_lock: Move stop to breakeven after threshold
    - trailing_stop: Trail stop at % below high water mark
    - stop_tighten: Tighten stop on specific day
    """
    # Standard exits
    stop_loss_spot: float       # BTC price trigger
    stop_loss_spot_pct: float   # % move from entry
    stop_loss_value: float      # Spread value trigger
    stop_loss_value_pct: float  # % of debit lost
    take_profit_pct: float      # % of max profit
    take_profit_value: float    # $ per contract
    max_hold_days: int
    max_hold_date: date
    scale_down_day: Optional[int]
    scale_down_date: Optional[date]
    scale_down_action: str      # "reduce_50pct" or "close_full"
    # Path-aware exits
    profit_lock_threshold: float = 0.30      # Lock profits after this % of max profit
    profit_lock_stop: Optional[float] = None # Stop price after profit lock (breakeven)
    trailing_stop_pct: float = 0.0           # Trail at this % below high (0 = disabled)
    stop_tighten_day: Optional[int] = None   # Day to tighten stop
    stop_tighten_date: Optional[date] = None # Date to tighten stop
    tightened_stop_pct: Optional[float] = None  # Tightened stop loss %


@dataclass
class PathProfile:
    """Path characteristics for the signal type."""
    shakeout_pct: float         # % of winners with shakeout path
    invalidation_pct: float     # % of winners invalidated before hit
    mae_p75: float              # 75th percentile MAE for winners
    clean_win_pct: float        # % of winners with clean paths
    is_shakeout_heavy: bool     # True if shakeout_pct >= 40%
    is_invalidation_heavy: bool # True if invalidation_pct >= 35%
    entry_strategy: str         # "single", "dca", or "scaled"
    entry_note: str             # Human-readable entry guidance


@dataclass
class TradeSetup:
    """Complete trade setup with all execution details."""
    # Signal info
    signal_date: date
    signal_type: str
    direction: str  # "LONG" or "SHORT" or "NEUTRAL"
    
    # Market context
    spot_price: float
    
    # Option details
    expiry: date
    dte: int
    long_leg: LegSetup
    short_leg: Optional[LegSetup]
    
    # Spread metrics
    spread_width: float
    spread_width_pct: float
    net_debit: float
    max_profit: float
    max_loss: float
    risk_reward: float
    breakeven: float
    
    # Adjusted metrics (after costs)
    execution_cost: float
    adjusted_max_profit: float
    net_edge_pct: float
    
    # Position sizing
    risk_budget: float
    contracts: int
    total_risk: float
    total_max_profit: float
    
    # Exit rules
    exit_rules: ExitRules
    
    # Validation
    validation_passed: bool
    validation_warnings: list[str] = field(default_factory=list)
    validation_blocking: list[str] = field(default_factory=list)
    
    # Extra legs (for multi-leg structures like iron condors)
    extra_legs: list[LegSetup] = field(default_factory=list)
    
    # Path profile (from historical analysis) - optional
    path_profile: Optional[PathProfile] = None
    
    # Policy version
    policy_version: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "signal_date": self.signal_date.isoformat(),
            "signal_type": self.signal_type,
            "direction": self.direction,
            "spot_price": self.spot_price,
            "expiry": self.expiry.isoformat(),
            "dte": self.dte,
            "legs": {
                "long": {
                    "symbol": self.long_leg.symbol,
                    "action": self.long_leg.action,
                    "strike": self.long_leg.strike,
                    "delta": self.long_leg.delta,
                    "iv": self.long_leg.iv,
                    "price": self.long_leg.price,
                    "open_interest": self.long_leg.open_interest,
                },
                "short": {
                    "symbol": self.short_leg.symbol,
                    "action": self.short_leg.action,
                    "strike": self.short_leg.strike,
                    "delta": self.short_leg.delta,
                    "iv": self.short_leg.iv,
                    "price": self.short_leg.price,
                    "open_interest": self.short_leg.open_interest,
                } if self.short_leg else None,
                "extra": [
                    {
                        "symbol": leg.symbol,
                        "action": leg.action,
                        "strike": leg.strike,
                        "delta": leg.delta,
                        "iv": leg.iv,
                        "price": leg.price,
                        "open_interest": leg.open_interest,
                    }
                    for leg in self.extra_legs
                ] if self.extra_legs else None,
            },
            "metrics": {
                "spread_width": self.spread_width,
                "spread_width_pct": self.spread_width_pct,
                "net_debit": self.net_debit,
                "max_profit": self.max_profit,
                "max_loss": self.max_loss,
                "risk_reward": self.risk_reward,
                "risk_reward_formatted": f"1:{self.risk_reward:.2f}",
                "breakeven": self.breakeven,
                "execution_cost": self.execution_cost,
                "adjusted_max_profit": self.adjusted_max_profit,
                "net_edge_pct": self.net_edge_pct,
            },
            "risk_reward_summary": {
                "ratio": self.risk_reward,
                "ratio_formatted": f"1:{self.risk_reward:.2f}",
                "max_loss_per_contract": self.max_loss,
                "max_profit_per_contract": self.max_profit,
                # Credit structures (condors): risk/reward based on credit economics
                # Debit structures (spreads): risk/reward based on spot move
                "risk_pct": self.max_loss / self.spot_price if self.direction == "NEUTRAL" else self.exit_rules.stop_loss_spot_pct,
                "risk_usd": self.max_loss if self.direction == "NEUTRAL" else self.spot_price * self.exit_rules.stop_loss_spot_pct,
                "reward_pct": self.max_profit / self.spot_price if self.direction == "NEUTRAL" else self.spread_width_pct * self.exit_rules.take_profit_pct,
                "reward_usd": self.max_profit if self.direction == "NEUTRAL" else self.spot_price * self.spread_width_pct * self.exit_rules.take_profit_pct,
            },
            "position": {
                "risk_budget": self.risk_budget,
                "contracts": self.contracts,
                "total_risk": self.total_risk,
                "total_max_profit": self.total_max_profit,
            },
            "exit_rules": {
                "stop_loss_spot": self.exit_rules.stop_loss_spot,
                "stop_loss_spot_pct": self.exit_rules.stop_loss_spot_pct,
                "stop_loss_value": self.exit_rules.stop_loss_value,
                "stop_loss_value_pct": self.exit_rules.stop_loss_value_pct,
                "take_profit_pct": self.exit_rules.take_profit_pct,
                "take_profit_value": self.exit_rules.take_profit_value,
                "max_hold_days": self.exit_rules.max_hold_days,
                "max_hold_date": self.exit_rules.max_hold_date.isoformat(),
                "scale_down_day": self.exit_rules.scale_down_day,
                "scale_down_date": self.exit_rules.scale_down_date.isoformat() if self.exit_rules.scale_down_date else None,
                "scale_down_action": self.exit_rules.scale_down_action,
                # Path-aware exit rules
                "profit_lock_threshold": self.exit_rules.profit_lock_threshold,
                "profit_lock_stop": self.exit_rules.profit_lock_stop,
                "trailing_stop_pct": self.exit_rules.trailing_stop_pct,
                "stop_tighten_day": self.exit_rules.stop_tighten_day,
                "stop_tighten_date": self.exit_rules.stop_tighten_date.isoformat() if self.exit_rules.stop_tighten_date else None,
                "tightened_stop_pct": self.exit_rules.tightened_stop_pct,
            },
            "path_profile": {
                "shakeout_pct": self.path_profile.shakeout_pct,
                "invalidation_pct": self.path_profile.invalidation_pct,
                "mae_p75": self.path_profile.mae_p75,
                "clean_win_pct": self.path_profile.clean_win_pct,
                "is_shakeout_heavy": self.path_profile.is_shakeout_heavy,
                "is_invalidation_heavy": self.path_profile.is_invalidation_heavy,
                "entry_strategy": self.path_profile.entry_strategy,
                "entry_note": self.path_profile.entry_note,
            } if self.path_profile else None,
            "validation": {
                "passed": self.validation_passed,
                "warnings": self.validation_warnings,
                "blocking": self.validation_blocking,
            },
            "policy_version": self.policy_version,
        }
    
    def to_telegram_message(self) -> str:
        """Format as Telegram message with markdown."""
        emoji = "🟢" if self.direction == "LONG" else "🔴" if self.direction == "SHORT" else "🟡"
        
        lines = [
            f"{emoji} *{self.signal_type} TRADE SETUP*",
            f"📅 {self.signal_date}",
            "",
            f"*Market:* BTC @ `${self.spot_price:,.0f}`",
            f"*Expiry:* {self.expiry} ({self.dte} DTE)",
            "",
            "━━━━━━━━━━",
            "*TRADE: Bear Put Spread*" if self.direction == "SHORT" else "*TRADE: Bull Call Spread*" if self.direction == "LONG" else "*TRADE: Iron Condor*",
            "━━━━━━━━━━",
            "",
        ]
        
        if self.direction == "NEUTRAL" and self.extra_legs:
            # Iron condor: show all 4 legs clearly
            lines.extend([
                f"*SELL CALL:* `{self.long_leg.symbol}`",
                f"  Strike: `${self.long_leg.strike:,.0f}` | Δ: `{self.long_leg.delta:.3f}` | IV: `{self.long_leg.iv*100:.1f}%`",
                f"  Bid: `${self.long_leg.price:,.2f}`",
                "",
                f"*SELL PUT:* `{self.short_leg.symbol}`",
                f"  Strike: `${self.short_leg.strike:,.0f}` | Δ: `{self.short_leg.delta:.3f}` | IV: `{self.short_leg.iv*100:.1f}%`",
                f"  Bid: `${self.short_leg.price:,.2f}`",
            ])
            # Show wing protection legs
            for leg in self.extra_legs:
                lines.extend([
                    "",
                    f"*BUY {'CALL' if leg.strike > self.long_leg.strike else 'PUT'}:* `{leg.symbol}`",
                    f"  Strike: `${leg.strike:,.0f}` | Δ: `{leg.delta:.3f}` | IV: `{leg.iv*100:.1f}%`",
                    f"  Ask: `${leg.price:,.2f}` (wing protection)",
                ])
        else:
            # Standard 2-leg spread
            lines.extend([
                f"*BUY:* `{self.long_leg.symbol}`",
                f"  Strike: `${self.long_leg.strike:,.0f}` | Δ: `{self.long_leg.delta:.3f}` | IV: `{self.long_leg.iv*100:.1f}%`",
                f"  Ask: `${self.long_leg.price:,.2f}`",
            ])
            
            if self.short_leg:
                lines.extend([
                    "",
                    f"*SELL:* `{self.short_leg.symbol}`",
                    f"  Strike: `${self.short_leg.strike:,.0f}` | Δ: `{self.short_leg.delta:.3f}` | IV: `{self.short_leg.iv*100:.1f}%`",
                    f"  Bid: `${self.short_leg.price:,.2f}`",
                ])
        
        lines.extend([
            "",
            "━━━━━━━━━━",
            "*METRICS*",
            "━━━━━━━━━━",
            f"Width: `${self.spread_width:,.0f}` ({self.spread_width_pct*100:.1f}%)",
            f"Net Debit: `${self.net_debit:,.2f}`",
            f"Max Profit: `${self.max_profit:,.2f}`",
            f"Max Loss: `${self.max_loss:,.2f}`",
            f"R:R: `1:{self.risk_reward:.2f}`",
            f"Breakeven: `${self.breakeven:,.0f}`",
            f"Net Edge: `{self.net_edge_pct*100:.1f}%`",
            "",
            "━━━━━━━━━━",
            "*POSITION*",
            "━━━━━━━━━━",
            f"Contracts: `{self.contracts}`",
            f"Total Risk: `${self.total_risk:,.2f}`",
            f"Total Max Profit: `${self.total_max_profit:,.2f}`",
            "",
            "━━━━━━━━━━",
            "*EXIT RULES*",
            "━━━━━━━━━━",
        ])
        
        if self.direction == "NEUTRAL" and self.short_leg:
            # Iron condor: symmetric risk — show value-based stop (primary) and breakevens (reference)
            lower_breakeven = self.short_leg.strike + self.net_debit  # net_debit is negative for credits
            lines.extend([
                f"🛑 Stop (Value): Close if spread value > `${self.exit_rules.stop_loss_value:,.0f}` ({self.exit_rules.stop_loss_value_pct*100:.0f}% of max loss)",
                f"📍 Upper Breakeven: `${self.exit_rules.stop_loss_spot:,.0f}` (reference)",
                f"📍 Lower Breakeven: `${lower_breakeven:,.0f}` (reference)",
            ])
        else:
            lines.extend([
                f"🛑 Stop (Spot): BTC {'>' if self.direction == 'SHORT' else '<'} `${self.exit_rules.stop_loss_spot:,.0f}` ({self.exit_rules.stop_loss_spot_pct*100:.1f}%)",
                f"🛑 Stop (Value): Spread {'>' if self.net_debit < 0 else '<'} `${self.exit_rules.stop_loss_value:,.0f}` ({self.exit_rules.stop_loss_value_pct*100:.0f}% {'of max loss' if self.net_debit < 0 else 'lost'})",
            ])
        
        lines.extend([
            f"✅ Take Profit: `{self.exit_rules.take_profit_pct*100:.0f}%` = `${self.exit_rules.take_profit_value:,.0f}`/contract",
            f"⏰ Max Hold: {self.exit_rules.max_hold_days}d (until {self.exit_rules.max_hold_date})",
        ])
        
        if self.exit_rules.scale_down_day:
            action = "CLOSE FULL" if self.exit_rules.scale_down_action == "close_full_position" else "Reduce 50%"
            lines.append(f"📉 Scale Down: Day {self.exit_rules.scale_down_day} → {action}")
        
        # Path-aware exit rules (only show if configured)
        path_aware_exits = []
        if self.exit_rules.profit_lock_threshold and self.exit_rules.profit_lock_threshold > 0:
            path_aware_exits.append(f"🔒 Profit Lock: Move stop to breakeven after `{self.exit_rules.profit_lock_threshold*100:.0f}%` profit")
        if self.exit_rules.trailing_stop_pct and self.exit_rules.trailing_stop_pct > 0:
            path_aware_exits.append(f"📈 Trailing Stop: `{self.exit_rules.trailing_stop_pct*100:.0f}%` below high water mark")
        if self.exit_rules.stop_tighten_day and self.exit_rules.tightened_stop_pct:
            path_aware_exits.append(f"⚡ Stop Tighten: Day {self.exit_rules.stop_tighten_day} → `{self.exit_rules.tightened_stop_pct*100:.1f}%`")
        
        if path_aware_exits:
            lines.append("")
            lines.append("*Dynamic Exits:*")
            lines.extend(path_aware_exits)
        
        # Validation status
        if not self.validation_passed:
            lines.extend([
                "",
                "⚠️ *VALIDATION FAILED*",
            ])
            for msg in self.validation_blocking:
                lines.append(f"  🛑 {msg[:60]}...")
        elif self.validation_warnings:
            lines.extend([
                "",
                "⚠️ *WARNINGS*",
            ])
            for msg in self.validation_warnings[:3]:  # Limit to 3
                lines.append(f"  ⚠️ {msg[:60]}...")
        
        # Path profile warning for shakeout-heavy signals
        if self.path_profile and (self.path_profile.is_shakeout_heavy or self.path_profile.is_invalidation_heavy):
            lines.extend([
                "",
                "━━━━━━━━━━",
                "⚡ *PATH PROFILE*",
                "━━━━━━━━━━",
            ])
            if self.path_profile.is_shakeout_heavy:
                lines.append(f"🔄 Shakeout Rate: `{self.path_profile.shakeout_pct*100:.0f}%` of winners")
            if self.path_profile.is_invalidation_heavy:
                lines.append(f"⚠️ Invalidation: `{self.path_profile.invalidation_pct*100:.0f}%` before hit")
            lines.append(f"📊 MAE p75: `{self.path_profile.mae_p75*100:.1f}%`")
            lines.append(f"💡 Entry: *{self.path_profile.entry_strategy.upper()}*")
            if self.path_profile.entry_note:
                lines.append(f"   {self.path_profile.entry_note}")
        
        lines.extend([
            "",
            "━━━━━━━━━━",
            "*ORDER ENTRY*",
            "━━━━━━━━━━",
        ])
        
        if self.direction == "NEUTRAL" and self.extra_legs:
            # Iron condor: 4-leg credit structure
            lines.extend([
                f"1\\. Sell: `{self.long_leg.symbol}` × {self.contracts} (short call)",
                f"2\\. Sell: `{self.short_leg.symbol}` × {self.contracts} (short put)",
            ])
            for i, leg in enumerate(self.extra_legs, start=3):
                leg_label = "long call wing" if leg.strike > self.long_leg.strike else "long put wing"
                lines.append(f"{i}\\. Buy: `{leg.symbol}` × {self.contracts} ({leg_label})")
            lines.append(f"{len(self.extra_legs) + 3}\\. Limit: `${abs(self.net_debit):,.2f}` net credit")
        else:
            # Standard 2-leg debit spread
            lines.append(f"1\\. Buy: `{self.long_leg.symbol}` × {self.contracts}")
            if self.short_leg:
                lines.append(f"2\\. Sell: `{self.short_leg.symbol}` × {self.contracts}")
                lines.append(f"3\\. Limit: `${self.net_debit:,.2f}` debit")
        
        return "\n".join(lines)
    
    def save_to_db(self, signal=None):
        """
        Persist this setup to the database.
        
        Args:
            signal: DailySignal instance (optional, will lookup if not provided)
        
        Returns:
            TradeSetupSnapshot instance
        """
        from execution.models import TradeSetupSnapshot
        from signals.models import DailySignal
        
        if signal is None:
            signal = DailySignal.objects.get(
                date=self.signal_date,
                trade_decision=self.signal_type
            )
        
        snapshot, created = TradeSetupSnapshot.objects.update_or_create(
            signal=signal,
            defaults={
                'setup_data': self.to_dict(),
                'signal_date': self.signal_date,
                'signal_type': self.signal_type,
                'direction': self.direction,
                'spot_price': self.spot_price,
                'net_debit': self.net_debit,
                'max_profit': self.max_profit,
                'max_loss': self.max_loss,
                'contracts': self.contracts,
                'validation_passed': self.validation_passed,
            }
        )
        return snapshot


class TradeSetupBuilder:
    """
    Builds complete trade setups from signals and option data.
    
    Integrates:
    - Signal data (direction, type)
    - Policy parameters (DTE, delta, spread width, exits)
    - Option snapshots (prices, greeks, liquidity)
    - Validation (risk checks, liquidity stress)
    """
    
    def __init__(self, policy: Optional[PolicyVersion] = None):
        self.policy = policy or get_policy()
        self.validator = TradeValidator(policy=self.policy)
    
    def build_setup(
        self,
        signal_date: date,
        signal_type: Optional[str] = None,
    ) -> Optional[TradeSetup]:
        """
        Build a complete trade setup for a signal date.
        
        Args:
            signal_date: Date of the signal
            signal_type: Override signal type (uses DB signal if None)
        
        Returns:
            TradeSetup or None if no valid setup found
        """
        # Get signal from DB if not overridden
        if signal_type is None:
            candidates = DailySignal.tradeable().filter(date=signal_date)
            signal = DailySignal.pick_highest_priority(candidates)
            if signal:
                signal_type = signal.trade_decision
            else:
                return None
        
        # Skip NO_TRADE
        if signal_type == "NO_TRADE":
            return None
        
        # Get spot price
        try:
            raw = RawDailyData.objects.get(date=signal_date)
            spot_price = float(raw.btc_close)
        except RawDailyData.DoesNotExist:
            return None
        
        # Determine direction and option type
        direction, option_type = self._get_direction_and_type(signal_type)
        if direction is None:
            return None
        
        # Handle iron condor separately (4-leg structure)
        if option_type == "condor":
            return self._build_condor_setup(signal_date, signal_type, spot_price)
        
        # Get policy parameters
        dte_cfg = self.policy.get_dte_target(signal_type)
        exit_cfg = self.policy.get_exit_params(signal_type)
        delta_target = self.policy.get_signal_delta(signal_type)
        spread_width_pct = self.policy.get_spread_width(signal_type)
        tier = self.policy.get_tier(signal_type)
        risk_budget = tier.risk_usd * tier.spread_pct
        
        # Get option snapshots
        latest_ts = OptionSnapshot.objects.filter(
            timestamp__date=signal_date,
            option_type=option_type,
            dte__gte=dte_cfg.min_dte,
            dte__lte=dte_cfg.max_dte,
        ).aggregate(Max('timestamp'))['timestamp__max']
        
        if not latest_ts:
            return None
        
        options = list(OptionSnapshot.objects.filter(
            timestamp=latest_ts,
            option_type=option_type,
            dte__gte=dte_cfg.min_dte,
            dte__lte=dte_cfg.max_dte,
        ))
        
        if not options:
            return None
        
        # Find long leg (closest to target delta)
        long_opt = min(options, key=lambda x: abs(float(x.delta or 0) - delta_target))
        
        # Find short leg (budget-constrained width)
        short_opt = self._find_best_short_leg(
            options, long_opt, option_type, direction, 
            spot_price, spread_width_pct, risk_budget
        )
        
        if not short_opt:
            return None
        
        # Calculate spread metrics
        width = abs(float(long_opt.strike) - float(short_opt.strike))
        net_debit = float(long_opt.ask) - float(short_opt.bid)
        max_profit = width - net_debit
        max_loss = net_debit
        
        if max_loss <= 0 or max_profit <= 0:
            return None
        
        risk_reward = max_profit / max_loss
        breakeven = float(long_opt.strike) - net_debit if option_type == "put" else float(long_opt.strike) + net_debit
        
        # Position sizing
        contracts = int(risk_budget / max_loss) if max_loss > 0 else 0
        if contracts < 1:
            contracts = 1  # Minimum 1 contract
        
        total_risk = contracts * max_loss
        total_max_profit = contracts * max_profit
        
        # Validate
        validation_result = self.validator.validate(SpreadPlan(
            signal_type=signal_type,
            direction=direction.lower(),
            option_type=option_type,
            long_strike=float(long_opt.strike),
            short_strike=float(short_opt.strike),
            expiry_dte=int(long_opt.dte),
            net_debit=net_debit,
            max_profit=max_profit,
            max_loss=max_loss,
            contracts=contracts,
            spot_price=spot_price,
            long_delta=float(long_opt.delta) if long_opt.delta else None,
            short_delta=float(short_opt.delta) if short_opt.delta else None,
            long_iv=float(long_opt.iv) if long_opt.iv else None,
            short_iv=float(short_opt.iv) if short_opt.iv else None,
            long_bid_ask_spread_pct=float(long_opt.spread_pct) if long_opt.spread_pct else None,
            short_bid_ask_spread_pct=float(short_opt.spread_pct) if short_opt.spread_pct else None,
            long_open_interest=int(long_opt.open_interest) if long_opt.open_interest else None,
            short_open_interest=int(short_opt.open_interest) if short_opt.open_interest else None,
        ))
        
        # Extract adjusted params
        adj = validation_result.adjusted_params
        execution_cost = adj.get('estimated_execution_cost', 0)
        adjusted_max_profit = adj.get('adjusted_max_profit', max_profit)
        net_edge_pct = adj.get('net_edge_pct', 0)
        
        # Build exit rules
        if direction == "SHORT":
            stop_spot = spot_price * (1 + exit_cfg.stop_loss_pct)
        else:
            stop_spot = spot_price * (1 - exit_cfg.stop_loss_pct)
        
        stop_value_pct = adj.get('option_value_stop_pct', 0.6)
        stop_value = net_debit * (1 - stop_value_pct)
        
        scale_action = adj.get('scale_down_action', 'reduce_50pct')
        
        exit_rules = ExitRules(
            stop_loss_spot=stop_spot,
            stop_loss_spot_pct=exit_cfg.stop_loss_pct,
            stop_loss_value=stop_value,
            stop_loss_value_pct=stop_value_pct,
            take_profit_pct=exit_cfg.take_profit_pct,
            take_profit_value=max_profit * exit_cfg.take_profit_pct,
            max_hold_days=exit_cfg.max_hold_days,
            max_hold_date=signal_date + timedelta(days=exit_cfg.max_hold_days),
            scale_down_day=exit_cfg.scale_down_day,
            scale_down_date=signal_date + timedelta(days=exit_cfg.scale_down_day) if exit_cfg.scale_down_day else None,
            scale_down_action=scale_action,
            # Path-aware exit parameters
            profit_lock_threshold=exit_cfg.profit_lock_threshold,
            profit_lock_stop=spot_price,  # Breakeven = entry spot price
            trailing_stop_pct=exit_cfg.trailing_stop_pct,
            stop_tighten_day=exit_cfg.stop_tighten_day,
            stop_tighten_date=signal_date + timedelta(days=exit_cfg.stop_tighten_day) if exit_cfg.stop_tighten_day else None,
            tightened_stop_pct=exit_cfg.stop_loss_pct * exit_cfg.stop_tighten_factor if exit_cfg.stop_tighten_day else None,
        )
        
        # Build leg setups
        long_leg = LegSetup(
            symbol=long_opt.symbol,
            action="BUY",
            strike=float(long_opt.strike),
            delta=float(long_opt.delta) if long_opt.delta else 0,
            iv=float(long_opt.iv) if long_opt.iv else 0,
            price=float(long_opt.ask),
            open_interest=int(long_opt.open_interest) if long_opt.open_interest else 0,
            bid_ask_spread_pct=float(long_opt.spread_pct) if long_opt.spread_pct else 0,
        )
        
        short_leg = LegSetup(
            symbol=short_opt.symbol,
            action="SELL",
            strike=float(short_opt.strike),
            delta=float(short_opt.delta) if short_opt.delta else 0,
            iv=float(short_opt.iv) if short_opt.iv else 0,
            price=float(short_opt.bid),
            open_interest=int(short_opt.open_interest) if short_opt.open_interest else 0,
            bid_ask_spread_pct=float(short_opt.spread_pct) if short_opt.spread_pct else 0,
        )
        
        # Extract validation messages
        warnings = [i.message for i in validation_result.warnings]
        blocking = [i.message for i in validation_result.blocking_issues]
        
        expiry_date = long_opt.expiry.date() if hasattr(long_opt.expiry, 'date') else long_opt.expiry
        
        # Build path profile
        path_data = self.policy.get_path_profile(signal_type)
        is_shakeout = self.policy.is_shakeout_heavy(signal_type)
        is_invalidation = self.policy.is_invalidation_heavy(signal_type)
        
        # Determine entry strategy based on path profile
        if is_shakeout:
            entry_strategy = "dca"
            # Use stop loss as DCA trigger (price moves against by stop_loss_pct)
            entry_note = f"33% initial, 67% DCA at +{exit_cfg.stop_loss_pct*100:.0f}% (shakeout-heavy)"
        elif is_invalidation:
            entry_strategy = "scaled"
            entry_note = "50% initial, 50% on confirmation (high invalidation)"
        else:
            entry_strategy = "single"
            entry_note = "Full position at entry"
        
        path_profile = PathProfile(
            shakeout_pct=path_data.get("shakeout_pct", 0),
            invalidation_pct=path_data.get("invalidation_pct", 0),
            mae_p75=path_data.get("mae_p75", 0),
            clean_win_pct=path_data.get("clean_win_pct", 1.0),
            is_shakeout_heavy=is_shakeout,
            is_invalidation_heavy=is_invalidation,
            entry_strategy=entry_strategy,
            entry_note=entry_note,
        )
        
        return TradeSetup(
            signal_date=signal_date,
            signal_type=signal_type,
            direction=direction,
            spot_price=spot_price,
            expiry=expiry_date,
            dte=int(long_opt.dte),
            long_leg=long_leg,
            short_leg=short_leg,
            spread_width=width,
            spread_width_pct=width / spot_price,
            net_debit=net_debit,
            max_profit=max_profit,
            max_loss=max_loss,
            risk_reward=risk_reward,
            breakeven=breakeven,
            execution_cost=execution_cost,
            adjusted_max_profit=adjusted_max_profit,
            net_edge_pct=net_edge_pct,
            risk_budget=risk_budget,
            contracts=contracts,
            total_risk=total_risk,
            total_max_profit=total_max_profit,
            exit_rules=exit_rules,
            path_profile=path_profile,
            validation_passed=validation_result.is_valid,
            validation_warnings=warnings,
            validation_blocking=blocking,
            policy_version=self.policy.version,
        )
    
    def _get_direction_and_type(self, signal_type: str) -> tuple[Optional[str], Optional[str]]:
        """Map signal type to direction and option type."""
        mapping = {
            "CALL": ("LONG", "call"),
            "OPTION_CALL": ("LONG", "call"),
            "BULL_PROBE": ("LONG", "call"),
            "PUT": ("SHORT", "put"),
            "OPTION_PUT": ("SHORT", "put"),
            "TACTICAL_PUT": ("SHORT", "put"),
            "BEAR_PROBE": ("SHORT", "put"),
            "MVRV_SHORT": ("SHORT", "put"),
            "IRON_CONDOR": ("NEUTRAL", "condor"),  # Special handling in build_setup
        }
        return mapping.get(signal_type, (None, None))
    
    def _find_best_short_leg(
        self,
        options: list[OptionSnapshot],
        long_opt: OptionSnapshot,
        option_type: str,
        direction: str,
        spot_price: float,
        policy_width_pct: float,
        risk_budget: float,
    ) -> Optional[OptionSnapshot]:
        """
        Find the best short leg balancing R:R, width, budget, and liquidity.
        
        Scoring weights:
        - R:R score (40%): Prefer R:R >= 1:1, penalize < 0.8
        - Width score (30%): Prefer closer to policy target width
        - Budget fit (20%): Prefer spreads that fit budget with 1+ contracts
        - Liquidity (10%): Prefer higher OI on short leg
        
        Minimum requirements:
        - R:R >= 0.5 (hard floor)
        - Fits budget (at least 1 contract)
        """
        policy_width_usd = spot_price * policy_width_pct
        candidates = []
        
        for opt in options:
            if opt.expiry != long_opt.expiry:
                continue
            
            # For puts: short strike < long strike
            # For calls: short strike > long strike
            if option_type == "put":
                if float(opt.strike) >= float(long_opt.strike):
                    continue
            else:
                if float(opt.strike) <= float(long_opt.strike):
                    continue
            
            width = abs(float(long_opt.strike) - float(opt.strike))
            net_debit = float(long_opt.ask) - float(opt.bid)
            
            if net_debit <= 0:
                continue
            
            max_profit = width - net_debit
            max_loss = net_debit
            
            if max_profit <= 0:
                continue
            
            rr = max_profit / max_loss
            
            # Hard floor: R:R must be at least 0.5
            if rr < 0.5:
                continue
            
            contracts = int(risk_budget / max_loss)
            oi = int(opt.open_interest) if opt.open_interest else 0
            
            # === SCORING ===
            
            # R:R score (0-1): optimal at 1.0-1.5, penalize below 0.8
            if rr >= 1.0:
                rr_score = min(1.0, 0.7 + (rr - 1.0) * 0.3)  # 0.7-1.0 for R:R 1.0-2.0
            elif rr >= 0.8:
                rr_score = 0.5 + (rr - 0.8) * 1.0  # 0.5-0.7 for R:R 0.8-1.0
            else:
                rr_score = rr * 0.625  # 0-0.5 for R:R 0-0.8
            
            # Width score (0-1): prefer closer to policy target
            # Allow 50% to 150% of target as acceptable range
            width_ratio = width / policy_width_usd if policy_width_usd > 0 else 0
            if 0.5 <= width_ratio <= 1.5:
                # Within acceptable range - score based on proximity to 1.0
                width_score = 1.0 - abs(width_ratio - 1.0) * 0.5
            elif width_ratio < 0.5:
                # Too narrow - penalize more
                width_score = width_ratio * 0.6  # 0-0.3 for 0-50% of target
            else:
                # Too wide - slight penalty
                width_score = max(0.3, 1.0 - (width_ratio - 1.5) * 0.2)
            
            # Budget fit score (0-1): prefer 1-3 contracts
            if contracts >= 3:
                budget_score = 1.0
            elif contracts >= 1:
                budget_score = 0.7 + contracts * 0.1  # 0.8-1.0 for 1-3 contracts
            else:
                budget_score = 0.3  # Doesn't fit budget
            
            # Liquidity score (0-1): prefer OI >= 50
            if oi >= 100:
                liq_score = 1.0
            elif oi >= 50:
                liq_score = 0.8 + (oi - 50) * 0.004  # 0.8-1.0 for 50-100
            elif oi >= 10:
                liq_score = 0.4 + (oi - 10) * 0.01  # 0.4-0.8 for 10-50
            else:
                liq_score = oi * 0.04  # 0-0.4 for 0-10
            
            # Weighted total score
            total_score = (
                rr_score * 0.40 +
                width_score * 0.30 +
                budget_score * 0.20 +
                liq_score * 0.10
            )
            
            candidates.append({
                'opt': opt,
                'score': total_score,
                'rr': rr,
                'rr_score': rr_score,
                'width': width,
                'width_score': width_score,
                'contracts': contracts,
                'budget_score': budget_score,
                'oi': oi,
                'liq_score': liq_score,
                'net_debit': net_debit,
                'max_profit': max_profit,
            })
        
        if not candidates:
            return None
        
        # Sort by total score descending
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Prefer candidates that fit budget (contracts >= 1)
        budget_fit = [c for c in candidates if c['contracts'] >= 1]
        if budget_fit:
            return budget_fit[0]['opt']
        
        # Fallback to best overall score
        return candidates[0]['opt']
    
    def _build_condor_setup(
        self,
        signal_date: date,
        signal_type: str,
        spot_price: float,
    ) -> Optional[TradeSetup]:
        """
        Build iron condor setup (4-leg structure).
        
        Iron condor structure:
        - SELL short call (OTM)
        - BUY long call (further OTM, wing protection)
        - SELL short put (OTM)
        - BUY long put (further OTM, wing protection)
        
        Net credit received = premium from selling - premium for buying wings
        Max profit = net credit (if price stays between short strikes)
        Max loss = wing width - net credit (if price moves beyond wings)
        """
        # Get policy parameters
        dte_cfg = self.policy.get_dte_target(signal_type)
        exit_cfg = self.policy.get_exit_params(signal_type)
        condor_cfg = self.policy.condor
        tier = self.policy.get_tier(signal_type)
        risk_budget = tier.risk_usd * tier.spread_pct
        
        # Get signal for MVRV-based strike targets (if available)
        try:
            signal = DailySignal.active().get(date=signal_date, trade_decision="IRON_CONDOR")
            target_short_call = signal.condor_short_call or spot_price * (1 + condor_cfg.spot_call_band)
            target_short_put = signal.condor_short_put or spot_price * (1 - condor_cfg.spot_put_band)
        except DailySignal.DoesNotExist:
            target_short_call = spot_price * (1 + condor_cfg.spot_call_band)
            target_short_put = spot_price * (1 - condor_cfg.spot_put_band)
        
        # Get latest timestamp for options on signal date
        latest_ts = OptionSnapshot.objects.filter(
            timestamp__date=signal_date,
            dte__gte=dte_cfg.min_dte,
            dte__lte=dte_cfg.max_dte,
        ).aggregate(Max('timestamp'))['timestamp__max']
        
        if not latest_ts:
            return None
        
        # Get call options
        call_options = list(OptionSnapshot.objects.filter(
            timestamp=latest_ts,
            option_type="call",
            dte__gte=dte_cfg.min_dte,
            dte__lte=dte_cfg.max_dte,
        ))
        
        # Get put options
        put_options = list(OptionSnapshot.objects.filter(
            timestamp=latest_ts,
            option_type="put",
            dte__gte=dte_cfg.min_dte,
            dte__lte=dte_cfg.max_dte,
        ))
        
        if not call_options or not put_options:
            return None
        
        # Find short call (closest to target, OTM)
        short_call = min(
            [o for o in call_options if float(o.strike) >= spot_price],
            key=lambda x: abs(float(x.strike) - target_short_call),
            default=None
        )
        if not short_call:
            return None
        
        # Find long call (wing, same expiry, higher strike)
        long_call_target = float(short_call.strike) + condor_cfg.wing_offset_usd
        long_call = min(
            [o for o in call_options 
             if o.expiry == short_call.expiry and float(o.strike) > float(short_call.strike)],
            key=lambda x: abs(float(x.strike) - long_call_target),
            default=None
        )
        if not long_call:
            return None
        
        # Find short put (closest to target, OTM)
        short_put = min(
            [o for o in put_options 
             if float(o.strike) <= spot_price and o.expiry == short_call.expiry],
            key=lambda x: abs(float(x.strike) - target_short_put),
            default=None
        )
        if not short_put:
            return None
        
        # Find long put (wing, same expiry, lower strike)
        long_put_target = float(short_put.strike) - condor_cfg.wing_offset_usd
        long_put = min(
            [o for o in put_options 
             if o.expiry == short_call.expiry and float(o.strike) < float(short_put.strike)],
            key=lambda x: abs(float(x.strike) - long_put_target),
            default=None
        )
        if not long_put:
            return None
        
        # Calculate condor metrics
        # Net credit = (short call bid + short put bid) - (long call ask + long put ask)
        net_credit = (
            float(short_call.bid) + float(short_put.bid) -
            float(long_call.ask) - float(long_put.ask)
        )
        
        # Wing widths
        call_wing_width = float(long_call.strike) - float(short_call.strike)
        put_wing_width = float(short_put.strike) - float(long_put.strike)
        wing_width = max(call_wing_width, put_wing_width)  # Use wider wing for max loss calc
        
        # Max profit = net credit (if price stays between short strikes)
        max_profit = net_credit
        
        # Max loss = wing width - net credit (if price moves beyond wings)
        max_loss = wing_width - net_credit
        
        if max_loss <= 0 or max_profit <= 0:
            return None
        
        risk_reward = max_profit / max_loss
        
        # Breakevens
        upper_breakeven = float(short_call.strike) + net_credit
        lower_breakeven = float(short_put.strike) - net_credit
        
        # Position sizing
        contracts = int(risk_budget / max_loss) if max_loss > 0 else 0
        if contracts < 1:
            contracts = 1
        
        total_risk = contracts * max_loss
        total_max_profit = contracts * max_profit
        
        # Build leg setups (use short call as primary "long" leg for display)
        # Note: For condors, we show all 4 legs — short legs as primary, wings in extra_legs
        long_leg = LegSetup(
            symbol=short_call.symbol,
            action="SELL",  # Short call
            strike=float(short_call.strike),
            delta=float(short_call.delta) if short_call.delta else 0,
            iv=float(short_call.iv) if short_call.iv else 0,
            price=float(short_call.bid),
            open_interest=int(short_call.open_interest) if short_call.open_interest else 0,
            bid_ask_spread_pct=float(short_call.spread_pct) if short_call.spread_pct else 0,
        )
        
        short_leg = LegSetup(
            symbol=short_put.symbol,
            action="SELL",  # Short put
            strike=float(short_put.strike),
            delta=float(short_put.delta) if short_put.delta else 0,
            iv=float(short_put.iv) if short_put.iv else 0,
            price=float(short_put.bid),
            open_interest=int(short_put.open_interest) if short_put.open_interest else 0,
            bid_ask_spread_pct=float(short_put.spread_pct) if short_put.spread_pct else 0,
        )
        
        # Wing legs (protective long options)
        extra_legs = [
            LegSetup(
                symbol=long_call.symbol,
                action="BUY",  # Long call wing
                strike=float(long_call.strike),
                delta=float(long_call.delta) if long_call.delta else 0,
                iv=float(long_call.iv) if long_call.iv else 0,
                price=float(long_call.ask),
                open_interest=int(long_call.open_interest) if long_call.open_interest else 0,
                bid_ask_spread_pct=float(long_call.spread_pct) if long_call.spread_pct else 0,
            ),
            LegSetup(
                symbol=long_put.symbol,
                action="BUY",  # Long put wing
                strike=float(long_put.strike),
                delta=float(long_put.delta) if long_put.delta else 0,
                iv=float(long_put.iv) if long_put.iv else 0,
                price=float(long_put.ask),
                open_interest=int(long_put.open_interest) if long_put.open_interest else 0,
                bid_ask_spread_pct=float(long_put.spread_pct) if long_put.spread_pct else 0,
            ),
        ]
        
        # Exit rules for condor
        # For credit structures, stop out when spread value rises (loss grows)
        # Stop triggers when unrealized loss exceeds stop_loss_pct of max_loss
        stop_loss_value = net_credit + (max_loss * exit_cfg.stop_loss_pct)
        exit_rules = ExitRules(
            stop_loss_spot=upper_breakeven,  # Upper breakeven as reference
            stop_loss_spot_pct=exit_cfg.stop_loss_pct,
            stop_loss_value=stop_loss_value,
            stop_loss_value_pct=exit_cfg.stop_loss_pct,
            take_profit_pct=exit_cfg.take_profit_pct,
            take_profit_value=max_profit * exit_cfg.take_profit_pct,
            max_hold_days=exit_cfg.max_hold_days,
            max_hold_date=signal_date + timedelta(days=exit_cfg.max_hold_days),
            scale_down_day=exit_cfg.scale_down_day,
            scale_down_date=signal_date + timedelta(days=exit_cfg.scale_down_day) if exit_cfg.scale_down_day else None,
            scale_down_action="close_full_position",  # Condors typically close full
            profit_lock_threshold=exit_cfg.profit_lock_threshold,
            profit_lock_stop=spot_price,
            trailing_stop_pct=exit_cfg.trailing_stop_pct,
            stop_tighten_day=exit_cfg.stop_tighten_day,
            stop_tighten_date=signal_date + timedelta(days=exit_cfg.stop_tighten_day) if exit_cfg.stop_tighten_day else None,
            tightened_stop_pct=exit_cfg.stop_loss_pct * exit_cfg.stop_tighten_factor if exit_cfg.stop_tighten_day else None,
        )
        
        # Path profile for condor
        path_data = self.policy.get_path_profile(signal_type)
        path_profile = PathProfile(
            shakeout_pct=path_data.get("shakeout_pct", 0),
            invalidation_pct=path_data.get("invalidation_pct", 0),
            mae_p75=path_data.get("mae_p75", 0),
            clean_win_pct=path_data.get("clean_win_pct", 1.0),
            is_shakeout_heavy=False,  # Condors don't have shakeout patterns
            is_invalidation_heavy=False,
            entry_strategy="single",
            entry_note="Full position at entry (premium selling)",
        )
        
        # Validation warnings
        warnings = []
        blocking = []
        
        # Check credit percentage
        credit_pct = net_credit / wing_width if wing_width > 0 else 0
        if credit_pct < condor_cfg.min_credit_pct:
            warnings.append(f"Credit {credit_pct:.1%} below minimum {condor_cfg.min_credit_pct:.1%}")
        
        # Check wing distances
        call_dist_pct = (float(short_call.strike) - spot_price) / spot_price
        put_dist_pct = (spot_price - float(short_put.strike)) / spot_price
        if call_dist_pct < 0.08:
            warnings.append(f"Short call only {call_dist_pct:.1%} OTM (prefer >8%)")
        if put_dist_pct < 0.08:
            warnings.append(f"Short put only {put_dist_pct:.1%} OTM (prefer >8%)")
        
        expiry_date = short_call.expiry.date() if hasattr(short_call.expiry, 'date') else short_call.expiry
        
        return TradeSetup(
            signal_date=signal_date,
            signal_type=signal_type,
            direction="NEUTRAL",
            spot_price=spot_price,
            expiry=expiry_date,
            dte=int(short_call.dte),
            long_leg=long_leg,  # Short call (primary display)
            short_leg=short_leg,  # Short put
            extra_legs=extra_legs,  # Wing protection legs
            spread_width=wing_width,
            spread_width_pct=wing_width / spot_price,
            net_debit=-net_credit,  # Negative debit = credit received
            max_profit=max_profit,
            max_loss=max_loss,
            risk_reward=risk_reward,
            breakeven=upper_breakeven,  # Show upper breakeven
            execution_cost=0,  # TODO: Calculate from policy
            adjusted_max_profit=max_profit,
            net_edge_pct=credit_pct,
            risk_budget=risk_budget,
            contracts=contracts,
            total_risk=total_risk,
            total_max_profit=total_max_profit,
            exit_rules=exit_rules,
            path_profile=path_profile,
            validation_passed=len(blocking) == 0,
            validation_warnings=warnings,
            validation_blocking=blocking,
            policy_version=self.policy.version,
        )
