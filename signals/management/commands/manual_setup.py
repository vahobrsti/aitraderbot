# signals/management/commands/manual_setup.py
"""
Manual Signal Override - Generate trade setup for any signal type.

Bypasses the fusion engine and generates a complete trade setup for
any specified signal type using REAL option chain data from OptionSnapshot.

Usage:
    # List all available signals
    python manage.py manual_setup --list
    
    # Generate setup for a specific signal (uses real option chain)
    python manage.py manual_setup --signal CALL
    python manage.py manual_setup --signal IRON_CONDOR
    python manage.py manual_setup --signal MVRV_SHORT
    
    # Generate all setups
    python manage.py manual_setup --all
    
    # Use specific date
    python manage.py manual_setup --signal PUT --date 2026-05-09
    
    # Output as JSON
    python manage.py manual_setup --signal CALL --json
    
    # Show theoretical setup only (no option chain lookup)
    python manage.py manual_setup --signal CALL --theoretical
"""
from datetime import datetime, timedelta
from decimal import Decimal
from django.core.management.base import BaseCommand
from django.db.models import Max

from execution.services.policy import get_policy
from execution.services.trade_setup import TradeSetupBuilder
from datafeed.models import RawDailyData, OptionSnapshot
from signals.options import (
    STRATEGY_MAP, DECISION_STRATEGY_MAP, compute_condor_strikes,
    CONDOR_TRAILING_DAYS
)
from signals.fusion import MarketState
from signals.income_gate import (
    evaluate_bull_put_gate, evaluate_bear_call_gate, IncomeGateConfig,
    compute_bull_put_score, compute_bear_call_score,
    compute_bear_call_target_mvrv, compute_bull_put_target_mvrv,
    dedupe_chain_to_latest,
)


# All available signal types
SIGNAL_TYPES = [
    # Core fusion signals
    "CALL",
    "PUT",
    "BULL_PROBE",
    "BEAR_PROBE",
    # Fallback/tactical signals
    "OPTION_CALL",
    "OPTION_PUT",
    "TACTICAL_PUT",
    "MVRV_SHORT",
    "IRON_CONDOR",
    # Income spread signals
    "BULL_PUT_SPREAD",
    "BEAR_CALL_SPREAD",
]

# Signal to MarketState mapping (for fusion-based signals)
SIGNAL_TO_STATE = {
    "CALL": [
        MarketState.STRONG_BULLISH,
        MarketState.EARLY_RECOVERY,
        MarketState.MOMENTUM_CONTINUATION,
        MarketState.BEAR_EXHAUSTION_LONG,
        MarketState.BEAR_RALLY_LONG,
    ],
    "PUT": [
        MarketState.DISTRIBUTION_RISK,
        MarketState.BEAR_CONTINUATION,
        MarketState.BEAR_CONTINUATION_SHORT,
        MarketState.LATE_DISTRIBUTION_SHORT,
    ],
    "BULL_PROBE": [MarketState.BULL_PROBE],
    "BEAR_PROBE": [MarketState.BEAR_PROBE],
}


class Command(BaseCommand):
    help = "Generate trade setup for any signal type (manual override) using real option chain data"

    def add_arguments(self, parser):
        parser.add_argument(
            "--signal",
            type=str,
            choices=SIGNAL_TYPES,
            help="Signal type to generate setup for",
        )
        parser.add_argument(
            "--list",
            action="store_true",
            help="List all available signal types with descriptions",
        )
        parser.add_argument(
            "--all",
            action="store_true",
            help="Generate setups for all signal types",
        )
        parser.add_argument(
            "--date",
            type=str,
            default=None,
            help="Target date (YYYY-MM-DD) - defaults to latest available",
        )
        parser.add_argument(
            "--json",
            action="store_true",
            help="Output as JSON",
        )
        parser.add_argument(
            "--spot",
            type=float,
            default=None,
            help="Override spot price (defaults to DB close price)",
        )
        parser.add_argument(
            "--theoretical",
            action="store_true",
            help="Show theoretical setup only (no option chain lookup)",
        )
        parser.add_argument(
            "--show-chain",
            action="store_true",
            dest="show_chain",
            help="Show full option chain candidates",
        )

    def handle(self, *args, **options):
        import json
        
        if options["list"]:
            self._list_signals()
            return
        
        # Get spot price and date
        if options["date"]:
            target_date = datetime.strptime(options["date"], "%Y-%m-%d").date()
        else:
            target_date = None
        
        spot, actual_date = self._get_spot_price(target_date, options.get("spot"))
        if spot is None:
            self.stderr.write(self.style.ERROR("No price data available"))
            return
        
        policy = get_policy()
        
        if options["all"]:
            signals_to_process = SIGNAL_TYPES
        elif options["signal"]:
            signals_to_process = [options["signal"].upper()]
        else:
            self.stderr.write(self.style.ERROR("Specify --signal, --all, or --list"))
            return
        
        results = []
        
        for signal_type in signals_to_process:
            if options.get("theoretical"):
                # Theoretical setup only
                setup = self._generate_theoretical_setup(signal_type, spot, actual_date, policy)
                results.append(setup)
                if not options["json"]:
                    self._print_theoretical_setup(setup)
            else:
                # Real option chain setup
                setup = self._generate_real_setup(
                    signal_type, spot, actual_date, policy, 
                    show_chain=options.get("show_chain", False)
                )
                if setup:
                    results.append(setup)
                    if not options["json"] and signal_type not in ("BULL_PUT_SPREAD", "BEAR_CALL_SPREAD"):
                        self._print_real_setup(setup, show_chain=options.get("show_chain", False))
                else:
                    # Fall back to theoretical if no option data
                    self.stdout.write(self.style.WARNING(
                        f"\n⚠️  No option chain data for {signal_type} - showing theoretical setup"
                    ))
                    setup = self._generate_theoretical_setup(signal_type, spot, actual_date, policy)
                    results.append(setup)
                    if not options["json"]:
                        self._print_theoretical_setup(setup)
        
        if options["json"]:
            self.stdout.write(json.dumps(results, indent=2, default=str))

    def _list_signals(self):
        """List all available signal types with descriptions."""
        policy = get_policy()
        
        self.stdout.write("\n" + "="*60)
        self.stdout.write("AVAILABLE SIGNAL TYPES")
        self.stdout.write("="*60)
        
        self.stdout.write("\n📈 LONG SIGNALS:")
        self.stdout.write("-"*60)
        for sig in ["CALL", "BULL_PROBE", "OPTION_CALL"]:
            exit_cfg = policy.get_exit_params(sig)
            spread_width = policy.get_spread_width(sig)
            dte_cfg = policy.get_dte_target(sig)
            if exit_cfg:
                rr = (spread_width * exit_cfg.take_profit_pct) / exit_cfg.stop_loss_pct
                self.stdout.write(
                    f"  {sig:<15} | DTE: {dte_cfg.min_dte}-{dte_cfg.max_dte}d | "
                    f"Stop: {exit_cfg.stop_loss_pct*100:.1f}% | R:R: 1:{rr:.2f}"
                )
        
        self.stdout.write("\n📉 SHORT SIGNALS:")
        self.stdout.write("-"*60)
        for sig in ["PUT", "BEAR_PROBE", "OPTION_PUT", "TACTICAL_PUT", "MVRV_SHORT"]:
            exit_cfg = policy.get_exit_params(sig)
            spread_width = policy.get_spread_width(sig)
            dte_cfg = policy.get_dte_target(sig)
            if exit_cfg:
                rr = (spread_width * exit_cfg.take_profit_pct) / exit_cfg.stop_loss_pct
                self.stdout.write(
                    f"  {sig:<15} | DTE: {dte_cfg.min_dte}-{dte_cfg.max_dte}d | "
                    f"Stop: {exit_cfg.stop_loss_pct*100:.1f}% | R:R: 1:{rr:.2f}"
                )
        
        self.stdout.write("\n🟡 NEUTRAL SIGNALS:")
        self.stdout.write("-"*60)
        for sig in ["IRON_CONDOR"]:
            exit_cfg = policy.get_exit_params(sig)
            spread_width = policy.get_spread_width(sig)
            dte_cfg = policy.get_dte_target(sig)
            if exit_cfg:
                rr = (spread_width * exit_cfg.take_profit_pct) / exit_cfg.stop_loss_pct
                self.stdout.write(
                    f"  {sig:<15} | DTE: {dte_cfg.min_dte}-{dte_cfg.max_dte}d | "
                    f"Stop: {exit_cfg.stop_loss_pct*100:.1f}% | R:R: 1:{rr:.2f}"
                )

        self.stdout.write("\n💰 INCOME SPREADS:")
        self.stdout.write("-"*60)
        self.stdout.write(
            f"  {'BULL_PUT_SPREAD':<15} | DTE: 9-21d (tactical) / 21-45d (income) | "
            f"Credit spread below MVRV floor"
        )
        self.stdout.write(
            f"  {'BEAR_CALL_SPREAD':<15} | DTE: 9-21d (tactical) / 21-45d (income) | "
            f"Credit spread above MVRV ceiling"
        )
        
        self.stdout.write("\n" + "="*60)
        self.stdout.write("Usage: python manage.py manual_setup --signal <SIGNAL_TYPE>")
        self.stdout.write("="*60 + "\n")

    def _get_spot_price(self, target_date, override_spot):
        """Get spot price from DB or override."""
        if override_spot:
            if target_date:
                return override_spot, target_date
            raw = RawDailyData.objects.order_by('-date').first()
            return override_spot, raw.date if raw else None
        
        if target_date:
            try:
                raw = RawDailyData.objects.get(date=target_date)
                return float(raw.btc_close), target_date
            except RawDailyData.DoesNotExist:
                return None, None
        else:
            raw = RawDailyData.objects.order_by('-date').first()
            if raw:
                return float(raw.btc_close), raw.date
            return None, None

    def _generate_real_setup(self, signal_type: str, spot: float, signal_date, policy, show_chain: bool = False) -> dict:
        """
        Generate setup using REAL option chain data from OptionSnapshot.
        Returns None if no option data available.
        """
        # Get policy parameters
        exit_cfg = policy.get_exit_params(signal_type)
        dte_cfg = policy.get_dte_target(signal_type)
        spread_width_pct = policy.get_spread_width(signal_type)
        delta_target = policy.get_signal_delta(signal_type)
        path_profile = policy.get_path_profile(signal_type)
        tier = policy.get_tier(signal_type)
        expected_edge = policy.get_expected_edge(signal_type)
        
        # Determine direction and option type
        if signal_type in ["CALL", "BULL_PROBE", "OPTION_CALL"]:
            direction = "LONG"
            option_type = "call"
        elif signal_type == "IRON_CONDOR":
            direction = "NEUTRAL"
            option_type = "condor"
        elif signal_type in ("BULL_PUT_SPREAD", "BEAR_CALL_SPREAD"):
            direction = "INCOME"
            option_type = "income"
        else:
            direction = "SHORT"
            option_type = "put"
        
        # For iron condor, handle separately
        if option_type == "condor":
            return self._generate_condor_setup(signal_type, spot, signal_date, policy)

        # For income spreads, handle separately
        if signal_type in ("BULL_PUT_SPREAD", "BEAR_CALL_SPREAD"):
            return self._generate_income_spread_setup(signal_type, spot, signal_date)
        
        # Find latest option snapshot timestamp for the date
        # Try signal_date first, then look for most recent
        latest_ts = OptionSnapshot.objects.filter(
            timestamp__date=signal_date,
            option_type=option_type,
        ).aggregate(Max('timestamp'))['timestamp__max']
        
        used_fallback_date = False
        if not latest_ts:
            # No same-day data — fall back to most recent chain
            latest_ts = OptionSnapshot.objects.filter(
                option_type=option_type,
            ).aggregate(Max('timestamp'))['timestamp__max']
            used_fallback_date = True
        
        if not latest_ts:
            return None
        
        # Warn if using a different date's snapshot
        if used_fallback_date:
            actual_date = latest_ts.date() if hasattr(latest_ts, 'date') else latest_ts
            self.stderr.write(self.style.WARNING(
                f"⚠️  No option chain for {signal_date}. Using snapshot from {actual_date}. "
                f"Prices may not reflect {signal_date} conditions."
            ))
        
        # Track pricing date mismatch for downstream consumers
        pricing_date_mismatch = used_fallback_date
        
        # Get all options in DTE range
        options = list(OptionSnapshot.objects.filter(
            timestamp=latest_ts,
            option_type=option_type,
            dte__gte=dte_cfg.min_dte,
            dte__lte=dte_cfg.max_dte,
        ).order_by('strike'))
        
        if not options:
            # Try wider DTE range
            options = list(OptionSnapshot.objects.filter(
                timestamp=latest_ts,
                option_type=option_type,
                dte__gte=dte_cfg.min_dte - 2,
                dte__lte=dte_cfg.max_dte + 5,
            ).order_by('strike'))
        
        if not options:
            return None
        
        # Get spot from option snapshot
        snapshot_spot = float(options[0].spot_price)
        
        # Find best long leg (closest to target delta)
        long_opt = min(options, key=lambda x: abs(float(x.delta or 0) - delta_target))
        
        # Find best short leg for spread
        spread_width_usd = snapshot_spot * spread_width_pct
        risk_budget = tier.risk_usd * tier.spread_pct
        
        short_opt = self._find_best_short_leg(
            options, long_opt, option_type, direction,
            snapshot_spot, spread_width_pct, risk_budget
        )
        
        if not short_opt:
            return None
        
        # Calculate spread metrics
        width = abs(float(long_opt.strike) - float(short_opt.strike))
        long_price = float(long_opt.ask) if long_opt.ask else float(long_opt.mid_price or 0)
        short_price = float(short_opt.bid) if short_opt.bid else float(short_opt.mid_price or 0)
        net_debit = long_price - short_price
        
        if net_debit <= 0:
            net_debit = width * 0.15  # Estimate if prices unavailable
        
        max_profit = width - net_debit
        max_loss = net_debit
        
        if max_loss <= 0 or max_profit <= 0:
            return None
        
        risk_reward = max_profit / max_loss
        
        # Calculate R:R from policy
        if exit_cfg:
            policy_rr = (spread_width_pct * exit_cfg.take_profit_pct) / exit_cfg.stop_loss_pct
        else:
            policy_rr = risk_reward
        
        # Position sizing
        contracts = int(risk_budget / max_loss) if max_loss > 0 else 1
        if contracts < 1:
            contracts = 1
        
        # Breakeven
        if option_type == "put":
            breakeven = float(long_opt.strike) - net_debit
        else:
            breakeven = float(long_opt.strike) + net_debit
        
        # Stop loss spot price
        if direction == "SHORT":
            stop_spot = snapshot_spot * (1 + exit_cfg.stop_loss_pct) if exit_cfg else snapshot_spot * 1.05
        else:
            stop_spot = snapshot_spot * (1 - exit_cfg.stop_loss_pct) if exit_cfg else snapshot_spot * 0.95
        
        # Build option chain candidates for display
        chain_candidates = []
        if show_chain:
            for opt in options:
                chain_candidates.append({
                    "symbol": opt.symbol,
                    "strike": float(opt.strike),
                    "delta": float(opt.delta) if opt.delta else None,
                    "iv": float(opt.iv) if opt.iv else None,
                    "bid": float(opt.bid) if opt.bid else None,
                    "ask": float(opt.ask) if opt.ask else None,
                    "oi": int(opt.open_interest) if opt.open_interest else 0,
                    "dte": float(opt.dte) if opt.dte else None,
                    "expiry": opt.expiry.strftime("%Y-%m-%d") if opt.expiry else None,
                })
        
        return {
            "signal_type": signal_type,
            "direction": direction,
            "option_type": option_type,
            "is_real_chain": True,
            "pricing_date_mismatch": pricing_date_mismatch,
            "snapshot_timestamp": latest_ts.isoformat(),
            "signal_date": signal_date,
            "spot_price": snapshot_spot,
            "policy_version": policy.version,
            
            # DTE
            "dte": {
                "target_min": dte_cfg.min_dte,
                "target_max": dte_cfg.max_dte,
                "target_optimal": dte_cfg.optimal_dte,
                "actual": int(long_opt.dte) if long_opt.dte else None,
            },
            
            # Delta
            "delta": {
                "target": abs(delta_target),
                "actual_long": float(long_opt.delta) if long_opt.delta else None,
                "actual_short": float(short_opt.delta) if short_opt.delta else None,
            },
            
            # Long leg (BUY)
            "long_leg": {
                "symbol": long_opt.symbol,
                "action": "BUY",
                "strike": float(long_opt.strike),
                "delta": float(long_opt.delta) if long_opt.delta else None,
                "iv": float(long_opt.iv) if long_opt.iv else None,
                "bid": float(long_opt.bid) if long_opt.bid else None,
                "ask": float(long_opt.ask) if long_opt.ask else None,
                "price": long_price,
                "open_interest": int(long_opt.open_interest) if long_opt.open_interest else 0,
                "expiry": long_opt.expiry.strftime("%Y-%m-%d %H:%M") if long_opt.expiry else None,
            },
            
            # Short leg (SELL)
            "short_leg": {
                "symbol": short_opt.symbol,
                "action": "SELL",
                "strike": float(short_opt.strike),
                "delta": float(short_opt.delta) if short_opt.delta else None,
                "iv": float(short_opt.iv) if short_opt.iv else None,
                "bid": float(short_opt.bid) if short_opt.bid else None,
                "ask": float(short_opt.ask) if short_opt.ask else None,
                "price": short_price,
                "open_interest": int(short_opt.open_interest) if short_opt.open_interest else 0,
                "expiry": short_opt.expiry.strftime("%Y-%m-%d %H:%M") if short_opt.expiry else None,
            },
            
            # Spread metrics
            "spread": {
                "width_usd": width,
                "width_pct": width / snapshot_spot,
                "target_width_pct": spread_width_pct,
                "net_debit": net_debit,
                "max_profit": max_profit,
                "max_loss": max_loss,
                "breakeven": breakeven,
            },
            
            # Risk/Reward
            "risk_reward": {
                "ratio": risk_reward,
                "ratio_formatted": f"1:{risk_reward:.2f}",
                "policy_ratio": policy_rr,
                "policy_ratio_formatted": f"1:{policy_rr:.2f}",
                "risk_pct": exit_cfg.stop_loss_pct if exit_cfg else None,
                "risk_usd": snapshot_spot * exit_cfg.stop_loss_pct if exit_cfg else None,
                "reward_pct": (width / snapshot_spot) * exit_cfg.take_profit_pct if exit_cfg else None,
                "reward_usd": max_profit * exit_cfg.take_profit_pct if exit_cfg else None,
            },
            
            # Exit rules
            "exit_rules": {
                "stop_loss_pct": exit_cfg.stop_loss_pct if exit_cfg else None,
                "stop_loss_spot": stop_spot,
                "take_profit_pct": exit_cfg.take_profit_pct if exit_cfg else None,
                "take_profit_value": max_profit * exit_cfg.take_profit_pct if exit_cfg else None,
                "max_hold_days": exit_cfg.max_hold_days if exit_cfg else None,
                "scale_down_day": exit_cfg.scale_down_day if exit_cfg else None,
                "profit_lock_threshold": exit_cfg.profit_lock_threshold if exit_cfg else None,
                "trailing_stop_pct": exit_cfg.trailing_stop_pct if exit_cfg else None,
                "stop_tighten_day": exit_cfg.stop_tighten_day if exit_cfg else None,
            } if exit_cfg else None,
            
            # Path profile
            "path_profile": {
                "shakeout_pct": path_profile.get("shakeout_pct", 0),
                "invalidation_pct": path_profile.get("invalidation_pct", 0),
                "mae_p75": path_profile.get("mae_p75", 0),
                "clean_win_pct": path_profile.get("clean_win_pct", 1.0),
                "is_shakeout_heavy": policy.is_shakeout_heavy(signal_type),
                "is_invalidation_heavy": policy.is_invalidation_heavy(signal_type),
            },
            
            # Position sizing
            "position": {
                "tier": tier.risk_usd,
                "spread_allocation_pct": tier.spread_pct,
                "risk_budget": risk_budget,
                "contracts": contracts,
                "total_risk": contracts * max_loss,
                "total_max_profit": contracts * max_profit,
            },
            
            # Expected edge
            "expected_edge_pct": expected_edge,
            
            # Option chain (if requested)
            "option_chain": chain_candidates if show_chain else None,
            
            # Rationale
            "rationale": self._get_rationale(signal_type),
        }

    def _find_best_short_leg(
        self,
        options: list,
        long_opt,
        option_type: str,
        direction: str,
        spot_price: float,
        policy_width_pct: float,
        risk_budget: float,
    ):
        """Find the best short leg for the spread."""
        policy_width_usd = spot_price * policy_width_pct
        best_opt = None
        best_score = -999
        
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
            
            long_price = float(long_opt.ask) if long_opt.ask else float(long_opt.mid_price or 0)
            short_price = float(opt.bid) if opt.bid else float(opt.mid_price or 0)
            net_debit = long_price - short_price
            
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
            
            contracts = int(risk_budget / max_loss) if max_loss > 0 else 0
            oi = int(opt.open_interest) if opt.open_interest else 0
            
            # Scoring
            # R:R score (0-1)
            if rr >= 1.0:
                rr_score = min(1.0, 0.7 + (rr - 1.0) * 0.3)
            elif rr >= 0.8:
                rr_score = 0.5 + (rr - 0.8) * 1.0
            else:
                rr_score = rr * 0.625
            
            # Width score (0-1)
            width_ratio = width / policy_width_usd if policy_width_usd > 0 else 0
            if 0.5 <= width_ratio <= 1.5:
                width_score = 1.0 - abs(width_ratio - 1.0) * 0.5
            elif width_ratio < 0.5:
                width_score = width_ratio * 0.6
            else:
                width_score = max(0.3, 1.0 - (width_ratio - 1.5) * 0.2)
            
            # Budget fit score (0-1)
            if contracts >= 3:
                budget_score = 1.0
            elif contracts >= 1:
                budget_score = 0.7 + contracts * 0.1
            else:
                budget_score = 0.3
            
            # Liquidity score (0-1)
            if oi >= 50:
                liq_score = 1.0
            elif oi >= 10:
                liq_score = 0.5 + (oi - 10) * 0.0125
            else:
                liq_score = oi * 0.05
            
            # Weighted score
            score = rr_score * 0.40 + width_score * 0.30 + budget_score * 0.20 + liq_score * 0.10
            
            if score > best_score:
                best_score = score
                best_opt = opt
        
        return best_opt

    def _generate_condor_setup(self, signal_type: str, spot: float, signal_date, policy) -> dict:
        """Generate iron condor setup using real option chain."""
        exit_cfg = policy.get_exit_params(signal_type)
        dte_cfg = policy.get_dte_target(signal_type)
        spread_width_pct = policy.get_spread_width(signal_type)
        path_profile = policy.get_path_profile(signal_type)
        tier = policy.get_tier(signal_type)
        expected_edge = policy.get_expected_edge(signal_type)
        
        # Get MVRV-based strikes
        condor_strikes = self._compute_condor_strikes(spot, signal_date)
        
        # Find latest option snapshot for the requested date
        latest_ts = OptionSnapshot.objects.filter(
            timestamp__date=signal_date,
        ).aggregate(Max('timestamp'))['timestamp__max']
        
        used_fallback_date = False
        if not latest_ts:
            latest_ts = OptionSnapshot.objects.aggregate(Max('timestamp'))['timestamp__max']
            used_fallback_date = True
        
        if not latest_ts:
            return None
        
        if used_fallback_date:
            actual_date = latest_ts.date() if hasattr(latest_ts, 'date') else latest_ts
            self.stderr.write(self.style.WARNING(
                f"⚠️  No option chain for {signal_date}. Using snapshot from {actual_date}."
            ))
        
        snapshot_spot = float(OptionSnapshot.objects.filter(timestamp=latest_ts).first().spot_price)
        
        # Find call options near short call strike
        call_options = list(OptionSnapshot.objects.filter(
            timestamp=latest_ts,
            option_type='call',
            dte__gte=dte_cfg.min_dte,
            dte__lte=dte_cfg.max_dte,
        ).order_by('strike'))
        
        # Find put options near short put strike
        put_options = list(OptionSnapshot.objects.filter(
            timestamp=latest_ts,
            option_type='put',
            dte__gte=dte_cfg.min_dte,
            dte__lte=dte_cfg.max_dte,
        ).order_by('strike'))
        
        if not call_options or not put_options:
            return None
        
        # Find short call (closest to condor_strikes.short_call, OTM)
        target_short_call = condor_strikes.short_call if condor_strikes else snapshot_spot * 1.10
        short_call = min(
            [o for o in call_options if float(o.strike) >= snapshot_spot],
            key=lambda x: abs(float(x.strike) - target_short_call),
            default=None
        )
        if not short_call:
            return None
        
        # Find short put (closest to condor_strikes.short_put, SAME EXPIRY as short call)
        target_short_put = condor_strikes.short_put if condor_strikes else snapshot_spot * 0.90
        same_expiry_puts = [o for o in put_options if o.expiry == short_call.expiry]
        if not same_expiry_puts:
            return None  # Cannot build valid condor without same-expiry puts
        short_put = min(
            [o for o in same_expiry_puts if float(o.strike) <= snapshot_spot],
            key=lambda x: abs(float(x.strike) - target_short_put),
            default=None
        )
        if not short_put:
            return None
        
        # Find long call wing (same expiry, higher strike than short call)
        condor_policy = policy.condor
        wing_offset = condor_policy.wing_offset_usd if hasattr(condor_policy, 'wing_offset_usd') else 5000
        long_call_target = float(short_call.strike) + wing_offset
        long_call_candidates = [o for o in call_options 
                                if o.expiry == short_call.expiry and float(o.strike) > float(short_call.strike)]
        long_call = min(long_call_candidates, key=lambda x: abs(float(x.strike) - long_call_target)) if long_call_candidates else None
        
        # Find long put wing (same expiry, lower strike than short put)
        long_put_target = float(short_put.strike) - wing_offset
        long_put_candidates = [o for o in put_options 
                               if o.expiry == short_call.expiry and float(o.strike) < float(short_put.strike)]
        long_put = min(long_put_candidates, key=lambda x: abs(float(x.strike) - long_put_target)) if long_put_candidates else None
        
        # Require all 4 legs for a valid condor — fail closed if wings unavailable
        if not long_call or not long_put:
            return None  # Cannot build valid iron condor without wing protection
        
        # Calculate real metrics
        net_credit = (
            float(short_call.bid or 0) + float(short_put.bid or 0) -
            float(long_call.ask or 0) - float(long_put.ask or 0)
        )
        
        # Fail closed if not a valid credit structure
        if net_credit <= 0:
            return None  # Not a viable income condor — zero or negative credit
        
        call_wing_width = float(long_call.strike) - float(short_call.strike)
        put_wing_width = float(short_put.strike) - float(long_put.strike)
        wing_width = max(call_wing_width, put_wing_width)
        max_profit = net_credit
        max_loss = wing_width - net_credit
        
        if max_loss <= 0 or max_profit <= 0:
            return None  # Invalid risk/reward structure
        
        real_rr = max_profit / max_loss
        upper_breakeven = float(short_call.strike) + net_credit
        lower_breakeven = float(short_put.strike) - net_credit
        
        # Calculate policy R:R for reference
        if exit_cfg:
            policy_rr = (spread_width_pct * exit_cfg.take_profit_pct) / exit_cfg.stop_loss_pct
        else:
            policy_rr = 0.74
        
        return {
            "signal_type": signal_type,
            "direction": "NEUTRAL",
            "option_type": "condor",
            "is_real_chain": True,
            "pricing_date_mismatch": used_fallback_date,
            "snapshot_timestamp": latest_ts.isoformat(),
            "signal_date": signal_date,
            "spot_price": snapshot_spot,
            "policy_version": policy.version,
            
            "dte": {
                "target_min": dte_cfg.min_dte,
                "target_max": dte_cfg.max_dte,
                "target_optimal": dte_cfg.optimal_dte,
                "actual": int(short_call.dte) if short_call.dte else None,
            },
            
            "short_call": {
                "symbol": short_call.symbol,
                "action": "SELL",
                "strike": float(short_call.strike),
                "delta": float(short_call.delta) if short_call.delta else None,
                "iv": float(short_call.iv) if short_call.iv else None,
                "bid": float(short_call.bid) if short_call.bid else None,
                "ask": float(short_call.ask) if short_call.ask else None,
                "open_interest": int(short_call.open_interest) if short_call.open_interest else 0,
                "distance_pct": (float(short_call.strike) - snapshot_spot) / snapshot_spot * 100,
                "expiry": short_call.expiry.strftime("%Y-%m-%d %H:%M") if short_call.expiry else None,
            },
            
            "short_put": {
                "symbol": short_put.symbol,
                "action": "SELL",
                "strike": float(short_put.strike),
                "delta": float(short_put.delta) if short_put.delta else None,
                "iv": float(short_put.iv) if short_put.iv else None,
                "bid": float(short_put.bid) if short_put.bid else None,
                "ask": float(short_put.ask) if short_put.ask else None,
                "open_interest": int(short_put.open_interest) if short_put.open_interest else 0,
                "distance_pct": (snapshot_spot - float(short_put.strike)) / snapshot_spot * 100,
                "expiry": short_put.expiry.strftime("%Y-%m-%d %H:%M") if short_put.expiry else None,
            },
            
            "mvrv_strikes": {
                "target_short_call": target_short_call,
                "target_short_put": target_short_put,
                "cost_basis": condor_strikes.cost_basis if condor_strikes else None,
                "mvrv_60d": condor_strikes.mvrv_60d if condor_strikes else None,
                "mvrv_drift": condor_strikes.mvrv_drift if condor_strikes else None,
            } if condor_strikes else None,
            
            "risk_reward": {
                "ratio": real_rr,
                "ratio_formatted": f"1:{real_rr:.2f}" if real_rr else "N/A",
                "policy_ratio": policy_rr,
                "policy_ratio_formatted": f"1:{policy_rr:.2f}" if policy_rr else "N/A",
                "risk_pct": exit_cfg.stop_loss_pct if exit_cfg else None,
                "risk_usd": snapshot_spot * exit_cfg.stop_loss_pct if exit_cfg else None,
            },
            
            # Full 4-leg structure
            "long_call_wing": {
                "symbol": long_call.symbol,
                "action": "BUY",
                "strike": float(long_call.strike),
                "delta": float(long_call.delta) if long_call.delta else None,
                "ask": float(long_call.ask) if long_call.ask else None,
                "open_interest": int(long_call.open_interest) if long_call.open_interest else 0,
            },
            "long_put_wing": {
                "symbol": long_put.symbol,
                "action": "BUY",
                "strike": float(long_put.strike),
                "delta": float(long_put.delta) if long_put.delta else None,
                "ask": float(long_put.ask) if long_put.ask else None,
                "open_interest": int(long_put.open_interest) if long_put.open_interest else 0,
            },
            
            # Real metrics from option chain
            "condor_metrics": {
                "net_credit": net_credit,
                "max_profit": max_profit,
                "max_loss": max_loss,
                "wing_width": wing_width,
                "upper_breakeven": upper_breakeven,
                "lower_breakeven": lower_breakeven,
            },
            
            "exit_rules": {
                "stop_loss_pct": exit_cfg.stop_loss_pct if exit_cfg else None,
                "take_profit_pct": exit_cfg.take_profit_pct if exit_cfg else None,
                "max_hold_days": exit_cfg.max_hold_days if exit_cfg else None,
                "scale_down_day": exit_cfg.scale_down_day if exit_cfg else None,
            } if exit_cfg else None,
            
            "path_profile": {
                "shakeout_pct": path_profile.get("shakeout_pct", 0),
                "invalidation_pct": path_profile.get("invalidation_pct", 0),
                "mae_p75": path_profile.get("mae_p75", 0),
            },
            
            "position": {
                "tier": tier.risk_usd,
                "risk_budget": tier.risk_usd * tier.spread_pct,
            },
            
            "expected_edge_pct": expected_edge,
            "rationale": self._get_rationale(signal_type),
        }

    def _generate_income_spread_setup(self, signal_type: str, spot: float, signal_date) -> dict:
        """
        Generate setup for income spreads (BULL_PUT_SPREAD / BEAR_CALL_SPREAD).
        Uses the income_gate module with real option chain data.
        """
        import pandas as pd
        from signals.fusion import add_fusion_features, fuse_signals

        # Load features for the date
        csv_path = "features_14d_5pct.csv"
        try:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            df = add_fusion_features(df)
            date_str = signal_date.strftime("%Y-%m-%d") if signal_date else df.index[-1].strftime("%Y-%m-%d")
            if date_str in df.index.strftime("%Y-%m-%d"):
                row = df.loc[date_str]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
            else:
                row = df.iloc[-1]
                date_str = row.name.strftime("%Y-%m-%d")
        except FileNotFoundError:
            self.stderr.write(self.style.ERROR(f"Feature CSV not found: {csv_path}"))
            return None

        fusion_result = fuse_signals(row)

        # Get option chain
        from datetime import datetime as dt
        target = signal_date or dt.now().date()
        chain_qs = OptionSnapshot.objects.filter(
            timestamp__date=target, underlying='BTC', exchange='deribit',
        ).values('symbol', 'exchange', 'timestamp', 'strike', 'option_type', 'delta', 'bid', 'ask', 'dte', 'spread_pct', 'spot_price')

        chain_df = pd.DataFrame.from_records(chain_qs)
        if len(chain_df) == 0:
            self.stderr.write(self.style.WARNING(f"No option chain for {target}"))
            return None

        # Collapse hourly snapshots to one row per contract (latest), so manual
        # setup mirrors live selection and tiers land on distinct strikes.
        chain_df = dedupe_chain_to_latest(chain_df)

        for col in ['strike', 'delta', 'bid', 'ask', 'dte', 'spread_pct', 'spot_price']:
            chain_df[col] = chain_df[col].apply(lambda x: float(x) if x is not None else None)

        chain_spot = chain_df['spot_price'].iloc[0]
        use_spot = spot if spot else chain_spot

        # Evaluate
        config = IncomeGateConfig()
        # Data-driven bear-call ceiling target (underwater regime) — mirror live
        # generation so manual setup selection matches SignalService for the
        # same date and chain.
        bear_target_mvrv = None
        if 'mvrv_60d' in df.columns:
            bear_target_mvrv = compute_bear_call_target_mvrv(
                df.loc[:date_str, 'mvrv_60d'], config
            )
        if signal_type == "BULL_PUT_SPREAD":
            # Data-driven bull-put floor target (reachable drawdown) — mirror
            # live generation so manual setup selection matches SignalService
            # for the same date and chain.
            bull_target_mvrv = None
            if 'mvrv_60d' in df.columns:
                bull_target_mvrv = compute_bull_put_target_mvrv(
                    df.loc[:date_str, 'mvrv_60d'], config
                )
            gate_result = evaluate_bull_put_gate(
                row, chain_df=chain_df, spot_price=use_spot,
                fusion_state=fusion_result.state.value, config=config,
                target_mvrv=bull_target_mvrv,
            )
            score_fn = compute_bull_put_score
            side_label = "BULL PUT SPREAD"
            direction = "LONG"
        else:
            gate_result = evaluate_bear_call_gate(
                row, chain_df=chain_df, spot_price=use_spot,
                fusion_state=fusion_result.state.value, config=config,
                target_mvrv=bear_target_mvrv,
            )
            score_fn = compute_bear_call_score
            side_label = "BEAR CALL SPREAD"
            direction = "SHORT"

        # Build output
        score, components = score_fn(row)
        mvrv_60d = float(row.get('mvrv_60d', row.get('mvrv_usd_60d', 0)))
        cost_basis = use_spot / mvrv_60d if mvrv_60d > 0 else 0

        self.stdout.write("\n" + "=" * 70)
        self.stdout.write(f"  💰 {side_label} — {date_str}")
        self.stdout.write("=" * 70)
        self.stdout.write(f"\n  BTC Spot: ${use_spot:,.0f}")
        self.stdout.write(f"  Fusion:   {fusion_result.state.value}")
        self.stdout.write(f"  Chain:    {len(chain_df)} contracts\n")

        self.stdout.write("  --- REGIME GATE ---")
        self.stdout.write(f"  Score: {score:.0f}/100 (threshold: 70)")
        for k, v in components.items():
            status = "✅" if v > 0 else "❌"
            self.stdout.write(f"    {status} {k}: +{v:.0f}")
        self.stdout.write(f"  Vetoes: {gate_result.veto_reasons if gate_result.veto_reasons else 'None'}")
        self.stdout.write(f"  Regime Eligible: {gate_result.regime_eligible}\n")

        self.stdout.write("  --- MVRV BOUNDARY ---")
        self.stdout.write(f"  MVRV-60D: {mvrv_60d:.4f}")
        self.stdout.write(f"  Cost basis: ${cost_basis:,.0f}")
        p10 = row.get('mvrv_60d_p10_180d')
        p90 = row.get('mvrv_60d_p90_180d')
        if signal_type == "BULL_PUT_SPREAD":
            if bull_target_mvrv:
                # Data-driven reachable floor (matches gate): cost_basis × target_mvrv
                floor = cost_basis * float(bull_target_mvrv)
                self.stdout.write(
                    f"  Floor: ${floor:,.0f} (cost_basis × target_mvrv={float(bull_target_mvrv):.4f} "
                    f"[P{config.bull_put_floor_percentile*100:.0f} trailing drawdown], "
                    f"{((use_spot-floor)/use_spot*100):.1f}% below)"
                )
            elif cost_basis <= use_spot:
                self.stdout.write(f"  Floor: ${cost_basis:,.0f} (cost_basis, {((use_spot-cost_basis)/use_spot*100):.1f}% below)")
            elif p10:
                floor = cost_basis * float(p10)
                self.stdout.write(f"  Floor: ${floor:,.0f} (cost_basis × P10={float(p10):.4f}, {((use_spot-floor)/use_spot*100):.1f}% below)")
        else:
            if mvrv_60d > 0 and mvrv_60d < 1.0 and bear_target_mvrv:
                # Underwater: data-driven trailing MVRV-rally ceiling (matches gate)
                ceiling = cost_basis * float(bear_target_mvrv)
                self.stdout.write(
                    f"  Ceiling: ${ceiling:,.0f} (cost_basis × target_mvrv={float(bear_target_mvrv):.4f} "
                    f"[P{config.bear_call_ceiling_percentile*100:.0f} trailing rally], "
                    f"{((ceiling-use_spot)/use_spot*100):.1f}% above)"
                )
            elif p90:
                ceiling = cost_basis * float(p90)
                self.stdout.write(f"  Ceiling: ${ceiling:,.0f} (cost_basis × P90={float(p90):.4f}, {((ceiling-use_spot)/use_spot*100):.1f}% above)")

        self.stdout.write("")

        if gate_result.eligible:
            self.stdout.write("  --- AVAILABLE SETUPS ---")
            tier_labels = {"low": "🟢 LOW RISK", "medium": "🟡 MEDIUM RISK", "high": "🔴 HIGH RISK"}
            for setup in gate_result.setups:
                label = tier_labels.get(setup.risk_tier, setup.risk_tier.upper())
                self.stdout.write(f"\n  {label} (delta {setup.short_delta:.2f}, ~{(1-setup.short_delta)*100:.0f}% POP)")
                if signal_type == "BULL_PUT_SPREAD":
                    self.stdout.write(f"    Short Put:  ${setup.short_strike:,.0f} ({setup.otm_pct*100:.1f}% below spot)")
                    self.stdout.write(f"    Long Put:   ${setup.long_strike:,.0f}")
                else:
                    self.stdout.write(f"    Short Call: ${setup.short_strike:,.0f} ({setup.otm_pct*100:.1f}% above spot)")
                    self.stdout.write(f"    Long Call:  ${setup.long_strike:,.0f}")
                self.stdout.write(f"    Width:      ${setup.spread_width:,.0f}")
                self.stdout.write(f"    Credit:     ${setup.credit:,.2f} ({setup.credit_width_pct*100:.1f}% of width)")
                self.stdout.write(f"    Max Loss:   ${setup.max_loss:,.2f}")
                self.stdout.write(f"    DTE:        {setup.dte} days")
                self.stdout.write(f"    R:R:        1:{setup.risk_reward:.2f}")

            self.stdout.write("\n  --- EXIT RULES (all tiers) ---")
            self.stdout.write(f"  Take Profit: 50% of credit received")
            self.stdout.write(f"  Stop Loss:   Spot moves 2% toward short strike")
            self.stdout.write(f"  Scale Down:  Day 12 → reduce to 25%")
            self.stdout.write(f"  Max Hold:    18 days")
        elif gate_result.regime_eligible:
            self.stdout.write(f"  ⚠️  Regime passed but NO TRADABLE SPREAD")
            self.stdout.write(f"  Chain rejection: {gate_result.chain_rejection_reason}")
        else:
            self.stdout.write(f"  ❌ REGIME NOT ELIGIBLE (score {score:.0f} < 70 or vetoed)")

        self.stdout.write("\n" + "=" * 70 + "\n")

        return {
            "signal_type": signal_type,
            "date": date_str,
            "direction": direction,
            "spot": use_spot,
            "score": score,
            "regime_eligible": gate_result.regime_eligible,
            "eligible": gate_result.eligible,
            "short_strike": gate_result.short_strike,
            "long_strike": gate_result.long_strike,
            "credit": gate_result.credit,
            "max_loss": gate_result.max_loss,
            "dte": gate_result.dte,
            "chain_rejection": gate_result.chain_rejection_reason,
            "vetoes": gate_result.veto_reasons,
        }

    def _get_rationale(self, signal_type: str) -> str:
        """Get rationale for signal type."""
        if signal_type in DECISION_STRATEGY_MAP:
            return DECISION_STRATEGY_MAP[signal_type].get("rationale", "")
        elif signal_type in SIGNAL_TO_STATE:
            states = SIGNAL_TO_STATE[signal_type]
            if states and states[0] in STRATEGY_MAP:
                return STRATEGY_MAP[states[0]].rationale
        return ""

    def _generate_theoretical_setup(self, signal_type: str, spot: float, signal_date, policy) -> dict:
        """Generate THEORETICAL setup (no option chain lookup)."""
        """Generate complete setup for a signal type."""
        exit_cfg = policy.get_exit_params(signal_type)
        dte_cfg = policy.get_dte_target(signal_type)
        spread_width_pct = policy.get_spread_width(signal_type)
        delta_target = policy.get_signal_delta(signal_type)
        path_profile = policy.get_path_profile(signal_type)
        tier = policy.get_tier(signal_type)
        expected_edge = policy.get_expected_edge(signal_type)
        
        # Determine direction
        if signal_type in ["CALL", "BULL_PROBE", "OPTION_CALL"]:
            direction = "LONG"
            option_type = "call"
        elif signal_type == "IRON_CONDOR":
            direction = "NEUTRAL"
            option_type = "condor"
        else:
            direction = "SHORT"
            option_type = "put"
        
        # Calculate R:R
        if exit_cfg:
            max_reward_pct = spread_width_pct * exit_cfg.take_profit_pct
            risk_pct = exit_cfg.stop_loss_pct
            risk_reward = max_reward_pct / risk_pct if risk_pct > 0 else 0
        else:
            risk_reward = 0
            max_reward_pct = 0
            risk_pct = 0
        
        # Calculate example strikes
        spread_width_usd = spot * spread_width_pct
        
        if direction == "LONG":
            # Slight ITM for calls: strike below spot
            long_strike = round(spot * 0.97, -2)  # ~3% ITM
            short_strike = round(long_strike + spread_width_usd, -2)
            stop_spot = spot * (1 - exit_cfg.stop_loss_pct) if exit_cfg else spot * 0.95
        elif direction == "SHORT":
            # Slight ITM for puts: strike above spot
            long_strike = round(spot * 1.03, -2)  # ~3% ITM
            short_strike = round(long_strike - spread_width_usd, -2)
            stop_spot = spot * (1 + exit_cfg.stop_loss_pct) if exit_cfg else spot * 1.05
        else:
            # Iron condor - use MVRV-based strikes, stop is % move either direction
            long_strike = None
            short_strike = None
            # For condor, stop triggers if underlying moves beyond wings
            stop_spot = None  # Condor uses % move, not directional spot price
        
        # Build setup dict
        setup = {
            "signal_type": signal_type,
            "direction": direction,
            "option_type": option_type,
            "signal_date": signal_date,
            "spot_price": spot,
            "policy_version": policy.version,
            
            # DTE
            "dte": {
                "min": dte_cfg.min_dte,
                "max": dte_cfg.max_dte,
                "optimal": dte_cfg.optimal_dte,
            },
            
            # Delta
            "delta_target": abs(delta_target),
            
            # Spread
            "spread": {
                "width_pct": spread_width_pct,
                "width_usd": spread_width_usd,
            },
            
            # Risk/Reward
            "risk_reward": {
                "ratio": risk_reward,
                "ratio_formatted": f"1:{risk_reward:.2f}",
                "risk_pct": risk_pct,
                "risk_usd": spot * risk_pct if risk_pct else 0,
                "reward_pct": max_reward_pct,
                "reward_usd": spot * max_reward_pct if max_reward_pct else 0,
            },
            
            # Exit rules
            "exit_rules": {
                "stop_loss_pct": exit_cfg.stop_loss_pct if exit_cfg else None,
                "stop_loss_spot": stop_spot,
                "take_profit_pct": exit_cfg.take_profit_pct if exit_cfg else None,
                "max_hold_days": exit_cfg.max_hold_days if exit_cfg else None,
                "scale_down_day": exit_cfg.scale_down_day if exit_cfg else None,
                "profit_lock_threshold": exit_cfg.profit_lock_threshold if exit_cfg else None,
                "trailing_stop_pct": exit_cfg.trailing_stop_pct if exit_cfg else None,
                "stop_tighten_day": exit_cfg.stop_tighten_day if exit_cfg else None,
            } if exit_cfg else None,
            
            # Path profile
            "path_profile": {
                "shakeout_pct": path_profile.get("shakeout_pct", 0),
                "invalidation_pct": path_profile.get("invalidation_pct", 0),
                "mae_p75": path_profile.get("mae_p75", 0),
                "clean_win_pct": path_profile.get("clean_win_pct", 1.0),
                "is_shakeout_heavy": policy.is_shakeout_heavy(signal_type),
                "is_invalidation_heavy": policy.is_invalidation_heavy(signal_type),
            },
            
            # Position sizing
            "position": {
                "tier": tier.risk_usd,
                "spread_allocation_pct": tier.spread_pct,
                "risk_budget": tier.risk_usd * tier.spread_pct,
            },
            
            # Expected edge
            "expected_edge_pct": expected_edge,
        }
        
        # Add example trade
        if direction == "NEUTRAL":
            # Iron condor strikes
            condor_strikes = self._compute_condor_strikes(spot, signal_date)
            if condor_strikes:
                setup["example_trade"] = {
                    "structure": "iron_condor",
                    "short_call": condor_strikes.short_call,
                    "short_put": condor_strikes.short_put,
                    "call_distance_pct": condor_strikes.call_distance_pct,
                    "put_distance_pct": condor_strikes.put_distance_pct,
                    "cost_basis": condor_strikes.cost_basis,
                    "mvrv_60d": condor_strikes.mvrv_60d,
                    "mvrv_drift": condor_strikes.mvrv_drift,
                }
        else:
            setup["example_trade"] = {
                "structure": f"{option_type}_spread",
                "long_strike": long_strike,
                "short_strike": short_strike,
                "action_long": "BUY",
                "action_short": "SELL",
            }
        
        # Get rationale from strategy maps
        if signal_type in DECISION_STRATEGY_MAP:
            setup["rationale"] = DECISION_STRATEGY_MAP[signal_type].get("rationale", "")
        elif signal_type in SIGNAL_TO_STATE:
            states = SIGNAL_TO_STATE[signal_type]
            if states and states[0] in STRATEGY_MAP:
                setup["rationale"] = STRATEGY_MAP[states[0]].rationale
        
        return setup

    def _compute_condor_strikes(self, spot: float, signal_date):
        """Compute iron condor strikes using MVRV drift projection."""
        try:
            raw = RawDailyData.objects.get(date=signal_date)
            mvrv_60d = float(raw.mvrv_usd_60d) if raw.mvrv_usd_60d else None
        except RawDailyData.DoesNotExist:
            raw = RawDailyData.objects.order_by('-date').first()
            mvrv_60d = float(raw.mvrv_usd_60d) if raw and raw.mvrv_usd_60d else None
        
        if not mvrv_60d:
            return None
        
        # Get trailing MVRV for drift calculation
        trailing_qs = RawDailyData.objects.filter(
            mvrv_usd_60d__isnull=False,
        ).order_by('-date')[:CONDOR_TRAILING_DAYS]
        trailing_mvrv = [r.mvrv_usd_60d for r in trailing_qs]
        
        if len(trailing_mvrv) >= CONDOR_TRAILING_DAYS:
            mvrv_trail_high = max(trailing_mvrv)
            mvrv_trail_low = min(trailing_mvrv)
        else:
            mvrv_trail_high = mvrv_trail_low = None
        
        return compute_condor_strikes(
            spot=spot,
            mvrv_60d=mvrv_60d,
            mvrv_trailing_high=mvrv_trail_high,
            mvrv_trailing_low=mvrv_trail_low,
        )

    def _print_real_setup(self, setup: dict, show_chain: bool = False):
        """Print REAL option chain setup in human-readable format."""
        direction = setup["direction"]
        emoji = "🟢" if direction == "LONG" else "🔴" if direction == "SHORT" else "🟡"
        
        self.stdout.write("\n" + "="*60)
        self.stdout.write(f"{emoji} {setup['signal_type']} SETUP (REAL OPTION CHAIN)")
        self.stdout.write("="*60)
        
        self.stdout.write(f"\n📅 Signal Date: {setup['signal_date']}")
        self.stdout.write(f"🕐 Snapshot: {setup['snapshot_timestamp']}")
        self.stdout.write(f"💰 Spot: ${setup['spot_price']:,.2f}")
        self.stdout.write(f"📊 Direction: {direction}")
        
        # DTE
        dte = setup["dte"]
        self.stdout.write(f"\n⏱️  DTE: Target {dte['target_min']}-{dte['target_max']}d | Actual: {dte['actual']}d")
        
        # Delta (only for spreads, not condors)
        if setup.get("delta"):
            delta = setup["delta"]
            long_delta = f"{delta['actual_long']:.3f}" if delta.get('actual_long') else "N/A"
            short_delta = f"{delta['actual_short']:.3f}" if delta.get('actual_short') else "N/A"
            self.stdout.write(f"Δ  Delta: Target {delta['target']:.2f} | Long: {long_delta} | Short: {short_delta}")
        
        # Option legs
        if setup.get("long_leg"):
            long = setup["long_leg"]
            self.stdout.write(f"\n{'─'*60}")
            self.stdout.write(self.style.SUCCESS(f"📗 LONG LEG (BUY)"))
            self.stdout.write(f"   Symbol:  {long['symbol']}")
            self.stdout.write(f"   Strike:  ${long['strike']:,.0f}")
            self.stdout.write(f"   Delta:   {long['delta']:.4f}" if long['delta'] else "   Delta:   N/A")
            self.stdout.write(f"   IV:      {long['iv']*100:.1f}%" if long['iv'] else "   IV:      N/A")
            self.stdout.write(f"   Bid/Ask: ${long['bid']:,.2f} / ${long['ask']:,.2f}" if long['bid'] and long['ask'] else "   Bid/Ask: N/A")
            self.stdout.write(f"   Price:   ${long['price']:,.2f} (use ASK)")
            self.stdout.write(f"   OI:      {long['open_interest']:,}")
            self.stdout.write(f"   Expiry:  {long['expiry']}")
        
        if setup.get("short_leg"):
            short = setup["short_leg"]
            self.stdout.write(f"\n{'─'*60}")
            self.stdout.write(self.style.ERROR(f"📕 SHORT LEG (SELL)"))
            self.stdout.write(f"   Symbol:  {short['symbol']}")
            self.stdout.write(f"   Strike:  ${short['strike']:,.0f}")
            self.stdout.write(f"   Delta:   {short['delta']:.4f}" if short['delta'] else "   Delta:   N/A")
            self.stdout.write(f"   IV:      {short['iv']*100:.1f}%" if short['iv'] else "   IV:      N/A")
            self.stdout.write(f"   Bid/Ask: ${short['bid']:,.2f} / ${short['ask']:,.2f}" if short['bid'] and short['ask'] else "   Bid/Ask: N/A")
            self.stdout.write(f"   Price:   ${short['price']:,.2f} (use BID)")
            self.stdout.write(f"   OI:      {short['open_interest']:,}")
            self.stdout.write(f"   Expiry:  {short['expiry']}")
        
        # Iron condor legs
        if setup.get("short_call"):
            self.stdout.write(f"\n{'─'*60}")
            sc = setup["short_call"]
            self.stdout.write(self.style.ERROR(f"📕 SHORT CALL (SELL)"))
            self.stdout.write(f"   Symbol:  {sc['symbol']}")
            self.stdout.write(f"   Strike:  ${sc['strike']:,.0f} (+{sc['distance_pct']:.1f}% from spot)")
            self.stdout.write(f"   Delta:   {sc['delta']:.4f}" if sc['delta'] else "   Delta:   N/A")
            self.stdout.write(f"   IV:      {sc['iv']*100:.1f}%" if sc['iv'] else "   IV:      N/A")
            self.stdout.write(f"   Bid:     ${sc['bid']:,.2f}" if sc['bid'] else "   Bid:     N/A")
            self.stdout.write(f"   OI:      {sc['open_interest']:,}")
        
        if setup.get("short_put"):
            sp = setup["short_put"]
            self.stdout.write(f"\n{'─'*60}")
            self.stdout.write(self.style.ERROR(f"📕 SHORT PUT (SELL)"))
            self.stdout.write(f"   Symbol:  {sp['symbol']}")
            self.stdout.write(f"   Strike:  ${sp['strike']:,.0f} (-{sp['distance_pct']:.1f}% from spot)")
            self.stdout.write(f"   Delta:   {sp['delta']:.4f}" if sp['delta'] else "   Delta:   N/A")
            self.stdout.write(f"   IV:      {sp['iv']*100:.1f}%" if sp['iv'] else "   IV:      N/A")
            self.stdout.write(f"   Bid:     ${sp['bid']:,.2f}" if sp['bid'] else "   Bid:     N/A")
            self.stdout.write(f"   OI:      {sp['open_interest']:,}")
        
        # Spread metrics
        if setup.get("spread"):
            spread = setup["spread"]
            self.stdout.write(f"\n{'─'*60}")
            self.stdout.write(f"📐 SPREAD METRICS")
            self.stdout.write(f"   Width:      ${spread['width_usd']:,.0f} ({spread['width_pct']*100:.1f}%)")
            self.stdout.write(f"   Net Debit:  ${spread['net_debit']:,.2f}")
            self.stdout.write(f"   Max Profit: ${spread['max_profit']:,.2f}")
            self.stdout.write(f"   Max Loss:   ${spread['max_loss']:,.2f}")
            self.stdout.write(f"   Breakeven:  ${spread['breakeven']:,.0f}")
        
        # R:R
        rr = setup["risk_reward"]
        self.stdout.write(f"\n{'─'*60}")
        self.stdout.write(f"⚖️  RISK:REWARD")
        self.stdout.write(f"   Actual R:R:  {rr['ratio_formatted']}")
        if rr.get('policy_ratio'):
            self.stdout.write(f"   Policy R:R:  {rr['policy_ratio_formatted']}")
        if rr.get('risk_pct'):
            self.stdout.write(f"   Risk:        {rr['risk_pct']*100:.1f}% = ${rr['risk_usd']:,.0f}")
        if rr.get('reward_pct'):
            self.stdout.write(f"   Reward:      {rr['reward_pct']*100:.1f}% = ${rr['reward_usd']:,.0f}")
        
        # Exit rules
        if setup.get("exit_rules"):
            exit_r = setup["exit_rules"]
            self.stdout.write(f"\n{'─'*60}")
            self.stdout.write(f"🚪 EXIT RULES")
            if exit_r.get('stop_loss_spot'):
                self.stdout.write(f"   Stop Loss:    {exit_r['stop_loss_pct']*100:.1f}% → ${exit_r['stop_loss_spot']:,.0f}")
            elif exit_r.get('stop_loss_pct'):
                self.stdout.write(f"   Stop Loss:    {exit_r['stop_loss_pct']*100:.1f}% (underlying move)")
            if exit_r.get('take_profit_pct'):
                tp_val = exit_r.get('take_profit_value', 0)
                self.stdout.write(f"   Take Profit:  {exit_r['take_profit_pct']*100:.0f}% = ${tp_val:,.2f}/contract")
            if exit_r.get('max_hold_days'):
                self.stdout.write(f"   Max Hold:     {exit_r['max_hold_days']} days")
            if exit_r.get('scale_down_day'):
                self.stdout.write(f"   Scale Down:   Day {exit_r['scale_down_day']}")
            if exit_r.get('trailing_stop_pct') and exit_r['trailing_stop_pct'] > 0:
                self.stdout.write(f"   Trailing:     {exit_r['trailing_stop_pct']*100:.0f}%")
            if exit_r.get('stop_tighten_day'):
                self.stdout.write(f"   Tighten Stop: Day {exit_r['stop_tighten_day']}")
        
        # Position sizing
        if setup.get("position"):
            pos = setup["position"]
            self.stdout.write(f"\n{'─'*60}")
            self.stdout.write(f"💼 POSITION SIZING")
            self.stdout.write(f"   Risk Budget:     ${pos['risk_budget']:,.0f}")
            self.stdout.write(f"   Contracts:       {pos.get('contracts', 'N/A')}")
            if pos.get('total_risk'):
                self.stdout.write(f"   Total Risk:      ${pos['total_risk']:,.2f}")
            if pos.get('total_max_profit'):
                self.stdout.write(f"   Total Max Profit: ${pos['total_max_profit']:,.2f}")
        
        # Path profile
        if setup.get("path_profile"):
            path = setup["path_profile"]
            self.stdout.write(f"\n{'─'*60}")
            self.stdout.write(f"📈 PATH PROFILE")
            self.stdout.write(f"   Shakeout:     {path['shakeout_pct']*100:.0f}%")
            self.stdout.write(f"   Invalidation: {path['invalidation_pct']*100:.0f}%")
            self.stdout.write(f"   MAE p75:      {path['mae_p75']*100:.1f}%")
            if path.get('is_shakeout_heavy'):
                self.stdout.write(self.style.WARNING("   ⚠️  SHAKEOUT-HEAVY: Consider DCA entry"))
            if path.get('is_invalidation_heavy'):
                self.stdout.write(self.style.WARNING("   ⚠️  HIGH INVALIDATION: Use scaled entry"))
        
        # Expected edge
        self.stdout.write(f"\n{'─'*60}")
        self.stdout.write(f"📊 Expected Edge: {setup['expected_edge_pct']*100:.1f}%")
        
        # Rationale
        if setup.get("rationale"):
            self.stdout.write(f"\n💡 Rationale: {setup['rationale'][:100]}...")
        
        # Option chain (if requested)
        if show_chain and setup.get("option_chain"):
            self.stdout.write(f"\n{'─'*60}")
            self.stdout.write(f"📋 OPTION CHAIN CANDIDATES")
            self.stdout.write(f"{'Symbol':<30} | {'Strike':>10} | {'Delta':>8} | {'IV':>6} | {'Bid':>10} | {'Ask':>10} | {'OI':>8}")
            self.stdout.write("-"*60)
            for opt in setup["option_chain"]:
                delta_str = f"{opt['delta']:.4f}" if opt['delta'] else "N/A"
                iv_str = f"{opt['iv']*100:.1f}%" if opt['iv'] else "N/A"
                bid_str = f"${opt['bid']:,.2f}" if opt['bid'] else "N/A"
                ask_str = f"${opt['ask']:,.2f}" if opt['ask'] else "N/A"
                self.stdout.write(
                    f"{opt['symbol']:<30} | ${opt['strike']:>9,.0f} | {delta_str:>8} | {iv_str:>6} | {bid_str:>10} | {ask_str:>10} | {opt['oi']:>8,}"
                )
        
        self.stdout.write("\n" + "="*60)

    def _print_theoretical_setup(self, setup: dict):
        """Print THEORETICAL setup in human-readable format."""
        direction = setup["direction"]
        emoji = "🟢" if direction == "LONG" else "🔴" if direction == "SHORT" else "🟡"
        
        self.stdout.write("\n" + "="*60)
        self.stdout.write(f"{emoji} {setup['signal_type']} SETUP (THEORETICAL)")
        self.stdout.write("="*60)
        
        self.stdout.write(f"\n📅 Date: {setup['signal_date']}")
        self.stdout.write(f"💰 Spot: ${setup['spot_price']:,.0f}")
        self.stdout.write(f"📊 Direction: {direction}")
        
        # DTE
        dte = setup["dte"]
        self.stdout.write(f"\n⏱️  DTE: {dte['min']}-{dte['max']}d (optimal: {dte['optimal']}d)")
        
        # Delta
        self.stdout.write(f"Δ  Delta Target: {setup['delta_target']:.2f}")
        
        # Spread
        spread = setup["spread"]
        self.stdout.write(f"\n📐 Spread Width: {spread['width_pct']*100:.0f}% = ${spread['width_usd']:,.0f}")
        
        # R:R
        rr = setup["risk_reward"]
        self.stdout.write(f"\n⚖️  RISK:REWARD")
        self.stdout.write(f"   Ratio: {rr['ratio_formatted']}")
        self.stdout.write(f"   Risk:   {rr['risk_pct']*100:.1f}% = ${rr['risk_usd']:,.0f}")
        self.stdout.write(f"   Reward: {rr['reward_pct']*100:.1f}% = ${rr['reward_usd']:,.0f}")
        
        # Exit rules
        if setup["exit_rules"]:
            exit_r = setup["exit_rules"]
            self.stdout.write(f"\n🚪 EXIT RULES")
            if exit_r['stop_loss_spot']:
                self.stdout.write(f"   Stop Loss:    {exit_r['stop_loss_pct']*100:.1f}% → ${exit_r['stop_loss_spot']:,.0f}")
            else:
                self.stdout.write(f"   Stop Loss:    {exit_r['stop_loss_pct']*100:.1f}% (underlying move)")
            self.stdout.write(f"   Take Profit:  {exit_r['take_profit_pct']*100:.0f}%")
            self.stdout.write(f"   Max Hold:     {exit_r['max_hold_days']} days")
            if exit_r['scale_down_day']:
                self.stdout.write(f"   Scale Down:   Day {exit_r['scale_down_day']}")
            if exit_r['trailing_stop_pct'] and exit_r['trailing_stop_pct'] > 0:
                self.stdout.write(f"   Trailing:     {exit_r['trailing_stop_pct']*100:.0f}%")
            if exit_r['stop_tighten_day']:
                self.stdout.write(f"   Tighten Stop: Day {exit_r['stop_tighten_day']}")
        
        # Path profile
        path = setup["path_profile"]
        self.stdout.write(f"\n📈 PATH PROFILE")
        self.stdout.write(f"   Shakeout:     {path['shakeout_pct']*100:.0f}%")
        self.stdout.write(f"   Invalidation: {path['invalidation_pct']*100:.0f}%")
        self.stdout.write(f"   MAE p75:      {path['mae_p75']*100:.1f}%")
        if path['is_shakeout_heavy']:
            self.stdout.write(self.style.WARNING("   ⚠️  SHAKEOUT-HEAVY: Consider DCA entry"))
        if path['is_invalidation_heavy']:
            self.stdout.write(self.style.WARNING("   ⚠️  HIGH INVALIDATION: Use scaled entry"))
        
        # Example trade
        trade = setup.get("example_trade", {})
        self.stdout.write(f"\n📍 EXAMPLE TRADE @ ${setup['spot_price']:,.0f}")
        if trade.get("structure") == "iron_condor":
            self.stdout.write(f"   Short Call: ${trade['short_call']:,.0f} (+{trade['call_distance_pct']:.1f}%)")
            self.stdout.write(f"   Short Put:  ${trade['short_put']:,.0f} (-{trade['put_distance_pct']:.1f}%)")
            self.stdout.write(f"   Cost Basis: ${trade['cost_basis']:,.0f}")
            self.stdout.write(f"   MVRV Drift: {trade['mvrv_drift']:.4f}")
        else:
            self.stdout.write(f"   BUY  {setup['option_type'].upper()}: ${trade['long_strike']:,.0f}")
            self.stdout.write(f"   SELL {setup['option_type'].upper()}: ${trade['short_strike']:,.0f}")
        
        # Expected edge
        self.stdout.write(f"\n📊 Expected Edge: {setup['expected_edge_pct']*100:.1f}%")
        
        # Rationale
        if setup.get("rationale"):
            self.stdout.write(f"\n💡 Rationale: {setup['rationale'][:80]}...")
        
        self.stdout.write("\n" + "="*60)