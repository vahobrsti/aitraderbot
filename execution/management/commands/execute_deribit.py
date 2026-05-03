"""
Execute a signal on Deribit using the greeks-aware entry engine.

This is the Deribit-specific replacement for execute_signal. It uses
OptionSnapshot data from Postgres to find optimal contracts based on
delta, IV, liquidity, and path-informed DTE targets.

Features:
- Policy-driven configuration (versioned, auditable)
- IV-aware scoring (buy legs favor low IV, sell legs favor high IV)
- Execution cost guardrails (reject if edge < costs)
- Limit orders with slippage caps (safer than market orders)
- Multi-leg atomicity with rollback/hedge on partial fills

Usage:
    # Plan only (show what would be traded)
    python manage.py execute_deribit --latest --account deribit-main --plan

    # Dry run (create intent, no orders)
    python manage.py execute_deribit --latest --account deribit-main --dry-run

    # Live execution
    python manage.py execute_deribit --latest --account deribit-main

    # Execute specific date
    python manage.py execute_deribit --date 2026-05-01 --account deribit-main

    # Override spot price (useful for testing)
    python manage.py execute_deribit --latest --account deribit-main --spot 95000
    
    # Use market orders instead of limit (not recommended)
    python manage.py execute_deribit --latest --account deribit-main --market-orders
"""
import logging
from datetime import date
from decimal import Decimal

from django.core.management.base import BaseCommand, CommandError

from signals.models import DailySignal
from execution.models import ExchangeAccount, ExecutionIntent
from execution.services.deribit_entry import DeribitEntryEngine
from execution.services.deribit_executor import DeribitExecutor, LegState
from execution.services.policy import get_policy
from execution.services.deribit_entry import EntryPlan

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Execute a signal on Deribit using greeks-aware entry engine"

    def add_arguments(self, parser):
        parser.add_argument("--date", type=str, help="Signal date (YYYY-MM-DD)")
        parser.add_argument("--latest", action="store_true", help="Use latest signal")
        parser.add_argument("--account", type=str, required=True, help="Deribit account name")
        parser.add_argument("--plan", action="store_true", help="Show entry plan only (no intent created)")
        parser.add_argument("--dry-run", action="store_true", help="Create intent but don't place orders")
        parser.add_argument("--spot", type=float, default=None, help="Override spot price")
        parser.add_argument("--force", action="store_true", help="Force even if intent exists")
        parser.add_argument(
            "--account-size", type=float, default=100_000,
            help="Account size in USD for tier sizing (default: 100000)"
        )
        parser.add_argument(
            "--market-orders", action="store_true",
            help="Use market orders instead of limit (not recommended for live)"
        )
        parser.add_argument(
            "--max-slippage", type=float, default=None,
            help="Max slippage %% for limit orders (default: from policy)"
        )

    def handle(self, *args, **options):
        # Load policy
        policy = get_policy()
        self.stdout.write(f"Policy: {policy.version}")
        
        # 1. Load signal
        signal = self._load_signal(options)
        self.stdout.write(
            f"\nSignal: {signal.date} | {signal.trade_decision} | "
            f"{signal.fusion_state} | size={signal.effective_size:.2f}"
        )

        # Check tradeable
        valid = ("CALL", "PUT", "OPTION_CALL", "OPTION_PUT", "TACTICAL_PUT",
                 "BULL_PROBE", "BEAR_PROBE", "IRON_CONDOR", "MVRV_SHORT")
        if signal.trade_decision.upper() not in valid:
            self.stdout.write(self.style.WARNING(
                f"Signal is {signal.trade_decision}, not executable"
            ))
            return

        # 2. Load account
        account = self._load_account(options["account"])
        self.stdout.write(f"Account: {account.name} ({account.exchange}, testnet={account.is_testnet})")

        # 3. Check existing intent
        if not options["force"] and not options["plan"]:
            existing = ExecutionIntent.objects.filter(
                signal=signal, account=account,
            ).exclude(status__in=["cancelled", "failed", "rejected"]).first()
            if existing:
                self.stdout.write(self.style.WARNING(
                    f"Intent exists: {existing.id} ({existing.status}). Use --force to override."
                ))
                return

        # 4. Build entry plan
        engine = DeribitEntryEngine(policy=policy)
        plan = engine.plan_entry(
            signal,
            account_size_usd=options["account_size"],
            spot_price=options.get("spot"),
        )

        if plan is None:
            self.stdout.write(self.style.WARNING("No entry plan generated (signal may be NO_TRADE or no instruments found)"))
            return

        # 5. Display plan
        self._print_plan(plan)

        if options["plan"]:
            return

        # 6. Execute
        executor = DeribitExecutor(account, policy=policy)
        dry_run = options.get("dry_run", False)
        use_limit = not options.get("market_orders", False)

        if dry_run:
            self.stdout.write(self.style.WARNING("\nDRY RUN — intent created, no orders placed"))
        
        if not use_limit:
            self.stdout.write(self.style.WARNING("⚠ Using MARKET orders (not recommended for live)"))

        result = executor.execute(
            plan, 
            signal, 
            dry_run=dry_run,
            use_limit_orders=use_limit,
            max_slippage_pct=options.get("max_slippage"),
        )

        # 7. Report
        self._print_result(result)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_signal(self, options) -> DailySignal:
        if options["latest"]:
            signal = DailySignal.objects.order_by("-date").first()
            if not signal:
                raise CommandError("No signals found")
            return signal
        elif options["date"]:
            try:
                d = date.fromisoformat(options["date"])
                return DailySignal.objects.get(date=d)
            except DailySignal.DoesNotExist:
                raise CommandError(f"No signal for {options['date']}")
        else:
            raise CommandError("Specify --date or --latest")

    def _load_account(self, name: str) -> ExchangeAccount:
        try:
            account = ExchangeAccount.objects.get(name=name)
        except ExchangeAccount.DoesNotExist:
            available = list(ExchangeAccount.objects.values_list("name", flat=True))
            raise CommandError(f"Account '{name}' not found. Available: {available}")

        if account.exchange != "deribit":
            raise CommandError(f"Account '{name}' is {account.exchange}, not deribit")

        return account

    def _print_plan(self, plan: EntryPlan):
        self.stdout.write(f"\n{'=' * 80}")
        self.stdout.write("ENTRY PLAN")
        self.stdout.write(f"{'=' * 80}")
        self.stdout.write(f"Decision:    {plan.trade_decision}")
        self.stdout.write(f"Direction:   {plan.direction}")
        self.stdout.write(f"Spot:        ${plan.spot_price:,.0f}")
        self.stdout.write(f"Risk budget: ${plan.total_risk_usd:,.0f}")
        self.stdout.write(f"Allocation:  {plan.naked_pct:.0%} naked / {plan.spread_pct:.0%} spread")
        if plan.iv_rank is not None:
            self.stdout.write(f"IV rank:     {plan.iv_rank:.0%}")
        self.stdout.write(f"Rationale:   {plan.rationale}")

        if plan.warnings:
            for w in plan.warnings:
                self.stdout.write(self.style.WARNING(f"  ⚠ {w}"))

        self.stdout.write(f"\n{'─' * 80}")
        self.stdout.write(f"{'Leg':12s} | {'Symbol':30s} | {'Side':4s} | {'Strike':>10s} | "
                          f"{'Delta':>6s} | {'IV':>6s} | {'Mid':>10s} | {'Spread':>6s} | {'DTE':>4s}")
        self.stdout.write(f"{'─' * 80}")

        for leg in plan.legs:
            delta_str = f"{leg.delta:+.3f}" if leg.delta else "  n/a"
            iv_str = f"{leg.iv:.1%}" if leg.iv else "  n/a"
            mid_str = f"{float(leg.mid_price):.4f}" if leg.mid_price else "     n/a"
            spread_str = f"{leg.spread_pct:.1%}" if leg.spread_pct else "  n/a"
            dte_str = f"{leg.dte:.0f}" if leg.dte else "n/a"

            self.stdout.write(
                f"{leg.leg_type.value:12s} | {leg.symbol:30s} | {leg.side:4s} | "
                f"${float(leg.strike):>9,.0f} | {delta_str:>6s} | {iv_str:>6s} | "
                f"{mid_str:>10s} | {spread_str:>6s} | {dte_str:>4s}"
            )

        # Estimate net cost for spreads/condors
        if len(plan.legs) >= 2:
            net = Decimal("0")
            for leg in plan.legs:
                if leg.mid_price:
                    if leg.side == "buy":
                        net -= leg.mid_price
                    else:
                        net += leg.mid_price

            label = "Net credit" if net > 0 else "Net debit"
            self.stdout.write(f"\n{label}: {float(net):.4f} BTC (${float(net) * plan.spot_price:,.0f} USD)")

    def _print_result(self, result):
        self.stdout.write(f"\n{'=' * 80}")
        self.stdout.write("EXECUTION RESULT")
        self.stdout.write(f"{'=' * 80}")

        if result.dry_run:
            self.stdout.write(self.style.WARNING("Mode: DRY RUN"))

        if result.success:
            self.stdout.write(self.style.SUCCESS("Status: SUCCESS"))
        else:
            self.stdout.write(self.style.ERROR("Status: FAILED"))

        if result.intent:
            self.stdout.write(f"Intent: {result.intent.id} ({result.intent.status})")
        
        # Show edge and cost analysis
        if result.edge_after_costs is not None:
            edge_style = self.style.SUCCESS if result.edge_after_costs > 0.02 else self.style.WARNING
            self.stdout.write(f"Edge after costs: {edge_style(f'{result.edge_after_costs:.1%}')}")
        if result.total_execution_cost is not None:
            self.stdout.write(f"Execution cost: {result.total_execution_cost:.1%}")

        # Show leg execution states
        if hasattr(result, 'leg_executions') and result.leg_executions:
            self.stdout.write(f"\nLeg Executions ({len(result.leg_executions)}):")
            for le in result.leg_executions:
                state_style = self.style.SUCCESS if le.state == LegState.FILLED else (
                    self.style.WARNING if le.state in (LegState.SUBMITTED, LegState.PARTIALLY_FILLED) 
                    else self.style.ERROR
                )
                self.stdout.write(
                    f"  {le.leg.leg_type.value:12s} | {le.leg.symbol:30s} | "
                    f"state={state_style(le.state.value)}"
                )
                if le.error:
                    self.stdout.write(self.style.ERROR(f"    error: {le.error}"))

        if result.orders:
            self.stdout.write(f"\nOrders ({len(result.orders)}):")
            for order in result.orders:
                status_style = self.style.SUCCESS if order.status == "filled" else self.style.WARNING
                self.stdout.write(
                    f"  {order.symbol} {order.side} {order.qty} | "
                    f"status={status_style(order.status)} | "
                    f"exchange_id={order.exchange_order_id or 'n/a'}"
                )
                if order.avg_fill_price:
                    self.stdout.write(f"    fill_price={order.avg_fill_price}")

        if result.errors:
            self.stdout.write(self.style.ERROR("\nErrors:"))
            for e in result.errors:
                self.stdout.write(self.style.ERROR(f"  • {e}"))

        if result.warnings:
            self.stdout.write(self.style.WARNING("\nWarnings:"))
            for w in result.warnings:
                self.stdout.write(self.style.WARNING(f"  • {w}"))
