"""
Scan OptionSnapshot data to show what the entry engine would select.

This is a diagnostic tool — it doesn't create intents or place orders.
Use it to validate that the greeks-aware selection is picking sensible
contracts before running execute_deribit.

Usage:
    # Show best entries for latest signal
    python manage.py scan_entries --latest

    # Show all call candidates with ranking
    python manage.py scan_entries --type call --dte-min 10 --dte-max 21

    # Show condor legs for latest signal
    python manage.py scan_entries --latest --condor

    # Show snapshot data freshness
    python manage.py scan_entries --status
"""
import logging
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

from django.core.management.base import BaseCommand, CommandError
from django.db.models import Count, Max, Min, Avg

from datafeed.models import OptionSnapshot
from signals.models import DailySignal
from execution.services.deribit_entry import (
    DeribitEntryEngine, DELTA_TARGETS, PATH_DTE_TARGETS,
)

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Scan OptionSnapshot data and show entry candidates"

    def add_arguments(self, parser):
        parser.add_argument("--latest", action="store_true", help="Use latest signal for context")
        parser.add_argument("--date", type=str, help="Signal date (YYYY-MM-DD)")
        parser.add_argument("--signal-type", type=str, default=None, dest="signal_type", help="Trade decision type (e.g., IRON_CONDOR)")
        parser.add_argument("--type", type=str, choices=["call", "put"], help="Filter by option type")
        parser.add_argument("--dte-min", type=int, default=7, help="Min DTE (default: 7)")
        parser.add_argument("--dte-max", type=int, default=30, help="Max DTE (default: 30)")
        parser.add_argument("--condor", action="store_true", help="Show condor leg candidates")
        parser.add_argument("--status", action="store_true", help="Show snapshot data status only")
        parser.add_argument("--top", type=int, default=10, help="Number of candidates to show")
        parser.add_argument("--spot", type=float, default=None, help="Override spot price")

    def handle(self, *args, **options):
        if options["status"]:
            self._print_status()
            return

        engine = DeribitEntryEngine()

        # Load signal if requested
        signal = None
        if options["latest"]:
            qs = DailySignal.tradeable()
            if options.get("signal_type"):
                signal = qs.filter(
                    trade_decision=options["signal_type"].upper()
                ).order_by("-date").first()
            else:
                latest = qs.order_by("-date").first()
                if latest:
                    signal = DailySignal.pick_highest_priority(
                        DailySignal.tradeable().filter(date=latest.date)
                    )
            if signal:
                self.stdout.write(
                    f"Signal: {signal.date} | {signal.trade_decision} | "
                    f"{signal.fusion_state} | size={signal.effective_size:.2f}"
                )

        spot = options.get("spot") or engine._latest_spot()
        if not spot:
            raise CommandError("No spot price available. Use --spot or ensure snapshots exist.")

        self.stdout.write(f"Spot: ${spot:,.0f}")

        if options["condor"] and signal:
            self._print_condor_scan(engine, signal, spot)
            return

        if signal and not options["type"]:
            # Show full entry plan
            plan = engine.plan_entry(signal, spot_price=spot)
            if plan:
                self._print_plan_summary(plan)
            else:
                self.stdout.write(self.style.WARNING("No plan generated"))
            return

        # Generic scan
        option_type = options["type"] or "call"
        dte_min = options["dte_min"]
        dte_max = options["dte_max"]
        top_n = options["top"]

        self._print_candidates(engine, option_type, dte_min, dte_max, spot, top_n)

    def _print_status(self):
        """Show snapshot data freshness and coverage."""
        self.stdout.write(f"\n{'=' * 70}")
        self.stdout.write("OPTION SNAPSHOT STATUS")
        self.stdout.write(f"{'=' * 70}")

        total = OptionSnapshot.objects.count()
        self.stdout.write(f"Total snapshots: {total:,}")

        if total == 0:
            self.stdout.write(self.style.ERROR("No snapshots found. Run collect_options first."))
            return

        # By exchange
        by_exchange = (
            OptionSnapshot.objects
            .values("exchange")
            .annotate(
                count=Count("id"),
                latest=Max("timestamp"),
                earliest=Min("timestamp"),
            )
        )
        for row in by_exchange:
            self.stdout.write(
                f"\n  {row['exchange']}: {row['count']:,} snapshots | "
                f"{row['earliest']} → {row['latest']}"
            )

        # Freshness
        now = datetime.now(timezone.utc)
        cutoff_2h = now - timedelta(hours=2)
        fresh = OptionSnapshot.objects.filter(timestamp__gte=cutoff_2h).count()
        self.stdout.write(f"\n  Fresh (< 2h old): {fresh:,}")

        # By option type
        by_type = (
            OptionSnapshot.objects
            .filter(timestamp__gte=cutoff_2h)
            .values("option_type")
            .annotate(count=Count("id"))
        )
        for row in by_type:
            self.stdout.write(f"    {row['option_type']}: {row['count']:,}")

        # DTE distribution
        self.stdout.write(f"\n  DTE distribution (fresh snapshots):")
        for dte_label, dte_lo, dte_hi in [
            ("7-14d", 7, 14), ("14-21d", 14, 21), ("21-30d", 21, 30), ("30-60d", 30, 60)
        ]:
            count = OptionSnapshot.objects.filter(
                timestamp__gte=cutoff_2h, dte__gte=dte_lo, dte__lt=dte_hi
            ).count()
            self.stdout.write(f"    {dte_label}: {count:,}")

        # Greeks coverage
        with_greeks = OptionSnapshot.objects.filter(
            timestamp__gte=cutoff_2h, delta__isnull=False
        ).count()
        self.stdout.write(f"\n  With greeks: {with_greeks:,} / {fresh:,}")

    def _print_candidates(self, engine, option_type, dte_min, dte_max, spot, top_n):
        """Show ranked candidates for a given option type."""
        candidates = engine._query_candidates(option_type, dte_min, dte_max)

        self.stdout.write(f"\n{'=' * 100}")
        self.stdout.write(f"{option_type.upper()} CANDIDATES (DTE {dte_min}-{dte_max}, {len(candidates)} found)")
        self.stdout.write(f"{'=' * 100}")

        if not candidates:
            self.stdout.write(self.style.WARNING("No candidates found. Check snapshot freshness."))
            return

        # Sort by delta proximity to 0.50
        target_delta = 0.50
        candidates.sort(
            key=lambda s: abs(abs(float(s.delta or 0)) - target_delta)
        )

        self.stdout.write(
            f"{'Symbol':35s} | {'Strike':>10s} | {'Delta':>7s} | {'IV':>6s} | "
            f"{'Bid':>10s} | {'Ask':>10s} | {'Spread':>6s} | {'OI':>8s} | {'DTE':>5s} | {'Moneyness':>10s}"
        )
        self.stdout.write("─" * 120)

        for snap in candidates[:top_n]:
            delta_str = f"{float(snap.delta):+.4f}" if snap.delta else "   n/a"
            iv_str = f"{float(snap.iv):.1%}" if snap.iv else "  n/a"
            bid_str = f"{float(snap.bid):.4f}" if snap.bid else "     n/a"
            ask_str = f"{float(snap.ask):.4f}" if snap.ask else "     n/a"
            spread_str = f"{snap.spread_pct:.1%}" if snap.spread_pct else "  n/a"
            oi_str = f"{float(snap.open_interest):,.0f}" if snap.open_interest else "     n/a"
            dte_str = f"{snap.dte:.1f}" if snap.dte else "  n/a"
            money_str = f"{snap.moneyness:+.2%}" if snap.moneyness else "      n/a"

            self.stdout.write(
                f"{snap.symbol:35s} | ${float(snap.strike):>9,.0f} | {delta_str:>7s} | "
                f"{iv_str:>6s} | {bid_str:>10s} | {ask_str:>10s} | {spread_str:>6s} | "
                f"{oi_str:>8s} | {dte_str:>5s} | {money_str:>10s}"
            )

    def _print_condor_scan(self, engine, signal, spot):
        """Show iron condor leg candidates."""
        self.stdout.write(f"\n{'=' * 80}")
        self.stdout.write("IRON CONDOR SCAN")
        self.stdout.write(f"{'=' * 80}")

        target_call = signal.condor_short_call or spot * 1.10
        target_put = signal.condor_short_put or spot * 0.90

        self.stdout.write(f"Target short call: ${target_call:,.0f} (+{(target_call/spot - 1)*100:.1f}%)")
        self.stdout.write(f"Target short put:  ${target_put:,.0f} (-{(1 - target_put/spot)*100:.1f}%)")

        if signal.condor_strike_meta:
            meta = signal.condor_strike_meta
            self.stdout.write(f"MVRV drift: {meta.get('mvrv_drift', 'n/a')}")
            self.stdout.write(f"Call source: {meta.get('call_source', 'n/a')}")
            self.stdout.write(f"Put source: {meta.get('put_source', 'n/a')}")

        plan = engine.plan_entry(signal, spot_price=spot)
        if plan and plan.is_condor:
            self._print_plan_summary(plan)
        else:
            self.stdout.write(self.style.WARNING("Could not build condor plan"))

    def _print_plan_summary(self, plan):
        """Print a compact plan summary."""
        self.stdout.write(f"\n{'─' * 80}")
        self.stdout.write(f"PLAN: {plan.trade_decision} ({plan.direction})")
        self.stdout.write(f"Risk: ${plan.total_risk_usd:,.0f} | Spot: ${plan.spot_price:,.0f}")
        self.stdout.write(f"{'─' * 80}")

        self.stdout.write(
            f"{'Leg':12s} | {'Symbol':30s} | {'Side':4s} | {'Strike':>10s} | "
            f"{'Delta':>7s} | {'IV':>6s} | {'Mid':>10s} | {'DTE':>5s}"
        )
        self.stdout.write("─" * 95)

        for leg in plan.legs:
            delta_str = f"{leg.delta:+.3f}" if leg.delta else "   n/a"
            iv_str = f"{leg.iv:.1%}" if leg.iv else "  n/a"
            mid_str = f"{float(leg.mid_price):.4f}" if leg.mid_price else "     n/a"
            dte_str = f"{leg.dte:.0f}" if leg.dte else " n/a"

            self.stdout.write(
                f"{leg.leg_type.value:12s} | {leg.symbol:30s} | {leg.side:4s} | "
                f"${float(leg.strike):>9,.0f} | {delta_str:>7s} | {iv_str:>6s} | "
                f"{mid_str:>10s} | {dte_str:>5s}"
            )

        if plan.warnings:
            for w in plan.warnings:
                self.stdout.write(self.style.WARNING(f"  ⚠ {w}"))

        self.stdout.write(f"\n{plan.rationale}")
