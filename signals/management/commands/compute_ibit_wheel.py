"""
Compute the IBIT wheel selection for the latest (or a given) income signal and
persist it as an IbitWheelSetup row. Exposed via the API — this command does not
send Telegram messages and does not place orders.

For the date's income-gate signals (BULL_PUT_SPREAD / BEAR_CALL_SPREAD), it
selects the IBIT wheel short leg per risk tier (cash-secured put / covered call)
from collected IBIT option snapshots using the income-gate chain filters, and
upserts one IbitWheelSetup per (signal_date, trade_decision).

Usage:
    python manage.py compute_ibit_wheel --latest
    python manage.py compute_ibit_wheel --date 2026-08-08 --dte-mode income
    python manage.py compute_ibit_wheel --latest --dry-run
"""
from datetime import date, datetime, timedelta, timezone

import pandas as pd
from django.core.management.base import BaseCommand, CommandError

from datafeed.models import OptionSnapshot
from signals.income_gate import IncomeGateConfig, dedupe_chain_to_latest
from signals.models import DailySignal, IbitWheelSetup
from signals.wheel import (
    leg_to_dict,
    select_wheel_legs,
    selection_hash,
    wheel_side_for_decision,
)

INCOME_DECISIONS = ("BULL_PUT_SPREAD", "BEAR_CALL_SPREAD")


class Command(BaseCommand):
    help = "Compute and persist the IBIT wheel setup (retrieve via API)"

    def add_arguments(self, parser):
        parser.add_argument("--date", type=str, help="Signal date (YYYY-MM-DD)")
        parser.add_argument("--latest", action="store_true", help="Use latest signal date")
        parser.add_argument(
            "--dte-mode", default="income", choices=["income", "tactical"],
            help="DTE window for selection (default: income = 21-45d)",
        )
        parser.add_argument(
            "--staleness-hours", type=int, default=24,
            help="Max age of IBIT snapshots to use (default: 24)",
        )
        parser.add_argument("--dry-run", action="store_true", help="Print without saving")

    def handle(self, *args, **options):
        target = self._resolve_date(options)
        self.stdout.write(f"Date: {target}")

        signals = list(
            DailySignal.active()
            .filter(date=target, trade_decision__in=INCOME_DECISIONS)
            .order_by("trade_decision")
        )
        if not signals:
            self.stdout.write(self.style.WARNING(
                f"No income signals (BULL_PUT_SPREAD/BEAR_CALL_SPREAD) for {target}"
            ))
            return

        chain_df, spot = self._load_ibit_chain(options["staleness_hours"])
        if chain_df is None or chain_df.empty or not spot:
            self.stdout.write(self.style.ERROR(
                "No fresh IBIT option snapshots (exchange=ibkr). Run "
                "collect_options --exchange ibkr first (US market hours)."
            ))
            return

        self.stdout.write(f"IBIT spot: ${spot:,.2f} | chain rows: {len(chain_df)}")

        config = IncomeGateConfig()
        dte_mode = options["dte_mode"]
        dry_run = options["dry_run"]

        for signal in signals:
            side = wheel_side_for_decision(signal.trade_decision)
            if side is None:
                continue

            legs = select_wheel_legs(chain_df, side, spot, config=config, dte_mode=dte_mode)
            self.stdout.write(
                f"\n{signal.trade_decision}: {len(legs)} wheel leg(s) selected"
            )
            for leg in legs:
                self.stdout.write(
                    f"  [{leg.risk_tier}] {leg.position} ${leg.strike:g} "
                    f"Δ{abs(leg.delta):.2f} credit ${leg.credit:.2f} "
                    f"({leg.otm_pct*100:.1f}% OTM, {leg.dte}d)"
                )

            if not legs:
                # No qualifying contract — leave any prior setup untouched.
                continue

            new_hash = selection_hash(legs)
            leg_dicts = [leg_to_dict(l) for l in legs]

            if dry_run:
                continue

            existing = IbitWheelSetup.objects.filter(
                signal_date=target, trade_decision=signal.trade_decision
            ).first()
            if existing and existing.selection_hash == new_hash:
                self.stdout.write(f"  ↳ Unchanged — setup already current")
                continue

            IbitWheelSetup.objects.update_or_create(
                signal_date=target,
                trade_decision=signal.trade_decision,
                defaults={
                    "side": side,
                    "position": legs[0].position,
                    "spot_price": spot,
                    "dte_mode": dte_mode,
                    "legs": leg_dicts,
                    "selection_hash": new_hash,
                },
            )
            self.stdout.write(self.style.SUCCESS(f"  ✓ Saved setup for {signal.trade_decision}"))

        if dry_run:
            self.stdout.write(self.style.WARNING("\nDRY RUN — nothing saved"))

    # ------------------------------------------------------------------
    def _resolve_date(self, options) -> date:
        if options.get("date"):
            try:
                return date.fromisoformat(options["date"])
            except ValueError:
                raise CommandError(f"Invalid date: {options['date']}")
        if options.get("latest"):
            latest = DailySignal.active().order_by("-date").first()
            if not latest:
                raise CommandError("No active signals found")
            return latest.date
        raise CommandError("Specify --date or --latest")

    def _load_ibit_chain(self, staleness_hours: int):
        cutoff = datetime.now(timezone.utc) - timedelta(hours=staleness_hours)
        records = list(
            OptionSnapshot.objects.filter(
                exchange="ibkr", timestamp__gte=cutoff
            ).values(
                "symbol", "exchange", "timestamp", "expiry", "strike",
                "option_type", "delta", "bid", "ask", "dte", "spread_pct", "spot_price",
            )
        )
        if not records:
            return None, None

        df = pd.DataFrame.from_records(records)
        df = dedupe_chain_to_latest(df)
        for col in ("strike", "delta", "bid", "ask", "dte", "spread_pct", "spot_price"):
            df[col] = df[col].apply(lambda x: float(x) if x is not None else None)

        spot = None
        if not df.empty and df["spot_price"].notna().any():
            spot = float(df["spot_price"].dropna().iloc[0])
        return df, spot
