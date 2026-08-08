"""
Publish the IBIT wheel selection to Telegram, alongside the BTC option setup.

For the latest (or a given) date, this finds the income-gate signals
(BULL_PUT_SPREAD / BEAR_CALL_SPREAD), selects the IBIT wheel short leg for each
(cash-secured put / covered call) from collected IBIT option snapshots using the
same income-gate chain-layer filters, and sends the result to the Telegram
channel together with the BTC income-spread setups already stored on the signal.

This publishes a signal only; it places no orders.

Usage:
    python manage.py publish_ibit_wheel --latest
    python manage.py publish_ibit_wheel --date 2026-08-07 --dte-mode income
    python manage.py publish_ibit_wheel --latest --dry-run
"""
from datetime import date, datetime, timedelta, timezone

import pandas as pd
from django.core.management.base import BaseCommand, CommandError

from datafeed.models import OptionSnapshot
from notifications.models import WheelPublication
from signals.income_gate import IncomeGateConfig, dedupe_chain_to_latest
from signals.models import DailySignal
from signals.wheel import select_wheel_legs, selection_hash, wheel_side_for_decision

INCOME_DECISIONS = ("BULL_PUT_SPREAD", "BEAR_CALL_SPREAD")


class Command(BaseCommand):
    help = "Publish IBIT wheel selection (+ BTC setup) to Telegram"

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
        parser.add_argument("--dry-run", action="store_true", help="Print instead of sending")
        parser.add_argument(
            "--force", action="store_true",
            help="Re-publish even if an identical selection was already sent",
        )

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
        if chain_df is None or chain_df.empty:
            self.stdout.write(self.style.ERROR(
                "No fresh IBIT option snapshots (exchange=ibkr). Run collect_options "
                "--exchange ibkr first."
            ))
            return
        if not spot:
            self.stdout.write(self.style.ERROR("Could not resolve IBIT spot from snapshots."))
            return

        self.stdout.write(f"IBIT spot: ${spot:,.2f} | chain rows: {len(chain_df)}")

        config = IncomeGateConfig()
        dry_run = options["dry_run"]
        notifier = None
        if not dry_run:
            from notifications.notifier import TelegramNotifier
            notifier = TelegramNotifier()

        sent_any = False
        for signal in signals:
            side = wheel_side_for_decision(signal.trade_decision)
            if side is None:
                continue

            legs = select_wheel_legs(
                chain_df, side, spot, config=config, dte_mode=options["dte_mode"],
            )
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
                continue

            if dry_run:
                sent_any = True
                continue

            # Dedup: skip if the identical selection was already published for
            # this (date, decision). The hourly market-hours cron would otherwise
            # resend the same alert every run. A changed selection (different
            # strike/expiry/tier) produces a new hash and is re-published.
            new_hash = selection_hash(legs)
            if not options["force"]:
                prior = WheelPublication.objects.filter(
                    signal_date=target, trade_decision=signal.trade_decision
                ).first()
                if prior and prior.selection_hash == new_hash:
                    self.stdout.write(
                        f"  ↳ Already published (unchanged) — skipping {signal.trade_decision}"
                    )
                    continue

            ok = notifier.send_ibit_wheel(
                signal_date=str(target),
                side=side,
                spot_price=spot,
                legs=legs,
                btc_signal_type=signal.trade_decision,
                btc_score=signal.income_spread_score,
                btc_setups=signal.income_spread_setups or [],
            )
            sent_any = sent_any or ok
            if ok:
                # Record only on success so a failed send retries next run.
                WheelPublication.objects.update_or_create(
                    signal_date=target,
                    trade_decision=signal.trade_decision,
                    defaults={"selection_hash": new_hash},
                )
                self.stdout.write(self.style.SUCCESS(f"  ✓ Sent {signal.trade_decision}"))
            else:
                self.stderr.write(self.style.ERROR(f"  Failed to send {signal.trade_decision}"))

        if dry_run:
            self.stdout.write(self.style.WARNING("\nDRY RUN — nothing sent"))
        elif not sent_any:
            self.stdout.write(self.style.WARNING("Nothing published (no qualifying legs)."))

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
