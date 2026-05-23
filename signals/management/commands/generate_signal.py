# signals/management/commands/generate_signal.py
"""
Generate and persist trading signals.
Designed for hourly cron execution.

Evaluates all independent gates and persists each qualifying signal.
Only sends notifications when a signal is new or has changed.
"""
from django.core.management.base import BaseCommand

from signals.services import SignalService


class Command(BaseCommand):
    help = "Generate trading signals and save to database. Supports multiple signals per day."

    def add_arguments(self, parser):
        parser.add_argument(
            "--long_model",
            type=str,
            default="models/long_model.joblib",
            help="Path to long model",
        )
        parser.add_argument(
            "--short_model",
            type=str,
            default="models/short_model.joblib",
            help="Path to short model",
        )
        parser.add_argument(
            "--horizon",
            type=int,
            default=14,
            help="Horizon days for feature building",
        )
        parser.add_argument(
            "--target_return",
            type=float,
            default=0.05,
            help="Target return threshold",
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Print detailed output",
        )
        parser.add_argument(
            "--notify",
            action="store_true",
            help="Send Telegram notification (only on new/changed signals)",
        )
        parser.add_argument(
            "--date",
            type=str,
            default=None,
            help="Target date (YYYY-MM-DD) - defaults to latest available",
        )
        parser.add_argument(
            "--persist",
            action="store_true",
            default=True,
            help="Persist signal to database (default: True)",
        )
        parser.add_argument(
            "--no-persist",
            action="store_false",
            dest="persist",
            help="Don't persist signal to database (dry run)",
        )
        parser.add_argument(
            "--include-setup",
            action="store_true",
            default=True,
            dest="include_setup",
            help="Include trade setup in Telegram notification (default: True)",
        )
        parser.add_argument(
            "--no-setup",
            action="store_false",
            dest="include_setup",
            help="Don't include trade setup in Telegram notification",
        )

    def handle(self, *args, **options):
        from datetime import datetime

        service = SignalService(
            long_model_path=options["long_model"],
            short_model_path=options["short_model"],
            horizon_days=options["horizon"],
            target_return=options["target_return"],
        )

        # Parse optional date
        target_date = None
        if options["date"]:
            target_date = datetime.strptime(options["date"], "%Y-%m-%d").date()

        try:
            # Generate all qualifying signals
            results = service.generate_all_signals(target_date)

            if not options["persist"]:
                self.stdout.write(self.style.WARNING("[DRY RUN] Signals not persisted"))
                for r in results:
                    self.stdout.write(
                        f"  {r.date} | {r.fusion_state} | {r.trade_decision} | size={r.effective_size:.2f}"
                    )
                return

            # Persist and detect changes
            persisted = service.persist_all_signals(results)

            if not persisted:
                # All results were NO_TRADE (not persisted)
                self.stdout.write(
                    f"OK: {results[0].date} | {results[0].fusion_state} | NO_TRADE (no signals qualify)"
                )
                return

            # Output results
            for signal, changed in persisted:
                status = "NEW" if changed else "unchanged"
                if options["verbose"]:
                    self._print_verbose(signal)
                else:
                    self.stdout.write(
                        f"OK: {signal.date} | {signal.trade_decision} | "
                        f"{signal.fusion_state} | size={signal.effective_size:.2f} [{status}]"
                    )

                # Notify only on new/changed signals
                if options["notify"] and changed:
                    self._send_telegram_notification(
                        signal, options["verbose"], options["include_setup"]
                    )

        except Exception as e:
            self.stderr.write(self.style.ERROR(f"Error: {e}"))
            raise

    def _print_verbose(self, signal):
        """Print detailed signal info."""
        self.stdout.write(f"\n{'='*50}")
        self.stdout.write(f"Signal: {signal.date} | {signal.trade_decision}")
        self.stdout.write(f"{'='*50}")
        self.stdout.write(f"p_long:         {signal.p_long:.3f}")
        self.stdout.write(f"p_short:        {signal.p_short:.3f}")
        self.stdout.write(f"Fusion State:   {signal.fusion_state}")
        self.stdout.write(f"Fusion Score:   {signal.fusion_score:+d}")
        self.stdout.write(f"Size Mult:      {signal.size_multiplier:.2f}")
        self.stdout.write(f"Effective Size: {signal.effective_size:.2f}")
        if signal.tactical_put_active:
            self.stdout.write(f"Tactical Put:   {signal.tactical_put_strategy}")
        if signal.stop_loss:
            self.stdout.write(f"Stop Loss:      {signal.stop_loss}")
        self.stdout.write(f"{'='*50}\n")

    def _send_telegram_notification(self, signal, verbose: bool, include_setup: bool = True):
        """Send Telegram notification for new/changed tradeable signals."""
        if signal.trade_decision == "NO_TRADE":
            # Check if this is a vetoed NO_TRADE (worth notifying)
            no_trade_reasons = signal.no_trade_reasons or []
            if "OVERLAY_VETO" not in no_trade_reasons:
                if verbose:
                    self.stdout.write("Skipping Telegram notification (NO_TRADE)")
                return

        try:
            from notifications.notifier import TelegramNotifier
            notifier = TelegramNotifier()
            success = notifier.send_from_model(signal, include_setup=include_setup)

            if success:
                self.stdout.write(
                    self.style.SUCCESS(
                        f"✓ Telegram sent: {signal.trade_decision}"
                    )
                )
            else:
                self.stderr.write(
                    self.style.WARNING("Telegram notification failed")
                )
        except ValueError as e:
            self.stderr.write(
                self.style.WARNING(f"Telegram not configured: {e}")
            )
        except Exception as e:
            self.stderr.write(
                self.style.ERROR(f"Telegram error: {e}")
            )
