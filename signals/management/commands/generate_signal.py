# signals/management/commands/generate_signal.py
"""
Generate and persist daily trading signal.
Designed for cron/scheduled execution.
"""
from django.core.management.base import BaseCommand

from signals.services import SignalService


class Command(BaseCommand):
    help = "Generate today's trading signal and save to database."

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
            help="Send Telegram notification (only for non-NO_TRADE signals)",
        )

    def handle(self, *args, **options):
        service = SignalService(
            long_model_path=options["long_model"],
            short_model_path=options["short_model"],
            horizon_days=options["horizon"],
            target_return=options["target_return"],
        )

        try:
            signal = service.generate_and_persist()
            
            if options["verbose"]:
                self.stdout.write(f"\n{'='*50}")
                self.stdout.write(f"Signal generated for: {signal.date}")
                self.stdout.write(f"{'='*50}")
                self.stdout.write(f"p_long:         {signal.p_long:.3f}")
                self.stdout.write(f"p_short:        {signal.p_short:.3f}")
                self.stdout.write(f"Fusion State:   {signal.fusion_state}")
                self.stdout.write(f"Fusion Score:   {signal.fusion_score:+d}")
                self.stdout.write(f"Decision:       {signal.trade_decision}")
                self.stdout.write(f"Size Mult:      {signal.size_multiplier:.2f}")
                if signal.tactical_put_active:
                    self.stdout.write(f"Tactical Put:   {signal.tactical_put_strategy}")
                self.stdout.write(f"{'='*50}\n")
            else:
                # Minimal output for cron
                self.stdout.write(
                    f"OK: {signal.date} | {signal.fusion_state} | {signal.trade_decision}"
                )
            
            # Send Telegram notification if requested
            if options["notify"]:
                self._send_telegram_notification(signal, options["verbose"])
                
        except Exception as e:
            self.stderr.write(self.style.ERROR(f"Error: {e}"))
            raise

    def _send_telegram_notification(self, signal, verbose: bool):
        """Send Telegram notification for non-NO_TRADE signals."""
        if signal.trade_decision == "NO_TRADE":
            if verbose:
                self.stdout.write("Skipping Telegram notification (NO_TRADE)")
            return
        
        try:
            from notifications.notifier import TelegramNotifier
            notifier = TelegramNotifier()
            success = notifier.send_from_model(signal)
            
            if success:
                self.stdout.write(
                    self.style.SUCCESS(f"✓ Telegram notification sent")
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

