"""
Send latest trading signal notification via Telegram.
Usage: python manage.py notify_signal [--verbose]
"""
from django.core.management.base import BaseCommand

from signals.models import DailySignal
from notifications.notifier import TelegramNotifier


class Command(BaseCommand):
    help = "Send latest trading signal to Telegram (if not NO_TRADE)"

    def add_arguments(self, parser):
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Print detailed output",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Send even if NO_TRADE (for testing)",
        )

    def handle(self, *args, **options):
        try:
            signal = DailySignal.objects.latest("date")
        except DailySignal.DoesNotExist:
            self.stderr.write(self.style.ERROR("No signals found in database"))
            return

        if options["verbose"]:
            self.stdout.write(f"Latest signal: {signal.date}")
            self.stdout.write(f"Decision: {signal.trade_decision}")
            self.stdout.write(f"Fusion State: {signal.fusion_state}")

        if signal.trade_decision == "NO_TRADE" and not options["force"]:
            self.stdout.write(
                self.style.WARNING(f"Skipping NO_TRADE signal for {signal.date}")
            )
            return

        try:
            notifier = TelegramNotifier()
            
            if options["force"] and signal.trade_decision == "NO_TRADE":
                # For testing, temporarily override the decision check
                from notifications.notifier import SignalMessage
                msg = SignalMessage(
                    date=str(signal.date),
                    trade_decision=signal.trade_decision,
                    fusion_state=signal.fusion_state,
                    fusion_score=signal.fusion_score,
                    fusion_confidence=signal.fusion_confidence,
                    p_long=signal.p_long,
                    p_short=signal.p_short,
                    size_multiplier=signal.size_multiplier,
                    option_structures=signal.option_structures,
                    strike_guidance=signal.strike_guidance,
                    dte_range=signal.dte_range,
                    strategy_rationale=signal.strategy_rationale,
                    tactical_put_active=signal.tactical_put_active,
                    tactical_put_strategy=signal.tactical_put_strategy,
                )
                # Bypass the NO_TRADE check by sending directly
                import asyncio
                message = notifier._format_message(msg)
                success = asyncio.run(notifier._send_async(message))
            else:
                success = notifier.send_from_model(signal)

            if success:
                self.stdout.write(
                    self.style.SUCCESS(f"✓ Sent notification for {signal.date}")
                )
            else:
                self.stderr.write(
                    self.style.ERROR(f"Failed to send notification")
                )

        except ValueError as e:
            self.stderr.write(
                self.style.ERROR(f"Configuration error: {e}")
            )
        except Exception as e:
            self.stderr.write(
                self.style.ERROR(f"Error: {e}")
            )
            raise
