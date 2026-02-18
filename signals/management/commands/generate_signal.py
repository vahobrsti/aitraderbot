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
            # Generate signal
            result = service.generate_signal(target_date)
            
            # Persist if requested
            if options["persist"]:
                signal = service.persist_signal(result)
            else:
                # Create a mock object with the same attributes for output
                from types import SimpleNamespace
                signal = SimpleNamespace(
                    date=result.date,
                    p_long=result.p_long,
                    p_short=result.p_short,
                    signal_option_call=result.signal_option_call,
                    signal_option_put=result.signal_option_put,
                    fusion_state=result.fusion_state,
                    fusion_confidence=result.fusion_confidence,
                    fusion_score=result.fusion_score,
                    trade_decision=result.trade_decision,
                    size_multiplier=result.size_multiplier,
                    tactical_put_active=result.tactical_put_active,
                    tactical_put_strategy=result.tactical_put_strategy,
                    no_trade_reasons=result.no_trade_reasons,
                    # Added missing fields for notification
                    option_structures=result.option_structures,
                    strike_guidance=result.strike_guidance,
                    dte_range=result.dte_range,
                    strategy_rationale=result.strategy_rationale,
                    score_components=result.score_components,
                    overlay_reason=result.overlay_reason,
                )
                self.stdout.write(self.style.WARNING("[DRY RUN] Signal not persisted"))
            
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
                if hasattr(signal, 'signal_option_call'):
                    self.stdout.write(f"Option Call:    {signal.signal_option_call}")
                    self.stdout.write(f"Option Put:     {signal.signal_option_put}")
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
        """Send Telegram notification for tradeable or vetoed signals."""
        # Check if overlay vetoed (fusion wanted to trade but overlay said no)
        no_trade_reasons = signal.no_trade_reasons or []
        is_overlay_veto = "OVERLAY_VETO" in no_trade_reasons
        
        # Check if option signal fired
        has_option_signal = getattr(signal, 'signal_option_call', 0) == 1 or getattr(signal, 'signal_option_put', 0) == 1
        is_option_trade = signal.trade_decision in ("OPTION_CALL", "OPTION_PUT")
        
        # Skip NO_TRADE unless it's a vetoed signal or an option signal fired
        if signal.trade_decision == "NO_TRADE" and not is_overlay_veto and not has_option_signal:
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

