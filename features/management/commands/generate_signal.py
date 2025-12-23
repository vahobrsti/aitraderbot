# features/management/commands/generate_signal.py
"""
Generate and persist daily trading signal.
Designed for cron/scheduled execution.
"""
from django.core.management.base import BaseCommand

from features.services import SignalService


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
                
        except Exception as e:
            self.stderr.write(self.style.ERROR(f"Error: {e}"))
            raise
