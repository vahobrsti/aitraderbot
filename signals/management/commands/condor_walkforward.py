"""
Management command: Walk-forward validation for iron condor range gate.

Usage:
    python manage.py condor_walkforward
    python manage.py condor_walkforward --train-months 18 --test-months 6
    python manage.py condor_walkforward --min-precision 0.70 --out reports/condor_wf
"""

from django.core.management.base import BaseCommand
from signals.research.condor_walkforward import run_walkforward


class Command(BaseCommand):
    help = "Run walk-forward validation for iron condor range gate"

    def add_arguments(self, parser):
        parser.add_argument("--train-months", type=int, default=12, help="Training window (months)")
        parser.add_argument("--test-months", type=int, default=3, help="Test window (months)")
        parser.add_argument("--step-months", type=int, default=3, help="Step between folds (months)")
        parser.add_argument("--min-precision", type=float, default=0.65, help="Min precision target")
        parser.add_argument("--out", type=str, default="reports/condor_wf", help="Output directory")

    def handle(self, *args, **options):
        self.stdout.write(self.style.NOTICE("Running condor gate walk-forward validation..."))
        folds = run_walkforward(
            train_months=options["train_months"],
            test_months=options["test_months"],
            step_months=options["step_months"],
            min_precision=options["min_precision"],
            out_dir=options["out"],
        )
        if folds:
            self.stdout.write(self.style.SUCCESS(f"Done — {len(folds)} folds evaluated."))
        else:
            self.stdout.write(self.style.WARNING("No valid folds. Need more data."))
