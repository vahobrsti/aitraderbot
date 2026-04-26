"""
Management command: Run chop/range analysis for iron condor entry filtering.

Usage:
    python manage.py chop_analysis
    python manage.py chop_analysis --out reports/chop
"""

from django.core.management.base import BaseCommand
from signals.research.chop_analysis import run_full_analysis


class Command(BaseCommand):
    help = "Analyze chop conditions vs 5-day range outcomes for iron condor filtering"

    def add_arguments(self, parser):
        parser.add_argument(
            "--out",
            type=str,
            default="reports/chop",
            help="Output directory for CSV reports (default: reports/chop)",
        )

    def handle(self, *args, **options):
        out_dir = options["out"]
        self.stdout.write(self.style.NOTICE(f"Running chop analysis, output → {out_dir}"))
        results = run_full_analysis(out_dir=out_dir)
        self.stdout.write(self.style.SUCCESS("Done."))
