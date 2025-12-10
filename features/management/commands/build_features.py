# features/management/commands/build_features.py

from django.core.management.base import BaseCommand
from pathlib import Path
import pandas as pd

from datafeed.models import RawDailyData
from features.feature_builder import build_features_and_labels_from_raw


class Command(BaseCommand):
    help = "Build feature dataset from RawDailyData and save to CSV"

    def add_arguments(self, parser):
        parser.add_argument(
            "--out",
            type=str,
            default="features_14d_5pct.csv",
            help="Output CSV path",
        )
        parser.add_argument(
            "--horizon",
            type=int,
            default=14,
            help="Prediction horizon in days (e.g. 14 for ~2 weeks)",
        )
        parser.add_argument(
            "--target_return",
            type=float,
            default=0.05,
            help="Target return threshold (e.g. 0.05 = +5%)",
        )

    def handle(self, *args, **options):
        out_path = Path(options["out"])
        horizon = options["horizon"]
        target_return = options["target_return"]

        # Pull raw data from DB
        qs = RawDailyData.objects.order_by("date").values()
        df = pd.DataFrame.from_records(qs)

        if df.empty:
            self.stderr.write(self.style.ERROR("No RawDailyData rows found."))
            return

        # Build features + labels
        feats = build_features_and_labels_from_raw(
            df,
            horizon_days=horizon,
            target_return=target_return,
        )

        # Save to CSV
        feats.to_csv(out_path)
        self.stdout.write(self.style.SUCCESS(f"Features written to {out_path}"))
