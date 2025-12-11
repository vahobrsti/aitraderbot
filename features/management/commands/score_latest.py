# features/management/commands/score_latest.py

from pathlib import Path

import joblib
import pandas as pd
from django.core.management.base import BaseCommand

from datafeed.models import RawDailyData
from features.feature_builder import build_features_and_labels_from_raw


class Command(BaseCommand):
    help = "Score the most recent day(s) using trained models and print suggested actions."

    def add_arguments(self, parser):
        parser.add_argument(
            "--days",
            type=int,
            default=120,
            help="How many recent days of raw data to pull for feature building.",
        )
        parser.add_argument(
            "--horizon",
            type=int,
            default=14,
        )
        parser.add_argument(
            "--target_return",
            type=float,
            default=0.05,
        )
        parser.add_argument(
            "--long_model",
            type=str,
            default="models/long_model.joblib",
        )
        parser.add_argument(
            "--short_model",
            type=str,
            default="models/short_model.joblib",
        )

    def handle(self, *args, **options):
        days = options["days"]
        horizon_days = options["horizon"]
        target_return = options["target_return"]

        long_model_path = Path(options["long_model"])
        short_model_path = Path(options["short_model"])

        # 1) Pull recent raw data
        qs = RawDailyData.objects.order_by("date")  # all rows, ascending
        df_raw = pd.DataFrame.from_records(qs.values())

        if df_raw.empty:
            self.stderr.write(self.style.ERROR("No RawDailyData found"))
            return

        # 2) Build features (this also builds labels, but we can ignore labels)
        feats = build_features_and_labels_from_raw(
            df_raw,
            horizon_days=horizon_days,
            target_return=target_return,
        )

        # Keep only the last row (most recent date with full horizon)
        latest_date = feats.index.max()
        latest_row = feats.loc[[latest_date]]

        # 3) Load models
        long_bundle = joblib.load(long_model_path)
        short_bundle = joblib.load(short_model_path)

        long_model = long_bundle["model"]
        long_feats = long_bundle["feature_names"]

        short_model = short_bundle["model"]
        short_feats = short_bundle["feature_names"]

        # 4) Score
        p_long = float(long_model.predict_proba(latest_row[long_feats])[:, 1][0])
        p_short = float(short_model.predict_proba(latest_row[short_feats])[:, 1][0])

        signal_option_call = int(latest_row["signal_option_call"].iloc[0])
        signal_option_put = int(latest_row["signal_option_put"].iloc[0])

        self.stdout.write(self.style.MIGRATE_HEADING(f"Date: {latest_date}"))
        self.stdout.write(f"p_long  = {p_long:.3f}")
        self.stdout.write(f"p_short = {p_short:.3f}")
        self.stdout.write(f"signal_option_call = {signal_option_call}")
        self.stdout.write(f"signal_option_put  = {signal_option_put}")

        # Simple decision logic example
        decision = "HOLD / NO TRADE"
        if p_long > 0.7 and signal_option_call == 1 and p_short < 0.5:
            decision = "BUY CALL (bullish setup)"
        elif p_short > 0.7 and signal_option_put == 1 and p_long < 0.5:
            decision = "BUY PUT (bearish setup)"

        self.stdout.write(self.style.SUCCESS(f"Decision: {decision}"))
