# ml/management/commands/score_dataset.py
"""
Score the full feature dataset using trained ML models.
Adds p_long, p_short probabilities and ml_*_signal columns.
"""

from pathlib import Path

import joblib
import pandas as pd
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Score full feature dataset with trained models and output probabilities"

    def add_arguments(self, parser):
        parser.add_argument(
            "--features_csv",
            type=str,
            default="features_14d_5pct.csv",
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
        parser.add_argument(
            "--out",
            type=str,
            default="features_scored.csv",
        )

    def handle(self, *args, **options):
        features_csv = Path(options["features_csv"])
        long_model_path = Path(options["long_model"])
        short_model_path = Path(options["short_model"])
        out_path = Path(options["out"])

        df = pd.read_csv(features_csv, parse_dates=["date"]).set_index("date")

        # Load models
        long_bundle = joblib.load(long_model_path)
        short_bundle = joblib.load(short_model_path)

        long_model = long_bundle["model"]
        long_feats = long_bundle["feature_names"]

        short_model = short_bundle["model"]
        short_feats = short_bundle["feature_names"]

        # Score
        df["p_long"] = long_model.predict_proba(df[long_feats])[:, 1]
        df["p_short"] = short_model.predict_proba(df[short_feats])[:, 1]

        # Optional: raw recommendation columns
        df["ml_long_signal"] = (df["p_long"] > 0.7).astype(int)
        df["ml_short_signal"] = (df["p_short"] > 0.7).astype(int)

        df.to_csv(out_path)
        self.stdout.write(self.style.SUCCESS(f"Wrote scored dataset to {out_path}"))
