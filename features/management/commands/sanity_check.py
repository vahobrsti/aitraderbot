# features/management/commands/sanity_check.py

from pathlib import Path

import pandas as pd
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Sanity check feature CSV: label distribution and option signal performance."

    def add_arguments(self, parser):
        parser.add_argument(
            "--csv",
            type=str,
            default="features_14d_5pct.csv",
            help="Path to the feature CSV file.",
        )

    def handle(self, *args, **options):
        csv_path = Path(options["csv"])

        if not csv_path.exists():
            self.stderr.write(self.style.ERROR(f"CSV file not found: {csv_path}"))
            return

        # --- Your original pandas code starts here ---
        df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")

        self.stdout.write(self.style.MIGRATE_HEADING("Basic info"))
        self.stdout.write(f"Shape: {df.shape}")
        self.stdout.write(f"Columns:\n{df.columns.tolist()}")

        self.stdout.write(self.style.MIGRATE_HEADING("Label head"))
        self.stdout.write(str(df[["label_good_move_long", "label_good_move_short"]].head()))

        self.stdout.write(self.style.MIGRATE_HEADING("Label means"))
        self.stdout.write(str(df[["label_good_move_long", "label_good_move_short"]].mean()))

        # Option signals
        call_days = df[df["signal_option_call"] == 1]
        put_days = df[df["signal_option_put"] == 1]

        self.stdout.write(self.style.MIGRATE_HEADING("Option signal stats"))

        if len(call_days) > 0:
            call_freq = len(call_days) / len(df)
            call_hit = call_days["label_good_move_long"].mean()
            self.stdout.write(f"Call signal frequency: {call_freq:.3f}")
            self.stdout.write(f"Call signal hit rate (long): {call_hit:.3f}")
        else:
            self.stdout.write("No call signal days found.")

        if len(put_days) > 0:
            put_freq = len(put_days) / len(df)
            put_hit = put_days["label_good_move_short"].mean()
            self.stdout.write(f"Put signal frequency: {put_freq:.3f}")
            self.stdout.write(f"Put signal hit rate (short): {put_hit:.3f}")
        else:
            self.stdout.write("No put signal days found.")
