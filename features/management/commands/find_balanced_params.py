# features/management/commands/find_balanced_params.py
"""
Find the combination of horizon (days) and target return (%) that produces
the most balanced long/short label distribution.

Tests horizons from 7 to 21 days and target returns from 5% to 10%.
"""

from django.core.management.base import BaseCommand
import pandas as pd

from datafeed.models import RawDailyData
from features.feature_builder import build_features_and_labels_from_raw


class Command(BaseCommand):
    help = "Find horizon/return combo with most balanced long/short labels"

    def add_arguments(self, parser):
        parser.add_argument(
            "--horizon-min", type=int, default=7, help="Min horizon days"
        )
        parser.add_argument(
            "--horizon-max", type=int, default=21, help="Max horizon days"
        )
        parser.add_argument(
            "--horizon-step", type=int, default=1, help="Horizon step size"
        )
        parser.add_argument(
            "--return-min", type=float, default=0.05, help="Min target return (e.g. 0.05)"
        )
        parser.add_argument(
            "--return-max", type=float, default=0.10, help="Max target return (e.g. 0.10)"
        )
        parser.add_argument(
            "--return-step", type=float, default=0.01, help="Return step size"
        )

    def handle(self, *args, **options):
        horizon_min = options["horizon_min"]
        horizon_max = options["horizon_max"]
        horizon_step = options["horizon_step"]
        return_min = options["return_min"]
        return_max = options["return_max"]
        return_step = options["return_step"]

        # Load raw data once
        self.stdout.write("Loading raw data from database...")
        qs = RawDailyData.objects.order_by("date").values()
        df = pd.DataFrame.from_records(qs)

        if df.empty:
            self.stderr.write(self.style.ERROR("No RawDailyData rows found."))
            return

        self.stdout.write(f"Loaded {len(df)} rows\n")

        results = []

        # Generate horizons and returns to test
        horizons = list(range(horizon_min, horizon_max + 1, horizon_step))
        returns = []
        r = return_min
        while r <= return_max + 0.0001:  # small epsilon for float comparison
            returns.append(round(r, 2))
            r += return_step

        total_combos = len(horizons) * len(returns)
        self.stdout.write(f"Testing {total_combos} combinations...\n")

        for i, horizon in enumerate(horizons):
            for j, target_return in enumerate(returns):
                combo_num = i * len(returns) + j + 1

                try:
                    feats = build_features_and_labels_from_raw(
                        df.copy(),
                        horizon_days=horizon,
                        target_return=target_return,
                    )

                    long_mean = feats["label_good_move_long"].mean()
                    short_mean = feats["label_good_move_short"].mean()
                    imbalance = abs(long_mean - short_mean)
                    total_signals = long_mean + short_mean

                    results.append({
                        "horizon": horizon,
                        "return_pct": int(target_return * 100),
                        "long_rate": long_mean,
                        "short_rate": short_mean,
                        "imbalance": imbalance,
                        "total_signals": total_signals,
                    })

                    self.stdout.write(
                        f"[{combo_num}/{total_combos}] "
                        f"H={horizon:2d}d R={int(target_return*100):2d}% | "
                        f"Long: {long_mean:.3f} Short: {short_mean:.3f} "
                        f"Imbalance: {imbalance:.3f}"
                    )

                except Exception as e:
                    self.stderr.write(f"Error for H={horizon}, R={target_return}: {e}")

        # Sort by imbalance (most balanced first)
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("imbalance")

        self.stdout.write("\n" + "=" * 70)
        self.stdout.write(self.style.SUCCESS("\nTOP 10 MOST BALANCED COMBINATIONS:"))
        self.stdout.write("=" * 70 + "\n")

        for idx, row in results_df.head(10).iterrows():
            self.stdout.write(
                f"  Horizon: {int(row['horizon']):2d} days | Return: {int(row['return_pct']):2d}% | "
                f"Long: {row['long_rate']:.3f} | Short: {row['short_rate']:.3f} | "
                f"Imbalance: {row['imbalance']:.4f}"
            )

        self.stdout.write("\n" + "=" * 70)
        self.stdout.write(self.style.WARNING("\nTOP 10 LEAST BALANCED (for comparison):"))
        self.stdout.write("=" * 70 + "\n")

        for idx, row in results_df.tail(10).iterrows():
            self.stdout.write(
                f"  Horizon: {int(row['horizon']):2d} days | Return: {int(row['return_pct']):2d}% | "
                f"Long: {row['long_rate']:.3f} | Short: {row['short_rate']:.3f} | "
                f"Imbalance: {row['imbalance']:.4f}"
            )

        # Best result summary
        best = results_df.iloc[0]
        self.stdout.write("\n" + "=" * 70)
        self.stdout.write(self.style.SUCCESS(
            f"\n🎯 BEST BALANCED: {int(best['horizon'])} days @ {int(best['return_pct'])}% return"
        ))
        self.stdout.write(
            f"   Long rate: {best['long_rate']:.3f} | Short rate: {best['short_rate']:.3f}"
        )
        self.stdout.write("=" * 70)
