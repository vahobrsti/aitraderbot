# signals/management/commands/analyze_window_sweep.py
"""
Data-driven rolling-window selection for metric normalisation.

Sweeps candidate windows (e.g. 60d vs 90d) for each metric and scores each by
how well its z-score separates good-move days, plus stability. Use this to pick
the window before wiring a normalised reading into the engine snapshot.

Usage:
    python manage.py analyze_window_sweep
    python manage.py analyze_window_sweep --csv features_14d_4pct.csv
    python manage.py analyze_window_sweep --windows 60,90 --by-year
"""
from pathlib import Path

import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand

from signals.research.window_sweep import (
    METRIC_BASES,
    DEFAULT_WINDOWS,
    DEFAULT_LABELS,
    sweep,
    sweep_by_year,
)


class Command(BaseCommand):
    help = "Sweep rolling windows and score each by good-move separation + stability."

    def add_arguments(self, parser):
        parser.add_argument("--csv", type=str, default="features_14d_5pct.csv")
        parser.add_argument(
            "--windows", type=str, default=None,
            help="Comma-separated windows, e.g. '60,90,120' (default 30,60,90,120,180)",
        )
        parser.add_argument(
            "--metrics", type=str, default=None,
            help=f"Comma-separated metrics (default: {','.join(METRIC_BASES)})",
        )
        parser.add_argument(
            "--labels", type=str, default=None,
            help="Comma-separated label columns (default long+short)",
        )
        parser.add_argument(
            "--by-year", action="store_true",
            help="Also print per-year AUC for the recommended window/direction",
        )

    def handle(self, *args, **options):
        csv_path = Path(options["csv"])
        if not csv_path.exists():
            self.stderr.write(f"CSV not found: {csv_path}")
            return

        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        windows = (
            [int(w) for w in options["windows"].split(",")]
            if options["windows"] else DEFAULT_WINDOWS
        )
        metrics = (
            options["metrics"].split(",") if options["metrics"]
            else list(METRIC_BASES.keys())
        )
        labels = options["labels"].split(",") if options["labels"] else DEFAULT_LABELS

        self.stdout.write(f"\nLoaded {len(df)} rows from {csv_path}")
        self.stdout.write(f"Windows: {windows} | Labels: {labels}\n")

        results = sweep(df, metrics=metrics, windows=windows, labels=labels)
        if results.empty:
            self.stderr.write("No results — check metric base columns exist in CSV.")
            return

        label_shorts = [l.replace("label_good_move_", "") for l in labels]

        for metric in metrics:
            sub = results[results["metric"] == metric]
            if sub.empty:
                continue
            base = METRIC_BASES[metric]
            self.stdout.write("=" * 78)
            self.stdout.write(f"{metric}  (base column: {base})")
            self.stdout.write("=" * 78)

            header = f"  {'window':>6} | "
            for s in label_shorts:
                header += f"{'auc_' + s:>12} {'spread_' + s:>12} | "
            header += f"{'autocorr1':>10} {'flips/yr':>9}"
            self.stdout.write(header)
            self.stdout.write("  " + "-" * (len(header) - 2))

            for _, r in sub.iterrows():
                line = f"  {int(r['window']):>6} | "
                for s in label_shorts:
                    auc = r.get(f"auc_{s}", np.nan)
                    spread = r.get(f"spread_{s}", np.nan)
                    mono = r.get(f"mono_{s}", False)
                    mflag = "*" if mono else " "
                    line += f"{self._f(auc):>12} {self._f(spread) + mflag:>12} | "
                line += f"{self._f(r['autocorr1']):>10} {self._f(r['flips_per_yr'], 1):>9}"
                self.stdout.write(line)

            self._recommend(sub, label_shorts)
            self.stdout.write("")

            if options["by_year"]:
                self._by_year(df, sub, metric, labels, label_shorts)

        self.stdout.write("(* = hit rate monotonic across z quintiles)")
        self.stdout.write("\nDone.\n")

    def _recommend(self, sub: pd.DataFrame, label_shorts: list[str]):
        """Pick the window with the strongest directional AUC (|auc-0.5|)."""
        best_window = None
        best_strength = -1.0
        best_dir = None
        for _, r in sub.iterrows():
            for s in label_shorts:
                auc = r.get(f"auc_{s}", np.nan)
                if pd.isna(auc):
                    continue
                strength = abs(auc - 0.5)
                if strength > best_strength:
                    best_strength = strength
                    best_window = int(r["window"])
                    best_dir = s
        if best_window is not None:
            self.stdout.write(
                f"  → strongest: window={best_window} on '{best_dir}' "
                f"(|AUC-0.5|={best_strength:.3f}) — confirm it holds per year (--by-year)"
            )

    def _by_year(self, df, sub, metric, labels, label_shorts):
        # Primary direction = the one with the strongest overall AUC for this metric
        strengths = {}
        for s in label_shorts:
            col = f"auc_{s}"
            if col in sub.columns:
                strengths[s] = (sub[col] - 0.5).abs().max()
        if not strengths:
            return
        primary_short = max(strengths, key=strengths.get)
        primary_label = f"label_good_move_{primary_short}"
        # Recommended window = best for that direction
        col = f"auc_{primary_short}"
        best_row = sub.loc[(sub[col] - 0.5).abs().idxmax()]
        window = int(best_row["window"])

        self.stdout.write(
            f"\n  Per-year AUC for window={window}, direction='{primary_short}':"
        )
        yr = sweep_by_year(df, metric, window, primary_label)
        for _, r in yr.iterrows():
            self.stdout.write(
                f"    {int(r['year'])}: AUC={self._f(r['auc'])}  (n={int(r['n'])})"
            )
        aucs = yr["auc"].dropna()
        if len(aucs):
            self.stdout.write(
                f"    across years: mean={aucs.mean():.3f}  min={aucs.min():.3f}"
            )

    @staticmethod
    def _f(val, prec=3):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "N/A"
        return f"{val:.{prec}f}"
