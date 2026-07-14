# signals/management/commands/analyze_engine.py
"""
Django command to print the essential engine snapshot for a single date:
per-metric value / 90-day z-score / position label, plus a confluence score.

Usage:
    python manage.py analyze_engine --explain
    python manage.py analyze_engine --explain --date 2024-11-20
"""
from pathlib import Path

import pandas as pd
from django.core.management.base import BaseCommand

from signals.engine_metrics import (
    collect_essential_metrics,
    ensure_normalized_columns,
)


class Command(BaseCommand):
    help = "Print the engine snapshot (value/z/label per metric + confluence score)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--csv", type=str, default="features_14d_5pct.csv",
            help="Input features CSV",
        )
        parser.add_argument(
            "--explain", action="store_true",
            help="Print the snapshot for the target date",
        )
        parser.add_argument(
            "--date", type=str, default=None,
            help="Target date (YYYY-MM-DD), defaults to the latest row",
        )

    def handle(self, *args, **options):
        csv_path = Path(options["csv"])
        if not csv_path.exists():
            self.stderr.write(f"CSV not found: {csv_path}")
            return

        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        if len(df) == 0:
            self.stderr.write("No rows in feature CSV. Nothing to analyze.")
            return
        df = ensure_normalized_columns(df)

        target_date = options.get("date")
        if target_date:
            df.index = pd.to_datetime(df.index)
            matching = [idx for idx in df.index if str(idx)[:10] == target_date]
            if not matching:
                self.stderr.write(f"Date {target_date} not found in data")
                return
            row = df.loc[matching[0]]
            date_str = str(matching[0])[:10]
        else:
            row = df.iloc[-1]
            date_str = str(df.index[-1])[:10]

        self._print(date_str, collect_essential_metrics(row))

    def _print(self, date_str: str, m: dict):
        def fmt(val, prec=2):
            return "N/A" if val is None else f"{val:.{prec}f}"

        fusion = m["fusion"]
        self.stdout.write("\n" + "=" * 60)
        self.stdout.write(f"ENGINE METRICS: {date_str} | {fusion['state'].upper()}")
        self.stdout.write("=" * 60)
        self.stdout.write(
            f"fusion: {fusion['state']} (score {fusion['score']:+d}, "
            f"{fusion['confidence']}, bear_mode={fusion['bear_mode']})"
        )

        self.stdout.write("\n" + "-" * 60)
        self.stdout.write(f"{'metric':<15}{'value':>14}{'z(90d)':>9}   label")
        self.stdout.write("-" * 60)
        for name, d in m["metrics"].items():
            self.stdout.write(
                f"{name:<15}{fmt(d['value'], 3):>14}{fmt(d['z_90']):>9}   {d['label']}"
            )

        fc = m["fusion_components"]
        self.stdout.write("\n" + "-" * 60)
        self.stdout.write("FUSION COMPONENTS (oriented bullish+, per-horizon -> sum)")
        self.stdout.write("-" * 60)
        mdia_h = "  ".join(f"{k}={v:+d}" for k, v in fc["mdia"]["horizons"].items())
        self.stdout.write(f"  mdia     {mdia_h}   sum={fc['mdia']['sum']:+.0f}")
        self.stdout.write(
            f"  whale    mega={fc['whale']['mega_sum']:+.0f}  "
            f"small={fc['whale']['small_sum']:+.0f}   sum={fc['whale']['sum']:+.0f}"
        )
        mvrv_h = "  ".join(f"{k}={v:+d}" for k, v in fc["mvrv_ls"]["horizons"].items())
        self.stdout.write(f"  mvrv_ls  {mvrv_h}   sum={fc['mvrv_ls']['sum']:+.0f}")

        s = m["score"]
        self.stdout.write("\n" + "-" * 60)
        self.stdout.write(f"ENGINE SCORE: {s['value']:+d}  ({s['direction'].upper()})")
        self.stdout.write("-" * 60)
        self.stdout.write(
            f"  net={s['net']:+.2f}  agreement={s['agreement']:.0%}  "
            f"active={s['active']}/{len(s['contributions'])}"
        )
        c = s["contributions"]
        norm = ["mvrv_60d", "sentiment", "exchange_flow", "mvrv_composite"]
        fus = ["mdia", "whale", "mvrv_ls"]
        self.stdout.write(
            "  metrics:  " + "  ".join(f"{k}={c[k]:+.2f}" for k in norm)
        )
        self.stdout.write(
            "  fusion:   " + "  ".join(f"{k}={c[k]:+.2f}" for k in fus)
        )

        self.stdout.write("\n" + "=" * 60 + "\nDone.\n")
