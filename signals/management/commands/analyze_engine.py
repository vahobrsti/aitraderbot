# signals/management/commands/analyze_engine.py
"""
Django command to print the essential engine metrics for a single date.

Mirrors the analyze_fusion command/endpoint pattern. Shows all canonical
buckets plus exchange flow balance, sentiment, mvrv_composite, mvrv_60d and
their z-scores for the requested (or latest) feature row.

Usage:
    python manage.py analyze_engine --explain
    python manage.py analyze_engine --explain --date 2024-11-20
"""
from pathlib import Path

import pandas as pd
from django.core.management.base import BaseCommand

from signals.engine_metrics import collect_essential_metrics


class Command(BaseCommand):
    help = "Print essential engine metrics (buckets, flow, sentiment, mvrv) for a date."

    def add_arguments(self, parser):
        parser.add_argument(
            "--csv",
            type=str,
            default="features_14d_5pct.csv",
            help="Input features CSV",
        )
        parser.add_argument(
            "--explain",
            action="store_true",
            help="Print the essential-metrics breakdown for the target date",
        )
        parser.add_argument(
            "--date",
            type=str,
            default=None,
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

        # Resolve target row
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

        metrics = collect_essential_metrics(row)
        self._print_metrics(date_str, metrics)

    # ── output ──────────────────────────────────────────────────────
    def _print_metrics(self, date_str: str, m: dict):
        def fmt(val, prec=4):
            return "N/A" if val is None else f"{val:.{prec}f}"

        fusion = m["fusion"]
        self.stdout.write("\n" + "=" * 70)
        self.stdout.write(f"ENGINE METRICS: {date_str} | {fusion['state'].upper()}")
        self.stdout.write("=" * 70)

        self.stdout.write(
            f"\nFUSION: score={fusion['score']:+d} | "
            f"confidence={fusion['confidence']} | "
            f"bear_mode={fusion['bear_mode']} | "
            f"cycle_day={fmt(fusion['cycle_day'], 0)}"
        )

        self.stdout.write("\n" + "-" * 70)
        self.stdout.write("BUCKETS")
        self.stdout.write("-" * 70)
        for name, val in m["buckets"].items():
            self.stdout.write(f"  {name:15s} {val}")

        self.stdout.write("\n" + "-" * 70)
        self.stdout.write("EXCHANGE FLOW BALANCE")
        self.stdout.write("-" * 70)
        ef = m["exchange_flow"]
        p = ef["pressure"]
        self.stdout.write(f"  >> {p['label']}  (z={fmt(p['value'], 2)})")
        self.stdout.write(f"  bucket                       {m['buckets']['exchange_flow']}")
        self.stdout.write(f"  flow_raw                     {fmt(ef['flow_raw'])}")
        for w in (2, 4, 7, 14, 21):
            self.stdout.write(f"  flow_sum_{w:<2}                   {fmt(ef[f'flow_sum_{w}'])}")
        self.stdout.write(f"  distribution_pressure_score  {fmt(ef['distribution_pressure_score'])}")
        self.stdout.write(f"  flow_pct_rank_180            {fmt(ef['flow_pct_rank_180'])}")
        self.stdout.write(f"  z: flow_z_90={fmt(ef['z_scores']['flow_z_90'], 2)}  flow_z_180={fmt(ef['z_scores']['flow_z_180'], 2)}")

        self.stdout.write("\n" + "-" * 70)
        self.stdout.write("SENTIMENT")
        self.stdout.write("-" * 70)
        s = m["sentiment"]
        self.stdout.write(f"  sentiment_norm               {fmt(s['sentiment_norm'], 2)}")
        self.stdout.write(f"  sentiment_roll_pct_180d      {fmt(s['sentiment_roll_pct_180d'])}")
        z = s["z_scores"]
        self.stdout.write(
            f"  z: 30d={fmt(z['sentiment_z_30d'], 2)}  90d={fmt(z['sentiment_z_90d'], 2)}  "
            f"180d={fmt(z['sentiment_z_180d'], 2)}  365d={fmt(z['sentiment_z_365d'], 2)}"
        )

        self.stdout.write("\n" + "-" * 70)
        self.stdout.write("MVRV COMPOSITE")
        self.stdout.write("-" * 70)
        mc = m["mvrv_composite"]
        self.stdout.write(f"  mvrv_composite_pct           {fmt(mc['mvrv_composite_pct'], 2)}")
        zc = mc["z_scores"]
        self.stdout.write(
            f"  z: 90d={fmt(zc['mvrv_comp_z_90d'], 2)}  180d={fmt(zc['mvrv_comp_z_180d'], 2)}  "
            f"365d={fmt(zc['mvrv_comp_z_365d'], 2)}"
        )

        self.stdout.write("\n" + "-" * 70)
        self.stdout.write("MVRV 60D")
        self.stdout.write("-" * 70)
        m60 = m["mvrv_60d"]
        self.stdout.write(f"  mvrv_60d                     {fmt(m60['mvrv_60d'], 3)}")
        self.stdout.write(f"  mvrv_60d_pct_rank            {fmt(m60['mvrv_60d_pct_rank'])}")
        self.stdout.write(f"  mvrv_60d_dist_from_max       {fmt(m60['mvrv_60d_dist_from_max'])}")
        self.stdout.write(
            f"  trend: falling={m60['is_falling']}  flattening={m60['is_flattening']}  rising={m60['is_rising']}"
        )

        self.stdout.write("\n" + "=" * 70)
        self.stdout.write("Done.\n")
