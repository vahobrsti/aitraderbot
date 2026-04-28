"""
Analyze price range during MVRV-60d flat-z periods.

Answers: "When MVRV-60d z-score stays flat for 7 days,
how far does price move (up and down) in that week?"

This directly informs iron condor wing safety.
"""
from django.core.management.base import BaseCommand
import pandas as pd
import numpy as np
from datafeed.models import RawDailyData


class Command(BaseCommand):
    help = "Analyze BTC price range during MVRV-60d flat-z periods"

    def add_arguments(self, parser):
        parser.add_argument(
            "--flat-threshold", type=float, default=0.5,
            help="Z-score abs threshold for 'flat' (default: 0.5)",
        )
        parser.add_argument(
            "--streak-days", type=int, default=7,
            help="Minimum consecutive flat days required (default: 7)",
        )
        parser.add_argument(
            "--forward-days", type=int, default=7,
            help="Forward window to measure price range (default: 7)",
        )

    def handle(self, *args, **options):
        flat_thresh = options["flat_threshold"]
        streak_days = options["streak_days"]
        fwd_days = options["forward_days"]

        # Load raw data
        qs = RawDailyData.objects.all().values(
            "date", "btc_close", "btc_high", "btc_low", "mvrv_usd_60d"
        )
        df = pd.DataFrame(list(qs)).sort_values("date").reset_index(drop=True)

        if df.empty or "mvrv_usd_60d" not in df.columns:
            self.stderr.write("No data or mvrv_usd_60d column missing.")
            return

        df = df.dropna(subset=["mvrv_usd_60d", "btc_close"]).reset_index(drop=True)

        # 1. Compute MVRV-60d delta z-score (same as mvrv_composite.py)
        mvrv_60d = df["mvrv_usd_60d"]
        delta_7d = mvrv_60d.diff(7)
        rolling_std = mvrv_60d.rolling(30, min_periods=10).std()
        delta_z = delta_7d / (rolling_std + 1e-9)
        df["delta_z"] = delta_z

        # 2. Flag flat days
        is_flat = (delta_z.abs() <= flat_thresh).astype(int)

        # 3. Consecutive flat streak
        streak = is_flat.groupby((is_flat != is_flat.shift()).cumsum()).cumsum()
        df["flat_streak"] = streak

        # 4. For each day with streak >= streak_days, compute forward price range
        #    Use only the FIRST day of each qualifying streak to avoid overlap
        qualified = df[streak >= streak_days].copy()

        # De-duplicate: only take first day of each streak block
        qualified["streak_block"] = (
            (qualified.index.to_series().diff() > 1).cumsum()
        )
        first_of_streak = qualified.groupby("streak_block").first().reset_index(drop=True)

        results = []
        for _, row in first_of_streak.iterrows():
            idx = df[df["date"] == row["date"]].index[0]
            fwd_slice = df.iloc[idx: idx + fwd_days + 1]

            if len(fwd_slice) < fwd_days:
                continue

            entry_price = row["btc_close"]
            max_price = fwd_slice["btc_high"].max()
            min_price = fwd_slice["btc_low"].min()

            max_up_pct = (max_price / entry_price - 1) * 100
            max_down_pct = (min_price / entry_price - 1) * 100
            total_range_pct = max_up_pct - max_down_pct

            results.append({
                "date": row["date"],
                "entry_price": entry_price,
                "mvrv_60d": row["mvrv_usd_60d"],
                "delta_z": row["delta_z"],
                "max_up_pct": round(max_up_pct, 2),
                "max_down_pct": round(max_down_pct, 2),
                "total_range_pct": round(total_range_pct, 2),
            })

        if not results:
            self.stdout.write("No qualifying flat-z periods found.")
            return

        res_df = pd.DataFrame(results)

        # 5. Summary statistics
        self.stdout.write(f"\n{'='*60}")
        self.stdout.write(f"MVRV-60d Flat-Z Price Range Analysis")
        self.stdout.write(f"{'='*60}")
        self.stdout.write(f"Flat threshold: |z| <= {flat_thresh}")
        self.stdout.write(f"Min streak: {streak_days} days")
        self.stdout.write(f"Forward window: {fwd_days} days")
        self.stdout.write(f"Qualifying periods: {len(res_df)}")
        self.stdout.write(f"{'='*60}\n")

        self.stdout.write("UPPER BAND (max upside during flat-z week):")
        self.stdout.write(f"  Mean:   +{res_df['max_up_pct'].mean():.2f}%")
        self.stdout.write(f"  Median: +{res_df['max_up_pct'].median():.2f}%")
        self.stdout.write(f"  P75:    +{res_df['max_up_pct'].quantile(0.75):.2f}%")
        self.stdout.write(f"  P90:    +{res_df['max_up_pct'].quantile(0.90):.2f}%")
        self.stdout.write(f"  P95:    +{res_df['max_up_pct'].quantile(0.95):.2f}%")
        self.stdout.write(f"  Max:    +{res_df['max_up_pct'].max():.2f}%")

        self.stdout.write(f"\nLOWER BAND (max downside during flat-z week):")
        self.stdout.write(f"  Mean:   {res_df['max_down_pct'].mean():.2f}%")
        self.stdout.write(f"  Median: {res_df['max_down_pct'].median():.2f}%")
        self.stdout.write(f"  P25:    {res_df['max_down_pct'].quantile(0.25):.2f}%")
        self.stdout.write(f"  P10:    {res_df['max_down_pct'].quantile(0.10):.2f}%")
        self.stdout.write(f"  P5:     {res_df['max_down_pct'].quantile(0.05):.2f}%")
        self.stdout.write(f"  Min:    {res_df['max_down_pct'].min():.2f}%")

        self.stdout.write(f"\nTOTAL RANGE (high-to-low spread):")
        self.stdout.write(f"  Mean:   {res_df['total_range_pct'].mean():.2f}%")
        self.stdout.write(f"  Median: {res_df['total_range_pct'].median():.2f}%")
        self.stdout.write(f"  P90:    {res_df['total_range_pct'].quantile(0.90):.2f}%")
        self.stdout.write(f"  P95:    {res_df['total_range_pct'].quantile(0.95):.2f}%")

        # 6. Iron condor safety assessment
        wings_pct = 10.0
        breaches_up = (res_df["max_up_pct"] > wings_pct).sum()
        breaches_down = (res_df["max_down_pct"] < -wings_pct).sum()
        breaches_any = ((res_df["max_up_pct"] > wings_pct) | (res_df["max_down_pct"] < -wings_pct)).sum()

        self.stdout.write(f"\nIRON CONDOR SAFETY (±{wings_pct}% wings):")
        self.stdout.write(f"  Periods breaching upper wing: {breaches_up}/{len(res_df)} ({breaches_up/len(res_df)*100:.1f}%)")
        self.stdout.write(f"  Periods breaching lower wing: {breaches_down}/{len(res_df)} ({breaches_down/len(res_df)*100:.1f}%)")
        self.stdout.write(f"  Periods breaching ANY wing:   {breaches_any}/{len(res_df)} ({breaches_any/len(res_df)*100:.1f}%)")
        self.stdout.write(f"  Safe rate (stayed in range):  {(len(res_df)-breaches_any)/len(res_df)*100:.1f}%")

        # 7. Segmented by MVRV level
        self.stdout.write(f"\n{'='*60}")
        self.stdout.write("SEGMENTED BY MVRV-60d LEVEL:")
        self.stdout.write(f"{'='*60}")

        bins = [0, 0.9, 1.0, 1.1, 1.3, float("inf")]
        labels = ["<0.9 (deep underval)", "0.9-1.0 (underval)", "1.0-1.1 (neutral)", "1.1-1.3 (profit)", ">1.3 (overheated)"]
        res_df["mvrv_bucket"] = pd.cut(res_df["mvrv_60d"], bins=bins, labels=labels)

        for bucket in labels:
            subset = res_df[res_df["mvrv_bucket"] == bucket]
            if subset.empty:
                continue
            safe = ((subset["max_up_pct"] <= wings_pct) & (subset["max_down_pct"] >= -wings_pct)).sum()
            self.stdout.write(
                f"  {bucket}: n={len(subset)}, "
                f"median range={subset['total_range_pct'].median():.1f}%, "
                f"safe rate={safe/len(subset)*100:.0f}%"
            )
