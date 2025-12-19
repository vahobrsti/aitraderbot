from django.core.management.base import BaseCommand
import pandas as pd
import numpy as np
from datafeed.models import RawDailyData
from features.feature_builder import build_features_and_labels_from_raw

class Command(BaseCommand):
    help = "Check if MVRV Delta Z thresholds are reasonable (target: 25-40% in trending buckets)"

    def handle(self, *args, **options):
        # 1. Fetch Data
        qs = RawDailyData.objects.order_by("date").values()
        df = pd.DataFrame.from_records(qs)

        if df.empty:
            self.stderr.write("No data found!")
            return

        # 2. Build Features
        feats = build_features_and_labels_from_raw(df)
        
        # 3. Analyze Buckets
        horizons = [2, 4, 7, 14]
        thresholds = [0.5, 0.8, 1.0, 1.2, 1.5]
        
        self.stdout.write("\n=== MVRV Long/Short Delta Z Threshold Validation ===")
        self.stdout.write("Target: 25-40% of days flagged as trending (|z| > threshold)\n")
        
        for h in horizons:
            col = f'mvrv_ls_delta_z_{h}d'
            if col not in feats.columns:
                continue
                
            series = feats[col].dropna()
            total = len(series)
            abs_series = series.abs()
            
            self.stdout.write(f"\n--- Horizon {h} Days (n={total}) ---")
            
            # Distribution stats
            median = abs_series.median()
            p75 = abs_series.quantile(0.75)
            p90 = abs_series.quantile(0.90)
            self.stdout.write(f"  |delta_z| distribution: median={median:.2f}, 75th={p75:.2f}, 90th={p90:.2f}")
            
            # Threshold sweep
            self.stdout.write("  Threshold sweep:")
            for thresh in thresholds:
                rising = (series > thresh).sum()
                falling = (series < -thresh).sum()
                pct_extreme = ((rising + falling) / total) * 100
                
                status = "✓" if 25 <= pct_extreme <= 40 else ("↑ too strict" if pct_extreme < 25 else "↓ too loose")
                self.stdout.write(f"    ±{thresh:.1f}: {pct_extreme:5.1f}% trending  {status}")
                
            self.stdout.write("")

        self.stdout.write("Done.\n")
