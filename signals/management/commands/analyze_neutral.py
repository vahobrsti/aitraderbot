# signals/management/commands/analyze_neutral.py
"""
MVRV-LS NEUTRAL Terrain Analysis

Purpose: Answer the key question:
"During MVRV-LS = NEUTRAL, were returns actually untradeable — or just unacknowledged?"

This command splits NEUTRAL days into subgroups and computes forward returns
to identify if any subgroup deserves a new trading lane.

Subgroups analyzed:
- rising_count >= 1 (improving terrain)
- falling_count >= 1 (deteriorating terrain)
- conflict = 1 (mixed signals)
- pure neutral (all zeros)
"""

from django.core.management.base import BaseCommand
from pathlib import Path
import pandas as pd
import numpy as np

from signals.fusion import fuse_signals, MarketState


class Command(BaseCommand):
    help = "Analyze forward returns within MVRV-LS NEUTRAL terrain"

    def add_arguments(self, parser):
        parser.add_argument(
            "--csv",
            type=str,
            default="features_14d_5pct.csv",
            help="Input features CSV",
        )
        parser.add_argument(
            "--year",
            type=int,
            default=None,
            help="Filter to specific year",
        )
        parser.add_argument(
            "--horizon",
            type=int,
            default=14,
            help="Forward return horizon in days",
        )
        parser.add_argument(
            "--threshold",
            type=float,
            default=0.05,
            help="Return threshold for hit rate (e.g., 0.05 = 5%)",
        )

    def handle(self, *args, **options):
        csv_path = Path(options["csv"])
        year = options.get("year")
        horizon = options["horizon"]
        threshold = options["threshold"]

        if not csv_path.exists():
            self.stderr.write(f"CSV not found: {csv_path}")
            return

        # Load features
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        self.stdout.write(f"\nLoaded {len(df)} rows from {csv_path}")
        
        if year:
            df = df.loc[f'{year}-01-01':f'{year}-12-31']
            self.stdout.write(f"Filtered to {year}: {len(df)} rows")

        # === IDENTIFY MVRV-LS STATUS FOR EACH DAY ===
        self.stdout.write("\n" + "=" * 80)
        self.stdout.write("STEP 1: Classifying MVRV-LS Status")
        self.stdout.write("=" * 80)
        
        results = []
        for idx, row in df.iterrows():
            fusion = fuse_signals(row)
            
            # Derive LS status
            ls_permissive = (
                row.get('mvrv_ls_regime_call_confirm', 0) == 1 or
                row.get('mvrv_ls_regime_call_confirm_recovery', 0) == 1 or
                row.get('mvrv_ls_weak_uptrend', 0) == 1
            )
            ls_bearish = (
                row.get('mvrv_ls_regime_distribution_warning', 0) == 1 or
                row.get('mvrv_ls_regime_put_confirm', 0) == 1
            )
            
            if ls_permissive:
                ls_status = 'PERMISSIVE'
            elif ls_bearish:
                ls_status = 'BEARISH'
            else:
                ls_status = 'NEUTRAL'
            
            # Get NEUTRAL subgroup indicators
            rising_count = int(row.get('mvrv_ls_rising_count', 0))
            falling_count = int(row.get('mvrv_ls_falling_count', 0))
            conflicts = 1 if 'conflicts' in str(fusion.components) else 0
            
            # Determine subgroup
            if ls_status == 'NEUTRAL':
                if rising_count >= 1:
                    subgroup = 'NEUTRAL-IMPROVING'
                elif falling_count >= 1:
                    subgroup = 'NEUTRAL-DETERIORATING'
                elif conflicts == 1:
                    subgroup = 'NEUTRAL-CONFLICTED'
                else:
                    subgroup = 'NEUTRAL-FLAT'
            else:
                subgroup = ls_status
            
            results.append({
                'date': idx,
                'ls_status': ls_status,
                'subgroup': subgroup,
                'fusion_state': fusion.state.value,
                'score': fusion.score,
                'rising_count': rising_count,
                'falling_count': falling_count,
                'close': row.get('close', np.nan),
            })
        
        analysis_df = pd.DataFrame(results)
        
        # Status distribution
        status_counts = analysis_df['ls_status'].value_counts()
        self.stdout.write("\nMVRV-LS Status Distribution:")
        for status, count in status_counts.items():
            pct = count / len(analysis_df) * 100
            self.stdout.write(f"  {status:15s} {count:4d} ({pct:5.1f}%)")
        
        # === ANALYZE NEUTRAL SUBGROUPS ===
        self.stdout.write("\n" + "=" * 80)
        self.stdout.write("STEP 2: NEUTRAL Subgroup Distribution")
        self.stdout.write("=" * 80)
        
        neutral_df = analysis_df[analysis_df['ls_status'] == 'NEUTRAL']
        subgroup_counts = neutral_df['subgroup'].value_counts()
        
        self.stdout.write(f"\nTotal NEUTRAL days: {len(neutral_df)}")
        for subgroup, count in subgroup_counts.items():
            pct = count / len(neutral_df) * 100
            self.stdout.write(f"  {subgroup:25s} {count:4d} ({pct:5.1f}%)")
        
        # === COMPUTE FORWARD RETURNS ===
        self.stdout.write("\n" + "=" * 80)
        self.stdout.write(f"STEP 3: Forward Returns Analysis ({horizon}d, threshold={threshold*100:.0f}%)")
        self.stdout.write("=" * 80)
        
        # Get price data from database
        from datafeed.models import RawDailyData
        price_qs = RawDailyData.objects.order_by('date').values('date', 'btc_close')
        price_df = pd.DataFrame.from_records(price_qs)
        
        if price_df.empty or 'btc_close' not in price_df.columns:
            self.stderr.write("WARNING: No price data found in database")
            return
        
        price_df['date'] = pd.to_datetime(price_df['date'])
        price_df = price_df.set_index('date').sort_index()
        
        # Compute forward returns
        price_df['fwd_return'] = price_df['btc_close'].shift(-horizon) / price_df['btc_close'] - 1
        
        # Compute max/min over horizon window
        fwd_max_list = []
        fwd_min_list = []
        for i in range(len(price_df)):
            if i + horizon < len(price_df):
                window = price_df['btc_close'].iloc[i+1:i+horizon+1]
                fwd_max_list.append(window.max() / price_df['btc_close'].iloc[i] - 1)
                fwd_min_list.append(window.min() / price_df['btc_close'].iloc[i] - 1)
            else:
                fwd_max_list.append(np.nan)
                fwd_min_list.append(np.nan)
        
        price_df['fwd_max'] = fwd_max_list
        price_df['fwd_min'] = fwd_min_list
        
        # Merge with analysis
        analysis_df['date_idx'] = pd.to_datetime(analysis_df['date'])
        analysis_df = analysis_df.merge(
            price_df[['fwd_return', 'fwd_max', 'fwd_min']].reset_index(),
            left_on='date_idx',
            right_on='date',
            how='left'
        )
        
        # === COMPUTE METRICS BY SUBGROUP ===
        self.stdout.write("\n📊 FORWARD RETURNS BY MVRV-LS STATUS")
        self.stdout.write("-" * 80)
        self.stdout.write(f"{'Subgroup':25s} | {'N':5s} | {'Avg Ret':8s} | {'Median':8s} | {'Hit>+{:.0f}%':8s} | {'Max DD':8s}".format(threshold*100))
        self.stdout.write("-" * 80)
        
        for subgroup in ['PERMISSIVE', 'NEUTRAL-IMPROVING', 'NEUTRAL-FLAT', 'NEUTRAL-DETERIORATING', 'NEUTRAL-CONFLICTED', 'BEARISH']:
            sub_df = analysis_df[analysis_df['subgroup'] == subgroup].dropna(subset=['fwd_return'])
            
            if len(sub_df) == 0:
                continue
            
            avg_ret = sub_df['fwd_return'].mean() * 100
            median_ret = sub_df['fwd_return'].median() * 100
            hit_rate = (sub_df['fwd_max'] >= threshold).mean() * 100
            max_dd = sub_df['fwd_min'].min() * 100
            
            # Highlight promising subgroups
            flag = ""
            if avg_ret > 2 and hit_rate > 50:
                flag = "⚡ PROMISING"
            elif avg_ret < -2:
                flag = "⚠️ BEARISH"
            
            self.stdout.write(
                f"{subgroup:25s} | {len(sub_df):5d} | {avg_ret:+7.2f}% | {median_ret:+7.2f}% | "
                f"{hit_rate:7.1f}% | {max_dd:+7.2f}% {flag}"
            )
        
        # === DETAILED NEUTRAL ANALYSIS ===
        self.stdout.write("\n" + "=" * 80)
        self.stdout.write("STEP 4: Conditional Analysis Within NEUTRAL")
        self.stdout.write("=" * 80)
        
        neutral_with_returns = analysis_df[
            (analysis_df['ls_status'] == 'NEUTRAL') & 
            (analysis_df['fwd_return'].notna())
        ]
        
        # By rising_count
        self.stdout.write("\n📊 NEUTRAL by rising_count:")
        for rc in sorted(neutral_with_returns['rising_count'].unique()):
            sub = neutral_with_returns[neutral_with_returns['rising_count'] == rc]
            avg = sub['fwd_return'].mean() * 100
            hit = (sub['fwd_max'] >= threshold).mean() * 100
            self.stdout.write(f"  rising_count={rc}: n={len(sub):3d}, avg={avg:+.1f}%, hit>{threshold*100:.0f}%={hit:.0f}%")
        
        # By falling_count
        self.stdout.write("\n📊 NEUTRAL by falling_count:")
        for fc in sorted(neutral_with_returns['falling_count'].unique()):
            sub = neutral_with_returns[neutral_with_returns['falling_count'] == fc]
            avg = sub['fwd_return'].mean() * 100
            hit = (sub['fwd_max'] >= threshold).mean() * 100
            self.stdout.write(f"  falling_count={fc}: n={len(sub):3d}, avg={avg:+.1f}%, hit>{threshold*100:.0f}%={hit:.0f}%")
        
        # By score within NEUTRAL
        self.stdout.write("\n📊 NEUTRAL by fusion score:")
        for score in sorted(neutral_with_returns['score'].unique()):
            sub = neutral_with_returns[neutral_with_returns['score'] == score]
            if len(sub) < 5:
                continue
            avg = sub['fwd_return'].mean() * 100
            hit = (sub['fwd_max'] >= threshold).mean() * 100
            flag = "⚡" if avg > 3 and hit > 60 else ""
            self.stdout.write(f"  score={score:+d}: n={len(sub):3d}, avg={avg:+.1f}%, hit>{threshold*100:.0f}%={hit:.0f}% {flag}")
        
        # === FINAL SUMMARY ===
        self.stdout.write("\n" + "=" * 80)
        self.stdout.write("SUMMARY: KEY FINDINGS")
        self.stdout.write("=" * 80)
        
        # Find best NEUTRAL subgroup
        best_subgroup = None
        best_hit = 0
        for subgroup in ['NEUTRAL-IMPROVING', 'NEUTRAL-FLAT', 'NEUTRAL-DETERIORATING']:
            sub = analysis_df[analysis_df['subgroup'] == subgroup].dropna(subset=['fwd_return'])
            if len(sub) >= 10:
                hit = (sub['fwd_max'] >= threshold).mean() * 100
                if hit > best_hit:
                    best_hit = hit
                    best_subgroup = subgroup
        
        if best_subgroup:
            self.stdout.write(f"\n🔑 Best NEUTRAL subgroup: {best_subgroup} (hit rate: {best_hit:.1f}%)")
            if best_hit > 50:
                self.stdout.write("   → This subgroup may deserve a new trading lane")
            else:
                self.stdout.write("   → No clear edge found - current system is appropriate")
        
        self.stdout.write("\nDone.\n")
