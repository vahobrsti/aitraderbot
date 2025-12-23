# signals/management/commands/diagnose_notrade.py
"""
NO_TRADE Forensics Diagnostic Command

Implements a 6-phase testing plan to answer:
"Which of MDIA, Whales, or MVRV-LS is responsible for most NO_TRADE outcomes?"

PHASE 1: Build NO_TRADE Forensics Dataset
PHASE 2: First-Order Attribution (who blocked first?)
PHASE 3: Counterfactual Unlock Tests (what-if experiments)
PHASE 4: Regime Near-Miss Analysis
PHASE 5: Score vs State Consistency Check
PHASE 6: Time-Series Block Analysis
"""

from django.core.management.base import BaseCommand
from pathlib import Path
import pandas as pd
import numpy as np

from signals.fusion import fuse_signals, MarketState


class Command(BaseCommand):
    help = "Diagnose NO_TRADE causes in fusion pipeline"

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
            "--output",
            type=str,
            default="notrade_forensics.csv",
            help="Output forensics CSV",
        )

    def handle(self, *args, **options):
        csv_path = Path(options["csv"])
        year = options.get("year")
        output_path = options["output"]

        if not csv_path.exists():
            self.stderr.write(f"CSV not found: {csv_path}")
            return

        # Load features
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        self.stdout.write(f"\nLoaded {len(df)} rows from {csv_path}")
        
        if year:
            df = df.loc[f'{year}-01-01':f'{year}-12-31']
            self.stdout.write(f"Filtered to {year}: {len(df)} rows")

        # === PHASE 1: Build NO_TRADE Forensics Dataset ===
        self.stdout.write("\n" + "=" * 80)
        self.stdout.write("PHASE 1: Building NO_TRADE Forensics Dataset")
        self.stdout.write("=" * 80)
        
        forensics = []
        for idx, row in df.iterrows():
            result = fuse_signals(row)
            
            # Derive LS_STATUS
            # PERMISSIVE = structural readiness for longs
            # BEARISH = structural rejection (strict: distribution_warning or put_confirm)
            # FRAGILE = early rollover (not bearish, just unstable)
            # NEUTRAL = structural swamp (no clear signal)
            ls_permissive = (
                row.get('mvrv_ls_regime_call_confirm', 0) == 1 or
                row.get('mvrv_ls_regime_call_confirm_recovery', 0) == 1 or
                row.get('mvrv_ls_weak_uptrend', 0) == 1
            )
            ls_bearish = (
                row.get('mvrv_ls_regime_distribution_warning', 0) == 1 or
                row.get('mvrv_ls_regime_put_confirm', 0) == 1
            )
            ls_fragile = row.get('mvrv_ls_early_rollover', 0) == 1
            
            if ls_permissive:
                ls_status = 'PERMISSIVE'
            elif ls_bearish:
                ls_status = 'BEARISH'
            elif ls_fragile:
                ls_status = 'FRAGILE'
            else:
                ls_status = 'NEUTRAL'
            
            # Derive MDIA_STATUS
            mdia_inflow = (
                row.get('mdia_regime_strong_inflow', 0) == 1 or
                row.get('mdia_regime_inflow', 0) == 1
            )
            mdia_distrib = row.get('mdia_regime_distribution', 0) == 1
            if mdia_inflow:
                mdia_status = 'INFLOW'
            elif mdia_distrib:
                mdia_status = 'DISTRIBUTION'
            else:
                mdia_status = 'NEUTRAL'
            
            # Derive WHALE_STATUS
            whale_sponsored = (
                row.get('whale_regime_broad_accum', 0) == 1 or
                row.get('whale_regime_strategic_accum', 0) == 1
            )
            whale_distrib = (
                row.get('whale_regime_distribution', 0) == 1 or
                row.get('whale_regime_distribution_strong', 0) == 1
            )
            if whale_sponsored:
                whale_status = 'SPONSORED'
            elif whale_distrib:
                whale_status = 'DISTRIBUTING'
            else:
                whale_status = 'NEUTRAL'
            
            forensics.append({
                'date': idx,
                'state': result.state.value,
                'score': result.score,
                'ls_status': ls_status,
                'mdia_status': mdia_status,
                'whale_status': whale_status,
                'components': str(result.components),
            })
        
        forensics_df = pd.DataFrame(forensics)
        forensics_df.to_csv(output_path, index=False)
        self.stdout.write(f"Saved forensics to {output_path}")
        
        # Filter to NO_TRADE only
        notrade_df = forensics_df[forensics_df['state'] == 'no_trade']
        total_days = len(forensics_df)
        notrade_days = len(notrade_df)
        
        self.stdout.write(f"\nTotal days: {total_days}")
        self.stdout.write(f"NO_TRADE days: {notrade_days} ({notrade_days/total_days*100:.1f}%)")
        
        # === PHASE 2: First-Order Attribution ===
        self.stdout.write("\n" + "=" * 80)
        self.stdout.write("PHASE 2: First-Order Attribution (Who Blocked First?)")
        self.stdout.write("=" * 80)
        
        # Attribution with priority order
        def attribute_blocker(row):
            # 1) MVRV-LS Block: LS_STATUS == NEUTRAL
            if row['ls_status'] == 'NEUTRAL':
                return 'MVRV-LS (neutral)'
            
            # 2) MDIA Block: LS permissive but MDIA neutral
            if row['ls_status'] == 'PERMISSIVE' and row['mdia_status'] == 'NEUTRAL':
                return 'MDIA (neutral)'
            
            # 3) Whale Block: LS permissive, MDIA not neutral, but whale not sponsored
            if row['ls_status'] == 'PERMISSIVE' and row['mdia_status'] != 'NEUTRAL' and row['whale_status'] != 'SPONSORED':
                return 'WHALE (not sponsored)'
            
            # 4) LS Bearish blocks long opportunities
            if row['ls_status'] == 'BEARISH':
                return 'MVRV-LS (bearish)'
            
            # Edge case
            return 'EDGE_CASE'
        
        notrade_df = notrade_df.copy()
        notrade_df['primary_blocker'] = notrade_df.apply(attribute_blocker, axis=1)
        
        # Responsibility Table
        blocker_counts = notrade_df['primary_blocker'].value_counts()
        
        self.stdout.write("\n📊 NO_TRADE RESPONSIBILITY TABLE")
        self.stdout.write("-" * 50)
        for blocker, count in blocker_counts.items():
            pct = count / notrade_days * 100
            bar = "█" * int(pct / 2)
            self.stdout.write(f"  {blocker:25s} {count:4d} ({pct:5.1f}%) {bar}")
        
        # === PHASE 3: Counterfactual Unlock Tests ===
        self.stdout.write("\n" + "=" * 80)
        self.stdout.write("PHASE 3: Counterfactual Unlock Tests (What-If Experiments)")
        self.stdout.write("=" * 80)
        
        # Test 3.1: LS unlock - what if LS neutral became permissive?
        ls_neutral_days = notrade_df[notrade_df['ls_status'] == 'NEUTRAL']
        ls_unlock_potential = len(ls_neutral_days)
        
        # Test 3.2: MDIA unlock - what if MDIA neutral became inflow?
        mdia_blocked = notrade_df[(notrade_df['ls_status'] == 'PERMISSIVE') & (notrade_df['mdia_status'] == 'NEUTRAL')]
        mdia_unlock_potential = len(mdia_blocked)
        
        # Test 3.3: Whale unlock - what if whale neutral became sponsored?
        whale_blocked = notrade_df[(notrade_df['ls_status'] == 'PERMISSIVE') & (notrade_df['mdia_status'] != 'NEUTRAL') & (notrade_df['whale_status'] != 'SPONSORED')]
        whale_unlock_potential = len(whale_blocked)
        
        self.stdout.write("\n📊 UNLOCK SENSITIVITY MATRIX")
        self.stdout.write("   (Δ NO_TRADE is an UPPER BOUND, not realized trades)")
        self.stdout.write("-" * 60)
        self.stdout.write(f"  {'Unlock Scenario':30s} | {'Δ NO_TRADE':12s} | Interpretation")
        self.stdout.write("-" * 60)
        self.stdout.write(f"  {'LS neutral → permissive':30s} | -{ls_unlock_potential:3d} ({ls_unlock_potential/notrade_days*100:5.1f}%) | Structural gate")
        self.stdout.write(f"  {'MDIA neutral → inflow':30s} | -{mdia_unlock_potential:3d} ({mdia_unlock_potential/notrade_days*100:5.1f}%) | Timing friction")
        self.stdout.write(f"  {'Whale neutral → sponsored':30s} | -{whale_unlock_potential:3d} ({whale_unlock_potential/notrade_days*100:5.1f}%) | Sponsorship strictness")
        
        # === PHASE 4: Near-Miss Analysis ===
        self.stdout.write("\n" + "=" * 80)
        self.stdout.write("PHASE 4: Regime Near-Miss Analysis")
        self.stdout.write("=" * 80)
        
        # High score but NO_TRADE
        high_score_notrade = notrade_df[notrade_df['score'] >= 2]
        low_score_notrade = notrade_df[notrade_df['score'] <= -2]
        
        self.stdout.write("\n📊 NEAR-MISS OPPORTUNITIES")
        self.stdout.write("-" * 50)
        self.stdout.write(f"  Score >= +2 but NO_TRADE: {len(high_score_notrade):3d} (potential LONG)")
        self.stdout.write(f"  Score <= -2 but NO_TRADE: {len(low_score_notrade):3d} (potential SHORT)")
        
        if len(high_score_notrade) > 0:
            self.stdout.write("\n  High-score near-misses breakdown:")
            for blocker, count in high_score_notrade['primary_blocker'].value_counts().items():
                self.stdout.write(f"    {blocker}: {count}")
        
        if len(low_score_notrade) > 0:
            self.stdout.write("\n  Low-score near-misses breakdown:")
            for blocker, count in low_score_notrade['primary_blocker'].value_counts().items():
                self.stdout.write(f"    {blocker}: {count}")
        
        # === PHASE 5: Score vs State Consistency ===
        self.stdout.write("\n" + "=" * 80)
        self.stdout.write("PHASE 5: Score vs State Consistency Check")
        self.stdout.write("=" * 80)
        
        # Score distribution for NO_TRADE
        score_dist = notrade_df['score'].value_counts().sort_index()
        
        self.stdout.write("\n📊 SCORE DISTRIBUTION IN NO_TRADE DAYS")
        self.stdout.write("-" * 50)
        for score, count in score_dist.items():
            bar = "█" * min(50, count // 5)
            flag = "⚠️ INCONSISTENT" if abs(score) >= 2 else ""
            self.stdout.write(f"  Score {score:+2d}: {count:4d} {bar} {flag}")
        
        # === PHASE 6: Time-Series Block Analysis ===
        self.stdout.write("\n" + "=" * 80)
        self.stdout.write("PHASE 6: Time-Series Block Analysis")
        self.stdout.write("=" * 80)
        
        # Calculate consecutive NO_TRADE streaks
        forensics_df['is_notrade'] = (forensics_df['state'] == 'no_trade').astype(int)
        forensics_df['streak_id'] = (forensics_df['is_notrade'] != forensics_df['is_notrade'].shift()).cumsum()
        
        notrade_streaks = forensics_df[forensics_df['is_notrade'] == 1].groupby('streak_id').size()
        
        if len(notrade_streaks) > 0:
            self.stdout.write("\n📊 CONSECUTIVE NO_TRADE STREAK ANALYSIS")
            self.stdout.write("-" * 50)
            self.stdout.write(f"  Total streaks: {len(notrade_streaks)}")
            self.stdout.write(f"  Average streak: {notrade_streaks.mean():.1f} days")
            self.stdout.write(f"  Max streak: {notrade_streaks.max()} days")
            self.stdout.write(f"  Median streak: {notrade_streaks.median():.0f} days")
            
            # Streak length distribution
            streak_bins = [1, 5, 10, 20, 50, 100]
            self.stdout.write("\n  Streak length distribution:")
            for i in range(len(streak_bins) - 1):
                count = ((notrade_streaks >= streak_bins[i]) & (notrade_streaks < streak_bins[i+1])).sum()
                self.stdout.write(f"    {streak_bins[i]:2d}-{streak_bins[i+1]:2d} days: {count} streaks")
            count = (notrade_streaks >= streak_bins[-1]).sum()
            self.stdout.write(f"    {streak_bins[-1]:2d}+ days: {count} streaks")
        
        # === FINAL SUMMARY ===
        self.stdout.write("\n" + "=" * 80)
        self.stdout.write("SUMMARY: PRIMARY BLOCKER IDENTIFICATION")
        self.stdout.write("=" * 80)
        
        top_blocker = blocker_counts.idxmax()
        top_pct = blocker_counts.iloc[0] / notrade_days * 100
        
        # Separate structural analysis
        ls_neutral_pct = blocker_counts.get('MVRV-LS (neutral)', 0) / notrade_days * 100
        ls_fragile_pct = blocker_counts.get('MVRV-LS (fragile)', 0) / notrade_days * 100
        ls_bearish_pct = blocker_counts.get('MVRV-LS (bearish)', 0) / notrade_days * 100
        
        self.stdout.write(f"\n🔑 PRIMARY BLOCKER: {top_blocker} ({top_pct:.1f}% of NO_TRADE days)")
        
        self.stdout.write("\n📊 MVRV-LS BREAKDOWN:")
        self.stdout.write(f"   NEUTRAL (missed opportunity risk):  {ls_neutral_pct:5.1f}%")
        self.stdout.write(f"   FRAGILE (early rollover instability): {ls_fragile_pct:5.1f}%")
        self.stdout.write(f"   BEARISH (correct avoidance):         {ls_bearish_pct:5.1f}%")
        
        if 'neutral' in top_blocker.lower():
            self.stdout.write("\n📝 INTERPRETATION:")
            self.stdout.write("   The system is STRUCTURALLY CONSERVATIVE (by design).")
            self.stdout.write("   MVRV-LS acts as a strong gatekeeper, requiring clear trend confirmation.")
            self.stdout.write("   This prevents false signals but may miss early moves.")
        elif 'fragile' in top_blocker.lower():
            self.stdout.write("\n📝 INTERPRETATION:")
            self.stdout.write("   Early rollover is causing hesitation.")
            self.stdout.write("   This is NOT bearish, just instability — structure is fragile.")
        elif 'bearish' in top_blocker.lower():
            self.stdout.write("\n📝 INTERPRETATION:")
            self.stdout.write("   MVRV-LS is correctly protecting you from bad longs.")
            self.stdout.write("   This is GOOD - don't 'fix' this.")
        elif 'MDIA' in top_blocker:
            self.stdout.write("\n📝 INTERPRETATION:")
            self.stdout.write("   TIMING is too strict.")
            self.stdout.write("   MDIA requires inflow confirmation even when structure is permissive.")
            self.stdout.write("   Consider adding weaker MDIA threshold or momentum lane.")
        elif 'WHALE' in top_blocker:
            self.stdout.write("\n📝 INTERPRETATION:")
            self.stdout.write("   SPONSORSHIP requirement is too heavy.")
            self.stdout.write("   Whales are neutral too often, blocking otherwise valid setups.")
            self.stdout.write("   Consider relaxing whale requirement for momentum continuation.")
        
        self.stdout.write("\n\nDone. Forensics saved to: " + output_path)

