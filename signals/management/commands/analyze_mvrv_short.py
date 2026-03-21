"""
MVRV Short Signal Analysis & Backtest

Strategy: Short when mvrv_7d >= threshold in bear mode with mvrv_60d >= 1.0 overlay
Target: 4% price drop within 5 days
DCA: $30 initial, $60 if price rises 4% against

Usage:
  python manage.py mvrv_short_signal                    # Run with defaults
  python manage.py mvrv_short_signal --mvrv7d 1.05     # Adjust mvrv_7d threshold
  python manage.py mvrv_short_signal --target 5        # 5% target instead of 4%
  python manage.py mvrv_short_signal --window 7        # 7-day window
  python manage.py mvrv_short_signal --walkforward     # Run walk-forward validation
"""
import os
import pandas as pd
from pathlib import Path
from django.core.management.base import BaseCommand
from django.conf import settings
from datafeed.models import RawDailyData


class Command(BaseCommand):
    help = "Backtest MVRV short signal for bear market tactical shorts"

    def add_arguments(self, parser):
        parser.add_argument("--mvrv7d", type=float, default=1.02, help="MVRV 7d threshold (default: 1.02)")
        parser.add_argument("--mvrv60d", type=float, default=1.0, help="MVRV 60d overlay threshold (default: 1.0)")
        parser.add_argument("--target", type=float, default=4.0, help="Target drop %% (default: 4.0)")
        parser.add_argument("--window", type=int, default=5, help="Window days (default: 5)")
        parser.add_argument("--bear-start", type=int, default=540, help="Bear mode start cycle day (default: 540)")
        parser.add_argument("--bear-end", type=int, default=900, help="Bear mode end cycle day (default: 900)")
        parser.add_argument("--walkforward", action="store_true", help="Run walk-forward validation")
        parser.add_argument("--today", action="store_true", help="Check today's signal status")

    def handle(self, *args, **options):
        mvrv_7d_threshold = options["mvrv7d"]
        mvrv_60d_threshold = options["mvrv60d"]
        target_pct = options["target"]
        window_days = options["window"]
        bear_start = options["bear_start"]
        bear_end = options["bear_end"]

        self.stdout.write(f"\nLoading data...")
        df = self.load_data(bear_start, bear_end)

        if options["today"]:
            self.check_today(df, mvrv_7d_threshold, mvrv_60d_threshold, bear_start, bear_end, target_pct, window_days)
            return

        if options["walkforward"]:
            self.run_walkforward(df, mvrv_7d_threshold, mvrv_60d_threshold, target_pct, window_days)
        else:
            self.run_backtest(df, mvrv_7d_threshold, mvrv_60d_threshold, target_pct, window_days)

    def load_data(self, bear_start, bear_end):
        csv_path = Path(settings.BASE_DIR) / 'features_14d_5pct.csv'
        features_df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

        bear_dates = features_df[
            (features_df['cycle_days_since_halving'] >= bear_start) &
            (features_df['cycle_days_since_halving'] <= bear_end)
        ].index
        bear_date_strs = set(d.strftime('%Y-%m-%d') for d in bear_dates)

        raw_qs = RawDailyData.objects.all().values(
            'date', 'btc_close', 'btc_low', 'btc_high', 'mvrv_usd_7d', 'mvrv_usd_60d'
        )
        df = pd.DataFrame(list(raw_qs))
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        df['is_bear'] = df['date'].dt.strftime('%Y-%m-%d').isin(bear_date_strs)
        df['year'] = df['date'].dt.year

        return df

    def which_hit_first(self, entry_close, df, start_idx, window_days, target_pct):
        drop_target = entry_close * (1 - target_pct / 100)
        rise_target = entry_close * (1 + target_pct / 100)

        for day in range(1, window_days + 1):
            idx = start_idx + day
            if idx >= len(df):
                break

            low = df.loc[idx, 'btc_low']
            high = df.loc[idx, 'btc_high']

            drop_hit = low <= drop_target
            rise_hit = high >= rise_target

            if drop_hit and rise_hit:
                return 'both_same_day'
            elif drop_hit:
                return 'drop'
            elif rise_hit:
                return 'rise'

        return 'neither'

    def run_backtest(self, df, mvrv_7d_threshold, mvrv_60d_threshold, target_pct, window_days):
        results = []

        for i, row in df.iterrows():
            if not row['is_bear']:
                continue
            if pd.isna(row['mvrv_usd_7d']) or pd.isna(row['btc_close']):
                continue
            if pd.isna(row['mvrv_usd_60d']) or row['mvrv_usd_60d'] < mvrv_60d_threshold:
                continue
            if row['mvrv_usd_7d'] < mvrv_7d_threshold:
                continue
            if i + window_days >= len(df):
                continue

            result = self.which_hit_first(row['btc_close'], df, i, window_days, target_pct)
            year = row['date'].year
            market = '2018' if year == 2018 else '2022' if year == 2022 else '2025-2026' if year >= 2025 else 'other'

            results.append({
                'date': row['date'],
                'market': market,
                'mvrv_7d': row['mvrv_usd_7d'],
                'mvrv_60d': row['mvrv_usd_60d'],
                'result': result
            })

        results_df = pd.DataFrame(results)
        self.print_results(results_df, mvrv_7d_threshold, mvrv_60d_threshold, target_pct, window_days)

    def print_results(self, results_df, mvrv_7d_threshold, mvrv_60d_threshold, target_pct, window_days):
        self.stdout.write(f"\n{'='*70}")
        self.stdout.write(f"MVRV SHORT SIGNAL BACKTEST")
        self.stdout.write(f"Entry: mvrv_7d >= {mvrv_7d_threshold} | Overlay: mvrv_60d >= {mvrv_60d_threshold}")
        self.stdout.write(f"Target: {target_pct}% drop | Window: {window_days} days")
        self.stdout.write(f"{'='*70}")

        if results_df.empty:
            self.stdout.write("No signals found!")
            return

        total = len(results_df)
        drops = (results_df['result'] == 'drop').sum()
        rises = (results_df['result'] == 'rise').sum()
        both = (results_df['result'] == 'both_same_day').sum()
        neither = (results_df['result'] == 'neither').sum()

        self.stdout.write(f"\nOVERALL ({total} signals):")
        self.stdout.write(f"  DROP first:    {drops:>3} ({drops/total*100:.1f}%)")
        self.stdout.write(f"  RISE first:    {rises:>3} ({rises/total*100:.1f}%)")
        self.stdout.write(f"  Both same day: {both:>3} ({both/total*100:.1f}%)")
        self.stdout.write(f"  Neither:       {neither:>3} ({neither/total*100:.1f}%)")

        if drops > 0 and rises > 0:
            ratio = drops / rises
            self.stdout.write(f"\n  Drop:Rise ratio: {ratio:.2f}:1")

        self.stdout.write(f"\n{'Market':<12} | {'Signals':>7} | {'Drop 1st':>8} | {'Rise 1st':>8} | {'Ratio':>6}")
        self.stdout.write("-" * 55)
        for market in ['2018', '2022', '2025-2026']:
            subset = results_df[results_df['market'] == market]
            if len(subset) > 0:
                n = len(subset)
                d = (subset['result'] == 'drop').sum()
                r = (subset['result'] == 'rise').sum()
                ratio_str = f"{d/r:.1f}:1" if r > 0 else "∞"
                self.stdout.write(f"{market:<12} | {n:>7} | {d:>3} ({d/n*100:>4.0f}%) | {r:>3} ({r/n*100:>4.0f}%) | {ratio_str:>6}")

        # DCA analysis
        self.stdout.write(f"\n{'='*70}")
        self.stdout.write("DCA STRATEGY ANALYSIS ($30 initial, $60 DCA at +4%)")
        self.stdout.write(f"{'='*70}")

        win_rate = drops / total if total > 0 else 0
        # Scenario A: drop first (win $1.20 on $30)
        a_profit = 30 * (target_pct / 100)
        # Scenario B1: rise first, then drop (win on avg entry)
        avg_entry = (30 * 100 + 60 * 104) / 90
        b1_profit = 90 * (avg_entry - 96) / avg_entry
        # Scenario B2: rise first, never drops (lose on $30)
        b2_loss = 30 * (target_pct / 100)

        # Estimate B1/B2 split (from earlier analysis: 54% eventually drop)
        b1_rate = 0.54
        b2_rate = 0.46

        ev = win_rate * a_profit + (1 - win_rate) * b1_rate * b1_profit - (1 - win_rate) * b2_rate * b2_loss

        self.stdout.write(f"\nScenario A ({win_rate*100:.0f}%): Drop first → Win ${a_profit:.2f}")
        self.stdout.write(f"Scenario B1 ({(1-win_rate)*b1_rate*100:.0f}%): Rise→DCA→Drop → Win ${b1_profit:.2f}")
        self.stdout.write(f"Scenario B2 ({(1-win_rate)*b2_rate*100:.0f}%): Rise→DCA→Stuck → Lose ${b2_loss:.2f}")
        self.stdout.write(f"\nExpected Value: ${ev:.2f} per $90 risked ({ev/90*100:.2f}%)")
        self.stdout.write(f"{'='*70}\n")

    def run_walkforward(self, df, mvrv_7d_threshold, mvrv_60d_threshold, target_pct, window_days):
        self.stdout.write(f"\n{'='*70}")
        self.stdout.write("WALK-FORWARD VALIDATION")
        self.stdout.write(f"Signal: mvrv_7d >= {mvrv_7d_threshold} AND mvrv_60d >= {mvrv_60d_threshold}")
        self.stdout.write(f"{'='*70}")

        periods = [
            ('2018', df[(df['year'] == 2018) & df['is_bear']].index.tolist()),
            ('2022', df[(df['year'] == 2022) & df['is_bear']].index.tolist()),
            ('2025-2026', df[(df['year'] >= 2025) & df['is_bear']].index.tolist()),
        ]

        self.stdout.write(f"\n--- FOLD 1: Train on 2018, Test on 2022 ---")
        train_results = self.run_period(df, periods[0][1], mvrv_7d_threshold, mvrv_60d_threshold, target_pct, window_days)
        test_results = self.run_period(df, periods[1][1], mvrv_7d_threshold, mvrv_60d_threshold, target_pct, window_days)
        self.print_fold("2018 (train)", train_results)
        self.print_fold("2022 (test)", test_results)

        self.stdout.write(f"\n--- FOLD 2: Train on 2018+2022, Test on 2025-2026 ---")
        train_indices = periods[0][1] + periods[1][1]
        train_results = self.run_period(df, train_indices, mvrv_7d_threshold, mvrv_60d_threshold, target_pct, window_days)
        test_results = self.run_period(df, periods[2][1], mvrv_7d_threshold, mvrv_60d_threshold, target_pct, window_days)
        self.print_fold("2018+2022 (train)", train_results)
        self.print_fold("2025-2026 (test)", test_results)

        self.stdout.write(f"\n{'='*70}")
        self.stdout.write("SUMMARY")
        self.stdout.write(f"{'='*70}")
        for period_name, indices in periods:
            results = self.run_period(df, indices, mvrv_7d_threshold, mvrv_60d_threshold, target_pct, window_days)
            if len(results) > 0:
                drops = (results['result'] == 'drop').sum()
                rises = (results['result'] == 'rise').sum()
                total = len(results)
                self.stdout.write(f"{period_name:<12}: {total:>3} signals | Drop: {drops:>2} ({drops/total*100:>5.1f}%) | Rise: {rises:>2} ({rises/total*100:>5.1f}%)")
        self.stdout.write(f"{'='*70}\n")

    def run_period(self, df, indices, mvrv_7d_threshold, mvrv_60d_threshold, target_pct, window_days):
        results = []
        for i in indices:
            if i >= len(df):
                continue
            row = df.loc[i]
            if pd.isna(row['mvrv_usd_7d']) or pd.isna(row['btc_close']):
                continue
            if pd.isna(row['mvrv_usd_60d']) or row['mvrv_usd_60d'] < mvrv_60d_threshold:
                continue
            if row['mvrv_usd_7d'] < mvrv_7d_threshold:
                continue
            if i + window_days >= len(df):
                continue

            result = self.which_hit_first(row['btc_close'], df, i, window_days, target_pct)
            results.append({'date': row['date'], 'result': result})

        return pd.DataFrame(results)

    def print_fold(self, label, results_df):
        if results_df.empty:
            self.stdout.write(f"  {label}: No signals")
            return

        total = len(results_df)
        drops = (results_df['result'] == 'drop').sum()
        rises = (results_df['result'] == 'rise').sum()
        self.stdout.write(f"  {label}: {total} signals | Drop first: {drops} ({drops/total*100:.0f}%) | Rise first: {rises} ({rises/total*100:.0f}%)")

    def check_today(self, df, mvrv_7d_threshold, mvrv_60d_threshold, bear_start, bear_end, target_pct, window_days):
        latest = df.iloc[-1]
        date = latest['date']
        mvrv_7d = latest['mvrv_usd_7d']
        mvrv_60d = latest['mvrv_usd_60d']
        btc_price = latest['btc_close']
        is_bear = latest['is_bear']

        signal_active = (
            is_bear and
            not pd.isna(mvrv_7d) and mvrv_7d >= mvrv_7d_threshold and
            not pd.isna(mvrv_60d) and mvrv_60d >= mvrv_60d_threshold
        )

        self.stdout.write(f"\n{'='*60}")
        self.stdout.write(f"MVRV SHORT SIGNAL - {date.strftime('%Y-%m-%d')}")
        self.stdout.write(f"{'='*60}")
        self.stdout.write(f"Bear Mode:     {'YES' if is_bear else 'NO'}")
        self.stdout.write(f"MVRV 7d:       {mvrv_7d:.4f} (threshold: {mvrv_7d_threshold})")
        self.stdout.write(f"MVRV 60d:      {mvrv_60d:.4f} (threshold: {mvrv_60d_threshold})")
        self.stdout.write(f"BTC Price:     ${btc_price:,.0f}")
        self.stdout.write(f"{'='*60}")

        if signal_active:
            target_price = btc_price * (1 - target_pct / 100)
            dca_trigger = btc_price * (1 + target_pct / 100)
            self.stdout.write(self.style.SUCCESS(f"🔴 SHORT SIGNAL ACTIVE"))
            self.stdout.write(f"\nExecution:")
            self.stdout.write(f"  Entry 1: 33% at ${btc_price:,.0f}")
            self.stdout.write(f"  Target:  ${target_price:,.0f} (-{target_pct}%)")
            self.stdout.write(f"  DCA at:  ${dca_trigger:,.0f} (+{target_pct}%) → add 67%")
            self.stdout.write(f"  Window:  {window_days} days")
        else:
            self.stdout.write(self.style.WARNING(f"⚪ NO SIGNAL"))
            if not is_bear:
                self.stdout.write(f"  Reason: Not in bear mode")
            elif pd.isna(mvrv_7d) or mvrv_7d < mvrv_7d_threshold:
                self.stdout.write(f"  Reason: MVRV 7d below threshold")
            elif pd.isna(mvrv_60d) or mvrv_60d < mvrv_60d_threshold:
                self.stdout.write(f"  Reason: MVRV 60d below breakeven")

        self.stdout.write(f"{'='*60}\n")
