# features/management/commands/sanity_check.py
"""
Sanity check for feature CSV files.
Reports label distribution (long/short) and option signal hit rates.
Shows per-trade detail with cooldown, similar to analyze_hit_rate.
"""

from pathlib import Path

import pandas as pd
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Sanity check feature CSV: label distribution and option signal hit rates"

    def add_arguments(self, parser):
        parser.add_argument(
            "--csv",
            type=str,
            default="features_14d_5pct.csv",
            help="Path to the feature CSV file.",
        )
        parser.add_argument(
            "--year",
            type=int,
            default=None,
            help="Filter to a specific year (e.g., 2024)",
        )
        parser.add_argument(
            "--cooldown",
            type=int,
            default=7,
            help="Cooldown days between option signals (default: 7, use 0 for no cooldown)",
        )
        parser.add_argument(
            "--direction",
            type=str,
            default=None,
            choices=["call", "put"],
            help="Filter by direction: call or put",
        )

    def handle(self, *args, **options):
        csv_path = Path(options["csv"])

        if not csv_path.exists():
            self.stderr.write(self.style.ERROR(f"CSV file not found: {csv_path}"))
            return

        df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")

        year = options.get("year")
        if year:
            df = df.loc[f'{year}-01-01':f'{year}-12-31']

        cooldown_days = options["cooldown"]
        direction = options.get("direction")

        # --- Header ---
        self.stdout.write(f"\n{'=' * 100}")
        self.stdout.write(f"SANITY CHECK — {csv_path}")
        self.stdout.write(f"{'=' * 100}")
        self.stdout.write(f"Rows: {len(df)} | Range: {df.index.min().date()} to {df.index.max().date()}")
        if year:
            self.stdout.write(f"Year filter: {year}")
        self.stdout.write(f"Cooldown: {cooldown_days}d")
        if direction:
            self.stdout.write(f"Direction filter: {direction.upper()}")
        self.stdout.write("")

        # --- Label Summary ---
        self.stdout.write(self.style.MIGRATE_HEADING("Label means"))
        self.stdout.write(str(df[["label_good_move_long", "label_good_move_short"]].mean()))

        # --- Build trades list ---
        trades = []

        if direction in (None, "call"):
            call_days = df[df["signal_option_call"] == 1]
            call_cd = self._apply_cooldown(call_days, cooldown_days) if cooldown_days > 0 else call_days
            for date in call_days.index:
                row = df.loc[date]
                kept = date in call_cd.index
                trades.append({
                    "date": date,
                    "direction": "CALL",
                    "hit": int(row["label_good_move_long"]),
                    "status": "KEPT" if kept else "cooldown",
                    "sent_norm": row.get("sentiment_norm", float("nan")),
                    "mvrv_underval": int(row.get("mvrv_comp_undervalued_90d", 0)),
                    "mvrv_new_low": int(row.get("mvrv_comp_new_low_180d", 0)),
                    "mvrv_near_bot": int(row.get("mvrv_comp_near_bottom_any", 0)),
                    "mvrv_60d_pct": row.get("mvrv_60d_pct_rank", float("nan")),
                })

        if direction in (None, "put"):
            put_days = df[df["signal_option_put"] == 1]
            put_cd = self._apply_cooldown(put_days, cooldown_days) if cooldown_days > 0 else put_days
            for date in put_days.index:
                row = df.loc[date]
                kept = date in put_cd.index
                trades.append({
                    "date": date,
                    "direction": "PUT",
                    "hit": int(row["label_good_move_short"]),
                    "status": "KEPT" if kept else "cooldown",
                    "sent_norm": row.get("sentiment_norm", float("nan")),
                    "mvrv_underval": int(row.get("mvrv_comp_undervalued_90d", 0)),
                    "mvrv_new_low": int(row.get("mvrv_comp_new_low_180d", 0)),
                    "mvrv_near_bot": int(row.get("mvrv_comp_near_bottom_any", 0)),
                    "mvrv_60d_pct": row.get("mvrv_60d_pct_rank", float("nan")),
                })

        if not trades:
            self.stdout.write("No option signal days found.")
            return

        trades_df = pd.DataFrame(trades).sort_values("date")

        # --- Summary by direction ---
        self.stdout.write(self.style.MIGRATE_HEADING("Hit rate summary"))
        for dir_name in ["CALL", "PUT"]:
            dir_df = trades_df[trades_df["direction"] == dir_name]
            if len(dir_df) == 0:
                continue
            raw_total = len(dir_df)
            raw_hits = dir_df["hit"].sum()
            raw_rate = raw_hits / raw_total * 100

            kept_df = dir_df[dir_df["status"] == "KEPT"]
            kept_total = len(kept_df)
            kept_hits = kept_df["hit"].sum()
            kept_rate = kept_hits / kept_total * 100 if kept_total > 0 else 0

            emoji = "🟢" if dir_name == "CALL" else "🔴"
            self.stdout.write(
                f"{emoji} {dir_name:4s}  "
                f"Raw: {raw_hits}/{raw_total} = {raw_rate:.1f}%  |  "
                f"After cooldown: {kept_hits}/{kept_total} = {kept_rate:.1f}%"
            )

        # --- Summary by year ---
        if year is None:
            self.stdout.write(self.style.MIGRATE_HEADING("Hit rate by year (KEPT only)"))
            kept_trades = trades_df[trades_df["status"] == "KEPT"].copy()
            kept_trades["year"] = kept_trades["date"].dt.year
            for yr in sorted(kept_trades["year"].unique()):
                yr_df = kept_trades[kept_trades["year"] == yr]
                yr_total = len(yr_df)
                yr_hits = yr_df["hit"].sum()
                yr_rate = yr_hits / yr_total * 100
                self.stdout.write(f"  {yr}: {yr_hits}/{yr_total} = {yr_rate:.1f}%")

        # --- Trade list ---
        self.stdout.write(self.style.MIGRATE_HEADING("Trade list"))
        self.stdout.write(
            f"{'Date':12s} | {'Dir':4s} | {'Hit':3s} | {'Status':8s} | "
            f"{'Sent':6s} | {'MVRV flags':12s} | {'60d_pct':7s}"
        )
        self.stdout.write("-" * 80)

        for _, t in trades_df.iterrows():
            hit_icon = "✅" if t["hit"] else "❌"
            dir_icon = "🟢" if t["direction"] == "CALL" else "🔴"
            flags = f"{t['mvrv_underval']}/{t['mvrv_new_low']}/{t['mvrv_near_bot']}"
            self.stdout.write(
                f"{t['date'].strftime('%Y-%m-%d'):12s} | {dir_icon}{t['direction']:3s} | "
                f"{hit_icon}  | {t['status']:8s} | "
                f"{t['sent_norm']:+5.2f} | "
                f"{flags:12s} | {t['mvrv_60d_pct']:.2f}"
            )

        # --- Overall ---
        kept_all = trades_df[trades_df["status"] == "KEPT"]
        total = len(kept_all)
        hits = kept_all["hit"].sum()
        rate = hits / total * 100 if total > 0 else 0
        self.stdout.write(f"\n{'=' * 80}")
        self.stdout.write(f"OVERALL (KEPT): {hits}/{total} = {rate:.1f}% hit rate")
        self.stdout.write(f"{'=' * 80}\n")

    @staticmethod
    def _apply_cooldown(signal_df, cooldown_days):
        """Keep only the first signal day in each cluster, skipping any within cooldown."""
        if signal_df.empty:
            return signal_df
        kept = []
        last_date = None
        for date in signal_df.index:
            if last_date is None or (date - last_date).days >= cooldown_days:
                kept.append(date)
                last_date = date
        return signal_df.loc[kept]
