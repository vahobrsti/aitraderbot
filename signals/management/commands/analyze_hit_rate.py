# signals/management/commands/analyze_hit_rate.py
"""
Analyze hit rates for all trades across all years.
Uses list_trades logic to enumerate trades, then checks if they hit target
using label_good_move_long / label_good_move_short from the features.

Usage:
    python manage.py analyze_hit_rate --csv features_14d_5pct.csv --year 2024
    python manage.py analyze_hit_rate  # All years, default CSV
"""

from django.core.management.base import BaseCommand
from pathlib import Path
import pandas as pd
import joblib
from collections import defaultdict

from signals.fusion import fuse_signals, add_fusion_features, MarketState
from signals.overlays import apply_overlays, get_size_multiplier
from signals.tactical_puts import tactical_put_inside_bull


class Command(BaseCommand):
    help = "Analyze hit rates for trades by year, type, and source"

    def add_arguments(self, parser):
        parser.add_argument(
            "--csv",
            type=str,
            default="features_14d_5pct.csv",
            help="Input features CSV (default: features_14d_5pct.csv)",
        )
        parser.add_argument(
            "--year",
            type=int,
            default=None,
            help="Filter to specific year (e.g., 2024). If not provided, analyzes all years.",
        )
        parser.add_argument(
            "--no-overlay",
            action="store_true",
            help="Disable overlay filtering - analyze all fusion-qualified trades",
        )
        parser.add_argument(
            "--no-cooldown",
            action="store_true",
            help="Disable cooldown - analyze every signal switch",
        )
        parser.add_argument(
            "--long-model",
            type=str,
            default="models/long_model.joblib",
            help="Path to long model",
        )
        parser.add_argument(
            "--short-model",
            type=str,
            default="models/short_model.joblib",
            help="Path to short model",
        )

    def handle(self, *args, **options):
        csv_path = Path(options["csv"])
        year_filter = options.get("year")
        no_overlay = options.get("no_overlay", False)
        no_cooldown = options.get("no_cooldown", False)
        long_model_path = Path(options["long_model"])
        short_model_path = Path(options["short_model"])

        if not csv_path.exists():
            self.stderr.write(f"CSV not found: {csv_path}")
            return

        # Load ML models
        long_bundle = joblib.load(long_model_path)
        short_bundle = joblib.load(short_model_path)
        long_model = long_bundle["model"]
        long_feats = long_bundle["feature_names"]
        short_model = short_bundle["model"]
        short_feats = short_bundle["feature_names"]

        # Load and prepare data
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        df = add_fusion_features(df)

        # Pre-compute ML probabilities for all rows
        df["p_long"] = long_model.predict_proba(df[long_feats])[:, 1]
        df["p_short"] = short_model.predict_proba(df[short_feats])[:, 1]

        # Define state categories
        long_states = {
            MarketState.STRONG_BULLISH,
            MarketState.EARLY_RECOVERY,
            MarketState.MOMENTUM_CONTINUATION,
            MarketState.BULL_PROBE,
        }
        short_states = {
            MarketState.DISTRIBUTION_RISK,
            MarketState.BEAR_CONTINUATION,
            MarketState.BEAR_PROBE,
        }

        # Cooldown days
        core_cooldown_days = 7
        probe_cooldown_days = 5
        tactical_cooldown_days = 7

        # Collect all trades with hit status
        all_trades = []

        # Get years to analyze
        all_years = sorted(df.index.year.unique())
        if year_filter:
            if year_filter not in all_years:
                self.stderr.write(f"Year {year_filter} not in data. Available: {all_years}")
                return
            years_to_analyze = [year_filter]
        else:
            years_to_analyze = all_years

        self.stdout.write(f"\n{'=' * 100}")
        self.stdout.write(f"ANALYZING HIT RATES - CSV: {csv_path}")
        self.stdout.write(f"{'=' * 100}")
        self.stdout.write(f"Years: {years_to_analyze}")
        self.stdout.write(f"Overlay: {'OFF' if no_overlay else 'ON'}")
        self.stdout.write(f"Cooldown: {'OFF' if no_cooldown else 'ON'}")
        self.stdout.write("")

        for year in years_to_analyze:
            df_year = df.loc[f"{year}-01-01":f"{year}-12-31"]
            if len(df_year) == 0:
                continue

            # Reset cooldown trackers per year
            last_long_date = None
            last_short_date = None
            last_tactical_date = None

            for date, row in df_year.iterrows():
                result = fuse_signals(row)
                date_str = date.strftime("%Y-%m-%d")

                # Handle overlay bypass
                if no_overlay:
                    size_mult = 1.0
                    long_veto = False
                    short_veto = False
                else:
                    overlay = apply_overlays(result, row)
                    size_mult = get_size_multiplier(overlay)
                    long_veto = overlay.long_veto_strength >= 2
                    short_veto = overlay.short_veto_strength >= 2

                is_bull_probe = result.state == MarketState.BULL_PROBE
                is_bear_probe = result.state == MarketState.BEAR_PROBE
                is_long_state = result.state in long_states and not is_bull_probe
                is_short_state = result.state in short_states and not is_bear_probe

                # Gate probes by score
                if is_bull_probe and result.score < 2:
                    continue
                if is_bear_probe and result.score > -2:
                    continue

                # Cooldown checks
                if not no_cooldown:
                    if is_long_state and last_long_date is not None:
                        if (date - last_long_date).days < core_cooldown_days:
                            continue
                    if is_short_state and last_short_date is not None:
                        if (date - last_short_date).days < core_cooldown_days:
                            continue
                    if is_bull_probe and last_long_date is not None:
                        if (date - last_long_date).days < probe_cooldown_days:
                            continue
                    if is_bear_probe and last_short_date is not None:
                        if (date - last_short_date).days < probe_cooldown_days:
                            continue

                # Probe sizing
                if is_bull_probe:
                    if result.score >= 4:
                        size_mult = min(size_mult, 0.60)
                    elif result.score == 3:
                        size_mult = min(size_mult, 0.50)
                    else:
                        size_mult = min(size_mult, 0.35)

                if is_bear_probe:
                    if result.score <= -4:
                        size_mult = min(size_mult, 0.60)
                    elif result.score == -3:
                        size_mult = min(size_mult, 0.50)
                    else:
                        size_mult = min(size_mult, 0.35)

                # === LONG TRADES ===
                if result.state in long_states and size_mult > 0:
                    if not long_veto:
                        trade_type = "BULL_PROBE" if is_bull_probe else "LONG"
                        # Check if hit target using label
                        hit = int(row.get("label_good_move_long", 0))
                        all_trades.append({
                            "date": date_str,
                            "year": year,
                            "type": trade_type,
                            "direction": "LONG",
                            "state": result.state.value,
                            "score": result.score,
                            "source": "rule",  # LONG is always rule-based
                            "hit": hit,
                            "p_long": row["p_long"],
                            "p_short": row["p_short"],
                        })
                        last_long_date = date

                # === SHORT TRADES ===
                if result.state in short_states and size_mult > 0:
                    if not short_veto:
                        trade_type = "BEAR_PROBE" if is_bear_probe else "PRIMARY_SHORT"
                        source = result.short_source or "rule"
                        hit = int(row.get("label_good_move_short", 0))
                        all_trades.append({
                            "date": date_str,
                            "year": year,
                            "type": trade_type,
                            "direction": "SHORT",
                            "state": result.state.value,
                            "score": result.score,
                            "source": source,
                            "hit": hit,
                            "p_long": row["p_long"],
                            "p_short": row["p_short"],
                        })
                        last_short_date = date

                # === TACTICAL PUTS ===
                if result.state in long_states:
                    if not no_cooldown and last_tactical_date is not None:
                        if (date - last_tactical_date).days < tactical_cooldown_days:
                            continue

                    tactical = tactical_put_inside_bull(result, row)
                    if tactical.active:
                        hit = int(row.get("label_good_move_short", 0))
                        all_trades.append({
                            "date": date_str,
                            "year": year,
                            "type": "TACTICAL_PUT",
                            "direction": "PUT",
                            "state": result.state.value,
                            "score": result.score,
                            "source": "tactical",
                            "hit": hit,
                            "p_long": row["p_long"],
                            "p_short": row["p_short"],
                        })
                        last_tactical_date = date

        # === PRINT RESULTS ===
        if not all_trades:
            self.stdout.write("No trades found.")
            return

        trades_df = pd.DataFrame(all_trades)

        # === SUMMARY BY YEAR ===
        self.stdout.write(f"\n{'=' * 100}")
        self.stdout.write("HIT RATE BY YEAR")
        self.stdout.write(f"{'=' * 100}")

        for year in years_to_analyze:
            year_df = trades_df[trades_df["year"] == year]
            if len(year_df) == 0:
                continue

            total = len(year_df)
            hits = year_df["hit"].sum()
            hit_rate = hits / total * 100 if total > 0 else 0

            self.stdout.write(f"\n{year}: {hits}/{total} = {hit_rate:.1f}%")

            # Breakdown by direction
            for direction in ["LONG", "SHORT", "PUT"]:
                dir_df = year_df[year_df["direction"] == direction]
                if len(dir_df) > 0:
                    d_total = len(dir_df)
                    d_hits = dir_df["hit"].sum()
                    d_rate = d_hits / d_total * 100 if d_total > 0 else 0
                    self.stdout.write(f"  {direction:6s}: {d_hits}/{d_total} = {d_rate:.1f}%")

        # === SUMMARY BY TYPE ===
        self.stdout.write(f"\n{'=' * 100}")
        self.stdout.write("HIT RATE BY TRADE TYPE")
        self.stdout.write(f"{'=' * 100}")

        for trade_type in ["LONG", "BULL_PROBE", "PRIMARY_SHORT", "BEAR_PROBE", "TACTICAL_PUT"]:
            type_df = trades_df[trades_df["type"] == trade_type]
            if len(type_df) > 0:
                t_total = len(type_df)
                t_hits = type_df["hit"].sum()
                t_rate = t_hits / t_total * 100 if t_total > 0 else 0
                self.stdout.write(f"{trade_type:15s}: {t_hits:3d}/{t_total:3d} = {t_rate:.1f}%")

        # === SUMMARY BY SHORT SOURCE ===
        self.stdout.write(f"\n{'=' * 100}")
        self.stdout.write("HIT RATE BY SHORT SOURCE (Shorts + Tactical Puts only)")
        self.stdout.write(f"{'=' * 100}")

        short_trades = trades_df[trades_df["direction"].isin(["SHORT", "PUT"])]
        for source in short_trades["source"].unique():
            src_df = short_trades[short_trades["source"] == source]
            s_total = len(src_df)
            s_hits = src_df["hit"].sum()
            s_rate = s_hits / s_total * 100 if s_total > 0 else 0
            self.stdout.write(f"{source:15s}: {s_hits:3d}/{s_total:3d} = {s_rate:.1f}%")

        # === SUMMARY BY STATE ===
        self.stdout.write(f"\n{'=' * 100}")
        self.stdout.write("HIT RATE BY MARKET STATE")
        self.stdout.write(f"{'=' * 100}")

        for state in trades_df["state"].unique():
            state_df = trades_df[trades_df["state"] == state]
            st_total = len(state_df)
            st_hits = state_df["hit"].sum()
            st_rate = st_hits / st_total * 100 if st_total > 0 else 0
            self.stdout.write(f"{state:25s}: {st_hits:3d}/{st_total:3d} = {st_rate:.1f}%")

        # === DETAILED TRADE LIST ===
        self.stdout.write(f"\n{'=' * 100}")
        self.stdout.write("DETAILED TRADE LIST")
        self.stdout.write(f"{'=' * 100}")
        self.stdout.write(
            f"{'Date':12s} | {'Type':15s} | {'Dir':6s} | {'State':22s} | {'Score':5s} | {'Source':10s} | {'Hit':3s} | {'p_long':6s} | {'p_short':7s}"
        )
        self.stdout.write("-" * 110)

        for _, t in trades_df.iterrows():
            hit_emoji = "✅" if t["hit"] == 1 else "❌"
            dir_emoji = "🟢" if t["direction"] == "LONG" else "🔴"
            self.stdout.write(
                f"{t['date']:12s} | {t['type']:15s} | {dir_emoji} {t['direction']:4s} | "
                f"{t['state']:22s} | {t['score']:+4d}  | {t['source']:10s} | {hit_emoji}  | "
                f"{t['p_long']:.2f}   | {t['p_short']:.2f}"
            )

        # === OVERALL SUMMARY ===
        self.stdout.write(f"\n{'=' * 100}")
        total_all = len(trades_df)
        hits_all = trades_df["hit"].sum()
        rate_all = hits_all / total_all * 100 if total_all > 0 else 0
        self.stdout.write(f"OVERALL: {hits_all}/{total_all} = {rate_all:.1f}% hit rate")
        self.stdout.write(f"{'=' * 100}\n")
