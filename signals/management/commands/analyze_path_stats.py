"""
Analyze forward path diagnostics for generated trades.

This command mirrors trade enumeration logic from analyze_hit_rate and adds
time-to-hit (TTH), MAE, MFE, overshoot, path-shape, and drawdown diagnostics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand

from datafeed.models import RawDailyData
from signals.fusion import MarketState, add_fusion_features, fuse_signals
from signals.overlays import apply_overlays, compute_efb_veto, get_size_multiplier
from signals.tactical_puts import tactical_put_inside_bull


class Command(BaseCommand):
    help = "Analyze path diagnostics (TTH, MAE, MFE, overshoot, path shape)"

    def add_arguments(self, parser):
        parser.add_argument("--csv", type=str, default="features_14d_5pct.csv")
        parser.add_argument("--year", type=int, default=None)
        parser.add_argument("--no-overlay", action="store_true")
        parser.add_argument("--no-cooldown", action="store_true")
        parser.add_argument("--long-model", type=str, default="models/long_model.joblib")
        parser.add_argument("--short-model", type=str, default="models/short_model.joblib")
        parser.add_argument("--type", type=str, default=None)
        parser.add_argument("--source", type=str, default=None)
        parser.add_argument("--direction", type=str, default=None)
        parser.add_argument("--state", type=str, default=None)
        parser.add_argument("--horizon", type=int, default=14, help="Forward path horizon in days")
        parser.add_argument("--target", type=float, default=0.03, help="Touch threshold (e.g. 0.03)")
        parser.add_argument(
            "--invalidation",
            type=float,
            default=None,
            help="Invalidation threshold against signal (defaults to --target)",
        )

    def handle(self, *args, **options):
        csv_path = Path(options["csv"])
        if not csv_path.exists():
            self.stderr.write(f"CSV not found: {csv_path}")
            return

        year_filter = options.get("year")
        no_overlay = options.get("no_overlay", False)
        no_cooldown = options.get("no_cooldown", False)
        horizon = int(options.get("horizon", 14))
        target = float(options.get("target", 0.03))
        invalidation = float(options.get("invalidation") or target)

        trades_df = self._build_trades_df(csv_path, options, year_filter, no_overlay, no_cooldown)
        if trades_df.empty:
            self.stdout.write("No trades found.")
            return

        # Apply optional filters
        for key in ["type", "source", "state"]:
            val = options.get(key)
            if val:
                trades_df = trades_df[trades_df[key] == val]
        direction_filter = options.get("direction")
        if direction_filter:
            # Backward compatibility: treat PUT filter as bearish side.
            normalized_direction = "SHORT" if direction_filter == "PUT" else direction_filter
            trades_df = trades_df[trades_df["direction"] == normalized_direction]
        if trades_df.empty:
            self.stdout.write("No trades match filters.")
            return

        price_df = self._load_prices()
        if price_df.empty:
            self.stderr.write("No RawDailyData price rows found.")
            return

        metrics_df = self._compute_path_metrics(trades_df, price_df, horizon, target, invalidation)
        if metrics_df.empty:
            self.stdout.write("No trades have sufficient forward path data.")
            return

        self.stdout.write("\n" + "=" * 100)
        self.stdout.write("PATH DIAGNOSTICS")
        self.stdout.write("=" * 100)
        self.stdout.write(
            f"Trades={len(metrics_df)} | horizon={horizon}d | target={target*100:.2f}% | invalidation={invalidation*100:.2f}%"
        )
        if year_filter:
            self.stdout.write(f"Year={year_filter}")

        self._print_tth(metrics_df)
        self._print_mae(metrics_df)
        self._print_mfe(metrics_df, target)
        self._print_overshoot(metrics_df)
        self._print_path_shapes(metrics_df)
        self._print_invalidation_vs_hit(metrics_df, horizon)
        self._print_loser_drawdown(metrics_df)

    def _build_trades_df(
        self,
        csv_path: Path,
        options: dict,
        year_filter: Optional[int],
        no_overlay: bool,
        no_cooldown: bool,
    ) -> pd.DataFrame:
        long_bundle = joblib.load(Path(options["long_model"]))
        short_bundle = joblib.load(Path(options["short_model"]))
        long_model = long_bundle["model"]
        short_model = short_bundle["model"]
        long_feats = long_bundle["feature_names"]
        short_feats = short_bundle["feature_names"]

        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        df = add_fusion_features(df)
        df["p_long"] = long_model.predict_proba(df[long_feats])[:, 1]
        df["p_short"] = short_model.predict_proba(df[short_feats])[:, 1]

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

        all_years = sorted(df.index.year.unique())
        if year_filter is not None:
            years = [year_filter] if year_filter in all_years else []
        else:
            years = all_years

        core_cooldown_days = 7
        probe_cooldown_days = 5
        tactical_cooldown_days = 7
        option_cooldown_days = 7
        all_trades: list[dict] = []

        for year in years:
            year_df = df.loc[f"{year}-01-01":f"{year}-12-31"]
            if year_df.empty:
                continue

            last_long_date = None
            last_short_date = None
            last_tactical_date = None
            last_option_call_date = None
            last_option_put_date = None

            for date, row in year_df.iterrows():
                result = fuse_signals(row)

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

                # Probes are now statically gated by the fusion engine itself.
                # No dynamic score thresholding needed here.

                can_long_fire = True
                can_short_fire = True
                if not no_cooldown:
                    if is_long_state and last_long_date is not None and (date - last_long_date).days < core_cooldown_days:
                        can_long_fire = False
                    if is_short_state and last_short_date is not None and (date - last_short_date).days < core_cooldown_days:
                        can_short_fire = False
                    if is_bull_probe and last_long_date is not None and (date - last_long_date).days < probe_cooldown_days:
                        can_long_fire = False
                    if is_bear_probe and last_short_date is not None and (date - last_short_date).days < probe_cooldown_days:
                        can_short_fire = False

                # Probe sizing (simplified: limit to 0.5x base overlay)
                if is_bull_probe or is_bear_probe:
                    size_mult = min(size_mult, 0.50)

                long_trade_fired = False
                if result.state in long_states and size_mult > 0 and not long_veto and can_long_fire:
                    all_trades.append(
                        {
                            "date": pd.Timestamp(date).normalize(),
                            "year": year,
                            "type": "BULL_PROBE" if is_bull_probe else "LONG",
                            "direction": "LONG",
                            "source": "rule",
                            "state": result.state.value,
                        }
                    )
                    last_long_date = date
                    long_trade_fired = True

                if result.state in short_states and size_mult > 0 and not short_veto and can_short_fire:
                    all_trades.append(
                        {
                            "date": pd.Timestamp(date).normalize(),
                            "year": year,
                            "type": "BEAR_PROBE" if is_bear_probe else "PRIMARY_SHORT",
                            "direction": "SHORT",
                            "source": result.short_source or "rule",
                            "state": result.state.value,
                        }
                    )
                    last_short_date = date

                tactical_cooldown_ok = (
                    no_cooldown or last_tactical_date is None or (date - last_tactical_date).days >= tactical_cooldown_days
                )
                if result.state in long_states and not long_trade_fired and tactical_cooldown_ok:
                    tactical = tactical_put_inside_bull(result, row)
                    if tactical.active:
                        all_trades.append(
                            {
                                "date": pd.Timestamp(date).normalize(),
                                "year": year,
                                "type": "TACTICAL_PUT",
                                "direction": "SHORT",
                                "source": "tactical",
                                "state": result.state.value,
                            }
                        )
                        last_tactical_date = date

                signal_call = int(row.get("signal_option_call", 0))
                signal_put = int(row.get("signal_option_put", 0))

                if signal_call == 1:
                    cooldown_ok = no_cooldown or last_option_call_date is None or (date - last_option_call_date).days >= option_cooldown_days
                    overlay_ok = no_overlay or size_mult > 0
                    if cooldown_ok and overlay_ok:
                        all_trades.append(
                            {
                                "date": pd.Timestamp(date).normalize(),
                                "year": year,
                                "type": "OPTION_CALL",
                                "direction": "LONG",
                                "source": "option_rule",
                                "state": result.state.value,
                            }
                        )
                        last_option_call_date = date

                if signal_put == 1:
                    cooldown_ok = no_cooldown or last_option_put_date is None or (date - last_option_put_date).days >= option_cooldown_days
                    overlay_ok = no_overlay or size_mult > 0
                    if cooldown_ok and overlay_ok:
                        efb_veto, _ = compute_efb_veto(row)
                        if efb_veto < 1:
                            all_trades.append(
                                {
                                    "date": pd.Timestamp(date).normalize(),
                                    "year": year,
                                    "type": "OPTION_PUT",
                                    "direction": "SHORT",
                                    "source": "option_rule",
                                    "state": result.state.value,
                                }
                            )
                            last_option_put_date = date

        return pd.DataFrame(all_trades)

    def _load_prices(self) -> pd.DataFrame:
        qs = RawDailyData.objects.order_by("date").values("date", "btc_close", "btc_high", "btc_low")
        px = pd.DataFrame.from_records(qs)
        if px.empty:
            return px
        px["date"] = pd.to_datetime(px["date"])
        px = px.set_index("date").sort_index()
        px.index = px.index.normalize()
        return px

    def _compute_path_metrics(
        self,
        trades_df: pd.DataFrame,
        price_df: pd.DataFrame,
        horizon: int,
        target: float,
        invalidation: float,
    ) -> pd.DataFrame:
        rows = []
        for _, t in trades_df.iterrows():
            dt = pd.Timestamp(t["date"])
            if dt not in price_df.index:
                continue
            entry = float(price_df.loc[dt, "btc_close"])
            if not np.isfinite(entry) or entry <= 0:
                continue

            path = price_df.loc[dt:].iloc[1 : horizon + 1]
            if len(path) < horizon:
                continue

            highs = path["btc_high"].to_numpy(dtype=float)
            lows = path["btc_low"].to_numpy(dtype=float)
            closes = path["btc_close"].to_numpy(dtype=float)
            if np.isnan(highs).any() or np.isnan(lows).any() or np.isnan(closes).any():
                continue

            is_long = t["direction"] == "LONG"
            if is_long:
                favorable_daily = highs / entry - 1.0
                adverse_daily = 1.0 - (lows / entry)  # positive adverse magnitude
                terminal_dir_ret = closes[-1] / entry - 1.0
            else:
                favorable_daily = 1.0 - (lows / entry)
                adverse_daily = (highs / entry) - 1.0
                terminal_dir_ret = 1.0 - (closes[-1] / entry)

            hit_idx = np.where(favorable_daily >= target)[0]
            hit_day = int(hit_idx[0] + 1) if len(hit_idx) else None
            hit = hit_day is not None

            mfe = float(np.max(favorable_daily))
            if hit:
                mae = float(np.max(adverse_daily[:hit_day]))
                tti_idx = np.where(adverse_daily[:hit_day] >= invalidation)[0]
            else:
                mae = float(np.max(adverse_daily))
                tti_idx = np.where(adverse_daily >= invalidation)[0]
            tti = int(tti_idx[0] + 1) if len(tti_idx) else horizon + 1
            invalid_before_hit = (tti < hit_day) if hit else (tti <= horizon)
            same_day_ambiguous = int(hit and (tti == hit_day))

            rows.append(
                {
                    **t.to_dict(),
                    "entry_date": dt.date().isoformat(),
                    "hit": int(hit),
                    "tth": hit_day,
                    "max_move_if_no_hit": (None if hit else mfe),
                    "mae": mae,
                    "mfe": mfe,
                    "terminal_dir_ret": float(terminal_dir_ret),
                    "tti_invalidation": tti,
                    "invalid_before_hit": int(invalid_before_hit),
                    "same_day_ambiguous": same_day_ambiguous,
                    "path_len": len(path),
                }
            )

        return pd.DataFrame(rows)

    def _fmt_pct(self, x: float) -> str:
        return f"{x*100:.2f}%"

    def _pct(self, s: pd.Series) -> float:
        return float(s.mean() * 100.0) if len(s) else 0.0

    def _print_tth(self, df: pd.DataFrame):
        hits = df[df["hit"] == 1].copy()
        misses = df[df["hit"] == 0].copy()
        if hits.empty:
            self.stdout.write("\nTTH: no winners.")
            return
        tth = hits["tth"].astype(int)
        all_n = len(df)

        in_1d = (df["hit"].eq(1) & df["tth"].eq(1)).mean() * 100
        in_2_3 = (df["hit"].eq(1) & df["tth"].between(2, 3)).mean() * 100
        in_4_7 = (df["hit"].eq(1) & df["tth"].between(4, 7)).mean() * 100
        in_8_14 = (df["hit"].eq(1) & df["tth"].between(8, 14)).mean() * 100

        self.stdout.write("\n" + "-" * 100)
        self.stdout.write("1) TIME-TO-HIT DISTRIBUTION (TTH)")
        self.stdout.write("-" * 100)
        self.stdout.write(f"Total trades: {all_n} | Winners: {len(hits)}")
        self.stdout.write(f"% hit in 1 day   : {in_1d:.1f}%")
        self.stdout.write(f"% hit in 2-3 days: {in_2_3:.1f}%")
        self.stdout.write(f"% hit in 4-7 days: {in_4_7:.1f}%")
        self.stdout.write(f"% hit in 8-14 days: {in_8_14:.1f}%")
        self.stdout.write(f"Median TTH: {float(tth.median()):.1f} days")
        self.stdout.write(f"75th pct TTH: {float(tth.quantile(0.75)):.1f} days")
        if not misses.empty:
            mm = misses["max_move_if_no_hit"].astype(float)
            self.stdout.write(
                "No-hit max move achieved: "
                f"avg={self._fmt_pct(float(mm.mean()))}, "
                f"median={self._fmt_pct(float(mm.median()))}, "
                f"75th={self._fmt_pct(float(mm.quantile(0.75)))}"
            )

    def _print_mae(self, df: pd.DataFrame):
        winners = df[df["hit"] == 1]
        losers = df[df["hit"] == 0]

        self.stdout.write("\n" + "-" * 100)
        self.stdout.write("2) MAX ADVERSE EXCURSION (MAE)")
        self.stdout.write("-" * 100)

        if winners.empty:
            self.stdout.write("No winners.")
        else:
            self.stdout.write(f"Average MAE (winners): {self._fmt_pct(float(winners['mae'].mean()))}")
            self.stdout.write(f"75th pct MAE (winners): {self._fmt_pct(float(winners['mae'].quantile(0.75)))}")
            self.stdout.write(f"Max MAE observed (winners): {self._fmt_pct(float(winners['mae'].max()))}")

        if losers.empty:
            self.stdout.write("No losers.")
        else:
            self.stdout.write(f"Average MAE (losers): {self._fmt_pct(float(losers['mae'].mean()))}")
            self.stdout.write(f"Max MAE observed (losers): {self._fmt_pct(float(losers['mae'].max()))}")

        bins = [0, 0.01, 0.02, 0.03, 0.05, 0.08, 10]
        labels = ["0-1%", "1-2%", "2-3%", "3-5%", "5-8%", "8%+"]
        hist = pd.cut(df["mae"], bins=bins, labels=labels, include_lowest=True).value_counts(normalize=True).reindex(labels, fill_value=0)
        self.stdout.write("MAE histogram:")
        for lbl in labels:
            self.stdout.write(f"  {lbl:>5}: {hist[lbl]*100:5.1f}%")

    def _print_mfe(self, df: pd.DataFrame, target: float):
        mfe = df["mfe"]
        self.stdout.write("\n" + "-" * 100)
        self.stdout.write("3) MAX FAVORABLE EXCURSION (MFE)")
        self.stdout.write("-" * 100)
        self.stdout.write(f"Average MFE: {self._fmt_pct(float(mfe.mean()))}")
        self.stdout.write(f"Median MFE: {self._fmt_pct(float(mfe.median()))}")
        self.stdout.write(f"75th pct MFE: {self._fmt_pct(float(mfe.quantile(0.75)))}")
        self.stdout.write(f"% trades exceeding 5%: {(mfe >= 0.05).mean()*100:.1f}%")
        self.stdout.write(f"% trades exceeding 7%: {(mfe >= 0.07).mean()*100:.1f}%")
        self.stdout.write(f"% trades reaching target ({target*100:.1f}%): {(mfe >= target).mean()*100:.1f}%")

    def _print_overshoot(self, df: pd.DataFrame):
        winners = df[df["hit"] == 1]
        self.stdout.write("\n" + "-" * 100)
        self.stdout.write("7) OVERSHOOT STATISTICS (CONDITIONAL ON WINNERS)")
        self.stdout.write("-" * 100)
        if winners.empty:
            self.stdout.write("No winners.")
            return
        self.stdout.write(f"% winners exceeding 4%: {(winners['mfe'] >= 0.04).mean()*100:.1f}%")
        self.stdout.write(f"% winners exceeding 5%: {(winners['mfe'] >= 0.05).mean()*100:.1f}%")
        self.stdout.write(f"% winners exceeding 6%: {(winners['mfe'] >= 0.06).mean()*100:.1f}%")

    def _print_path_shapes(self, df: pd.DataFrame):
        winners = df[df["hit"] == 1].copy()
        self.stdout.write("\n" + "-" * 100)
        self.stdout.write("4) PATH SHAPE CLASSIFICATION (WINNERS)")
        self.stdout.write("-" * 100)
        if winners.empty:
            self.stdout.write("No winners.")
            return

        # Heuristic classes:
        # A) Clean expansion: fast hit + low MAE
        # B) Shakeout then expansion: larger MAE before hit
        # C) Slow grind: later hit, contained MAE
        # D) Overshoot then mean reversion: large MFE with strong giveback
        giveback_ratio = (winners["mfe"] - winners["terminal_dir_ret"]) / winners["mfe"].clip(lower=1e-9)
        cond_d = (winners["mfe"] >= 0.06) & (giveback_ratio >= 0.40)
        cond_b = winners["mae"] >= 0.02
        cond_a = (winners["tth"] <= 3) & (winners["mae"] <= 0.015)
        cond_c = (winners["tth"] >= 6) & (winners["mae"] <= 0.02)
        winners["shape"] = np.select([cond_d, cond_b, cond_a, cond_c], ["D", "B", "A", "C"], default="OTHER")

        mapping = {
            "A": "Clean expansion",
            "B": "Shakeout then expansion",
            "C": "Slow grind",
            "D": "Overshoot then mean reversion",
            "OTHER": "Other / mixed path",
        }
        vc = winners["shape"].value_counts()
        for k in ["A", "B", "C", "D", "OTHER"]:
            n = int(vc.get(k, 0))
            pct = (n / len(winners)) * 100
            self.stdout.write(f"{k}) {mapping[k]:28s}: {n:4d} ({pct:5.1f}%)")

    def _print_loser_drawdown(self, df: pd.DataFrame):
        losers = df[df["hit"] == 0]
        self.stdout.write("\n" + "-" * 100)
        self.stdout.write("6) LOSER DRAWDOWN PROFILE")
        self.stdout.write("-" * 100)
        if losers.empty:
            self.stdout.write("No losers.")
            return

        self.stdout.write(f"Average adverse move (losers): {self._fmt_pct(float(losers['mae'].mean()))}")
        self.stdout.write(f"Worst historical loser (adverse move): {self._fmt_pct(float(losers['mae'].max()))}")
        self.stdout.write(
            "Time until invalidation (days): "
            f"median={float(losers['tti_invalidation'].median()):.1f}, "
            f"75th={float(losers['tti_invalidation'].quantile(0.75)):.1f}, "
            f"max={int(losers['tti_invalidation'].max())}"
        )

    def _print_invalidation_vs_hit(self, df: pd.DataFrame, horizon: int):
        self.stdout.write("\n" + "-" * 100)
        self.stdout.write("5) INVALIDATION VS HIT ORDERING")
        self.stdout.write("-" * 100)

        winners = df[df["hit"] == 1]
        losers = df[df["hit"] == 0]
        if winners.empty:
            self.stdout.write("No winners.")
            return

        winner_invalid_first = winners["invalid_before_hit"].mean() * 100
        winner_same_day_ambiguous = winners["same_day_ambiguous"].mean() * 100
        clean_winners = 100.0 - winner_invalid_first
        self.stdout.write(f"% winners invalidated before hit (strict): {winner_invalid_first:.1f}%")
        self.stdout.write(f"% winners ambiguous same-day (hit and invalidation): {winner_same_day_ambiguous:.1f}%")
        self.stdout.write(f"% clean winners (no invalidation before hit): {clean_winners:.1f}%")

        if not losers.empty:
            loser_invalidated = (losers["tti_invalidation"] <= horizon).mean() * 100
            self.stdout.write(f"% losers that reached invalidation: {loser_invalidated:.1f}%")
