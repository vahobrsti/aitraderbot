"""
Train option response model from OptionSnapshot history.

Builds training rows with features:
- DTE
- moneyness
- IV level and IV change
- time elapsed
- spread/liquidity regime
- option direction (call/put)
- structure tag (naked/spread/condor)
- BTC underlying move

Then fits LeveragePredictor and runs walk-forward diagnostics.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand

from execution.services.option_pricer import LeveragePredictor, TrainingRow


class Command(BaseCommand):
    help = "Train learned option response mapping from OptionSnapshot pairs"

    def add_arguments(self, parser):
        parser.add_argument("--horizon-days", type=int, default=1, help="Target pairing horizon in days")
        parser.add_argument("--min-samples-per-bucket", type=int, default=5, help="Min rows per bucket to keep stats")
        parser.add_argument("--walk-forward-splits", type=int, default=4, help="Time splits for diagnostics")
        parser.add_argument("--min-snapshots-per-symbol", type=int, default=8)
        parser.add_argument("--underlying", type=str, default="BTC")
        parser.add_argument("--start", type=str, default=None, help="Start timestamp/date filter")
        parser.add_argument("--end", type=str, default=None, help="End timestamp/date filter")
        parser.add_argument("--out-model", type=str, default="models/option_response_predictor.json")
        parser.add_argument("--out-rows", type=str, default="", help="Optional CSV export of training rows")

    def handle(self, *args, **options):
        from datafeed.models import OptionSnapshot
        from django.db.models import Count

        horizon_days = int(options["horizon_days"])
        min_samples_per_bucket = int(options["min_samples_per_bucket"])
        min_snapshots_per_symbol = int(options["min_snapshots_per_symbol"])
        n_splits = int(options["walk_forward_splits"])

        # Build base queryset without order_by (aggregation needs it removed)
        base_qs = OptionSnapshot.objects.filter(underlying=options["underlying"])
        if options["start"]:
            base_qs = base_qs.filter(timestamp__gte=options["start"])
        if options["end"]:
            base_qs = base_qs.filter(timestamp__lte=options["end"])
        
        # For symbol counting, don't use order_by (breaks aggregation)
        qs = base_qs.order_by("timestamp")

        symbol_counts = (
            base_qs.values("symbol")
            .annotate(count=Count("id"))
            .filter(count__gte=min_snapshots_per_symbol)
        )
        symbols = [row["symbol"] for row in symbol_counts]
        if not symbols:
            self.stdout.write(self.style.ERROR("No symbols with enough snapshots for training."))
            return

        self.stdout.write(f"Eligible symbols: {len(symbols)}")
        rows_df = self._build_rows_df(qs, symbols, horizon_days)
        if rows_df.empty:
            self.stdout.write(self.style.ERROR("No training rows created from snapshot pairs."))
            return

        self.stdout.write(f"Training rows: {len(rows_df)}")
        if options["out_rows"]:
            out_rows = Path(options["out_rows"])
            out_rows.parent.mkdir(parents=True, exist_ok=True)
            rows_df.to_csv(out_rows, index=False)
            self.stdout.write(f"Rows exported: {out_rows}")

        self._print_feature_coverage(rows_df)
        self._walk_forward_eval(rows_df, min_samples_per_bucket=min_samples_per_bucket, n_splits=n_splits)

        predictor = LeveragePredictor()
        for row in rows_df.itertuples(index=False):
            predictor.add_training_row(
                TrainingRow(
                    dte=float(row.dte),
                    moneyness=float(row.moneyness),
                    iv=float(row.iv),
                    spread_pct=float(row.spread_pct),
                    option_type=str(row.option_type),
                    btc_change_pct=float(row.btc_change_pct),
                    iv_change=float(row.iv_change),
                    days_held=int(row.days_held),
                    option_return_pct=float(row.option_return_pct),
                )
            )
        predictor.fit(min_samples_per_bucket=min_samples_per_bucket)
        out_model = Path(options["out_model"])
        predictor.save(out_model)
        self.stdout.write(self.style.SUCCESS(f"Saved predictor: {out_model}"))

    def _build_rows_df(self, qs, symbols, horizon_days: int) -> pd.DataFrame:
        from datetime import timedelta

        rows = []
        for symbol in symbols:
            snaps = list(
                qs.filter(symbol=symbol).values(
                    "timestamp",
                    "symbol",
                    "option_type",
                    "spot_price",
                    "strike",
                    "mid_price",
                    "iv",
                    "spread_pct",
                    "dte",
                )
            )
            if len(snaps) < 2:
                continue

            for i, entry in enumerate(snaps[:-1]):
                entry_ts = entry["timestamp"]
                target_ts = entry_ts + timedelta(days=horizon_days)
                best_j = None
                best_diff = timedelta(days=9999)
                for j in range(i + 1, len(snaps)):
                    diff = abs(snaps[j]["timestamp"] - target_ts)
                    if diff < best_diff:
                        best_diff = diff
                        best_j = j
                    if snaps[j]["timestamp"] > target_ts + timedelta(days=1):
                        break
                if best_j is None or best_diff > timedelta(days=2):
                    continue

                exit_snap = snaps[best_j]
                entry_mid = float(entry["mid_price"] or 0)
                exit_mid = float(exit_snap["mid_price"] or 0)
                entry_spot = float(entry["spot_price"] or 0)
                exit_spot = float(exit_snap["spot_price"] or 0)
                strike = float(entry["strike"] or 0)
                if entry_mid <= 0 or exit_mid <= 0 or entry_spot <= 0 or exit_spot <= 0 or strike <= 0:
                    continue

                dte = float(entry["dte"] or 0)
                moneyness = float((strike - entry_spot) / entry_spot)
                iv_entry = float(entry["iv"] or 0.60)
                iv_exit = float(exit_snap["iv"] or iv_entry)
                spread_pct = float(entry["spread_pct"] or 0.02)
                days_held = max(1, int((exit_snap["timestamp"] - entry_ts).total_seconds() // 86400))

                rows.append(
                    {
                        "entry_ts": entry_ts,
                        "symbol": symbol,
                        "option_type": entry["option_type"],
                        "dte": dte,
                        "moneyness": moneyness,
                        "iv": iv_entry,
                        "iv_change": iv_exit - iv_entry,
                        "spread_pct": spread_pct,
                        "liquidity_regime": self._liquidity_regime(spread_pct),
                        "days_held": days_held,
                        "btc_change_pct": (exit_spot - entry_spot) / entry_spot,
                        "option_return_pct": (exit_mid - entry_mid) / entry_mid,
                        "structure": self._infer_structure(moneyness, dte, spread_pct),
                    }
                )

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("entry_ts").reset_index(drop=True)

    def _walk_forward_eval(self, rows_df: pd.DataFrame, min_samples_per_bucket: int, n_splits: int) -> None:
        n = len(rows_df)
        if n < 50 or n_splits < 2:
            self.stdout.write("Skipping walk-forward diagnostics (insufficient rows/splits).")
            return

        split_points = [int(n * k / (n_splits + 1)) for k in range(1, n_splits + 1)]
        self.stdout.write("\nWalk-forward diagnostics:")
        self.stdout.write("split | train_rows | test_rows | mae_p50 | pinball_p10 | pinball_p90 | cov_10_90")
        for sp in split_points:
            train = rows_df.iloc[:sp]
            test = rows_df.iloc[sp : min(sp + max(20, (n - sp) // 2), n)]
            if len(train) < 20 or len(test) < 20:
                continue

            predictor = LeveragePredictor()
            for row in train.itertuples(index=False):
                predictor.add_training_row(
                    TrainingRow(
                        dte=float(row.dte),
                        moneyness=float(row.moneyness),
                        iv=float(row.iv),
                        spread_pct=float(row.spread_pct),
                        option_type=str(row.option_type),
                        btc_change_pct=float(row.btc_change_pct),
                        iv_change=float(row.iv_change),
                        days_held=int(row.days_held),
                        option_return_pct=float(row.option_return_pct),
                    )
                )
            predictor.fit(min_samples_per_bucket=min_samples_per_bucket)

            y = test["option_return_pct"].to_numpy(dtype=float)
            p10, p50, p90 = [], [], []
            for row in test.itertuples(index=False):
                pred = predictor.predict(
                    dte=float(row.dte),
                    moneyness=float(row.moneyness),
                    option_type=str(row.option_type),
                    btc_change_pct=float(row.btc_change_pct),
                    iv=float(row.iv),
                    spread_pct=float(row.spread_pct),
                    iv_change=float(row.iv_change),
                    days_held=int(row.days_held),
                )
                p10.append(pred.p10)
                p50.append(pred.p50)
                p90.append(pred.p90)

            p10 = np.array(p10, dtype=float)
            p50 = np.array(p50, dtype=float)
            p90 = np.array(p90, dtype=float)
            mae = float(np.mean(np.abs(y - p50)))
            pin10 = self._pinball(y, p10, 0.10)
            pin90 = self._pinball(y, p90, 0.90)
            cov = float(np.mean((y >= p10) & (y <= p90)))
            self.stdout.write(
                f"{sp:5d} | {len(train):10d} | {len(test):8d} | {mae:7.4f} | {pin10:11.4f} | {pin90:11.4f} | {cov:8.3f}"
            )

    def _print_feature_coverage(self, rows_df: pd.DataFrame) -> None:
        self.stdout.write("\nFeature coverage:")
        self.stdout.write(f"- Date range: {rows_df['entry_ts'].min()} -> {rows_df['entry_ts'].max()}")
        self.stdout.write(f"- Option types: {rows_df['option_type'].value_counts().to_dict()}")
        self.stdout.write(f"- Structures: {rows_df['structure'].value_counts().to_dict()}")
        self.stdout.write(f"- Liquidity regimes: {rows_df['liquidity_regime'].value_counts().to_dict()}")
        self.stdout.write(
            "- BTC change pct quantiles: "
            + str(rows_df["btc_change_pct"].quantile([0.1, 0.5, 0.9]).round(4).to_dict())
        )
        self.stdout.write(
            "- Option return pct quantiles: "
            + str(rows_df["option_return_pct"].quantile([0.1, 0.5, 0.9]).round(4).to_dict())
        )

    def _pinball(self, y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
        err = y_true - y_pred
        return float(np.mean(np.maximum(q * err, (q - 1) * err)))

    def _liquidity_regime(self, spread_pct: float) -> str:
        if spread_pct < 0.02:
            return "tight"
        if spread_pct < 0.05:
            return "normal"
        if spread_pct < 0.10:
            return "wide"
        return "illiquid"

    def _infer_structure(self, moneyness: float, dte: float, spread_pct: float) -> str:
        # Snapshot-only fallback heuristic. Replace with signal-linked structure when available.
        abs_m = abs(moneyness)
        if dte >= 10 and abs_m >= 0.08 and spread_pct <= 0.06:
            return "spread"
        if dte >= 10 and 0.03 <= abs_m <= 0.12 and spread_pct > 0.06:
            return "condor"
        return "naked"
