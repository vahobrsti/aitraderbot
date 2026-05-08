"""
Train option response model from OptionSnapshot history.

Supports two model types:
- bucket: Bucket-based quantile lookup (interpretable, JSON output)
- gbm: Gradient Boosting quantile regression (better generalization, joblib output)

Builds training rows with features:
- DTE, moneyness, IV level and IV change
- time elapsed, spread/liquidity regime
- option direction (call/put)
- BTC underlying move

Usage:
    # Train bucket model (default)
    python manage.py train_option_response --model-type bucket
    
    # Train GBM model
    python manage.py train_option_response --model-type gbm --out-model models/option_response_gbm.joblib
    
    # With walk-forward validation
    python manage.py train_option_response --model-type gbm --walk-forward-splits 5
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand

from execution.services.option_pricer import LeveragePredictor, GBMPredictor, TrainingRow


class Command(BaseCommand):
    help = "Train learned option response mapping from OptionSnapshot pairs"

    def add_arguments(self, parser):
        parser.add_argument(
            "--model-type",
            choices=["bucket", "gbm"],
            default="gbm",
            help="Model type: bucket (JSON lookup) or gbm (sklearn GBM, joblib)"
        )
        parser.add_argument("--horizon-days", type=int, default=1, help="Target pairing horizon in days")
        parser.add_argument("--min-samples-per-bucket", type=int, default=5, help="Min rows per bucket (bucket model)")
        parser.add_argument("--walk-forward-splits", type=int, default=4, help="Time splits for diagnostics")
        parser.add_argument("--min-snapshots-per-symbol", type=int, default=8)
        parser.add_argument("--underlying", type=str, default="BTC")
        parser.add_argument("--start", type=str, default=None, help="Start timestamp/date filter")
        parser.add_argument("--end", type=str, default=None, help="End timestamp/date filter")
        parser.add_argument(
            "--out-model",
            type=str,
            default=None,
            help="Output model path (default: models/option_response_gbm.joblib or .json)"
        )
        parser.add_argument("--out-rows", type=str, default="", help="Optional CSV export of training rows")
        
        # GBM hyperparameters
        parser.add_argument("--n-estimators", type=int, default=200, help="GBM: number of trees")
        parser.add_argument("--max-depth", type=int, default=5, help="GBM: max tree depth")
        parser.add_argument("--learning-rate", type=float, default=0.05, help="GBM: learning rate")
        parser.add_argument("--min-samples-leaf", type=int, default=20, help="GBM: min samples per leaf")

    def handle(self, *args, **options):
        from datafeed.models import OptionSnapshot
        from django.db.models import Count

        model_type = options["model_type"]
        horizon_days = int(options["horizon_days"])
        min_samples_per_bucket = int(options["min_samples_per_bucket"])
        min_snapshots_per_symbol = int(options["min_snapshots_per_symbol"])
        n_splits = int(options["walk_forward_splits"])
        
        # Default output path based on model type
        if options["out_model"]:
            out_model = Path(options["out_model"])
        else:
            if model_type == "gbm":
                out_model = Path("models/option_response_gbm.joblib")
            else:
                out_model = Path("models/option_response_predictor.json")

        self.stdout.write(f"\n{'='*60}")
        self.stdout.write(f"TRAINING OPTION RESPONSE MODEL")
        self.stdout.write(f"{'='*60}")
        self.stdout.write(f"Model type: {model_type.upper()}")
        self.stdout.write(f"Output: {out_model}")
        self.stdout.write(f"Horizon: {horizon_days} day(s)")

        # Build base queryset
        base_qs = OptionSnapshot.objects.filter(underlying=options["underlying"])
        if options["start"]:
            base_qs = base_qs.filter(timestamp__gte=options["start"])
        if options["end"]:
            base_qs = base_qs.filter(timestamp__lte=options["end"])
        
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

        # Train based on model type
        if model_type == "gbm":
            self._train_gbm(rows_df, out_model, options, n_splits)
        else:
            self._train_bucket(rows_df, out_model, min_samples_per_bucket, n_splits)

    def _train_gbm(self, rows_df: pd.DataFrame, out_model: Path, options: dict, n_splits: int) -> None:
        """Train GBM quantile regression model."""
        self.stdout.write(f"\n{'='*60}")
        self.stdout.write("TRAINING GBM MODEL")
        self.stdout.write(f"{'='*60}")
        
        # Walk-forward evaluation first
        if n_splits >= 2:
            self._walk_forward_eval_gbm(rows_df, options, n_splits)
        
        # Train final model on all data
        self.stdout.write("\nTraining final model on all data...")
        predictor = GBMPredictor()
        
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
        
        metrics = predictor.fit(
            n_estimators=options["n_estimators"],
            max_depth=options["max_depth"],
            learning_rate=options["learning_rate"],
            min_samples_leaf=options["min_samples_leaf"],
        )
        
        self.stdout.write(f"\nTraining metrics:")
        self.stdout.write(f"  MAE (p50): {metrics.get('mae_p50', 0):.4f}")
        self.stdout.write(f"  Coverage (10-90): {metrics.get('coverage_10_90', 0):.1%}")
        for q in [10, 25, 50, 75, 90]:
            self.stdout.write(f"  Pinball p{q}: {metrics.get(f'pinball_p{q}', 0):.4f}")
        
        # Feature importance
        importance = predictor.get_feature_importance()
        if importance:
            self.stdout.write(f"\nFeature importance (p50 model):")
            p50_imp = importance.get('p50', {})
            for feat, imp in sorted(p50_imp.items(), key=lambda x: -x[1]):
                self.stdout.write(f"  {feat}: {imp:.3f}")
        
        predictor.save(out_model)
        self.stdout.write(self.style.SUCCESS(f"\nSaved GBM predictor: {out_model}"))

    def _walk_forward_eval_gbm(self, rows_df: pd.DataFrame, options: dict, n_splits: int) -> None:
        """Walk-forward evaluation for GBM model."""
        n = len(rows_df)
        if n < 100:
            self.stdout.write("Skipping walk-forward (insufficient rows).")
            return

        split_points = [int(n * k / (n_splits + 1)) for k in range(1, n_splits + 1)]
        
        self.stdout.write("\nWalk-forward diagnostics (GBM):")
        self.stdout.write("split | train | test  | mae_p50 | pin_p10 | pin_p90 | cov_10_90")
        self.stdout.write("-" * 70)
        
        for sp in split_points:
            train = rows_df.iloc[:sp]
            test = rows_df.iloc[sp:min(sp + max(50, (n - sp) // 2), n)]
            
            if len(train) < 100 or len(test) < 50:
                continue
            
            predictor = GBMPredictor()
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
            
            predictor.fit(
                n_estimators=options["n_estimators"],
                max_depth=options["max_depth"],
                learning_rate=options["learning_rate"],
                min_samples_leaf=options["min_samples_leaf"],
            )
            
            # Evaluate on test set
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
            
            p10 = np.array(p10)
            p50 = np.array(p50)
            p90 = np.array(p90)
            
            mae = float(np.mean(np.abs(y - p50)))
            pin10 = self._pinball(y, p10, 0.10)
            pin90 = self._pinball(y, p90, 0.90)
            cov = float(np.mean((y >= p10) & (y <= p90)))
            
            self.stdout.write(
                f"{sp:5d} | {len(train):5d} | {len(test):5d} | {mae:7.4f} | {pin10:7.4f} | {pin90:7.4f} | {cov:8.1%}"
            )

    def _train_bucket(self, rows_df: pd.DataFrame, out_model: Path, min_samples_per_bucket: int, n_splits: int) -> None:
        """Train bucket-based quantile model."""
        self.stdout.write(f"\n{'='*60}")
        self.stdout.write("TRAINING BUCKET MODEL")
        self.stdout.write(f"{'='*60}")
        
        # Walk-forward evaluation
        if n_splits >= 2:
            self._walk_forward_eval_bucket(rows_df, min_samples_per_bucket, n_splits)
        
        # Train final model
        self.stdout.write("\nTraining final model on all data...")
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
        predictor.save(out_model)
        self.stdout.write(self.style.SUCCESS(f"\nSaved bucket predictor: {out_model}"))
        self.stdout.write(f"Buckets created: {len(predictor.bucket_stats)}")

    def _walk_forward_eval_bucket(self, rows_df: pd.DataFrame, min_samples_per_bucket: int, n_splits: int) -> None:
        """Walk-forward evaluation for bucket model."""
        n = len(rows_df)
        if n < 50:
            self.stdout.write("Skipping walk-forward (insufficient rows).")
            return

        split_points = [int(n * k / (n_splits + 1)) for k in range(1, n_splits + 1)]
        
        self.stdout.write("\nWalk-forward diagnostics (Bucket):")
        self.stdout.write("split | train | test  | mae_p50 | pin_p10 | pin_p90 | cov_10_90")
        self.stdout.write("-" * 70)
        
        for sp in split_points:
            train = rows_df.iloc[:sp]
            test = rows_df.iloc[sp:min(sp + max(20, (n - sp) // 2), n)]
            
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

            p10 = np.array(p10)
            p50 = np.array(p50)
            p90 = np.array(p90)
            
            mae = float(np.mean(np.abs(y - p50)))
            pin10 = self._pinball(y, p10, 0.10)
            pin90 = self._pinball(y, p90, 0.90)
            cov = float(np.mean((y >= p10) & (y <= p90)))
            
            self.stdout.write(
                f"{sp:5d} | {len(train):5d} | {len(test):5d} | {mae:7.4f} | {pin10:7.4f} | {pin90:7.4f} | {cov:8.1%}"
            )

    def _build_rows_df(self, qs, symbols, horizon_days: int) -> pd.DataFrame:
        from datetime import timedelta

        rows = []
        for symbol in symbols:
            snaps = list(
                qs.filter(symbol=symbol).values(
                    "timestamp", "symbol", "option_type", "spot_price",
                    "strike", "mid_price", "iv", "spread_pct", "dte",
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
                moneyness = (strike - entry_spot) / entry_spot
                iv_entry = float(entry["iv"] or 0.60)
                iv_exit = float(exit_snap["iv"] or iv_entry)
                spread_pct = float(entry["spread_pct"] or 0.02)
                days_held = max(1, int((exit_snap["timestamp"] - entry_ts).total_seconds() // 86400))

                rows.append({
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
                })

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("entry_ts").reset_index(drop=True)

    def _print_feature_coverage(self, rows_df: pd.DataFrame) -> None:
        self.stdout.write(f"\n{'='*60}")
        self.stdout.write("FEATURE COVERAGE")
        self.stdout.write(f"{'='*60}")
        self.stdout.write(f"Date range: {rows_df['entry_ts'].min()} -> {rows_df['entry_ts'].max()}")
        self.stdout.write(f"Option types: {rows_df['option_type'].value_counts().to_dict()}")
        self.stdout.write(f"Structures: {rows_df['structure'].value_counts().to_dict()}")
        self.stdout.write(f"Liquidity regimes: {rows_df['liquidity_regime'].value_counts().to_dict()}")
        self.stdout.write(
            f"BTC change quantiles: {rows_df['btc_change_pct'].quantile([0.1, 0.5, 0.9]).round(4).to_dict()}"
        )
        self.stdout.write(
            f"Option return quantiles: {rows_df['option_return_pct'].quantile([0.1, 0.5, 0.9]).round(4).to_dict()}"
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
        abs_m = abs(moneyness)
        if dte >= 10 and abs_m >= 0.08 and spread_pct <= 0.06:
            return "spread"
        if dte >= 10 and 0.03 <= abs_m <= 0.12 and spread_pct > 0.06:
            return "condor"
        return "naked"
