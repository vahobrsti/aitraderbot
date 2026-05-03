"""
Calibrate execution policy parameters from path analysis metrics.

Builds per-signal entry/exit overrides and writes:
execution/data/policy_calibration.json
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from django.core.management.base import BaseCommand

from signals.management.commands.analyze_path_stats import Command as PathStatsCommand


TYPE_TO_SIGNAL = {
    "LONG": "CALL",
    "PRIMARY_SHORT": "PUT",
    "BULL_PROBE": "BULL_PROBE",
    "BEAR_PROBE": "BEAR_PROBE",
    "TACTICAL_PUT": "TACTICAL_PUT",
    "OPTION_CALL": "OPTION_CALL",
    "OPTION_PUT": "OPTION_PUT",
    "MVRV_SHORT": "MVRV_SHORT",
    "IRON_CONDOR": "IRON_CONDOR",
}


class Command(BaseCommand):
    help = "Calibrate policy from analyze_path_stats and write JSON overrides"

    def add_arguments(self, parser):
        parser.add_argument("--csv", type=str, default="features_14d_5pct.csv")
        parser.add_argument("--horizon", type=int, default=14)
        parser.add_argument("--target", type=float, default=0.05)
        parser.add_argument("--invalidation", type=float, default=0.04)
        parser.add_argument("--year", type=int, default=None)
        parser.add_argument("--min-samples", type=int, default=12)

    def handle(self, *args, **options):
        cmd = PathStatsCommand()
        trades_df = cmd._build_trades_df(
            Path(options["csv"]),
            {
                "csv": options["csv"],
                "long_model": "models/long_model.joblib",
                "short_model": "models/short_model.joblib",
            },
            options.get("year"),
            no_overlay=False,
            no_cooldown=False,
        )
        if trades_df.empty:
            self.stdout.write(self.style.ERROR("No trades found for calibration."))
            return

        price_df = cmd._load_prices()
        metrics_df = cmd._compute_path_metrics(
            trades_df,
            price_df,
            int(options["horizon"]),
            float(options["target"]),
            float(options["invalidation"]),
        )
        if metrics_df.empty:
            self.stdout.write(self.style.ERROR("No path metrics available for calibration."))
            return

        payload = self._build_payload(
            metrics_df=metrics_df,
            horizon=int(options["horizon"]),
            min_samples=int(options["min_samples"]),
        )

        out_path = Path("execution/data/policy_calibration.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        self.stdout.write(self.style.SUCCESS(f"Calibration saved to {out_path}"))
        self.stdout.write(json.dumps(payload["metadata"], indent=2))

    def _build_payload(self, metrics_df: pd.DataFrame, horizon: int, min_samples: int) -> dict:
        out = {
            "metadata": {
                "source": "analyze_path_stats",
                "horizon_days": horizon,
                "min_samples": min_samples,
                "total_trades": int(len(metrics_df)),
            },
            "dte_targets": {},
            "spread_width_pct": {},
            "exit_params": {},
            "expected_edge_by_signal": {},
        }

        for trade_type, group in metrics_df.groupby("type"):
            signal = TYPE_TO_SIGNAL.get(str(trade_type))
            if not signal or len(group) < min_samples:
                continue

            winners = group[group["hit"] == 1]
            hit_rate = float(group["hit"].mean())
            
            # IRON_CONDOR uses different metrics (range-bound, no TTH)
            is_condor = signal == "IRON_CONDOR"
            
            if is_condor:
                # For condors: use 7-day horizon, MAE is max move in either direction
                tth_p75 = 7  # Fixed 7-day hold for condors
                tth_med = 7
                # MFE for condors is inverted (1 - max_move), so we need MAE instead
                mae_w_p75 = float(winners["mae"].quantile(0.75)) if len(winners) else 0.08
                # Spread width for condors is the wing width (typically 10%)
                mfe_p75 = 0.10  # Standard condor wing width
            else:
                # Directional trades: use TTH-based calibration
                tth_series = winners["tth"].dropna()
                tth_p75 = int(round(tth_series.quantile(0.75))) if len(tth_series) else max(3, int(horizon * 0.5))
                tth_med = int(round(tth_series.median())) if len(tth_series) else max(2, int(horizon * 0.35))
                mfe_p75 = float(group["mfe"].quantile(0.75))
                mae_w_p75 = float(winners["mae"].quantile(0.75)) if len(winners) else float(group["mae"].quantile(0.50))

            min_dte = max(7, tth_p75 + 2)
            max_dte = min(max(min_dte + 2, tth_p75 + 6), horizon + 10)
            optimal_dte = max(min_dte, min(max_dte, tth_med + 2))

            spread_width = min(max(mfe_p75, 0.05), 0.15)
            stop_loss = min(max(mae_w_p75, 0.02), 0.08)
            
            # Take-profit is % of max profit to capture, NOT related to spread width
            # For debit spreads: 50-70% of max profit is standard
            # Higher win rate signals can afford to take profit earlier
            # Lower win rate signals need to let winners run
            if is_condor:
                take_profit = 0.50  # Condors: take 50% of max credit
            elif hit_rate >= 0.80:
                take_profit = 0.50  # High win rate: take profit early
            elif hit_rate >= 0.65:
                take_profit = 0.60  # Medium win rate: balanced
            else:
                take_profit = 0.70  # Lower win rate: let winners run
            
            max_hold = max(3, min(horizon, tth_p75 + 2))
            scale_down = max(2, min(max_hold - 1, tth_med + 1)) if max_hold >= 4 else None

            out["dte_targets"][signal] = {
                "min_dte": int(min_dte),
                "max_dte": int(max_dte),
                "optimal_dte": int(optimal_dte),
            }
            out["spread_width_pct"][signal] = round(spread_width, 4)
            out["exit_params"][signal] = {
                "stop_loss_pct": round(stop_loss, 4),
                "take_profit_pct": round(take_profit, 4),
                "max_hold_days": int(max_hold),
                "scale_down_day": (int(scale_down) if scale_down is not None else None),
            }
            out["expected_edge_by_signal"][signal] = round(hit_rate * spread_width, 4)

        return out
