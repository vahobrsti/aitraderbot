# features/management/commands/score_latest.py
"""
Score the most recent day(s) using trained models and signal fusion.
Includes tactical puts detection for puts inside bull regimes.
"""

from pathlib import Path

import joblib
import pandas as pd
from django.core.management.base import BaseCommand

from datafeed.models import RawDailyData
from features.feature_builder import build_features_and_labels_from_raw
from features.signals.fusion import fuse_signals, MarketState, Confidence, FusionResult
from features.signals.overlays import apply_overlays, get_size_multiplier, get_dte_multiplier
from features.signals.tactical_puts import tactical_put_inside_bull, TacticalPutStrategy
from features.signals.options import get_strategy, generate_trade_signal


class Command(BaseCommand):
    help = "Score the most recent day(s) using trained models and print suggested actions."

    def add_arguments(self, parser):
        parser.add_argument(
            "--days",
            type=int,
            default=120,
            help="How many recent days of raw data to pull for feature building.",
        )
        parser.add_argument(
            "--horizon",
            type=int,
            default=14,
        )
        parser.add_argument(
            "--target_return",
            type=float,
            default=0.05,
        )
        parser.add_argument(
            "--long_model",
            type=str,
            default="models/long_model.joblib",
        )
        parser.add_argument(
            "--short_model",
            type=str,
            default="models/short_model.joblib",
        )

    def handle(self, *args, **options):
        days = options["days"]
        horizon_days = options["horizon"]
        target_return = options["target_return"]

        long_model_path = Path(options["long_model"])
        short_model_path = Path(options["short_model"])

        # 1) Pull recent raw data
        qs = RawDailyData.objects.order_by("date")  # all rows, ascending
        df_raw = pd.DataFrame.from_records(qs.values())

        if df_raw.empty:
            self.stderr.write(self.style.ERROR("No RawDailyData found"))
            return

        # 2) Build features (this also builds labels, but we can ignore labels)
        feats = build_features_and_labels_from_raw(
            df_raw,
            horizon_days=horizon_days,
            target_return=target_return,
        )

        # Keep only the last row (most recent date with full horizon)
        latest_date = feats.index.max()
        latest_row = feats.loc[[latest_date]]
        row = feats.loc[latest_date]

        # 3) Load models
        long_bundle = joblib.load(long_model_path)
        short_bundle = joblib.load(short_model_path)

        long_model = long_bundle["model"]
        long_feats = long_bundle["feature_names"]

        short_model = short_bundle["model"]
        short_feats = short_bundle["feature_names"]

        # 4) ML Score
        p_long = float(long_model.predict_proba(latest_row[long_feats])[:, 1][0])
        p_short = float(short_model.predict_proba(latest_row[short_feats])[:, 1][0])

        signal_option_call = int(latest_row["signal_option_call"].iloc[0])
        signal_option_put = int(latest_row["signal_option_put"].iloc[0])

        # 5) Fusion Analysis
        fusion_result = fuse_signals(row)
        overlay = apply_overlays(fusion_result, row)
        size_mult = get_size_multiplier(overlay)
        dte_mult = get_dte_multiplier(overlay)
        
        # 6) Tactical Put Check
        tactical_result = tactical_put_inside_bull(fusion_result, row)

        # === OUTPUT ===
        self.stdout.write("")
        self.stdout.write("=" * 60)
        self.stdout.write(self.style.MIGRATE_HEADING(f"SIGNAL ANALYSIS: {latest_date}"))
        self.stdout.write("=" * 60)
        
        # ML Scores
        self.stdout.write("\n--- ML MODEL SCORES ---")
        self.stdout.write(f"p_long  = {p_long:.3f}")
        self.stdout.write(f"p_short = {p_short:.3f}")
        self.stdout.write(f"signal_option_call = {signal_option_call}")
        self.stdout.write(f"signal_option_put  = {signal_option_put}")
        
        # Fusion State
        state_emoji = "🟢" if fusion_result.state in {MarketState.STRONG_BULLISH, MarketState.EARLY_RECOVERY, MarketState.MOMENTUM_CONTINUATION} else \
                     "🔴" if fusion_result.state in {MarketState.DISTRIBUTION_RISK, MarketState.BEAR_CONTINUATION} else "⚪"
        
        self.stdout.write("\n--- FUSION STATE ---")
        self.stdout.write(f"{state_emoji} State: {fusion_result.state.value}")
        self.stdout.write(f"   Confidence: {fusion_result.confidence.value}")
        self.stdout.write(f"   Score: {fusion_result.score:+d}")
        
        # Overlay
        self.stdout.write("\n--- OVERLAY ---")
        self.stdout.write(f"   {overlay.reason}")
        self.stdout.write(f"   Size Multiplier: {size_mult:.2f}")
        self.stdout.write(f"   DTE Multiplier: {dte_mult:.2f}")
        
        # Tactical Put
        if tactical_result.active:
            str_label = "FULL" if tactical_result.strength == 2 else "PARTIAL"
            self.stdout.write("\n--- 🔻 TACTICAL PUT TRIGGERED ---")
            self.stdout.write(f"   Strength: {str_label}")
            self.stdout.write(f"   Strategy: {tactical_result.strategy.value}")
            self.stdout.write(f"   Size: {tactical_result.size_mult:.2f}")
            self.stdout.write(f"   DTE: {tactical_result.dte_mult:.2f}")
            self.stdout.write(f"   Reason: {tactical_result.reason}")
        
        # === DECISION LOGIC ===
        self.stdout.write("\n" + "=" * 60)
        self.stdout.write("TRADE DECISIONS")
        self.stdout.write("=" * 60)
        
        decisions = []
        
        # Check fusion-based trades
        if fusion_result.state != MarketState.NO_TRADE and size_mult > 0:
            is_long = fusion_result.state in {MarketState.STRONG_BULLISH, MarketState.EARLY_RECOVERY, MarketState.MOMENTUM_CONTINUATION}
            is_short = fusion_result.state in {MarketState.DISTRIBUTION_RISK, MarketState.BEAR_CONTINUATION}
            
            if is_long:
                strategy = get_strategy(fusion_result.state)
                decisions.append({
                    'type': 'CALL',
                    'reason': f"Fusion: {fusion_result.state.value}",
                    'confidence': fusion_result.confidence.value,
                    'size': size_mult,
                    'structures': [s.value for s in strategy.primary_structures],
                })
            
            if is_short:
                strategy = get_strategy(fusion_result.state)
                decisions.append({
                    'type': 'PUT',
                    'reason': f"Fusion: {fusion_result.state.value}",
                    'confidence': fusion_result.confidence.value,
                    'size': size_mult,
                    'structures': [s.value for s in strategy.primary_structures],
                })
        
        # Check tactical put
        if tactical_result.active:
            decisions.append({
                'type': 'TACTICAL PUT',
                'reason': tactical_result.reason,
                'confidence': 'tactical',
                'size': tactical_result.size_mult,
                'structures': [tactical_result.strategy.value],
            })
        
        # Print decisions
        if decisions:
            for d in decisions:
                self.stdout.write(f"\n✅ {d['type']}")
                self.stdout.write(f"   Reason: {d['reason'][:50]}")
                self.stdout.write(f"   Confidence: {d['confidence']}")
                self.stdout.write(f"   Size: {d['size']:.2f}")
                self.stdout.write(f"   Structures: {', '.join(d['structures'])}")
        else:
            self.stdout.write("\n⚪ NO TRADE")
            self.stdout.write(f"   State: {fusion_result.state.value}")
            if size_mult == 0:
                self.stdout.write(f"   Reason: Overlay vetoed (size_mult=0)")
        
        # Legacy simple decision (for backwards compatibility)
        self.stdout.write("\n" + "-" * 60)
        decision = "HOLD / NO TRADE"
        if p_long > 0.7 and signal_option_call == 1 and p_short < 0.5:
            decision = "BUY CALL (ML bullish setup)"
        elif p_short > 0.7 and signal_option_put == 1 and p_long < 0.5:
            decision = "BUY PUT (ML bearish setup)"
        elif tactical_result.active:
            decision = f"TACTICAL PUT ({tactical_result.strategy.value})"
        
        self.stdout.write(self.style.SUCCESS(f"Legacy Decision: {decision}"))
        self.stdout.write("")
