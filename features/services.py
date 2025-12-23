"""
Service layer for signal generation and persistence.
Encapsulates ML scoring, fusion engine, and database operations.
"""
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from datafeed.models import RawDailyData
from features.feature_builder import build_features_and_labels_from_raw
from features.models import DailySignal
from features.signals.fusion import fuse_signals, MarketState
from features.signals.overlays import apply_overlays, get_size_multiplier, get_dte_multiplier
from features.signals.tactical_puts import tactical_put_inside_bull


@dataclass
class SignalResult:
    """Container for all signal outputs."""
    date: date
    p_long: float
    p_short: float
    signal_option_call: int
    signal_option_put: int
    fusion_state: str
    fusion_confidence: str
    fusion_score: int
    overlay_reason: str
    size_multiplier: float
    dte_multiplier: float
    tactical_put_active: bool
    tactical_put_strategy: str
    tactical_put_size: float
    trade_decision: str
    trade_notes: str


class SignalService:
    """
    Service for generating and persisting daily trading signals.
    
    Usage:
        service = SignalService()
        result = service.generate_signal()
        service.persist_signal(result)
    """
    
    def __init__(
        self,
        long_model_path: str = "models/long_model.joblib",
        short_model_path: str = "models/short_model.joblib",
        horizon_days: int = 14,
        target_return: float = 0.05,
    ):
        self.long_model_path = Path(long_model_path)
        self.short_model_path = Path(short_model_path)
        self.horizon_days = horizon_days
        self.target_return = target_return
        
        # Lazy load models
        self._long_bundle = None
        self._short_bundle = None
    
    def _load_models(self):
        """Load ML models if not already loaded."""
        if self._long_bundle is None:
            self._long_bundle = joblib.load(self.long_model_path)
        if self._short_bundle is None:
            self._short_bundle = joblib.load(self.short_model_path)
    
    def generate_signal(self, target_date: Optional[date] = None) -> SignalResult:
        """
        Generate signal for the specified date (or latest available).
        
        Args:
            target_date: Specific date to score, or None for latest.
            
        Returns:
            SignalResult with all signal components.
        """
        self._load_models()
        
        # 1) Pull raw data
        qs = RawDailyData.objects.order_by("date")
        df_raw = pd.DataFrame.from_records(qs.values())
        
        if df_raw.empty:
            raise ValueError("No RawDailyData found in database")
        
        # 2) Build features
        feats = build_features_and_labels_from_raw(
            df_raw,
            horizon_days=self.horizon_days,
            target_return=self.target_return,
        )
        
        # 3) Get target row
        if target_date is not None:
            if target_date not in feats.index:
                raise ValueError(f"No features available for {target_date}")
            latest_date = target_date
        else:
            latest_date = feats.index.max()
        
        latest_row = feats.loc[[latest_date]]
        row = feats.loc[latest_date]
        
        # 4) ML Predictions
        long_model = self._long_bundle["model"]
        long_feats = self._long_bundle["feature_names"]
        short_model = self._short_bundle["model"]
        short_feats = self._short_bundle["feature_names"]
        
        p_long = float(long_model.predict_proba(latest_row[long_feats])[:, 1][0])
        p_short = float(short_model.predict_proba(latest_row[short_feats])[:, 1][0])
        
        signal_option_call = int(latest_row["signal_option_call"].iloc[0])
        signal_option_put = int(latest_row["signal_option_put"].iloc[0])
        
        # 5) Fusion Engine
        fusion_result = fuse_signals(row)
        overlay = apply_overlays(fusion_result, row)
        size_mult = get_size_multiplier(overlay)
        dte_mult = get_dte_multiplier(overlay)
        
        # 6) Tactical Put
        tactical_result = tactical_put_inside_bull(fusion_result, row)
        
        # 7) Determine trade decision
        trade_decision, trade_notes = self._determine_trade_decision(
            fusion_result, size_mult, tactical_result, p_long, p_short,
            signal_option_call, signal_option_put
        )
        
        return SignalResult(
            date=latest_date,
            p_long=p_long,
            p_short=p_short,
            signal_option_call=signal_option_call,
            signal_option_put=signal_option_put,
            fusion_state=fusion_result.state.value,
            fusion_confidence=fusion_result.confidence.value,
            fusion_score=fusion_result.score,
            overlay_reason=overlay.reason,
            size_multiplier=size_mult,
            dte_multiplier=dte_mult,
            tactical_put_active=tactical_result.active,
            tactical_put_strategy=tactical_result.strategy.value if tactical_result.active else "",
            tactical_put_size=tactical_result.size_mult if tactical_result.active else 0.0,
            trade_decision=trade_decision,
            trade_notes=trade_notes,
        )
    
    def _determine_trade_decision(
        self, fusion_result, size_mult, tactical_result,
        p_long, p_short, signal_option_call, signal_option_put
    ) -> tuple[str, str]:
        """Determine final trade decision and notes."""
        notes_parts = []
        
        # Check for tactical put first
        if tactical_result.active:
            return "TACTICAL_PUT", tactical_result.reason
        
        # Check fusion-based trades
        if fusion_result.state != MarketState.NO_TRADE and size_mult > 0:
            is_long = fusion_result.state in {
                MarketState.STRONG_BULLISH,
                MarketState.EARLY_RECOVERY,
                MarketState.MOMENTUM_CONTINUATION
            }
            is_short = fusion_result.state in {
                MarketState.DISTRIBUTION_RISK,
                MarketState.BEAR_CONTINUATION
            }
            
            if is_long:
                notes_parts.append(f"Fusion: {fusion_result.state.value}")
                return "CALL", "; ".join(notes_parts)
            
            if is_short:
                notes_parts.append(f"Fusion: {fusion_result.state.value}")
                return "PUT", "; ".join(notes_parts)
        
        # Check probes
        if fusion_result.state == MarketState.BULL_PROBE and size_mult > 0:
            return "CALL", f"Bull probe, score={fusion_result.score}"
        
        if fusion_result.state == MarketState.BEAR_PROBE and size_mult > 0:
            return "PUT", f"Bear probe, score={fusion_result.score}"
        
        # No trade
        if size_mult == 0:
            return "NO_TRADE", "Overlay vetoed (size_mult=0)"
        
        return "NO_TRADE", f"State: {fusion_result.state.value}"
    
    def persist_signal(self, result: SignalResult) -> DailySignal:
        """
        Persist signal to database, updating if date already exists.
        
        Args:
            result: SignalResult to persist.
            
        Returns:
            DailySignal model instance.
        """
        signal, created = DailySignal.objects.update_or_create(
            date=result.date,
            defaults={
                'p_long': result.p_long,
                'p_short': result.p_short,
                'signal_option_call': result.signal_option_call,
                'signal_option_put': result.signal_option_put,
                'fusion_state': result.fusion_state,
                'fusion_confidence': result.fusion_confidence,
                'fusion_score': result.fusion_score,
                'overlay_reason': result.overlay_reason,
                'size_multiplier': result.size_multiplier,
                'dte_multiplier': result.dte_multiplier,
                'tactical_put_active': result.tactical_put_active,
                'tactical_put_strategy': result.tactical_put_strategy,
                'tactical_put_size': result.tactical_put_size,
                'trade_decision': result.trade_decision,
                'trade_notes': result.trade_notes,
            }
        )
        return signal
    
    def generate_and_persist(self, target_date: Optional[date] = None) -> DailySignal:
        """
        Convenience method to generate and persist in one call.
        
        Args:
            target_date: Specific date to score, or None for latest.
            
        Returns:
            DailySignal model instance.
        """
        result = self.generate_signal(target_date)
        return self.persist_signal(result)
