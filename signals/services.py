"""
Service layer for signal generation and persistence.
Encapsulates ML scoring, fusion engine, and database operations.
"""
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from datafeed.models import RawDailyData
from features.feature_builder import build_features_and_labels_from_raw
from signals.models import DailySignal
from signals.fusion import fuse_signals, MarketState
from signals.overlays import apply_overlays, get_size_multiplier, get_dte_multiplier
from signals.tactical_puts import tactical_put_inside_bull
from signals.options import get_strategy_summary

# Option signal constants
OPTION_SIGNAL_COOLDOWN_DAYS = 7
OPTION_SIGNAL_SIZE_MULT = 0.50


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
    # Option strategy fields
    option_structures: str
    strike_guidance: str
    dte_range: str
    strategy_rationale: str
    # NO_TRADE diagnostics
    no_trade_reasons: list
    decision_trace: list
    score_components: dict
    effective_size: float
    # Versioning
    decision_version: str
    model_versions: dict
    # Short signal source tracking
    short_source: Optional[str] = None  # 'rule' or 'score' for short setups


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
        
        # 7) Option signal cooldown check
        option_call_ok = signal_option_call == 1 and self._check_option_cooldown(latest_date, "OPTION_CALL")
        option_put_ok = signal_option_put == 1 and self._check_option_cooldown(latest_date, "OPTION_PUT")
        
        trade_decision, trade_notes, no_trade_reasons, decision_trace = self._determine_trade_decision(
            fusion_result, size_mult, dte_mult, tactical_result, p_long, p_short,
            signal_option_call, signal_option_put, overlay,
            option_call_ok=option_call_ok, option_put_ok=option_put_ok,
        )
        
        # 8) Get option strategy recommendation
        strategy_summary = get_strategy_summary(fusion_result.state)
        
        # Compute effective values
        if trade_decision == "NO_TRADE":
            effective_size = 0.0
            size_mult = 0.0
            dte_mult = 0.0
        elif trade_decision in ("OPTION_CALL", "OPTION_PUT"):
            # Option signals use reduced sizing
            effective_size = OPTION_SIGNAL_SIZE_MULT
            size_mult = OPTION_SIGNAL_SIZE_MULT
        else:
            effective_size = size_mult
        
        # Build model version info
        model_versions = {
            'long': self.long_model_path.name,
            'short': self.short_model_path.name,
        }
        decision_version = "2025-12-28.1"
        
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
            option_structures=strategy_summary["primary_structures"],
            strike_guidance=strategy_summary["strike_guidance"],
            dte_range=strategy_summary["dte_range"],
            strategy_rationale=strategy_summary["rationale"],
            no_trade_reasons=no_trade_reasons,
            decision_trace=decision_trace,
            score_components=fusion_result.components,
            effective_size=effective_size,
            decision_version=decision_version,
            model_versions=model_versions,
            short_source=fusion_result.short_source,
        )
    
    def _check_option_cooldown(self, current_date, decision_type: str) -> bool:
        """
        Check if enough days have passed since last OPTION_CALL/OPTION_PUT trade.
        Returns True if cooldown has passed (OK to trade).
        """
        cutoff = current_date - timedelta(days=OPTION_SIGNAL_COOLDOWN_DAYS)
        last_signal = (
            DailySignal.objects
            .filter(trade_decision=decision_type, date__gte=cutoff, date__lt=current_date)
            .order_by("-date")
            .first()
        )
        return last_signal is None

    def _determine_trade_decision(
        self, fusion_result, size_mult, dte_mult, tactical_result,
        p_long, p_short, signal_option_call, signal_option_put, overlay,
        option_call_ok=False, option_put_ok=False,
    ) -> tuple[str, str, list, list]:
        """
        Determine final trade decision, notes, and diagnostics.
        
        Priority:
            1. Tactical Put
            2. Fusion state (CALL/PUT/probes)
            3. Option signal fallback (when fusion=NO_TRADE, with cooldown + overlay)
            4. NO_TRADE
        
        Returns:
            Tuple of (decision, notes, no_trade_reasons, decision_trace)
        """
        no_trade_reasons = []
        decision_trace = []
        
        # Build decision trace step by step
        fusion_step = f"fusion={fusion_result.state.value}(score={fusion_result.score},conf={fusion_result.confidence.value})"
        decision_trace.append(fusion_step)
        
        # Check for tactical put first
        if tactical_result.active:
            decision_trace.append(f"tactical_put=active(strategy={tactical_result.strategy.value}) -> TRADE")
            return "TACTICAL_PUT", tactical_result.reason, [], decision_trace
        else:
            decision_trace.append("tactical_put=inactive")
        
        # Check fusion state — if fusion has a directional view, use it
        if fusion_result.state != MarketState.NO_TRADE:
            # Check confidence
            if fusion_result.confidence.value == "low":
                no_trade_reasons.append("CONFIDENCE_TOO_LOW")
            
            # Check overlay
            overlay_step = f"overlay(size={size_mult},dte={dte_mult},reason={overlay.reason[:30]})"
            decision_trace.append(overlay_step)
            
            if size_mult == 0:
                no_trade_reasons.append("OVERLAY_VETO")
                decision_trace.append("stop_reason=OVERLAY_VETO")
                return "NO_TRADE", "Overlay vetoed (size_mult=0)", no_trade_reasons, decision_trace
            
            # Check fusion-based trades
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
                decision_trace.append(f"state={fusion_result.state.value} -> CALL")
                return "CALL", f"Fusion: {fusion_result.state.value}", [], decision_trace
            
            if is_short:
                source_label = f"[{fusion_result.short_source}]" if fusion_result.short_source else ""
                decision_trace.append(f"state={fusion_result.state.value} {source_label} -> PUT")
                return "PUT", f"Fusion: {fusion_result.state.value} {source_label}".strip(), [], decision_trace
            
            # Check probes
            if fusion_result.state == MarketState.BULL_PROBE:
                decision_trace.append(f"bull_probe(score={fusion_result.score}) -> CALL")
                return "CALL", f"Bull probe, score={fusion_result.score}", [], decision_trace
            
            if fusion_result.state == MarketState.BEAR_PROBE:
                source_label = f"[{fusion_result.short_source}]" if fusion_result.short_source else ""
                decision_trace.append(f"bear_probe(score={fusion_result.score}) {source_label} -> PUT")
                return "PUT", f"Bear probe, score={fusion_result.score} {source_label}".strip(), [], decision_trace
            
            # Catch-all for non-tradeable fusion states
            no_trade_reasons.append("STATE_NOT_TRADEABLE")
            decision_trace.append("stop_reason=STATE_NOT_TRADEABLE")
            return "NO_TRADE", f"State: {fusion_result.state.value}", no_trade_reasons, decision_trace
        
        # --- Fusion is NO_TRADE — check option signal fallback ---
        decision_trace.append("fusion=NO_TRADE")
        
        # Option CALL fallback
        if option_call_ok:
            # Apply overlay check
            if size_mult == 0:
                decision_trace.append("option_call=fired but overlay_veto -> NO_TRADE")
                no_trade_reasons.append("FUSION_STATE_NO_TRADE")
                no_trade_reasons.append("OPTION_CALL_OVERLAY_VETO")
                return "NO_TRADE", "Option call fired but overlay vetoed", no_trade_reasons, decision_trace
            decision_trace.append(f"option_call=fired(cooldown_ok) -> OPTION_CALL (size={OPTION_SIGNAL_SIZE_MULT})")
            return "OPTION_CALL", "Rule: MVRV cheap (2+ flags) + Sentiment fear", [], decision_trace
        elif signal_option_call == 1:
            decision_trace.append("option_call=fired but cooldown_active -> skip")
        
        # Option PUT fallback
        if option_put_ok:
            if size_mult == 0:
                decision_trace.append("option_put=fired but overlay_veto -> NO_TRADE")
                no_trade_reasons.append("FUSION_STATE_NO_TRADE")
                no_trade_reasons.append("OPTION_PUT_OVERLAY_VETO")
                return "NO_TRADE", "Option put fired but overlay vetoed", no_trade_reasons, decision_trace
            decision_trace.append(f"option_put=fired(cooldown_ok) -> OPTION_PUT (size={OPTION_SIGNAL_SIZE_MULT})")
            return "OPTION_PUT", "Rule: MVRV overheated + Sentiment greed + Whale distrib", [], decision_trace
        elif signal_option_put == 1:
            decision_trace.append("option_put=fired but cooldown_active -> skip")
        
        # Default NO_TRADE
        no_trade_reasons.append("FUSION_STATE_NO_TRADE")
        decision_trace.append("stop_reason=FUSION_STATE_NO_TRADE")
        return "NO_TRADE", f"State: {fusion_result.state.value}", no_trade_reasons, decision_trace
    
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
                'option_structures': result.option_structures,
                'strike_guidance': result.strike_guidance,
                'dte_range': result.dte_range,
                'strategy_rationale': result.strategy_rationale,
                'no_trade_reasons': result.no_trade_reasons,
                'decision_trace': result.decision_trace,
                'score_components': result.score_components,
                'effective_size': result.effective_size,
                'decision_version': result.decision_version,
                'model_versions': result.model_versions,
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
