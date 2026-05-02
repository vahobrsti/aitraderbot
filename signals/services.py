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
from signals.overlays import apply_overlays, get_size_multiplier, get_dte_multiplier, compute_efb_veto
from signals.tactical_puts import tactical_put_inside_bull
from signals.options import get_strategy_with_path_risk, get_decision_strategy_summary, DECISION_STRATEGY_MAP, format_stop_loss_string, compute_condor_strikes
from signals.mvrv_short import check_mvrv_short_signal
from signals.condor_gate import evaluate_condor_gate, CondorGateResult, CONDOR_COOLDOWN_DAYS, compute_vol_metrics

# Cooldown constants — single source of truth for live + backtest alignment
# These values must match analyze_path_stats.py for calibration consistency
OPTION_SIGNAL_COOLDOWN_DAYS = 5   # was 7 — OPTION_CALL hits 81%, let more through
OPTION_SIGNAL_SIZE_MULT = 0.75   # base size for option signals (before overlay scaling)
MVRV_SHORT_COOLDOWN_DAYS = 5     # cooldown for MVRV short signal
MVRV_SHORT_SIZE_MULT = 0.33      # initial entry size (33% of position, DCA adds 67%)
CORE_SIGNAL_COOLDOWN_DAYS = 7
PROBE_COOLDOWN_DAYS = 5
TACTICAL_PUT_COOLDOWN_DAYS = 7

# Per-state path-risk rates from analyze_path_stats (14d horizon, 5% target, 4% invalidation, 569 trades)
# Recalibrated 2026-03-05 after fusion+bear-market revamp.
# Format: MarketState -> (invalid_before_hit_rate, same_day_ambiguous_rate)
PATH_RISK_BY_STATE = {
    MarketState.STRONG_BULLISH:          (0.286, 0.000),  # n=11, was 0.0% (n=1)
    MarketState.EARLY_RECOVERY:          (0.067, 0.000),  # n=17, very clean
    MarketState.MOMENTUM_CONTINUATION:   (0.263, 0.000),  # n=191, dropped below 30% threshold
    MarketState.BULL_PROBE:              (0.190, 0.000),  # n=83, moderate
    MarketState.DISTRIBUTION_RISK:       (0.174, 0.000),  # n=35, was 50% (n=5)
    MarketState.BEAR_CONTINUATION:       (0.167, 0.000),  # n=7, clean
    MarketState.BEAR_PROBE:              (0.550, 0.000),  # n=36, messiest — triggers path-risk
    MarketState.BEAR_EXHAUSTION_LONG:    (0.500, 0.000),  # n=5, unchanged
    MarketState.BEAR_RALLY_LONG:         (0.212, 0.000),  # n=66, improved
    MarketState.BEAR_CONTINUATION_SHORT: (0.250, 0.000),  # n=4, improved
    MarketState.LATE_DISTRIBUTION_SHORT: (0.286, 0.000),  # n=59, was 53% (n=17)
}


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
    stop_loss: str
    # Numeric execution fields for exchange integration
    stop_loss_pct: Optional[float]
    scale_down_day: Optional[int]
    max_hold_days: Optional[int]
    spread_width_pct: Optional[float]
    take_profit_pct: Optional[float]
    # NO_TRADE diagnostics
    no_trade_reasons: list
    decision_trace: list
    score_components: dict
    effective_size: float
    # Versioning
    decision_version: str
    model_versions: dict
    # Short signal source tracking
    short_source: Optional[str] = None  # Tracks specific rule origin for short setups
    # Iron condor gate
    condor_score: float = 0.0
    condor_eligible: bool = False
    condor_veto_reasons: list = None
    condor_score_components: dict = None
    # Iron condor MVRV-based strikes
    condor_short_call: Optional[float] = None
    condor_short_put: Optional[float] = None
    condor_cost_basis: Optional[float] = None
    condor_strike_meta: dict = None


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
        
        # 6) Core and tactical cooldown checks
        core_call_ok = self._check_trade_cooldown(latest_date, "CALL", CORE_SIGNAL_COOLDOWN_DAYS)
        core_put_ok = self._check_trade_cooldown(latest_date, "PUT", CORE_SIGNAL_COOLDOWN_DAYS)
        tactical_put_ok = self._check_trade_cooldown(latest_date, "TACTICAL_PUT", TACTICAL_PUT_COOLDOWN_DAYS)

        # 7) Tactical Put
        tactical_result = tactical_put_inside_bull(
            fusion_result,
            row,
            cooldown_active=not tactical_put_ok,
        )

        # 8) Option signal cooldown check
        option_call_ok = signal_option_call == 1 and self._check_option_cooldown(latest_date, "OPTION_CALL")
        option_put_ok = signal_option_put == 1 and self._check_option_cooldown(latest_date, "OPTION_PUT")
        
        # 9) MVRV Short signal check (bear market tactical short)
        # Add raw MVRV data to row for signal check
        try:
            raw_data = RawDailyData.objects.get(date=latest_date)
            row_with_mvrv = row.copy()
            row_with_mvrv['mvrv_usd_7d'] = raw_data.mvrv_usd_7d
            row_with_mvrv['mvrv_usd_60d'] = raw_data.mvrv_usd_60d
            row_with_mvrv['btc_close'] = raw_data.btc_close
        except RawDailyData.DoesNotExist:
            row_with_mvrv = row
        
        mvrv_short_signal = check_mvrv_short_signal(row_with_mvrv)
        mvrv_short_ok = mvrv_short_signal.active and self._check_trade_cooldown(
            latest_date, "MVRV_SHORT", MVRV_SHORT_COOLDOWN_DAYS
        )
        
        # 10) Iron Condor gate evaluation
        # Compute vol metrics from raw OHLC (no lookahead)
        ohlc_qs = RawDailyData.objects.filter(date__lte=latest_date).order_by("date").values(
            "btc_open", "btc_high", "btc_low", "btc_close"
        )
        ohlc_df = pd.DataFrame.from_records(ohlc_qs)
        if len(ohlc_df) >= 31:
            atr_ratio, gap_pct = compute_vol_metrics(ohlc_df)
        else:
            atr_ratio, gap_pct = None, None

        # Expanding median of distribution_pressure_score (no lookahead)
        dp_val = row.get("distribution_pressure_score", None)
        if dp_val is not None:
            dp_series = feats.loc[:latest_date, "distribution_pressure_score"]
            dp_expanding_median = float(dp_series.expanding().median().iloc[-1])
        else:
            dp_expanding_median = None

        condor_gate = evaluate_condor_gate(
            row,
            fusion_state=fusion_result.state.value,
            dp_expanding_median=dp_expanding_median,
            atr_ratio=atr_ratio,
            gap_pct=gap_pct,
        )
        condor_cooldown_ok = self._check_trade_cooldown(
            latest_date, "IRON_CONDOR", CONDOR_COOLDOWN_DAYS
        )
        condor_ok = condor_gate.eligible and condor_cooldown_ok

        # 11) Compute MVRV-based condor strikes (always, for diagnostics)
        condor_strikes = None
        if row_with_mvrv is not None:
            mvrv_60d_val = getattr(raw_data, 'mvrv_usd_60d', None) if raw_data else None
            btc_close_val = getattr(raw_data, 'btc_close', None) if raw_data else None
            if mvrv_60d_val and btc_close_val:
                # Get trailing 7d MVRV range for drift calculation
                from signals.options import CONDOR_TRAILING_DAYS
                trailing_qs = RawDailyData.objects.filter(
                    date__lte=latest_date,
                    mvrv_usd_60d__isnull=False,
                ).order_by('-date')[:CONDOR_TRAILING_DAYS]
                trailing_mvrv = [r.mvrv_usd_60d for r in trailing_qs]
                if len(trailing_mvrv) >= CONDOR_TRAILING_DAYS:
                    mvrv_trail_high = max(trailing_mvrv)
                    mvrv_trail_low = min(trailing_mvrv)
                else:
                    mvrv_trail_high = None
                    mvrv_trail_low = None

                condor_strikes = compute_condor_strikes(
                    spot=btc_close_val,
                    mvrv_60d=mvrv_60d_val,
                    mvrv_trailing_high=mvrv_trail_high,
                    mvrv_trailing_low=mvrv_trail_low,
                )
        
        trade_decision, trade_notes, no_trade_reasons, decision_trace = self._determine_trade_decision(
            fusion_result, size_mult, dte_mult, tactical_result, p_long, p_short,
            signal_option_call, signal_option_put, overlay, row,
            core_call_ok=core_call_ok, core_put_ok=core_put_ok,
            option_call_ok=option_call_ok, option_put_ok=option_put_ok,
            mvrv_short_ok=mvrv_short_ok, mvrv_short_signal=mvrv_short_signal,
            condor_ok=condor_ok, condor_gate=condor_gate,
        )
        
        # 10) Get strategy recommendation based on final trade decision
        if trade_decision in DECISION_STRATEGY_MAP:
            strategy_summary = get_decision_strategy_summary(trade_decision)
        else:
            strategy_summary = self._get_strategy_summary_with_path_risk(fusion_result.state)
        
        # Compute effective values
        if trade_decision == "NO_TRADE":
            effective_size = 0.0
            size_mult = 0.0
            dte_mult = 0.0
        elif trade_decision in ("OPTION_CALL", "OPTION_PUT"):
            # Option signals use base size scaled by overlay (soft scaling)
            effective_size = min(OPTION_SIGNAL_SIZE_MULT, OPTION_SIGNAL_SIZE_MULT * size_mult)
            size_mult = effective_size
        elif trade_decision == "MVRV_SHORT":
            # MVRV short uses fixed initial size (DCA handled separately)
            effective_size = MVRV_SHORT_SIZE_MULT
            size_mult = effective_size
        elif trade_decision == "IRON_CONDOR":
            # Iron condor uses fixed size (premium selling)
            effective_size = 0.50  # 50% of base — conservative for premium selling
            size_mult = effective_size
        else:
            effective_size = size_mult
        
        # Build model version info
        model_versions = {
            'long': self.long_model_path.name,
            'short': self.short_model_path.name,
        }
        decision_version = "2026-04-26.1"
        
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
            stop_loss=strategy_summary.get("stop_loss", ""),
            stop_loss_pct=strategy_summary.get("stop_loss_pct"),
            scale_down_day=strategy_summary.get("scale_down_day"),
            max_hold_days=strategy_summary.get("max_hold_days"),
            spread_width_pct=strategy_summary.get("spread_width_pct"),
            take_profit_pct=strategy_summary.get("take_profit_pct"),
            no_trade_reasons=no_trade_reasons,
            decision_trace=decision_trace,
            score_components=fusion_result.components,
            effective_size=effective_size,
            decision_version=decision_version,
            model_versions=model_versions,
            short_source=fusion_result.short_source,
            condor_score=condor_gate.score,
            condor_eligible=condor_gate.eligible,
            condor_veto_reasons=condor_gate.veto_reasons,
            condor_score_components=condor_gate.score_components,
            condor_short_call=condor_strikes.short_call if condor_strikes else None,
            condor_short_put=condor_strikes.short_put if condor_strikes else None,
            condor_cost_basis=condor_strikes.cost_basis if condor_strikes else None,
            condor_strike_meta={
                'call_source': condor_strikes.call_source,
                'put_source': condor_strikes.put_source,
                'call_distance_pct': condor_strikes.call_distance_pct,
                'put_distance_pct': condor_strikes.put_distance_pct,
                'mvrv_60d': condor_strikes.mvrv_60d,
                'mvrv_drift': condor_strikes.mvrv_drift,
                'mvrv_ceiling': condor_strikes.mvrv_ceiling,
                'mvrv_floor': condor_strikes.mvrv_floor,
                'spot': condor_strikes.spot,
            } if condor_strikes else {},
        )

    def _get_strategy_summary_with_path_risk(self, state: MarketState) -> dict:
        if state == MarketState.NO_TRADE:
            return {
                "primary_structures": "",
                "strike_guidance": "",
                "dte_range": "",
                "rationale": "",
                "stop_loss": "",
                "stop_loss_pct": None,
                "scale_down_day": None,
                "max_hold_days": None,
                "spread_width_pct": None,
                "take_profit_pct": None,
            }

        rates = PATH_RISK_BY_STATE.get(state)
        if rates is not None:
            inv, amb = rates
        else:
            inv, amb = None, None

        strategy = get_strategy_with_path_risk(
            state=state,
            invalid_before_hit_rate=inv,
            same_day_ambiguous_rate=amb,
        )

        structures = ", ".join(s.value for s in strategy.primary_structures)
        dte_range = f"{strategy.dte.min_dte}-{strategy.dte.max_dte}d"
        rationale = strategy.rationale
        
        # Extract numeric fields from spread guidance
        stop_loss_pct = None
        scale_down_day = None
        max_hold_days = None
        spread_width_pct = None
        take_profit_pct = None
        
        if strategy.spread is not None:
            stop_loss_pct = strategy.spread.stop_loss_pct
            scale_down_day = strategy.spread.scale_down_day
            max_hold_days = strategy.spread.max_hold_days
            spread_width_pct = strategy.spread.width_pct
            take_profit_pct = strategy.spread.take_profit_pct
            rationale = (
                f"{rationale} "
                f"[spread width={strategy.spread.width_pct*100:.0f}%, "
                f"take-profit={strategy.spread.take_profit_pct*100:.0f}%, "
                f"max-hold={strategy.spread.max_hold_days}d]"
            )

        stop_loss = format_stop_loss_string(strategy.spread)

        return {
            "primary_structures": structures,
            "strike_guidance": strategy.strike_guidance.value,
            "dte_range": dte_range,
            "rationale": rationale,
            "stop_loss": stop_loss,
            "stop_loss_pct": stop_loss_pct,
            "scale_down_day": scale_down_day,
            "max_hold_days": max_hold_days,
            "spread_width_pct": spread_width_pct,
            "take_profit_pct": take_profit_pct,
        }
    
    def _check_trade_cooldown(self, current_date: date, decision_type: str, cooldown_days: int) -> bool:
        """Return True when decision_type can fire on current_date."""
        cutoff = current_date - timedelta(days=cooldown_days)
        last_signal = (
            DailySignal.objects
            .filter(trade_decision=decision_type, date__gte=cutoff, date__lt=current_date)
            .order_by("-date")
            .first()
        )
        return last_signal is None

    def _check_option_cooldown(self, current_date, decision_type: str) -> bool:
        """
        Check if enough days have passed since last OPTION_CALL/OPTION_PUT trade.
        Returns True if cooldown has passed (OK to trade).
        """
        return self._check_trade_cooldown(current_date, decision_type, OPTION_SIGNAL_COOLDOWN_DAYS)

    def _determine_trade_decision(
        self, fusion_result, size_mult, dte_mult, tactical_result,
        p_long, p_short, signal_option_call, signal_option_put, overlay, row,
        core_call_ok=True, core_put_ok=True,
        option_call_ok=False, option_put_ok=False,
        mvrv_short_ok=False, mvrv_short_signal=None,
        condor_ok=False, condor_gate=None,
    ) -> tuple[str, str, list, list]:
        """
        Determine final trade decision, notes, and diagnostics.
        
        Priority:
            1. Fusion state (CALL/PUT/probes)
            2. Tactical Put (when fusion=NO_TRADE or CALL is cooldown-blocked)
            3. Option signal fallback (when fusion=NO_TRADE, with cooldown + overlay)
            4. MVRV Short (bear market tactical)
            5. Iron Condor (range gate passes in chop)
            6. NO_TRADE
        
        Returns:
            Tuple of (decision, notes, no_trade_reasons, decision_trace)
        """
        no_trade_reasons = []
        decision_trace = []
        
        # Build decision trace step by step
        fusion_step = f"fusion={fusion_result.state.value}(score={fusion_result.score},conf={fusion_result.confidence.value})"
        decision_trace.append(fusion_step)
        
        # Check fusion state first — if fusion has a directional view, use it
        # TRANSITION_CHOP is treated like NO_TRADE for fallback purposes
        fusion_has_direction = fusion_result.state not in (MarketState.NO_TRADE, MarketState.TRANSITION_CHOP)
        if fusion_has_direction:
            # Log tactical put status for tracing
            if tactical_result.active:
                decision_trace.append(f"tactical_put=active(strategy={tactical_result.strategy.value})")
            else:
                decision_trace.append("tactical_put=inactive")
            
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
                MarketState.MOMENTUM_CONTINUATION,
                MarketState.BEAR_EXHAUSTION_LONG,
                MarketState.BEAR_RALLY_LONG
            }
            is_short = fusion_result.state in {
                MarketState.DISTRIBUTION_RISK,
                MarketState.BEAR_CONTINUATION,
                MarketState.BEAR_CONTINUATION_SHORT,
                MarketState.LATE_DISTRIBUTION_SHORT
            }

            core_decision = None
            core_notes = ""
            core_cooldown_ok = False

            if is_long:
                core_decision = "CALL"
                core_notes = f"Fusion: {fusion_result.state.value}"
                core_cooldown_ok = core_call_ok
            elif is_short:
                source_label = f"[{fusion_result.short_source}]" if fusion_result.short_source else ""
                core_decision = "PUT"
                core_notes = f"Fusion: {fusion_result.state.value} {source_label}".strip()
                core_cooldown_ok = core_put_ok
            elif fusion_result.state == MarketState.BULL_PROBE:
                core_decision = "CALL"
                core_notes = f"Bull probe, score={fusion_result.score}"
                core_cooldown_ok = core_call_ok
            elif fusion_result.state == MarketState.BEAR_PROBE:
                source_label = f"[{fusion_result.short_source}]" if fusion_result.short_source else ""
                core_decision = "PUT"
                core_notes = f"Bear probe, score={fusion_result.score} {source_label}".strip()
                core_cooldown_ok = core_put_ok

            if core_decision is not None:
                if core_cooldown_ok:
                    if tactical_result.active:
                        decision_trace.append("fusion takes priority over tactical_put")
                    decision_trace.append(f"state={fusion_result.state.value} -> {core_decision}")
                    return core_decision, core_notes, [], decision_trace

                cooldown_reason = f"{core_decision}_COOLDOWN_ACTIVE"
                no_trade_reasons.append(cooldown_reason)
                decision_trace.append(f"state={fusion_result.state.value} -> {core_decision} blocked by cooldown")

                if core_decision == "CALL" and tactical_result.active:
                    decision_trace.append(f"fallback=tactical_put(strategy={tactical_result.strategy.value}) -> TRADE")
                    return "TACTICAL_PUT", tactical_result.reason, [], decision_trace

                decision_trace.append(f"stop_reason={cooldown_reason}")
                return "NO_TRADE", f"{core_decision} cooldown active", no_trade_reasons, decision_trace
            
            # Catch-all for non-tradeable fusion states
            no_trade_reasons.append("STATE_NOT_TRADEABLE")
            decision_trace.append("stop_reason=STATE_NOT_TRADEABLE")
            return "NO_TRADE", f"State: {fusion_result.state.value}", no_trade_reasons, decision_trace
        
        # --- Fusion is NO_TRADE or TRANSITION_CHOP — check fallbacks ---
        decision_trace.append(f"fusion={fusion_result.state.value} (no direction)")
        
        if tactical_result.active:
            decision_trace.append(f"tactical_put=active(strategy={tactical_result.strategy.value}) -> TRADE")
            return "TACTICAL_PUT", tactical_result.reason, [], decision_trace
        else:
            decision_trace.append("tactical_put=inactive")
        
        # Option CALL fallback
        if option_call_ok:
            # Apply overlay check
            if size_mult == 0:
                decision_trace.append("option_call=fired but overlay_veto -> NO_TRADE")
                no_trade_reasons.append("FUSION_STATE_NO_TRADE")
                no_trade_reasons.append("OPTION_CALL_OVERLAY_VETO")
                return "NO_TRADE", "Option call fired but overlay vetoed", no_trade_reasons, decision_trace
            # Apply soft overlay scaling to option size
            scaled_size = min(OPTION_SIGNAL_SIZE_MULT, OPTION_SIGNAL_SIZE_MULT * size_mult)
            decision_trace.append(f"option_call=fired(cooldown_ok) -> OPTION_CALL (base={OPTION_SIGNAL_SIZE_MULT}, overlay={size_mult:.2f}, effective={scaled_size:.2f})")
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
            # Check EFB veto for option puts
            efb_veto, efb_reason = compute_efb_veto(row)
            if efb_veto >= 1:
                decision_trace.append(f"option_put=fired but {efb_reason} -> NO_TRADE")
                no_trade_reasons.append("EFB_VETO_OPTION_PUT")
                return "NO_TRADE", f"Option put fired but EFB vetoed ({efb_reason})", no_trade_reasons, decision_trace
            # Apply soft overlay scaling to option size
            scaled_size = min(OPTION_SIGNAL_SIZE_MULT, OPTION_SIGNAL_SIZE_MULT * size_mult)
            decision_trace.append(f"option_put=fired(cooldown_ok) -> OPTION_PUT (base={OPTION_SIGNAL_SIZE_MULT}, overlay={size_mult:.2f}, effective={scaled_size:.2f})")
            return "OPTION_PUT", "Rule: MVRV overheated + Sentiment greed + Whale distrib", [], decision_trace
        elif signal_option_put == 1:
            decision_trace.append("option_put=fired but cooldown_active -> skip")
        
        # MVRV Short fallback (bear market tactical short)
        if mvrv_short_ok and mvrv_short_signal is not None and mvrv_short_signal.active:
            decision_trace.append(
                f"mvrv_short=active(7d={mvrv_short_signal.mvrv_7d:.3f}, 60d={mvrv_short_signal.mvrv_60d:.3f}) "
                f"-> MVRV_SHORT (size={MVRV_SHORT_SIZE_MULT}, target={mvrv_short_signal.target_pct}%)"
            )
            return (
                "MVRV_SHORT",
                f"Rule: Bear mode + MVRV 7d >= 1.02 + MVRV 60d >= 1.0 | "
                f"Entry: {MVRV_SHORT_SIZE_MULT*100:.0f}% at ${mvrv_short_signal.btc_price:,.0f}, "
                f"DCA at ${mvrv_short_signal.dca_trigger_price:,.0f}, "
                f"Target: ${mvrv_short_signal.target_price:,.0f}",
                [],
                decision_trace
            )
        elif mvrv_short_signal is not None and mvrv_short_signal.active:
            decision_trace.append(f"mvrv_short=active but cooldown_active -> skip")
        elif mvrv_short_signal is not None and not mvrv_short_signal.active:
            decision_trace.append(f"mvrv_short=inactive({mvrv_short_signal.reason})")
        
        # Iron Condor fallback (range-bound strategy when everything is flat)
        if condor_ok and condor_gate is not None:
            decision_trace.append(
                f"condor_gate=eligible(score={condor_gate.score:.0f}, thresh={condor_gate.threshold:.0f}) "
                f"-> IRON_CONDOR"
            )
            notes = f"Range gate: score={condor_gate.score:.0f}, chop state"
            return "IRON_CONDOR", notes, [], decision_trace
        elif condor_gate is not None:
            reasons = []
            if condor_gate.score < condor_gate.threshold:
                reasons.append(f"score={condor_gate.score:.0f}<{condor_gate.threshold:.0f}")
            if condor_gate.veto_reasons:
                reasons.append(f"vetoes={','.join(condor_gate.veto_reasons)}")
            if condor_gate.eligible and not condor_ok:
                reasons.append("cooldown")
            decision_trace.append(f"condor_gate=blocked({'; '.join(reasons)})")
        
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
                'short_source': result.short_source or "",
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
                'stop_loss': result.stop_loss,
                'stop_loss_pct': result.stop_loss_pct,
                'scale_down_day': result.scale_down_day,
                'max_hold_days': result.max_hold_days,
                'spread_width_pct': result.spread_width_pct,
                'take_profit_pct': result.take_profit_pct,
                'no_trade_reasons': result.no_trade_reasons,
                'decision_trace': result.decision_trace,
                'score_components': result.score_components,
                'effective_size': result.effective_size,
                'decision_version': result.decision_version,
                'model_versions': result.model_versions,
                'condor_score': result.condor_score,
                'condor_eligible': result.condor_eligible,
                'condor_veto_reasons': result.condor_veto_reasons or [],
                'condor_score_components': result.condor_score_components or {},
                'condor_short_call': result.condor_short_call,
                'condor_short_put': result.condor_short_put,
                'condor_cost_basis': result.condor_cost_basis,
                'condor_strike_meta': result.condor_strike_meta or {},
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
