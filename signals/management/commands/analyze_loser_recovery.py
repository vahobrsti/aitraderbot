"""
Analyze loser recovery potential for failed trades.

This command identifies trades that are likely failing at checkpoint (TTH p75)
and measures the forward return if you flipped direction at that point.

The thesis: If a trade hasn't hit by TTH p75 AND price is adverse, 
the probability of recovery is low. Flipping direction may recover losses.

POLICY INTEGRATION:
- Checkpoint days derived from policy.get_path_profile() TTH p75 values
- Adverse thresholds calculated as policy mae_p75 / 2
- Falls back to hardcoded values if policy data unavailable

Usage:
    python manage.py analyze_loser_recovery
    python manage.py analyze_loser_recovery --type MVRV_SHORT
    python manage.py analyze_loser_recovery --type BEAR_PROBE --adverse-threshold 0.02
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand

from datafeed.models import RawDailyData
from signals.fusion import MarketState, add_fusion_features, fuse_signals
from signals.overlays import apply_overlays, compute_efb_veto, get_size_multiplier
from signals.tactical_puts import tactical_put_inside_bull
from signals.condor_gate import compute_range_score, check_hard_vetoes, DEFAULT_SCORE_THRESHOLD, CONDOR_COOLDOWN_DAYS
from execution.services.policy import get_policy
from execution.services.recovery import RecoveryDecisionEngine


# TTH p75 values from entry_policy_design.md (calibrated from path analysis)
# FALLBACK VALUES - Used when policy data is unavailable
TTH_P75_BY_TYPE_FALLBACK = {
    "LONG": 7,
    "PUT": 6,
    "PRIMARY_SHORT": 6,
    "BULL_PROBE": 5,
    "BEAR_PROBE": 9,  # 8.5 rounded up
    "TACTICAL_PUT": 6,  # 5.5 rounded up
    "OPTION_CALL": 5,
    "OPTION_PUT": 3,
    "MVRV_SHORT": 10,
    "IRON_CONDOR": 7,  # Fixed 7-day horizon
}

# MAE(W) p75 values - FALLBACK VALUES - Used when policy data is unavailable
MAE_W_P75_BY_TYPE_FALLBACK = {
    "LONG": 0.0471,
    "PUT": 0.0441,
    "PRIMARY_SHORT": 0.0441,
    "BULL_PROBE": 0.0384,
    "BEAR_PROBE": 0.0653,
    "TACTICAL_PUT": 0.0308,
    "OPTION_CALL": 0.0848,
    "OPTION_PUT": 0.0682,
    "MVRV_SHORT": 0.0719,
    "IRON_CONDOR": 0.0676,
}


@dataclass
class PolicyConfigAdapter:
    """
    Adapts PolicyVersion to provide recovery analysis parameters.
    
    Delegates to RecoveryDecisionEngine for checkpoint days and adverse thresholds,
    with fallback to hardcoded values if the service is unavailable.
    Uses a cached engine instance keyed by policy version to avoid repeated loading
    while invalidating on policy changes.
    """
    
    _engine_cache: Optional[RecoveryDecisionEngine] = None
    _cached_version: Optional[str] = None
    
    @classmethod
    def _get_engine(cls) -> RecoveryDecisionEngine:
        """Get or create a cached RecoveryDecisionEngine instance. Invalidates on policy version change."""
        policy = get_policy()
        if cls._engine_cache is None or cls._cached_version != policy.version:
            cls._engine_cache = RecoveryDecisionEngine(policy)
            cls._cached_version = policy.version
        return cls._engine_cache
    
    @classmethod
    def reset_cache(cls):
        """Reset the cached engine (useful for testing)."""
        cls._engine_cache = None
        cls._cached_version = None
    
    @classmethod
    def get_checkpoint_day(cls, signal_type: str) -> int:
        """
        Get checkpoint day for a signal type based on TTH p75 from policy.
        
        Delegates to RecoveryDecisionEngine.get_checkpoint_day().
        Falls back to hardcoded values if unavailable.
        """
        try:
            return cls._get_engine().get_checkpoint_day(signal_type)
        except Exception:
            return TTH_P75_BY_TYPE_FALLBACK.get(signal_type, 7)
    
    @classmethod
    def get_adverse_threshold(cls, signal_type: str, override: Optional[float] = None) -> float:
        """
        Get adverse threshold for a signal type.
        
        Delegates to RecoveryDecisionEngine.get_adverse_threshold().
        Falls back to hardcoded values if unavailable.
        
        Args:
            signal_type: Signal type
            override: User-provided override value (takes precedence)
            
        Returns:
            Adverse threshold as decimal (e.g., 0.0235 for 2.35%)
        """
        if override is not None:
            return override
            
        try:
            return cls._get_engine().get_adverse_threshold(signal_type)
        except Exception:
            return MAE_W_P75_BY_TYPE_FALLBACK.get(signal_type, 0.04) / 2


@dataclass
class RecoveryCandidate:
    """A trade that qualifies for recovery analysis."""
    date: str
    trade_type: str
    direction: str
    entry_price: float
    checkpoint_day: int
    checkpoint_price: float
    checkpoint_return: float  # Return at checkpoint (negative = adverse for original)
    remaining_days: int
    forward_return: float  # Return from checkpoint to horizon end
    original_hit: bool
    recovery_hit: bool  # Would flipping have hit target?
    recovery_return: float  # Return if you flipped at checkpoint
    flip_threshold: float = 0.05  # Threshold for FLIP recommendation
    cut_threshold: float = 0.02   # Threshold for CUT recommendation
    
    @property
    def recommendation(self) -> str:
        """
        Get trading recommendation based on recovery potential.
        
        Logic:
        - recovery_return > flip_threshold → "FLIP" (strong recovery potential)
        - recovery_return < cut_threshold → "CUT" (weak recovery potential, cut losses)
        - else → "HOLD" (moderate recovery potential, hold original position)
        
        Returns:
            "FLIP", "CUT", or "HOLD"
        """
        if self.recovery_return > self.flip_threshold:
            return "FLIP"
        elif self.recovery_return < self.cut_threshold:
            return "CUT"
        else:
            return "HOLD"


class Command(BaseCommand):
    help = "Analyze loser recovery potential at TTH p75 checkpoint"

    def add_arguments(self, parser):
        parser.add_argument("--csv", type=str, default="features_14d_5pct.csv")
        parser.add_argument("--year", type=int, default=None)
        parser.add_argument("--no-overlay", action="store_true")
        parser.add_argument("--no-cooldown", action="store_true")
        parser.add_argument("--long-model", type=str, default="models/long_model.joblib")
        parser.add_argument("--short-model", type=str, default="models/short_model.joblib")
        parser.add_argument("--type", type=str, default=None, help="Filter by trade type")
        parser.add_argument("--horizon", type=int, default=14, help="Forward path horizon in days")
        parser.add_argument("--target", type=float, default=0.05, help="Target move threshold")
        parser.add_argument(
            "--adverse-threshold",
            type=float,
            default=None,
            help="Min adverse move at checkpoint to trigger recovery (default: use policy mae_p75 / 2, fallback to hardcoded values)",
        )
        parser.add_argument(
            "--recovery-target",
            type=float,
            default=0.03,
            help="Target for recovery trade (default: 3%% - smaller than original)",
        )
        parser.add_argument(
            "--recovery-flip-threshold",
            type=float,
            default=0.05,
            help="Minimum recovery MFE to recommend FLIP (default: 5%%)",
        )
        parser.add_argument(
            "--recovery-cut-threshold", 
            type=float,
            default=0.02,
            help="Maximum recovery MFE to recommend CUT (default: 2%%)",
        )
        parser.add_argument(
            "--show-threshold-impact",
            action="store_true",
            help="Show impact of different threshold values on recommendation distribution",
        )
        parser.add_argument(
            "--checkpoint-mode",
            type=str,
            default="tth_p75",
            choices=["tth_p75", "tth_median", "fixed"],
            help="How to determine checkpoint day",
        )
        parser.add_argument(
            "--fixed-checkpoint",
            type=int,
            default=7,
            help="Fixed checkpoint day (only used if checkpoint-mode=fixed)",
        )
        parser.add_argument(
            "--simulate-policy",
            action="store_true",
            help="Run policy backtesting simulation to calculate P&L impact of FLIP/HOLD/CUT decisions vs hold-to-expiry",
        )
        parser.add_argument(
            "--sensitivity-analysis",
            action="store_true",
            help="Run sensitivity analysis testing different recovery thresholds (1%%, 2%%, 3%%, 5%%, 10%%) to find optimal threshold selection",
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
        target = float(options.get("target", 0.05))
        recovery_target = float(options.get("recovery_target", 0.03))
        flip_threshold = float(options.get("recovery_flip_threshold", 0.05))
        cut_threshold = float(options.get("recovery_cut_threshold", 0.02))
        show_threshold_impact = options.get("show_threshold_impact", False)
        adverse_threshold = options.get("adverse_threshold")
        type_filter = options.get("type")
        checkpoint_mode = options.get("checkpoint_mode", "tth_p75")
        fixed_checkpoint = int(options.get("fixed_checkpoint", 7))
        simulate_policy = options.get("simulate_policy", False)
        sensitivity_analysis = options.get("sensitivity_analysis", False)

        # Validate inputs
        if horizon < 3:
            self.stderr.write("--horizon must be >= 3")
            return
        if checkpoint_mode == "fixed":
            if fixed_checkpoint < 1:
                self.stderr.write("--fixed-checkpoint must be >= 1")
                return
            if fixed_checkpoint >= horizon:
                self.stderr.write(f"--fixed-checkpoint ({fixed_checkpoint}) must be less than --horizon ({horizon})")
                return

        # Build trades using same logic as analyze_path_stats
        trades_df = self._build_trades_df(csv_path, options, year_filter, no_overlay, no_cooldown)
        if trades_df.empty:
            self.stdout.write("No trades found.")
            return

        if type_filter:
            trades_df = trades_df[trades_df["type"] == type_filter]
        if trades_df.empty:
            self.stdout.write("No trades match filters.")
            return

        # Exclude IRON_CONDOR (neutral, no directional recovery)
        trades_df = trades_df[trades_df["type"] != "IRON_CONDOR"]
        if trades_df.empty:
            self.stdout.write("No directional trades to analyze.")
            return

        price_df = self._load_prices()
        if price_df.empty:
            self.stderr.write("No price data found.")
            return

        # Analyze recovery potential
        candidates = self._analyze_recovery(
            trades_df, price_df, horizon, target, recovery_target,
            adverse_threshold, checkpoint_mode, fixed_checkpoint, flip_threshold, cut_threshold
        )

        # Run sensitivity analysis if requested
        sensitivity_results = None
        if sensitivity_analysis:
            sensitivity_results = self._run_sensitivity_analysis(
                trades_df, price_df, horizon, target, adverse_threshold,
                checkpoint_mode, fixed_checkpoint, flip_threshold, cut_threshold
            )

        self._print_results(candidates, horizon, target, recovery_target, type_filter, flip_threshold, cut_threshold, show_threshold_impact, simulate_policy, sensitivity_analysis)
        
        # Print sensitivity analysis results if available
        if sensitivity_results:
            self._print_sensitivity_analysis_results(sensitivity_results)

    def _build_trades_df(
        self,
        csv_path: Path,
        options: dict,
        year_filter: Optional[int],
        no_overlay: bool,
        no_cooldown: bool,
    ) -> pd.DataFrame:
        """Build trades DataFrame - mirrors analyze_path_stats logic."""
        long_bundle = joblib.load(Path(options["long_model"]))
        short_bundle = joblib.load(Path(options["short_model"]))
        long_model = long_bundle["model"]
        short_model = short_bundle["model"]
        long_feats = long_bundle["feature_names"]
        short_feats = short_bundle["feature_names"]

        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        df = add_fusion_features(df)

        prob_df = pd.DataFrame({
            "p_long": long_model.predict_proba(df[long_feats])[:, 1],
            "p_short": short_model.predict_proba(df[short_feats])[:, 1],
        }, index=df.index)
        df = pd.concat([df, prob_df], axis=1)

        long_states = {
            MarketState.STRONG_BULLISH,
            MarketState.EARLY_RECOVERY,
            MarketState.MOMENTUM_CONTINUATION,
            MarketState.BULL_PROBE,
            MarketState.BEAR_EXHAUSTION_LONG,
            MarketState.BEAR_RALLY_LONG,
        }
        short_states = {
            MarketState.DISTRIBUTION_RISK,
            MarketState.BEAR_CONTINUATION,
            MarketState.BEAR_PROBE,
            MarketState.BEAR_CONTINUATION_SHORT,
            MarketState.LATE_DISTRIBUTION_SHORT,
        }

        all_years = sorted(df.index.year.unique())
        if year_filter is not None:
            years = [year_filter] if year_filter in all_years else []
        else:
            years = all_years

        from signals.services import (
            CORE_SIGNAL_COOLDOWN_DAYS,
            PROBE_COOLDOWN_DAYS,
            TACTICAL_PUT_COOLDOWN_DAYS,
            OPTION_SIGNAL_COOLDOWN_DAYS,
            MVRV_SHORT_COOLDOWN_DAYS,
        )

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
            last_mvrv_short_date = None

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

                can_long_fire = True
                can_short_fire = True
                if not no_cooldown:
                    if is_long_state and last_long_date is not None and (date - last_long_date).days <= CORE_SIGNAL_COOLDOWN_DAYS:
                        can_long_fire = False
                    if is_short_state and last_short_date is not None and (date - last_short_date).days <= CORE_SIGNAL_COOLDOWN_DAYS:
                        can_short_fire = False
                    if is_bull_probe and last_long_date is not None and (date - last_long_date).days <= PROBE_COOLDOWN_DAYS:
                        can_long_fire = False
                    if is_bear_probe and last_short_date is not None and (date - last_short_date).days <= PROBE_COOLDOWN_DAYS:
                        can_short_fire = False

                if is_bull_probe or is_bear_probe:
                    size_mult = min(size_mult, 0.50)

                long_trade_fired = False
                if result.state in long_states and size_mult > 0 and not long_veto and can_long_fire:
                    all_trades.append({
                        "date": pd.Timestamp(date).normalize(),
                        "year": year,
                        "type": "BULL_PROBE" if is_bull_probe else "LONG",
                        "direction": "LONG",
                        "source": "rule",
                        "state": result.state.value,
                    })
                    last_long_date = date
                    long_trade_fired = True

                if result.state in short_states and size_mult > 0 and not short_veto and can_short_fire:
                    all_trades.append({
                        "date": pd.Timestamp(date).normalize(),
                        "year": year,
                        "type": "BEAR_PROBE" if is_bear_probe else "PRIMARY_SHORT",
                        "direction": "SHORT",
                        "source": result.short_source or "rule",
                        "state": result.state.value,
                    })
                    last_short_date = date

                tactical_cooldown_ok = (
                    no_cooldown or last_tactical_date is None or (date - last_tactical_date).days > TACTICAL_PUT_COOLDOWN_DAYS
                )
                if result.state in long_states and not long_trade_fired and tactical_cooldown_ok:
                    tactical = tactical_put_inside_bull(result, row)
                    if tactical.active:
                        all_trades.append({
                            "date": pd.Timestamp(date).normalize(),
                            "year": year,
                            "type": "TACTICAL_PUT",
                            "direction": "SHORT",
                            "source": "tactical",
                            "state": result.state.value,
                        })
                        last_tactical_date = date

                signal_call = int(row.get("signal_option_call", 0))
                signal_put = int(row.get("signal_option_put", 0))

                fusion_traded = (result.state in long_states and size_mult > 0 and not long_veto and can_long_fire) or \
                               (result.state in short_states and size_mult > 0 and not short_veto and can_short_fire)

                option_call_fired = False
                if signal_call == 1 and not fusion_traded:
                    cooldown_ok = no_cooldown or last_option_call_date is None or (date - last_option_call_date).days > OPTION_SIGNAL_COOLDOWN_DAYS
                    overlay_ok = no_overlay or size_mult > 0
                    if cooldown_ok and overlay_ok:
                        all_trades.append({
                            "date": pd.Timestamp(date).normalize(),
                            "year": year,
                            "type": "OPTION_CALL",
                            "direction": "LONG",
                            "source": "option_rule",
                            "state": result.state.value,
                        })
                        last_option_call_date = date
                        option_call_fired = True

                option_put_fired = False
                if signal_put == 1 and not fusion_traded and not option_call_fired:
                    cooldown_ok = no_cooldown or last_option_put_date is None or (date - last_option_put_date).days > OPTION_SIGNAL_COOLDOWN_DAYS
                    overlay_ok = no_overlay or size_mult > 0
                    if cooldown_ok and overlay_ok:
                        efb_veto, _ = compute_efb_veto(row)
                        if efb_veto < 1:
                            all_trades.append({
                                "date": pd.Timestamp(date).normalize(),
                                "year": year,
                                "type": "OPTION_PUT",
                                "direction": "SHORT",
                                "source": "option_rule",
                                "state": result.state.value,
                            })
                            last_option_put_date = date
                            option_put_fired = True

                fusion_no_signal = result.state in (MarketState.NO_TRADE, MarketState.TRANSITION_CHOP)
                if fusion_no_signal and not option_call_fired and not option_put_fired:
                    from signals.mvrv_short import check_mvrv_short_signal
                    mvrv_signal = check_mvrv_short_signal(row)
                    if mvrv_signal.active:
                        mvrv_cooldown_ok = no_cooldown or last_mvrv_short_date is None or (date - last_mvrv_short_date).days > MVRV_SHORT_COOLDOWN_DAYS
                        if mvrv_cooldown_ok:
                            all_trades.append({
                                "date": pd.Timestamp(date).normalize(),
                                "year": year,
                                "type": "MVRV_SHORT",
                                "direction": "SHORT",
                                "source": "mvrv_rule",
                                "state": result.state.value,
                            })
                            last_mvrv_short_date = date

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

    def _get_checkpoint_day(self, trade_type: str, mode: str, fixed: int) -> int:
        """Get checkpoint day based on mode."""
        if mode == "fixed":
            return fixed
        elif mode == "tth_median":
            # Approximate median as p75 * 0.6
            checkpoint_day = PolicyConfigAdapter.get_checkpoint_day(trade_type)
            return max(2, int(checkpoint_day * 0.6))
        else:  # tth_p75
            return PolicyConfigAdapter.get_checkpoint_day(trade_type)

    def _analyze_recovery(
        self,
        trades_df: pd.DataFrame,
        price_df: pd.DataFrame,
        horizon: int,
        target: float,
        recovery_target: float,
        adverse_threshold: Optional[float],
        checkpoint_mode: str,
        fixed_checkpoint: int,
        flip_threshold: float,
        cut_threshold: float,
    ) -> list[RecoveryCandidate]:
        """Analyze each trade for recovery potential."""
        candidates = []

        for _, t in trades_df.iterrows():
            dt = pd.Timestamp(t["date"])
            if dt not in price_df.index:
                continue

            entry_price = float(price_df.loc[dt, "btc_close"])
            if not np.isfinite(entry_price) or entry_price <= 0:
                continue

            trade_type = t["type"]
            direction = t["direction"]
            is_long = direction == "LONG"

            # Get checkpoint day
            checkpoint_day = self._get_checkpoint_day(trade_type, checkpoint_mode, fixed_checkpoint)
            if checkpoint_day >= horizon:
                checkpoint_day = horizon - 2  # Leave room for recovery

            # Get full path
            path = price_df.loc[dt:].iloc[1:horizon + 1]
            if len(path) < horizon:
                continue

            closes = path["btc_close"].to_numpy(dtype=float)
            highs = path["btc_high"].to_numpy(dtype=float)
            lows = path["btc_low"].to_numpy(dtype=float)

            if np.isnan(closes).any() or np.isnan(highs).any() or np.isnan(lows).any():
                continue

            # Check if original trade hit target
            if is_long:
                favorable = highs / entry_price - 1.0
            else:
                favorable = 1.0 - lows / entry_price

            original_hit = np.any(favorable >= target)

            # Check if trade already hit target on or before checkpoint day
            # Since favorable is computed from intraday highs/lows, a hit on checkpoint day
            # means the trade would already be closed — it should not become a recovery candidate
            pre_checkpoint_favorable = favorable[:checkpoint_day]
            hit_before_checkpoint = np.any(pre_checkpoint_favorable >= target)
            if hit_before_checkpoint:
                continue  # Already hit target, not a recovery candidate

            # Get checkpoint price and return
            checkpoint_price = closes[checkpoint_day - 1]  # -1 because path starts at day 1
            if is_long:
                checkpoint_return = checkpoint_price / entry_price - 1.0
            else:
                checkpoint_return = 1.0 - checkpoint_price / entry_price

            # Determine adverse threshold using PolicyConfigAdapter
            if adverse_threshold is not None:
                thresh = adverse_threshold
            else:
                # Use policy-derived MAE(W) p75 / 2, with fallback to hardcoded values
                thresh = PolicyConfigAdapter.get_adverse_threshold(trade_type)

            # Only consider trades that are adverse at checkpoint
            # checkpoint_return < 0 means price moved against original direction
            is_adverse = checkpoint_return < -thresh

            if not is_adverse:
                continue  # Not a recovery candidate

            # Calculate forward return from checkpoint to horizon end
            remaining_days = horizon - checkpoint_day
            remaining_path_closes = closes[checkpoint_day:]
            remaining_path_highs = highs[checkpoint_day:]
            remaining_path_lows = lows[checkpoint_day:]

            if len(remaining_path_closes) == 0:
                continue

            # Forward return in original direction (from checkpoint)
            if is_long:
                forward_favorable = remaining_path_highs / checkpoint_price - 1.0
                forward_adverse = 1.0 - remaining_path_lows / checkpoint_price
            else:
                forward_favorable = 1.0 - remaining_path_lows / checkpoint_price
                forward_adverse = remaining_path_highs / checkpoint_price - 1.0

            forward_return = remaining_path_closes[-1] / checkpoint_price - 1.0
            if not is_long:
                forward_return = -forward_return

            # Recovery trade: flip direction at checkpoint
            # If original was LONG and failing, go SHORT from checkpoint
            # Recovery return = adverse direction from checkpoint
            if is_long:
                recovery_favorable = 1.0 - remaining_path_lows / checkpoint_price
            else:
                recovery_favorable = remaining_path_highs / checkpoint_price - 1.0

            recovery_hit = np.any(recovery_favorable >= recovery_target)
            recovery_return = float(np.max(recovery_favorable))

            candidates.append(RecoveryCandidate(
                date=dt.strftime("%Y-%m-%d"),
                trade_type=trade_type,
                direction=direction,
                entry_price=entry_price,
                checkpoint_day=checkpoint_day,
                checkpoint_price=checkpoint_price,
                checkpoint_return=checkpoint_return,
                remaining_days=remaining_days,
                forward_return=forward_return,
                original_hit=original_hit,
                recovery_hit=recovery_hit,
                recovery_return=recovery_return,
                flip_threshold=flip_threshold,
                cut_threshold=cut_threshold,
            ))

        return candidates

    def _analyze_optimal_thresholds(self, df: pd.DataFrame) -> list[dict]:
        """
        Analyze different threshold combinations to find optimal performance.
        
        Tests various cut and flip threshold combinations and scores them based on:
        - Win rate (primary factor)
        - Cut rate (secondary factor - rewards cutting losing trades)
        - Flip rate balance (penalty for extreme flip rates)
        
        Args:
            df: DataFrame of recovery candidates
            
        Returns:
            List of threshold results sorted by performance score (best first)
        """
        results = []
        
        # Test threshold combinations
        cut_thresholds = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]  # 0.5% to 4%
        flip_thresholds = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]     # 3% to 10%
        
        for cut_thresh in cut_thresholds:
            for flip_thresh in flip_thresholds:
                if cut_thresh >= flip_thresh:
                    continue  # Skip invalid combinations
                
                # Calculate recommendations with these thresholds
                recommendations = []
                for _, row in df.iterrows():
                    recovery_mfe = row["recovery_return"]
                    if recovery_mfe > flip_thresh:
                        rec = "FLIP"
                    elif recovery_mfe < cut_thresh:
                        rec = "CUT"
                    else:
                        rec = "HOLD"
                    recommendations.append(rec)
                
                # Calculate performance metrics
                rec_series = pd.Series(recommendations, index=df.index)
                flip_trades = df[rec_series == "FLIP"]
                hold_trades = df[rec_series == "HOLD"]
                cut_trades = df[rec_series == "CUT"]
                
                # Calculate expected wins following these recommendations
                expected_wins = 0
                total_decisions = len(df)
                
                if len(flip_trades) > 0:
                    expected_wins += flip_trades["recovery_hit"].sum()
                    
                if len(hold_trades) > 0:
                    expected_wins += hold_trades["original_hit"].sum()
                
                # CUT trades contribute 0 wins but avoid losses
                
                win_rate = expected_wins / total_decisions * 100 if total_decisions > 0 else 0
                
                # Calculate strategy distribution
                flip_count = len(flip_trades)
                hold_count = len(hold_trades)
                cut_count = len(cut_trades)
                
                flip_rate = flip_count / total_decisions * 100 if total_decisions > 0 else 0
                cut_rate = cut_count / total_decisions * 100 if total_decisions > 0 else 0
                
                # Performance score calculation
                # Base score is win rate
                score = win_rate
                
                # Bonus for cutting losing trades (up to +10 points)
                score += min(cut_rate * 0.5, 10)
                
                # Penalty for extreme flip rates (prefer ~40% flip rate)
                optimal_flip_rate = 40.0
                flip_penalty = abs(flip_rate - optimal_flip_rate) * 0.1
                score -= flip_penalty
                
                results.append({
                    'cut_threshold': cut_thresh,
                    'flip_threshold': flip_thresh,
                    'win_rate': win_rate,
                    'flip_count': flip_count,
                    'hold_count': hold_count,
                    'cut_count': cut_count,
                    'flip_rate': flip_rate,
                    'cut_rate': cut_rate,
                    'score': score
                })
        
        # Sort by score (best first)
        return sorted(results, key=lambda x: x['score'], reverse=True)

    def _generate_policy_suggestions(self, df: pd.DataFrame, flip_threshold: float, cut_threshold: float) -> list[dict]:
        """
        Generate policy enhancement suggestions based on signal type analysis.
        
        Analyzes which signal types have highest flip edge and suggests optimal
        recovery parameters per signal type.
        
        Args:
            df: DataFrame of recovery candidates
            flip_threshold: Current flip threshold
            cut_threshold: Current cut threshold
            
        Returns:
            List of policy suggestions per signal type
        """
        suggestions = []
        
        for signal_type in sorted(df["trade_type"].unique()):
            type_df = df[df["trade_type"] == signal_type]
            n = len(type_df)
            
            if n == 0:
                continue
                
            # Calculate edge (recovery hit rate - original hit rate for adverse trades)
            orig_hit_rate = type_df["original_hit"].mean()
            recovery_hit_rate = type_df["recovery_hit"].mean()
            edge = (recovery_hit_rate - orig_hit_rate) * 100
            
            # Calculate optimal thresholds for this signal type
            if n >= 5:  # Need sufficient data for threshold optimization
                type_optimal = self._analyze_optimal_thresholds(type_df)
                if type_optimal:
                    best_flip_thresh = type_optimal[0]['flip_threshold']
                    best_cut_thresh = type_optimal[0]['cut_threshold']
                else:
                    best_flip_thresh = flip_threshold
                    best_cut_thresh = cut_threshold
            else:
                # Use global thresholds for small samples
                best_flip_thresh = flip_threshold
                best_cut_thresh = cut_threshold
            
            # Calculate signal-specific recovery target based on MFE distribution
            recovery_mfe_values = type_df["recovery_return"].values
            if len(recovery_mfe_values) > 0:
                # Use 75th percentile as recovery target (achievable but not too easy)
                recovery_target = np.percentile(recovery_mfe_values, 75)
                # Ensure minimum 2% target
                recovery_target = max(recovery_target, 0.02)
                # Cap at 8% to be realistic
                recovery_target = min(recovery_target, 0.08)
            else:
                recovery_target = 0.03  # Default 3%
            
            suggestions.append({
                'signal_type': signal_type,
                'sample_size': n,
                'edge': edge,
                'flip_threshold': best_flip_thresh,
                'cut_threshold': best_cut_thresh,
                'recovery_target': recovery_target,
                'orig_hit_rate': orig_hit_rate * 100,
                'recovery_hit_rate': recovery_hit_rate * 100,
            })
        
        # Sort by edge (highest first)
        return sorted(suggestions, key=lambda x: x['edge'], reverse=True)

    def _show_threshold_impact_analysis(self, df: pd.DataFrame, current_flip_threshold: float, current_cut_threshold: float):
        """
        Show impact of different threshold values on recommendation distribution.
        
        Tests various threshold combinations and shows how they affect the
        distribution of FLIP/HOLD/CUT recommendations and expected performance.
        
        Args:
            df: DataFrame of recovery candidates
            current_flip_threshold: Current flip threshold
            current_cut_threshold: Current cut threshold
        """
        self.stdout.write("Testing different threshold combinations:")
        self.stdout.write("")
        
        # Test a range of threshold combinations
        test_combinations = [
            (0.01, 0.03),  # Conservative: 1% cut, 3% flip
            (0.015, 0.04), # Moderate-Conservative: 1.5% cut, 4% flip
            (0.02, 0.05),  # Current default: 2% cut, 5% flip
            (0.025, 0.06), # Moderate-Aggressive: 2.5% cut, 6% flip
            (0.03, 0.07),  # Aggressive: 3% cut, 7% flip
            (0.035, 0.08), # Very Aggressive: 3.5% cut, 8% flip
        ]
        
        self.stdout.write(f"{'Cut Thresh':>10} | {'Flip Thresh':>11} | {'FLIP':>4} | {'HOLD':>4} | {'CUT':>3} | {'Win Rate':>8} | {'vs Current':>10}")
        self.stdout.write("-" * 75)
        
        # Calculate current performance for comparison
        current_recommendations = []
        for _, row in df.iterrows():
            recovery_mfe = row["recovery_return"]
            if recovery_mfe > current_flip_threshold:
                rec = "FLIP"
            elif recovery_mfe < current_cut_threshold:
                rec = "CUT"
            else:
                rec = "HOLD"
            current_recommendations.append(rec)
        
        current_performance = self._calculate_strategy_performance(df, current_recommendations)
        
        for cut_thresh, flip_thresh in test_combinations:
            # Calculate recommendations with these thresholds
            recommendations = []
            for _, row in df.iterrows():
                recovery_mfe = row["recovery_return"]
                if recovery_mfe > flip_thresh:
                    rec = "FLIP"
                elif recovery_mfe < cut_thresh:
                    rec = "CUT"
                else:
                    rec = "HOLD"
                recommendations.append(rec)
            
            # Calculate performance
            performance = self._calculate_strategy_performance(df, recommendations)
            
            # Count recommendations
            rec_counts = pd.Series(recommendations).value_counts()
            flip_count = rec_counts.get("FLIP", 0)
            hold_count = rec_counts.get("HOLD", 0)
            cut_count = rec_counts.get("CUT", 0)
            
            # Performance difference vs current
            diff = performance - current_performance
            diff_str = f"{diff:+.1f}%" if diff != 0 else "  0.0%"
            
            # Highlight current settings
            marker = " *" if (cut_thresh == current_cut_threshold and flip_thresh == current_flip_threshold) else "  "
            
            self.stdout.write(
                f"{cut_thresh*100:>9.1f}% | {flip_thresh*100:>10.1f}% | {flip_count:>4} | {hold_count:>4} | "
                f"{cut_count:>3} | {performance:>7.1f}% | {diff_str:>9}{marker}"
            )
        
        self.stdout.write("")
        self.stdout.write("* = Current settings")
        self.stdout.write("")
        
        # Show sensitivity analysis
        self.stdout.write("Threshold Sensitivity Analysis:")
        self.stdout.write("")
        
        # Test flip threshold sensitivity (keeping cut threshold constant)
        self.stdout.write(f"Flip Threshold Sensitivity (Cut Threshold = {current_cut_threshold*100:.1f}%):")
        flip_test_values = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
        
        for flip_thresh in flip_test_values:
            recommendations = []
            for _, row in df.iterrows():
                recovery_mfe = row["recovery_return"]
                if recovery_mfe > flip_thresh:
                    rec = "FLIP"
                elif recovery_mfe < current_cut_threshold:
                    rec = "CUT"
                else:
                    rec = "HOLD"
                recommendations.append(rec)
            
            performance = self._calculate_strategy_performance(df, recommendations)
            rec_counts = pd.Series(recommendations).value_counts()
            flip_count = rec_counts.get("FLIP", 0)
            
            marker = " *" if flip_thresh == current_flip_threshold else "  "
            self.stdout.write(f"  {flip_thresh*100:>5.1f}%: {flip_count:>3} FLIP trades, {performance:>5.1f}% win rate{marker}")
        
        self.stdout.write("")
        
        # Test cut threshold sensitivity (keeping flip threshold constant)
        self.stdout.write(f"Cut Threshold Sensitivity (Flip Threshold = {current_flip_threshold*100:.1f}%):")
        cut_test_values = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]
        
        for cut_thresh in cut_test_values:
            recommendations = []
            for _, row in df.iterrows():
                recovery_mfe = row["recovery_return"]
                if recovery_mfe > current_flip_threshold:
                    rec = "FLIP"
                elif recovery_mfe < cut_thresh:
                    rec = "CUT"
                else:
                    rec = "HOLD"
                recommendations.append(rec)
            
            performance = self._calculate_strategy_performance(df, recommendations)
            rec_counts = pd.Series(recommendations).value_counts()
            cut_count = rec_counts.get("CUT", 0)
            
            marker = " *" if cut_thresh == current_cut_threshold else "  "
            self.stdout.write(f"  {cut_thresh*100:>5.1f}%: {cut_count:>3} CUT trades, {performance:>5.1f}% win rate{marker}")
        
        self.stdout.write("")
        self.stdout.write("Key Insights:")
        
        # Find optimal thresholds from the test combinations
        best_performance = 0
        best_combination = None
        
        for cut_thresh, flip_thresh in test_combinations:
            recommendations = []
            for _, row in df.iterrows():
                recovery_mfe = row["recovery_return"]
                if recovery_mfe > flip_thresh:
                    rec = "FLIP"
                elif recovery_mfe < cut_thresh:
                    rec = "CUT"
                else:
                    rec = "HOLD"
                recommendations.append(rec)
            
            performance = self._calculate_strategy_performance(df, recommendations)
            
            if performance > best_performance:
                best_performance = performance
                best_combination = (cut_thresh, flip_thresh)
        
        if best_combination:
            cut_opt, flip_opt = best_combination
            improvement = best_performance - current_performance
            
            if improvement > 1:
                self.stdout.write(f"  • Optimal thresholds: {cut_opt*100:.1f}% cut, {flip_opt*100:.1f}% flip (+{improvement:.1f}% improvement)")
            else:
                self.stdout.write(f"  • Current thresholds are near-optimal (best improvement: +{improvement:.1f}%)")
        
        # Analyze threshold stability
        performances = []
        for cut_thresh, flip_thresh in test_combinations:
            recommendations = []
            for _, row in df.iterrows():
                recovery_mfe = row["recovery_return"]
                if recovery_mfe > flip_thresh:
                    rec = "FLIP"
                elif recovery_mfe < cut_thresh:
                    rec = "CUT"
                else:
                    rec = "HOLD"
                recommendations.append(rec)
            
            performance = self._calculate_strategy_performance(df, recommendations)
            performances.append(performance)
        
        performance_std = np.std(performances)
        
        if performance_std < 2:
            self.stdout.write("  • Strategy performance is stable across different thresholds")
        else:
            self.stdout.write("  • Strategy performance is sensitive to threshold selection")

    def _calculate_strategy_performance(self, df: pd.DataFrame, recommendations: list[str]) -> float:
        """
        Calculate expected win rate for a given set of recommendations.
        
        Args:
            df: DataFrame of recovery candidates
            recommendations: List of recommendations ("FLIP", "HOLD", "CUT") for each trade
            
        Returns:
            Expected win rate as percentage
        """
        if len(df) == 0 or len(recommendations) == 0:
            return 0.0
        
        expected_wins = 0
        total_decisions = len(df)
        
        for i, (_, row) in enumerate(df.iterrows()):
            if i >= len(recommendations):
                break
                
            rec = recommendations[i]
            
            if rec == "FLIP":
                # If we flip, we win if recovery_hit is True
                if row["recovery_hit"]:
                    expected_wins += 1
            elif rec == "HOLD":
                # If we hold, we win if original_hit is True
                if row["original_hit"]:
                    expected_wins += 1
            # For CUT, we assume 0 wins (cut losses early)
        
        return expected_wins / total_decisions * 100 if total_decisions > 0 else 0.0

    def _run_sensitivity_analysis(
        self,
        trades_df: pd.DataFrame,
        price_df: pd.DataFrame,
        horizon: int,
        target: float,
        adverse_threshold: Optional[float],
        checkpoint_mode: str,
        fixed_checkpoint: int,
        flip_threshold: float,
        cut_threshold: float,
    ) -> dict:
        """
        Run sensitivity analysis testing different recovery thresholds.
        
        Tests recovery thresholds of 1%, 2%, 3%, 5%, 10% to find optimal
        threshold selection based on risk/reward analysis.
        
        Args:
            trades_df: DataFrame with trade data
            price_df: DataFrame with price data
            horizon: Forward path horizon in days
            target: Original target threshold
            adverse_threshold: Adverse threshold override
            checkpoint_mode: Checkpoint mode
            fixed_checkpoint: Fixed checkpoint day
            flip_threshold: Threshold for FLIP recommendation
            cut_threshold: Threshold for CUT recommendation
            
        Returns:
            Dictionary with sensitivity analysis results
        """
        # Test different recovery thresholds
        recovery_thresholds = [0.01, 0.02, 0.03, 0.05, 0.10]  # 1%, 2%, 3%, 5%, 10%
        
        results = []
        
        for recovery_target in recovery_thresholds:
            # Analyze recovery potential with this threshold
            candidates = self._analyze_recovery(
                trades_df, price_df, horizon, target, recovery_target,
                adverse_threshold, checkpoint_mode, fixed_checkpoint, 
                flip_threshold, cut_threshold
            )
            
            if not candidates:
                continue
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame([vars(c) for c in candidates])
            
            # Calculate key metrics
            total_candidates = len(candidates)
            orig_hit_rate = df["original_hit"].mean() * 100
            recovery_hit_rate = df["recovery_hit"].mean() * 100
            net_edge = recovery_hit_rate - orig_hit_rate
            
            # Calculate recommendations
            recommendations = []
            for _, row in df.iterrows():
                candidate = RecoveryCandidate(
                    date=row["date"],
                    trade_type=row["trade_type"],
                    direction=row["direction"],
                    entry_price=row["entry_price"],
                    checkpoint_day=row["checkpoint_day"],
                    checkpoint_price=row["checkpoint_price"],
                    checkpoint_return=row["checkpoint_return"],
                    remaining_days=row["remaining_days"],
                    forward_return=row["forward_return"],
                    original_hit=row["original_hit"],
                    recovery_hit=row["recovery_hit"],
                    recovery_return=row["recovery_return"],
                    flip_threshold=flip_threshold,
                    cut_threshold=cut_threshold,
                )
                recommendations.append(candidate.recommendation)
            
            # Count recommendations
            rec_counts = pd.Series(recommendations).value_counts()
            flip_count = rec_counts.get("FLIP", 0)
            hold_count = rec_counts.get("HOLD", 0)
            cut_count = rec_counts.get("CUT", 0)
            
            # Calculate expected performance with recovery policy
            flip_trades = df[pd.Series(recommendations) == "FLIP"]
            hold_trades = df[pd.Series(recommendations) == "HOLD"]
            
            expected_wins = 0
            if len(flip_trades) > 0:
                expected_wins += flip_trades["recovery_hit"].sum()
            if len(hold_trades) > 0:
                expected_wins += hold_trades["original_hit"].sum()
            
            expected_win_rate = expected_wins / total_candidates * 100 if total_candidates > 0 else 0
            improvement_vs_hold_all = expected_win_rate - orig_hit_rate
            
            # Calculate risk metrics
            avg_recovery_mfe = df["recovery_return"].mean() * 100
            recovery_mfe_std = df["recovery_return"].std() * 100
            
            # Calculate success rates by strategy
            flip_success_rate = 0
            hold_success_rate = 0
            
            if len(flip_trades) > 0:
                flip_success_rate = flip_trades["recovery_hit"].mean() * 100
            if len(hold_trades) > 0:
                hold_success_rate = hold_trades["original_hit"].mean() * 100
            
            # Risk/reward score calculation
            # Higher recovery threshold = higher bar for success but potentially higher reward
            # Score balances win rate improvement with strategy distribution
            risk_reward_score = (
                improvement_vs_hold_all * 2 +  # Primary factor: improvement
                (flip_success_rate - 50) * 0.5 +  # Bonus for high flip success rate
                (recovery_hit_rate - 50) * 0.3 +  # Bonus for high overall recovery rate
                min(flip_count / total_candidates * 100, 50) * 0.2  # Bonus for reasonable flip rate (cap at 50%)
            )
            
            results.append({
                'recovery_threshold': recovery_target,
                'total_candidates': total_candidates,
                'orig_hit_rate': orig_hit_rate,
                'recovery_hit_rate': recovery_hit_rate,
                'net_edge': net_edge,
                'expected_win_rate': expected_win_rate,
                'improvement_vs_hold_all': improvement_vs_hold_all,
                'flip_count': flip_count,
                'hold_count': hold_count,
                'cut_count': cut_count,
                'flip_success_rate': flip_success_rate,
                'hold_success_rate': hold_success_rate,
                'avg_recovery_mfe': avg_recovery_mfe,
                'recovery_mfe_std': recovery_mfe_std,
                'risk_reward_score': risk_reward_score,
            })
        
        return {
            'results': results,
            'optimal_threshold': max(results, key=lambda x: x['risk_reward_score']) if results else None,
            'test_thresholds': recovery_thresholds,
        }

    def _print_sensitivity_analysis_results(self, sensitivity_results: dict):
        """
        Print sensitivity analysis results showing optimal threshold selection.
        
        Args:
            sensitivity_results: Dictionary with sensitivity analysis results
        """
        if not sensitivity_results or not sensitivity_results.get('results'):
            self.stdout.write("No sensitivity analysis results available.")
            return
        
        results = sensitivity_results['results']
        optimal = sensitivity_results['optimal_threshold']
        
        self.stdout.write("\n" + "=" * 120)
        self.stdout.write("RECOVERY THRESHOLD SENSITIVITY ANALYSIS")
        self.stdout.write("=" * 120)
        
        self.stdout.write("Testing different recovery thresholds to find optimal risk/reward balance...")
        self.stdout.write("")
        
        # === SENSITIVITY TABLE ===
        self.stdout.write("📊 THRESHOLD SENSITIVITY TABLE")
        self.stdout.write("-" * 120)
        self.stdout.write(
            f"{'Recovery':>8} | {'Candidates':>10} | {'Orig Hit':>8} | {'Recov Hit':>9} | "
            f"{'Net Edge':>8} | {'Expected':>8} | {'Improvement':>11} | {'FLIP':>4} | {'HOLD':>4} | "
            f"{'CUT':>3} | {'Risk/Reward':>11}"
        )
        self.stdout.write(
            f"{'Threshold':>8} | {'':>10} | {'Rate':>8} | {'Rate':>9} | "
            f"{'':>8} | {'Win Rate':>8} | {'vs Hold-All':>11} | {'':>4} | {'':>4} | "
            f"{'':>3} | {'Score':>11}"
        )
        self.stdout.write("-" * 120)
        
        for result in results:
            recovery_thresh = result['recovery_threshold']
            candidates = result['total_candidates']
            orig_hit = result['orig_hit_rate']
            recov_hit = result['recovery_hit_rate']
            net_edge = result['net_edge']
            expected_win = result['expected_win_rate']
            improvement = result['improvement_vs_hold_all']
            flip_count = result['flip_count']
            hold_count = result['hold_count']
            cut_count = result['cut_count']
            score = result['risk_reward_score']
            
            # Highlight optimal threshold
            marker = " *" if optimal and recovery_thresh == optimal['recovery_threshold'] else "  "
            
            self.stdout.write(
                f"{recovery_thresh*100:>7.1f}% | {candidates:>10} | {orig_hit:>7.1f}% | "
                f"{recov_hit:>8.1f}% | {net_edge:>+7.1f}% | {expected_win:>7.1f}% | "
                f"{improvement:>+10.1f}% | {flip_count:>4} | {hold_count:>4} | "
                f"{cut_count:>3} | {score:>+10.2f}{marker}"
            )
        
        self.stdout.write("")
        self.stdout.write("* = Optimal threshold based on risk/reward score")
        self.stdout.write("")
        
        # === OPTIMAL THRESHOLD ANALYSIS ===
        if optimal:
            self.stdout.write("🎯 OPTIMAL THRESHOLD ANALYSIS")
            self.stdout.write("-" * 60)
            
            opt_thresh = optimal['recovery_threshold']
            opt_improvement = optimal['improvement_vs_hold_all']
            opt_flip_success = optimal['flip_success_rate']
            opt_score = optimal['risk_reward_score']
            
            self.stdout.write(f"Optimal Recovery Threshold: {opt_thresh*100:.1f}%")
            self.stdout.write(f"Expected Win Rate Improvement: {opt_improvement:+.1f}%")
            self.stdout.write(f"FLIP Strategy Success Rate: {opt_flip_success:.1f}%")
            self.stdout.write(f"Risk/Reward Score: {opt_score:+.2f}")
            self.stdout.write("")
            
            # Compare with other thresholds
            self.stdout.write("Comparison with other thresholds:")
            
            # Find best and worst performing thresholds
            best_improvement = max(results, key=lambda x: x['improvement_vs_hold_all'])
            worst_improvement = min(results, key=lambda x: x['improvement_vs_hold_all'])
            highest_edge = max(results, key=lambda x: x['net_edge'])
            
            if best_improvement['recovery_threshold'] != opt_thresh:
                self.stdout.write(
                    f"  • Highest improvement: {best_improvement['recovery_threshold']*100:.1f}% "
                    f"({best_improvement['improvement_vs_hold_all']:+.1f}% vs {opt_improvement:+.1f}%)"
                )
            
            if highest_edge['recovery_threshold'] != opt_thresh:
                self.stdout.write(
                    f"  • Highest net edge: {highest_edge['recovery_threshold']*100:.1f}% "
                    f"({highest_edge['net_edge']:+.1f}% vs {optimal['net_edge']:+.1f}%)"
                )
            
            self.stdout.write("")
        
        # === THRESHOLD SELECTION INSIGHTS ===
        self.stdout.write("💡 THRESHOLD SELECTION INSIGHTS")
        self.stdout.write("-" * 60)
        
        # Analyze trends across thresholds
        improvements = [r['improvement_vs_hold_all'] for r in results]
        net_edges = [r['net_edge'] for r in results]
        flip_counts = [r['flip_count'] for r in results]
        
        # Find trends
        if len(results) >= 3:
            # Check if improvement generally increases or decreases with threshold
            low_thresh_avg = np.mean([r['improvement_vs_hold_all'] for r in results[:2]])
            high_thresh_avg = np.mean([r['improvement_vs_hold_all'] for r in results[-2:]])
            
            if high_thresh_avg > low_thresh_avg + 1:
                trend_insight = "Higher recovery thresholds tend to perform better"
            elif low_thresh_avg > high_thresh_avg + 1:
                trend_insight = "Lower recovery thresholds tend to perform better"
            else:
                trend_insight = "Performance is relatively stable across thresholds"
            
            self.stdout.write(f"Performance Trend: {trend_insight}")
            
            # Analyze flip count trends
            if flip_counts[0] > flip_counts[-1] * 1.5:
                flip_trend = "Lower thresholds generate more FLIP recommendations"
            elif flip_counts[-1] > flip_counts[0] * 1.5:
                flip_trend = "Higher thresholds generate more FLIP recommendations"
            else:
                flip_trend = "FLIP recommendation count is stable across thresholds"
            
            self.stdout.write(f"Strategy Distribution: {flip_trend}")
        
        # Risk analysis
        max_improvement = max(improvements)
        min_improvement = min(improvements)
        improvement_range = max_improvement - min_improvement
        
        if improvement_range > 5:
            risk_assessment = "HIGH SENSITIVITY - Threshold selection is critical"
        elif improvement_range > 2:
            risk_assessment = "MODERATE SENSITIVITY - Threshold selection matters"
        else:
            risk_assessment = "LOW SENSITIVITY - Performance is stable across thresholds"
        
        self.stdout.write(f"Sensitivity Assessment: {risk_assessment}")
        self.stdout.write(f"Performance Range: {min_improvement:+.1f}% to {max_improvement:+.1f}% ({improvement_range:.1f}% spread)")
        
        self.stdout.write("")
        
        # === IMPLEMENTATION RECOMMENDATIONS ===
        self.stdout.write("🚀 IMPLEMENTATION RECOMMENDATIONS")
        self.stdout.write("-" * 60)
        
        if optimal:
            opt_thresh = optimal['recovery_threshold']
            opt_improvement = optimal['improvement_vs_hold_all']
            
            # Generate recommendation based on optimal threshold performance
            if opt_improvement > 5:
                recommendation = f"🟢 STRONGLY RECOMMEND {opt_thresh*100:.1f}% recovery threshold"
                rationale = "Significant improvement with good risk/reward balance"
            elif opt_improvement > 2:
                recommendation = f"🟡 RECOMMEND {opt_thresh*100:.1f}% recovery threshold"
                rationale = "Moderate improvement with acceptable risk"
            elif opt_improvement > 0:
                recommendation = f"🔵 CONSIDER {opt_thresh*100:.1f}% recovery threshold"
                rationale = "Marginal improvement, low implementation risk"
            else:
                recommendation = "🔴 NOT RECOMMENDED - No clear benefit"
                rationale = "Current strategy appears optimal"
            
            self.stdout.write(f"Primary Recommendation: {recommendation}")
            self.stdout.write(f"Rationale: {rationale}")
            
            # Alternative recommendations
            if len(results) > 1:
                # Find second-best option
                sorted_results = sorted(results, key=lambda x: x['risk_reward_score'], reverse=True)
                if len(sorted_results) > 1:
                    second_best = sorted_results[1]
                    score_diff = optimal['risk_reward_score'] - second_best['risk_reward_score']
                    
                    if score_diff < 2:  # Close performance
                        self.stdout.write(
                            f"Alternative Option: {second_best['recovery_threshold']*100:.1f}% threshold "
                            f"(score: {second_best['risk_reward_score']:+.2f}, similar performance)"
                        )
            
            self.stdout.write("")
            
            # Configuration recommendations
            self.stdout.write("Configuration Parameters:")
            self.stdout.write(f"  --recovery-target {opt_thresh:.3f}")
            
            # Suggest complementary threshold adjustments
            if opt_thresh <= 0.02:  # Low recovery threshold
                self.stdout.write("  # Consider lower flip threshold for more aggressive strategy")
                self.stdout.write(f"  --recovery-flip-threshold 0.040")
            elif opt_thresh >= 0.05:  # High recovery threshold
                self.stdout.write("  # Consider higher flip threshold for more conservative strategy")
                self.stdout.write(f"  --recovery-flip-threshold 0.070")
            
        self.stdout.write("")
        
        # === RISK CONSIDERATIONS ===
        self.stdout.write("⚠️ RISK CONSIDERATIONS")
        self.stdout.write("-" * 50)
        
        if optimal:
            opt_flip_success = optimal['flip_success_rate']
            opt_flip_count = optimal['flip_count']
            opt_total = optimal['total_candidates']
            
            # Assess FLIP strategy risk
            if opt_flip_success < 40:
                flip_risk = "🔴 HIGH RISK - Low FLIP success rate"
            elif opt_flip_success < 60:
                flip_risk = "🟡 MODERATE RISK - Average FLIP success rate"
            else:
                flip_risk = "🟢 LOW RISK - High FLIP success rate"
            
            self.stdout.write(f"FLIP Strategy Risk: {flip_risk} ({opt_flip_success:.1f}% success)")
            
            # Assess strategy concentration risk
            flip_concentration = opt_flip_count / opt_total * 100
            if flip_concentration > 70:
                concentration_risk = "🔴 HIGH CONCENTRATION - Over-reliance on FLIP strategy"
            elif flip_concentration > 50:
                concentration_risk = "🟡 MODERATE CONCENTRATION - Balanced strategy mix"
            else:
                concentration_risk = "🟢 LOW CONCENTRATION - Diversified strategy approach"
            
            self.stdout.write(f"Strategy Concentration: {concentration_risk} ({flip_concentration:.1f}% FLIP)")
            
            # Market regime sensitivity
            recovery_mfe_std = optimal['recovery_mfe_std']
            if recovery_mfe_std > 5:
                regime_risk = "🔴 HIGH SENSITIVITY - Performance may vary significantly across market conditions"
            elif recovery_mfe_std > 3:
                regime_risk = "🟡 MODERATE SENSITIVITY - Some variation expected across market conditions"
            else:
                regime_risk = "🟢 LOW SENSITIVITY - Stable performance across market conditions"
            
            self.stdout.write(f"Market Regime Sensitivity: {regime_risk}")
        
        self.stdout.write("")
        
        # === MONITORING RECOMMENDATIONS ===
        self.stdout.write("📊 MONITORING RECOMMENDATIONS")
        self.stdout.write("-" * 50)
        
        self.stdout.write("Key Metrics to Track:")
        self.stdout.write("  • FLIP strategy success rate (target: >50%)")
        self.stdout.write("  • Overall win rate improvement vs baseline")
        self.stdout.write("  • Strategy distribution (FLIP/HOLD/CUT ratios)")
        self.stdout.write("  • Recovery MFE distribution changes")
        
        self.stdout.write("")
        self.stdout.write("Review Triggers:")
        self.stdout.write("  • FLIP success rate drops below 40% for 30+ days")
        self.stdout.write("  • Win rate improvement becomes negative")
        self.stdout.write("  • Significant market regime changes")
        self.stdout.write("  • Strategy distribution shifts dramatically")
        
        self.stdout.write("")

    def _print_recovery_strategy_comparison(
        self,
        df: pd.DataFrame,
        flip_threshold: float,
        cut_threshold: float,
        target: float,
        recovery_target: float,
    ):
        """
        Print recovery strategy comparison showing quantitative benefits of recovery policy
        vs current hold-until-expiry approach.
        
        Compares:
        1. Current "hold until expiry" strategy vs proposed recovery policy
        2. Expected P&L improvement from recovery policy
        3. Win rate improvements by signal type
        
        Args:
            df: DataFrame of recovery candidates
            flip_threshold: Threshold for FLIP recommendation
            cut_threshold: Threshold for CUT recommendation
            target: Original target threshold
            recovery_target: Recovery target threshold
        """
        if len(df) == 0:
            self.stdout.write("No recovery candidates available for comparison.")
            return
        
        self.stdout.write("Quantitative Analysis: Hold Until Expiry vs Recovery Policy")
        self.stdout.write("")
        
        # Calculate current recommendations for all candidates
        recommendations = []
        for _, row in df.iterrows():
            recovery_mfe = row["recovery_return"]
            if recovery_mfe > flip_threshold:
                rec = "FLIP"
            elif recovery_mfe < cut_threshold:
                rec = "CUT"
            else:
                rec = "HOLD"
            recommendations.append(rec)
        
        df_with_rec = df.copy()
        df_with_rec["recommendation"] = recommendations
        
        # === STRATEGY COMPARISON OVERVIEW ===
        self.stdout.write("📊 STRATEGY COMPARISON OVERVIEW")
        self.stdout.write("-" * 60)
        
        total_candidates = len(df)
        
        # Current "Hold Until Expiry" Strategy
        hold_all_wins = df["original_hit"].sum()
        hold_all_win_rate = (hold_all_wins / total_candidates * 100) if total_candidates > 0 else 0
        
        # Proposed Recovery Policy Strategy
        flip_trades = df_with_rec[df_with_rec["recommendation"] == "FLIP"]
        hold_trades = df_with_rec[df_with_rec["recommendation"] == "HOLD"]
        cut_trades = df_with_rec[df_with_rec["recommendation"] == "CUT"]
        
        recovery_policy_wins = 0
        if len(flip_trades) > 0:
            recovery_policy_wins += flip_trades["recovery_hit"].sum()
        if len(hold_trades) > 0:
            recovery_policy_wins += hold_trades["original_hit"].sum()
        # CUT trades contribute 0 wins but avoid further losses
        
        recovery_policy_win_rate = (recovery_policy_wins / total_candidates * 100) if total_candidates > 0 else 0
        
        # Calculate improvement
        win_rate_improvement = recovery_policy_win_rate - hold_all_win_rate
        
        self.stdout.write(f"Current Strategy (Hold Until Expiry):")
        self.stdout.write(f"  • Total Trades: {total_candidates}")
        self.stdout.write(f"  • Wins: {hold_all_wins}")
        self.stdout.write(f"  • Win Rate: {hold_all_win_rate:.1f}%")
        self.stdout.write("")
        
        self.stdout.write(f"Proposed Recovery Policy:")
        self.stdout.write(f"  • FLIP Trades: {len(flip_trades)} (Recovery wins: {flip_trades['recovery_hit'].sum() if len(flip_trades) > 0 else 0})")
        self.stdout.write(f"  • HOLD Trades: {len(hold_trades)} (Original wins: {hold_trades['original_hit'].sum() if len(hold_trades) > 0 else 0})")
        self.stdout.write(f"  • CUT Trades: {len(cut_trades)} (Losses avoided)")
        self.stdout.write(f"  • Total Wins: {recovery_policy_wins}")
        self.stdout.write(f"  • Win Rate: {recovery_policy_win_rate:.1f}%")
        self.stdout.write("")
        
        # Improvement metrics
        self.stdout.write("📈 IMPROVEMENT METRICS")
        self.stdout.write("-" * 40)
        self.stdout.write(f"Win Rate Improvement: {win_rate_improvement:+.1f}%")
        
        if win_rate_improvement > 0:
            relative_improvement = (win_rate_improvement / hold_all_win_rate * 100) if hold_all_win_rate > 0 else 0
            self.stdout.write(f"Relative Improvement: {relative_improvement:+.1f}%")
        
        # Expected P&L improvement (assuming equal position sizes)
        # This is a simplified calculation - in practice would need actual P&L data
        additional_wins = recovery_policy_wins - hold_all_wins
        if additional_wins > 0:
            # Assume average win = target% gain, average loss = -50% of target (simplified)
            avg_win_pnl = target * 100  # Convert to percentage points
            avg_loss_pnl = -target * 50  # Assume losses are 50% of target
            
            # Calculate expected P&L improvement per trade
            pnl_improvement_per_trade = (additional_wins / total_candidates) * avg_win_pnl
            
            self.stdout.write(f"Additional Wins: {additional_wins}")
            self.stdout.write(f"Expected P&L Improvement: {pnl_improvement_per_trade:+.2f}% per trade")
        else:
            self.stdout.write("Expected P&L Improvement: No improvement")
        
        self.stdout.write("")
        
        # === WIN RATE IMPROVEMENTS BY SIGNAL TYPE ===
        self.stdout.write("🎯 WIN RATE IMPROVEMENTS BY SIGNAL TYPE")
        self.stdout.write("-" * 80)
        self.stdout.write(f"{'Signal Type':<15} | {'N':>3} | {'Hold All':>8} | {'Recovery':>8} | {'Improvement':>11} | {'Strategy Mix':<20}")
        self.stdout.write("-" * 80)
        
        signal_improvements = []
        
        for signal_type in sorted(df["trade_type"].unique()):
            signal_df = df_with_rec[df_with_rec["trade_type"] == signal_type]
            n = len(signal_df)
            
            if n == 0:
                continue
            
            # Hold all strategy for this signal type
            hold_all_signal_wins = signal_df["original_hit"].sum()
            hold_all_signal_rate = (hold_all_signal_wins / n * 100) if n > 0 else 0
            
            # Recovery policy for this signal type
            signal_flip = signal_df[signal_df["recommendation"] == "FLIP"]
            signal_hold = signal_df[signal_df["recommendation"] == "HOLD"]
            signal_cut = signal_df[signal_df["recommendation"] == "CUT"]
            
            recovery_signal_wins = 0
            if len(signal_flip) > 0:
                recovery_signal_wins += signal_flip["recovery_hit"].sum()
            if len(signal_hold) > 0:
                recovery_signal_wins += signal_hold["original_hit"].sum()
            
            recovery_signal_rate = (recovery_signal_wins / n * 100) if n > 0 else 0
            
            # Improvement
            signal_improvement = recovery_signal_rate - hold_all_signal_rate
            
            # Strategy mix
            flip_count = len(signal_flip)
            hold_count = len(signal_hold)
            cut_count = len(signal_cut)
            strategy_mix = f"F:{flip_count} H:{hold_count} C:{cut_count}"
            
            self.stdout.write(
                f"{signal_type:<15} | {n:>3} | {hold_all_signal_rate:>7.1f}% | "
                f"{recovery_signal_rate:>7.1f}% | {signal_improvement:>+10.1f}% | {strategy_mix:<20}"
            )
            
            signal_improvements.append({
                'signal_type': signal_type,
                'n': n,
                'improvement': signal_improvement,
                'hold_rate': hold_all_signal_rate,
                'recovery_rate': recovery_signal_rate
            })
        
        self.stdout.write("")
        
        # === KEY INSIGHTS ===
        self.stdout.write("💡 KEY INSIGHTS")
        self.stdout.write("-" * 40)
        
        # Find best and worst performing signal types
        if signal_improvements:
            best_signal = max(signal_improvements, key=lambda x: x['improvement'])
            worst_signal = min(signal_improvements, key=lambda x: x['improvement'])
            
            self.stdout.write(f"Best Performing Signal: {best_signal['signal_type']} ({best_signal['improvement']:+.1f}% improvement)")
            self.stdout.write(f"Worst Performing Signal: {worst_signal['signal_type']} ({worst_signal['improvement']:+.1f}% improvement)")
            
            # Count signals with positive improvement
            positive_signals = [s for s in signal_improvements if s['improvement'] > 0]
            self.stdout.write(f"Signals with Positive Improvement: {len(positive_signals)}/{len(signal_improvements)}")
            
            # Average improvement across all signals
            avg_improvement = sum(s['improvement'] for s in signal_improvements) / len(signal_improvements)
            self.stdout.write(f"Average Improvement Across Signals: {avg_improvement:+.1f}%")
        
        self.stdout.write("")
        
        # === IMPLEMENTATION IMPACT ===
        self.stdout.write("🚀 IMPLEMENTATION IMPACT")
        self.stdout.write("-" * 50)
        
        # Calculate the impact of implementing the recovery policy
        if win_rate_improvement > 5:
            impact_level = "🟢 HIGH IMPACT"
            recommendation = "Strongly recommend immediate implementation"
        elif win_rate_improvement > 2:
            impact_level = "🟡 MODERATE IMPACT"
            recommendation = "Recommend implementation with monitoring"
        elif win_rate_improvement > 0:
            impact_level = "🔵 LOW IMPACT"
            recommendation = "Consider implementation for marginal gains"
        else:
            impact_level = "🔴 NO IMPACT"
            recommendation = "Current strategy appears optimal"
        
        self.stdout.write(f"Impact Level: {impact_level}")
        self.stdout.write(f"Recommendation: {recommendation}")
        self.stdout.write("")
        
        # Calculate potential annual impact (assuming regular trading)
        # This is illustrative - actual impact depends on trading frequency
        if total_candidates > 0 and win_rate_improvement > 0:
            # Assume this represents a sample period, extrapolate to annual
            sample_period_days = 365  # Assume 1 year sample
            trades_per_year = total_candidates  # Simplified assumption
            
            additional_wins_per_year = (additional_wins / total_candidates) * trades_per_year
            
            self.stdout.write("📅 PROJECTED ANNUAL IMPACT (Illustrative)")
            self.stdout.write(f"Additional Wins per Year: {additional_wins_per_year:.1f}")
            self.stdout.write(f"Win Rate Improvement: {win_rate_improvement:+.1f}%")
            
            if additional_wins_per_year >= 1:
                self.stdout.write("✅ Significant annual improvement expected")
            else:
                self.stdout.write("⚠️ Modest annual improvement expected")
        
        self.stdout.write("")
        
        # === RISK CONSIDERATIONS ===
        self.stdout.write("⚠️ RISK CONSIDERATIONS")
        self.stdout.write("-" * 40)
        
        # Calculate the risk of the recovery policy
        flip_failure_rate = 0
        if len(flip_trades) > 0:
            flip_failures = len(flip_trades) - flip_trades["recovery_hit"].sum()
            flip_failure_rate = (flip_failures / len(flip_trades) * 100) if len(flip_trades) > 0 else 0
        
        cut_opportunity_cost = 0
        if len(cut_trades) > 0:
            # Opportunity cost: trades we cut that would have recovered
            cut_would_have_won = cut_trades["original_hit"].sum() + cut_trades["recovery_hit"].sum()
            cut_opportunity_cost = (cut_would_have_won / len(cut_trades) * 100) if len(cut_trades) > 0 else 0
        
        self.stdout.write("Risk Metrics:")
        if len(flip_trades) > 0:
            self.stdout.write(f"  • FLIP Strategy Failure Rate: {flip_failure_rate:.1f}%")
        if len(cut_trades) > 0:
            self.stdout.write(f"  • CUT Strategy Opportunity Cost: {cut_opportunity_cost:.1f}%")
        
        # Overall risk assessment
        if flip_failure_rate > 50:
            self.stdout.write("  ⚠️ High failure rate on FLIP trades - consider more conservative thresholds")
        elif flip_failure_rate > 30:
            self.stdout.write("  ⚠️ Moderate failure rate on FLIP trades - monitor performance")
        else:
            self.stdout.write("  ✅ Acceptable failure rate on FLIP trades")
        
        if cut_opportunity_cost > 30:
            self.stdout.write("  ⚠️ High opportunity cost on CUT trades - consider lower cut threshold")
        elif cut_opportunity_cost > 15:
            self.stdout.write("  ⚠️ Moderate opportunity cost on CUT trades - monitor performance")
        else:
            self.stdout.write("  ✅ Acceptable opportunity cost on CUT trades")
        
        self.stdout.write("")

    def _simulate_policy_backtesting(
        self,
        candidates: list[RecoveryCandidate],
        horizon: int,
        target: float,
        recovery_target: float,
        flip_threshold: float,
        cut_threshold: float,
    ) -> dict:
        """
        Simulate policy backtesting to calculate P&L impact of recovery decisions.
        
        Compares three strategies:
        1. Hold-to-expiry (baseline): Always hold original position until horizon end
        2. Recovery policy: Follow FLIP/HOLD/CUT recommendations based on thresholds
        3. Always-flip: Always flip direction at checkpoint (for comparison)
        
        Args:
            candidates: List of recovery candidates
            horizon: Forward path horizon in days
            target: Original target threshold
            recovery_target: Recovery target threshold
            flip_threshold: Threshold for FLIP recommendation
            cut_threshold: Threshold for CUT recommendation
            
        Returns:
            Dictionary with simulation results including P&L metrics
        """
        if not candidates:
            return {}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame([vars(c) for c in candidates])
        
        # Calculate recommendations for all candidates
        recommendations = []
        for _, row in df.iterrows():
            candidate = RecoveryCandidate(
                date=row["date"],
                trade_type=row["trade_type"],
                direction=row["direction"],
                entry_price=row["entry_price"],
                checkpoint_day=row["checkpoint_day"],
                checkpoint_price=row["checkpoint_price"],
                checkpoint_return=row["checkpoint_return"],
                remaining_days=row["remaining_days"],
                forward_return=row["forward_return"],
                original_hit=row["original_hit"],
                recovery_hit=row["recovery_hit"],
                recovery_return=row["recovery_return"],
                flip_threshold=flip_threshold,
                cut_threshold=cut_threshold,
            )
            recommendations.append(candidate.recommendation)
        
        df["recommendation"] = recommendations
        
        # Simulation parameters
        total_trades = len(df)
        
        # === STRATEGY 1: Hold-to-Expiry (Baseline) ===
        hold_wins = df["original_hit"].sum()
        hold_losses = total_trades - hold_wins
        hold_win_rate = (hold_wins / total_trades * 100) if total_trades > 0 else 0
        
        # Calculate P&L for hold strategy
        # Assume: Win = +target%, Loss = -50% (simplified loss assumption)
        hold_pnl_per_trade = (hold_wins * target + hold_losses * (-target * 0.5)) / total_trades if total_trades > 0 else 0
        hold_total_pnl = hold_pnl_per_trade * total_trades
        
        # === STRATEGY 2: Recovery Policy ===
        flip_trades = df[df["recommendation"] == "FLIP"]
        hold_trades = df[df["recommendation"] == "HOLD"]
        cut_trades = df[df["recommendation"] == "CUT"]
        
        # Calculate wins for recovery policy
        policy_wins = 0
        policy_losses = 0
        policy_cuts = len(cut_trades)
        
        # FLIP trades: win if recovery hits, lose if recovery misses
        if len(flip_trades) > 0:
            flip_wins = flip_trades["recovery_hit"].sum()
            flip_losses = len(flip_trades) - flip_wins
            policy_wins += flip_wins
            policy_losses += flip_losses
        
        # HOLD trades: win if original hits, lose if original misses
        if len(hold_trades) > 0:
            hold_wins_policy = hold_trades["original_hit"].sum()
            hold_losses_policy = len(hold_trades) - hold_wins_policy
            policy_wins += hold_wins_policy
            policy_losses += hold_losses_policy
        
        # CUT trades: assume small loss (cut early to avoid larger loss)
        # Assume cutting saves 75% of potential loss
        cut_loss_per_trade = -target * 0.125  # 12.5% loss instead of 50%
        
        policy_win_rate = (policy_wins / total_trades * 100) if total_trades > 0 else 0
        
        # Calculate P&L for recovery policy
        policy_pnl = (
            policy_wins * target +  # Wins get full target
            policy_losses * (-target * 0.5) +  # Losses lose 50% of target
            policy_cuts * cut_loss_per_trade  # Cuts lose 12.5% of target
        )
        policy_pnl_per_trade = policy_pnl / total_trades if total_trades > 0 else 0
        
        # === STRATEGY 3: Always-Flip (for comparison) ===
        flip_all_wins = df["recovery_hit"].sum()
        flip_all_losses = total_trades - flip_all_wins
        flip_all_win_rate = (flip_all_wins / total_trades * 100) if total_trades > 0 else 0
        
        # Calculate P&L for always-flip strategy
        flip_all_pnl_per_trade = (flip_all_wins * recovery_target + flip_all_losses * (-recovery_target * 0.5)) / total_trades if total_trades > 0 else 0
        flip_all_total_pnl = flip_all_pnl_per_trade * total_trades
        
        # === COMPARATIVE ANALYSIS ===
        policy_vs_hold_pnl = policy_pnl - hold_total_pnl
        policy_vs_hold_pnl_per_trade = policy_pnl_per_trade - hold_pnl_per_trade
        policy_vs_hold_win_rate = policy_win_rate - hold_win_rate
        
        flip_all_vs_hold_pnl = flip_all_total_pnl - hold_total_pnl
        flip_all_vs_hold_pnl_per_trade = flip_all_pnl_per_trade - hold_pnl_per_trade
        flip_all_vs_hold_win_rate = flip_all_win_rate - hold_win_rate
        
        # === RISK METRICS ===
        # Calculate maximum drawdown scenarios
        # Worst case: all FLIP trades fail, all HOLD trades fail, CUT trades still lose
        worst_case_policy_pnl = (
            len(flip_trades) * (-recovery_target * 0.5) +
            len(hold_trades) * (-target * 0.5) +
            len(cut_trades) * cut_loss_per_trade
        )
        worst_case_hold_pnl = total_trades * (-target * 0.5)
        
        # Best case: all recommendations work perfectly
        best_case_policy_pnl = (
            len(flip_trades) * recovery_target +
            len(hold_trades) * target +
            len(cut_trades) * 0  # CUT trades avoid losses
        )
        best_case_hold_pnl = hold_wins * target + hold_losses * (-target * 0.5)
        
        # === STATISTICAL SIGNIFICANCE ===
        # Calculate confidence intervals (simplified)
        policy_win_rate_std = np.sqrt(policy_win_rate * (100 - policy_win_rate) / total_trades) if total_trades > 0 else 0
        hold_win_rate_std = np.sqrt(hold_win_rate * (100 - hold_win_rate) / total_trades) if total_trades > 0 else 0
        
        # Z-score for win rate difference
        win_rate_diff_std = np.sqrt(policy_win_rate_std**2 + hold_win_rate_std**2)
        z_score = abs(policy_vs_hold_win_rate) / win_rate_diff_std if win_rate_diff_std > 0 else 0
        
        # Statistical significance (approximate)
        is_significant = z_score > 1.96  # 95% confidence level
        
        return {
            # Basic metrics
            "total_trades": total_trades,
            "target": target,
            "recovery_target": recovery_target,
            
            # Hold-to-expiry strategy
            "hold_wins": hold_wins,
            "hold_losses": hold_losses,
            "hold_win_rate": hold_win_rate,
            "hold_pnl_per_trade": hold_pnl_per_trade,
            "hold_total_pnl": hold_total_pnl,
            
            # Recovery policy strategy
            "policy_wins": policy_wins,
            "policy_losses": policy_losses,
            "policy_cuts": policy_cuts,
            "policy_win_rate": policy_win_rate,
            "policy_pnl_per_trade": policy_pnl_per_trade,
            "policy_total_pnl": policy_pnl,
            
            # Always-flip strategy
            "flip_all_wins": flip_all_wins,
            "flip_all_losses": flip_all_losses,
            "flip_all_win_rate": flip_all_win_rate,
            "flip_all_pnl_per_trade": flip_all_pnl_per_trade,
            "flip_all_total_pnl": flip_all_total_pnl,
            
            # Comparative metrics
            "policy_vs_hold_pnl": policy_vs_hold_pnl,
            "policy_vs_hold_pnl_per_trade": policy_vs_hold_pnl_per_trade,
            "policy_vs_hold_win_rate": policy_vs_hold_win_rate,
            
            "flip_all_vs_hold_pnl": flip_all_vs_hold_pnl,
            "flip_all_vs_hold_pnl_per_trade": flip_all_vs_hold_pnl_per_trade,
            "flip_all_vs_hold_win_rate": flip_all_vs_hold_win_rate,
            
            # Risk metrics
            "worst_case_policy_pnl": worst_case_policy_pnl,
            "worst_case_hold_pnl": worst_case_hold_pnl,
            "best_case_policy_pnl": best_case_policy_pnl,
            "best_case_hold_pnl": best_case_hold_pnl,
            
            # Statistical metrics
            "z_score": z_score,
            "is_significant": is_significant,
            "policy_win_rate_std": policy_win_rate_std,
            
            # Strategy breakdown
            "flip_count": len(flip_trades),
            "hold_count": len(hold_trades),
            "cut_count": len(cut_trades),
            
            # Detailed trade outcomes
            "flip_wins": flip_trades["recovery_hit"].sum() if len(flip_trades) > 0 else 0,
            "flip_losses": len(flip_trades) - flip_trades["recovery_hit"].sum() if len(flip_trades) > 0 else 0,
            "hold_wins_policy": hold_trades["original_hit"].sum() if len(hold_trades) > 0 else 0,
            "hold_losses_policy": len(hold_trades) - hold_trades["original_hit"].sum() if len(hold_trades) > 0 else 0,
        }

    def _print_policy_simulation_results(self, simulation: dict):
        """
        Print detailed policy simulation results.
        
        Args:
            simulation: Dictionary with simulation results from _simulate_policy_backtesting
        """
        if not simulation:
            self.stdout.write("No simulation data available.")
            return
        
        self.stdout.write("\n" + "=" * 120)
        self.stdout.write("POLICY BACKTESTING SIMULATION RESULTS")
        self.stdout.write("=" * 120)
        
        total_trades = simulation["total_trades"]
        target = simulation["target"]
        recovery_target = simulation["recovery_target"]
        
        # === STRATEGY COMPARISON OVERVIEW ===
        self.stdout.write("📊 STRATEGY COMPARISON OVERVIEW")
        self.stdout.write("-" * 80)
        
        self.stdout.write(f"{'Strategy':<20} | {'Wins':>5} | {'Losses':>6} | {'Cuts':>4} | {'Win Rate':>8} | {'P&L/Trade':>10} | {'Total P&L':>10}")
        self.stdout.write("-" * 80)
        
        # Hold-to-expiry (baseline)
        self.stdout.write(
            f"{'Hold-to-Expiry':<20} | {simulation['hold_wins']:>5} | {simulation['hold_losses']:>6} | "
            f"{'N/A':>4} | {simulation['hold_win_rate']:>7.1f}% | {simulation['hold_pnl_per_trade']*100:>+9.2f}% | "
            f"{simulation['hold_total_pnl']*100:>+9.2f}%"
        )
        
        # Recovery policy
        self.stdout.write(
            f"{'Recovery Policy':<20} | {simulation['policy_wins']:>5} | {simulation['policy_losses']:>6} | "
            f"{simulation['policy_cuts']:>4} | {simulation['policy_win_rate']:>7.1f}% | {simulation['policy_pnl_per_trade']*100:>+9.2f}% | "
            f"{simulation['policy_total_pnl']*100:>+9.2f}%"
        )
        
        # Always-flip (for reference)
        self.stdout.write(
            f"{'Always-Flip':<20} | {simulation['flip_all_wins']:>5} | {simulation['flip_all_losses']:>6} | "
            f"{'N/A':>4} | {simulation['flip_all_win_rate']:>7.1f}% | {simulation['flip_all_pnl_per_trade']*100:>+9.2f}% | "
            f"{simulation['flip_all_total_pnl']*100:>+9.2f}%"
        )
        
        self.stdout.write("")
        
        # === PERFORMANCE IMPROVEMENTS ===
        self.stdout.write("📈 PERFORMANCE IMPROVEMENTS")
        self.stdout.write("-" * 60)
        
        policy_improvement = simulation["policy_vs_hold_pnl_per_trade"] * 100
        policy_win_rate_improvement = simulation["policy_vs_hold_win_rate"]
        
        flip_all_improvement = simulation["flip_all_vs_hold_pnl_per_trade"] * 100
        flip_all_win_rate_improvement = simulation["flip_all_vs_hold_win_rate"]
        
        self.stdout.write("Recovery Policy vs Hold-to-Expiry:")
        self.stdout.write(f"  • P&L Improvement per Trade: {policy_improvement:+.2f}%")
        self.stdout.write(f"  • Win Rate Improvement: {policy_win_rate_improvement:+.1f}%")
        self.stdout.write(f"  • Total P&L Improvement: {simulation['policy_vs_hold_pnl']*100:+.2f}%")
        
        self.stdout.write("")
        self.stdout.write("Always-Flip vs Hold-to-Expiry (Reference):")
        self.stdout.write(f"  • P&L Improvement per Trade: {flip_all_improvement:+.2f}%")
        self.stdout.write(f"  • Win Rate Improvement: {flip_all_win_rate_improvement:+.1f}%")
        self.stdout.write(f"  • Total P&L Improvement: {simulation['flip_all_vs_hold_pnl']*100:+.2f}%")
        
        self.stdout.write("")
        
        # === STRATEGY BREAKDOWN ===
        self.stdout.write("🎯 RECOVERY POLICY STRATEGY BREAKDOWN")
        self.stdout.write("-" * 70)
        
        flip_count = simulation["flip_count"]
        hold_count = simulation["hold_count"]
        cut_count = simulation["cut_count"]
        
        self.stdout.write(f"Strategy Distribution:")
        self.stdout.write(f"  • FLIP Trades: {flip_count:>3} ({flip_count/total_trades*100:>5.1f}%) - Strong recovery potential")
        self.stdout.write(f"  • HOLD Trades: {hold_count:>3} ({hold_count/total_trades*100:>5.1f}%) - Moderate recovery potential")
        self.stdout.write(f"  • CUT Trades:  {cut_count:>3} ({cut_count/total_trades*100:>5.1f}%) - Weak recovery potential")
        
        self.stdout.write("")
        self.stdout.write("Strategy Performance:")
        
        if flip_count > 0:
            flip_wins = simulation["flip_wins"]
            flip_losses = simulation["flip_losses"]
            flip_success_rate = flip_wins / flip_count * 100
            self.stdout.write(f"  • FLIP Strategy: {flip_wins}/{flip_count} wins ({flip_success_rate:.1f}% success rate)")
        
        if hold_count > 0:
            hold_wins_policy = simulation["hold_wins_policy"]
            hold_losses_policy = simulation["hold_losses_policy"]
            hold_success_rate = hold_wins_policy / hold_count * 100
            self.stdout.write(f"  • HOLD Strategy: {hold_wins_policy}/{hold_count} wins ({hold_success_rate:.1f}% success rate)")
        
        if cut_count > 0:
            self.stdout.write(f"  • CUT Strategy: {cut_count} trades cut early (loss minimization)")
        
        self.stdout.write("")
        
        # === RISK ANALYSIS ===
        self.stdout.write("⚠️ RISK ANALYSIS")
        self.stdout.write("-" * 50)
        
        worst_case_policy = simulation["worst_case_policy_pnl"] * 100
        worst_case_hold = simulation["worst_case_hold_pnl"] * 100
        best_case_policy = simulation["best_case_policy_pnl"] * 100
        best_case_hold = simulation["best_case_hold_pnl"] * 100
        
        self.stdout.write("Scenario Analysis:")
        self.stdout.write(f"  Worst Case:")
        self.stdout.write(f"    • Recovery Policy: {worst_case_policy:+.2f}% total P&L")
        self.stdout.write(f"    • Hold-to-Expiry:  {worst_case_hold:+.2f}% total P&L")
        self.stdout.write(f"    • Risk Difference: {(worst_case_policy - worst_case_hold):+.2f}%")
        
        self.stdout.write(f"  Best Case:")
        self.stdout.write(f"    • Recovery Policy: {best_case_policy:+.2f}% total P&L")
        self.stdout.write(f"    • Hold-to-Expiry:  {best_case_hold:+.2f}% total P&L")
        self.stdout.write(f"    • Upside Difference: {(best_case_policy - best_case_hold):+.2f}%")
        
        # Risk assessment
        downside_protection = worst_case_policy > worst_case_hold
        upside_potential = best_case_policy > best_case_hold
        
        self.stdout.write("")
        self.stdout.write("Risk Assessment:")
        if downside_protection and upside_potential:
            risk_assessment = "🟢 FAVORABLE - Better upside and downside protection"
        elif upside_potential and not downside_protection:
            risk_assessment = "🟡 MIXED - Better upside but higher downside risk"
        elif downside_protection and not upside_potential:
            risk_assessment = "🟡 CONSERVATIVE - Better downside protection but limited upside"
        else:
            risk_assessment = "🔴 UNFAVORABLE - Higher risk with limited benefit"
        
        self.stdout.write(f"  • {risk_assessment}")
        
        self.stdout.write("")
        
        # === STATISTICAL SIGNIFICANCE ===
        self.stdout.write("📊 STATISTICAL SIGNIFICANCE")
        self.stdout.write("-" * 50)
        
        z_score = simulation["z_score"]
        is_significant = simulation["is_significant"]
        confidence_level = "95%" if is_significant else "<95%"
        
        self.stdout.write(f"Statistical Analysis (Sample Size: {total_trades}):")
        self.stdout.write(f"  • Z-Score: {z_score:.2f}")
        self.stdout.write(f"  • Confidence Level: {confidence_level}")
        self.stdout.write(f"  • Statistical Significance: {'Yes' if is_significant else 'No'}")
        
        if is_significant:
            self.stdout.write("  ✅ Results are statistically significant at 95% confidence level")
        else:
            self.stdout.write("  ⚠️ Results may not be statistically significant - larger sample needed")
        
        self.stdout.write("")
        
        # === IMPLEMENTATION RECOMMENDATIONS ===
        self.stdout.write("🚀 IMPLEMENTATION RECOMMENDATIONS")
        self.stdout.write("-" * 60)
        
        # Decision logic based on results
        pnl_improvement = policy_improvement
        win_rate_improvement = policy_win_rate_improvement
        
        if pnl_improvement > 2 and win_rate_improvement > 5 and is_significant:
            recommendation = "🟢 STRONGLY RECOMMEND - Significant improvement with statistical confidence"
            priority = "HIGH PRIORITY"
        elif pnl_improvement > 1 and win_rate_improvement > 2:
            recommendation = "🟡 RECOMMEND - Moderate improvement, monitor performance"
            priority = "MEDIUM PRIORITY"
        elif pnl_improvement > 0 and win_rate_improvement > 0:
            recommendation = "🔵 CONSIDER - Marginal improvement, low risk"
            priority = "LOW PRIORITY"
        else:
            recommendation = "🔴 NOT RECOMMENDED - Limited or negative improvement"
            priority = "NO ACTION"
        
        self.stdout.write(f"Overall Recommendation: {recommendation}")
        self.stdout.write(f"Implementation Priority: {priority}")
        
        self.stdout.write("")
        self.stdout.write("Key Success Factors:")
        
        if flip_count > 0:
            flip_success_rate = simulation["flip_wins"] / flip_count * 100
            if flip_success_rate > 60:
                self.stdout.write(f"  ✅ FLIP strategy highly effective ({flip_success_rate:.1f}% success rate)")
            elif flip_success_rate > 40:
                self.stdout.write(f"  ⚠️ FLIP strategy moderately effective ({flip_success_rate:.1f}% success rate)")
            else:
                self.stdout.write(f"  ❌ FLIP strategy underperforming ({flip_success_rate:.1f}% success rate)")
        
        if cut_count > 0:
            cut_benefit = cut_count / total_trades * 100
            if cut_benefit > 20:
                self.stdout.write(f"  ✅ CUT strategy prevents significant losses ({cut_benefit:.1f}% of trades)")
            elif cut_benefit > 10:
                self.stdout.write(f"  ⚠️ CUT strategy provides moderate protection ({cut_benefit:.1f}% of trades)")
            else:
                self.stdout.write(f"  ❌ CUT strategy limited impact ({cut_benefit:.1f}% of trades)")
        
        # Expected annual impact (illustrative)
        if total_trades > 0:
            annual_trades_estimate = total_trades  # Assume this represents annual sample
            annual_pnl_improvement = simulation["policy_vs_hold_pnl"] * 100
            
            self.stdout.write("")
            self.stdout.write("Projected Annual Impact (Illustrative):")
            self.stdout.write(f"  • Estimated Annual Trades: {annual_trades_estimate}")
            self.stdout.write(f"  • Expected P&L Improvement: {annual_pnl_improvement:+.2f}% annually")
            self.stdout.write(f"  • Per-Trade Improvement: {policy_improvement:+.2f}%")
            
            if annual_pnl_improvement > 5:
                self.stdout.write("  🚀 Significant annual improvement expected")
            elif annual_pnl_improvement > 2:
                self.stdout.write("  📈 Moderate annual improvement expected")
            elif annual_pnl_improvement > 0:
                self.stdout.write("  📊 Modest annual improvement expected")
            else:
                self.stdout.write("  ⚠️ Limited or no annual improvement expected")
        
        self.stdout.write("")
        
        # === NEXT STEPS ===
        self.stdout.write("📋 NEXT STEPS")
        self.stdout.write("-" * 30)
        
        if pnl_improvement > 1:
            self.stdout.write("1. ✅ Implement recovery policy in execution system")
            self.stdout.write("2. 📊 Set up monitoring and performance tracking")
            self.stdout.write("3. 🔧 Fine-tune thresholds based on live performance")
            self.stdout.write("4. 📈 Expand to additional signal types if successful")
        elif pnl_improvement > 0:
            self.stdout.write("1. 🧪 Run extended backtesting with larger sample size")
            self.stdout.write("2. 📊 Test on different market conditions")
            self.stdout.write("3. 🔧 Optimize thresholds for better performance")
            self.stdout.write("4. ⚠️ Consider paper trading before live implementation")
        else:
            self.stdout.write("1. 🔍 Investigate why recovery policy underperforms")
            self.stdout.write("2. 📊 Analyze different market regimes separately")
            self.stdout.write("3. 🔧 Consider alternative threshold strategies")
            self.stdout.write("4. ❌ Current policy may not be suitable for implementation")
        
    def _print_results(
        self,
        candidates: list[RecoveryCandidate],
        horizon: int,
        target: float,
        recovery_target: float,
        type_filter: Optional[str],
        flip_threshold: float,
        cut_threshold: float,
        show_threshold_impact: bool = False,
        simulate_policy: bool = False,
        sensitivity_analysis: bool = False,
    ):
        """Print analysis results."""
        if not candidates:
            self.stdout.write("No recovery candidates found (no trades adverse at checkpoint).")
            return

        # Convert to DataFrame for analysis
        df = pd.DataFrame([vars(c) for c in candidates])

        # === EXECUTIVE SUMMARY ===
        self.stdout.write("\n" + "=" * 110)
        self.stdout.write("EXECUTIVE SUMMARY - RECOVERY POLICY ANALYZER")
        self.stdout.write("=" * 110)
        
        # Key Metrics
        total_candidates = len(candidates)
        orig_hit_rate = df["original_hit"].mean() * 100
        recovery_hit_rate = df["recovery_hit"].mean() * 100
        net_edge = recovery_hit_rate - orig_hit_rate
        
        # Calculate recommendations for all candidates
        recommendations = []
        for _, row in df.iterrows():
            candidate = RecoveryCandidate(
                date=row["date"],
                trade_type=row["trade_type"],
                direction=row["direction"],
                entry_price=row["entry_price"],
                checkpoint_day=row["checkpoint_day"],
                checkpoint_price=row["checkpoint_price"],
                checkpoint_return=row["checkpoint_return"],
                remaining_days=row["remaining_days"],
                forward_return=row["forward_return"],
                original_hit=row["original_hit"],
                recovery_hit=row["recovery_hit"],
                recovery_return=row["recovery_return"],
                flip_threshold=flip_threshold,
                cut_threshold=cut_threshold,
            )
            recommendations.append(candidate.recommendation)
        
        df["recommendation"] = recommendations
        rec_counts = pd.Series(recommendations).value_counts()
        flip_count = rec_counts.get("FLIP", 0)
        hold_count = rec_counts.get("HOLD", 0)
        cut_count = rec_counts.get("CUT", 0)
        
        # Calculate expected performance with recovery policy
        flip_trades = df[df["recommendation"] == "FLIP"]
        hold_trades = df[df["recommendation"] == "HOLD"]
        cut_trades = df[df["recommendation"] == "CUT"]
        
        expected_wins = 0
        if len(flip_trades) > 0:
            expected_wins += flip_trades["recovery_hit"].sum()
        if len(hold_trades) > 0:
            expected_wins += hold_trades["original_hit"].sum()
        # CUT trades contribute 0 wins but avoid losses
        
        expected_win_rate = expected_wins / total_candidates * 100 if total_candidates > 0 else 0
        improvement_vs_hold_all = expected_win_rate - orig_hit_rate
        
        self.stdout.write("📊 KEY METRICS")
        self.stdout.write(f"   • Total Recovery Candidates: {total_candidates}")
        self.stdout.write(f"   • Net Edge (Recovery vs Hold): {net_edge:+.1f}%")
        self.stdout.write(f"   • Recovery Success Rate: {recovery_hit_rate:.1f}%")
        self.stdout.write(f"   • Late Recovery Rate: {orig_hit_rate:.1f}%")
        self.stdout.write("")
        
        self.stdout.write("🎯 RECOVERY POLICY RECOMMENDATIONS")
        self.stdout.write(f"   • FLIP Strategy: {flip_count} trades ({flip_count/total_candidates*100:.1f}%) - Strong recovery potential")
        self.stdout.write(f"   • HOLD Strategy: {hold_count} trades ({hold_count/total_candidates*100:.1f}%) - Moderate recovery potential")
        self.stdout.write(f"   • CUT Strategy: {cut_count} trades ({cut_count/total_candidates*100:.1f}%) - Weak recovery potential")
        self.stdout.write("")
        
        # High-level recommendation based on net edge
        if net_edge > 15:
            strategy_rec = "🟢 STRONG FLIP BIAS - Implement aggressive recovery policy"
        elif net_edge > 5:
            strategy_rec = "🟡 MODERATE FLIP BIAS - Implement selective recovery policy"
        elif net_edge > -5:
            strategy_rec = "⚪ NEUTRAL - Use three-way strategy (FLIP/HOLD/CUT)"
        else:
            strategy_rec = "🔴 HOLD BIAS - Recovery policy shows limited benefit"
        
        self.stdout.write("📈 STRATEGIC RECOMMENDATION")
        self.stdout.write(f"   • {strategy_rec}")
        self.stdout.write(f"   • Expected Win Rate with Recovery Policy: {expected_win_rate:.1f}%")
        self.stdout.write(f"   • Improvement vs Hold-All Strategy: {improvement_vs_hold_all:+.1f}%")
        self.stdout.write("")
        
        # Policy Enhancement Suggestions Preview
        policy_suggestions = self._generate_policy_suggestions(df, flip_threshold, cut_threshold)
        high_edge_signals = [s for s in policy_suggestions if s['edge'] > 20]
        
        self.stdout.write("⚙️ POLICY ENHANCEMENT SUGGESTIONS")
        if high_edge_signals:
            self.stdout.write("   • High Priority Signal Types (Edge > 20%):")
            for s in high_edge_signals[:3]:  # Show top 3
                self.stdout.write(f"     - {s['signal_type']}: {s['edge']:+.1f}% edge, {s['sample_size']} samples")
        
        # Calculate optimal thresholds
        optimal_results = self._analyze_optimal_thresholds(df)
        if optimal_results:
            best_result = optimal_results[0]
            current_performance = expected_win_rate
            optimal_performance = best_result['win_rate']
            threshold_improvement = optimal_performance - current_performance
            
            if threshold_improvement > 1:
                self.stdout.write(f"   • Optimal Thresholds: {best_result['cut_threshold']*100:.1f}% cut, {best_result['flip_threshold']*100:.1f}% flip")
                self.stdout.write(f"   • Additional Improvement Potential: {threshold_improvement:+.1f}%")
        
        self.stdout.write("")
        
        # Expected improvement from implementing recovery policy
        self.stdout.write("💡 EXPECTED IMPROVEMENT FROM RECOVERY POLICY")
        baseline_hold_all = orig_hit_rate
        baseline_flip_all = recovery_hit_rate
        
        if improvement_vs_hold_all > 5:
            impact_assessment = "🚀 SIGNIFICANT IMPACT - Strongly recommend implementation"
        elif improvement_vs_hold_all > 2:
            impact_assessment = "📈 MODERATE IMPACT - Recommend implementation"
        elif improvement_vs_hold_all > 0:
            impact_assessment = "📊 MINOR IMPACT - Consider implementation"
        else:
            impact_assessment = "⚠️ LIMITED IMPACT - Current strategy may be sufficient"
        
        self.stdout.write(f"   • Current Hold-All Strategy: {baseline_hold_all:.1f}% win rate")
        self.stdout.write(f"   • Recovery Policy Strategy: {expected_win_rate:.1f}% win rate")
        self.stdout.write(f"   • Net Improvement: {improvement_vs_hold_all:+.1f}%")
        self.stdout.write(f"   • Impact Assessment: {impact_assessment}")
        
        # Show configuration parameters
        self.stdout.write("")
        self.stdout.write("🔧 CURRENT CONFIGURATION")
        self.stdout.write(f"   • Horizon: {horizon}d | Original Target: {target*100:.1f}% | Recovery Target: {recovery_target*100:.1f}%")
        self.stdout.write(f"   • Flip Threshold: {flip_threshold*100:.1f}% | Cut Threshold: {cut_threshold*100:.1f}%")
        if type_filter:
            self.stdout.write(f"   • Type Filter: {type_filter}")
        
        self.stdout.write("\n" + "=" * 110)
        self.stdout.write("DETAILED ANALYSIS")
        self.stdout.write("=" * 110)

        # === SUMMARY BY TYPE ===
        self.stdout.write("-" * 110)
        self.stdout.write("SUMMARY BY TRADE TYPE")
        self.stdout.write("-" * 110)
        self.stdout.write(
            f"{'Type':<15} | {'N':>4} | {'Orig Hit%':>9} | {'Recov Hit%':>10} | "
            f"{'Avg Chkpt Ret':>13} | {'Avg Recov Ret':>13} | {'Edge':>6}"
        )
        self.stdout.write("-" * 110)

        for trade_type in sorted(df["trade_type"].unique()):
            type_df = df[df["trade_type"] == trade_type]
            n = len(type_df)
            orig_hit_pct = type_df["original_hit"].mean() * 100
            recov_hit_pct = type_df["recovery_hit"].mean() * 100
            avg_chkpt_ret = type_df["checkpoint_return"].mean() * 100
            avg_recov_ret = type_df["recovery_return"].mean() * 100
            # Edge = recovery hit rate - (100 - original hit rate for these candidates)
            # These are already filtered to adverse candidates, so original hit rate here
            # is the "late recovery" rate for originally failing trades
            edge = recov_hit_pct - orig_hit_pct

            self.stdout.write(
                f"{trade_type:<15} | {n:>4} | {orig_hit_pct:>8.1f}% | {recov_hit_pct:>9.1f}% | "
                f"{avg_chkpt_ret:>+12.1f}% | {avg_recov_ret:>+12.1f}% | {edge:>+5.1f}%"
            )

        # === OVERALL STATS ===
        self.stdout.write("\n" + "-" * 110)
        self.stdout.write("OVERALL STATISTICS")
        self.stdout.write("-" * 110)

        total_n = len(df)
        orig_eventually_hit = df["original_hit"].sum()
        recov_would_hit = df["recovery_hit"].sum()

        self.stdout.write(f"Trades adverse at checkpoint: {total_n}")
        self.stdout.write(f"Original trades that eventually hit anyway: {orig_eventually_hit} ({orig_eventually_hit/total_n*100:.1f}%)")
        self.stdout.write(f"Recovery trades that would hit: {recov_would_hit} ({recov_would_hit/total_n*100:.1f}%)")
        self.stdout.write("")
        self.stdout.write(f"Average checkpoint return: {df['checkpoint_return'].mean()*100:+.2f}%")
        self.stdout.write(f"Average recovery MFE: {df['recovery_return'].mean()*100:+.2f}%")
        self.stdout.write(f"Median recovery MFE: {df['recovery_return'].median()*100:+.2f}%")
        self.stdout.write(f"75th pct recovery MFE: {df['recovery_return'].quantile(0.75)*100:+.2f}%")

        # === RECOMMENDATION DISTRIBUTION ===
        self.stdout.write("\n" + "-" * 110)
        self.stdout.write("RECOMMENDATION DISTRIBUTION")
        self.stdout.write("-" * 110)
        
        # Calculate recommendations for all candidates
        recommendations = []
        for _, row in df.iterrows():
            candidate = RecoveryCandidate(
                date=row["date"],
                trade_type=row["trade_type"],
                direction=row["direction"],
                entry_price=row["entry_price"],
                checkpoint_day=row["checkpoint_day"],
                checkpoint_price=row["checkpoint_price"],
                checkpoint_return=row["checkpoint_return"],
                remaining_days=row["remaining_days"],
                forward_return=row["forward_return"],
                original_hit=row["original_hit"],
                recovery_hit=row["recovery_hit"],
                recovery_return=row["recovery_return"],
                flip_threshold=flip_threshold,
                cut_threshold=cut_threshold,
            )
            recommendations.append(candidate.recommendation)
        
        rec_counts = pd.Series(recommendations).value_counts()
        for rec in ["FLIP", "HOLD", "CUT"]:
            count = rec_counts.get(rec, 0)
            pct = count / total_n * 100 if total_n > 0 else 0
            self.stdout.write(f"{rec}: {count:>4} trades ({pct:>5.1f}%)")
        
        self.stdout.write("")
        self.stdout.write("Recommendation Logic:")
        self.stdout.write(f"  FLIP: Recovery MFE > {flip_threshold*100:.0f}% (strong recovery potential)")
        self.stdout.write(f"  HOLD: Recovery MFE {cut_threshold*100:.0f}-{flip_threshold*100:.0f}% (moderate recovery potential)")
        self.stdout.write(f"  CUT:  Recovery MFE < {cut_threshold*100:.0f}% (weak recovery potential, cut losses)")

        # === DECISION MATRIX ===
        self.stdout.write("\n" + "-" * 110)
        self.stdout.write("THREE-WAY DECISION MATRIX: FLIP vs HOLD vs CUT Analysis")
        self.stdout.write("-" * 110)

        # Calculate recommendations for all candidates
        recommendations = []
        for _, row in df.iterrows():
            candidate = RecoveryCandidate(
                date=row["date"],
                trade_type=row["trade_type"],
                direction=row["direction"],
                entry_price=row["entry_price"],
                checkpoint_day=row["checkpoint_day"],
                checkpoint_price=row["checkpoint_price"],
                checkpoint_return=row["checkpoint_return"],
                remaining_days=row["remaining_days"],
                forward_return=row["forward_return"],
                original_hit=row["original_hit"],
                recovery_hit=row["recovery_hit"],
                recovery_return=row["recovery_return"],
                flip_threshold=flip_threshold,
                cut_threshold=cut_threshold,
            )
            recommendations.append(candidate.recommendation)
        
        df["recommendation"] = recommendations
        
        # Analyze success rates by recommendation
        flip_trades = df[df["recommendation"] == "FLIP"]
        hold_trades = df[df["recommendation"] == "HOLD"] 
        cut_trades = df[df["recommendation"] == "CUT"]
        
        self.stdout.write("Strategy Performance Analysis:")
        self.stdout.write("-" * 60)
        
        if len(flip_trades) > 0:
            flip_success = flip_trades["recovery_hit"].mean() * 100
            flip_orig_success = flip_trades["original_hit"].mean() * 100
            self.stdout.write(f"FLIP Strategy ({len(flip_trades)} trades, Recovery MFE > {flip_threshold*100:.0f}%):")
            self.stdout.write(f"  Recovery success rate: {flip_success:>5.1f}% (if flipped)")
            self.stdout.write(f"  Original success rate: {flip_orig_success:>5.1f}% (if held)")
            self.stdout.write(f"  Net advantage:         {flip_success - flip_orig_success:>+5.1f}%")
        else:
            self.stdout.write(f"FLIP Strategy: No trades qualify (Recovery MFE > {flip_threshold*100:.0f}%)")
            
        if len(hold_trades) > 0:
            hold_success = hold_trades["original_hit"].mean() * 100
            hold_recov_success = hold_trades["recovery_hit"].mean() * 100
            self.stdout.write(f"HOLD Strategy ({len(hold_trades)} trades, Recovery MFE {cut_threshold*100:.0f}-{flip_threshold*100:.0f}%):")
            self.stdout.write(f"  Original success rate: {hold_success:>5.1f}% (if held)")
            self.stdout.write(f"  Recovery success rate: {hold_recov_success:>5.1f}% (if flipped)")
            self.stdout.write(f"  Net advantage:         {hold_success - hold_recov_success:>+5.1f}%")
        else:
            self.stdout.write(f"HOLD Strategy: No trades qualify (Recovery MFE {cut_threshold*100:.0f}-{flip_threshold*100:.0f}%)")
            
        if len(cut_trades) > 0:
            cut_orig_loss = (1 - cut_trades["original_hit"].mean()) * 100
            cut_recov_loss = (1 - cut_trades["recovery_hit"].mean()) * 100
            self.stdout.write(f"CUT Strategy ({len(cut_trades)} trades, Recovery MFE < {cut_threshold*100:.0f}%):")
            self.stdout.write(f"  Original loss rate:    {cut_orig_loss:>5.1f}% (if held)")
            self.stdout.write(f"  Recovery loss rate:    {cut_recov_loss:>5.1f}% (if flipped)")
            self.stdout.write(f"  Loss avoidance:        {min(cut_orig_loss, cut_recov_loss):>5.1f}% (cut early)")
        else:
            self.stdout.write(f"CUT Strategy: No trades qualify (Recovery MFE < {cut_threshold*100:.0f}%)")

        self.stdout.write("")
        self.stdout.write("Traditional 2x2 Matrix (for reference):")
        self.stdout.write("-" * 60)
        
        # 4 outcomes:
        # 1. Original eventually hits, recovery would also hit (both win)
        # 2. Original eventually hits, recovery would miss (should have held)
        # 3. Original misses, recovery would hit (should have flipped)
        # 4. Original misses, recovery would miss (both lose)

        both_hit = ((df["original_hit"]) & (df["recovery_hit"])).sum()
        orig_only = ((df["original_hit"]) & (~df["recovery_hit"])).sum()
        recov_only = ((~df["original_hit"]) & (df["recovery_hit"])).sum()
        both_miss = ((~df["original_hit"]) & (~df["recovery_hit"])).sum()

        self.stdout.write(f"Both original and recovery hit:     {both_hit:>4} ({both_hit/total_n*100:>5.1f}%) - Either strategy works")
        self.stdout.write(f"Only original hits (late recovery): {orig_only:>4} ({orig_only/total_n*100:>5.1f}%) - Should have held")
        self.stdout.write(f"Only recovery hits (flip wins):     {recov_only:>4} ({recov_only/total_n*100:>5.1f}%) - Should have flipped")
        self.stdout.write(f"Both miss:                          {both_miss:>4} ({both_miss/total_n*100:>5.1f}%) - Cut losses early")

        # === RECOMMENDATION ===
        self.stdout.write("\n" + "-" * 110)
        self.stdout.write("THREE-WAY RECOMMENDATION ANALYSIS")
        self.stdout.write("-" * 110)

        # Calculate strategy distribution
        rec_counts = pd.Series(recommendations).value_counts()
        flip_count = rec_counts.get("FLIP", 0)
        hold_count = rec_counts.get("HOLD", 0)
        cut_count = rec_counts.get("CUT", 0)
        
        self.stdout.write("Optimal Strategy Distribution:")
        self.stdout.write(f"  FLIP: {flip_count:>3} trades ({flip_count/total_n*100:>5.1f}%) - Strong recovery potential (MFE > 5%)")
        self.stdout.write(f"  HOLD: {hold_count:>3} trades ({hold_count/total_n*100:>5.1f}%) - Moderate recovery potential (MFE 2-5%)")
        self.stdout.write(f"  CUT:  {cut_count:>3} trades ({cut_count/total_n*100:>5.1f}%) - Weak recovery potential (MFE < 2%)")
        self.stdout.write("")
        
        # Calculate expected outcomes if following recommendations
        expected_wins = 0
        total_decisions = 0
        
        if len(flip_trades) > 0:
            flip_wins = flip_trades["recovery_hit"].sum()
            expected_wins += flip_wins
            total_decisions += len(flip_trades)
            
        if len(hold_trades) > 0:
            hold_wins = hold_trades["original_hit"].sum()
            expected_wins += hold_wins
            total_decisions += len(hold_trades)
            
        # For CUT trades, we assume cutting losses early (no wins, but also no further losses)
        total_decisions += len(cut_trades)
        
        if total_decisions > 0:
            expected_win_rate = expected_wins / total_decisions * 100
            self.stdout.write(f"Expected Win Rate Following Recommendations: {expected_win_rate:.1f}%")
        
        # Compare to baseline strategies
        baseline_hold_all = df["original_hit"].mean() * 100
        baseline_flip_all = df["recovery_hit"].mean() * 100
        
        self.stdout.write(f"Baseline Hold All Strategy:          {baseline_hold_all:.1f}%")
        self.stdout.write(f"Baseline Flip All Strategy:          {baseline_flip_all:.1f}%")
        
        if total_decisions > 0:
            improvement_vs_hold = expected_win_rate - baseline_hold_all
            improvement_vs_flip = expected_win_rate - baseline_flip_all
            self.stdout.write(f"Improvement vs Hold All:             {improvement_vs_hold:+.1f}%")
            self.stdout.write(f"Improvement vs Flip All:             {improvement_vs_flip:+.1f}%")

        self.stdout.write("")
        self.stdout.write("Key Insights:")
        
        # Traditional flip vs hold edge for comparison
        flip_advantage = recov_only / total_n * 100
        hold_advantage = orig_only / total_n * 100
        net_edge = flip_advantage - hold_advantage

        if net_edge > 5:
            self.stdout.write(f"• Traditional flip strategy has +{net_edge:.1f}% edge over holding")
        elif net_edge < -5:
            self.stdout.write(f"• Traditional hold strategy has +{-net_edge:.1f}% edge over flipping")
        else:
            self.stdout.write(f"• Traditional flip vs hold is neutral ({net_edge:+.1f}% edge)")
            
        if cut_count > 0:
            cut_loss_avoidance = both_miss / total_n * 100
            self.stdout.write(f"• Cut losses strategy could avoid {cut_loss_avoidance:.1f}% of total losing trades")
            
        # === ENHANCED MFE DISTRIBUTION ANALYSIS ===
        self.stdout.write("\n" + "-" * 110)
        self.stdout.write("RECOVERY MFE DISTRIBUTION ANALYSIS")
        self.stdout.write("-" * 110)
        
        recovery_mfe_values = df["recovery_return"].values
        
        # Detailed percentile analysis
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        self.stdout.write("Recovery MFE Percentiles:")
        for p in percentiles:
            value = np.percentile(recovery_mfe_values, p) * 100
            self.stdout.write(f"  {p:>2}th percentile: {value:>6.2f}%")
        
        # Quartile analysis
        q1 = np.percentile(recovery_mfe_values, 25)
        q2 = np.percentile(recovery_mfe_values, 50)  # median
        q3 = np.percentile(recovery_mfe_values, 75)
        iqr = q3 - q1
        
        self.stdout.write("")
        self.stdout.write("Quartile Analysis:")
        self.stdout.write(f"  Q1 (25th percentile): {q1*100:>6.2f}%")
        self.stdout.write(f"  Q2 (median):          {q2*100:>6.2f}%")
        self.stdout.write(f"  Q3 (75th percentile): {q3*100:>6.2f}%")
        self.stdout.write(f"  IQR (Q3 - Q1):       {iqr*100:>6.2f}%")
        
        # Additional statistics
        mean_mfe = np.mean(recovery_mfe_values) * 100
        std_mfe = np.std(recovery_mfe_values) * 100
        min_mfe = np.min(recovery_mfe_values) * 100
        max_mfe = np.max(recovery_mfe_values) * 100
        
        self.stdout.write("")
        self.stdout.write("Additional Statistics:")
        self.stdout.write(f"  Mean:                 {mean_mfe:>6.2f}%")
        self.stdout.write(f"  Standard Deviation:   {std_mfe:>6.2f}%")
        self.stdout.write(f"  Minimum:              {min_mfe:>6.2f}%")
        self.stdout.write(f"  Maximum:              {max_mfe:>6.2f}%")
        
        # === OPTIMAL THRESHOLD IDENTIFICATION ===
        self.stdout.write("\n" + "-" * 110)
        self.stdout.write("OPTIMAL THRESHOLD ANALYSIS")
        self.stdout.write("-" * 110)
        
        # Test different threshold combinations to find optimal performance
        optimal_results = self._analyze_optimal_thresholds(df)
        
        self.stdout.write("Threshold Performance Analysis:")
        self.stdout.write(f"{'Cut Thresh':>10} | {'Flip Thresh':>11} | {'Win Rate':>8} | {'FLIP':>4} | {'HOLD':>4} | {'CUT':>3} | {'Score':>5}")
        self.stdout.write("-" * 70)
        
        for result in optimal_results[:10]:  # Show top 10 combinations
            self.stdout.write(
                f"{result['cut_threshold']*100:>9.1f}% | {result['flip_threshold']*100:>10.1f}% | "
                f"{result['win_rate']:>7.1f}% | {result['flip_count']:>4} | {result['hold_count']:>4} | "
                f"{result['cut_count']:>3} | {result['score']:>5.2f}"
            )
        
        # Recommend optimal thresholds
        best_result = optimal_results[0]
        self.stdout.write("")
        self.stdout.write("THRESHOLD RECOMMENDATIONS:")
        self.stdout.write(f"  Optimal Cut Threshold:  {best_result['cut_threshold']*100:>5.1f}% (vs current {cut_threshold*100:.1f}%)")
        self.stdout.write(f"  Optimal Flip Threshold: {best_result['flip_threshold']*100:>5.1f}% (vs current {flip_threshold*100:.1f}%)")
        self.stdout.write(f"  Expected Win Rate:      {best_result['win_rate']:>5.1f}% (vs current {expected_win_rate:.1f}%)")
        self.stdout.write(f"  Performance Score:      {best_result['score']:>5.2f}")
        
        # Explain the scoring methodology
        self.stdout.write("")
        self.stdout.write("Scoring Methodology:")
        self.stdout.write("  Score = Win Rate + (0.5 * Cut Rate) - (0.1 * |Flip Rate - 40%|)")
        self.stdout.write("  • Rewards higher win rates")
        self.stdout.write("  • Rewards cutting losing trades early")
        self.stdout.write("  • Penalizes extreme flip rates (prefers ~40% flip rate)")
        
        # Data-driven insights
        self.stdout.write("")
        self.stdout.write("Data-Driven Insights:")
        
        # Analyze success rates by MFE ranges
        mfe_ranges = [
            (0.00, 0.01, "0-1%"),
            (0.01, 0.02, "1-2%"), 
            (0.02, 0.03, "2-3%"),
            (0.03, 0.05, "3-5%"),
            (0.05, 0.10, "5-10%"),
            (0.10, float('inf'), ">10%")
        ]
        
        for min_mfe, max_mfe, label in mfe_ranges:
            range_df = df[(df["recovery_return"] >= min_mfe) & (df["recovery_return"] < max_mfe)]
            if len(range_df) > 0:
                recovery_success = range_df["recovery_hit"].mean() * 100
                original_success = range_df["original_hit"].mean() * 100
                count = len(range_df)
                self.stdout.write(f"  MFE {label:>6}: {count:>3} trades, Recovery: {recovery_success:>5.1f}%, Original: {original_success:>5.1f}%")
        
        self.stdout.write(f"  Current thresholds: CUT < {cut_threshold*100:.0f}%, HOLD {cut_threshold*100:.0f}-{flip_threshold*100:.0f}%, FLIP > {flip_threshold*100:.0f}%")

        # === RECOVERY STRATEGY COMPARISON ===
        self.stdout.write("\n" + "-" * 110)
        self.stdout.write("RECOVERY STRATEGY COMPARISON")
        self.stdout.write("-" * 110)
        
        self._print_recovery_strategy_comparison(df, flip_threshold, cut_threshold, target, recovery_target)

        # === POLICY ENHANCEMENT SUGGESTIONS ===
        self.stdout.write("\n" + "-" * 110)
        self.stdout.write("POLICY ENHANCEMENT SUGGESTIONS")
        self.stdout.write("-" * 110)
        
        policy_suggestions = self._generate_policy_suggestions(df, flip_threshold, cut_threshold)
        
        self.stdout.write("Signal Type Analysis for Policy Enhancement:")
        self.stdout.write("-" * 70)
        self.stdout.write(f"{'Signal Type':<15} | {'Edge':>6} | {'Flip Thresh':>11} | {'Cut Thresh':>10} | {'Recovery Target':>15}")
        self.stdout.write("-" * 70)
        
        for suggestion in policy_suggestions:
            self.stdout.write(
                f"{suggestion['signal_type']:<15} | {suggestion['edge']:>+5.1f}% | "
                f"{suggestion['flip_threshold']*100:>10.1f}% | {suggestion['cut_threshold']*100:>9.1f}% | "
                f"{suggestion['recovery_target']*100:>14.1f}%"
            )
        
        self.stdout.write("")
        self.stdout.write("Policy Parameter Recommendations:")
        
        # Calculate overall optimal thresholds
        overall_optimal = optimal_results[0]
        self.stdout.write(f"  recovery_flip_threshold: {overall_optimal['flip_threshold']:.3f}  # {overall_optimal['flip_threshold']*100:.1f}% - Strong recovery potential")
        self.stdout.write(f"  recovery_cut_threshold:  {overall_optimal['cut_threshold']:.3f}  # {overall_optimal['cut_threshold']*100:.1f}% - Weak recovery potential")
        
        # Signal-specific recovery targets based on analysis
        self.stdout.write("")
        self.stdout.write("Signal-Specific Recovery Targets:")
        for suggestion in policy_suggestions:
            if suggestion['sample_size'] >= 3:  # Only suggest for signals with sufficient data
                self.stdout.write(f"  {suggestion['signal_type']}: {suggestion['recovery_target']:.3f}  # {suggestion['recovery_target']*100:.1f}% target")
        
        self.stdout.write("")
        self.stdout.write("Implementation Notes:")
        self.stdout.write("  • Add these parameters to execution/services/policy.py RecoveryConfig")
        self.stdout.write("  • Use signal-specific thresholds for better performance")
        self.stdout.write("  • Consider implementing adaptive thresholds based on market conditions")
        
        # Expected improvement calculation
        current_performance = expected_win_rate if total_decisions > 0 else baseline_hold_all
        optimal_performance = overall_optimal['win_rate']
        improvement = optimal_performance - current_performance
        
        self.stdout.write("")
        self.stdout.write("Expected Performance Improvement:")
        self.stdout.write(f"  Current Strategy Win Rate:  {current_performance:>5.1f}%")
        self.stdout.write(f"  Optimal Strategy Win Rate:  {optimal_performance:>5.1f}%")
        self.stdout.write(f"  Expected Improvement:       {improvement:>+5.1f}%")
        
        if improvement > 2:
            self.stdout.write("  ✅ Significant improvement potential - recommend implementing recovery policy")
        elif improvement > 0:
            self.stdout.write("  ⚠️  Moderate improvement potential - consider implementing recovery policy")
        else:
            self.stdout.write("  ❌ Limited improvement potential - current strategy may be sufficient")

        # === POLICY ENHANCEMENT SUGGESTIONS ===
        self.stdout.write("\n" + "-" * 110)
        self.stdout.write("POLICY ENHANCEMENT SUGGESTIONS")
        self.stdout.write("-" * 110)
        
        policy_suggestions = self._generate_policy_suggestions(df, flip_threshold, cut_threshold)
        
        self.stdout.write("Signal Type Analysis & Recommendations:")
        self.stdout.write("")
        self.stdout.write(f"{'Signal Type':<15} | {'N':>3} | {'Edge':>6} | {'Flip Thresh':>11} | {'Cut Thresh':>10} | {'Recovery Target':>14}")
        self.stdout.write("-" * 80)
        
        for suggestion in policy_suggestions:
            self.stdout.write(
                f"{suggestion['signal_type']:<15} | {suggestion['sample_size']:>3} | "
                f"{suggestion['edge']:>+5.1f}% | {suggestion['flip_threshold']*100:>10.1f}% | "
                f"{suggestion['cut_threshold']*100:>9.1f}% | {suggestion['recovery_target']*100:>13.1f}%"
            )
        
        self.stdout.write("")
        self.stdout.write("Policy Configuration Recommendations:")
        self.stdout.write("")
        
        # Identify signal types with highest flip edge
        high_edge_signals = [s for s in policy_suggestions if s['edge'] > 20]
        medium_edge_signals = [s for s in policy_suggestions if 10 <= s['edge'] <= 20]
        low_edge_signals = [s for s in policy_suggestions if s['edge'] < 10]
        
        if high_edge_signals:
            self.stdout.write("🟢 HIGH PRIORITY - Strong Recovery Potential (Edge > 20%):")
            for s in high_edge_signals:
                self.stdout.write(f"  • {s['signal_type']}: {s['edge']:+.1f}% edge, recommend aggressive recovery policy")
                self.stdout.write(f"    - recovery_flip_threshold: {s['flip_threshold']:.3f}")
                self.stdout.write(f"    - recovery_cut_threshold: {s['cut_threshold']:.3f}")
                self.stdout.write(f"    - recovery_target: {s['recovery_target']:.3f}")
        
        if medium_edge_signals:
            self.stdout.write("")
            self.stdout.write("🟡 MEDIUM PRIORITY - Moderate Recovery Potential (Edge 10-20%):")
            for s in medium_edge_signals:
                self.stdout.write(f"  • {s['signal_type']}: {s['edge']:+.1f}% edge, recommend standard recovery policy")
                self.stdout.write(f"    - recovery_flip_threshold: {s['flip_threshold']:.3f}")
                self.stdout.write(f"    - recovery_cut_threshold: {s['cut_threshold']:.3f}")
                self.stdout.write(f"    - recovery_target: {s['recovery_target']:.3f}")
        
        if low_edge_signals:
            self.stdout.write("")
            self.stdout.write("🔴 LOW PRIORITY - Limited Recovery Potential (Edge < 10%):")
            for s in low_edge_signals:
                self.stdout.write(f"  • {s['signal_type']}: {s['edge']:+.1f}% edge, consider conservative recovery policy")
                self.stdout.write(f"    - recovery_flip_threshold: {s['flip_threshold']:.3f}")
                self.stdout.write(f"    - recovery_cut_threshold: {s['cut_threshold']:.3f}")
                self.stdout.write(f"    - recovery_target: {s['recovery_target']:.3f}")
        
        # Overall policy recommendations
        self.stdout.write("")
        self.stdout.write("Overall Policy Enhancement Recommendations:")
        self.stdout.write("")
        
        # Calculate weighted averages based on sample sizes
        total_samples = sum(s['sample_size'] for s in policy_suggestions)
        if total_samples > 0:
            weighted_flip_thresh = sum(s['flip_threshold'] * s['sample_size'] for s in policy_suggestions) / total_samples
            weighted_cut_thresh = sum(s['cut_threshold'] * s['sample_size'] for s in policy_suggestions) / total_samples
            weighted_recovery_target = sum(s['recovery_target'] * s['sample_size'] for s in policy_suggestions) / total_samples
            
            self.stdout.write("1. Global Recovery Policy Parameters:")
            self.stdout.write(f"   - recovery_flip_threshold: {weighted_flip_thresh:.3f} ({weighted_flip_thresh*100:.1f}%)")
            self.stdout.write(f"   - recovery_cut_threshold: {weighted_cut_thresh:.3f} ({weighted_cut_thresh*100:.1f}%)")
            self.stdout.write(f"   - recovery_target: {weighted_recovery_target:.3f} ({weighted_recovery_target*100:.1f}%)")
        
        # Signal-specific recommendations
        self.stdout.write("")
        self.stdout.write("2. Signal-Specific Recovery Targets:")
        for s in policy_suggestions:
            if s['sample_size'] >= 5:  # Only recommend for signals with sufficient data
                self.stdout.write(f"   - {s['signal_type']}: {s['recovery_target']*100:.1f}% target (based on {s['sample_size']} samples)")
        
        # Implementation guidance
        self.stdout.write("")
        self.stdout.write("3. Implementation Guidance:")
        
        # Calculate overall improvement potential
        current_baseline = df["original_hit"].mean() * 100
        flip_all_baseline = df["recovery_hit"].mean() * 100
        
        # Calculate expected performance with recommendations
        expected_wins = 0
        total_decisions = 0
        
        for suggestion in policy_suggestions:
            signal_df = df[df["trade_type"] == suggestion['signal_type']]
            if len(signal_df) == 0:
                continue
                
            flip_thresh = suggestion['flip_threshold']
            cut_thresh = suggestion['cut_threshold']
            
            flip_trades = signal_df[signal_df["recovery_return"] > flip_thresh]
            hold_trades = signal_df[(signal_df["recovery_return"] >= cut_thresh) & (signal_df["recovery_return"] <= flip_thresh)]
            
            expected_wins += flip_trades["recovery_hit"].sum()
            expected_wins += hold_trades["original_hit"].sum()
            total_decisions += len(signal_df)
        
        if total_decisions > 0:
            expected_performance = expected_wins / total_decisions * 100
            improvement = expected_performance - current_baseline
            
            self.stdout.write(f"   - Expected improvement: {improvement:+.1f}% vs current hold-all strategy")
            self.stdout.write(f"   - Expected performance: {expected_performance:.1f}% win rate")
            
            if improvement > 5:
                self.stdout.write("   - ✅ RECOMMENDED: Implement recovery policy immediately")
            elif improvement > 2:
                self.stdout.write("   - ⚠️  CONSIDER: Recovery policy shows moderate improvement")
            else:
                self.stdout.write("   - ❌ CAUTION: Limited improvement potential")
        
        # Risk management recommendations
        self.stdout.write("")
        self.stdout.write("4. Risk Management Considerations:")
        
        # Calculate volatility of recovery returns by signal type
        high_volatility_signals = []
        for s in policy_suggestions:
            signal_df = df[df["trade_type"] == s['signal_type']]
            if len(signal_df) >= 3:
                recovery_std = signal_df["recovery_return"].std()
                if recovery_std > 0.05:  # > 5% standard deviation
                    high_volatility_signals.append((s['signal_type'], recovery_std))
        
        if high_volatility_signals:
            self.stdout.write("   - High volatility signals (consider conservative thresholds):")
            for signal_type, std in high_volatility_signals:
                self.stdout.write(f"     • {signal_type}: {std*100:.1f}% recovery MFE standard deviation")
        
        # Identify signals with low sample sizes
        low_sample_signals = [s for s in policy_suggestions if s['sample_size'] < 5]
        if low_sample_signals:
            self.stdout.write("   - Limited data signals (use global thresholds):")
            for s in low_sample_signals:
                self.stdout.write(f"     • {s['signal_type']}: Only {s['sample_size']} samples")

        # === THRESHOLD IMPACT ANALYSIS ===
        if show_threshold_impact:
            self.stdout.write("\n" + "-" * 110)
            self.stdout.write("THRESHOLD IMPACT ANALYSIS")
            self.stdout.write("-" * 110)
            
            self._show_threshold_impact_analysis(df, flip_threshold, cut_threshold)

        # === POLICY BACKTESTING SIMULATION ===
        if simulate_policy:
            simulation_results = self._simulate_policy_backtesting(
                candidates, horizon, target, recovery_target, flip_threshold, cut_threshold
            )
            self._print_policy_simulation_results(simulation_results)

        # === SENSITIVITY ANALYSIS ===
        if sensitivity_analysis:
            self.stdout.write("\n" + "=" * 120)
            self.stdout.write("SENSITIVITY ANALYSIS")
            self.stdout.write("=" * 120)
            self.stdout.write("📊 Sensitivity analysis results will be displayed after the main analysis.")
            self.stdout.write("Testing recovery thresholds: 1%, 2%, 3%, 5%, 10%")
            self.stdout.write("")

        # === DETAILED TRADE LIST ===
        self.stdout.write("\n" + "-" * 120)
        self.stdout.write("DETAILED TRADE LIST (sorted by recovery potential)")
        self.stdout.write("-" * 120)
        self.stdout.write(
            f"{'Date':<12} | {'Type':<15} | {'Dir':<5} | {'Chkpt':>5} | "
            f"{'Chkpt Ret':>9} | {'Recov MFE':>9} | {'Orig Hit':>8} | {'Recov Hit':>9} | {'Rec':>4}"
        )
        self.stdout.write("-" * 120)

        # Sort by recovery return descending
        df_sorted = df.sort_values("recovery_return", ascending=False)

        for _, row in df_sorted.iterrows():
            orig_emoji = "✅" if row["original_hit"] else "❌"
            recov_emoji = "✅" if row["recovery_hit"] else "❌"
            dir_emoji = "🟢" if row["direction"] == "LONG" else "🔴"
            
            # Create RecoveryCandidate to get recommendation
            candidate = RecoveryCandidate(
                date=row["date"],
                trade_type=row["trade_type"],
                direction=row["direction"],
                entry_price=row["entry_price"],
                checkpoint_day=row["checkpoint_day"],
                checkpoint_price=row["checkpoint_price"],
                checkpoint_return=row["checkpoint_return"],
                remaining_days=row["remaining_days"],
                forward_return=row["forward_return"],
                original_hit=row["original_hit"],
                recovery_hit=row["recovery_hit"],
                recovery_return=row["recovery_return"],
                flip_threshold=flip_threshold,
                cut_threshold=cut_threshold,
            )

            self.stdout.write(
                f"{row['date']:<12} | {row['trade_type']:<15} | {dir_emoji} {row['direction']:<3} | "
                f"D{row['checkpoint_day']:>3}  | {row['checkpoint_return']*100:>+8.2f}% | "
                f"{row['recovery_return']*100:>+8.2f}% | {orig_emoji:>7} | {recov_emoji:>8} | {candidate.recommendation:>4}"
            )

        self.stdout.write("\n" + "=" * 120)
