"""
Deribit Entry Engine V2 - Policy-driven with IV-aware selection.

Improvements over V1:
- Uses versioned PolicyVersion for all parameters
- IV-aware scoring via InstrumentSelector
- Spread construction for all signal types (including MVRV_SHORT)
- Execution cost estimation in plan
- Uses signal date spot price for historical analysis

Usage:
    from execution.services.deribit_entry_v2 import DeribitEntryEngineV2
    from execution.services.policy import get_policy
    
    engine = DeribitEntryEngineV2()
    plan = engine.plan_entry(signal, account_size_usd=100_000)
"""
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional

from datafeed.models import OptionSnapshot, RawDailyData
from signals.models import DailySignal
from execution.services.policy import get_policy, PolicyVersion
from execution.services.instrument_selector import InstrumentSelector, ScoredCandidate
from execution.services.deribit_entry import LegType, LegPlan, EntryPlan

logger = logging.getLogger(__name__)


class DeribitEntryEngineV2:
    """
    Policy-driven entry engine with IV-aware instrument selection.
    
    Key improvements:
    1. All parameters from versioned policy
    2. IV scoring is side-aware (buy vs sell legs)
    3. Spreads enabled for all signals including MVRV_SHORT
    4. Uses signal date spot for historical analysis
    """

    def __init__(self, policy: Optional[PolicyVersion] = None):
        self.policy = policy or get_policy()
        self.selector = InstrumentSelector(policy=self.policy)

    def plan_entry(
        self,
        signal: DailySignal,
        account_size_usd: float = 100_000,
        spot_price: Optional[float] = None,
    ) -> Optional[EntryPlan]:
        """
        Build an EntryPlan from a DailySignal.
        
        Args:
            signal: The DailySignal to execute
            account_size_usd: Account size for tier sizing
            spot_price: Override spot price (uses signal date if None)
        
        Returns:
            EntryPlan or None if no suitable instruments found.
        """
        decision = signal.trade_decision.upper()

        if decision == "NO_TRADE":
            return None

        if decision == "IRON_CONDOR":
            return self._plan_condor(signal, account_size_usd, spot_price)

        return self._plan_directional(signal, account_size_usd, spot_price)

    def _resolve_spot(self, signal: DailySignal, override: Optional[float]) -> Optional[float]:
        """
        Resolve spot price, preferring signal date data for historical analysis.
        """
        if override is not None:
            return override
        
        # Try to get spot from signal date (for historical accuracy)
        try:
            raw = RawDailyData.objects.get(date=signal.date)
            if raw.btc_close:
                return float(raw.btc_close)
        except RawDailyData.DoesNotExist:
            pass
        
        # Fallback to latest snapshot
        return self.selector.get_latest_spot()

    def _plan_directional(
        self,
        signal: DailySignal,
        account_size_usd: float,
        spot_price: Optional[float],
    ) -> Optional[EntryPlan]:
        """Plan directional trade (calls, puts, spreads)."""
        decision = signal.trade_decision.upper()

        # Map decision -> direction + option_type
        decision_map = {
            "CALL":         ("long",  "call"),
            "OPTION_CALL":  ("long",  "call"),
            "PUT":          ("short", "put"),
            "OPTION_PUT":   ("short", "put"),
            "TACTICAL_PUT": ("long",  "put"),
            "BULL_PROBE":   ("long",  "call"),
            "BEAR_PROBE":   ("short", "put"),
            "MVRV_SHORT":   ("short", "put"),
        }
        if decision not in decision_map:
            logger.warning(f"Unknown decision: {decision}")
            return None

        direction, option_type = decision_map[decision]

        # Resolve spot
        spot = self._resolve_spot(signal, spot_price)
        if spot is None:
            logger.error("Cannot resolve spot price")
            return None

        # Get policy parameters
        dte_cfg = self.policy.get_dte_target(decision)
        strike_guidance = signal.strike_guidance or "slight_itm"
        target_delta = self.policy.get_delta_target(strike_guidance, option_type)
        abs_target_delta = abs(target_delta)

        # Find best contract for LONG leg (always buy the primary option)
        candidates = self.selector.find_candidates(
            option_type=option_type,
            side="buy",  # Primary leg is always bought
            dte_min=dte_cfg.min_dte,
            dte_max=dte_cfg.max_dte,
            staleness_hours=self.policy.snapshot_staleness_hours,
        )
        if not candidates:
            logger.warning(f"No {option_type} candidates in DTE {dte_cfg.min_dte}-{dte_cfg.max_dte}")
            return None

        # Rank with IV-aware scoring (side="buy" for long leg)
        best = self.selector.rank_candidates(
            candidates,
            target_delta=abs_target_delta,
            optimal_dte=dte_cfg.optimal_dte,
            side="buy",
        )
        if best is None:
            return None

        legs: list[LegPlan] = []
        warnings: list[str] = []

        # Check if spread is enabled for this signal
        spread_enabled = self.policy.is_spread_enabled(decision)
        spread_width_pct = self.policy.get_spread_width(decision)
        has_spread = spread_enabled and spread_width_pct > 0

        # Primary (long) leg
        primary_leg = self._snapshot_to_leg(
            best.snapshot,
            side="buy",
            leg_type=LegType.SINGLE if not has_spread else LegType.LONG_LEG,
        )
        legs.append(primary_leg)

        # Spread short leg (if enabled)
        if has_spread:
            short_leg = self._find_spread_short_leg(
                best.snapshot, option_type, direction, spread_width_pct, spot
            )
            if short_leg:
                legs.append(short_leg)
            else:
                warnings.append("Could not find spread short leg; falling back to naked only")

        # Sizing from policy
        tier = self.policy.get_tier(decision)
        size_mult = signal.effective_size or signal.size_multiplier or 1.0
        tier_risk = tier.risk_usd * size_mult

        # Cap at max account risk
        max_risk = account_size_usd * self.policy.max_account_risk_pct
        tier_risk = min(tier_risk, max_risk)

        # IV rank
        iv_rank = self.selector.compute_iv_rank(option_type)

        # Estimate execution costs
        num_legs = len(legs)
        exec_cost = self.policy.execution_costs.total_cost_multi_leg(num_legs, is_market=False)

        rationale = (
            f"{decision} via {strike_guidance} {option_type} | "
            f"delta={primary_leg.delta:.2f}, IV={primary_leg.iv:.1%}, "
            f"DTE={primary_leg.dte:.0f}, spread={primary_leg.spread_pct:.1%}, "
            f"exec_cost={exec_cost:.1%}"
        ) if primary_leg.delta and primary_leg.iv else f"{decision} {option_type}"

        return EntryPlan(
            signal_date=signal.date,
            trade_decision=decision,
            direction=direction,
            legs=legs,
            total_risk_usd=tier_risk,
            naked_pct=tier.naked_pct if not has_spread else 0.0,
            spread_pct=tier.spread_pct if has_spread else 0.0,
            rationale=rationale,
            spot_price=spot,
            iv_rank=iv_rank,
            warnings=warnings,
        )

    def _plan_condor(
        self,
        signal: DailySignal,
        account_size_usd: float,
        spot_price: Optional[float],
    ) -> Optional[EntryPlan]:
        """Build 4-leg iron condor using MVRV drift strikes from signal."""
        spot = self._resolve_spot(signal, spot_price)
        if spot is None:
            return None

        dte_cfg = self.policy.get_dte_target("IRON_CONDOR")
        condor_cfg = self.policy.condor

        # MVRV drift strike targets from signal pipeline
        target_short_call = signal.condor_short_call or spot * (1 + condor_cfg.spot_call_band)
        target_short_put = signal.condor_short_put or spot * (1 - condor_cfg.spot_put_band)

        legs: list[LegPlan] = []
        warnings: list[str] = []

        # --- Short call (SELL - use IV-aware scoring for sell side) ---
        call_candidates = self.selector.find_candidates(
            option_type="call",
            side="sell",
            dte_min=dte_cfg.min_dte,
            dte_max=dte_cfg.max_dte,
        )
        short_call = self._find_nearest_from_candidates(call_candidates, target_short_call)
        if not short_call:
            warnings.append("No short call strike found")
            return None

        # --- Long call (wing) ---
        long_call_target = float(short_call.strike) + condor_cfg.wing_offset_usd
        long_call = self._find_nearest_from_candidates(
            call_candidates, long_call_target, same_expiry_as=short_call
        )

        # --- Short put (SELL) ---
        put_candidates = self.selector.find_candidates(
            option_type="put",
            side="sell",
            dte_min=dte_cfg.min_dte,
            dte_max=dte_cfg.max_dte,
        )
        short_put = self._find_nearest_from_candidates(
            put_candidates, target_short_put, same_expiry_as=short_call
        )
        if not short_put:
            warnings.append("No short put strike found")
            return None

        # --- Long put (wing) ---
        long_put_target = float(short_put.strike) - condor_cfg.wing_offset_usd
        long_put = self._find_nearest_from_candidates(
            put_candidates, long_put_target, same_expiry_as=short_call
        )

        # Build legs
        legs.append(self._snapshot_to_leg(short_call, side="sell", leg_type=LegType.SHORT_CALL))
        if long_call:
            legs.append(self._snapshot_to_leg(long_call, side="buy", leg_type=LegType.LONG_CALL))
        else:
            warnings.append("No long call wing; condor is uncapped on upside")

        legs.append(self._snapshot_to_leg(short_put, side="sell", leg_type=LegType.SHORT_PUT))
        if long_put:
            legs.append(self._snapshot_to_leg(long_put, side="buy", leg_type=LegType.LONG_PUT))
        else:
            warnings.append("No long put wing; condor is uncapped on downside")

        # Estimate credit
        credit = Decimal("0")
        for leg in legs:
            if leg.mid_price:
                if leg.side == "sell":
                    credit += leg.mid_price
                else:
                    credit -= leg.mid_price

        # Check minimum credit threshold
        wing_width = condor_cfg.wing_offset_usd
        credit_usd = float(credit) * spot
        credit_pct = credit_usd / wing_width if wing_width > 0 else 0
        
        if credit_pct < condor_cfg.min_credit_pct:
            warnings.append(
                f"Credit {credit_pct:.1%} below minimum {condor_cfg.min_credit_pct:.1%}"
            )

        tier = self.policy.get_tier("IRON_CONDOR")
        tier_risk = tier.risk_usd

        call_dist = (float(short_call.strike) - spot) / spot * 100
        put_dist = (spot - float(short_put.strike)) / spot * 100

        # Execution cost for 4-leg
        exec_cost = self.policy.execution_costs.total_cost_multi_leg(4, is_market=False)

        rationale = (
            f"IRON_CONDOR | short call {short_call.strike} (+{call_dist:.1f}%), "
            f"short put {short_put.strike} (-{put_dist:.1f}%) | "
            f"est credit={float(credit):.4f} BTC ({credit_pct:.1%} of wing) | "
            f"DTE={short_call.dte:.0f} | exec_cost={exec_cost:.1%}"
        )

        return EntryPlan(
            signal_date=signal.date,
            trade_decision="IRON_CONDOR",
            direction="neutral",
            legs=legs,
            total_risk_usd=tier_risk,
            naked_pct=0.0,
            spread_pct=1.0,
            rationale=rationale,
            spot_price=spot,
            warnings=warnings,
        )

    def _find_spread_short_leg(
        self,
        long_snap: OptionSnapshot,
        option_type: str,
        direction: str,
        width_pct: float,
        spot_price: float,
    ) -> Optional[LegPlan]:
        """
        Find the short leg for a vertical spread.
        
        For bull call spread: sell higher strike call
        For bear put spread: sell lower strike put
        """
        width_amount = spot_price * width_pct

        if option_type == "call":
            # Bull call spread: short leg is higher strike
            target_short_strike = float(long_snap.strike) + width_amount
        else:
            # Bear put spread: short leg is lower strike
            target_short_strike = float(long_snap.strike) - width_amount

        # Find short leg with IV-aware scoring (side="sell")
        short_snap = self.selector.find_nearest_strike(
            option_type=option_type,
            target_strike=target_short_strike,
            dte_min=int(long_snap.dte or 7) - 2,
            dte_max=int(long_snap.dte or 21) + 2,
            same_expiry_as=long_snap,
        )

        if short_snap is None:
            return None

        return self._snapshot_to_leg(
            short_snap,
            side="sell",
            leg_type=LegType.SHORT_LEG,
        )

    def _find_nearest_from_candidates(
        self,
        candidates: list[OptionSnapshot],
        target_strike: float,
        same_expiry_as: Optional[OptionSnapshot] = None,
    ) -> Optional[OptionSnapshot]:
        """Find nearest strike from pre-filtered candidates."""
        if same_expiry_as and same_expiry_as.expiry:
            target_expiry = same_expiry_as.expiry
            candidates = [
                c for c in candidates
                if c.expiry and abs((c.expiry - target_expiry).total_seconds()) < 3600
            ]
        
        if not candidates:
            return None
        
        return min(candidates, key=lambda s: abs(float(s.strike) - target_strike))

    def _snapshot_to_leg(
        self,
        snap: OptionSnapshot,
        side: str,
        leg_type: LegType,
    ) -> LegPlan:
        """Convert an OptionSnapshot to a LegPlan."""
        expiry_date = snap.expiry.date() if hasattr(snap.expiry, "date") else snap.expiry
        return LegPlan(
            symbol=snap.symbol,
            side=side,
            option_type=snap.option_type,
            strike=snap.strike,
            expiry=expiry_date,
            leg_type=leg_type,
            delta=float(snap.delta) if snap.delta else None,
            iv=float(snap.iv) if snap.iv else None,
            mid_price=snap.mid_price,
            bid=snap.bid,
            ask=snap.ask,
            spread_pct=snap.spread_pct,
            open_interest=snap.open_interest,
            dte=snap.dte,
        )
