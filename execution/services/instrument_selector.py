"""
Instrument Selector - Marketplace-aware contract selection and scoring.

Handles:
- IV-aware scoring (different logic for buy vs sell legs)
- Liquidity filtering with configurable thresholds
- Delta/DTE targeting from policy
- Execution cost estimation

Usage:
    from execution.services.instrument_selector import InstrumentSelector
    from execution.services.policy import get_policy
    
    selector = InstrumentSelector(policy=get_policy())
    
    # Find best contract for a long put
    candidates = selector.find_candidates(
        option_type="put",
        side="buy",
        dte_min=10,
        dte_max=14,
    )
    
    best = selector.rank_candidates(
        candidates,
        target_delta=-0.60,
        optimal_dte=12,
        side="buy",
    )
"""
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Optional

from django.db.models import Subquery, OuterRef

from datafeed.models import OptionSnapshot
from execution.services.policy import PolicyVersion, get_policy

logger = logging.getLogger(__name__)


# Legacy enums for backward compatibility with order_builder.py
class StrikeSelection(Enum):
    """Strike selection result (legacy)."""
    ATM = "atm"
    ITM = "itm"
    OTM = "otm"


class SpreadSelection(Enum):
    """Spread type selection (legacy)."""
    BULL_CALL = "bull_call"
    BEAR_PUT = "bear_put"
    BULL_PUT = "bull_put"
    BEAR_CALL = "bear_call"


@dataclass
class ScoredCandidate:
    """Candidate with scoring breakdown."""
    snapshot: OptionSnapshot
    total_score: float
    delta_score: float
    dte_score: float
    liquidity_score: float
    iv_score: float
    execution_cost_estimate: float


class InstrumentSelector:
    """
    Marketplace-aware instrument selection with IV-aware scoring.
    
    Key improvement: IV scoring is side-aware.
    - For BUY legs: lower IV = higher score (cheaper premium)
    - For SELL legs: higher IV = higher score (more premium collected)
    """
    
    def __init__(self, policy: Optional[PolicyVersion] = None):
        self.policy = policy or get_policy()
    
    def find_candidates(
        self,
        option_type: str,
        side: str,  # 'buy' or 'sell'
        dte_min: int,
        dte_max: int,
        exchange: str = "deribit",
        staleness_hours: Optional[int] = None,
    ) -> list[OptionSnapshot]:
        """
        Find candidate contracts matching filters.
        
        Args:
            option_type: 'call' or 'put'
            side: 'buy' or 'sell' (affects IV scoring later)
            dte_min: Minimum days to expiry
            dte_max: Maximum days to expiry
            exchange: Exchange name
            staleness_hours: Max snapshot age (uses policy default if None)
        
        Returns:
            List of OptionSnapshot candidates passing liquidity filters.
        """
        staleness = staleness_hours or self.policy.snapshot_staleness_hours
        cutoff = datetime.now(timezone.utc) - timedelta(hours=staleness)
        
        # Query latest snapshot per symbol
        try:
            # Postgres: use DISTINCT ON
            qs = (
                OptionSnapshot.objects
                .filter(
                    exchange=exchange,
                    option_type=option_type,
                    timestamp__gte=cutoff,
                    dte__gte=dte_min,
                    dte__lte=dte_max,
                )
                .exclude(bid__isnull=True)
                .exclude(bid__lte=0)
                .order_by("symbol", "-timestamp")
                .distinct("symbol")
            )
            results = list(qs)
        except Exception:
            # SQLite fallback
            results = self._query_fallback(
                option_type, dte_min, dte_max, cutoff, exchange
            )
        
        # Apply liquidity filters
        liq = self.policy.liquidity
        filtered = []
        for snap in results:
            if snap.open_interest and snap.open_interest < liq.min_open_interest:
                continue
            if snap.spread_pct and snap.spread_pct > liq.max_spread_pct:
                continue
            if snap.bid and snap.bid < liq.min_bid_btc:
                continue
            filtered.append(snap)
        
        logger.debug(
            f"Found {len(filtered)} {option_type} candidates "
            f"(DTE {dte_min}-{dte_max}, {len(results)} before liquidity filter)"
        )
        return filtered
    
    def _query_fallback(
        self, option_type, dte_min, dte_max, cutoff, exchange
    ) -> list[OptionSnapshot]:
        """SQLite-compatible fallback (no DISTINCT ON)."""
        latest_ts = (
            OptionSnapshot.objects
            .filter(symbol=OuterRef("symbol"), exchange=exchange)
            .order_by("-timestamp")
            .values("timestamp")[:1]
        )
        qs = (
            OptionSnapshot.objects
            .filter(
                exchange=exchange,
                option_type=option_type,
                timestamp__gte=cutoff,
                dte__gte=dte_min,
                dte__lte=dte_max,
                timestamp=Subquery(latest_ts),
            )
            .exclude(bid__isnull=True)
            .exclude(bid__lte=0)
        )
        return list(qs)
    
    def rank_candidates(
        self,
        candidates: list[OptionSnapshot],
        target_delta: float,
        optimal_dte: int,
        side: str,  # 'buy' or 'sell' - CRITICAL for IV scoring
    ) -> Optional[ScoredCandidate]:
        """
        Rank candidates by composite score with IV-aware logic.
        
        Args:
            candidates: List of OptionSnapshot to rank
            target_delta: Target absolute delta (e.g., 0.60)
            optimal_dte: Optimal days to expiry
            side: 'buy' or 'sell' - determines IV scoring direction
        
        Returns:
            Best ScoredCandidate, or None if no candidates.
        """
        if not candidates:
            return None
        
        weights = self.policy.scoring_weights
        scored = []
        
        for snap in candidates:
            # Delta score: closer to target = higher score
            delta_score = 0.0
            if snap.delta is not None:
                delta_diff = abs(abs(float(snap.delta)) - abs(target_delta))
                delta_score = max(0, 1.0 - delta_diff / 0.30)
            
            # DTE score: closer to optimal = higher score
            dte_score = 0.0
            if snap.dte is not None:
                dte_diff = abs(snap.dte - optimal_dte)
                dte_score = max(0, 1.0 - dte_diff / 15.0)
            
            # Liquidity score: tighter spread + higher OI = higher score
            liq_score = 0.0
            if snap.spread_pct is not None:
                liq_score = max(0, 1.0 - snap.spread_pct / self.policy.liquidity.max_spread_pct)
            if snap.open_interest:
                oi_bonus = min(float(snap.open_interest) / 100.0, 1.0)
                liq_score = (liq_score + oi_bonus) / 2.0
            
            # IV score: SIDE-AWARE
            # - BUY legs: lower IV = higher score (cheaper to buy)
            # - SELL legs: higher IV = higher score (more premium to collect)
            iv_score = 0.5  # neutral default
            if snap.iv is not None:
                iv_val = float(snap.iv)
                if side == "buy":
                    # Lower IV is better for buying
                    iv_score = max(0, 1.0 - iv_val)
                else:
                    # Higher IV is better for selling
                    iv_score = min(1.0, iv_val)
            
            # Composite score
            total = (
                weights.get("delta", 0.40) * delta_score
                + weights.get("dte", 0.30) * dte_score
                + weights.get("liquidity", 0.20) * liq_score
                + weights.get("iv", 0.10) * iv_score
            )
            
            # Estimate execution cost
            exec_cost = self._estimate_execution_cost(snap)
            
            scored.append(ScoredCandidate(
                snapshot=snap,
                total_score=total,
                delta_score=delta_score,
                dte_score=dte_score,
                liquidity_score=liq_score,
                iv_score=iv_score,
                execution_cost_estimate=exec_cost,
            ))
        
        # Sort by total score descending
        scored.sort(key=lambda x: x.total_score, reverse=True)
        
        best = scored[0]
        logger.debug(
            f"Best candidate: {best.snapshot.symbol} "
            f"(score={best.total_score:.3f}, delta={best.delta_score:.2f}, "
            f"dte={best.dte_score:.2f}, liq={best.liquidity_score:.2f}, "
            f"iv={best.iv_score:.2f}, side={side})"
        )
        return best
    
    def _estimate_execution_cost(self, snap: OptionSnapshot) -> float:
        """
        Estimate execution cost as % of premium.
        
        Includes: fees + half spread + expected slippage
        """
        costs = self.policy.execution_costs
        
        # Base fee (assume taker for market orders)
        fee_pct = costs.taker_fee_pct
        
        # Spread cost (half the bid-ask spread)
        spread_cost = 0.0
        if snap.spread_pct:
            spread_cost = snap.spread_pct / 2.0
        
        # Slippage estimate
        slippage = costs.slippage_pct
        
        return fee_pct + spread_cost + slippage
    
    def find_nearest_strike(
        self,
        option_type: str,
        target_strike: float,
        dte_min: int,
        dte_max: int,
        same_expiry_as: Optional[OptionSnapshot] = None,
        exchange: str = "deribit",
        staleness_hours: Optional[int] = None,
    ) -> Optional[OptionSnapshot]:
        """
        Find the snapshot closest to target_strike.
        
        Used for spread construction (finding short leg near long leg).
        """
        candidates = self.find_candidates(
            option_type=option_type,
            side="sell",  # Typically finding short leg
            dte_min=dte_min,
            dte_max=dte_max,
            exchange=exchange,
            staleness_hours=staleness_hours,
        )
        
        if same_expiry_as and same_expiry_as.expiry:
            # Filter to same expiry (within 1 hour tolerance)
            target_expiry = same_expiry_as.expiry
            candidates = [
                c for c in candidates
                if c.expiry and abs((c.expiry - target_expiry).total_seconds()) < 3600
            ]
        
        if not candidates:
            return None
        
        return min(candidates, key=lambda s: abs(float(s.strike) - target_strike))
    
    def compute_iv_rank(
        self,
        option_type: str,
        lookback_days: int = 30,
        exchange: str = "deribit",
    ) -> Optional[float]:
        """
        Compute IV rank: where current ATM IV sits relative to last N days.
        
        Returns 0.0 (lowest) to 1.0 (highest).
        """
        from django.db.models import Min, Max
        
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        recent_cutoff = datetime.now(timezone.utc) - timedelta(hours=6)
        
        # Get current ATM IV
        current = (
            OptionSnapshot.objects
            .filter(
                exchange=exchange,
                option_type=option_type,
                timestamp__gte=recent_cutoff,
                iv__isnull=False,
            )
            .order_by("-timestamp")
            .first()
        )
        if not current or not current.iv:
            return None
        
        current_iv = float(current.iv)
        
        # Get historical IV range
        agg = (
            OptionSnapshot.objects
            .filter(
                exchange=exchange,
                option_type=option_type,
                timestamp__gte=cutoff,
                iv__isnull=False,
                dte__gte=7,
                dte__lte=30,
            )
            .aggregate(
                iv_min=Min("iv"),
                iv_max=Max("iv"),
            )
        )
        
        iv_min = float(agg["iv_min"]) if agg["iv_min"] else None
        iv_max = float(agg["iv_max"]) if agg["iv_max"] else None
        
        if iv_min is None or iv_max is None or iv_max == iv_min:
            return None
        
        return (current_iv - iv_min) / (iv_max - iv_min)
    
    def get_latest_spot(self, exchange: str = "deribit") -> Optional[float]:
        """Get latest spot price from most recent snapshot."""
        snap = (
            OptionSnapshot.objects
            .filter(exchange=exchange, spot_price__gt=0)
            .order_by("-timestamp")
            .values_list("spot_price", flat=True)
            .first()
        )
        return float(snap) if snap else None
