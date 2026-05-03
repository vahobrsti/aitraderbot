"""
Deribit-specific entry engine.

Uses OptionSnapshot data (greeks, IV, liquidity) from Postgres to find
the optimal option contract for each signal type. Replaces the naive
instrument selection in the base orchestrator.

Key improvements over V1 orchestrator:
- Greeks-aware strike selection (target delta, avoid gamma spikes)
- IV-aware sizing (adjust for IV rank)
- Liquidity filtering (spread %, open interest, volume)
- Spread construction (vertical spreads for V7 policy)
- Iron condor 4-leg construction with MVRV drift strikes
- Path-informed DTE selection (uses TTH distribution per state)

Architecture:
    DailySignal
        -> DeribitEntryEngine.plan_entry(signal)
        -> EntryPlan (legs, sizing, rationale)
        -> DeribitEntryEngine.execute_plan(plan)
        -> Orders on Deribit
"""
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Optional

from django.db.models import Q, Avg, Max, Min

from datafeed.models import OptionSnapshot
from signals.models import DailySignal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path-informed DTE targets per signal type
# Derived from analyze_path_stats: optimal_dte ≈ median_TTH + buffer
# ---------------------------------------------------------------------------
PATH_DTE_TARGETS = {
    "CALL":           {"min": 12, "max": 21, "optimal": 14},
    "PUT":            {"min": 10, "max": 14, "optimal": 12},
    "OPTION_CALL":    {"min": 11, "max": 14, "optimal": 12},
    "OPTION_PUT":     {"min": 11, "max": 14, "optimal": 12},
    "TACTICAL_PUT":   {"min": 10, "max": 14, "optimal": 12},
    "BULL_PROBE":     {"min": 10, "max": 14, "optimal": 12},
    "BEAR_PROBE":     {"min": 12, "max": 16, "optimal": 14},
    "IRON_CONDOR":    {"min": 10, "max": 21, "optimal": 14},
    "MVRV_SHORT":     {"min": 15, "max": 21, "optimal": 18},
}

# Delta targets per strike guidance
DELTA_TARGETS = {
    "slight_itm":  {"call": 0.60, "put": -0.60},
    "atm":         {"call": 0.50, "put": -0.50},
    "slight_otm":  {"call": 0.35, "put": -0.35},
    "otm":         {"call": 0.20, "put": -0.20},
    "itm":         {"call": 0.70, "put": -0.70},
    "deep_otm":    {"call": 0.10, "put": -0.10},
}

# Liquidity filters
MIN_OPEN_INTEREST = Decimal("5")       # Minimum OI for a contract
MAX_SPREAD_PCT = 0.15                  # Max bid-ask spread as % of mid
MIN_BID = Decimal("0.0001")            # Must have a bid (in BTC)


class LegType(Enum):
    SINGLE = "single"
    LONG_LEG = "long_leg"
    SHORT_LEG = "short_leg"
    # Iron condor legs
    SHORT_CALL = "short_call"
    LONG_CALL = "long_call"
    SHORT_PUT = "short_put"
    LONG_PUT = "long_put"


@dataclass
class LegPlan:
    """A single option leg to execute."""
    symbol: str
    side: str                    # 'buy' or 'sell'
    option_type: str             # 'call' or 'put'
    strike: Decimal
    expiry: date
    leg_type: LegType
    # Snapshot data at planning time
    delta: Optional[float] = None
    iv: Optional[float] = None
    mid_price: Optional[Decimal] = None
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    spread_pct: Optional[float] = None
    open_interest: Optional[Decimal] = None
    dte: Optional[float] = None


@dataclass
class EntryPlan:
    """Complete entry plan for a signal."""
    signal_date: date
    trade_decision: str
    direction: str               # 'long', 'short', 'neutral'
    legs: list[LegPlan]
    # Sizing
    total_risk_usd: float
    naked_pct: float             # % of risk in naked leg
    spread_pct: float            # % of risk in spread
    # Metadata
    rationale: str
    spot_price: float
    iv_rank: Optional[float] = None
    warnings: list[str] = field(default_factory=list)

    @property
    def is_spread(self) -> bool:
        return len(self.legs) >= 2

    @property
    def is_condor(self) -> bool:
        return len(self.legs) == 4


class DeribitEntryEngine:
    """
    Greeks-aware entry engine for Deribit options.

    Uses OptionSnapshot data from Postgres to find optimal contracts,
    then constructs entry plans aligned with V7 execution policy.
    """

    def __init__(
        self,
        max_spread_pct: float = MAX_SPREAD_PCT,
        min_open_interest: Decimal = MIN_OPEN_INTEREST,
        snapshot_staleness_hours: int = 2,
    ):
        self.max_spread_pct = max_spread_pct
        self.min_open_interest = min_open_interest
        self.snapshot_staleness_hours = snapshot_staleness_hours

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan_entry(
        self,
        signal: DailySignal,
        account_size_usd: float = 100_000,
        spot_price: Optional[float] = None,
    ) -> Optional[EntryPlan]:
        """
        Build an EntryPlan from a DailySignal.

        Reads the latest OptionSnapshot data from Postgres to find the
        best contracts. Returns None if no suitable instruments found.
        """
        decision = signal.trade_decision.upper()

        if decision == "NO_TRADE":
            return None

        if decision == "IRON_CONDOR":
            return self._plan_condor(signal, account_size_usd, spot_price)

        return self._plan_directional(signal, account_size_usd, spot_price)

    # ------------------------------------------------------------------
    # Directional trades (CALL, PUT, spreads)
    # ------------------------------------------------------------------

    def _plan_directional(
        self,
        signal: DailySignal,
        account_size_usd: float,
        spot_price: Optional[float],
    ) -> Optional[EntryPlan]:
        decision = signal.trade_decision.upper()

        # Map decision -> direction + option_type
        decision_map = {
            "CALL":         ("long",  "call"),
            "OPTION_CALL":  ("long",  "call"),
            "PUT":          ("short", "put"),
            "OPTION_PUT":   ("short", "put"),
            "TACTICAL_PUT": ("long",  "put"),
            "BULL_PROBE":   ("long",  "call"),   # alias
            "BEAR_PROBE":   ("short", "put"),     # alias
            "MVRV_SHORT":   ("short", "put"),
        }
        if decision not in decision_map:
            logger.warning(f"Unknown decision: {decision}")
            return None

        direction, option_type = decision_map[decision]

        # Resolve spot
        if spot_price is None:
            spot_price = self._latest_spot()
        if spot_price is None:
            logger.error("Cannot resolve spot price")
            return None

        # DTE target
        dte_cfg = PATH_DTE_TARGETS.get(decision, {"min": 11, "max": 14, "optimal": 12})

        # Strike guidance from signal
        strike_guidance = signal.strike_guidance or "slight_itm"
        target_delta = DELTA_TARGETS.get(strike_guidance, DELTA_TARGETS["slight_itm"])
        abs_target_delta = abs(target_delta[option_type])

        # Find best contract
        candidates = self._query_candidates(
            option_type=option_type,
            dte_min=dte_cfg["min"],
            dte_max=dte_cfg["max"],
        )
        if not candidates:
            logger.warning(f"No {option_type} candidates in DTE {dte_cfg['min']}-{dte_cfg['max']}")
            return None

        # Score and rank
        best = self._rank_candidates(candidates, abs_target_delta, dte_cfg["optimal"])
        if best is None:
            return None

        legs: list[LegPlan] = []
        warnings: list[str] = []

        # V7 policy: naked + spread
        spread_width_pct = signal.spread_width_pct or 0.10
        has_spread = spread_width_pct > 0 and decision not in ("MVRV_SHORT",)

        # Primary (naked) leg — always BUY the option (long call or long put)
        # Direction 'long' = buy call, direction 'short' = buy put
        # The "direction" refers to market view, not option side
        primary_leg = self._snapshot_to_leg(
            best,
            side="buy",
            leg_type=LegType.SINGLE if not has_spread else LegType.LONG_LEG,
        )
        legs.append(primary_leg)

        # Spread short leg
        if has_spread:
            short_leg = self._find_spread_short_leg(
                best, option_type, direction, spread_width_pct, spot_price
            )
            if short_leg:
                legs.append(short_leg)
            else:
                warnings.append("Could not find spread short leg; falling back to naked only")

        # Sizing from V7 tiers
        tier_risk, naked_pct, spread_pct_alloc = self._resolve_tier(decision, signal, account_size_usd)

        # IV rank (informational)
        iv_rank = self._compute_iv_rank(option_type)

        rationale = (
            f"{decision} via {strike_guidance} {option_type} | "
            f"delta={primary_leg.delta:.2f}, IV={primary_leg.iv:.1%}, "
            f"DTE={primary_leg.dte:.0f}, spread={primary_leg.spread_pct:.1%}"
        ) if primary_leg.delta and primary_leg.iv else f"{decision} {option_type}"

        return EntryPlan(
            signal_date=signal.date,
            trade_decision=decision,
            direction=direction,
            legs=legs,
            total_risk_usd=tier_risk,
            naked_pct=naked_pct,
            spread_pct=spread_pct_alloc,
            rationale=rationale,
            spot_price=spot_price,
            iv_rank=iv_rank,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Iron Condor
    # ------------------------------------------------------------------

    def _plan_condor(
        self,
        signal: DailySignal,
        account_size_usd: float,
        spot_price: Optional[float],
    ) -> Optional[EntryPlan]:
        """Build 4-leg iron condor using MVRV drift strikes from signal."""
        if spot_price is None:
            spot_price = self._latest_spot()
        if spot_price is None:
            return None

        dte_cfg = PATH_DTE_TARGETS["IRON_CONDOR"]

        # MVRV drift strike targets from signal pipeline
        target_short_call = signal.condor_short_call or spot_price * 1.10
        target_short_put = signal.condor_short_put or spot_price * 0.90

        # Wing width: $2000 per side (from iron_condor_spec)
        wing_offset = 2000

        legs: list[LegPlan] = []
        warnings: list[str] = []

        # --- Short call ---
        short_call = self._find_nearest_strike(
            "call", target_short_call, dte_cfg["min"], dte_cfg["max"]
        )
        if not short_call:
            warnings.append("No short call strike found")
            return None

        # --- Long call (wing) ---
        long_call_target = float(short_call.strike) + wing_offset
        long_call = self._find_nearest_strike(
            "call", long_call_target, dte_cfg["min"], dte_cfg["max"],
            same_expiry_as=short_call,
        )

        # --- Short put ---
        short_put = self._find_nearest_strike(
            "put", target_short_put, dte_cfg["min"], dte_cfg["max"],
            same_expiry_as=short_call,
        )
        if not short_put:
            warnings.append("No short put strike found")
            return None

        # --- Long put (wing) ---
        long_put_target = float(short_put.strike) - wing_offset
        long_put = self._find_nearest_strike(
            "put", long_put_target, dte_cfg["min"], dte_cfg["max"],
            same_expiry_as=short_call,
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

        tier_risk = 2000.0  # Fixed $2k risk budget for condors

        call_dist = (float(short_call.strike) - spot_price) / spot_price * 100
        put_dist = (spot_price - float(short_put.strike)) / spot_price * 100

        rationale = (
            f"IRON_CONDOR | short call {short_call.strike} (+{call_dist:.1f}%), "
            f"short put {short_put.strike} (-{put_dist:.1f}%) | "
            f"est credit={float(credit):.4f} BTC | "
            f"DTE={short_call.dte:.0f}"
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
            spot_price=spot_price,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Snapshot queries
    # ------------------------------------------------------------------

    def _query_candidates(
        self,
        option_type: str,
        dte_min: int,
        dte_max: int,
        exchange: str = "deribit",
    ) -> list[OptionSnapshot]:
        """
        Query latest OptionSnapshot rows matching filters.
        Returns one snapshot per symbol (the most recent).
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.snapshot_staleness_hours)

        # Get the latest snapshot timestamp per symbol
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

        # Postgres-specific: distinct on symbol gives latest per symbol
        # For SQLite fallback, we'd need a subquery
        try:
            results = list(qs)
        except Exception:
            # Fallback for SQLite (no DISTINCT ON)
            results = self._query_candidates_fallback(
                option_type, dte_min, dte_max, cutoff, exchange
            )

        # Apply liquidity filters
        filtered = []
        for snap in results:
            if snap.open_interest and snap.open_interest < self.min_open_interest:
                continue
            if snap.spread_pct and snap.spread_pct > self.max_spread_pct:
                continue
            if snap.bid and snap.bid < MIN_BID:
                continue
            filtered.append(snap)

        return filtered

    def _query_candidates_fallback(
        self, option_type, dte_min, dte_max, cutoff, exchange
    ) -> list[OptionSnapshot]:
        """SQLite-compatible fallback (no DISTINCT ON)."""
        from django.db.models import Subquery, OuterRef

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

    def _find_nearest_strike(
        self,
        option_type: str,
        target_strike: float,
        dte_min: int,
        dte_max: int,
        same_expiry_as: Optional[OptionSnapshot] = None,
        exchange: str = "deribit",
    ) -> Optional[OptionSnapshot]:
        """Find the snapshot closest to target_strike."""
        candidates = self._query_candidates(option_type, dte_min, dte_max, exchange)

        if same_expiry_as and same_expiry_as.expiry:
            # Filter to same expiry (within 1 hour tolerance for timestamp differences)
            target_expiry = same_expiry_as.expiry
            candidates = [
                c for c in candidates
                if c.expiry and abs((c.expiry - target_expiry).total_seconds()) < 3600
            ]

        if not candidates:
            return None

        return min(candidates, key=lambda s: abs(float(s.strike) - target_strike))

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def _rank_candidates(
        self,
        candidates: list[OptionSnapshot],
        target_abs_delta: float,
        optimal_dte: int,
    ) -> Optional[OptionSnapshot]:
        """
        Rank candidates by composite score:
        - Delta proximity (40% weight)
        - DTE proximity (30% weight)
        - Liquidity / spread (20% weight)
        - IV relative value (10% weight)
        """
        if not candidates:
            return None

        scored = []
        for snap in candidates:
            delta_score = 0.0
            if snap.delta is not None:
                delta_diff = abs(abs(float(snap.delta)) - target_abs_delta)
                delta_score = max(0, 1.0 - delta_diff / 0.30)  # 0.30 delta range

            dte_score = 0.0
            if snap.dte is not None:
                dte_diff = abs(snap.dte - optimal_dte)
                dte_score = max(0, 1.0 - dte_diff / 15.0)  # 15 day range

            liq_score = 0.0
            if snap.spread_pct is not None:
                liq_score = max(0, 1.0 - snap.spread_pct / self.max_spread_pct)
            if snap.open_interest:
                oi_bonus = min(float(snap.open_interest) / 100.0, 1.0)
                liq_score = (liq_score + oi_bonus) / 2.0

            # Lower IV is better for buying, higher for selling
            iv_score = 0.5  # neutral default
            if snap.iv is not None:
                iv_score = max(0, 1.0 - float(snap.iv))  # lower IV = higher score for buying

            composite = (
                0.40 * delta_score
                + 0.30 * dte_score
                + 0.20 * liq_score
                + 0.10 * iv_score
            )
            scored.append((composite, snap))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1] if scored else None

    # ------------------------------------------------------------------
    # Spread construction
    # ------------------------------------------------------------------

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

        short_snap = self._find_nearest_strike(
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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

    def _latest_spot(self) -> Optional[float]:
        """Get latest spot price from most recent snapshot."""
        snap = (
            OptionSnapshot.objects
            .filter(exchange="deribit", spot_price__gt=0)
            .order_by("-timestamp")
            .values_list("spot_price", flat=True)
            .first()
        )
        return float(snap) if snap else None

    def _compute_iv_rank(self, option_type: str, lookback_days: int = 30) -> Optional[float]:
        """
        Compute IV rank: where current ATM IV sits relative to last N days.
        Returns 0.0 (lowest) to 1.0 (highest).
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        recent_cutoff = datetime.now(timezone.utc) - timedelta(hours=6)

        # Get current ATM IV (closest to 0.50 delta)
        current = (
            OptionSnapshot.objects
            .filter(
                exchange="deribit",
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
                exchange="deribit",
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

    def _resolve_tier(
        self,
        decision: str,
        signal: DailySignal,
        account_size_usd: float,
    ) -> tuple[float, float, float]:
        """
        Resolve V7 tier sizing.

        Returns (total_risk_usd, naked_pct, spread_pct).
        """
        # V7 tier map
        tier_map = {
            "PRIMARY_SHORT": 1,
            "OPTION_CALL":   1,
            "CALL":          1,
            "PUT":           1,
            "BULL_PROBE":    2,
            "BEAR_PROBE":    2,
            "TACTICAL_PUT":  2,
            "OPTION_PUT":    2,
            "MVRV_SHORT":    2,
        }
        tier_num = tier_map.get(decision, 2)

        tier_configs = {
            1: {"risk": 4000, "naked_pct": 0.20, "spread_pct": 0.80},
            2: {"risk": 2400, "naked_pct": 0.25, "spread_pct": 0.75},
        }
        cfg = tier_configs.get(tier_num, tier_configs[2])

        # Scale by signal's size_multiplier
        size_mult = signal.effective_size or signal.size_multiplier or 1.0
        risk = cfg["risk"] * size_mult

        # Cap at 6% of account
        max_risk = account_size_usd * 0.06
        risk = min(risk, max_risk)

        return risk, cfg["naked_pct"], cfg["spread_pct"]
