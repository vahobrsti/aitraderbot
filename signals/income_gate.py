"""
Income gate evaluation for directional credit spreads.

Implements Bull Put Spread and Bear Call Spread gates using the same
scoring + veto pattern as condor_gate.py. Produces two layers of truth:
1. Regime eligibility (on-chain scoring + vetoes)
2. Spread selection (option chain filtering + credit validation)

Uses MVRV-derived strike boundaries:
- Bull put: floor = spot / mvrv_60d (cost basis of recent buyers)
- Bear call: ceiling = (spot / mvrv_composite) * mvrv_composite_p90_180d
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from signals.options import OptionStructure, SpreadGuidance, format_stop_loss_string


# ============================================================
# DATA MODELS
# ============================================================


@dataclass
class IncomeGateConfig:
    """Configurable parameters for income gates."""

    # Scoring
    score_threshold: float = 70.0

    # Option chain filters
    min_delta: float = 0.12
    max_delta: float = 0.35
    max_bid_ask_spread_pct: float = 0.15
    credit_delta_scalar: float = 0.55  # required credit = delta × scalar (P50 line)
    credit_delta_scalar_floor: float = 0.40  # absolute minimum scalar (excellent setups)
    min_spread_width_pct: float = 0.03  # spread_width / spot minimum
    max_spread_width_pct: float = 0.08  # spread_width / spot maximum
    min_otm_pct: float = 0.04  # short strike must be at least 4% OTM

    # MVRV floor/ceiling validity
    min_mvrv_for_bull_put: float = 0.0  # No minimum — P5 fallback handles underwater case
    mvrv_strong_floor_threshold: float = 1.03  # MVRV above this = strong floor

    # DTE windows
    tactical_min_dte: int = 9
    tactical_max_dte: int = 21
    income_min_dte: int = 21
    income_max_dte: int = 45

    # Bear-call ceiling — data-driven MVRV rally target (underwater regime only)
    # When mvrv_60d < 1 the static cost_basis × p90_180d ceiling projects the
    # short call unreachably far OTM. Instead, estimate the ceiling from how far
    # mvrv_60d itself rallies over the option horizon in the trailing window.
    # Percentile is a tunable hypothesis (see compute_bear_call_target_mvrv).
    bear_call_ceiling_percentile: float = 0.60  # percentile of trailing forward MVRV rallies
    bear_call_ceiling_lookback: int = 180       # trailing days sampled
    bear_call_rally_hmin: int = 7               # forward window min (decoupled from DTE knobs)
    bear_call_rally_hmax: int = 21              # forward window max (decoupled from DTE knobs)

    # Position management
    cooldown_days: int = 5
    max_concurrent: int = 1

    # Volatility
    atr_expansion_threshold: float = 1.5


@dataclass
class SpreadCandidate:
    """Internal: a validated spread pair from the option chain."""

    short_strike: float
    long_strike: float
    credit: float
    spread_width: float
    dte: int
    max_loss: float
    short_delta: float


@dataclass
class SpreadSetup:
    """A risk-tiered spread setup presented to the human for decision."""

    risk_tier: str  # "low", "medium", "high"
    short_strike: float
    long_strike: float
    credit: float
    spread_width: float
    dte: int
    max_loss: float
    short_delta: float
    credit_width_pct: float  # credit / spread_width
    otm_pct: float  # distance from spot as %
    risk_reward: float  # credit / max_loss


@dataclass
class IncomeGateResult:
    """Result of an income spread gate evaluation."""

    # --- Regime layer (always populated) ---
    score: float  # 0-100 additive score
    regime_eligible: bool  # True if score >= threshold and no vetoes
    eligible: bool  # True if regime_eligible AND chain_valid
    veto_reasons: list = field(default_factory=list)
    score_components: dict = field(default_factory=dict)
    threshold: float = 70.0
    structure: OptionStructure = OptionStructure.NO_TRADE

    # --- Chain layer (populated only when regime_eligible and chain provided) ---
    chain_valid: bool = False
    chain_rejection_reason: Optional[str] = None
    spread_guidance: Optional[SpreadGuidance] = None

    # --- Multi-setup output (human picks the risk tier) ---
    setups: list = field(default_factory=list)  # List[SpreadSetup], up to 3 tiers

    # --- Legacy single-spread fields (populated from best setup for backward compat) ---
    short_strike: Optional[float] = None
    long_strike: Optional[float] = None
    credit: Optional[float] = None
    dte: Optional[int] = None
    max_loss: Optional[float] = None


# ============================================================
# CHAIN NORMALIZATION
# ============================================================

# Column mappings from exchange-specific names to canonical names
_COLUMN_ALIASES = {
    "strike": ["strike"],
    "side": ["option_type", "side"],
    "delta": ["delta"],
    "bid": ["bid", "bid_price"],
    "ask": ["ask", "ask_price"],
    "dte": ["dte", "days_to_expiry"],
    "spread_pct": ["spread_pct"],
}

_REQUIRED_COLUMNS = {"strike", "side", "delta", "bid", "ask", "dte"}


def dedupe_chain_to_latest(chain_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse intraday option snapshots to one row per contract (latest snapshot).

    The OptionSnapshot table stores multiple intraday snapshots per contract
    (hourly). If the raw query result is fed straight into spread selection,
    every contract appears N times with slightly different delta (spot drifts
    hour to hour) and a DTE that flips across the integer-day boundary as the
    fractional day count ticks down (e.g. 15.0 → 14.99 within the same day,
    int-truncated to 15 then 14). The tier selector then buckets these as
    distinct candidates and can pick the *same* contract for two risk tiers
    (identical strikes, different DTE/credit).

    Keep only the most recent snapshot per contract so each contract is a single
    candidate with one delta and one DTE. Contract identity is the exchange
    symbol when present, else (strike, option_type/side, expiry) — always scoped
    by exchange, since a symbol is only unique within an exchange. Callers should
    still filter the query to a single exchange; this dedup is a safety net, not
    a venue selector.

    Args:
        chain_df: Raw chain DataFrame including a ``timestamp`` column (and
            ideally ``symbol``). If ``timestamp`` is absent the frame is
            returned unchanged (already deduped or single-snapshot source).

    Returns:
        Deduplicated DataFrame, one row per contract.
    """
    if chain_df is None or chain_df.empty:
        return chain_df
    if "timestamp" not in chain_df.columns:
        return chain_df

    df = chain_df.sort_values("timestamp")
    if "symbol" in df.columns:
        subset = ["symbol"]
    else:
        subset = [
            c for c in ("strike", "option_type", "side", "expiry")
            if c in df.columns
        ]
    # A contract symbol is only unique within an exchange — the snapshot table's
    # uniqueness is (timestamp, symbol, exchange). Include exchange in the key so
    # mixed-exchange input is never silently collapsed across venues (callers are
    # expected to scope to a single exchange; this is a defensive backstop).
    if "exchange" in df.columns:
        subset = subset + ["exchange"]
    if subset:
        df = df.drop_duplicates(subset=subset, keep="last")
    return df.reset_index(drop=True)


def normalize_chain_columns(chain_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Normalize column names from Deribit/Bybit schemas to canonical names.

    Handles: option_type/side, days_to_expiry/dte, signed vs absolute delta.
    Returns DataFrame with canonical columns, or None if required columns missing.
    """
    if chain_df is None or chain_df.empty:
        return None

    df = chain_df.copy()
    existing_cols = set(df.columns)

    # Map aliases to canonical names
    for canonical, aliases in _COLUMN_ALIASES.items():
        if canonical not in existing_cols:
            for alias in aliases:
                if alias in existing_cols:
                    df = df.rename(columns={alias: canonical})
                    break

    # Check required columns
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        return None

    # Normalize side to lowercase "put"/"call"
    df["side"] = df["side"].astype(str).str.lower().str.strip()

    # Convert delta to absolute value (puts have negative delta)
    df["delta"] = df["delta"].apply(
        lambda x: abs(float(x)) if pd.notna(x) else 0.0
    )

    # Compute spread_pct if missing
    if "spread_pct" not in df.columns:
        mid = (df["bid"].astype(float) + df["ask"].astype(float)) / 2
        df["spread_pct"] = ((df["ask"].astype(float) - df["bid"].astype(float)) / mid).where(mid > 0, other=1.0)

    # Ensure numeric types
    for col in ("strike", "delta", "bid", "ask", "dte"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with NaN in required numeric columns
    df = df.dropna(subset=["strike", "delta", "bid", "ask", "dte"])

    return df


# ============================================================
# MVRV-DERIVED STRIKE BOUNDARIES
# ============================================================


def compute_strike_boundaries(
    spot_price: float,
    mvrv_60d: Optional[float],
    mvrv_60d_p10_180d: Optional[float] = None,
    mvrv_60d_p90_180d: Optional[float] = None,
) -> tuple[Optional[float], Optional[float]]:
    """
    Compute MVRV-derived price boundaries for strike selection.

    Floor logic (bull put):
    - cost_basis = spot / mvrv_60d
    - If mvrv_60d >= 1 (buyers profitable): floor = cost_basis (below spot)
    - If mvrv_60d < 1 (buyers underwater): floor = cost_basis × P10

    Ceiling logic (bear call):
    - ceiling = cost_basis × P90 (where price would be at 6-month MVRV P90)

    Returns:
        (floor_price, ceiling_price)
        - floor_price: bull put short strikes must be at or below this
        - ceiling_price: bear call short strikes must be at or above this
        Either may be None if MVRV data is unavailable.
    """
    floor_price = None
    ceiling_price = None

    if mvrv_60d and mvrv_60d > 0:
        cost_basis = spot_price / mvrv_60d

        # Floor
        if cost_basis <= spot_price:
            # Buyers profitable → cost basis is a natural support
            floor_price = cost_basis
        elif mvrv_60d_p10_180d and mvrv_60d_p10_180d > 0:
            # Buyers underwater → use P10 as tighter anchor
            floor_price = cost_basis * mvrv_60d_p10_180d
        else:
            floor_price = cost_basis

        # Ceiling
        if mvrv_60d_p90_180d and mvrv_60d_p90_180d > 0:
            ceiling_price = cost_basis * mvrv_60d_p90_180d

    return floor_price, ceiling_price


# ============================================================
# BULL PUT SPREAD — SCORING & VETOES
# ============================================================


def compute_bull_put_score(row: pd.Series) -> tuple[float, dict]:
    """
    Compute additive score (0-100) for bull put spread eligibility.

    Components:
        mdia_inflow (+25): Fresh capital entering
        whale_sponsored (+20): Smart money accumulating
        mvrv_macro_bullish (+20): Macro structure supports upside
        no_option_put (+15): No conflicting put signal
        sentiment_safe (+10): Not in extreme greed
        whale_not_distributing (+10): No active distribution
    """
    components = {}
    score = 0.0

    # MDIA inflow (+25)
    mdia_inflow = (
        row.get("mdia_regime_inflow", 0) == 1
        or row.get("mdia_regime_strong_inflow", 0) == 1
    )
    mdia_contrib = 25.0 if mdia_inflow else 0.0
    score += mdia_contrib
    components["mdia_inflow"] = mdia_contrib

    # Whale sponsored (+20)
    whale_sponsored = (
        row.get("whale_regime_broad_accum", 0) == 1
        or row.get("whale_regime_strategic_accum", 0) == 1
    )
    whale_contrib = 20.0 if whale_sponsored else 0.0
    score += whale_contrib
    components["whale_sponsored"] = whale_contrib

    # MVRV macro bullish (+20)
    mvrv_bullish = (
        row.get("mvrv_ls_regime_call_confirm", 0) == 1
        or row.get("mvrv_ls_regime_call_confirm_recovery", 0) == 1
        or row.get("mvrv_ls_regime_call_confirm_trend", 0) == 1
    )
    mvrv_contrib = 20.0 if mvrv_bullish else 0.0
    score += mvrv_contrib
    components["mvrv_macro_bullish"] = mvrv_contrib

    # No option put signal (+15)
    no_put = row.get("signal_option_put", 0) == 0
    no_put_contrib = 15.0 if no_put else 0.0
    score += no_put_contrib
    components["no_option_put"] = no_put_contrib

    # Sentiment safe — not extreme greed (+10)
    sent_safe = row.get("sent_bucket_extreme_greed", 0) == 0
    sent_contrib = 10.0 if sent_safe else 0.0
    score += sent_contrib
    components["sentiment_safe"] = sent_contrib

    # Whale not distributing (+10)
    whale_not_distrib = (
        row.get("whale_regime_distribution", 0) == 0
        and row.get("whale_regime_distribution_strong", 0) == 0
    )
    whale_nd_contrib = 10.0 if whale_not_distrib else 0.0
    score += whale_nd_contrib
    components["whale_not_distributing"] = whale_nd_contrib

    score = max(0.0, min(100.0, score))
    return score, components


def check_bull_put_vetoes(
    row: pd.Series,
    atr_ratio: Optional[float] = None,
    fusion_state: Optional[str] = None,
    higher_priority_active: bool = False,
    condor_eligible: bool = False,
    config: IncomeGateConfig = None,
) -> list[str]:
    """
    Check hard veto conditions that block bull put spread entry.

    Returns list of veto reason strings (empty = no vetoes).
    """
    if config is None:
        config = IncomeGateConfig()

    vetoes = []

    # MVRV floor invalid — no cost-basis support below spot
    # Reject when MVRV-60d is missing, NaN, non-positive, or below threshold
    mvrv_60d = row.get("mvrv_60d", row.get("mvrv_usd_60d"))
    mvrv_60d_valid = False
    if mvrv_60d is not None:
        try:
            mvrv_val = float(mvrv_60d)
            if mvrv_val != mvrv_val:  # NaN check
                mvrv_60d_valid = False
            elif mvrv_val <= 0:
                mvrv_60d_valid = False
            elif mvrv_val >= config.min_mvrv_for_bull_put:
                mvrv_60d_valid = True
        except (ValueError, TypeError):
            mvrv_60d_valid = False
    if not mvrv_60d_valid:
        vetoes.append("MVRV_FLOOR_INVALID")

    # MVRV below P10 band — current MVRV is in extreme low territory,
    # no valid floor anchor exists within the 90% historical range
    if mvrv_60d_valid:
        mvrv_60d_p10 = row.get("mvrv_60d_p10_180d")
        if mvrv_60d_p10 is not None:
            try:
                p10_val = float(mvrv_60d_p10)
                if p10_val == p10_val and float(mvrv_60d) < p10_val:
                    vetoes.append("MVRV_BELOW_P10_BAND")
            except (ValueError, TypeError):
                pass

    # Directional signal conflict
    if row.get("signal_option_put", 0) == 1:
        vetoes.append("OPTION_PUT_ACTIVE")

    # Sentiment persistence
    if row.get("sent_extreme_greed_persist_5d", 0) == 1:
        vetoes.append("EXTREME_GREED_PERSIST_5D")

    # MVRV macro bearish
    mvrv_bearish = (
        row.get("mvrv_ls_regime_put_confirm", 0) == 1
        or row.get("mvrv_ls_regime_bear_continuation", 0) == 1
        or row.get("mvrv_ls_early_rollover", 0) == 1
        or row.get("mvrv_ls_weak_downtrend", 0) == 1
        or row.get("mvrv_ls_regime_distribution_warning", 0) == 1
    )
    if mvrv_bearish:
        vetoes.append("MVRV_MACRO_BEARISH")

    # Whale distribution strong
    if row.get("whale_regime_distribution_strong", 0) == 1:
        vetoes.append("WHALE_DISTRIB_STRONG")

    # Higher-priority signal (passed by caller)
    if higher_priority_active:
        vetoes.append("HIGHER_PRIORITY_SIGNAL")

    # ATR expansion
    if atr_ratio is not None and atr_ratio > config.atr_expansion_threshold:
        vetoes.append(
            f"ATR_EXPANSION({atr_ratio:.2f}>{config.atr_expansion_threshold})"
        )

    # Fusion state conflict
    if fusion_state in ("bear_continuation", "distribution_risk"):
        vetoes.append("FUSION_STATE_BEARISH")

    # Condor precedence
    if condor_eligible:
        vetoes.append("CONDOR_PRECEDENCE")

    return vetoes


# ============================================================
# BEAR CALL SPREAD — SCORING & VETOES
# ============================================================


def compute_bear_call_score(row: pd.Series) -> tuple[float, dict]:
    """
    Compute additive score (0-100) for bear call spread eligibility.

    Components:
        no_mdia_inflow_or_aging (+25): No fresh capital / aging
        whale_distribution (+20): Smart money distributing
        mvrv_macro_bearish (+20): Macro structure supports downside
        no_option_call (+15): No conflicting call signal
        sentiment_greed_or_flat (+10): Greed or flattening sentiment
        whale_not_accumulating (+10): No active accumulation
    """
    components = {}
    score = 0.0

    # No MDIA inflow or aging (+25)
    no_inflow = (
        row.get("mdia_regime_inflow", 0) == 0
        and row.get("mdia_regime_strong_inflow", 0) == 0
    )
    aging = row.get("mdia_regime_aging", 0) == 1
    mdia_contrib = 25.0 if (no_inflow or aging) else 0.0
    score += mdia_contrib
    components["no_mdia_inflow_or_aging"] = mdia_contrib

    # Whale distribution (+20)
    whale_distrib = (
        row.get("whale_regime_distribution", 0) == 1
        or row.get("whale_regime_distribution_strong", 0) == 1
    )
    whale_contrib = 20.0 if whale_distrib else 0.0
    score += whale_contrib
    components["whale_distribution"] = whale_contrib

    # MVRV macro bearish (+20)
    mvrv_bearish = (
        row.get("mvrv_ls_regime_put_confirm", 0) == 1
        or row.get("mvrv_ls_regime_bear_continuation", 0) == 1
        or row.get("mvrv_ls_early_rollover", 0) == 1
        or row.get("mvrv_ls_weak_downtrend", 0) == 1
        or row.get("mvrv_ls_regime_distribution_warning", 0) == 1
    )
    mvrv_contrib = 20.0 if mvrv_bearish else 0.0
    score += mvrv_contrib
    components["mvrv_macro_bearish"] = mvrv_contrib

    # No option call signal (+15)
    no_call = row.get("signal_option_call", 0) == 0
    no_call_contrib = 15.0 if no_call else 0.0
    score += no_call_contrib
    components["no_option_call"] = no_call_contrib

    # Sentiment greed or flattening (+10)
    sent_greed_flat = (
        row.get("sent_bucket_greed", 0) == 1
        or row.get("sent_is_flattening", 0) == 1
    )
    sent_contrib = 10.0 if sent_greed_flat else 0.0
    score += sent_contrib
    components["sentiment_greed_or_flat"] = sent_contrib

    # Whale not accumulating (+10)
    whale_not_accum = (
        row.get("whale_regime_broad_accum", 0) == 0
        and row.get("whale_regime_strategic_accum", 0) == 0
    )
    whale_na_contrib = 10.0 if whale_not_accum else 0.0
    score += whale_na_contrib
    components["whale_not_accumulating"] = whale_na_contrib

    score = max(0.0, min(100.0, score))
    return score, components


def check_bear_call_vetoes(
    row: pd.Series,
    atr_ratio: Optional[float] = None,
    fusion_state: Optional[str] = None,
    higher_priority_active: bool = False,
    condor_eligible: bool = False,
    config: IncomeGateConfig = None,
) -> list[str]:
    """
    Check hard veto conditions that block bear call spread entry.

    Returns list of veto reason strings (empty = no vetoes).
    """
    if config is None:
        config = IncomeGateConfig()

    vetoes = []

    # MVRV ceiling invalid — no resistance anchor above spot
    # Uses mvrv_composite_pct (current composite value) and mvrv_comp_max_180d
    # (180-day max). If current composite is at or above the 180d max, there's
    # no historical resistance overhead to anchor the short call against.
    # Falls back to mvrv_composite / mvrv_composite_p90_180d if available.
    mvrv_composite_pct = row.get("mvrv_composite_pct")
    mvrv_comp_max_180d = row.get("mvrv_comp_max_180d")
    mvrv_composite = row.get("mvrv_composite")
    mvrv_composite_p90 = row.get("mvrv_composite_p90_180d")

    ceiling_valid = True  # assume valid unless proven otherwise
    if mvrv_composite is not None and mvrv_composite_p90 is not None:
        # Primary path: raw composite values
        try:
            mvrv_c = float(mvrv_composite)
            p90 = float(mvrv_composite_p90)
            if mvrv_c == mvrv_c and p90 == p90 and mvrv_c > 0 and p90 > 0:
                if p90 <= mvrv_c:
                    ceiling_valid = False
        except (ValueError, TypeError):
            pass
    elif mvrv_composite_pct is not None and mvrv_comp_max_180d is not None:
        # Fallback: composite pct vs 180d max
        # If composite_pct >= comp_max_180d, we're at or above the ceiling
        try:
            comp_pct = float(mvrv_composite_pct)
            comp_max = float(mvrv_comp_max_180d)
            if comp_pct == comp_pct and comp_max == comp_max:
                if comp_pct >= comp_max:
                    ceiling_valid = False
        except (ValueError, TypeError):
            pass

    if not ceiling_valid:
        vetoes.append("MVRV_CEILING_INVALID")

    # MVRV above P90 band — current MVRV is in extreme high territory,
    # no valid ceiling anchor exists within the 90% historical range
    mvrv_60d_raw = row.get("mvrv_60d", row.get("mvrv_usd_60d"))
    mvrv_60d_p90 = row.get("mvrv_60d_p90_180d")
    if mvrv_60d_raw is not None and mvrv_60d_p90 is not None:
        try:
            mvrv_val = float(mvrv_60d_raw)
            p90_val = float(mvrv_60d_p90)
            if mvrv_val == mvrv_val and p90_val == p90_val and mvrv_val > p90_val:
                vetoes.append("MVRV_ABOVE_P90_BAND")
        except (ValueError, TypeError):
            pass

    # Directional signal conflict
    if row.get("signal_option_call", 0) == 1:
        vetoes.append("OPTION_CALL_ACTIVE")

    # MDIA strong inflow
    if row.get("mdia_regime_strong_inflow", 0) == 1:
        vetoes.append("MDIA_STRONG_INFLOW")

    # MVRV macro bullish
    mvrv_bullish = (
        row.get("mvrv_ls_regime_call_confirm", 0) == 1
        or row.get("mvrv_ls_regime_call_confirm_recovery", 0) == 1
        or row.get("mvrv_ls_regime_call_confirm_trend", 0) == 1
    )
    if mvrv_bullish:
        vetoes.append("MVRV_MACRO_BULLISH")

    # Whale sponsored
    whale_sponsored = (
        row.get("whale_regime_broad_accum", 0) == 1
        or row.get("whale_regime_strategic_accum", 0) == 1
    )
    if whale_sponsored:
        vetoes.append("WHALE_SPONSORED")

    # Higher-priority signal (passed by caller)
    if higher_priority_active:
        vetoes.append("HIGHER_PRIORITY_SIGNAL")

    # ATR expansion
    if atr_ratio is not None and atr_ratio > config.atr_expansion_threshold:
        vetoes.append(
            f"ATR_EXPANSION({atr_ratio:.2f}>{config.atr_expansion_threshold})"
        )

    # Fusion state conflict
    if fusion_state in ("strong_bullish", "early_recovery", "momentum"):
        vetoes.append("FUSION_STATE_BULLISH")

    # Condor precedence
    if condor_eligible:
        vetoes.append("CONDOR_PRECEDENCE")

    return vetoes


# ============================================================
# OPTION CHAIN FILTERING & SPREAD SELECTION
# ============================================================


def filter_option_chain(
    chain_df: pd.DataFrame,
    side: str,
    spot_price: float,
    config: IncomeGateConfig,
    dte_mode: str = "tactical",
    strike_boundary: Optional[float] = None,
) -> pd.DataFrame:
    """
    Filter option chain for spread candidates.

    Pipeline:
        1. Side filter (put or call)
        2. OTM filter (puts < spot, calls > spot)
        3. MVRV boundary filter (if provided)
        4. Delta filter (min_delta <= abs(delta) <= max_delta)
        5. DTE filter (tactical or income window)
        6. Bid/ask filter (spread_pct <= max)
        7. Sort by abs(delta) descending (more credit first)

    Args:
        chain_df: Normalized option chain DataFrame.
        side: "put" or "call".
        spot_price: Current BTC spot price.
        config: Gate configuration.
        dte_mode: "tactical" (9-21d) or "income" (21-45d).
        strike_boundary: MVRV-derived boundary. For puts: keep <= boundary.
                         For calls: keep >= boundary.

    Returns:
        Filtered and sorted DataFrame of candidates.
    """
    df = chain_df.copy()

    # 1. Side filter
    df = df[df["side"] == side]

    # 2. OTM filter
    if side == "put":
        df = df[df["strike"] < spot_price]
    else:
        df = df[df["strike"] > spot_price]

    # 2b. Minimum OTM distance — reject strikes too close to spot
    min_otm_distance = spot_price * config.min_otm_pct
    if side == "put":
        df = df[df["strike"] <= spot_price - min_otm_distance]
    else:
        df = df[df["strike"] >= spot_price + min_otm_distance]

    # 3. MVRV boundary filter (hard — no fallback, reject if no candidates pass)
    if strike_boundary is not None and not df.empty:
        if side == "put":
            df = df[df["strike"] <= strike_boundary]
        else:
            df = df[df["strike"] >= strike_boundary]

    # 4. Delta filter
    df = df[
        (df["delta"] >= config.min_delta) & (df["delta"] <= config.max_delta)
    ]

    # 5. DTE filter
    if dte_mode == "tactical":
        min_dte = config.tactical_min_dte
        max_dte = config.tactical_max_dte
    else:
        min_dte = config.income_min_dte
        max_dte = config.income_max_dte
    df = df[(df["dte"] >= min_dte) & (df["dte"] <= max_dte)]

    # 6. Bid/ask filter
    if "spread_pct" in df.columns:
        df = df[df["spread_pct"] <= config.max_bid_ask_spread_pct]

    # 7. Sort by delta descending (higher delta = more credit)
    df = df.sort_values("delta", ascending=False)

    return df.reset_index(drop=True)


def select_spread(
    filtered_chain: pd.DataFrame,
    full_chain: pd.DataFrame,
    side: str,
    spot_price: float,
    config: IncomeGateConfig,
    mvrv_60d: Optional[float] = None,
    strike_boundary: Optional[float] = None,
) -> Optional[SpreadCandidate]:
    """
    Select the best spread from filtered chain candidates (single result).

    This is the backward-compatible interface. For multi-tier selection,
    use select_spread_tiers() instead.

    Returns:
        SpreadCandidate if a valid spread is found, None otherwise.
    """
    candidates = _find_all_valid_spreads(
        filtered_chain, full_chain, side, spot_price, config, mvrv_60d, strike_boundary
    )
    if not candidates:
        return None
    # Return the first (highest-delta = most credit) valid candidate
    return candidates[0]


# Delta boundaries for risk tiers
_TIER_DELTA_RANGES = {
    "low": (0.12, 0.20),    # Far OTM, less credit, higher POP
    "medium": (0.20, 0.28), # Middle ground
    "high": (0.28, 0.35),   # Closer to ATM, more credit, lower POP
}


def select_spread_tiers(
    filtered_chain: pd.DataFrame,
    full_chain: pd.DataFrame,
    side: str,
    spot_price: float,
    config: IncomeGateConfig,
    mvrv_60d: Optional[float] = None,
    strike_boundary: Optional[float] = None,
) -> list:
    """
    Select up to 3 spread setups across risk tiers (low/medium/high).

    Each tier targets a different delta range:
    - Low risk:    delta 0.12-0.20 (far OTM, ~80-88% POP, less credit)
    - Medium risk: delta 0.20-0.28 (balanced, ~72-80% POP)
    - High risk:   delta 0.28-0.35 (closer to ATM, ~65-72% POP, more credit)

    The human chooses which tier to trade based on their conviction and
    risk appetite. All returned setups pass the adaptive credit threshold.

    Returns:
        List[SpreadSetup] — 0 to 3 setups, one per tier that has a valid spread.
    """
    all_candidates = _find_all_valid_spreads(
        filtered_chain, full_chain, side, spot_price, config, mvrv_60d, strike_boundary
    )
    if not all_candidates:
        return []

    setups = []
    for tier_name, (d_min, d_max) in _TIER_DELTA_RANGES.items():
        # Find best candidate in this delta range
        tier_candidates = [
            c for c in all_candidates
            if d_min <= c.short_delta < d_max
        ]
        if not tier_candidates:
            continue
        # Pick highest credit ratio within the tier
        best = max(tier_candidates, key=lambda c: c.credit / c.spread_width)

        if side == "put":
            otm_pct = (spot_price - best.short_strike) / spot_price
        else:
            otm_pct = (best.short_strike - spot_price) / spot_price

        setups.append(SpreadSetup(
            risk_tier=tier_name,
            short_strike=best.short_strike,
            long_strike=best.long_strike,
            credit=best.credit,
            spread_width=best.spread_width,
            dte=best.dte,
            max_loss=best.max_loss,
            short_delta=best.short_delta,
            credit_width_pct=best.credit / best.spread_width,
            otm_pct=otm_pct,
            risk_reward=best.credit / best.max_loss if best.max_loss > 0 else 0.0,
        ))

    return setups


def _find_all_valid_spreads(
    filtered_chain: pd.DataFrame,
    full_chain: pd.DataFrame,
    side: str,
    spot_price: float,
    config: IncomeGateConfig,
    mvrv_60d: Optional[float] = None,
    strike_boundary: Optional[float] = None,
) -> list:
    """
    Find all valid spreads from the filtered chain that pass the credit threshold.

    Returns list of SpreadCandidate sorted by delta descending (highest credit first).
    """
    if filtered_chain.empty:
        return []

    min_width = spot_price * config.min_spread_width_pct
    max_width = spot_price * config.max_spread_width_pct

    # Long leg candidates: same side, from full chain
    side_chain = full_chain[full_chain["side"] == side].copy()
    side_chain["dte_int"] = side_chain["dte"].astype(int)

    candidates = []
    for _, short_row in filtered_chain.iterrows():
        short_strike = float(short_row["strike"])
        short_bid = float(short_row["bid"])
        short_dte = int(short_row["dte"])
        short_delta = float(short_row["delta"])

        # Find long leg: same side, same expiry, further OTM, within width range
        same_expiry = side_chain[side_chain["dte_int"] == short_dte]
        if side == "put":
            long_candidates = same_expiry[
                (same_expiry["strike"] < short_strike)
                & (same_expiry["strike"] >= short_strike - max_width)
                & (same_expiry["strike"] <= short_strike - min_width)
            ]
        else:
            long_candidates = same_expiry[
                (same_expiry["strike"] > short_strike)
                & (same_expiry["strike"] <= short_strike + max_width)
                & (same_expiry["strike"] >= short_strike + min_width)
            ]

        if long_candidates.empty:
            continue

        # Pick the long leg closest to min_width (tightest spread = best credit ratio)
        if side == "put":
            long_candidates = long_candidates.sort_values("strike", ascending=False)
        else:
            long_candidates = long_candidates.sort_values("strike", ascending=True)

        long_row = long_candidates.iloc[0]
        long_strike = float(long_row["strike"])
        long_ask = float(long_row["ask"])

        spread_width = abs(short_strike - long_strike)
        credit = short_bid - long_ask

        if credit <= 0 or spread_width <= 0:
            continue

        # Adaptive credit threshold
        required_credit_pct = _compute_required_credit_pct(
            side=side,
            short_strike=short_strike,
            short_delta=short_delta,
            spot_price=spot_price,
            mvrv_60d=mvrv_60d,
            strike_boundary=strike_boundary,
            config=config,
        )

        credit_ratio = credit / spread_width
        if credit_ratio < required_credit_pct:
            continue

        max_loss = spread_width - credit
        candidates.append(SpreadCandidate(
            short_strike=short_strike,
            long_strike=long_strike,
            credit=credit,
            spread_width=spread_width,
            dte=short_dte,
            max_loss=max_loss,
            short_delta=short_delta,
        ))

    return candidates


def _compute_required_credit_pct(
    side: str,
    short_strike: float,
    short_delta: float,
    spot_price: float,
    mvrv_60d: Optional[float],
    strike_boundary: Optional[float],
    config: IncomeGateConfig,
) -> float:
    """
    Compute minimum required credit/width ratio scaled to delta.

    A real trader demands more credit when closer to ATM (higher risk) and
    accepts less when further OTM (lower risk). The formula:

        required = short_delta × scalar

    Base scalar is config.credit_delta_scalar (0.55), which targets P50
    of observed BTC option spreads — selecting the better-than-median half.

    Quality adjustments reduce the scalar when the setup is safer:
    - MVRV is constructive (>1.03 for puts, strike >= boundary for calls): -0.05
    - Strike is well behind MVRV boundary (extra cushion): -0.02
    - Well OTM (>6% from spot): -0.03

    Never goes below config.credit_delta_scalar_floor (0.40) × delta.

    Examples at scalar 0.55:
        delta 0.15 → require ≥ 8.3% credit/width
        delta 0.20 → require ≥ 11.0%
        delta 0.25 → require ≥ 13.8%
        delta 0.30 → require ≥ 16.5%
        delta 0.35 → require ≥ 19.3%
    """
    scalar = config.credit_delta_scalar  # 0.55

    # Compute OTM distance
    if side == "put":
        otm_pct = (spot_price - short_strike) / spot_price
    else:
        otm_pct = (short_strike - spot_price) / spot_price

    # MVRV quality bonus — reduce scalar when on-chain anchor is strong
    if side == "put" and mvrv_60d is not None:
        if mvrv_60d >= config.mvrv_strong_floor_threshold:
            scalar -= 0.05
            # Additional: strike below MVRV floor = extra cushion
            if strike_boundary is not None and short_strike <= strike_boundary:
                scalar -= 0.02
    elif side == "call":
        if strike_boundary is not None and short_strike >= strike_boundary:
            scalar -= 0.05
            # Extra: strike well above ceiling
            if spot_price > 0:
                boundary_dist = (short_strike - strike_boundary) / spot_price
                if boundary_dist > 0.02:
                    scalar -= 0.02

    # Well OTM bonus (>6% from spot = more room to be wrong)
    if otm_pct > 0.06:
        scalar -= 0.03

    # Floor: never below absolute minimum scalar
    scalar = max(config.credit_delta_scalar_floor, scalar)

    # Required credit = delta × scalar
    return short_delta * scalar


# ============================================================
# EVALUATE FUNCTIONS
# ============================================================

# Default SpreadGuidance for income spreads
_BULL_PUT_GUIDANCE = SpreadGuidance(
    width_pct=0.05,
    take_profit_pct=0.50,
    max_hold_days=18,
    stop_loss_pct=0.02,
    scale_down_day=12,
)

_BEAR_CALL_GUIDANCE = SpreadGuidance(
    width_pct=0.05,
    take_profit_pct=0.50,
    max_hold_days=18,
    stop_loss_pct=0.02,
    scale_down_day=12,
)


def evaluate_bull_put_gate(
    row: pd.Series,
    chain_df: Optional[pd.DataFrame],
    spot_price: float,
    config: IncomeGateConfig = None,
    atr_ratio: Optional[float] = None,
    fusion_state: Optional[str] = None,
    higher_priority_active: bool = False,
    condor_eligible: bool = False,
    dte_mode: str = "tactical",
) -> IncomeGateResult:
    """
    Full bull put spread gate evaluation: score + vetoes + chain selection.

    Two-layer evaluation:
        1. Regime: scoring + vetoes (independent of chain)
        2. Chain: MVRV boundary + filtering + spread selection

    Args:
        row: Feature row (pd.Series).
        chain_df: Option chain DataFrame (may be None).
        spot_price: Current BTC spot price.
        config: Gate configuration.
        atr_ratio: ATR(7)/ATR(30) for vol expansion veto.
        fusion_state: Fusion state string.
        higher_priority_active: Whether a higher-priority signal is active.
        condor_eligible: Whether the condor gate passed.
        dte_mode: "tactical" or "income".

    Returns:
        IncomeGateResult with eligibility decision.
    """
    if config is None:
        config = IncomeGateConfig()

    # --- Layer 1: Regime ---
    score, components = compute_bull_put_score(row)
    vetoes = check_bull_put_vetoes(
        row, atr_ratio, fusion_state, higher_priority_active, condor_eligible, config
    )
    regime_eligible = (score >= config.score_threshold) and (len(vetoes) == 0)

    if not regime_eligible:
        return IncomeGateResult(
            score=score,
            regime_eligible=False,
            eligible=False,
            veto_reasons=vetoes,
            score_components=components,
            threshold=config.score_threshold,
            structure=OptionStructure.SHORT_PUT_SPREAD,
        )

    # --- Layer 2: Chain ---
    if chain_df is None or (hasattr(chain_df, "empty") and chain_df.empty):
        return IncomeGateResult(
            score=score,
            regime_eligible=True,
            eligible=False,
            veto_reasons=vetoes,
            score_components=components,
            threshold=config.score_threshold,
            structure=OptionStructure.SHORT_PUT_SPREAD,
            chain_valid=False,
            chain_rejection_reason="NO_CHAIN_DATA",
        )

    normalized = normalize_chain_columns(chain_df)
    if normalized is None or normalized.empty:
        return IncomeGateResult(
            score=score,
            regime_eligible=True,
            eligible=False,
            veto_reasons=vetoes,
            score_components=components,
            threshold=config.score_threshold,
            structure=OptionStructure.SHORT_PUT_SPREAD,
            chain_valid=False,
            chain_rejection_reason="MISSING_COLUMNS",
        )

    # Compute MVRV floor for bull put spread
    # Primary: cost_basis = spot / mvrv_60d (where recent buyers break even)
    # When mvrv_60d >= 1 (buyers profitable): cost_basis < spot → good floor
    # When mvrv_60d < 1 (buyers underwater): cost_basis > spot → use P10 fallback
    mvrv_60d = row.get("mvrv_60d", row.get("mvrv_usd_60d"))
    mvrv_60d_p10 = row.get("mvrv_60d_p10_180d")
    floor_price = None
    if mvrv_60d and float(mvrv_60d) > 0:
        cost_basis = spot_price / float(mvrv_60d)
        if cost_basis <= spot_price:
            # Normal case: buyers in profit, cost basis is below spot → use it
            floor_price = cost_basis
        else:
            # Underwater case: cost basis above spot → use P10 as tighter anchor
            if mvrv_60d_p10 and float(mvrv_60d_p10) > 0:
                floor_price = cost_basis * float(mvrv_60d_p10)
            else:
                # No P10 available, use cost basis anyway (hard boundary will reject if no strikes)
                floor_price = cost_basis

    # Filter chain
    filtered = filter_option_chain(
        normalized, "put", spot_price, config, dte_mode, strike_boundary=floor_price
    )

    if filtered.empty:
        return IncomeGateResult(
            score=score,
            regime_eligible=True,
            eligible=False,
            veto_reasons=vetoes,
            score_components=components,
            threshold=config.score_threshold,
            structure=OptionStructure.SHORT_PUT_SPREAD,
            chain_valid=False,
            chain_rejection_reason="NO_CONTRACTS_PASS_FILTERS",
        )

    # Select spread tiers (low/medium/high risk)
    setups = select_spread_tiers(
        filtered, normalized, "put", spot_price, config,
        mvrv_60d=float(mvrv_60d) if mvrv_60d else None,
        strike_boundary=floor_price,
    )

    if not setups:
        return IncomeGateResult(
            score=score,
            regime_eligible=True,
            eligible=False,
            veto_reasons=vetoes,
            score_components=components,
            threshold=config.score_threshold,
            structure=OptionStructure.SHORT_PUT_SPREAD,
            chain_valid=False,
            chain_rejection_reason="NO_ACCEPTABLE_SPREAD",
        )

    # Use medium-risk as the default (fallback to first available)
    default_setup = next(
        (s for s in setups if s.risk_tier == "medium"),
        setups[0]
    )

    return IncomeGateResult(
        score=score,
        regime_eligible=True,
        eligible=True,
        veto_reasons=vetoes,
        score_components=components,
        threshold=config.score_threshold,
        structure=OptionStructure.SHORT_PUT_SPREAD,
        chain_valid=True,
        spread_guidance=_BULL_PUT_GUIDANCE,
        setups=setups,
        short_strike=default_setup.short_strike,
        long_strike=default_setup.long_strike,
        credit=default_setup.credit,
        dte=default_setup.dte,
        max_loss=default_setup.max_loss,
    )


def compute_bear_call_target_mvrv(
    mvrv_series: pd.Series,
    config: IncomeGateConfig = None,
) -> Optional[float]:
    """
    Data-driven ceiling target for bear call spreads in underwater regimes.

    Problem this solves
    --------------------
    The static macro ceiling (``cost_basis × mvrv_60d_p90_180d``) projects the
    short call strike unreachably far OTM when holders are underwater
    (``mvrv_60d < 1``): cost_basis sits well above spot, so the P90 multiplier
    pushes the ceiling ~20%+ OTM — beyond any strike that still carries sellable
    delta. ``filter_option_chain`` then returns nothing and the gate emits
    NO_TRADE (observed on 2026-06-18 and 2026-06-25).

    Approach
    --------
    Estimate a *reachable* ceiling from the recent behaviour of mvrv_60d itself:
    over the trailing window, how far does mvrv_60d rally within the option's
    holding horizon? Take a percentile of those forward rallies as the projected
    upside and scale the current mvrv_60d by it::

        target_mvrv = current_mvrv × (1 + pct)

    Because ``cost_basis = spot / current_mvrv``, the downstream ceiling
    (``cost_basis × target_mvrv``) reduces exactly to ``spot × (1 + pct)``: the
    current_mvrv level cancels, so the ceiling's OTM distance equals the
    percentile forward MVRV rally. The MVRV-native form is kept for
    interpretability and parity with the profitable-regime path.

    Anti-leakage
    ------------
    Only starting points whose *entire* forward ``[hmin, hmax]`` window lies
    within the observed series are used, so the live computation matches a
    properly lagged backtest (the most recent ~hmax days are never used as
    starting points).

    Modeling notes / tradeoffs
    --------------------------
    - Multiplicative scaling: a depressed current_mvrv produces a proportional
      (not fixed) absolute rebound. Intentional — keeps the ceiling conservative
      when valuations are deeply compressed.
    - Trailing-window assumption: a fresh panic or violent post-capitulation
      rebound can leave the trailing distribution stale in either direction.
    - Percentile is a hypothesis (default P60), not a tuned optimum. It should
      be calibrated against realized breach-rate / credit tradeoff over history,
      not by how many contracts survive on any single day.

    Args:
        mvrv_series: Trailing mvrv_60d values up to and including the signal
            date, in chronological order. The last value is treated as current.
        config: Gate configuration (percentile, lookback, horizon knobs).

    Returns:
        target_mvrv, or None if data is insufficient — in which case the caller
        should fall back to the static P90 ceiling.
    """
    if config is None:
        config = IncomeGateConfig()

    if mvrv_series is None:
        return None

    vals = pd.to_numeric(mvrv_series, errors="coerce").dropna().to_numpy()
    if vals.size == 0:
        return None

    current_mvrv = float(vals[-1])
    if current_mvrv <= 0:
        return None

    lookback = int(config.bear_call_ceiling_lookback)
    hmin = int(config.bear_call_rally_hmin)
    hmax = int(config.bear_call_rally_hmax)

    # Keep the trailing lookback plus the forward span needed to fully observe
    # windows for the most recent eligible starting points.
    if vals.size > (lookback + hmax):
        window = vals[-(lookback + hmax):]
    else:
        window = vals
    n = window.size

    # Forward rally for each starting point with a fully-observed [hmin, hmax]
    # window. last_start is inclusive and guarantees t + hmax <= n - 1.
    rallies = []
    last_start = n - hmax - 1
    for t in range(0, last_start + 1):
        base = window[t]
        if not (base > 0):
            continue
        fwd_max = window[t + hmin : t + hmax + 1].max()
        rallies.append(fwd_max / base - 1.0)

    MIN_SAMPLES = 30
    if len(rallies) < MIN_SAMPLES:
        return None

    pct = float(np.percentile(rallies, config.bear_call_ceiling_percentile * 100.0))
    # A downtrend-only window can yield a negative percentile, which would pull
    # the ceiling below spot — nonsensical for a resistance. Floor at 0 so the
    # ceiling is at least spot (delta + min_otm filters still apply downstream).
    pct = max(pct, 0.0)
    return current_mvrv * (1.0 + pct)


def evaluate_bear_call_gate(
    row: pd.Series,
    chain_df: Optional[pd.DataFrame],
    spot_price: float,
    config: IncomeGateConfig = None,
    atr_ratio: Optional[float] = None,
    fusion_state: Optional[str] = None,
    higher_priority_active: bool = False,
    condor_eligible: bool = False,
    dte_mode: str = "tactical",
    target_mvrv: Optional[float] = None,
) -> IncomeGateResult:
    """
    Full bear call spread gate evaluation: score + vetoes + chain selection.

    Two-layer evaluation:
        1. Regime: scoring + vetoes (independent of chain)
        2. Chain: MVRV boundary + filtering + spread selection

    Args:
        row: Feature row (pd.Series).
        chain_df: Option chain DataFrame (may be None).
        spot_price: Current BTC spot price.
        config: Gate configuration.
        atr_ratio: ATR(7)/ATR(30) for vol expansion veto.
        fusion_state: Fusion state string.
        higher_priority_active: Whether a higher-priority signal is active.
        condor_eligible: Whether the condor gate passed.
        dte_mode: "tactical" or "income".
        target_mvrv: Data-driven MVRV ceiling target for the underwater regime
            (from compute_bear_call_target_mvrv). When provided and mvrv_60d < 1,
            it replaces the static p90_180d ceiling, which projects unreachably
            far OTM when holders are underwater. None → static P90 fallback.

    Returns:
        IncomeGateResult with eligibility decision.
    """
    if config is None:
        config = IncomeGateConfig()

    # --- Layer 1: Regime ---
    score, components = compute_bear_call_score(row)
    vetoes = check_bear_call_vetoes(
        row, atr_ratio, fusion_state, higher_priority_active, condor_eligible, config
    )
    regime_eligible = (score >= config.score_threshold) and (len(vetoes) == 0)

    if not regime_eligible:
        return IncomeGateResult(
            score=score,
            regime_eligible=False,
            eligible=False,
            veto_reasons=vetoes,
            score_components=components,
            threshold=config.score_threshold,
            structure=OptionStructure.SHORT_CALL_SPREAD,
        )

    # --- Layer 2: Chain ---
    if chain_df is None or (hasattr(chain_df, "empty") and chain_df.empty):
        return IncomeGateResult(
            score=score,
            regime_eligible=True,
            eligible=False,
            veto_reasons=vetoes,
            score_components=components,
            threshold=config.score_threshold,
            structure=OptionStructure.SHORT_CALL_SPREAD,
            chain_valid=False,
            chain_rejection_reason="NO_CHAIN_DATA",
        )

    normalized = normalize_chain_columns(chain_df)
    if normalized is None or normalized.empty:
        return IncomeGateResult(
            score=score,
            regime_eligible=True,
            eligible=False,
            veto_reasons=vetoes,
            score_components=components,
            threshold=config.score_threshold,
            structure=OptionStructure.SHORT_CALL_SPREAD,
            chain_valid=False,
            chain_rejection_reason="MISSING_COLUMNS",
        )

    # Compute MVRV ceiling for bear call spread.
    #
    # Underwater (mvrv_60d < 1): the static cost_basis × p90_180d ceiling projects
    # the short call ~20%+ OTM (cost_basis sits far above spot), beyond any strike
    # with sellable delta. When a data-driven target_mvrv is supplied, use it —
    # it reflects how far mvrv_60d actually rallies over the option horizon in the
    # trailing window, yielding a reachable ceiling. See compute_bear_call_target_mvrv.
    #
    # Profitable (mvrv_60d >= 1): cost_basis < spot, the P90 ceiling is already
    # meaningful and reachable — keep it unchanged.
    mvrv_60d = row.get("mvrv_60d", row.get("mvrv_usd_60d"))
    mvrv_60d_p90 = row.get("mvrv_60d_p90_180d")
    mvrv_composite_pct = row.get("mvrv_composite_pct")
    mvrv_comp_max_180d = row.get("mvrv_comp_max_180d")

    ceiling_price = None
    if mvrv_60d and float(mvrv_60d) > 0:
        cost_basis = spot_price / float(mvrv_60d)
        if float(mvrv_60d) < 1.0 and target_mvrv is not None and float(target_mvrv) > 0:
            # Underwater + data-driven target available: reachable ceiling.
            ceiling_price = cost_basis * float(target_mvrv)
        elif mvrv_60d_p90 and float(mvrv_60d_p90) > 0:
            # P90: where price would be if MVRV-60D hits its 6-month 90th percentile
            ceiling_price = cost_basis * float(mvrv_60d_p90)
        elif mvrv_composite_pct is not None and mvrv_comp_max_180d is not None:
            # Fallback: composite max (pct format)
            try:
                comp_pct = float(mvrv_composite_pct)
                comp_max = float(mvrv_comp_max_180d)
                raw_composite = 1.0 + comp_pct / 100.0
                raw_max = 1.0 + comp_max / 100.0
                if raw_composite > 0 and raw_max > 0:
                    ceiling_price = (spot_price / raw_composite) * raw_max
            except (ValueError, TypeError):
                pass

    # Filter chain
    filtered = filter_option_chain(
        normalized, "call", spot_price, config, dte_mode, strike_boundary=ceiling_price
    )

    if filtered.empty:
        return IncomeGateResult(
            score=score,
            regime_eligible=True,
            eligible=False,
            veto_reasons=vetoes,
            score_components=components,
            threshold=config.score_threshold,
            structure=OptionStructure.SHORT_CALL_SPREAD,
            chain_valid=False,
            chain_rejection_reason="NO_CONTRACTS_PASS_FILTERS",
        )

    # Select spread tiers (low/medium/high risk)
    setups = select_spread_tiers(
        filtered, normalized, "call", spot_price, config,
        mvrv_60d=None,  # not used for calls
        strike_boundary=ceiling_price,
    )

    if not setups:
        return IncomeGateResult(
            score=score,
            regime_eligible=True,
            eligible=False,
            veto_reasons=vetoes,
            score_components=components,
            threshold=config.score_threshold,
            structure=OptionStructure.SHORT_CALL_SPREAD,
            chain_valid=False,
            chain_rejection_reason="NO_ACCEPTABLE_SPREAD",
        )

    # Use medium-risk as the default (fallback to first available)
    default_setup = next(
        (s for s in setups if s.risk_tier == "medium"),
        setups[0]
    )

    return IncomeGateResult(
        score=score,
        regime_eligible=True,
        eligible=True,
        veto_reasons=vetoes,
        score_components=components,
        threshold=config.score_threshold,
        structure=OptionStructure.SHORT_CALL_SPREAD,
        chain_valid=True,
        spread_guidance=_BEAR_CALL_GUIDANCE,
        setups=setups,
        short_strike=default_setup.short_strike,
        long_strike=default_setup.long_strike,
        credit=default_setup.credit,
        dte=default_setup.dte,
        max_loss=default_setup.max_loss,
    )
