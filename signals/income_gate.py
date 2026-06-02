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
    min_delta: float = 0.15
    max_delta: float = 0.30
    max_bid_ask_spread_pct: float = 0.15
    min_credit_pct: float = 0.25  # credit / spread_width minimum
    min_spread_width_pct: float = 0.03  # spread_width / spot minimum
    max_spread_width_pct: float = 0.08  # spread_width / spot maximum

    # DTE windows
    tactical_min_dte: int = 9
    tactical_max_dte: int = 21
    income_min_dte: int = 21
    income_max_dte: int = 45

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
    mvrv_composite: Optional[float],
    mvrv_composite_p90_180d: Optional[float],
) -> tuple[Optional[float], Optional[float]]:
    """
    Compute MVRV-derived price boundaries for strike selection.

    Returns:
        (floor_price, ceiling_price)
        - floor_price: bull put short strikes must be at or below this
        - ceiling_price: bear call short strikes must be at or above this
        Either may be None if MVRV data is unavailable.
    """
    floor_price = None
    ceiling_price = None

    if mvrv_60d and mvrv_60d > 0:
        floor_price = spot_price / mvrv_60d

    if mvrv_composite and mvrv_composite > 0 and mvrv_composite_p90_180d:
        cost_basis_composite = spot_price / mvrv_composite
        ceiling_price = cost_basis_composite * mvrv_composite_p90_180d

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

    # 3. MVRV boundary filter (soft — fall back if eliminates all)
    if strike_boundary is not None and not df.empty:
        if side == "put":
            boundary_filtered = df[df["strike"] <= strike_boundary]
        else:
            boundary_filtered = df[df["strike"] >= strike_boundary]
        # Only apply if it doesn't eliminate everything
        if not boundary_filtered.empty:
            df = boundary_filtered

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
) -> Optional[SpreadCandidate]:
    """
    Select the best spread from filtered chain candidates.

    Iterates short leg candidates from filtered_chain, finds a matching
    long leg from full_chain at the correct width, and validates
    credit/spread_width ratio.

    Args:
        filtered_chain: Pre-filtered chain for short leg candidates.
        full_chain: Full normalized chain for long leg lookup.
        side: "put" or "call".
        spot_price: Current BTC spot price.
        config: Gate configuration.

    Returns:
        SpreadCandidate if a valid spread is found, None otherwise.
    """
    if filtered_chain.empty:
        return None

    min_width = spot_price * config.min_spread_width_pct
    max_width = spot_price * config.max_spread_width_pct

    # Long leg candidates: same side, from full chain
    side_chain = full_chain[full_chain["side"] == side]

    for _, short_row in filtered_chain.iterrows():
        short_strike = float(short_row["strike"])
        short_bid = float(short_row["bid"])
        short_dte = int(short_row["dte"])
        short_delta = float(short_row["delta"])

        # Find long leg: same side, same expiry, further OTM, within width range
        same_expiry = side_chain[side_chain["dte"] == short_dte]
        if side == "put":
            # Long put is below short put
            long_candidates = same_expiry[
                (same_expiry["strike"] < short_strike)
                & (same_expiry["strike"] >= short_strike - max_width)
                & (same_expiry["strike"] <= short_strike - min_width)
            ]
        else:
            # Long call is above short call
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

        # Compute spread metrics
        spread_width = abs(short_strike - long_strike)
        credit = short_bid - long_ask

        if credit <= 0:
            continue

        if spread_width <= 0:
            continue

        credit_ratio = credit / spread_width
        if credit_ratio < config.min_credit_pct:
            continue

        max_loss = spread_width - credit

        return SpreadCandidate(
            short_strike=short_strike,
            long_strike=long_strike,
            credit=credit,
            spread_width=spread_width,
            dte=short_dte,
            max_loss=max_loss,
            short_delta=short_delta,
        )

    return None


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

    # Compute MVRV floor
    mvrv_60d = row.get("mvrv_60d", row.get("mvrv_usd_60d"))
    floor_price = None
    if mvrv_60d and float(mvrv_60d) > 0:
        floor_price = spot_price / float(mvrv_60d)

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

    # Select spread
    candidate = select_spread(filtered, normalized, "put", spot_price, config)

    if candidate is None:
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
        short_strike=candidate.short_strike,
        long_strike=candidate.long_strike,
        credit=candidate.credit,
        dte=candidate.dte,
        max_loss=candidate.max_loss,
    )


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

    # Compute MVRV ceiling
    mvrv_composite = row.get("mvrv_composite")
    mvrv_composite_p90 = row.get("mvrv_composite_p90_180d")
    ceiling_price = None
    if mvrv_composite and float(mvrv_composite) > 0 and mvrv_composite_p90:
        cost_basis = spot_price / float(mvrv_composite)
        ceiling_price = cost_basis * float(mvrv_composite_p90)

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

    # Select spread
    candidate = select_spread(filtered, normalized, "call", spot_price, config)

    if candidate is None:
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
        short_strike=candidate.short_strike,
        long_strike=candidate.long_strike,
        credit=candidate.credit,
        dte=candidate.dte,
        max_loss=candidate.max_loss,
    )
