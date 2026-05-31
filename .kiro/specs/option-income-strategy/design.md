# Technical Design Document: Option Income Strategy Module

## Overview

The Option Income Strategy module provides a comprehensive framework for income-generating BTC option strategies on **Deribit Linear USDC Options only** (Phase 1). The module has **two primary functions**:

1. **Signal Generation**: Generate income strategy signals (CSP_ACCUMULATION, BULL_PUT_SPREAD, BEAR_CALL_SPREAD) using existing metrics to expand signal variety beyond directional-only signals
2. **Safe Execution**: Validate, execute atomically, and track income strategy positions with strict safety constraints

### Key Design Constraints

1. **Deribit Linear USDC Only**: All options are European-style, cash-settled in USDC. No physical BTC delivery, no early assignment.
2. **Atomic or Sequential-Safe Execution**: Multi-leg strategies require atomic combo orders OR long-leg-first sequential execution with verification.
3. **Exchange-Verified Collateral**: Execution requires authenticated exchange state, not user-supplied values.
4. **No Inverse BTC Options**: Phase 1 excludes inverse options due to BTC collateral convexity risk.
5. **Integration with Existing Signals**: Reuses existing metrics (MVRV, IV rank, fusion state) and signal infrastructure.

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Option Income Strategy Module                           │
│                    (Deribit Linear USDC Options Only)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │              IncomeSignalGenerator (NEW)                          │       │
│  │  Generates: CSP_ACCUMULATION, BULL_PUT_SPREAD, BEAR_CALL_SPREAD   │       │
│  │  Uses: fusion_state, iv_rank, mvrv_60d, signal_option_*           │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                   │                                          │
│                                   ▼                                          │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐       │
│  │ StrategyClassifier│    │  SafetyValidator │    │  PayoffCalculator│       │
│  │                  │    │  (Enhanced)      │    │  (USDC-based)    │       │
│  └────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘       │
│           │                       │                       │                  │
│           ▼                       ▼                       ▼                  │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                    IncomeStrategyEngine                           │       │
│  │  (orchestrates classification, validation, calculation)           │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                   │                                          │
│           ┌───────────────────────┼───────────────────────┐                  │
│           ▼                       ▼                       ▼                  │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐       │
│  │   RiskEngine     │    │RecommendationEngine│   │ CSPIncomeTracker │       │
│  │                  │    │                  │    │                  │       │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘       │
│                                   │                                          │
│                                   ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │              AtomicExecutionController                            │       │
│  │  (enforces long-leg-first, partial fill recovery)                 │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Integration Layer                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  SignalService            │  Extended with evaluate_income_signals()        │
│  TradeSetupBuilder        │  Extends for income strategy setup              │
│  TradeValidator           │  Adds income-specific validation rules          │
│  DeribitExecutor          │  Extended for atomic/sequential-safe execution  │
│  PolicyVersion            │  Adds income strategy policy parameters         │
│  OptionSnapshot           │  Reuses existing option data model              │
│  DailySignal              │  Extended with income signal fields             │
│  API Views                │  New endpoints (simulation vs verified modes)   │
└─────────────────────────────────────────────────────────────────────────────┘
```


### Component Interaction Flow

```
Signal Generation Flow (NEW):
1. SignalService.generate_signal() called (existing)
         │
         ▼
2. IncomeSignalGenerator.evaluate(row, fusion_result, iv_rank)
   ├── Check CSP_ACCUMULATION conditions
   ├── Check BULL_PUT_SPREAD conditions
   ├── Check BEAR_CALL_SPREAD conditions
   ├── Apply cooldowns
   └── Returns IncomeSignalResult (signal type, rationale, strike guidance)
         │
         ▼
3. IF income signal active:
   Persist to DailySignal with trade_decision = signal type
         │
         ▼
4. Execution flow (when user acts on signal):
   IncomeStrategyEngine.build_setup(signal, account_state)
   → Full validation, payoff calculation, risk assessment

Execution Flow (existing, enhanced):
1. User Request (strategy type + parameters)
         │
         ▼
2. StrategyClassifier.classify(strategy_type)
   → Returns StrategyProfile with category, requirements, risk attributes
         │
         ▼
3. SafetyValidator.validate(strategy, account_state)
   ├── Check against prohibited strategies (BLOCKING)
   ├── Verify collateral sufficiency (BLOCKING)
   ├── Verify protective legs present (BLOCKING)
   └── Returns ValidationResult (pass/fail + rejection details)
         │
         ▼
4. IF validation passes:
   PayoffCalculator.calculate(strategy, option_data)
   → Returns PayoffMetrics (max_profit, max_loss, breakevens, etc.)
         │
         ▼
5. RiskEngine.assess(strategy, payoff, account_state)
   ├── Check position-level limits
   ├── Check portfolio-level limits
   ├── Calculate risk metrics
   └── Returns RiskAssessment (approved/blocked + warnings)
         │
         ▼
6. RecommendationEngine.recommend(market_regime, account_state)
   → Returns ranked list of suitable strategies for current conditions
         │
         ▼
7. IncomeStrategySetup created and returned
   → Includes legs, metrics, validation, risk assessment, exit rules
```

## Components

### 0. Income Signal Generator (NEW)

Generates income strategy signals using **5-axis regime classification**. The strategy itself matters less than the regime detection.

#### Core Principle

Premium selling performance is dominated by:
1. **Volatility regime** - IV richness AND IV vs RV spread
2. **Trend vs chop** - Most important for condors and spreads
3. **Expansion vs compression** - Avoid selling into breakouts
4. **Macro valuation** - Accumulation desirability

#### 5-Axis Regime Classification

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class DirectionState(Enum):
    STRONG_BULLISH = "strong_bullish"
    MILD_BULLISH = "mild_bullish"
    NEUTRAL = "neutral"
    MILD_BEARISH = "mild_bearish"
    STRONG_BEARISH = "strong_bearish"


class IVRegime(Enum):
    LOW = "low"           # IV rank < 30
    MODERATE = "moderate" # IV rank 30-50
    ELEVATED = "elevated" # IV rank 50-70
    EXTREME = "extreme"   # IV rank > 70


class TrendRegime(Enum):
    STRONG_TREND = "strong_trend"     # Momentum persistent - AVOID premium selling
    MILD_TREND = "mild_trend"         # Drift with pullbacks - directional spreads only
    CHOP = "chop"                     # Range behavior - ideal for condors
    COMPRESSION = "compression"       # Narrowing range - expansion risk coming
    EXPANSION = "expansion"           # Breakout - AVOID premium selling


class ValuationState(Enum):
    DEEP_VALUE = "deep_value"         # MVRV < 0.90
    UNDERVALUED = "undervalued"       # MVRV 0.90-0.98
    FAIR = "fair"                     # MVRV 0.98-1.05
    OVERVALUED = "overvalued"         # MVRV 1.05-1.15
    EXTREME = "extreme"               # MVRV > 1.15


@dataclass
class RegimeAxis:
    """Single axis of regime classification with confidence."""
    state: str
    confidence: float  # 0.0-1.0


@dataclass
class RegimeClassification:
    """Complete 5-axis regime classification."""
    direction: RegimeAxis
    iv_regime: RegimeAxis
    iv_rv_favorable: RegimeAxis  # IV > RV check
    trend_regime: RegimeAxis
    valuation: RegimeAxis
    
    # Computed metrics
    iv_rank: float
    iv_current: float
    realized_vol_7d: float
    iv_rv_ratio: float
    atr_ratio: float  # 7d ATR / 30d ATR
```

#### IV vs Realized Volatility (CRITICAL)

This is the most important missing piece from the original design.

```python
def compute_iv_rv_metrics(
    iv_current: float,
    realized_vol_7d: float,
) -> tuple[float, float, bool]:
    """
    Compute IV vs RV spread and ratio.
    
    Returns:
        iv_rv_spread: IV - RV (positive = IV overpriced)
        iv_rv_ratio: IV / RV (>1.2 = favorable for selling)
        is_favorable: True if IV > RV by sufficient margin
    """
    if realized_vol_7d <= 0:
        return 0.0, 0.0, False
    
    iv_rv_spread = iv_current - realized_vol_7d
    iv_rv_ratio = iv_current / realized_vol_7d
    
    # Require IV at least 20% above realized for premium selling
    is_favorable = iv_rv_ratio > 1.2
    
    return iv_rv_spread, iv_rv_ratio, is_favorable


# Example scenarios:
# IV 60%, RV 35% → ratio 1.71 → EXCELLENT for selling
# IV 60%, RV 55% → ratio 1.09 → MARGINAL, avoid
# IV 60%, RV 80% → ratio 0.75 → DANGEROUS, do not sell
```

#### Trend vs Chop Detection

```python
def classify_trend_regime(
    fusion_state: str,
    atr_ratio: float,
    mvrv_ls_neutral: bool,
    sent_is_flattening: bool,
    gap_pct: float,
) -> RegimeAxis:
    """
    Classify trend regime - most important for premium selling.
    
    Args:
        fusion_state: From fusion engine
        atr_ratio: 7d ATR / 30d ATR (>1.3 = expansion, <0.7 = compression)
        mvrv_ls_neutral: MVRV LS level is neutral
        sent_is_flattening: Sentiment is flattening
        gap_pct: |open - prev_close| / prev_close
    """
    # Expansion detection (AVOID premium selling)
    if atr_ratio > 1.5 or gap_pct > 0.04:
        return RegimeAxis(state=TrendRegime.EXPANSION.value, confidence=0.90)
    
    if atr_ratio > 1.3:
        return RegimeAxis(state=TrendRegime.EXPANSION.value, confidence=0.75)
    
    # Strong trend detection (AVOID premium selling)
    strong_trend_states = {
        "strong_bullish", "bear_continuation", "bear_continuation_short",
        "momentum", "early_recovery"
    }
    if fusion_state in strong_trend_states:
        return RegimeAxis(state=TrendRegime.STRONG_TREND.value, confidence=0.80)
    
    # Compression detection (caution - expansion coming)
    if atr_ratio < 0.7 and sent_is_flattening:
        return RegimeAxis(state=TrendRegime.COMPRESSION.value, confidence=0.75)
    
    # Chop detection (IDEAL for condors)
    chop_states = {"no_trade", "transition_chop"}
    if fusion_state in chop_states and mvrv_ls_neutral:
        return RegimeAxis(state=TrendRegime.CHOP.value, confidence=0.85)
    
    if fusion_state in chop_states:
        return RegimeAxis(state=TrendRegime.CHOP.value, confidence=0.70)
    
    # Mild trend (directional spreads OK)
    return RegimeAxis(state=TrendRegime.MILD_TREND.value, confidence=0.65)
```

#### Strategy Selection Logic

```python
# Simplified selector based on regime combination
REGIME_STRATEGY_MAP = {
    # Undervalued + panic IV + capitulation → CSP
    ("undervalued", "elevated", True, "chop"): "CSP_ACCUMULATION",
    ("deep_value", "extreme", True, "chop"): "CSP_ACCUMULATION",
    ("deep_value", "elevated", True, "mild_trend"): "CSP_ACCUMULATION",
    
    # Mild bullish + rich IV + IV > RV → Bull put spread
    ("fair", "elevated", True, "chop"): "BULL_PUT_SPREAD",
    ("fair", "elevated", True, "mild_trend"): "BULL_PUT_SPREAD",
    
    # Mild bearish + rich IV + IV > RV → Bear call spread
    ("overvalued", "elevated", True, "chop"): "BEAR_CALL_SPREAD",
    ("overvalued", "elevated", True, "mild_trend"): "BEAR_CALL_SPREAD",
    
    # Chop + rich IV + low RV → Iron condor (handled by condor_gate.py)
    ("fair", "elevated", True, "chop"): "IRON_CONDOR",
}

# CRITICAL: These regimes BLOCK all premium selling
PREMIUM_SELLING_BLOCKED = {
    TrendRegime.STRONG_TREND,
    TrendRegime.EXPANSION,
}
```

#### Global Vetoes

```python
def check_global_vetoes(
    regime: RegimeClassification,
    signal_option_call: int,
    signal_option_put: int,
    mvrv_short_active: bool,
) -> list[str]:
    """
    Check global vetoes that block ALL income signals.
    
    Returns list of veto reasons (empty = no vetoes).
    """
    vetoes = []
    
    # Veto 1: Strong trend detected
    if regime.trend_regime.state == TrendRegime.STRONG_TREND.value:
        vetoes.append(f"Strong trend detected - trend will run you over")
    
    # Veto 2: Expansion regime
    if regime.trend_regime.state == TrendRegime.EXPANSION.value:
        vetoes.append(f"Expansion regime - breakout will destroy spreads")
    
    # Veto 3: IV < RV (realized vol exceeding implied)
    if regime.iv_rv_ratio < 1.0:
        vetoes.append(f"IV < RV ({regime.iv_rv_ratio:.2f}) - realized vol exceeding implied")
    
    # Veto 4: Existing directional signal active
    if signal_option_call == 1:
        vetoes.append("OPTION_CALL signal active")
    if signal_option_put == 1:
        vetoes.append("OPTION_PUT signal active")
    if mvrv_short_active:
        vetoes.append("MVRV_SHORT signal active")
    
    # Veto 5: IV too low
    if regime.iv_rank < 0.30:
        vetoes.append(f"IV rank too low ({regime.iv_rank:.2f}) - premium not worth risk")
    
    return vetoes
```

See `signals/income_signals.py` for full implementation.

### 1. Strategy Classifier

Classifies and profiles option income strategies.

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class StrategyCategory(Enum):
    """Primary strategy categories."""
    INCOME = "income"
    ACCUMULATION = "accumulation"
    DIRECTIONAL = "directional"


class StrategyType(Enum):
    """Supported income strategy types (Deribit Linear USDC only)."""
    CASH_SECURED_PUT = "cash_secured_put"
    # SYNTHETIC_COVERED_CALL removed - naked short call has undefined loss
    BULL_PUT_SPREAD = "bull_put_spread"
    BEAR_CALL_SPREAD = "bear_call_spread"  # Use this instead of naked covered calls
    IRON_CONDOR = "iron_condor"
    PUT_DEBIT_SPREAD = "put_debit_spread"
    CALL_DEBIT_SPREAD = "call_debit_spread"


class CollateralType(Enum):
    """Type of collateral required (Phase 1: USDC only)."""
    USDC = "usdc"           # USDC collateral (Phase 1)
    # BTC = "btc"           # Future: inverse options
    # SPREAD_MARGIN = "spread_margin"  # Future: portfolio margin


class SettlementType(Enum):
    """Settlement type for options."""
    CASH_USDC = "cash_usdc"     # Cash-settled in USDC (Phase 1)
    # CASH_BTC = "cash_btc"     # Future: inverse options
    # PHYSICAL = "physical"      # NOT supported on Deribit


class ExerciseStyle(Enum):
    """Option exercise style."""
    EUROPEAN = "european"       # All Deribit options
    # AMERICAN = "american"     # NOT supported


@dataclass
class StrategyProfile:
    """
    Complete profile of an income strategy.
    
    Defines the strategy's characteristics, requirements, and risk attributes.
    All strategies in Phase 1 are Deribit Linear USDC options only.
    """
    strategy_type: StrategyType
    category: StrategyCategory
    
    # Strategy attributes
    is_income: bool              # Generates premium income
    is_accumulation: bool        # Designed to synthetically accumulate BTC exposure
    is_defined_risk: bool        # Max loss known at entry (MUST be True AND verified)
    is_fully_collateralized: bool  # Collateral covers max loss without liquidation
    is_credit: bool              # Receives net premium
    
    # Requirements (Phase 1: USDC only)
    requires_usdc_collateral: bool = True  # Always True for Phase 1
    collateral_type: CollateralType = CollateralType.USDC
    settlement_type: SettlementType = SettlementType.CASH_USDC
    exercise_style: ExerciseStyle = ExerciseStyle.EUROPEAN
    num_legs: int = 1            # Number of option legs
    
    # Risk attributes (European options - no early assignment)
    has_settlement_risk: bool    # Can settle ITM at expiry
    has_early_assignment_risk: bool = False  # Always False for European options
    max_loss_formula: str        # Formula for max loss calculation (USDC)
    max_profit_formula: str      # Formula for max profit calculation (USDC)


class StrategyClassifier:
    """
    Classifies and profiles option income strategies.
    
    Phase 1: Deribit Linear USDC options only.
    All strategies are European-style, cash-settled in USDC.
    """
    
    # Strategy profiles registry
    PROFILES: dict[StrategyType, StrategyProfile] = {
        StrategyType.CASH_SECURED_PUT: StrategyProfile(
            strategy_type=StrategyType.CASH_SECURED_PUT,
            category=StrategyCategory.ACCUMULATION,
            is_income=True,
            is_accumulation=True,
            is_defined_risk=True,
            is_fully_collateralized=True,
            is_credit=True,
            num_legs=1,
            has_settlement_risk=True,
            max_loss_formula="(strike * contract_size * qty) - premium_usdc",
            max_profit_formula="premium_usdc",
        ),
        # SYNTHETIC_COVERED_CALL removed - naked short call has undefined upside loss
        # Use BEAR_CALL_SPREAD instead for call-side income with defined risk
        StrategyType.BULL_PUT_SPREAD: StrategyProfile(
            strategy_type=StrategyType.BULL_PUT_SPREAD,
            category=StrategyCategory.INCOME,
            is_income=True,
            is_accumulation=False,
            is_defined_risk=True,
            is_fully_collateralized=True,
            is_credit=True,
            num_legs=2,
            has_settlement_risk=True,
            max_loss_formula="(spread_width * contract_size * qty) - net_credit_usdc",
            max_profit_formula="net_credit_usdc",
        ),
        # ... additional profiles defined similarly
    }
    
    @classmethod
    def classify(cls, strategy_type: StrategyType) -> StrategyProfile:
        """Get the profile for a strategy type."""
        if strategy_type not in cls.PROFILES:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        return cls.PROFILES[strategy_type]
    
    @classmethod
    def get_income_strategies(cls) -> list[StrategyProfile]:
        """Get all income-generating strategies."""
        return [p for p in cls.PROFILES.values() if p.is_income]
    
    @classmethod
    def get_accumulation_strategies(cls) -> list[StrategyProfile]:
        """Get all accumulation strategies."""
        return [p for p in cls.PROFILES.values() if p.is_accumulation]
    
    @classmethod
    def validate_exchange_support(cls, strategy_type: StrategyType, exchange: str) -> bool:
        """Validate strategy is supported on exchange."""
        if exchange != "deribit":
            return False  # Phase 1: Deribit only
        return strategy_type in cls.PROFILES
```


### 2. Safety Validator

Enforces all safety constraints to prevent blow-up risk.

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class RejectionSeverity(Enum):
    """Severity of validation rejection."""
    BLOCKING = "blocking"    # Cannot proceed
    WARNING = "warning"      # Can proceed with caution


class RejectionCode(Enum):
    """Rejection codes for unsafe strategies."""
    # BLOCKING - Prohibited strategies
    NAKED_CALL_PROHIBITED = "naked_call_prohibited"
    NAKED_PUT_INSUFFICIENT_COLLATERAL = "naked_put_insufficient_collateral"
    SHORT_STRADDLE_PROHIBITED = "short_straddle_prohibited"
    SHORT_STRANGLE_PROHIBITED = "short_strangle_prohibited"
    UNCOVERED_RATIO_PROHIBITED = "uncovered_ratio_prohibited"
    LEVERAGED_SHORT_PROHIBITED = "leveraged_short_prohibited"
    
    # BLOCKING - Resource constraints
    INSUFFICIENT_COLLATERAL_VERIFIED = "insufficient_collateral_verified"
    RISK_LIMIT_EXCEEDED = "risk_limit_exceeded"
    EXCHANGE_VERIFICATION_REQUIRED = "exchange_verification_required"  # Execution requires authenticated state
    
    # BLOCKING - Protective leg validation failures
    PROTECTIVE_LEG_INVALID = "protective_leg_invalid"
    PROTECTIVE_LEG_EXPIRY_MISMATCH = "protective_leg_expiry_mismatch"
    PROTECTIVE_LEG_TYPE_MISMATCH = "protective_leg_type_mismatch"
    PROTECTIVE_LEG_SETTLEMENT_MISMATCH = "protective_leg_settlement_mismatch"
    PROTECTIVE_LEG_QUANTITY_INSUFFICIENT = "protective_leg_quantity_insufficient"
    PROTECTIVE_LEG_STRIKE_INVALID = "protective_leg_strike_invalid"
    PROTECTIVE_LEG_UNDERLYING_MISMATCH = "protective_leg_underlying_mismatch"
    
    # BLOCKING - Execution safety
    PARTIAL_FILL_NAKED_EXPOSURE = "partial_fill_naked_exposure"
    LONG_LEG_NOT_FILLED = "long_leg_not_filled"
    ATOMIC_EXECUTION_UNAVAILABLE = "atomic_execution_unavailable"
    
    # BLOCKING - Exchange/product constraints (Phase 1)
    INVERSE_OPTIONS_NOT_SUPPORTED = "inverse_options_not_supported"
    NON_USDC_SETTLEMENT_NOT_SUPPORTED = "non_usdc_settlement_not_supported"
    EXCHANGE_NOT_SUPPORTED = "exchange_not_supported"
    
    # BLOCKING - Risk limits
    SPREAD_WIDTH_EXCEEDED = "spread_width_exceeded"
    
    # WARNING - Risk advisories
    HIGH_SETTLEMENT_RISK = "high_settlement_risk"
    LOW_PREMIUM_EFFICIENCY = "low_premium_efficiency"
    HIGH_IV_PREMIUM_WARNING = "high_iv_premium_warning"
    LOW_IV_PREMIUM_WARNING = "low_iv_premium_warning"
    GAMMA_RISK_WARNING = "gamma_risk_warning"
    DIRECTIONAL_RISK_WARNING = "directional_risk_warning"
    SIMULATION_UNVERIFIED = "simulation_unverified"
    LIQUIDITY_WARNING = "liquidity_warning"


@dataclass
class RejectionDetail:
    """Details of a validation rejection."""
    code: RejectionCode
    severity: RejectionSeverity
    message: str
    details: dict = field(default_factory=dict)
    suggested_alternative: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of safety validation."""
    is_valid: bool
    rejections: list[RejectionDetail] = field(default_factory=list)
    warnings: list[RejectionDetail] = field(default_factory=list)
    
    @property
    def has_blocking_issues(self) -> bool:
        return any(r.severity == RejectionSeverity.BLOCKING for r in self.rejections)
    
    @property
    def blocking_rejections(self) -> list[RejectionDetail]:
        return [r for r in self.rejections if r.severity == RejectionSeverity.BLOCKING]


@dataclass
class AccountState:
    """
    Current account state for validation.
    
    IMPORTANT: For execution, this MUST be populated from authenticated
    exchange API calls, NOT user-supplied values.
    """
    available_usdc: float           # USDC balance from exchange
    total_account_value_usdc: float # Total account value in USDC
    current_income_exposure_usdc: float
    current_settlement_exposure_usdc: float
    
    # Validation metadata
    is_exchange_verified: bool = False  # True if from authenticated API
    verification_timestamp: Optional[datetime] = None
    exchange: str = "deribit"
    
    # Phase 1: No BTC collateral (inverse options not supported)
    # btc_holdings: float  # Future: for inverse options


@dataclass
class ProtectiveLegValidation:
    """Result of protective leg validation."""
    is_valid: bool
    long_leg_symbol: str
    short_leg_symbol: str
    checks_passed: list[str]
    checks_failed: list[str]
    rejection_code: Optional[RejectionCode] = None


class SafetyValidator:
    """
    Validates strategies against safety constraints.
    
    Enforces the non-negotiable safety rules:
    - No naked short options
    - No undefined loss exposure
    - No leveraged short structures
    - Full collateralization required (USDC)
    - Protective legs mandatory for spreads (complete validation)
    - Exchange-verified collateral for execution
    """
    
    # Prohibited strategy patterns
    PROHIBITED_PATTERNS = {
        "naked_call": RejectionCode.NAKED_CALL_PROHIBITED,
        "naked_put_insufficient": RejectionCode.NAKED_PUT_INSUFFICIENT_COLLATERAL,
        "short_straddle": RejectionCode.SHORT_STRADDLE_PROHIBITED,
        "short_strangle": RejectionCode.SHORT_STRANGLE_PROHIBITED,
        "ratio_uncovered": RejectionCode.UNCOVERED_RATIO_PROHIBITED,
        "leveraged_short": RejectionCode.LEVERAGED_SHORT_PROHIBITED,
    }
    
    def __init__(self, policy: "IncomeStrategyPolicy"):
        self.policy = policy
    
    def validate(
        self,
        strategy: "IncomeStrategyRequest",
        account: AccountState,
        require_exchange_verification: bool = True,
    ) -> ValidationResult:
        """
        Validate a strategy against all safety constraints.
        
        Args:
            strategy: The strategy to validate
            account: Account state (must be exchange-verified for execution)
            require_exchange_verification: If True, reject unverified account state
        
        Returns ValidationResult with pass/fail and detailed rejections.
        """
        rejections = []
        warnings = []
        
        # Check 0: Exchange verification requirement (BLOCKING for execution)
        if require_exchange_verification and not account.is_exchange_verified:
            rejections.append(RejectionDetail(
                code=RejectionCode.EXCHANGE_VERIFICATION_REQUIRED,
                severity=RejectionSeverity.BLOCKING,
                message="Execution requires authenticated exchange state. Cannot proceed with unverified account balances.",
                details={"is_exchange_verified": False},
                suggested_alternative="Use simulation mode for preview, or provide account_id for exchange-verified execution",
            ))
        
        # Check 1: Exchange/product support (Phase 1)
        product_rejection = self._check_product_support(strategy)
        if product_rejection:
            rejections.append(product_rejection)
        
        # Check 2: Prohibited strategy patterns
        pattern_rejection = self._check_prohibited_patterns(strategy)
        if pattern_rejection:
            rejections.append(pattern_rejection)
        
        # Check 3: Protective leg validation (complete)
        protection_rejection = self._check_protective_legs_complete(strategy)
        if protection_rejection:
            rejections.append(protection_rejection)
        
        # Check 4: Collateral sufficiency (USDC)
        collateral_rejection = self._check_collateral_usdc(strategy, account)
        if collateral_rejection:
            rejections.append(collateral_rejection)
        
        # Check 5: Risk limits
        risk_rejection = self._check_risk_limits(strategy, account)
        if risk_rejection:
            rejections.append(risk_rejection)
        
        # Check 6: Liquidity gates
        liquidity_warnings = self._check_liquidity(strategy)
        warnings.extend(liquidity_warnings)
        
        # Check 7: Other warnings (non-blocking)
        warnings.extend(self._check_warnings(strategy, account))
        
        is_valid = not any(
            r.severity == RejectionSeverity.BLOCKING for r in rejections
        )
        
        return ValidationResult(
            is_valid=is_valid,
            rejections=rejections,
            warnings=warnings,
        )
    
    def _check_product_support(
        self, 
        strategy: "IncomeStrategyRequest"
    ) -> Optional[RejectionDetail]:
        """Check if product is supported in Phase 1."""
        if strategy.settlement_currency != "USDC":
            return RejectionDetail(
                code=RejectionCode.NON_USDC_SETTLEMENT_NOT_SUPPORTED,
                severity=RejectionSeverity.BLOCKING,
                message=f"Only USDC-settled options supported in Phase 1. Got: {strategy.settlement_currency}",
                details={"settlement_currency": strategy.settlement_currency},
                suggested_alternative="Use Deribit Linear USDC options",
            )
        if strategy.exchange != "deribit":
            return RejectionDetail(
                code=RejectionCode.EXCHANGE_NOT_SUPPORTED,
                severity=RejectionSeverity.BLOCKING,
                message=f"Only Deribit supported in Phase 1. Got: {strategy.exchange}",
                details={"exchange": strategy.exchange},
            )
        return None
    
    def _check_protective_legs_complete(
        self,
        strategy: "IncomeStrategyRequest",
    ) -> Optional[RejectionDetail]:
        """
        Verify EVERY short leg has a valid protective long leg.
        
        For two-leg spreads: one short leg, one protective long leg.
        For iron condors: two short legs, each with its own protective long leg.
        
        Each short-long pair must satisfy ALL requirements:
        - Same underlying (BTC)
        - Same expiry
        - Same settlement currency (USDC)
        - Same option type (call protects call, put protects put)
        - Correct strike ordering
        - Equal or greater quantity on long leg
        - No ratio tail (no uncovered short exposure)
        """
        if not strategy.is_spread:
            return None
        
        # Get all legs grouped by role
        short_legs = [leg for leg in strategy.legs if leg.action == "SELL"]
        long_legs = [leg for leg in strategy.legs if leg.action == "BUY"]
        
        # CRITICAL: Every short leg must have a protective long leg
        if len(short_legs) == 0:
            return None  # No short legs to protect (debit spread)
        
        if len(long_legs) < len(short_legs):
            return RejectionDetail(
                code=RejectionCode.PROTECTIVE_LEG_QUANTITY_INSUFFICIENT,
                severity=RejectionSeverity.BLOCKING,
                message=f"Not enough protective legs: {len(short_legs)} short legs but only {len(long_legs)} long legs",
                details={"short_leg_count": len(short_legs), "long_leg_count": len(long_legs)},
                suggested_alternative="Add protective long legs for each short leg",
            )
        
        # Validate each short leg has a matching protective long leg
        for short_leg in short_legs:
            # Find matching long leg (same option type)
            matching_longs = [
                long_leg for long_leg in long_legs 
                if long_leg.option_type == short_leg.option_type
            ]
            
            if not matching_longs:
                return RejectionDetail(
                    code=RejectionCode.PROTECTIVE_LEG_TYPE_MISMATCH,
                    severity=RejectionSeverity.BLOCKING,
                    message=f"No protective long {short_leg.option_type} for short {short_leg.option_type} at strike {short_leg.strike}",
                    details={"short_leg": short_leg.symbol, "option_type": short_leg.option_type},
                )
            
            # Use the first matching long leg for validation
            long_leg = matching_longs[0]
            
            # Validate the pair
            rejection = self._validate_protective_pair(short_leg, long_leg)
            if rejection:
                return rejection
        
        return None
    
    def _validate_protective_pair(
        self,
        short_leg: "OptionLeg",
        long_leg: "OptionLeg",
    ) -> Optional[RejectionDetail]:
        """
        Validate a single short-long protective pair.
        
        Returns RejectionDetail if validation fails, None if valid.
        """
        # Check 1: Same underlying
        if long_leg.underlying != short_leg.underlying:
            return RejectionDetail(
                code=RejectionCode.PROTECTIVE_LEG_UNDERLYING_MISMATCH,
                severity=RejectionSeverity.BLOCKING,
                message=f"Underlying mismatch: long={long_leg.underlying}, short={short_leg.underlying}",
                details={"long_underlying": long_leg.underlying, "short_underlying": short_leg.underlying},
            )
        
        # Check 2: Same expiry
        if long_leg.expiry != short_leg.expiry:
            return RejectionDetail(
                code=RejectionCode.PROTECTIVE_LEG_EXPIRY_MISMATCH,
                severity=RejectionSeverity.BLOCKING,
                message=f"Expiry mismatch: long={long_leg.expiry}, short={short_leg.expiry}",
                details={"long_expiry": str(long_leg.expiry), "short_expiry": str(short_leg.expiry)},
            )
        
        # Check 3: Same settlement currency
        if long_leg.settlement_currency != short_leg.settlement_currency:
            return RejectionDetail(
                code=RejectionCode.PROTECTIVE_LEG_SETTLEMENT_MISMATCH,
                severity=RejectionSeverity.BLOCKING,
                message=f"Settlement mismatch: long={long_leg.settlement_currency}, short={short_leg.settlement_currency}",
                details={"long_settlement": long_leg.settlement_currency, "short_settlement": short_leg.settlement_currency},
            )
        
        # Check 4: Same option type (already verified in caller, but double-check)
        if long_leg.option_type != short_leg.option_type:
            return RejectionDetail(
                code=RejectionCode.PROTECTIVE_LEG_TYPE_MISMATCH,
                severity=RejectionSeverity.BLOCKING,
                message=f"Option type mismatch: long={long_leg.option_type}, short={short_leg.option_type}",
                details={"long_type": long_leg.option_type, "short_type": short_leg.option_type},
            )
        
        # Check 5: Correct strike ordering
        if long_leg.option_type == "put":
            # Put spread: long strike <= short strike (long is further OTM)
            if long_leg.strike > short_leg.strike:
                return RejectionDetail(
                    code=RejectionCode.PROTECTIVE_LEG_STRIKE_INVALID,
                    severity=RejectionSeverity.BLOCKING,
                    message=f"Put spread: long strike ({long_leg.strike}) must be <= short strike ({short_leg.strike})",
                    details={"long_strike": long_leg.strike, "short_strike": short_leg.strike},
                )
        else:  # call
            # Call spread: long strike >= short strike (long is further OTM)
            if long_leg.strike < short_leg.strike:
                return RejectionDetail(
                    code=RejectionCode.PROTECTIVE_LEG_STRIKE_INVALID,
                    severity=RejectionSeverity.BLOCKING,
                    message=f"Call spread: long strike ({long_leg.strike}) must be >= short strike ({short_leg.strike})",
                    details={"long_strike": long_leg.strike, "short_strike": short_leg.strike},
                )
        
        # Check 6: Quantity coverage (no ratio tail / uncovered exposure)
        if long_leg.quantity < short_leg.quantity:
            return RejectionDetail(
                code=RejectionCode.PROTECTIVE_LEG_QUANTITY_INSUFFICIENT,
                severity=RejectionSeverity.BLOCKING,
                message=f"Long leg quantity ({long_leg.quantity}) must be >= short leg quantity ({short_leg.quantity})",
                details={"long_qty": long_leg.quantity, "short_qty": short_leg.quantity},
                suggested_alternative="Reduce short leg quantity to match long leg",
            )
        
        return None
    
    def _check_collateral_usdc(
        self,
        strategy: "IncomeStrategyRequest",
        account: AccountState,
    ) -> Optional[RejectionDetail]:
        """Verify sufficient USDC collateral for the strategy."""
        profile = StrategyClassifier.classify(strategy.strategy_type)
        
        # Calculate required collateral based on strategy type
        if strategy.strategy_type == StrategyType.CASH_SECURED_PUT:
            # Full strike value as collateral (worst case: BTC → 0)
            required = strategy.strike * strategy.contract_size * strategy.quantity
        elif strategy.is_spread:
            # Spread width as collateral
            required = strategy.spread_width * strategy.contract_size * strategy.quantity
        else:
            # Single leg: per exchange margin requirements
            required = strategy.max_loss_usdc
        
        if account.available_usdc < required:
            return RejectionDetail(
                code=RejectionCode.INSUFFICIENT_COLLATERAL_VERIFIED,
                severity=RejectionSeverity.BLOCKING,
                message=f"Insufficient USDC. Required: ${required:,.0f}, Available: ${account.available_usdc:,.0f}",
                details={
                    "required_usdc": required, 
                    "available_usdc": account.available_usdc,
                    "is_exchange_verified": account.is_exchange_verified,
                },
                suggested_alternative="Reduce position size or add USDC",
            )
        
        return None
    
    def _check_liquidity(
        self,
        strategy: "IncomeStrategyRequest",
    ) -> list[RejectionDetail]:
        """Check liquidity gates: bid/ask spread, size, staleness."""
        warnings = []
        
        for leg in strategy.legs:
            # Bid-ask spread check
            if leg.spread_pct and leg.spread_pct > self.policy.max_spread_pct:
                warnings.append(RejectionDetail(
                    code=RejectionCode.LIQUIDITY_WARNING,
                    severity=RejectionSeverity.WARNING,
                    message=f"Wide bid-ask spread on {leg.symbol}: {leg.spread_pct*100:.1f}%",
                    details={"symbol": leg.symbol, "spread_pct": leg.spread_pct},
                ))
            
            # Open interest check
            if leg.open_interest and leg.open_interest < self.policy.min_open_interest:
                warnings.append(RejectionDetail(
                    code=RejectionCode.LIQUIDITY_WARNING,
                    severity=RejectionSeverity.WARNING,
                    message=f"Low open interest on {leg.symbol}: {leg.open_interest}",
                    details={"symbol": leg.symbol, "open_interest": leg.open_interest},
                ))
        
        return warnings


### 3. Payoff Calculator

Calculates complete payoff metrics for income strategies.

```python
from dataclasses import dataclass
from typing import Optional
from decimal import Decimal


@dataclass
class PayoffMetrics:
    """Complete payoff metrics for a strategy."""
    # Core payoff
    max_profit: float           # Maximum possible profit (USD)
    max_loss: float             # Maximum possible loss (USD) - MUST be defined
    breakeven_price: float      # Single breakeven for simple strategies
    breakeven_prices: list[float]  # Multiple breakevens for complex strategies
    
    # Risk/reward
    risk_reward_ratio: float    # max_profit / max_loss
    probability_of_profit: float  # Estimated PoP based on delta
    expected_value: float       # Probability-weighted expected return
    
    # Collateral
    collateral_required: float  # Total capital locked
    buying_power_reduction: float  # Impact on available margin
    capital_efficiency: float   # Annualized return on collateral
    
    # Settlement risk (European options - no early assignment)
    itm_probability_at_expiry: float  # ITM probability at expiry
    settlement_risk: str  # HIGH/MEDIUM/LOW based on moneyness and DTE
    max_settlement_payout_usdc: float  # Maximum cash settlement amount in USDC


@dataclass
class OptionLeg:
    """Single option leg in a strategy."""
    action: str          # "BUY" or "SELL"
    option_type: str     # "call" or "put"
    strike: float
    premium: float       # Per-contract premium
    delta: float
    iv: float
    dte: int
    quantity: int


class PayoffCalculator:
    """
    Calculates payoff metrics for income strategies.
    
    Supports single-leg, two-leg spreads, and four-leg iron condors.
    """
    
    def calculate(
        self,
        strategy_type: StrategyType,
        legs: list[OptionLeg],
        spot_price: float,
        contract_size: float = 1.0,  # BTC per contract
        days_to_expiry: int = 30,
    ) -> PayoffMetrics:
        """Calculate complete payoff metrics for a strategy."""
        
        if strategy_type == StrategyType.CASH_SECURED_PUT:
            return self._calc_cash_secured_put(legs[0], spot_price, contract_size)
        elif strategy_type in (StrategyType.BULL_PUT_SPREAD, StrategyType.BEAR_CALL_SPREAD):
            return self._calc_credit_spread(legs, spot_price, contract_size)
        elif strategy_type == StrategyType.IRON_CONDOR:
            return self._calc_iron_condor(legs, spot_price, contract_size)
        elif strategy_type in (StrategyType.PUT_DEBIT_SPREAD, StrategyType.CALL_DEBIT_SPREAD):
            return self._calc_debit_spread(legs, spot_price, contract_size)
        else:
            raise ValueError(f"Unsupported strategy type: {strategy_type}")
    
    def _calc_cash_secured_put(
        self,
        put_leg: OptionLeg,
        spot_price: float,
        contract_size: float,
    ) -> PayoffMetrics:
        """Calculate payoff for cash-secured put."""
        premium = put_leg.premium * contract_size * put_leg.quantity
        strike = put_leg.strike
        
        # Max profit = premium received
        max_profit = premium
        
        # Max loss = strike - premium (if BTC goes to 0)
        # But realistically, loss is (strike - spot_at_expiry) - premium
        max_loss = (strike * contract_size * put_leg.quantity) - premium
        
        # Breakeven = strike - premium per share
        breakeven = strike - (put_leg.premium)
        
        # Collateral = full strike value (cash-secured)
        collateral = strike * contract_size * put_leg.quantity
        
        # Capital efficiency (annualized)
        days = put_leg.dte if put_leg.dte > 0 else 30
        annual_factor = 365 / days
        capital_efficiency = (max_profit / collateral) * annual_factor
        
        # ITM probability at expiry ≈ delta for puts (simplified)
        itm_prob = abs(put_leg.delta)
        
        return PayoffMetrics(
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_price=breakeven,
            breakeven_prices=[breakeven],
            risk_reward_ratio=max_profit / max_loss if max_loss > 0 else 0,
            probability_of_profit=1 - itm_prob,
            expected_value=max_profit * (1 - itm_prob) - max_loss * itm_prob,
            collateral_required=collateral,
            buying_power_reduction=collateral,
            capital_efficiency=capital_efficiency,
            itm_probability_at_expiry=itm_prob,
            settlement_risk=self._assess_settlement_risk(put_leg, spot_price),
            max_settlement_payout_usdc=strike * contract_size * put_leg.quantity,
        )

    
    def _calc_credit_spread(
        self,
        legs: list[OptionLeg],
        spot_price: float,
        contract_size: float,
    ) -> PayoffMetrics:
        """Calculate payoff for credit spread (bull put or bear call)."""
        # Identify short and long legs
        short_leg = next(l for l in legs if l.action == "SELL")
        long_leg = next(l for l in legs if l.action == "BUY")
        
        qty = short_leg.quantity
        
        # Net credit received
        net_credit = (short_leg.premium - long_leg.premium) * contract_size * qty
        
        # Spread width
        width = abs(short_leg.strike - long_leg.strike)
        
        # Max profit = net credit
        max_profit = net_credit
        
        # Max loss = width - net credit
        max_loss = (width * contract_size * qty) - net_credit
        
        # Breakeven depends on spread type
        if short_leg.option_type == "put":
            # Bull put spread: breakeven = short strike - net credit per share
            breakeven = short_leg.strike - (net_credit / (contract_size * qty))
        else:
            # Bear call spread: breakeven = short strike + net credit per share
            breakeven = short_leg.strike + (net_credit / (contract_size * qty))
        
        # Collateral = spread width (defined risk)
        collateral = width * contract_size * qty
        
        # Capital efficiency
        days = short_leg.dte if short_leg.dte > 0 else 30
        capital_efficiency = (max_profit / collateral) * (365 / days)
        
        # PoP approximation using short leg delta
        pop = 1 - abs(short_leg.delta)
        
        return PayoffMetrics(
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_price=breakeven,
            breakeven_prices=[breakeven],
            risk_reward_ratio=max_profit / max_loss if max_loss > 0 else 0,
            probability_of_profit=pop,
            expected_value=max_profit * pop - max_loss * (1 - pop),
            collateral_required=collateral,
            buying_power_reduction=collateral,
            capital_efficiency=capital_efficiency,
            itm_probability_at_expiry=abs(short_leg.delta),
            settlement_risk=self._assess_settlement_risk(short_leg, spot_price),
            max_settlement_payout_usdc=short_leg.strike * contract_size * qty,
        )
    
    def _calc_iron_condor(
        self,
        legs: list[OptionLeg],
        spot_price: float,
        contract_size: float,
    ) -> PayoffMetrics:
        """Calculate payoff for iron condor (4 legs)."""
        # Separate put spread and call spread legs
        put_legs = [l for l in legs if l.option_type == "put"]
        call_legs = [l for l in legs if l.option_type == "call"]
        
        # Calculate each spread
        put_spread = self._calc_credit_spread(put_legs, spot_price, contract_size)
        call_spread = self._calc_credit_spread(call_legs, spot_price, contract_size)
        
        # Combined metrics
        max_profit = put_spread.max_profit + call_spread.max_profit
        max_loss = max(put_spread.max_loss, call_spread.max_loss)  # Only one side can lose
        
        # Two breakevens
        breakevens = [put_spread.breakeven_price, call_spread.breakeven_price]
        
        # Collateral = max of either spread (not sum, since only one can be ITM)
        collateral = max(put_spread.collateral_required, call_spread.collateral_required)
        
        return PayoffMetrics(
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_price=spot_price,  # Midpoint for reference
            breakeven_prices=sorted(breakevens),
            risk_reward_ratio=max_profit / max_loss if max_loss > 0 else 0,
            probability_of_profit=put_spread.probability_of_profit * call_spread.probability_of_profit,
            expected_value=max_profit * 0.6 - max_loss * 0.4,  # Simplified
            collateral_required=collateral,
            buying_power_reduction=collateral,
            capital_efficiency=(max_profit / collateral) * (365 / put_legs[0].dte),
            itm_probability_at_expiry=max(put_spread.itm_probability_at_expiry, call_spread.itm_probability_at_expiry),
            settlement_risk="LOW",  # Spreads have defined risk
            max_settlement_payout_usdc=max(put_spread.max_settlement_payout_usdc, call_spread.max_settlement_payout_usdc),
        )
    
    def _assess_settlement_risk(
        self,
        leg: OptionLeg,
        spot_price: float,
    ) -> str:
        """
        Assess settlement risk for a short option at expiry.
        
        NOTE: Deribit options are European and cash-settled.
        There is NO early assignment. This assesses the risk of
        the option settling ITM at expiry.
        """
        moneyness = (leg.strike - spot_price) / spot_price
        
        if leg.option_type == "put":
            # Put is ITM if strike > spot
            itm_pct = moneyness
        else:
            # Call is ITM if strike < spot
            itm_pct = -moneyness
        
        if itm_pct > 0.05 and leg.dte < 7:
            return "HIGH"
        elif itm_pct > 0.02 or leg.dte < 14:
            return "MEDIUM"
        else:
            return "LOW"
```


### 4. Risk Engine

Enforces position and portfolio-level risk limits.

```python
from dataclasses import dataclass
from typing import Optional


@dataclass
class RiskLimits:
    """Risk limit configuration."""
    max_single_position_risk_pct: float = 0.05      # 5% of account per position
    max_total_income_exposure_pct: float = 0.30     # 30% of account in income strategies
    max_settlement_exposure_pct: float = 0.50       # 50% of account exposed to ITM settlement
    max_spread_width_usd: float = 5000              # Max spread width
    max_single_expiry_concentration_pct: float = 0.20  # 20% of income in single expiry


@dataclass
class RiskAssessment:
    """Risk assessment result."""
    is_approved: bool
    account_risk_pct: float
    margin_utilization_pct: float
    concentration_risk: str  # LOW/MEDIUM/HIGH
    blocking_issues: list[RejectionDetail]
    warnings: list[RejectionDetail]


class RiskEngine:
    """
    Enforces risk limits for income strategies.
    
    Prevents excessive concentration, over-leveraging, and blow-up risk.
    """
    
    def __init__(self, limits: Optional[RiskLimits] = None):
        self.limits = limits or RiskLimits()
    
    def assess(
        self,
        strategy: "IncomeStrategyRequest",
        payoff: PayoffMetrics,
        account: AccountState,
    ) -> RiskAssessment:
        """Assess risk for a proposed strategy."""
        blocking = []
        warnings = []
        
        # Calculate risk metrics (all USDC-denominated)
        account_risk_pct = payoff.max_loss / account.total_account_value_usdc
        margin_util_pct = payoff.collateral_required / account.available_usdc
        
        # Check position-level risk
        if account_risk_pct > self.limits.max_single_position_risk_pct:
            blocking.append(RejectionDetail(
                code=RejectionCode.RISK_LIMIT_EXCEEDED,
                severity=RejectionSeverity.BLOCKING,
                message=f"Position risk {account_risk_pct*100:.1f}% exceeds limit {self.limits.max_single_position_risk_pct*100:.0f}%",
                details={
                    "position_risk_pct": account_risk_pct,
                    "limit_pct": self.limits.max_single_position_risk_pct,
                    "max_loss_usdc": payoff.max_loss,
                },
                suggested_alternative="Reduce position size or use tighter spread",
            ))
        
        # Check total income exposure (USDC)
        new_total_exposure = account.current_income_exposure_usdc + payoff.collateral_required
        total_exposure_pct = new_total_exposure / account.total_account_value_usdc
        
        if total_exposure_pct > self.limits.max_total_income_exposure_pct:
            blocking.append(RejectionDetail(
                code=RejectionCode.RISK_LIMIT_EXCEEDED,
                severity=RejectionSeverity.BLOCKING,
                message=f"Total income exposure {total_exposure_pct*100:.1f}% would exceed limit {self.limits.max_total_income_exposure_pct*100:.0f}%",
                details={
                    "current_exposure_usdc": account.current_income_exposure_usdc,
                    "new_exposure_usdc": payoff.collateral_required,
                    "total_pct": total_exposure_pct,
                },
            ))
        
        # Check settlement exposure (ITM at expiry risk)
        new_settlement_exposure = account.current_settlement_exposure_usdc + payoff.max_settlement_payout_usdc
        settlement_pct = new_settlement_exposure / account.total_account_value_usdc
        
        if settlement_pct > self.limits.max_settlement_exposure_pct:
            blocking.append(RejectionDetail(
                code=RejectionCode.RISK_LIMIT_EXCEEDED,
                severity=RejectionSeverity.BLOCKING,
                message=f"Settlement exposure {settlement_pct*100:.1f}% would exceed limit {self.limits.max_settlement_exposure_pct*100:.0f}%",
                details={
                    "max_settlement_payout_usdc": payoff.max_settlement_payout_usdc,
                    "total_settlement_pct": settlement_pct,
                },
            ))
        
        # Warnings
        if payoff.settlement_risk == "HIGH":
            warnings.append(RejectionDetail(
                code=RejectionCode.HIGH_SETTLEMENT_RISK,
                severity=RejectionSeverity.WARNING,
                message="High settlement risk - position is ITM with low DTE",
            ))
        
        # Concentration risk assessment
        concentration = self._assess_concentration(strategy, account)
        
        return RiskAssessment(
            is_approved=len(blocking) == 0,
            account_risk_pct=account_risk_pct,
            margin_utilization_pct=margin_util_pct,
            concentration_risk=concentration,
            blocking_issues=blocking,
            warnings=warnings,
        )
    
    def _assess_concentration(
        self,
        strategy: "IncomeStrategyRequest",
        account: AccountState,
    ) -> str:
        """Assess concentration risk."""
        # Implementation checks expiry concentration, strike concentration, etc.
        return "LOW"  # Placeholder
```


### 5. Recommendation Engine

Provides market-regime-based strategy recommendations.

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class MarketRegime(Enum):
    """Market regime classification."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    RANGE_BOUND = "range_bound"
    HIGH_IV = "high_iv"
    LOW_IV = "low_iv"


@dataclass
class StrategyRecommendation:
    """A strategy recommendation with context."""
    strategy_type: StrategyType
    rank: int                    # 1 = best fit
    fit_score: float             # 0-1 score
    rationale: str               # Why this strategy fits
    strike_guidance: str         # OTM distance recommendation
    dte_guidance: str            # DTE recommendation
    size_guidance: str           # Position size recommendation
    warnings: list[str]          # Regime-specific warnings


class RecommendationEngine:
    """
    Recommends strategies based on market regime.
    
    Integrates with existing fusion engine signals for regime classification.
    """
    
    # Strategy fit matrix by regime (only defined-risk strategies)
    REGIME_STRATEGY_FIT: dict[MarketRegime, dict[StrategyType, float]] = {
        MarketRegime.BULLISH: {
            StrategyType.CASH_SECURED_PUT: 0.95,
            StrategyType.BULL_PUT_SPREAD: 0.90,
            StrategyType.BEAR_CALL_SPREAD: 0.20,  # Avoid in bullish
            StrategyType.IRON_CONDOR: 0.40,
        },
        MarketRegime.BEARISH: {
            StrategyType.BEAR_CALL_SPREAD: 0.95,  # Defined-risk call income
            StrategyType.PUT_DEBIT_SPREAD: 0.80,
            StrategyType.CASH_SECURED_PUT: 0.30,  # Risk of ITM settlement at high prices
            StrategyType.IRON_CONDOR: 0.40,
        },
        MarketRegime.RANGE_BOUND: {
            StrategyType.IRON_CONDOR: 0.95,
            StrategyType.BULL_PUT_SPREAD: 0.70,
            StrategyType.BEAR_CALL_SPREAD: 0.70,
            StrategyType.CASH_SECURED_PUT: 0.65,
        },
        MarketRegime.HIGH_IV: {
            StrategyType.IRON_CONDOR: 0.95,
            StrategyType.BULL_PUT_SPREAD: 0.90,
            StrategyType.BEAR_CALL_SPREAD: 0.90,
            StrategyType.CASH_SECURED_PUT: 0.85,
            StrategyType.PUT_DEBIT_SPREAD: 0.30,  # Expensive
            StrategyType.CALL_DEBIT_SPREAD: 0.30,
        },
        MarketRegime.LOW_IV: {
            StrategyType.PUT_DEBIT_SPREAD: 0.85,
            StrategyType.CALL_DEBIT_SPREAD: 0.85,
            StrategyType.CASH_SECURED_PUT: 0.65,
            StrategyType.IRON_CONDOR: 0.40,  # Low premium
            StrategyType.BULL_PUT_SPREAD: 0.40,
        },
    }
    
    # Strike guidance by regime (only defined-risk strategies)
    STRIKE_GUIDANCE: dict[MarketRegime, dict[StrategyType, str]] = {
        MarketRegime.BULLISH: {
            StrategyType.CASH_SECURED_PUT: "5-10% OTM (below support)",
            StrategyType.BULL_PUT_SPREAD: "Short leg 5% OTM, long leg 10% OTM",
            StrategyType.BEAR_CALL_SPREAD: "Short leg 10-15% OTM, long leg 15-20% OTM",
        },
        MarketRegime.BEARISH: {
            StrategyType.BEAR_CALL_SPREAD: "Short leg 5% OTM, long leg 10% OTM",
            StrategyType.CASH_SECURED_PUT: "10-15% OTM (more conservative in bearish)",
        },
        MarketRegime.RANGE_BOUND: {
            StrategyType.IRON_CONDOR: "Short legs at range boundaries (±10%)",
        },
    }
    
    # DTE guidance by regime
    DTE_GUIDANCE: dict[MarketRegime, str] = {
        MarketRegime.BULLISH: "30-45 DTE for optimal theta decay",
        MarketRegime.BEARISH: "21-30 DTE for faster premium capture",
        MarketRegime.RANGE_BOUND: "30-45 DTE for iron condors",
        MarketRegime.HIGH_IV: "21-30 DTE to capture elevated premium",
        MarketRegime.LOW_IV: "45-60 DTE for debit spreads, 30 DTE for credit",
    }

    
    def __init__(self, policy: "IncomeStrategyPolicy"):
        self.policy = policy
    
    def classify_regime(
        self,
        fusion_state: str,
        fusion_confidence: str,
        iv_rank: float,
    ) -> MarketRegime:
        """
        Classify market regime from fusion engine signals.
        
        Integrates with existing DailySignal fusion_state.
        """
        # IV-based regime takes precedence if extreme
        if iv_rank > 0.60:
            return MarketRegime.HIGH_IV
        if iv_rank < 0.30:
            return MarketRegime.LOW_IV
        
        # Direction-based regime from fusion state
        bullish_states = {"STRONG_BULLISH", "BULLISH", "WEAK_BULLISH"}
        bearish_states = {"STRONG_BEARISH", "BEARISH", "WEAK_BEARISH"}
        neutral_states = {"NEUTRAL", "CHOPPY", "CONSOLIDATION"}
        
        if fusion_state in bullish_states:
            return MarketRegime.BULLISH
        elif fusion_state in bearish_states:
            return MarketRegime.BEARISH
        elif fusion_state in neutral_states:
            return MarketRegime.RANGE_BOUND
        else:
            return MarketRegime.RANGE_BOUND  # Default
    
    def recommend(
        self,
        regime: MarketRegime,
        account: AccountState,
        preferences: Optional[dict] = None,
    ) -> list[StrategyRecommendation]:
        """
        Get ranked strategy recommendations for current regime.
        
        Returns strategies sorted by fit score, filtered by account constraints.
        All strategies require USDC collateral (Phase 1: no BTC collateral).
        """
        recommendations = []
        fit_scores = self.REGIME_STRATEGY_FIT.get(regime, {})
        
        for strategy_type, fit_score in sorted(fit_scores.items(), key=lambda x: -x[1]):
            # Skip if account doesn't have USDC collateral
            profile = StrategyClassifier.classify(strategy_type)
            
            # Phase 1: All strategies require USDC collateral only
            if account.available_usdc <= 0:
                continue
            
            # Build recommendation
            rec = StrategyRecommendation(
                strategy_type=strategy_type,
                rank=len(recommendations) + 1,
                fit_score=fit_score,
                rationale=self._get_rationale(strategy_type, regime),
                strike_guidance=self.STRIKE_GUIDANCE.get(regime, {}).get(
                    strategy_type, "5-10% OTM"
                ),
                dte_guidance=self.DTE_GUIDANCE.get(regime, "30-45 DTE"),
                size_guidance=self._get_size_guidance(strategy_type, account),
                warnings=self._get_regime_warnings(strategy_type, regime),
            )
            recommendations.append(rec)
        
        return recommendations
    
    def _get_rationale(self, strategy_type: StrategyType, regime: MarketRegime) -> str:
        """Get rationale for strategy in regime."""
        rationales = {
            (StrategyType.CASH_SECURED_PUT, MarketRegime.BULLISH): 
                "Bullish bias favors selling puts - collect premium while waiting for synthetic BTC exposure at lower prices",
            (StrategyType.BEAR_CALL_SPREAD, MarketRegime.BEARISH):
                "Bearish bias favors selling call spreads - generate defined-risk income from downward pressure",
            (StrategyType.IRON_CONDOR, MarketRegime.RANGE_BOUND):
                "Range-bound market ideal for iron condors - profit from time decay within defined range",
            (StrategyType.IRON_CONDOR, MarketRegime.HIGH_IV):
                "High IV inflates premiums - sell premium with defined risk via iron condor",
        }
        return rationales.get(
            (strategy_type, regime), 
            f"{strategy_type.value} suitable for {regime.value} conditions"
        )
    
    def _get_size_guidance(self, strategy_type: StrategyType, account: AccountState) -> str:
        """Get position sizing guidance (USDC-denominated)."""
        max_risk = account.total_account_value_usdc * 0.05  # 5% max risk
        return f"Size to risk max ${max_risk:,.0f} USDC (5% of account)"
    
    def _get_regime_warnings(self, strategy_type: StrategyType, regime: MarketRegime) -> list[str]:
        """Get regime-specific warnings."""
        warnings = []
        
        if regime == MarketRegime.HIGH_IV:
            if strategy_type in (StrategyType.PUT_DEBIT_SPREAD, StrategyType.CALL_DEBIT_SPREAD):
                warnings.append("Debit spreads expensive in high IV - consider waiting for IV crush")
        
        if regime == MarketRegime.LOW_IV:
            if strategy_type in (StrategyType.IRON_CONDOR, StrategyType.BULL_PUT_SPREAD):
                warnings.append("Low IV compresses premiums - may not be worth the risk")
        
        if regime == MarketRegime.BEARISH:
            if strategy_type == StrategyType.CASH_SECURED_PUT:
                warnings.append("Selling puts in bearish market risks ITM settlement at unfavorable prices")
        
        return warnings
```


### 6. CSP Income Tracker

Tracks cash-secured put income strategy performance (replaces wheel strategy which required naked calls).

```python
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional


@dataclass
class CSPIncomeMetrics:
    """Performance metrics for cash-secured put income strategy."""
    total_premium_collected_usdc: float
    total_settlement_paid_usdc: float  # Paid when puts expire ITM
    net_income_usdc: float  # premium - settlement paid
    annualized_yield_pct: float
    positions_opened: int
    positions_closed: int
    otm_expiry_rate: float  # % of positions expired OTM (max profit)


@dataclass
class CSPPosition:
    """A single cash-secured put position."""
    symbol: str
    strike: float
    expiry: date
    quantity: float
    premium_received_usdc: float
    collateral_locked_usdc: float
    opened_at: datetime
    closed_at: Optional[datetime] = None
    settlement_paid_usdc: float = 0.0  # Paid if ITM at expiry
    status: str = "open"  # open, expired_otm, expired_itm, closed_early


class CSPIncomeTracker:
    """
    Tracks cash-secured put income strategy performance.
    
    IMPORTANT: Deribit options are European and cash-settled.
    - NO physical BTC delivery
    - NO early assignment
    - Short put ITM at expiry = PAY cash settlement (this is a LOSS)
    - Short put OTM at expiry = keep premium (this is a GAIN)
    
    NOTE: Traditional "wheel strategy" is NOT supported because the call
    phase requires naked short calls which have undefined upside loss.
    Use bear call spreads for defined-risk call-side income instead.
    """
    
    def __init__(self, policy: "IncomeStrategyPolicy"):
        self.policy = policy
    
    def get_metrics(self, tracker_id: str) -> CSPIncomeMetrics:
        """Calculate performance metrics for CSP income strategy."""
        positions = self._get_positions(tracker_id)
        
        total_premium = sum(p.premium_received_usdc for p in positions)
        total_settlement = sum(p.settlement_paid_usdc for p in positions)
        closed_positions = [p for p in positions if p.status != "open"]
        otm_count = len([p for p in closed_positions if p.status == "expired_otm"])
        
        # Calculate annualized yield
        if positions:
            first_position = min(positions, key=lambda p: p.opened_at)
            days_active = (datetime.now() - first_position.opened_at).days
            total_collateral = sum(p.collateral_locked_usdc for p in positions) / len(positions)
            if days_active > 0 and total_collateral > 0:
                net_income = total_premium - total_settlement
                annualized_yield = (net_income / total_collateral) * (365 / days_active)
            else:
                annualized_yield = 0
        else:
            annualized_yield = 0
        
        return CSPIncomeMetrics(
            total_premium_collected_usdc=total_premium,
            total_settlement_paid_usdc=total_settlement,
            net_income_usdc=total_premium - total_settlement,
            annualized_yield_pct=annualized_yield,
            positions_opened=len(positions),
            positions_closed=len(closed_positions),
            otm_expiry_rate=otm_count / max(1, len(closed_positions)),
        )
    
    def handle_cash_settlement(
        self,
        position_id: str,
        settlement_price: float,  # BTC price at settlement
    ) -> CSPPosition:
        """
        Handle a cash settlement event for a short put.
        
        Short put settles ITM: PAY cash = (strike - settlement_price) * qty
        This is a LOSS for the put seller.
        
        Short put settles OTM: Keep premium, no settlement payment.
        This is a GAIN for the put seller.
        
        NOTE: There is NO physical BTC delivery on Deribit.
        """
        position = self._get_position(position_id)
        
        if settlement_price < position.strike:
            # Put settled ITM - we PAY cash settlement (this is a LOSS)
            settlement_amount = (position.strike - settlement_price) * position.quantity
            position.settlement_paid_usdc = settlement_amount
            position.status = "expired_itm"
        else:
            # Put settled OTM - we keep premium (this is a GAIN)
            position.settlement_paid_usdc = 0
            position.status = "expired_otm"
        
        position.closed_at = datetime.now()
        self._save_position(position)
        return position
    
    def _get_positions(self, tracker_id: str) -> list[CSPPosition]:
        """Load positions from database."""
        ...
    
    def _get_position(self, position_id: str) -> CSPPosition:
        """Load single position from database."""
        ...
    
    def _save_position(self, position: CSPPosition):
        """Persist position to database."""
        ...
```


## Data Models

### New Django Models

```python
# execution/models.py additions

class IncomePosition(models.Model):
    """
    Tracks active income strategy positions.
    
    Extends ExecutionIntent for income-specific tracking.
    """
    STRATEGY_CHOICES = [
        ('cash_secured_put', 'Cash-Secured Put'),
        ('bull_put_spread', 'Bull Put Spread'),
        ('bear_call_spread', 'Bear Call Spread'),  # Defined-risk call income
        ('iron_condor', 'Iron Condor'),
        ('put_debit_spread', 'Put Debit Spread'),
        ('call_debit_spread', 'Call Debit Spread'),
    ]
    # NOTE: synthetic_covered_call and wheel phases removed - naked calls have undefined loss
    
    CATEGORY_CHOICES = [
        ('income', 'Income'),
        ('accumulation', 'Accumulation'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    account = models.ForeignKey(ExchangeAccount, on_delete=models.CASCADE)
    
    # Strategy identification
    strategy_type = models.CharField(max_length=30, choices=STRATEGY_CHOICES)
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES)
    
    # Position details
    entry_date = models.DateField()
    expiry_date = models.DateField()
    
    # Legs (JSON for flexibility)
    legs = models.JSONField(help_text="List of option legs with strikes, premiums, etc.")
    
    # Metrics at entry
    net_premium = models.DecimalField(max_digits=12, decimal_places=4)
    max_profit = models.DecimalField(max_digits=12, decimal_places=2)
    max_loss = models.DecimalField(max_digits=12, decimal_places=2)
    collateral_locked = models.DecimalField(max_digits=12, decimal_places=2)
    breakeven_price = models.DecimalField(max_digits=12, decimal_places=2)
    
    # Current state
    current_value = models.DecimalField(max_digits=12, decimal_places=2, null=True)
    current_pnl = models.DecimalField(max_digits=12, decimal_places=2, null=True)
    pnl_pct_of_max = models.FloatField(null=True)
    
    # Status
    STATUS_CHOICES = [
        ('open', 'Open'),
        ('closed', 'Closed'),
        ('assigned', 'Assigned'),
        ('rolled', 'Rolled'),
        ('expired', 'Expired'),
    ]
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='open')
    closed_at = models.DateTimeField(null=True, blank=True)
    close_reason = models.CharField(max_length=50, blank=True)
    
    # CSP tracker linkage (replaces wheel)
    csp_tracker = models.ForeignKey('CSPIncomeTracker', on_delete=models.SET_NULL, null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'income_position'
        ordering = ['-entry_date']


class CSPIncomeTracker(models.Model):
    """
    Tracks cash-secured put income strategy performance.
    
    NOTE: Traditional "wheel strategy" is NOT supported because the call
    phase requires naked short calls which have undefined upside loss.
    Use bear call spreads for defined-risk call-side income instead.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    account = models.ForeignKey(ExchangeAccount, on_delete=models.CASCADE)
    name = models.CharField(max_length=100, help_text="User-friendly name for this tracker")
    
    # USDC tracking
    initial_capital_usdc = models.DecimalField(max_digits=12, decimal_places=2)
    current_usdc = models.DecimalField(max_digits=12, decimal_places=2)
    
    # Position tracking
    positions_opened = models.IntegerField(default=0)
    positions_closed = models.IntegerField(default=0)
    
    # Income tracking
    total_premium_collected_usdc = models.DecimalField(max_digits=12, decimal_places=4, default=0)
    total_settlement_paid_usdc = models.DecimalField(max_digits=12, decimal_places=4, default=0)
    
    # Current position reference
    current_position = models.ForeignKey(
        IncomePosition, 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True,
        related_name='csp_tracker_current'
    )
    
    # Status
    is_active = models.BooleanField(default=True)
    started_at = models.DateTimeField(auto_now_add=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'csp_income_tracker'
    
    @property
    def net_income_usdc(self) -> float:
        """Calculate net income (premium - settlement paid)."""
        return float(self.total_premium_collected_usdc) - float(self.total_settlement_paid_usdc)
    
    @property
    def annualized_yield(self) -> float:
        """Calculate annualized yield on initial capital."""
        days_active = (datetime.now() - self.started_at).days
        if days_active > 0 and self.initial_capital > 0:
            return (float(self.total_premium_collected) / float(self.initial_capital)) * (365 / days_active)
        return 0
```


## API Endpoints

### New Income Strategy Endpoints

```python
# api/income_views.py

class IncomeStrategyListView(APIView):
    """
    GET /api/v1/income/strategies/
    
    List all available income strategies with their profiles.
    """
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        strategies = []
        for strategy_type in StrategyType:
            profile = StrategyClassifier.classify(strategy_type)
            strategies.append({
                "type": strategy_type.value,
                "category": profile.category.value,
                "is_income": profile.is_income,
                "is_accumulation": profile.is_accumulation,
                "requires_usdc_collateral": profile.requires_usdc_collateral,
                "settlement_type": profile.settlement_type.value,
                "exercise_style": profile.exercise_style.value,
                "num_legs": profile.num_legs,
                "note": "Deribit Linear USDC options only. European-style, cash-settled.",
            })
        return Response({"strategies": strategies})


class IncomeRecommendView(APIView):
    """
    GET /api/v1/income/recommend/
    
    Get strategy recommendations based on current market regime.
    
    NOTE: This endpoint uses user-supplied balances for SIMULATION only.
    Results are marked as "unverified" and cannot be used for execution.
    
    Query params:
        iv_rank: Current IV rank (0-1)
        usdc_available: USDC available (SIMULATION ONLY)
    """
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        # Get latest signal for regime classification
        latest_signal = DailySignal.active().order_by('-date').first()
        if not latest_signal:
            return Response({"error": "No signal data available"}, status=404)
        
        iv_rank = float(request.query_params.get('iv_rank', 0.5))
        usdc_available = float(request.query_params.get('usdc_available', 0))
        
        # Build SIMULATION account state (NOT exchange-verified)
        account = AccountState(
            available_usdc=usdc_available,
            total_account_value_usdc=usdc_available,
            current_income_exposure_usdc=0,
            current_settlement_exposure_usdc=0,
            is_exchange_verified=False,  # SIMULATION MODE
            verification_timestamp=None,
            exchange="deribit",
        )
        
        # Get recommendations
        engine = RecommendationEngine(get_income_policy())
        regime = engine.classify_regime(
            latest_signal.fusion_state,
            latest_signal.fusion_confidence,
            iv_rank,
        )
        
        recommendations = engine.recommend(regime, account)
        
        return Response({
            "regime": regime.value,
            "fusion_state": latest_signal.fusion_state,
            "iv_rank": iv_rank,
            "validation_mode": "simulation",  # CLEARLY MARKED
            "warning": "Results based on user-supplied balances. Not verified against exchange.",
            "recommendations": [
                {
                    "rank": r.rank,
                    "strategy": r.strategy_type.value,
                    "fit_score": r.fit_score,
                    "rationale": r.rationale,
                    "strike_guidance": r.strike_guidance,
                    "dte_guidance": r.dte_guidance,
                    "size_guidance": r.size_guidance,
                    "warnings": r.warnings,
                }
                for r in recommendations
            ],
        })


class IncomeValidateView(APIView):
    """
    POST /api/v1/income/validate/
    
    Validate a strategy against AUTHENTICATED exchange state.
    This is the ONLY endpoint that can approve strategies for execution.
    
    Request body:
        strategy_type: Strategy type
        strike: Strike price
        expiry_date: Expiry date
        quantity: Number of contracts
        account_id: Exchange account ID (for authenticated balance check)
    """
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        account_id = request.data.get('account_id')
        if not account_id:
            return Response({"error": "account_id required for verified validation"}, status=400)
        
        # Get authenticated exchange account
        try:
            exchange_account = ExchangeAccount.objects.get(id=account_id, is_active=True)
        except ExchangeAccount.DoesNotExist:
            return Response({"error": "Exchange account not found"}, status=404)
        
        # Fetch REAL balances from exchange
        adapter = DeribitAdapter.from_account(exchange_account)
        exchange_balances = adapter.get_account_summary()
        
        # Build VERIFIED account state
        account = AccountState(
            available_usdc=float(exchange_balances.get('available_usdc', 0)),
            total_account_value_usdc=float(exchange_balances.get('equity_usdc', 0)),
            current_income_exposure_usdc=0,  # Calculate from open positions
            current_settlement_exposure_usdc=0,
            is_exchange_verified=True,  # VERIFIED MODE
            verification_timestamp=datetime.now(timezone.utc),
            exchange="deribit",
        )
        
        # Parse strategy request
        strategy_type = StrategyType(request.data.get('strategy_type'))
        # ... build strategy request ...
        
        # Validate with exchange-verified state
        validator = SafetyValidator(get_income_policy())
        result = validator.validate(
            strategy=strategy_request,
            account=account,
            require_exchange_verification=True,
        )
        
        return Response({
            "is_valid": result.is_valid,
            "validation_mode": "exchange_verified",
            "verification_timestamp": account.verification_timestamp.isoformat(),
            "exchange_usdc_balance": account.available_usdc,
            "rejections": [
                {
                    "code": r.code.value,
                    "severity": r.severity.value,
                    "message": r.message,
                    "details": r.details,
                }
                for r in result.rejections
            ],
            "warnings": [
                {
                    "code": w.code.value,
                    "message": w.message,
                }
                for w in result.warnings
            ],
        })


class IncomeSetupView(APIView):
    """
    POST /api/v1/income/setup/
    
    Build a complete income strategy setup.
    
    For SIMULATION: accepts user-supplied balances, marked as unverified.
    For EXECUTION: requires account_id for exchange-verified validation.
    
    Request body:
        strategy_type: Strategy type (e.g., "cash_secured_put")
        strike: Strike price (optional, will recommend if not provided)
        expiry_date: Expiry date (YYYY-MM-DD)
        quantity: Number of contracts
        
        # For simulation (unverified):
        usdc_available: USDC available (simulation only)
        
        # For execution (verified):
        account_id: Exchange account ID (required for execution)
        mode: "simulation" or "execution" (default: simulation)
    """
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        mode = request.data.get('mode', 'simulation')
        
        if mode == 'execution':
            # EXECUTION MODE: Requires exchange-verified state
            account_id = request.data.get('account_id')
            if not account_id:
                return Response({
                    "error": "account_id required for execution mode",
                    "hint": "Use mode='simulation' for preview without exchange verification",
                }, status=400)
            
            # Fetch real balances from exchange
            exchange_account = ExchangeAccount.objects.get(id=account_id)
            adapter = DeribitAdapter.from_account(exchange_account)
            exchange_balances = adapter.get_account_summary()
            
            account = AccountState(
                available_usdc=float(exchange_balances.get('available_usdc', 0)),
                total_account_value_usdc=float(exchange_balances.get('equity_usdc', 0)),
                current_income_exposure_usdc=0,
                current_settlement_exposure_usdc=0,
                is_exchange_verified=True,
                verification_timestamp=datetime.now(timezone.utc),
                exchange="deribit",
            )
        else:
            # SIMULATION MODE: User-supplied balances (unverified)
            usdc_available = float(request.data.get('usdc_available', 0))
            
            if usdc_available <= 0:
                return Response({
                    "error": "usdc_available required for simulation mode",
                }, status=400)
            
            account = AccountState(
                available_usdc=usdc_available,
                total_account_value_usdc=usdc_available,
                current_income_exposure_usdc=0,
                current_settlement_exposure_usdc=0,
                is_exchange_verified=False,
                verification_timestamp=None,
                exchange="deribit",
            )
        
        # Parse strategy request
        strategy_type = StrategyType(request.data.get('strategy_type'))
        strike = request.data.get('strike')
        expiry_date = request.data.get('expiry_date')
        quantity = int(request.data.get('quantity', 1))
        
        # Build setup using IncomeStrategyBuilder
        builder = IncomeStrategyBuilder()
        setup = builder.build_setup(
            strategy_type=strategy_type,
            strike=strike,
            expiry_date=expiry_date,
            quantity=quantity,
            account=account,
        )
        
        if setup is None:
            return Response({"error": "Could not build setup"}, status=400)
        
        response_data = setup.to_dict()
        response_data["validation_mode"] = "exchange_verified" if account.is_exchange_verified else "simulation"
        
        if not account.is_exchange_verified:
            response_data["warning"] = (
                "SIMULATION MODE: Balances not verified against exchange. "
                "This setup cannot be executed. Use mode='execution' with account_id for real trades."
            )
        
        return Response(response_data)


class CSPIncomeStatusView(APIView):
    """
    GET /api/v1/income/csp/status/
    GET /api/v1/income/csp/status/<tracker_id>/
    
    Get cash-secured put income strategy status and metrics.
    
    NOTE: Traditional "wheel strategy" is NOT supported because the call
    phase requires naked short calls which have undefined upside loss.
    """
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]
    
    def get(self, request, tracker_id=None):
        if tracker_id:
            tracker = CSPIncomeTracker.objects.get(id=tracker_id)
            csp_manager = CSPIncomeTracker(get_income_policy())
            metrics = csp_manager.get_metrics(tracker_id)
            
            return Response({
                "tracker_id": str(tracker.id),
                "name": tracker.name,
                "note": "Cash-settled puts only. Short put ITM = PAY settlement (loss). No physical BTC delivery.",
                "metrics": {
                    "total_premium_collected_usdc": float(tracker.total_premium_collected_usdc),
                    "total_settlement_paid_usdc": float(tracker.total_settlement_paid_usdc),
                    "net_income_usdc": metrics.net_income_usdc,
                    "annualized_yield_pct": metrics.annualized_yield_pct,
                    "positions_opened": tracker.positions_opened,
                    "positions_closed": tracker.positions_closed,
                    "otm_expiry_rate": metrics.otm_expiry_rate,
                },
                "current_position": tracker.current_position.id if tracker.current_position else None,
            })
        else:
            # List all trackers for account
            trackers = CSPIncomeTracker.objects.filter(is_active=True)
            return Response({
                "trackers": [
                    {
                        "id": str(t.id),
                        "name": t.name,
                        "total_premium_usdc": float(t.total_premium_collected_usdc),
                        "net_income_usdc": t.net_income_usdc,
                    }
                    for t in trackers
                ]
            })
```


## Policy Configuration

### Income Strategy Policy Extension

```python
# execution/services/policy.py additions

@dataclass
class IncomeStrategyConfig:
    """Configuration for income strategy module."""
    # Risk limits
    max_single_position_risk_pct: float = 0.05
    max_total_income_exposure_pct: float = 0.30
    max_settlement_exposure_pct: float = 0.50
    max_spread_width_usd: float = 5000
    max_single_expiry_concentration_pct: float = 0.20
    
    # Exit rules
    credit_spread_profit_target_pct: float = 0.50  # Close at 50% profit
    credit_spread_loss_limit_pct: float = 2.00     # Close at 200% loss
    iron_condor_management_dte: int = 21           # Manage at 21 DTE
    
    # CSP parameters (replaces wheel - wheel required naked calls)
    csp_put_otm_pct: float = 0.07                 # 7% OTM for puts
    csp_target_dte: int = 30                       # 30 DTE target
    
    # Premium efficiency thresholds
    min_annualized_yield_pct: float = 0.10        # 10% min annualized
    min_risk_reward_ratio: float = 0.20           # 1:5 min R:R for credit


def get_income_policy() -> IncomeStrategyConfig:
    """Get current income strategy policy configuration."""
    return IncomeStrategyConfig()
```

## Integration Points

### Integration with Existing TradeSetupBuilder

```python
# execution/services/trade_setup.py additions

class TradeSetupBuilder:
    # ... existing code ...
    
    def build_income_setup(
        self,
        strategy_type: StrategyType,
        strike: Optional[float] = None,
        expiry_date: Optional[date] = None,
        quantity: int = 1,
        account: Optional[AccountState] = None,
    ) -> Optional["IncomeStrategySetup"]:
        """
        Build a complete income strategy setup.
        
        Integrates with existing option snapshot selection and validation.
        """
        # Get option snapshots using existing logic
        options = self._get_option_snapshots(expiry_date)
        if not options:
            return None
        
        # Build strategy-specific setup
        if strategy_type == StrategyType.CASH_SECURED_PUT:
            return self._build_csp_setup(options, strike, quantity, account)
        elif strategy_type == StrategyType.BULL_PUT_SPREAD:
            return self._build_bull_put_spread(options, strike, quantity, account)
        elif strategy_type == StrategyType.BEAR_CALL_SPREAD:
            return self._build_bear_call_spread(options, strike, quantity, account)
        # ... etc
        # NOTE: SYNTHETIC_COVERED_CALL not supported - naked calls have undefined loss
        
        return None
```

## Correctness Properties

### Property 1: No Undefined Risk

*For any* strategy returned by the system, the `max_loss` field SHALL be a finite, positive number. The system SHALL NOT return any strategy where `max_loss` is infinite, undefined, or cannot be calculated.

**Validates: Safety Constraints S1, S2, S3**

### Property 2: Collateral Sufficiency

*For any* cash-secured put, the `collateral_required_usdc` SHALL equal `strike * contract_size * quantity`. *For any* synthetic covered call, the `collateral_required_usdc` SHALL meet exchange margin requirements. The system SHALL NOT approve any strategy where available USDC collateral is less than required.

**Validates: Safety Constraint S4, Requirement 2.3**

### Property 3: Protective Leg Presence and Completeness

*For any* credit spread or iron condor, every short option leg SHALL have a corresponding long option leg that meets ALL of the following:
- Same underlying (BTC)
- Same expiry date
- Same settlement currency (USDC)
- Same option type (call/put)
- Correct strike ordering (long put ≤ short put; long call ≥ short call)
- Equal or greater quantity

The system SHALL reject any spread where a short leg lacks complete protection.

**Validates: Safety Constraint S5, Requirement 2.3**

### Property 4: Rejection Completeness

*For any* strategy in the prohibited list (naked calls, naked puts on margin, short straddles, short strangles, ratio spreads with uncovered tails, leveraged short structures), the system SHALL return `is_valid=False` with the appropriate rejection code.

**Validates: Safety Constraint S6, Requirement 2.2**

### Property 5: Payoff Calculation Accuracy

*For any* cash-secured put with strike S and premium P, `max_profit` SHALL equal `P * contract_size * qty` and `max_loss` SHALL equal `(S - P) * contract_size * qty`. *For any* credit spread with width W and net credit C, `max_profit` SHALL equal `C` and `max_loss` SHALL equal `(W - C) * contract_size * qty`.

**Validates: Requirement 3.1**

### Property 6: Risk Limit Enforcement

*For any* strategy where `max_loss / account_value > max_single_position_risk_pct`, the system SHALL return `is_approved=False` with `RISK_LIMIT_EXCEEDED` code.

**Validates: Requirement 4.1, 4.4**

### Property 7: Regime Classification Consistency

*For any* IV rank > 0.60, the system SHALL classify regime as `HIGH_IV`. *For any* IV rank < 0.30, the system SHALL classify regime as `LOW_IV`. *For any* fusion_state in bullish set, the system SHALL classify regime as `BULLISH` (unless IV override applies).

**Validates: Requirement 5.1**

### Property 8: CSP Settlement Handling

*For any* short put ITM cash settlement event, the system SHALL record a settlement payment (loss) equal to `(strike - settlement_price) * quantity`. *For any* short put OTM expiry, the system SHALL record the full premium as income (gain). Note: These are CASH SETTLEMENTS, not physical assignments. There is NO physical BTC delivery on Deribit.

**Validates: Requirement 6.2, 6.3**

## File Structure

```
execution/
├── services/
│   ├── policy.py                    # Extended with IncomeStrategyConfig
│   ├── trade_setup.py               # Extended with build_income_setup()
│   ├── trade_validator.py           # Extended with income validation
│   └── income/
│       ├── __init__.py
│       ├── classifier.py            # StrategyClassifier
│       ├── validator.py             # SafetyValidator
│       ├── payoff.py                # PayoffCalculator
│       ├── risk.py                  # RiskEngine
│       ├── recommendation.py        # RecommendationEngine
│       └── csp_tracker.py           # CSPIncomeTracker (replaces wheel)
├── models.py                        # Extended with IncomePosition, CSPIncomeTracker

api/
├── urls.py                          # Extended with income endpoints
├── income_views.py                  # New income strategy views
└── income_serializers.py            # New serializers

datafeed/
└── models.py                        # Existing OptionSnapshot (reused)
```

## Dependencies

| Dependency | Purpose |
|------------|---------|
| `execution.services.policy` | Policy configuration |
| `execution.services.trade_validator` | Validation infrastructure |
| `datafeed.models.OptionSnapshot` | Option market data |
| `signals.models.DailySignal` | Regime classification |
| `execution.models.ExchangeAccount` | Account management |
