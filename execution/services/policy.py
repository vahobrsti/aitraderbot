"""
Execution Policy Configuration - Versioned, Auditable

Single source of truth for all execution parameters. Versioned for
reproducibility and safe recalibration.

Usage:
    from execution.services.policy import get_policy, PolicyVersion
    
    policy = get_policy()  # Current active policy
    policy = get_policy(version="2026-05-01.1")  # Specific version
    
    # Access parameters
    tier = policy.get_tier("MVRV_SHORT")
    dte = policy.get_dte_target("PUT")
    delta = policy.get_delta_target("slight_itm", "put")
"""
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional
import json


@dataclass
class TierConfig:
    """Risk tier configuration."""
    risk_usd: float
    naked_pct: float
    spread_pct: float
    max_concurrent: int = 2


@dataclass
class DTEConfig:
    """DTE targeting configuration."""
    min_dte: int
    max_dte: int
    optimal_dte: int


@dataclass
class LiquidityConfig:
    """Liquidity filtering thresholds."""
    min_open_interest: Decimal = Decimal("5")
    max_spread_pct: float = 0.15
    min_bid_btc: Decimal = Decimal("0.0001")
    max_slippage_pct: float = 0.02  # Max acceptable slippage from mid


@dataclass
class ExecutionCostConfig:
    """Execution cost estimates for edge calculation."""
    maker_fee_pct: float = 0.0003  # 0.03% maker
    taker_fee_pct: float = 0.0005  # 0.05% taker
    spread_cost_pct: float = 0.01  # Estimated half-spread cost
    slippage_pct: float = 0.005    # Expected slippage
    
    def total_cost_per_leg(self, is_market: bool = True) -> float:
        """Total estimated cost per leg as % of notional."""
        fee = self.taker_fee_pct if is_market else self.maker_fee_pct
        return fee + self.spread_cost_pct + self.slippage_pct
    
    def total_cost_multi_leg(self, num_legs: int, is_market: bool = True) -> float:
        """Total cost for multi-leg trade."""
        return self.total_cost_per_leg(is_market) * num_legs


@dataclass
class CondorConfig:
    """Iron condor specific configuration."""
    wing_offset_usd: float = 2000
    min_credit_pct: float = 0.15  # Min credit as % of wing width
    max_concurrent: int = 1
    drift_multiplier: float = 1.5
    spot_call_band: float = 0.10
    spot_put_band: float = 0.10


@dataclass 
class PolicyVersion:
    """
    Complete execution policy - versioned and immutable once deployed.
    
    Version format: YYYY-MM-DD.N (date + revision number)
    """
    version: str
    description: str
    
    # Tier configurations
    tiers: dict[str, TierConfig] = field(default_factory=dict)
    
    # Signal -> Tier mapping
    signal_tier_map: dict[str, int] = field(default_factory=dict)
    
    # DTE targets per signal type
    dte_targets: dict[str, DTEConfig] = field(default_factory=dict)
    
    # Delta targets per strike guidance
    delta_targets: dict[str, dict[str, float]] = field(default_factory=dict)
    
    # Spread configuration
    spread_enabled: dict[str, bool] = field(default_factory=dict)
    spread_width_pct: dict[str, float] = field(default_factory=dict)
    
    # Liquidity and execution
    liquidity: LiquidityConfig = field(default_factory=LiquidityConfig)
    execution_costs: ExecutionCostConfig = field(default_factory=ExecutionCostConfig)
    
    # Condor specific
    condor: CondorConfig = field(default_factory=CondorConfig)
    
    # Scoring weights
    scoring_weights: dict[str, float] = field(default_factory=dict)
    
    # Global constraints
    max_account_risk_pct: float = 0.06  # Max 6% of account at risk
    snapshot_staleness_hours: int = 2
    
    def get_tier(self, signal_type: str) -> TierConfig:
        """Get tier config for a signal type."""
        tier_num = self.signal_tier_map.get(signal_type, 2)
        return self.tiers.get(str(tier_num), self.tiers["2"])
    
    def get_dte_target(self, signal_type: str) -> DTEConfig:
        """Get DTE target for a signal type."""
        return self.dte_targets.get(
            signal_type, 
            DTEConfig(min_dte=11, max_dte=14, optimal_dte=12)
        )
    
    def get_delta_target(self, strike_guidance: str, option_type: str) -> float:
        """Get target delta for strike guidance and option type."""
        guidance = self.delta_targets.get(
            strike_guidance, 
            self.delta_targets.get("slight_itm", {"call": 0.60, "put": -0.60})
        )
        return guidance.get(option_type, 0.50 if option_type == "call" else -0.50)
    
    def is_spread_enabled(self, signal_type: str) -> bool:
        """Check if spread construction is enabled for signal type."""
        return self.spread_enabled.get(signal_type, True)
    
    def get_spread_width(self, signal_type: str) -> float:
        """Get spread width % for signal type."""
        return self.spread_width_pct.get(signal_type, 0.10)
    
    def estimate_edge_after_costs(
        self, 
        expected_return_pct: float,
        num_legs: int,
        is_market: bool = True,
    ) -> float:
        """
        Estimate edge after execution costs.
        
        Returns negative if costs exceed expected return.
        """
        total_cost = self.execution_costs.total_cost_multi_leg(num_legs, is_market)
        return expected_return_pct - total_cost
    
    def to_dict(self) -> dict:
        """Serialize policy to dict for storage."""
        return {
            "version": self.version,
            "description": self.description,
            "tiers": {k: vars(v) for k, v in self.tiers.items()},
            "signal_tier_map": self.signal_tier_map,
            "dte_targets": {k: vars(v) for k, v in self.dte_targets.items()},
            "delta_targets": self.delta_targets,
            "spread_enabled": self.spread_enabled,
            "spread_width_pct": self.spread_width_pct,
            "liquidity": vars(self.liquidity),
            "execution_costs": vars(self.execution_costs),
            "condor": vars(self.condor),
            "scoring_weights": self.scoring_weights,
            "max_account_risk_pct": self.max_account_risk_pct,
            "snapshot_staleness_hours": self.snapshot_staleness_hours,
        }


# =============================================================================
# POLICY VERSIONS
# =============================================================================

POLICY_V1 = PolicyVersion(
    version="2026-05-03.1",
    description="V7 path-stable policy with spread support for all signals",
    
    tiers={
        "1": TierConfig(risk_usd=4000, naked_pct=0.20, spread_pct=0.80),
        "2": TierConfig(risk_usd=2400, naked_pct=0.25, spread_pct=0.75),
    },
    
    signal_tier_map={
        "PRIMARY_SHORT": 1,
        "OPTION_CALL": 1,
        "CALL": 1,
        "PUT": 1,
        "BULL_PROBE": 2,
        "BEAR_PROBE": 2,
        "TACTICAL_PUT": 2,
        "OPTION_PUT": 2,
        "MVRV_SHORT": 2,
        "IRON_CONDOR": 2,
    },
    
    dte_targets={
        "CALL": DTEConfig(min_dte=12, max_dte=21, optimal_dte=14),
        "PUT": DTEConfig(min_dte=10, max_dte=14, optimal_dte=12),
        "OPTION_CALL": DTEConfig(min_dte=11, max_dte=14, optimal_dte=12),
        "OPTION_PUT": DTEConfig(min_dte=11, max_dte=14, optimal_dte=12),
        "TACTICAL_PUT": DTEConfig(min_dte=10, max_dte=14, optimal_dte=12),
        "BULL_PROBE": DTEConfig(min_dte=10, max_dte=14, optimal_dte=12),
        "BEAR_PROBE": DTEConfig(min_dte=12, max_dte=16, optimal_dte=14),
        "IRON_CONDOR": DTEConfig(min_dte=10, max_dte=21, optimal_dte=14),
        "MVRV_SHORT": DTEConfig(min_dte=10, max_dte=14, optimal_dte=12),  # Shortened for options
    },
    
    delta_targets={
        "slight_itm": {"call": 0.60, "put": -0.60},
        "atm": {"call": 0.50, "put": -0.50},
        "slight_otm": {"call": 0.35, "put": -0.35},
        "otm": {"call": 0.20, "put": -0.20},
        "itm": {"call": 0.70, "put": -0.70},
        "deep_otm": {"call": 0.10, "put": -0.10},
    },
    
    # Enable spreads for ALL signals including MVRV_SHORT
    spread_enabled={
        "CALL": True,
        "PUT": True,
        "OPTION_CALL": True,
        "OPTION_PUT": True,
        "TACTICAL_PUT": True,
        "BULL_PROBE": True,
        "BEAR_PROBE": True,
        "MVRV_SHORT": True,  # NOW ENABLED
        "IRON_CONDOR": True,
    },
    
    spread_width_pct={
        "CALL": 0.10,
        "PUT": 0.10,
        "OPTION_CALL": 0.10,
        "OPTION_PUT": 0.10,
        "TACTICAL_PUT": 0.08,
        "BULL_PROBE": 0.07,
        "BEAR_PROBE": 0.07,
        "MVRV_SHORT": 0.065,  # ~$5k width at $77k spot (like your trade)
        "IRON_CONDOR": 0.10,
    },
    
    liquidity=LiquidityConfig(
        min_open_interest=Decimal("5"),
        max_spread_pct=0.15,
        min_bid_btc=Decimal("0.0001"),
        max_slippage_pct=0.02,
    ),
    
    execution_costs=ExecutionCostConfig(
        maker_fee_pct=0.0003,
        taker_fee_pct=0.0005,
        spread_cost_pct=0.01,
        slippage_pct=0.005,
    ),
    
    condor=CondorConfig(
        wing_offset_usd=2000,
        min_credit_pct=0.15,
        max_concurrent=1,
        drift_multiplier=1.5,
        spot_call_band=0.10,
        spot_put_band=0.10,
    ),
    
    # Scoring weights for candidate ranking
    scoring_weights={
        "delta": 0.40,
        "dte": 0.30,
        "liquidity": 0.20,
        "iv": 0.10,
    },
    
    max_account_risk_pct=0.06,
    snapshot_staleness_hours=2,
)


# Active policy registry
_POLICIES = {
    "2026-05-03.1": POLICY_V1,
}

_ACTIVE_VERSION = "2026-05-03.1"


def get_policy(version: Optional[str] = None) -> PolicyVersion:
    """
    Get execution policy by version.
    
    Args:
        version: Specific version string, or None for active policy.
    
    Returns:
        PolicyVersion instance.
    """
    v = version or _ACTIVE_VERSION
    if v not in _POLICIES:
        raise ValueError(f"Unknown policy version: {v}. Available: {list(_POLICIES.keys())}")
    return _POLICIES[v]


def get_active_version() -> str:
    """Get the currently active policy version string."""
    return _ACTIVE_VERSION


def list_versions() -> list[str]:
    """List all available policy versions."""
    return list(_POLICIES.keys())
