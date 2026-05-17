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
from pathlib import Path
from typing import Optional
import copy
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
class RecoveryConfig:
    """
    Recovery policy parameters per signal type.
    
    Recovery parameters for trades that are adverse at checkpoint:
    - recovery_flip_threshold: MFE threshold above which to flip the trade direction
    - recovery_cut_threshold: MFE threshold below which to cut losses
    - recovery_target: Profit target for recovery trades (typically lower than original)
    """
    recovery_flip_threshold: float = 0.05    # 5% MFE threshold to recommend flipping
    recovery_cut_threshold: float = 0.02     # 2% MFE threshold below which to cut losses
    recovery_target: float = 0.03            # 3% profit target for recovery trades


@dataclass
class ExitConfig:
    """
    Exit policy parameters per signal type.
    
    Path-aware exit features:
    - Trailing stop: Activates after profit_lock_threshold, trails at trailing_stop_pct
    - Time-based tightening: Stop tightens by stop_tighten_factor at stop_tighten_day
    - Profit lock: Move stop to breakeven after profit_lock_threshold reached
    """
    stop_loss_pct: float
    take_profit_pct: float
    max_hold_days: int
    scale_down_day: Optional[int] = None
    # Path-aware exit parameters
    profit_lock_threshold: float = 0.30      # Lock profits after 30% of max profit
    trailing_stop_pct: float = 0.0           # 0 = no trailing, >0 = trail at this % below high
    stop_tighten_day: Optional[int] = None   # Day to tighten stop (None = no tightening)
    stop_tighten_factor: float = 0.5         # Multiply stop by this factor on tighten day
    # Recovery policy integration
    recovery: Optional[RecoveryConfig] = None  # Recovery parameters for adverse trades


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
    exit_params: dict[str, ExitConfig] = field(default_factory=dict)
    expected_edge_by_signal: dict[str, float] = field(default_factory=dict)
    
    # Per-signal delta targets (calibrated from MAE analysis)
    signal_delta_targets: dict[str, float] = field(default_factory=dict)
    
    # Path profile data (from analyze_path_stats)
    # shakeout_pct: % of winners that experience shakeout before hitting target
    # invalidation_pct: % of winners invalidated before hit
    path_profiles: dict[str, dict] = field(default_factory=dict)
    
    # Recovery policy parameters per signal type
    recovery_configs: dict[str, RecoveryConfig] = field(default_factory=dict)
    
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

    def get_signal_delta(self, signal_type: str) -> float:
        """
        Get target delta for a signal type.
        
        Calibrated from MAE analysis:
        - Higher MAE = need more ITM cushion (higher absolute delta)
        - Lower MAE = can use ATM or slight OTM
        """
        return self.signal_delta_targets.get(signal_type, 0.55)

    def get_exit_params(self, signal_type: str) -> Optional[ExitConfig]:
        """Get exit params for a signal type, if configured."""
        return self.exit_params.get(signal_type)

    def get_expected_edge(self, signal_type: str, default: float = 0.08) -> float:
        """Get expected edge for cost checks."""
        return self.expected_edge_by_signal.get(signal_type, default)
    
    def get_path_profile(self, signal_type: str) -> dict:
        """
        Get path profile for a signal type.
        
        Returns dict with:
        - shakeout_pct: % of winners with shakeout path
        - invalidation_pct: % of winners invalidated before hit
        - mae_p75: 75th percentile MAE for winners
        - clean_win_pct: % of winners with clean paths
        """
        return self.path_profiles.get(signal_type, {
            "shakeout_pct": 0.20,
            "invalidation_pct": 0.25,
            "mae_p75": 0.05,
            "clean_win_pct": 0.75,
        })
    
    def has_path_profile(self, signal_type: str) -> bool:
        """Check if a signal type has an explicitly configured path profile."""
        return signal_type in self.path_profiles
    
    def is_shakeout_heavy(self, signal_type: str) -> bool:
        """
        Check if signal type has shakeout-heavy paths (>40% shakeout).
        
        Shakeout-heavy signals need:
        - DCA entry strategy
        - Wider stops
        - ITM strikes for cushion
        """
        profile = self.get_path_profile(signal_type)
        return profile.get("shakeout_pct", 0) >= 0.40
    
    def is_invalidation_heavy(self, signal_type: str) -> bool:
        """
        Check if signal type has high invalidation rate (>35%).
        
        High invalidation signals need:
        - ITM strikes
        - Wider stops
        - Smaller position size
        """
        profile = self.get_path_profile(signal_type)
        return profile.get("invalidation_pct", 0) >= 0.35
    
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
    
    def get_recovery_config(self, signal_type: str) -> RecoveryConfig:
        """
        Get recovery configuration for a signal type.
        
        Checks top-level recovery_configs first, then falls back to
        ExitConfig.recovery if available, then defaults.
        
        Args:
            signal_type: Signal type (e.g., "CALL", "PUT", "MVRV_SHORT")
            
        Returns:
            RecoveryConfig with thresholds and targets for recovery decisions
        """
        # Primary: top-level recovery_configs
        if signal_type in self.recovery_configs:
            return self.recovery_configs[signal_type]
        
        # Fallback: ExitConfig.recovery field (if populated via calibration)
        exit_cfg = self.exit_params.get(signal_type)
        if exit_cfg and exit_cfg.recovery:
            return exit_cfg.recovery
        
        # Default
        return RecoveryConfig()
    
    def get_recovery_flip_threshold(self, signal_type: str) -> float:
        """Get recovery flip threshold for a signal type."""
        return self.get_recovery_config(signal_type).recovery_flip_threshold
    
    def get_recovery_cut_threshold(self, signal_type: str) -> float:
        """Get recovery cut threshold for a signal type."""
        return self.get_recovery_config(signal_type).recovery_cut_threshold
    
    def get_recovery_target(self, signal_type: str) -> float:
        """Get recovery target for a signal type."""
        return self.get_recovery_config(signal_type).recovery_target
    
    def to_dict(self) -> dict:
        """Serialize policy to dict for storage."""
        return {
            "version": self.version,
            "description": self.description,
            "tiers": {k: vars(v) for k, v in self.tiers.items()},
            "signal_tier_map": self.signal_tier_map,
            "dte_targets": {k: vars(v) for k, v in self.dte_targets.items()},
            "delta_targets": self.delta_targets,
            "signal_delta_targets": self.signal_delta_targets,
            "spread_enabled": self.spread_enabled,
            "spread_width_pct": self.spread_width_pct,
            "exit_params": {
                k: {
                    **{field: val for field, val in vars(v).items() if field != "recovery"},
                    "recovery": vars(v.recovery) if v.recovery else None,
                }
                for k, v in self.exit_params.items()
            },
            "expected_edge_by_signal": self.expected_edge_by_signal,
            "path_profiles": self.path_profiles,
            "recovery_configs": {k: vars(v) for k, v in self.recovery_configs.items()},
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
    version="2026-05-03.3",
    description="Path-analysis calibrated policy with corrected IRON_CONDOR (63.1% win rate, 7d range-bound)",
    
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
    
    # DTE targets calibrated from TTH p75 + 2d buffer
    # Source: analyze_path_stats (14d horizon, 5% target, 4% invalidation)
    dte_targets={
        "CALL": DTEConfig(min_dte=9, max_dte=14, optimal_dte=11),       # TTH p75=7d
        "PUT": DTEConfig(min_dte=8, max_dte=12, optimal_dte=10),        # TTH p75=6d
        "OPTION_CALL": DTEConfig(min_dte=7, max_dte=12, optimal_dte=9), # TTH p75=5d, fast signal
        "OPTION_PUT": DTEConfig(min_dte=5, max_dte=10, optimal_dte=7),  # TTH p75=3d, fastest
        "TACTICAL_PUT": DTEConfig(min_dte=8, max_dte=12, optimal_dte=10),  # TTH p75=5.5d
        "BULL_PROBE": DTEConfig(min_dte=7, max_dte=11, optimal_dte=9),  # TTH p75=5d, fast
        "BEAR_PROBE": DTEConfig(min_dte=11, max_dte=16, optimal_dte=13),  # TTH p75=8.5d, slow
        "IRON_CONDOR": DTEConfig(min_dte=9, max_dte=13, optimal_dte=9),  # 7d horizon + 2d buffer
        "MVRV_SHORT": DTEConfig(min_dte=12, max_dte=18, optimal_dte=14),  # TTH p75=10d, slowest
    },
    
    delta_targets={
        "slight_itm": {"call": 0.60, "put": -0.60},
        "atm": {"call": 0.50, "put": -0.50},
        "slight_otm": {"call": 0.35, "put": -0.35},
        "otm": {"call": 0.20, "put": -0.20},
        "itm": {"call": 0.70, "put": -0.70},
        "deep_otm": {"call": 0.10, "put": -0.10},
    },
    
    # Per-signal delta targets based on MAE analysis
    # Higher MAE = need more ITM cushion
    signal_delta_targets={
        "CALL": 0.60,          # MAE(W) p75=4.71%, slight ITM
        "PUT": -0.60,          # MAE(W) p75=4.41%, slight ITM
        "OPTION_CALL": 0.65,   # MAE(W) p75=8.48%, need ITM cushion
        "OPTION_PUT": -0.65,   # MAE(W) p75=6.82%, need ITM cushion
        "TACTICAL_PUT": -0.40, # MAE(W) p75=3.08%, hedge so slight OTM
        "BULL_PROBE": 0.55,    # MAE(W) p75=3.84%, ATM to slight ITM
        "BEAR_PROBE": -0.55,   # MAE(W) p75=6.53%, need cushion
        "IRON_CONDOR": 0.20,   # OTM wings for premium selling
        "MVRV_SHORT": -0.60,   # MAE(W) p75=7.19%, shakeout-heavy
    },
    
    # Path profile data from analyze_path_stats (14d horizon, 5% target)
    # shakeout_pct: % of winners with "shakeout then expansion" path shape
    # invalidation_pct: % of winners invalidated before hitting target
    # mae_p75: 75th percentile MAE for winners
    # clean_win_pct: % of winners with clean paths (no invalidation)
    path_profiles={
        "CALL": {"shakeout_pct": 0.21, "invalidation_pct": 0.27, "mae_p75": 0.0471, "clean_win_pct": 0.729},
        "PUT": {"shakeout_pct": 0.18, "invalidation_pct": 0.29, "mae_p75": 0.0441, "clean_win_pct": 0.714},
        "OPTION_CALL": {"shakeout_pct": 0.35, "invalidation_pct": 0.54, "mae_p75": 0.0848, "clean_win_pct": 0.458},
        "OPTION_PUT": {"shakeout_pct": 0.31, "invalidation_pct": 0.44, "mae_p75": 0.0682, "clean_win_pct": 0.562},
        "TACTICAL_PUT": {"shakeout_pct": 0.15, "invalidation_pct": 0.25, "mae_p75": 0.0308, "clean_win_pct": 0.750},
        "BULL_PROBE": {"shakeout_pct": 0.19, "invalidation_pct": 0.22, "mae_p75": 0.0384, "clean_win_pct": 0.781},
        "BEAR_PROBE": {"shakeout_pct": 0.40, "invalidation_pct": 0.40, "mae_p75": 0.0653, "clean_win_pct": 0.600},
        "MVRV_SHORT": {"shakeout_pct": 0.57, "invalidation_pct": 0.43, "mae_p75": 0.0719, "clean_win_pct": 0.571},
        "IRON_CONDOR": {"shakeout_pct": 0.0, "invalidation_pct": 0.0, "mae_p75": 0.0676, "clean_win_pct": 1.0},
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
        "MVRV_SHORT": True,
        "IRON_CONDOR": True,
    },
    
    # Spread width calibrated from MFE p75 × 0.65-0.70
    spread_width_pct={
        "CALL": 0.10,          # MFE p75=15.03% × 0.67 = 10%
        "PUT": 0.09,           # MFE p75=14.28% × 0.65 = 9.3%
        "OPTION_CALL": 0.10,   # MFE p75=15.26% × 0.67 = 10%
        "OPTION_PUT": 0.12,    # MFE p75=23.08% × 0.52 = 12% (capped)
        "TACTICAL_PUT": 0.07,  # MFE p75=10.40% × 0.67 = 7%
        "BULL_PROBE": 0.12,    # MFE p75=28.24% × 0.42 = 12% (capped, highest MFE)
        "BEAR_PROBE": 0.08,    # MFE p75=12.13% × 0.67 = 8%
        "IRON_CONDOR": 0.10,   # Standard condor width
        "MVRV_SHORT": 0.09,    # MFE p75=13.90% × 0.65 = 9%
    },
    
    # Exit params calibrated from MAE(W) p75 for stop loss
    # Path-aware enhancements:
    # - Shakeout-heavy (MVRV_SHORT, BEAR_PROBE): trailing stop + delayed tightening
    # - Invalidation-heavy (OPTION_CALL, OPTION_PUT): lower take-profit + early tightening
    # - Clean signals (CALL, PUT, BULL_PROBE): standard exits
    exit_params={
        # Clean signals - standard exits
        "CALL": ExitConfig(
            stop_loss_pct=0.045, take_profit_pct=0.70, max_hold_days=9, scale_down_day=6,
            profit_lock_threshold=0.30, trailing_stop_pct=0.0,
            stop_tighten_day=6, stop_tighten_factor=0.6,
            recovery=RecoveryConfig(recovery_flip_threshold=0.05, recovery_cut_threshold=0.02, recovery_target=0.03),
        ),
        "PUT": ExitConfig(
            stop_loss_pct=0.045, take_profit_pct=0.70, max_hold_days=8, scale_down_day=5,
            profit_lock_threshold=0.30, trailing_stop_pct=0.0,
            stop_tighten_day=5, stop_tighten_factor=0.6,
            recovery=RecoveryConfig(recovery_flip_threshold=0.05, recovery_cut_threshold=0.02, recovery_target=0.03),
        ),
        # Invalidation-heavy signals - lower take-profit, early tightening
        "OPTION_CALL": ExitConfig(
            stop_loss_pct=0.085, take_profit_pct=0.50, max_hold_days=7, scale_down_day=4,
            profit_lock_threshold=0.25, trailing_stop_pct=0.0,
            stop_tighten_day=3, stop_tighten_factor=0.5,  # Tighten early due to 54% invalidation
            recovery=RecoveryConfig(recovery_flip_threshold=0.05, recovery_cut_threshold=0.02, recovery_target=0.03),
        ),
        "OPTION_PUT": ExitConfig(
            stop_loss_pct=0.070, take_profit_pct=0.50, max_hold_days=5, scale_down_day=3,
            profit_lock_threshold=0.25, trailing_stop_pct=0.0,
            stop_tighten_day=2, stop_tighten_factor=0.5,  # Tighten early due to 44% invalidation
            recovery=RecoveryConfig(recovery_flip_threshold=0.05, recovery_cut_threshold=0.02, recovery_target=0.03),
        ),
        # Clean hedge signal
        "TACTICAL_PUT": ExitConfig(
            stop_loss_pct=0.035, take_profit_pct=0.70, max_hold_days=8, scale_down_day=5,
            profit_lock_threshold=0.30, trailing_stop_pct=0.0,
            stop_tighten_day=5, stop_tighten_factor=0.6,
            recovery=RecoveryConfig(recovery_flip_threshold=0.05, recovery_cut_threshold=0.02, recovery_target=0.03),
        ),
        # Clean probe signal
        "BULL_PROBE": ExitConfig(
            stop_loss_pct=0.040, take_profit_pct=0.70, max_hold_days=7, scale_down_day=5,
            profit_lock_threshold=0.30, trailing_stop_pct=0.0,
            stop_tighten_day=4, stop_tighten_factor=0.6,
            recovery=RecoveryConfig(recovery_flip_threshold=0.05, recovery_cut_threshold=0.02, recovery_target=0.03),
        ),
        # Shakeout-heavy signals - trailing stop, delayed tightening, wider initial stop
        "BEAR_PROBE": ExitConfig(
            stop_loss_pct=0.065, take_profit_pct=0.60, max_hold_days=11, scale_down_day=7,
            profit_lock_threshold=0.25, trailing_stop_pct=0.03,  # 3% trailing after profit lock
            stop_tighten_day=8, stop_tighten_factor=0.7,  # Delayed tightening for 40% shakeout
            recovery=RecoveryConfig(recovery_flip_threshold=0.05, recovery_cut_threshold=0.02, recovery_target=0.03),
        ),
        "MVRV_SHORT": ExitConfig(
            stop_loss_pct=0.070, take_profit_pct=0.60, max_hold_days=12, scale_down_day=7,
            profit_lock_threshold=0.25, trailing_stop_pct=0.04,  # 4% trailing after profit lock
            stop_tighten_day=9, stop_tighten_factor=0.7,  # Delayed tightening for 57% shakeout
            recovery=RecoveryConfig(recovery_flip_threshold=0.05, recovery_cut_threshold=0.02, recovery_target=0.03),
        ),
        # Range-bound strategy - tighter take-profit, no trailing
        "IRON_CONDOR": ExitConfig(
            stop_loss_pct=0.068, take_profit_pct=0.50, max_hold_days=9, scale_down_day=8,
            profit_lock_threshold=0.30, trailing_stop_pct=0.0,
            stop_tighten_day=7, stop_tighten_factor=0.5,
            # No recovery config for IRON_CONDOR (neutral direction)
        ),
    },
    
    # Expected edge = win_rate × spread_width (from path analysis)
    expected_edge_by_signal={
        "CALL": 0.069,         # 69.4% × 10%
        "PUT": 0.068,          # 75.7% × 9%
        "OPTION_CALL": 0.092,  # 92.3% × 10%
        "OPTION_PUT": 0.096,   # 80.0% × 12%
        "TACTICAL_PUT": 0.034, # 48.0% × 7%
        "BULL_PROBE": 0.085,   # 71.1% × 12%
        "BEAR_PROBE": 0.047,   # 58.8% × 8%
        "IRON_CONDOR": 0.063,  # 63.1% × 10%
        "MVRV_SHORT": 0.060,   # 66.7% × 9%
    },
    
    # Recovery configurations per signal type
    # Based on analysis findings showing +32% net edge for flipping vs holding
    # Thresholds calibrated from recovery MFE distribution analysis
    recovery_configs={
        "CALL": RecoveryConfig(
            recovery_flip_threshold=0.05,  # 5% MFE threshold for flip recommendation
            recovery_cut_threshold=0.02,   # 2% MFE threshold for cut losses
            recovery_target=0.03,          # 3% profit target for recovery trades
        ),
        "PUT": RecoveryConfig(
            recovery_flip_threshold=0.05,
            recovery_cut_threshold=0.02,
            recovery_target=0.03,
        ),
        "OPTION_CALL": RecoveryConfig(
            recovery_flip_threshold=0.05,  # Always flip - 100% edge found
            recovery_cut_threshold=0.02,
            recovery_target=0.03,
        ),
        "OPTION_PUT": RecoveryConfig(
            recovery_flip_threshold=0.05,
            recovery_cut_threshold=0.02,
            recovery_target=0.03,
        ),
        "TACTICAL_PUT": RecoveryConfig(
            recovery_flip_threshold=0.05,  # +44.4% edge for flipping
            recovery_cut_threshold=0.02,
            recovery_target=0.03,
        ),
        "BULL_PROBE": RecoveryConfig(
            recovery_flip_threshold=0.05,  # +22.2% edge for flipping
            recovery_cut_threshold=0.02,
            recovery_target=0.03,
        ),
        "BEAR_PROBE": RecoveryConfig(
            recovery_flip_threshold=0.05,  # +33.3% edge for flipping
            recovery_cut_threshold=0.02,
            recovery_target=0.03,
        ),
        "MVRV_SHORT": RecoveryConfig(
            recovery_flip_threshold=0.05,  # +14.3% edge for flipping
            recovery_cut_threshold=0.02,
            recovery_target=0.03,
        ),
        # IRON_CONDOR excluded from recovery analysis (neutral direction)
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
    "2026-05-03.3": POLICY_V1,
}

_ACTIVE_VERSION = "2026-05-03.3"
_CALIBRATION_PATH = Path(__file__).resolve().parents[1] / "data" / "policy_calibration.json"


def _coerce_dte_config(payload: dict) -> DTEConfig:
    return DTEConfig(
        min_dte=int(payload["min_dte"]),
        max_dte=int(payload["max_dte"]),
        optimal_dte=int(payload["optimal_dte"]),
    )


def _coerce_exit_config(payload: dict, base_config: Optional[ExitConfig] = None) -> ExitConfig:
    """
    Create ExitConfig from calibration payload, preserving path-aware fields from base.
    
    The calibration file only contains basic exit params (stop_loss_pct, take_profit_pct,
    max_hold_days, scale_down_day). Path-aware fields (trailing_stop_pct, profit_lock_threshold,
    stop_tighten_day, stop_tighten_factor) and recovery config are preserved from the base policy.
    """
    return ExitConfig(
        stop_loss_pct=float(payload["stop_loss_pct"]),
        take_profit_pct=float(payload["take_profit_pct"]),
        max_hold_days=int(payload["max_hold_days"]),
        scale_down_day=(None if payload.get("scale_down_day") is None else int(payload["scale_down_day"])),
        # Preserve path-aware fields from base config if available
        profit_lock_threshold=base_config.profit_lock_threshold if base_config else 0.30,
        trailing_stop_pct=base_config.trailing_stop_pct if base_config else 0.0,
        stop_tighten_day=base_config.stop_tighten_day if base_config else None,
        stop_tighten_factor=base_config.stop_tighten_factor if base_config else 0.5,
        # Preserve recovery config from base config if available
        recovery=base_config.recovery if base_config else None,
    )


def _apply_calibration(policy: PolicyVersion) -> PolicyVersion:
    """
    Apply disk calibration overrides to a copied policy.
    Falls back silently to defaults if file is missing/invalid.
    """
    if not _CALIBRATION_PATH.exists():
        return policy
    try:
        payload = json.loads(_CALIBRATION_PATH.read_text())
    except Exception:
        return policy

    calibrated = copy.deepcopy(policy)
    for signal_type, dte_cfg in payload.get("dte_targets", {}).items():
        calibrated.dte_targets[signal_type] = _coerce_dte_config(dte_cfg)
    for signal_type, width in payload.get("spread_width_pct", {}).items():
        calibrated.spread_width_pct[signal_type] = float(width)
    for signal_type, exit_cfg in payload.get("exit_params", {}).items():
        # Pass the base config to preserve path-aware fields
        base_config = policy.exit_params.get(signal_type)
        calibrated.exit_params[signal_type] = _coerce_exit_config(exit_cfg, base_config)
    for signal_type, edge in payload.get("expected_edge_by_signal", {}).items():
        calibrated.expected_edge_by_signal[signal_type] = float(edge)
    return calibrated


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
    return _apply_calibration(_POLICIES[v])


def get_active_version() -> str:
    """Get the currently active policy version string."""
    return _ACTIVE_VERSION


def list_versions() -> list[str]:
    """List all available policy versions."""
    return list(_POLICIES.keys())
