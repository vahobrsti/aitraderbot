"""
Trade Construction Validator - Pre-flight checks before order placement.

Validates trade plans against policy rules and execution realism constraints.
Flags issues that would cause silent drift or execution failures.

VALIDATOR SPEC TABLE
====================
| Code                        | Threshold                      | Severity | Action                    |
|-----------------------------|--------------------------------|----------|---------------------------|
| SCALE_DOWN_NOT_EXECUTABLE   | contracts=1 & scale_down set   | WARNING  | Fallback: close full      |
| TAKE_PROFIT_TOO_CONSERVATIVE| TP<30% & R:R>1                 | WARNING  | Suggest 50-70% TP         |
| STOP_LOSS_BASIS_MISMATCH    | Always (info)                  | INFO     | Add option-value stop     |
| WIDTH_DEVIATION_FROM_POLICY | >30% deviation                 | WARNING  | Flag as budget override   |
| HIGH_EXECUTION_COST_IMPACT  | costs >10% of max profit       | WARNING  | Suggest limit orders      |
| WIDE_BID_ASK_*              | spread > max_spread_pct        | WARNING  | Review liquidity          |
| LOW_OPEN_INTEREST_*         | OI < min_open_interest         | WARNING  | Review liquidity          |
| POOR_RISK_REWARD            | R:R < 0.5:1                    | BLOCKING | Reject trade              |
| SUBOPTIMAL_RISK_REWARD      | R:R < 1:1                      | WARNING  | Flag for review           |
| POSITION_EXCEEDS_BUDGET     | risk > budget * 1.1            | BLOCKING | Reject trade              |
| POSITION_UNDERUTILIZED      | risk < budget * 0.5            | INFO     | Suggest tighter spread    |
| INSUFFICIENT_NET_EDGE       | net_edge < MIN_NET_EDGE        | BLOCKING | Reject trade              |
| LIQUIDITY_INSUFFICIENT_SIZE | OI < contracts * 10            | WARNING  | Reduce size or split      |
| SPREAD_IMPACT_AT_SIZE       | spread_cost > 2% at size       | WARNING  | Use limit orders          |

Usage:
    from execution.services.trade_validator import TradeValidator, ValidationResult
    
    validator = TradeValidator()
    result = validator.validate(plan)
    
    if result.has_blocking_issues:
        print("Cannot execute:", result.blocking_issues)
    if result.warnings:
        print("Warnings:", result.warnings)
"""
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Optional

from execution.services.policy import get_policy, PolicyVersion


class IssueSeverity(Enum):
    """Severity levels for validation issues."""
    BLOCKING = "blocking"      # Cannot execute, must fix
    WARNING = "warning"        # Can execute, but flagged for review
    INFO = "info"              # Informational, no action needed


@dataclass
class ValidationIssue:
    """A single validation issue."""
    code: str
    severity: IssueSeverity
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Complete validation result for a trade plan."""
    is_valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    adjusted_params: dict = field(default_factory=dict)  # Suggested fixes
    
    @property
    def blocking_issues(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == IssueSeverity.BLOCKING]
    
    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == IssueSeverity.WARNING]
    
    @property
    def has_blocking_issues(self) -> bool:
        return len(self.blocking_issues) > 0


@dataclass
class SpreadPlan:
    """Simplified spread plan for validation."""
    signal_type: str
    direction: str              # 'long' or 'short' (trade direction)
    option_type: str            # 'call' or 'put'
    long_strike: float
    short_strike: float
    expiry_dte: int
    net_debit: float            # Cost to enter (positive = debit)
    max_profit: float
    max_loss: float
    contracts: int
    spot_price: float
    # Optional greeks
    long_delta: Optional[float] = None
    short_delta: Optional[float] = None
    long_iv: Optional[float] = None
    short_iv: Optional[float] = None
    # Liquidity
    long_bid_ask_spread_pct: Optional[float] = None
    short_bid_ask_spread_pct: Optional[float] = None
    long_open_interest: Optional[int] = None
    short_open_interest: Optional[int] = None


class TradeValidator:
    """
    Validates trade construction against policy and execution constraints.
    
    Checks:
    1. Scale-down executability (can't halve 1 contract)
    2. Take-profit threshold vs risk/reward
    3. Stop-loss basis (spot vs option value)
    4. Width deviation from policy
    5. Execution cost impact on edge
    6. Liquidity thresholds
    7. Position sizing constraints
    8. Minimum net edge gate (NEW)
    9. Liquidity stress by size tier (NEW)
    """
    
    # Minimum net edge as % of debit required to execute
    # net_edge = (tp_value - expected_costs) / debit
    MIN_NET_EDGE_PCT = 0.05  # 5% minimum net edge
    
    # Liquidity stress multipliers
    # OI should be at least this multiple of contracts for safe execution
    OI_SAFETY_MULTIPLIER = 10  # OI >= contracts * 10
    
    # Spread impact threshold at size (% of debit)
    MAX_SPREAD_IMPACT_AT_SIZE = 0.02  # 2%
    
    def __init__(self, policy: Optional[PolicyVersion] = None):
        self.policy = policy or get_policy()
    
    def validate(
        self,
        plan: SpreadPlan,
    ) -> ValidationResult:
        """
        Run all validation checks on a spread plan.
        
        Returns ValidationResult with issues and suggested adjustments.
        """
        issues: list[ValidationIssue] = []
        adjusted_params: dict = {}
        
        # Run all checks
        issues.extend(self._check_scale_down_executability(plan, adjusted_params))
        issues.extend(self._check_take_profit_threshold(plan, adjusted_params))
        issues.extend(self._check_stop_loss_basis(plan, adjusted_params))
        issues.extend(self._check_width_deviation(plan, adjusted_params))
        issues.extend(self._check_execution_costs(plan, adjusted_params))
        issues.extend(self._check_liquidity(plan))
        issues.extend(self._check_risk_reward_minimum(plan))
        issues.extend(self._check_position_sizing(plan))
        issues.extend(self._check_minimum_net_edge(plan, adjusted_params))
        issues.extend(self._check_liquidity_at_size(plan, adjusted_params))
        
        is_valid = not any(i.severity == IssueSeverity.BLOCKING for i in issues)
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            adjusted_params=adjusted_params,
        )
    
    def _check_scale_down_executability(
        self,
        plan: SpreadPlan,
        adjusted_params: dict,
    ) -> list[ValidationIssue]:
        """
        Issue #1: Scale-down with 1 contract is not executable.
        
        If contracts=1 and scale_down_day is set, we can't halve the position.
        Fallback: close full position on scale_down_day trigger.
        """
        issues = []
        exit_cfg = self.policy.get_exit_params(plan.signal_type)
        
        if exit_cfg and exit_cfg.scale_down_day and plan.contracts == 1:
            issues.append(ValidationIssue(
                code="SCALE_DOWN_NOT_EXECUTABLE",
                severity=IssueSeverity.WARNING,
                message=(
                    f"Scale-down on day {exit_cfg.scale_down_day} not executable with 1 contract. "
                    f"Fallback: close full position if scale-down trigger met."
                ),
                details={
                    "contracts": plan.contracts,
                    "scale_down_day": exit_cfg.scale_down_day,
                    "fallback_action": "close_full_position",
                },
            ))
            adjusted_params["scale_down_action"] = "close_full_position"
        
        return issues
    
    def _check_take_profit_threshold(
        self,
        plan: SpreadPlan,
        adjusted_params: dict,
    ) -> list[ValidationIssue]:
        """
        Issue #2: Take-profit threshold vs risk/reward.
        
        If TP% is very low (e.g., 8%) but R:R is favorable (e.g., 1:1.3),
        we're leaving significant edge on the table.
        
        Recommendation: TP should be at least 50% of max profit for spreads
        with R:R > 1:1, unless win rate is exceptionally high.
        """
        issues = []
        exit_cfg = self.policy.get_exit_params(plan.signal_type)
        
        if not exit_cfg or plan.max_profit <= 0:
            return issues
        
        tp_pct = exit_cfg.take_profit_pct
        rr_ratio = plan.max_profit / plan.max_loss if plan.max_loss > 0 else 0
        tp_dollar = plan.max_profit * tp_pct
        
        # Expected edge from policy
        expected_edge = self.policy.get_expected_edge(plan.signal_type, default=0.06)
        
        # If R:R > 1 and TP < 30%, flag as potentially too conservative
        if rr_ratio > 1.0 and tp_pct < 0.30:
            # Calculate what TP would need to be to maintain edge
            # edge = win_rate * avg_win - (1 - win_rate) * avg_loss
            # With TP at 8%, avg_win = 8% of max_profit
            # This materially lowers expectancy
            
            suggested_tp = max(0.50, min(0.70, rr_ratio * 0.40))
            
            issues.append(ValidationIssue(
                code="TAKE_PROFIT_TOO_CONSERVATIVE",
                severity=IssueSeverity.WARNING,
                message=(
                    f"Take-profit at {tp_pct*100:.0f}% targets ${tp_dollar:.0f} "
                    f"while max profit is ${plan.max_profit:.0f} (R:R 1:{rr_ratio:.2f}). "
                    f"This materially lowers expectancy. Consider {suggested_tp*100:.0f}%."
                ),
                details={
                    "current_tp_pct": tp_pct,
                    "tp_dollar": tp_dollar,
                    "max_profit": plan.max_profit,
                    "rr_ratio": rr_ratio,
                    "suggested_tp_pct": suggested_tp,
                    "expected_edge": expected_edge,
                },
            ))
            adjusted_params["suggested_take_profit_pct"] = suggested_tp
        
        return issues
    
    def _check_stop_loss_basis(
        self,
        plan: SpreadPlan,
        adjusted_params: dict,
    ) -> list[ValidationIssue]:
        """
        Issue #3: Stop-loss basis mismatch.
        
        Policy stop is on BTC spot, but position risk is in option spread value.
        Spot-based stops can misfire due to IV/time decay effects.
        
        Recommendation: Add option-value stop (% of debit lost).
        """
        issues = []
        exit_cfg = self.policy.get_exit_params(plan.signal_type)
        
        if not exit_cfg:
            return issues
        
        spot_stop_pct = exit_cfg.stop_loss_pct
        
        # Calculate spot stop price
        if plan.direction == "short":  # Bearish trade, stop if price rises
            spot_stop_price = plan.spot_price * (1 + spot_stop_pct)
        else:  # Bullish trade, stop if price falls
            spot_stop_price = plan.spot_price * (1 - spot_stop_pct)
        
        # Recommend option-value stop as backup
        # Standard: stop if spread loses 50-70% of debit
        option_stop_pct = 0.60  # Stop if 60% of debit lost
        option_stop_value = plan.net_debit * option_stop_pct
        
        issues.append(ValidationIssue(
            code="STOP_LOSS_BASIS_MISMATCH",
            severity=IssueSeverity.INFO,
            message=(
                f"Spot-based stop at ${spot_stop_price:,.0f} ({spot_stop_pct*100:.1f}% move). "
                f"Add option-value stop: close if spread value drops to "
                f"${plan.net_debit - option_stop_value:,.0f} ({option_stop_pct*100:.0f}% of debit lost)."
            ),
            details={
                "spot_stop_pct": spot_stop_pct,
                "spot_stop_price": spot_stop_price,
                "option_stop_pct": option_stop_pct,
                "option_stop_trigger": plan.net_debit * (1 - option_stop_pct),
                "net_debit": plan.net_debit,
            },
        ))
        adjusted_params["option_value_stop_pct"] = option_stop_pct
        adjusted_params["option_value_stop_trigger"] = plan.net_debit * (1 - option_stop_pct)
        
        return issues
    
    def _check_width_deviation(
        self,
        plan: SpreadPlan,
        adjusted_params: dict,
    ) -> list[ValidationIssue]:
        """
        Issue #4: Width selection deviates from policy intent.
        
        If actual width differs significantly from policy width due to
        budget constraints, flag as explicit override.
        """
        issues = []
        
        policy_width_pct = self.policy.get_spread_width(plan.signal_type)
        policy_width_usd = plan.spot_price * policy_width_pct
        actual_width = abs(plan.long_strike - plan.short_strike)
        
        deviation_pct = abs(actual_width - policy_width_usd) / policy_width_usd if policy_width_usd > 0 else 0
        
        # Flag if deviation > 30%
        if deviation_pct > 0.30:
            reason = "budget_constraint" if actual_width < policy_width_usd else "liquidity_constraint"
            
            issues.append(ValidationIssue(
                code="WIDTH_DEVIATION_FROM_POLICY",
                severity=IssueSeverity.WARNING,
                message=(
                    f"Spread width ${actual_width:,.0f} ({actual_width/plan.spot_price*100:.1f}%) "
                    f"deviates {deviation_pct*100:.0f}% from policy target "
                    f"${policy_width_usd:,.0f} ({policy_width_pct*100:.0f}%). "
                    f"Reason: {reason}."
                ),
                details={
                    "policy_width_pct": policy_width_pct,
                    "policy_width_usd": policy_width_usd,
                    "actual_width_usd": actual_width,
                    "actual_width_pct": actual_width / plan.spot_price,
                    "deviation_pct": deviation_pct,
                    "reason": reason,
                },
            ))
            adjusted_params["width_override_reason"] = reason
        
        return issues
    
    def _check_execution_costs(
        self,
        plan: SpreadPlan,
        adjusted_params: dict,
    ) -> list[ValidationIssue]:
        """
        Issue #5: Execution costs impact on edge.
        
        Include fees, spread cost, and slippage in sizing and TP threshold.
        """
        issues = []
        
        exec_costs = self.policy.execution_costs
        num_legs = 2  # Spread has 2 legs
        
        # Total cost per contract (entry + exit = 4 leg executions)
        total_cost_pct = exec_costs.total_cost_multi_leg(num_legs * 2, is_market=True)
        total_cost_usd = plan.net_debit * total_cost_pct
        
        # Also estimate slippage on entry
        entry_slippage = plan.net_debit * exec_costs.slippage_pct
        
        # Adjusted max profit after costs
        adjusted_max_profit = plan.max_profit - total_cost_usd
        
        # Check if costs eat significant portion of profit
        cost_impact_pct = total_cost_usd / plan.max_profit if plan.max_profit > 0 else 0
        
        if cost_impact_pct > 0.10:  # Costs > 10% of max profit
            issues.append(ValidationIssue(
                code="HIGH_EXECUTION_COST_IMPACT",
                severity=IssueSeverity.WARNING,
                message=(
                    f"Execution costs ~${total_cost_usd:.0f} ({cost_impact_pct*100:.1f}% of max profit). "
                    f"Adjusted max profit: ${adjusted_max_profit:.0f}. "
                    f"Consider limit orders to reduce taker fees."
                ),
                details={
                    "total_cost_usd": total_cost_usd,
                    "total_cost_pct": total_cost_pct,
                    "entry_slippage": entry_slippage,
                    "adjusted_max_profit": adjusted_max_profit,
                    "cost_impact_pct": cost_impact_pct,
                },
            ))
        
        adjusted_params["estimated_execution_cost"] = total_cost_usd
        adjusted_params["adjusted_max_profit"] = adjusted_max_profit
        
        return issues
    
    def _check_liquidity(self, plan: SpreadPlan) -> list[ValidationIssue]:
        """Check liquidity thresholds for both legs."""
        issues = []
        liq_cfg = self.policy.liquidity
        
        # Check bid-ask spread
        for leg_name, spread_pct in [
            ("long", plan.long_bid_ask_spread_pct),
            ("short", plan.short_bid_ask_spread_pct),
        ]:
            if spread_pct and spread_pct > liq_cfg.max_spread_pct:
                issues.append(ValidationIssue(
                    code=f"WIDE_BID_ASK_{leg_name.upper()}",
                    severity=IssueSeverity.WARNING,
                    message=(
                        f"{leg_name.title()} leg bid-ask spread {spread_pct*100:.1f}% "
                        f"exceeds threshold {liq_cfg.max_spread_pct*100:.0f}%."
                    ),
                    details={"leg": leg_name, "spread_pct": spread_pct},
                ))
        
        # Check open interest
        for leg_name, oi in [
            ("long", plan.long_open_interest),
            ("short", plan.short_open_interest),
        ]:
            if oi and oi < int(liq_cfg.min_open_interest):
                issues.append(ValidationIssue(
                    code=f"LOW_OPEN_INTEREST_{leg_name.upper()}",
                    severity=IssueSeverity.WARNING,
                    message=(
                        f"{leg_name.title()} leg OI={oi} below minimum {liq_cfg.min_open_interest}."
                    ),
                    details={"leg": leg_name, "open_interest": oi},
                ))
        
        return issues
    
    def _check_risk_reward_minimum(self, plan: SpreadPlan) -> list[ValidationIssue]:
        """Check minimum risk/reward ratio."""
        issues = []
        
        if plan.max_loss <= 0:
            return issues
        
        rr_ratio = plan.max_profit / plan.max_loss
        
        # Minimum R:R of 0.5:1 for any trade
        if rr_ratio < 0.5:
            issues.append(ValidationIssue(
                code="POOR_RISK_REWARD",
                severity=IssueSeverity.BLOCKING,
                message=(
                    f"Risk/reward ratio 1:{rr_ratio:.2f} is below minimum 1:0.5. "
                    f"Consider wider spread or different strikes."
                ),
                details={"rr_ratio": rr_ratio, "min_rr": 0.5},
            ))
        elif rr_ratio < 1.0:
            issues.append(ValidationIssue(
                code="SUBOPTIMAL_RISK_REWARD",
                severity=IssueSeverity.WARNING,
                message=(
                    f"Risk/reward ratio 1:{rr_ratio:.2f} is below 1:1. "
                    f"Acceptable but not ideal."
                ),
                details={"rr_ratio": rr_ratio},
            ))
        
        return issues
    
    def _check_position_sizing(self, plan: SpreadPlan) -> list[ValidationIssue]:
        """Check position sizing constraints."""
        issues = []
        
        tier = self.policy.get_tier(plan.signal_type)
        risk_budget = tier.risk_usd * tier.spread_pct
        
        total_risk = plan.max_loss * plan.contracts
        
        # Check if position exceeds budget
        if total_risk > risk_budget * 1.1:  # 10% tolerance
            issues.append(ValidationIssue(
                code="POSITION_EXCEEDS_BUDGET",
                severity=IssueSeverity.BLOCKING,
                message=(
                    f"Total risk ${total_risk:,.0f} exceeds budget ${risk_budget:,.0f}."
                ),
                details={
                    "total_risk": total_risk,
                    "risk_budget": risk_budget,
                    "contracts": plan.contracts,
                },
            ))
        
        # Check if position is too small (< 50% of budget utilized)
        if total_risk < risk_budget * 0.5 and plan.contracts >= 1:
            issues.append(ValidationIssue(
                code="POSITION_UNDERUTILIZED",
                severity=IssueSeverity.INFO,
                message=(
                    f"Only ${total_risk:,.0f} of ${risk_budget:,.0f} budget utilized "
                    f"({total_risk/risk_budget*100:.0f}%). Consider tighter spread."
                ),
                details={
                    "total_risk": total_risk,
                    "risk_budget": risk_budget,
                    "utilization_pct": total_risk / risk_budget,
                },
            ))
        
        return issues
    
    def _check_minimum_net_edge(
        self,
        plan: SpreadPlan,
        adjusted_params: dict,
    ) -> list[ValidationIssue]:
        """
        Issue #8: Minimum net edge gate.
        
        net_edge = (tp_value - expected_costs) / debit
        
        Blocks trades where net edge after costs and TP is below threshold.
        This ensures we're not taking trades that look good on R:R but
        have insufficient expected value after realistic execution.
        """
        issues = []
        exit_cfg = self.policy.get_exit_params(plan.signal_type)
        
        if not exit_cfg or plan.net_debit <= 0:
            return issues
        
        # Calculate expected take-profit value
        tp_pct = exit_cfg.take_profit_pct
        tp_value = plan.max_profit * tp_pct
        
        # Get execution costs (already calculated, or compute fresh)
        exec_costs = self.policy.execution_costs
        num_legs = 2
        total_cost_pct = exec_costs.total_cost_multi_leg(num_legs * 2, is_market=True)
        expected_costs = plan.net_debit * total_cost_pct
        
        # Net edge = (expected_win - costs) / capital_at_risk
        net_edge = (tp_value - expected_costs) / plan.net_debit
        
        adjusted_params["net_edge_pct"] = net_edge
        adjusted_params["tp_value"] = tp_value
        adjusted_params["expected_costs"] = expected_costs
        
        if net_edge < self.MIN_NET_EDGE_PCT:
            issues.append(ValidationIssue(
                code="INSUFFICIENT_NET_EDGE",
                severity=IssueSeverity.BLOCKING,
                message=(
                    f"Net edge {net_edge*100:.1f}% is below minimum {self.MIN_NET_EDGE_PCT*100:.0f}%. "
                    f"TP value ${tp_value:.0f} - costs ${expected_costs:.0f} = "
                    f"${tp_value - expected_costs:.0f} on ${plan.net_debit:.0f} debit. "
                    f"Consider wider spread or better entry price."
                ),
                details={
                    "net_edge_pct": net_edge,
                    "min_net_edge_pct": self.MIN_NET_EDGE_PCT,
                    "tp_value": tp_value,
                    "expected_costs": expected_costs,
                    "net_profit": tp_value - expected_costs,
                    "debit": plan.net_debit,
                },
            ))
        elif net_edge < self.MIN_NET_EDGE_PCT * 2:
            # Marginal edge - warn but allow
            issues.append(ValidationIssue(
                code="MARGINAL_NET_EDGE",
                severity=IssueSeverity.WARNING,
                message=(
                    f"Net edge {net_edge*100:.1f}% is marginal. "
                    f"Expected profit after costs: ${tp_value - expected_costs:.0f}."
                ),
                details={
                    "net_edge_pct": net_edge,
                    "net_profit": tp_value - expected_costs,
                },
            ))
        
        return issues
    
    def _check_liquidity_at_size(
        self,
        plan: SpreadPlan,
        adjusted_params: dict,
    ) -> list[ValidationIssue]:
        """
        Issue #9: Liquidity stress by size tier.
        
        OI/spread checks that pass at 1 lot can fail at scaled size.
        - OI should be >= contracts * OI_SAFETY_MULTIPLIER
        - Spread impact at size should be < MAX_SPREAD_IMPACT_AT_SIZE
        """
        issues = []
        
        if plan.contracts <= 1:
            # Single contract - basic liquidity checks are sufficient
            return issues
        
        # Check OI relative to position size
        min_oi_needed = plan.contracts * self.OI_SAFETY_MULTIPLIER
        
        for leg_name, oi in [
            ("long", plan.long_open_interest),
            ("short", plan.short_open_interest),
        ]:
            if oi and oi < min_oi_needed:
                issues.append(ValidationIssue(
                    code=f"LIQUIDITY_INSUFFICIENT_SIZE_{leg_name.upper()}",
                    severity=IssueSeverity.WARNING,
                    message=(
                        f"{leg_name.title()} leg OI={oi} is insufficient for {plan.contracts} contracts. "
                        f"Recommend OI >= {min_oi_needed} (contracts × {self.OI_SAFETY_MULTIPLIER}). "
                        f"Consider reducing size or splitting execution."
                    ),
                    details={
                        "leg": leg_name,
                        "open_interest": oi,
                        "contracts": plan.contracts,
                        "min_oi_needed": min_oi_needed,
                        "oi_multiplier": self.OI_SAFETY_MULTIPLIER,
                    },
                ))
        
        # Check spread impact at size
        # Larger orders face more slippage - estimate as spread_pct * sqrt(contracts)
        for leg_name, spread_pct in [
            ("long", plan.long_bid_ask_spread_pct),
            ("short", plan.short_bid_ask_spread_pct),
        ]:
            if spread_pct:
                # Simple model: spread impact scales with sqrt of size
                import math
                size_factor = math.sqrt(plan.contracts)
                estimated_impact = spread_pct * size_factor
                
                if estimated_impact > self.MAX_SPREAD_IMPACT_AT_SIZE:
                    issues.append(ValidationIssue(
                        code=f"SPREAD_IMPACT_AT_SIZE_{leg_name.upper()}",
                        severity=IssueSeverity.WARNING,
                        message=(
                            f"{leg_name.title()} leg spread impact at {plan.contracts} contracts "
                            f"estimated at {estimated_impact*100:.1f}% (base spread {spread_pct*100:.1f}% × √{plan.contracts}). "
                            f"Exceeds {self.MAX_SPREAD_IMPACT_AT_SIZE*100:.0f}% threshold. "
                            f"Use limit orders and consider splitting execution."
                        ),
                        details={
                            "leg": leg_name,
                            "base_spread_pct": spread_pct,
                            "contracts": plan.contracts,
                            "estimated_impact_pct": estimated_impact,
                            "max_impact_pct": self.MAX_SPREAD_IMPACT_AT_SIZE,
                        },
                    ))
        
        # Suggest execution strategy for larger sizes
        if plan.contracts >= 3:
            adjusted_params["execution_strategy"] = "split_iceberg"
            adjusted_params["suggested_tranches"] = max(2, plan.contracts // 2)
        
        return issues


def validate_spread_plan(
    signal_type: str,
    direction: str,
    option_type: str,
    long_strike: float,
    short_strike: float,
    expiry_dte: int,
    net_debit: float,
    contracts: int,
    spot_price: float,
    long_delta: Optional[float] = None,
    short_delta: Optional[float] = None,
    long_iv: Optional[float] = None,
    short_iv: Optional[float] = None,
    long_bid_ask_spread_pct: Optional[float] = None,
    short_bid_ask_spread_pct: Optional[float] = None,
    long_open_interest: Optional[int] = None,
    short_open_interest: Optional[int] = None,
) -> ValidationResult:
    """
    Convenience function to validate a spread plan.
    
    Example:
        result = validate_spread_plan(
            signal_type="MVRV_SHORT",
            direction="short",
            option_type="put",
            long_strike=80000,
            short_strike=76000,
            expiry_dte=12,
            net_debit=1733.27,
            contracts=1,
            spot_price=78653,
        )
        
        if result.has_blocking_issues:
            for issue in result.blocking_issues:
                print(f"BLOCKING: {issue.message}")
    """
    width = abs(long_strike - short_strike)
    max_profit = width - net_debit
    max_loss = net_debit
    
    plan = SpreadPlan(
        signal_type=signal_type,
        direction=direction,
        option_type=option_type,
        long_strike=long_strike,
        short_strike=short_strike,
        expiry_dte=expiry_dte,
        net_debit=net_debit,
        max_profit=max_profit,
        max_loss=max_loss,
        contracts=contracts,
        spot_price=spot_price,
        long_delta=long_delta,
        short_delta=short_delta,
        long_iv=long_iv,
        short_iv=short_iv,
        long_bid_ask_spread_pct=long_bid_ask_spread_pct,
        short_bid_ask_spread_pct=short_bid_ask_spread_pct,
        long_open_interest=long_open_interest,
        short_open_interest=short_open_interest,
    )
    
    validator = TradeValidator()
    return validator.validate(plan)
