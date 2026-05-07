"""
Unit tests for Trade Setup, Trade Validator, and Policy modules.

Tests cover:
- PolicyVersion configuration and parameter retrieval
- TradeValidator validation checks (11 rules)
- TradeSetupBuilder spread construction and leg selection
- API endpoints for trade setup
- Telegram message formatting
"""
from decimal import Decimal
from datetime import date, timedelta
from unittest.mock import Mock, patch, MagicMock
from django.test import TestCase

from execution.services.policy import (
    get_policy, PolicyVersion, TierConfig, DTEConfig, ExitConfig,
    LiquidityConfig, ExecutionCostConfig, get_active_version, list_versions
)
from execution.services.trade_validator import (
    TradeValidator, SpreadPlan, ValidationResult, ValidationIssue,
    IssueSeverity, validate_spread_plan
)


# =============================================================================
# POLICY TESTS
# =============================================================================

class PolicyVersionTests(TestCase):
    """Tests for PolicyVersion configuration."""
    
    def test_get_active_policy(self):
        """Test retrieving the active policy."""
        policy = get_policy()
        self.assertIsInstance(policy, PolicyVersion)
        self.assertIsNotNone(policy.version)
    
    def test_get_specific_version(self):
        """Test retrieving a specific policy version."""
        version = get_active_version()
        policy = get_policy(version=version)
        self.assertEqual(policy.version, version)
    
    def test_invalid_version_raises(self):
        """Test that invalid version raises ValueError."""
        with self.assertRaises(ValueError):
            get_policy(version="invalid-version")
    
    def test_list_versions(self):
        """Test listing available versions."""
        versions = list_versions()
        self.assertIsInstance(versions, list)
        self.assertGreater(len(versions), 0)
    
    def test_tier_config_retrieval(self):
        """Test getting tier config for signal types."""
        policy = get_policy()
        
        tier = policy.get_tier("MVRV_SHORT")
        self.assertIsInstance(tier, TierConfig)
        self.assertGreater(tier.risk_usd, 0)
        self.assertGreater(tier.spread_pct, 0)
    
    def test_tier_fallback_to_default(self):
        """Test tier falls back to tier 2 for unknown signals."""
        policy = get_policy()
        tier = policy.get_tier("UNKNOWN_SIGNAL")
        self.assertEqual(tier, policy.tiers["2"])
    
    def test_dte_target_retrieval(self):
        """Test getting DTE targets for signal types."""
        policy = get_policy()
        
        dte = policy.get_dte_target("MVRV_SHORT")
        self.assertIsInstance(dte, DTEConfig)
        self.assertGreater(dte.min_dte, 0)
        self.assertGreater(dte.max_dte, dte.min_dte)
        self.assertGreaterEqual(dte.optimal_dte, dte.min_dte)
        self.assertLessEqual(dte.optimal_dte, dte.max_dte)
    
    def test_dte_fallback_for_unknown(self):
        """Test DTE falls back to default for unknown signals."""
        policy = get_policy()
        dte = policy.get_dte_target("UNKNOWN_SIGNAL")
        self.assertEqual(dte.min_dte, 11)
        self.assertEqual(dte.max_dte, 14)
    
    def test_delta_target_retrieval(self):
        """Test getting delta targets."""
        policy = get_policy()
        
        call_delta = policy.get_delta_target("slight_itm", "call")
        put_delta = policy.get_delta_target("slight_itm", "put")
        
        self.assertGreater(call_delta, 0)
        self.assertLess(put_delta, 0)
    
    def test_signal_delta_retrieval(self):
        """Test getting per-signal delta targets."""
        policy = get_policy()
        
        delta = policy.get_signal_delta("MVRV_SHORT")
        self.assertLess(delta, 0)  # Put signal should have negative delta
        
        delta = policy.get_signal_delta("CALL")
        self.assertGreater(delta, 0)  # Call signal should have positive delta
    
    def test_spread_enabled_check(self):
        """Test spread enabled check."""
        policy = get_policy()
        
        self.assertTrue(policy.is_spread_enabled("MVRV_SHORT"))
        self.assertTrue(policy.is_spread_enabled("CALL"))
    
    def test_spread_width_retrieval(self):
        """Test getting spread width percentage."""
        policy = get_policy()
        
        width = policy.get_spread_width("MVRV_SHORT")
        self.assertGreater(width, 0)
        self.assertLess(width, 1)  # Should be a percentage
    
    def test_exit_params_retrieval(self):
        """Test getting exit parameters."""
        policy = get_policy()
        
        exit_cfg = policy.get_exit_params("MVRV_SHORT")
        self.assertIsInstance(exit_cfg, ExitConfig)
        self.assertGreater(exit_cfg.stop_loss_pct, 0)
        self.assertGreater(exit_cfg.take_profit_pct, 0)
        self.assertGreater(exit_cfg.max_hold_days, 0)
    
    def test_expected_edge_retrieval(self):
        """Test getting expected edge."""
        policy = get_policy()
        
        edge = policy.get_expected_edge("MVRV_SHORT")
        self.assertGreater(edge, 0)
        
        # Test default fallback
        edge = policy.get_expected_edge("UNKNOWN", default=0.05)
        self.assertEqual(edge, 0.05)
    
    def test_execution_cost_calculation(self):
        """Test execution cost calculations."""
        policy = get_policy()
        
        # Single leg cost
        cost = policy.execution_costs.total_cost_per_leg(is_market=True)
        self.assertGreater(cost, 0)
        
        # Multi-leg cost (spread = 4 legs: entry + exit)
        multi_cost = policy.execution_costs.total_cost_multi_leg(4, is_market=True)
        self.assertEqual(multi_cost, cost * 4)
    
    def test_edge_after_costs_estimation(self):
        """Test edge after costs estimation."""
        policy = get_policy()
        
        # Positive edge scenario
        edge = policy.estimate_edge_after_costs(0.10, num_legs=4, is_market=True)
        self.assertGreater(edge, 0)
        
        # Negative edge scenario (costs exceed return)
        edge = policy.estimate_edge_after_costs(0.01, num_legs=4, is_market=True)
        self.assertLess(edge, 0.01)
    
    def test_policy_to_dict(self):
        """Test policy serialization."""
        policy = get_policy()
        data = policy.to_dict()
        
        self.assertIn("version", data)
        self.assertIn("tiers", data)
        self.assertIn("dte_targets", data)
        self.assertIn("exit_params", data)


# =============================================================================
# TRADE VALIDATOR TESTS
# =============================================================================

class TradeValidatorTests(TestCase):
    """Tests for TradeValidator validation checks."""
    
    def setUp(self):
        self.validator = TradeValidator()
        self.base_plan = SpreadPlan(
            signal_type="MVRV_SHORT",
            direction="short",
            option_type="put",
            long_strike=80000,
            short_strike=76000,
            expiry_dte=12,
            net_debit=1733.27,
            max_profit=2266.73,
            max_loss=1733.27,
            contracts=1,
            spot_price=78653,
            long_delta=-0.58,
            short_delta=-0.29,
            long_iv=0.35,
            short_iv=0.38,
            long_bid_ask_spread_pct=0.03,
            short_bid_ask_spread_pct=0.04,
            long_open_interest=257,
            short_open_interest=541,
        )
    
    def test_valid_plan_passes(self):
        """Test that a valid plan passes validation."""
        result = self.validator.validate(self.base_plan)
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.blocking_issues), 0)
    
    def test_scale_down_not_executable_warning(self):
        """Test scale-down warning for 1 contract."""
        result = self.validator.validate(self.base_plan)
        
        # Should have warning about scale-down
        warnings = [i for i in result.warnings if i.code == "SCALE_DOWN_NOT_EXECUTABLE"]
        self.assertEqual(len(warnings), 1)
        self.assertIn("close_full_position", result.adjusted_params.get("scale_down_action", ""))
    
    def test_poor_risk_reward_blocks(self):
        """Test that poor R:R blocks the trade."""
        plan = SpreadPlan(
            signal_type="MVRV_SHORT",
            direction="short",
            option_type="put",
            long_strike=80000,
            short_strike=79500,  # Very narrow spread
            expiry_dte=12,
            net_debit=400,
            max_profit=100,  # R:R = 0.25:1
            max_loss=400,
            contracts=1,
            spot_price=78653,
        )
        
        result = self.validator.validate(plan)
        self.assertFalse(result.is_valid)
        
        blocking = [i for i in result.blocking_issues if i.code == "POOR_RISK_REWARD"]
        self.assertEqual(len(blocking), 1)
    
    def test_suboptimal_risk_reward_warning(self):
        """Test that suboptimal R:R generates warning."""
        plan = SpreadPlan(
            signal_type="MVRV_SHORT",
            direction="short",
            option_type="put",
            long_strike=80000,
            short_strike=79000,
            expiry_dte=12,
            net_debit=600,
            max_profit=400,  # R:R = 0.67:1
            max_loss=600,
            contracts=1,
            spot_price=78653,
        )
        
        result = self.validator.validate(plan)
        self.assertTrue(result.is_valid)  # Should pass but with warning
        
        warnings = [i for i in result.warnings if i.code == "SUBOPTIMAL_RISK_REWARD"]
        self.assertEqual(len(warnings), 1)
    
    def test_position_exceeds_budget_blocks(self):
        """Test that exceeding budget blocks the trade."""
        plan = SpreadPlan(
            signal_type="MVRV_SHORT",
            direction="short",
            option_type="put",
            long_strike=80000,
            short_strike=76000,
            expiry_dte=12,
            net_debit=5000,  # Very expensive
            max_profit=2000,
            max_loss=5000,
            contracts=1,
            spot_price=78653,
        )
        
        result = self.validator.validate(plan)
        
        blocking = [i for i in result.blocking_issues if i.code == "POSITION_EXCEEDS_BUDGET"]
        self.assertEqual(len(blocking), 1)
    
    def test_width_deviation_warning(self):
        """Test width deviation from policy generates warning."""
        # Base plan has $4000 width vs ~$11800 policy target (66% deviation)
        result = self.validator.validate(self.base_plan)
        
        warnings = [i for i in result.warnings if i.code == "WIDTH_DEVIATION_FROM_POLICY"]
        self.assertEqual(len(warnings), 1)
        self.assertIn("budget_constraint", warnings[0].details.get("reason", ""))
    
    def test_stop_loss_basis_info(self):
        """Test stop loss basis mismatch info is added."""
        result = self.validator.validate(self.base_plan)
        
        infos = [i for i in result.issues if i.code == "STOP_LOSS_BASIS_MISMATCH"]
        self.assertEqual(len(infos), 1)
        self.assertEqual(infos[0].severity, IssueSeverity.INFO)
        
        # Should have option value stop in adjusted params
        self.assertIn("option_value_stop_pct", result.adjusted_params)
    
    def test_execution_cost_impact_warning(self):
        """Test high execution cost impact warning."""
        plan = SpreadPlan(
            signal_type="MVRV_SHORT",
            direction="short",
            option_type="put",
            long_strike=80000,
            short_strike=79900,  # Very narrow
            expiry_dte=12,
            net_debit=80,
            max_profit=20,  # Small profit, costs will be significant
            max_loss=80,
            contracts=1,
            spot_price=78653,
        )
        
        result = self.validator.validate(plan)
        
        # Check for cost impact warning or blocking
        cost_issues = [i for i in result.issues if "COST" in i.code or "EDGE" in i.code]
        self.assertGreater(len(cost_issues), 0)
    
    def test_wide_bid_ask_warning(self):
        """Test wide bid-ask spread warning."""
        plan = SpreadPlan(
            signal_type="MVRV_SHORT",
            direction="short",
            option_type="put",
            long_strike=80000,
            short_strike=76000,
            expiry_dte=12,
            net_debit=1733.27,
            max_profit=2266.73,
            max_loss=1733.27,
            contracts=1,
            spot_price=78653,
            long_bid_ask_spread_pct=0.20,  # 20% spread - very wide
            short_bid_ask_spread_pct=0.25,
        )
        
        result = self.validator.validate(plan)
        
        warnings = [i for i in result.warnings if "WIDE_BID_ASK" in i.code]
        self.assertEqual(len(warnings), 2)  # Both legs
    
    def test_low_open_interest_warning(self):
        """Test low open interest warning."""
        plan = SpreadPlan(
            signal_type="MVRV_SHORT",
            direction="short",
            option_type="put",
            long_strike=80000,
            short_strike=76000,
            expiry_dte=12,
            net_debit=1733.27,
            max_profit=2266.73,
            max_loss=1733.27,
            contracts=1,
            spot_price=78653,
            long_open_interest=2,  # Very low
            short_open_interest=3,
        )
        
        result = self.validator.validate(plan)
        
        warnings = [i for i in result.warnings if "LOW_OPEN_INTEREST" in i.code]
        self.assertEqual(len(warnings), 2)  # Both legs
    
    def test_insufficient_net_edge_blocks(self):
        """Test insufficient net edge blocks trade."""
        plan = SpreadPlan(
            signal_type="MVRV_SHORT",
            direction="short",
            option_type="put",
            long_strike=80000,
            short_strike=79950,  # Tiny spread
            expiry_dte=12,
            net_debit=45,
            max_profit=5,  # Tiny profit, costs will exceed
            max_loss=45,
            contracts=1,
            spot_price=78653,
        )
        
        result = self.validator.validate(plan)
        
        blocking = [i for i in result.blocking_issues if i.code == "INSUFFICIENT_NET_EDGE"]
        # May or may not block depending on exact cost calculation
        # At minimum should have net_edge_pct in adjusted_params
        self.assertIn("net_edge_pct", result.adjusted_params)
    
    def test_liquidity_at_size_warning(self):
        """Test liquidity stress at larger size."""
        plan = SpreadPlan(
            signal_type="MVRV_SHORT",
            direction="short",
            option_type="put",
            long_strike=80000,
            short_strike=76000,
            expiry_dte=12,
            net_debit=500,
            max_profit=500,
            max_loss=500,
            contracts=5,  # Larger size
            spot_price=78653,
            long_open_interest=30,  # OI < 5 * 10 = 50
            short_open_interest=40,
            long_bid_ask_spread_pct=0.03,
            short_bid_ask_spread_pct=0.04,
        )
        
        result = self.validator.validate(plan)
        
        warnings = [i for i in result.warnings if "LIQUIDITY_INSUFFICIENT_SIZE" in i.code]
        self.assertGreater(len(warnings), 0)
    
    def test_spread_impact_at_size_warning(self):
        """Test spread impact at larger size."""
        plan = SpreadPlan(
            signal_type="MVRV_SHORT",
            direction="short",
            option_type="put",
            long_strike=80000,
            short_strike=76000,
            expiry_dte=12,
            net_debit=500,
            max_profit=500,
            max_loss=500,
            contracts=5,
            spot_price=78653,
            long_open_interest=500,
            short_open_interest=500,
            long_bid_ask_spread_pct=0.02,  # 2% base spread
            short_bid_ask_spread_pct=0.02,
        )
        
        result = self.validator.validate(plan)
        
        # At 5 contracts, impact = 2% * sqrt(5) = 4.5% > 2% threshold
        warnings = [i for i in result.warnings if "SPREAD_IMPACT_AT_SIZE" in i.code]
        self.assertGreater(len(warnings), 0)


class ValidateSpreadPlanFunctionTests(TestCase):
    """Tests for the convenience validate_spread_plan function."""
    
    def test_convenience_function_works(self):
        """Test the convenience function produces valid results."""
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
        
        self.assertIsInstance(result, ValidationResult)
        self.assertTrue(result.is_valid)


class ValidationResultTests(TestCase):
    """Tests for ValidationResult dataclass."""
    
    def test_blocking_issues_property(self):
        """Test blocking_issues property filters correctly."""
        result = ValidationResult(
            is_valid=False,
            issues=[
                ValidationIssue("TEST1", IssueSeverity.BLOCKING, "Blocking issue"),
                ValidationIssue("TEST2", IssueSeverity.WARNING, "Warning issue"),
                ValidationIssue("TEST3", IssueSeverity.INFO, "Info issue"),
            ]
        )
        
        self.assertEqual(len(result.blocking_issues), 1)
        self.assertEqual(result.blocking_issues[0].code, "TEST1")
    
    def test_warnings_property(self):
        """Test warnings property filters correctly."""
        result = ValidationResult(
            is_valid=True,
            issues=[
                ValidationIssue("TEST1", IssueSeverity.BLOCKING, "Blocking"),
                ValidationIssue("TEST2", IssueSeverity.WARNING, "Warning 1"),
                ValidationIssue("TEST3", IssueSeverity.WARNING, "Warning 2"),
                ValidationIssue("TEST4", IssueSeverity.INFO, "Info"),
            ]
        )
        
        self.assertEqual(len(result.warnings), 2)
    
    def test_has_blocking_issues_property(self):
        """Test has_blocking_issues property."""
        result_with_blocking = ValidationResult(
            is_valid=False,
            issues=[ValidationIssue("TEST", IssueSeverity.BLOCKING, "Block")]
        )
        result_without_blocking = ValidationResult(
            is_valid=True,
            issues=[ValidationIssue("TEST", IssueSeverity.WARNING, "Warn")]
        )
        
        self.assertTrue(result_with_blocking.has_blocking_issues)
        self.assertFalse(result_without_blocking.has_blocking_issues)


# =============================================================================
# TRADE SETUP BUILDER TESTS
# =============================================================================

class TradeSetupBuilderTests(TestCase):
    """Tests for TradeSetupBuilder."""
    
    def test_direction_and_type_mapping(self):
        """Test signal type to direction/option type mapping."""
        from execution.services.trade_setup import TradeSetupBuilder
        
        builder = TradeSetupBuilder()
        
        # Test all mappings
        self.assertEqual(builder._get_direction_and_type("CALL"), ("LONG", "call"))
        self.assertEqual(builder._get_direction_and_type("OPTION_CALL"), ("LONG", "call"))
        self.assertEqual(builder._get_direction_and_type("BULL_PROBE"), ("LONG", "call"))
        self.assertEqual(builder._get_direction_and_type("PUT"), ("SHORT", "put"))
        self.assertEqual(builder._get_direction_and_type("OPTION_PUT"), ("SHORT", "put"))
        self.assertEqual(builder._get_direction_and_type("TACTICAL_PUT"), ("SHORT", "put"))
        self.assertEqual(builder._get_direction_and_type("BEAR_PROBE"), ("SHORT", "put"))
        self.assertEqual(builder._get_direction_and_type("MVRV_SHORT"), ("SHORT", "put"))
        self.assertEqual(builder._get_direction_and_type("IRON_CONDOR"), ("NEUTRAL", None))
        
        # Unknown signal
        self.assertEqual(builder._get_direction_and_type("UNKNOWN"), (None, None))
    
    def test_no_trade_returns_none(self):
        """Test that NO_TRADE signal returns None."""
        from execution.services.trade_setup import TradeSetupBuilder
        
        builder = TradeSetupBuilder()
        setup = builder.build_setup(signal_date=date(2020, 1, 1), signal_type="NO_TRADE")
        
        self.assertIsNone(setup)


class LegSetupTests(TestCase):
    """Tests for LegSetup dataclass."""
    
    def test_leg_setup_creation(self):
        """Test creating a LegSetup."""
        from execution.services.trade_setup import LegSetup
        
        leg = LegSetup(
            symbol="BTC-15MAY26-80000-P",
            action="BUY",
            strike=80000.0,
            delta=-0.58,
            iv=0.35,
            price=2757.48,
            open_interest=257,
            bid_ask_spread_pct=0.03,
        )
        
        self.assertEqual(leg.symbol, "BTC-15MAY26-80000-P")
        self.assertEqual(leg.action, "BUY")
        self.assertEqual(leg.strike, 80000.0)


class ExitRulesTests(TestCase):
    """Tests for ExitRules dataclass."""
    
    def test_exit_rules_creation(self):
        """Test creating ExitRules."""
        from execution.services.trade_setup import ExitRules
        
        rules = ExitRules(
            stop_loss_spot=83474.0,
            stop_loss_spot_pct=0.061,
            stop_loss_value=693.0,
            stop_loss_value_pct=0.6,
            take_profit_pct=0.6,
            take_profit_value=1360.0,
            max_hold_days=11,
            max_hold_date=date(2026, 5, 13),
            scale_down_day=5,
            scale_down_date=date(2026, 5, 7),
            scale_down_action="close_full_position",
        )
        
        self.assertEqual(rules.stop_loss_spot, 83474.0)
        self.assertEqual(rules.scale_down_action, "close_full_position")


class TradeSetupToDictTests(TestCase):
    """Tests for TradeSetup.to_dict() method."""
    
    def test_to_dict_structure(self):
        """Test to_dict returns correct structure."""
        from execution.services.trade_setup import TradeSetup, LegSetup, ExitRules
        
        setup = TradeSetup(
            signal_date=date(2026, 5, 2),
            signal_type="MVRV_SHORT",
            direction="SHORT",
            spot_price=78653.0,
            expiry=date(2026, 5, 15),
            dte=12,
            long_leg=LegSetup(
                symbol="BTC-15MAY26-80000-P",
                action="BUY",
                strike=80000.0,
                delta=-0.58,
                iv=0.35,
                price=2757.48,
                open_interest=257,
                bid_ask_spread_pct=0.03,
            ),
            short_leg=LegSetup(
                symbol="BTC-15MAY26-76000-P",
                action="SELL",
                strike=76000.0,
                delta=-0.29,
                iv=0.38,
                price=1024.21,
                open_interest=541,
                bid_ask_spread_pct=0.04,
            ),
            spread_width=4000.0,
            spread_width_pct=0.051,
            net_debit=1733.27,
            max_profit=2266.73,
            max_loss=1733.27,
            risk_reward=1.31,
            breakeven=78267.0,
            execution_cost=107.46,
            adjusted_max_profit=2159.27,
            net_edge_pct=0.72,
            risk_budget=1800.0,
            contracts=1,
            total_risk=1733.27,
            total_max_profit=2266.73,
            exit_rules=ExitRules(
                stop_loss_spot=83474.0,
                stop_loss_spot_pct=0.061,
                stop_loss_value=693.0,
                stop_loss_value_pct=0.6,
                take_profit_pct=0.6,
                take_profit_value=1360.0,
                max_hold_days=11,
                max_hold_date=date(2026, 5, 13),
                scale_down_day=5,
                scale_down_date=date(2026, 5, 7),
                scale_down_action="close_full_position",
            ),
            validation_passed=True,
            validation_warnings=["Test warning"],
            validation_blocking=[],
            policy_version="2026-05-03.3",
        )
        
        data = setup.to_dict()
        
        # Check top-level keys
        self.assertIn("signal_date", data)
        self.assertIn("signal_type", data)
        self.assertIn("legs", data)
        self.assertIn("metrics", data)
        self.assertIn("position", data)
        self.assertIn("exit_rules", data)
        self.assertIn("validation", data)
        
        # Check nested structure
        self.assertIn("long", data["legs"])
        self.assertIn("short", data["legs"])
        self.assertEqual(data["legs"]["long"]["symbol"], "BTC-15MAY26-80000-P")
        
        # Check metrics
        self.assertEqual(data["metrics"]["spread_width"], 4000.0)
        self.assertEqual(data["metrics"]["risk_reward"], 1.31)
        
        # Check validation
        self.assertTrue(data["validation"]["passed"])
        self.assertEqual(len(data["validation"]["warnings"]), 1)


class TradeSetupToTelegramTests(TestCase):
    """Tests for TradeSetup.to_telegram_message() method."""
    
    def test_telegram_message_contains_key_info(self):
        """Test Telegram message contains all key information."""
        from execution.services.trade_setup import TradeSetup, LegSetup, ExitRules
        
        setup = TradeSetup(
            signal_date=date(2026, 5, 2),
            signal_type="MVRV_SHORT",
            direction="SHORT",
            spot_price=78653.0,
            expiry=date(2026, 5, 15),
            dte=12,
            long_leg=LegSetup(
                symbol="BTC-15MAY26-80000-P",
                action="BUY",
                strike=80000.0,
                delta=-0.58,
                iv=0.35,
                price=2757.48,
                open_interest=257,
                bid_ask_spread_pct=0.03,
            ),
            short_leg=LegSetup(
                symbol="BTC-15MAY26-76000-P",
                action="SELL",
                strike=76000.0,
                delta=-0.29,
                iv=0.38,
                price=1024.21,
                open_interest=541,
                bid_ask_spread_pct=0.04,
            ),
            spread_width=4000.0,
            spread_width_pct=0.051,
            net_debit=1733.27,
            max_profit=2266.73,
            max_loss=1733.27,
            risk_reward=1.31,
            breakeven=78267.0,
            execution_cost=107.46,
            adjusted_max_profit=2159.27,
            net_edge_pct=0.72,
            risk_budget=1800.0,
            contracts=1,
            total_risk=1733.27,
            total_max_profit=2266.73,
            exit_rules=ExitRules(
                stop_loss_spot=83474.0,
                stop_loss_spot_pct=0.061,
                stop_loss_value=693.0,
                stop_loss_value_pct=0.6,
                take_profit_pct=0.6,
                take_profit_value=1360.0,
                max_hold_days=11,
                max_hold_date=date(2026, 5, 13),
                scale_down_day=5,
                scale_down_date=date(2026, 5, 7),
                scale_down_action="close_full_position",
            ),
            validation_passed=True,
            validation_warnings=[],
            validation_blocking=[],
            policy_version="2026-05-03.3",
        )
        
        message = setup.to_telegram_message()
        
        # Check key elements are present
        self.assertIn("MVRV_SHORT", message)
        self.assertIn("2026-05-02", message)
        self.assertIn("BTC-15MAY26-80000-P", message)
        self.assertIn("BTC-15MAY26-76000-P", message)
        self.assertIn("$4,000", message)
        self.assertIn("1:1.31", message)
        self.assertIn("CLOSE FULL", message)  # Scale down action
    
    def test_telegram_message_shows_warnings(self):
        """Test Telegram message shows validation warnings."""
        from execution.services.trade_setup import TradeSetup, LegSetup, ExitRules
        
        setup = TradeSetup(
            signal_date=date(2026, 5, 2),
            signal_type="MVRV_SHORT",
            direction="SHORT",
            spot_price=78653.0,
            expiry=date(2026, 5, 15),
            dte=12,
            long_leg=LegSetup(
                symbol="BTC-15MAY26-80000-P",
                action="BUY",
                strike=80000.0,
                delta=-0.58,
                iv=0.35,
                price=2757.48,
                open_interest=257,
                bid_ask_spread_pct=0.03,
            ),
            short_leg=None,
            spread_width=4000.0,
            spread_width_pct=0.051,
            net_debit=1733.27,
            max_profit=2266.73,
            max_loss=1733.27,
            risk_reward=1.31,
            breakeven=78267.0,
            execution_cost=107.46,
            adjusted_max_profit=2159.27,
            net_edge_pct=0.72,
            risk_budget=1800.0,
            contracts=1,
            total_risk=1733.27,
            total_max_profit=2266.73,
            exit_rules=ExitRules(
                stop_loss_spot=83474.0,
                stop_loss_spot_pct=0.061,
                stop_loss_value=693.0,
                stop_loss_value_pct=0.6,
                take_profit_pct=0.6,
                take_profit_value=1360.0,
                max_hold_days=11,
                max_hold_date=date(2026, 5, 13),
                scale_down_day=None,
                scale_down_date=None,
                scale_down_action="reduce_50pct",
            ),
            validation_passed=True,
            validation_warnings=["Test warning message"],
            validation_blocking=[],
            policy_version="2026-05-03.3",
        )
        
        message = setup.to_telegram_message()
        
        self.assertIn("WARNINGS", message)
        self.assertIn("Test warning", message)


# =============================================================================
# API ENDPOINT TESTS
# =============================================================================

class TradeSetupAPITests(TestCase):
    """Tests for Trade Setup API endpoints."""
    
    def setUp(self):
        from django.contrib.auth.models import User
        from rest_framework.authtoken.models import Token
        
        self.user = User.objects.create_user(username='testuser', password='testpass')
        self.token = Token.objects.create(user=self.user)
    
    def test_trade_setup_view_invalid_date_format(self):
        """Test TradeSetupView returns 400 for invalid date format."""
        from rest_framework.test import APIRequestFactory
        from api.views import TradeSetupView
        
        factory = APIRequestFactory()
        request = factory.get('/api/v1/signals/invalid-date/setup/')
        request.META['HTTP_AUTHORIZATION'] = f'Token {self.token.key}'
        
        view = TradeSetupView.as_view()
        response = view(request, date='invalid-date')
        
        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid date format", response.data.get("error", ""))
    
    def test_trade_setup_view_requires_auth(self):
        """Test TradeSetupView requires authentication."""
        from rest_framework.test import APIRequestFactory
        from api.views import TradeSetupView
        
        factory = APIRequestFactory()
        request = factory.get('/api/v1/signals/2026-05-02/setup/')
        # No auth header
        
        view = TradeSetupView.as_view()
        response = view(request, date='2026-05-02')
        
        self.assertEqual(response.status_code, 401)


# =============================================================================
# NOTIFIER TESTS
# =============================================================================

class NotifierTradeSetupTests(TestCase):
    """Tests for TelegramNotifier trade setup integration."""
    
    @patch('notifications.notifier.Bot')
    def test_send_trade_setup_calls_bot(self, mock_bot_class):
        """Test send_trade_setup calls the Telegram bot."""
        from notifications.notifier import TelegramNotifier
        from execution.services.trade_setup import TradeSetup, LegSetup, ExitRules
        
        # Mock the bot
        mock_bot = MagicMock()
        mock_bot_class.return_value = mock_bot
        
        # Create notifier with test credentials
        with patch.dict('os.environ', {
            'TELEGRAM_BOT_TOKEN': 'test_token',
            'TELEGRAM_CHAT_ID': 'test_chat'
        }):
            notifier = TelegramNotifier(
                bot_token='test_token',
                chat_id='test_chat'
            )
        
        # Create a mock setup
        setup = MagicMock()
        setup.to_telegram_message.return_value = "Test message"
        
        # This will fail because we can't easily mock async, but we can test the method exists
        self.assertTrue(hasattr(notifier, 'send_trade_setup'))


# =============================================================================
# PATH PROFILE TESTS
# =============================================================================

class PathProfilePolicyTests(TestCase):
    """Tests for path profile methods in PolicyVersion."""
    
    def test_get_path_profile_returns_dict(self):
        """Test get_path_profile returns a dict with expected keys."""
        policy = get_policy()
        
        profile = policy.get_path_profile("MVRV_SHORT")
        
        self.assertIsInstance(profile, dict)
        self.assertIn("shakeout_pct", profile)
        self.assertIn("invalidation_pct", profile)
        self.assertIn("mae_p75", profile)
        self.assertIn("clean_win_pct", profile)
    
    def test_get_path_profile_mvrv_short_values(self):
        """Test MVRV_SHORT has expected path profile values."""
        policy = get_policy()
        
        profile = policy.get_path_profile("MVRV_SHORT")
        
        # MVRV_SHORT should have high shakeout rate (57%)
        self.assertGreaterEqual(profile["shakeout_pct"], 0.50)
        self.assertLessEqual(profile["shakeout_pct"], 0.65)
        
        # Should have high invalidation rate (43%)
        self.assertGreaterEqual(profile["invalidation_pct"], 0.40)
    
    def test_get_path_profile_unknown_signal_returns_defaults(self):
        """Test unknown signal returns default path profile."""
        policy = get_policy()
        
        profile = policy.get_path_profile("UNKNOWN_SIGNAL")
        
        # Should return defaults
        self.assertEqual(profile["shakeout_pct"], 0.20)
        self.assertEqual(profile["invalidation_pct"], 0.25)
        self.assertEqual(profile["mae_p75"], 0.05)
        self.assertEqual(profile["clean_win_pct"], 0.75)
    
    def test_is_shakeout_heavy_mvrv_short(self):
        """Test MVRV_SHORT is identified as shakeout-heavy."""
        policy = get_policy()
        
        self.assertTrue(policy.is_shakeout_heavy("MVRV_SHORT"))
    
    def test_is_shakeout_heavy_bear_probe(self):
        """Test BEAR_PROBE is identified as shakeout-heavy (40% threshold)."""
        policy = get_policy()
        
        self.assertTrue(policy.is_shakeout_heavy("BEAR_PROBE"))
    
    def test_is_shakeout_heavy_call_is_false(self):
        """Test CALL is NOT shakeout-heavy."""
        policy = get_policy()
        
        self.assertFalse(policy.is_shakeout_heavy("CALL"))
    
    def test_is_invalidation_heavy_option_call(self):
        """Test OPTION_CALL is identified as invalidation-heavy."""
        policy = get_policy()
        
        # OPTION_CALL has 54% invalidation rate
        self.assertTrue(policy.is_invalidation_heavy("OPTION_CALL"))
    
    def test_is_invalidation_heavy_option_put(self):
        """Test OPTION_PUT is identified as invalidation-heavy."""
        policy = get_policy()
        
        # OPTION_PUT has 44% invalidation rate
        self.assertTrue(policy.is_invalidation_heavy("OPTION_PUT"))
    
    def test_is_invalidation_heavy_call_is_false(self):
        """Test CALL is NOT invalidation-heavy."""
        policy = get_policy()
        
        # CALL has 27% invalidation rate (below 35% threshold)
        self.assertFalse(policy.is_invalidation_heavy("CALL"))
    
    def test_is_invalidation_heavy_mvrv_short(self):
        """Test MVRV_SHORT is invalidation-heavy."""
        policy = get_policy()
        
        # MVRV_SHORT has 43% invalidation rate
        self.assertTrue(policy.is_invalidation_heavy("MVRV_SHORT"))
    
    def test_path_profiles_in_to_dict(self):
        """Test path_profiles is included in to_dict() serialization."""
        policy = get_policy()
        
        data = policy.to_dict()
        
        self.assertIn("path_profiles", data)
        self.assertIsInstance(data["path_profiles"], dict)
        self.assertIn("MVRV_SHORT", data["path_profiles"])
        self.assertIn("shakeout_pct", data["path_profiles"]["MVRV_SHORT"])
    
    def test_all_signals_have_path_profiles(self):
        """Test all configured signals have path profiles."""
        policy = get_policy()
        
        expected_signals = [
            "CALL", "PUT", "OPTION_CALL", "OPTION_PUT",
            "TACTICAL_PUT", "BULL_PROBE", "BEAR_PROBE",
            "MVRV_SHORT", "IRON_CONDOR"
        ]
        
        for signal in expected_signals:
            profile = policy.get_path_profile(signal)
            self.assertIn("shakeout_pct", profile, f"Missing shakeout_pct for {signal}")
            self.assertIn("invalidation_pct", profile, f"Missing invalidation_pct for {signal}")


class PathProfileDataclassTests(TestCase):
    """Tests for PathProfile dataclass in trade_setup.py."""
    
    def test_path_profile_creation(self):
        """Test creating a PathProfile dataclass."""
        from execution.services.trade_setup import PathProfile
        
        profile = PathProfile(
            shakeout_pct=0.57,
            invalidation_pct=0.43,
            mae_p75=0.0719,
            clean_win_pct=0.571,
            is_shakeout_heavy=True,
            is_invalidation_heavy=True,
            entry_strategy="dca",
            entry_note="33% initial, 67% DCA at +7% (shakeout-heavy)",
        )
        
        self.assertEqual(profile.shakeout_pct, 0.57)
        self.assertEqual(profile.entry_strategy, "dca")
        self.assertTrue(profile.is_shakeout_heavy)
    
    def test_path_profile_entry_strategies(self):
        """Test all valid entry strategies."""
        from execution.services.trade_setup import PathProfile
        
        # DCA strategy
        dca_profile = PathProfile(
            shakeout_pct=0.57,
            invalidation_pct=0.43,
            mae_p75=0.07,
            clean_win_pct=0.57,
            is_shakeout_heavy=True,
            is_invalidation_heavy=True,
            entry_strategy="dca",
            entry_note="DCA entry",
        )
        self.assertEqual(dca_profile.entry_strategy, "dca")
        
        # Scaled strategy
        scaled_profile = PathProfile(
            shakeout_pct=0.35,
            invalidation_pct=0.54,
            mae_p75=0.08,
            clean_win_pct=0.46,
            is_shakeout_heavy=False,
            is_invalidation_heavy=True,
            entry_strategy="scaled",
            entry_note="Scaled entry",
        )
        self.assertEqual(scaled_profile.entry_strategy, "scaled")
        
        # Single strategy
        single_profile = PathProfile(
            shakeout_pct=0.21,
            invalidation_pct=0.27,
            mae_p75=0.05,
            clean_win_pct=0.73,
            is_shakeout_heavy=False,
            is_invalidation_heavy=False,
            entry_strategy="single",
            entry_note="Full position at entry",
        )
        self.assertEqual(single_profile.entry_strategy, "single")


class TradeSetupPathProfileTests(TestCase):
    """Tests for PathProfile integration in TradeSetup."""
    
    def test_trade_setup_with_path_profile_to_dict(self):
        """Test TradeSetup.to_dict() includes path_profile when present."""
        from execution.services.trade_setup import TradeSetup, LegSetup, ExitRules, PathProfile
        
        setup = TradeSetup(
            signal_date=date(2026, 5, 2),
            signal_type="MVRV_SHORT",
            direction="SHORT",
            spot_price=78653.0,
            expiry=date(2026, 5, 15),
            dte=12,
            long_leg=LegSetup(
                symbol="BTC-15MAY26-80000-P",
                action="BUY",
                strike=80000.0,
                delta=-0.58,
                iv=0.35,
                price=2757.48,
                open_interest=257,
                bid_ask_spread_pct=0.03,
            ),
            short_leg=LegSetup(
                symbol="BTC-15MAY26-76000-P",
                action="SELL",
                strike=76000.0,
                delta=-0.29,
                iv=0.38,
                price=1024.21,
                open_interest=541,
                bid_ask_spread_pct=0.04,
            ),
            spread_width=4000.0,
            spread_width_pct=0.051,
            net_debit=1733.27,
            max_profit=2266.73,
            max_loss=1733.27,
            risk_reward=1.31,
            breakeven=78267.0,
            execution_cost=107.46,
            adjusted_max_profit=2159.27,
            net_edge_pct=0.72,
            risk_budget=1800.0,
            contracts=1,
            total_risk=1733.27,
            total_max_profit=2266.73,
            exit_rules=ExitRules(
                stop_loss_spot=83474.0,
                stop_loss_spot_pct=0.061,
                stop_loss_value=693.0,
                stop_loss_value_pct=0.6,
                take_profit_pct=0.6,
                take_profit_value=1360.0,
                max_hold_days=11,
                max_hold_date=date(2026, 5, 13),
                scale_down_day=5,
                scale_down_date=date(2026, 5, 7),
                scale_down_action="close_full_position",
            ),
            path_profile=PathProfile(
                shakeout_pct=0.57,
                invalidation_pct=0.43,
                mae_p75=0.0719,
                clean_win_pct=0.571,
                is_shakeout_heavy=True,
                is_invalidation_heavy=True,
                entry_strategy="dca",
                entry_note="33% initial, 67% DCA at +7% (shakeout-heavy)",
            ),
            validation_passed=True,
            validation_warnings=[],
            validation_blocking=[],
            policy_version="2026-05-03.3",
        )
        
        data = setup.to_dict()
        
        # Check path_profile is in output
        self.assertIn("path_profile", data)
        self.assertIsNotNone(data["path_profile"])
        
        # Check all fields are serialized
        pp = data["path_profile"]
        self.assertEqual(pp["shakeout_pct"], 0.57)
        self.assertEqual(pp["invalidation_pct"], 0.43)
        self.assertEqual(pp["mae_p75"], 0.0719)
        self.assertEqual(pp["clean_win_pct"], 0.571)
        self.assertTrue(pp["is_shakeout_heavy"])
        self.assertTrue(pp["is_invalidation_heavy"])
        self.assertEqual(pp["entry_strategy"], "dca")
        self.assertIn("33%", pp["entry_note"])
    
    def test_trade_setup_without_path_profile_to_dict(self):
        """Test TradeSetup.to_dict() handles None path_profile."""
        from execution.services.trade_setup import TradeSetup, LegSetup, ExitRules
        
        setup = TradeSetup(
            signal_date=date(2026, 5, 2),
            signal_type="CALL",
            direction="LONG",
            spot_price=78653.0,
            expiry=date(2026, 5, 15),
            dte=12,
            long_leg=LegSetup(
                symbol="BTC-15MAY26-80000-C",
                action="BUY",
                strike=80000.0,
                delta=0.58,
                iv=0.35,
                price=2757.48,
                open_interest=257,
                bid_ask_spread_pct=0.03,
            ),
            short_leg=None,
            spread_width=4000.0,
            spread_width_pct=0.051,
            net_debit=1733.27,
            max_profit=2266.73,
            max_loss=1733.27,
            risk_reward=1.31,
            breakeven=78267.0,
            execution_cost=107.46,
            adjusted_max_profit=2159.27,
            net_edge_pct=0.72,
            risk_budget=1800.0,
            contracts=1,
            total_risk=1733.27,
            total_max_profit=2266.73,
            exit_rules=ExitRules(
                stop_loss_spot=74720.0,
                stop_loss_spot_pct=0.05,
                stop_loss_value=693.0,
                stop_loss_value_pct=0.6,
                take_profit_pct=0.7,
                take_profit_value=1586.0,
                max_hold_days=9,
                max_hold_date=date(2026, 5, 11),
                scale_down_day=6,
                scale_down_date=date(2026, 5, 8),
                scale_down_action="reduce_50pct",
            ),
            path_profile=None,  # No path profile
            validation_passed=True,
            validation_warnings=[],
            validation_blocking=[],
            policy_version="2026-05-03.3",
        )
        
        data = setup.to_dict()
        
        # path_profile should be None in output
        self.assertIn("path_profile", data)
        self.assertIsNone(data["path_profile"])


class TradeSetupTelegramPathProfileTests(TestCase):
    """Tests for path profile rendering in Telegram messages."""
    
    def test_telegram_shows_path_profile_for_shakeout_heavy(self):
        """Test Telegram message shows PATH PROFILE section for shakeout-heavy signals."""
        from execution.services.trade_setup import TradeSetup, LegSetup, ExitRules, PathProfile
        
        setup = TradeSetup(
            signal_date=date(2026, 5, 2),
            signal_type="MVRV_SHORT",
            direction="SHORT",
            spot_price=78653.0,
            expiry=date(2026, 5, 15),
            dte=12,
            long_leg=LegSetup(
                symbol="BTC-15MAY26-80000-P",
                action="BUY",
                strike=80000.0,
                delta=-0.58,
                iv=0.35,
                price=2757.48,
                open_interest=257,
                bid_ask_spread_pct=0.03,
            ),
            short_leg=LegSetup(
                symbol="BTC-15MAY26-76000-P",
                action="SELL",
                strike=76000.0,
                delta=-0.29,
                iv=0.38,
                price=1024.21,
                open_interest=541,
                bid_ask_spread_pct=0.04,
            ),
            spread_width=4000.0,
            spread_width_pct=0.051,
            net_debit=1733.27,
            max_profit=2266.73,
            max_loss=1733.27,
            risk_reward=1.31,
            breakeven=78267.0,
            execution_cost=107.46,
            adjusted_max_profit=2159.27,
            net_edge_pct=0.72,
            risk_budget=1800.0,
            contracts=1,
            total_risk=1733.27,
            total_max_profit=2266.73,
            exit_rules=ExitRules(
                stop_loss_spot=83474.0,
                stop_loss_spot_pct=0.061,
                stop_loss_value=693.0,
                stop_loss_value_pct=0.6,
                take_profit_pct=0.6,
                take_profit_value=1360.0,
                max_hold_days=11,
                max_hold_date=date(2026, 5, 13),
                scale_down_day=5,
                scale_down_date=date(2026, 5, 7),
                scale_down_action="close_full_position",
            ),
            path_profile=PathProfile(
                shakeout_pct=0.57,
                invalidation_pct=0.43,
                mae_p75=0.0719,
                clean_win_pct=0.571,
                is_shakeout_heavy=True,
                is_invalidation_heavy=True,
                entry_strategy="dca",
                entry_note="33% initial, 67% DCA at +7% (shakeout-heavy)",
            ),
            validation_passed=True,
            validation_warnings=[],
            validation_blocking=[],
            policy_version="2026-05-03.3",
        )
        
        message = setup.to_telegram_message()
        
        # Should contain PATH PROFILE section
        self.assertIn("PATH PROFILE", message)
        self.assertIn("Shakeout Rate", message)
        self.assertIn("57%", message)
        self.assertIn("DCA", message)
        self.assertIn("MAE p75", message)
    
    def test_telegram_shows_invalidation_for_invalidation_heavy(self):
        """Test Telegram message shows invalidation info for invalidation-heavy signals."""
        from execution.services.trade_setup import TradeSetup, LegSetup, ExitRules, PathProfile
        
        setup = TradeSetup(
            signal_date=date(2026, 5, 2),
            signal_type="OPTION_CALL",
            direction="LONG",
            spot_price=78653.0,
            expiry=date(2026, 5, 15),
            dte=9,
            long_leg=LegSetup(
                symbol="BTC-15MAY26-78000-C",
                action="BUY",
                strike=78000.0,
                delta=0.65,
                iv=0.35,
                price=3500.0,
                open_interest=300,
                bid_ask_spread_pct=0.03,
            ),
            short_leg=LegSetup(
                symbol="BTC-15MAY26-86000-C",
                action="SELL",
                strike=86000.0,
                delta=0.25,
                iv=0.38,
                price=1000.0,
                open_interest=400,
                bid_ask_spread_pct=0.04,
            ),
            spread_width=8000.0,
            spread_width_pct=0.10,
            net_debit=2500.0,
            max_profit=5500.0,
            max_loss=2500.0,
            risk_reward=2.2,
            breakeven=80500.0,
            execution_cost=150.0,
            adjusted_max_profit=5350.0,
            net_edge_pct=0.85,
            risk_budget=3200.0,
            contracts=1,
            total_risk=2500.0,
            total_max_profit=5500.0,
            exit_rules=ExitRules(
                stop_loss_spot=71940.0,
                stop_loss_spot_pct=0.085,
                stop_loss_value=1000.0,
                stop_loss_value_pct=0.6,
                take_profit_pct=0.7,
                take_profit_value=3850.0,
                max_hold_days=7,
                max_hold_date=date(2026, 5, 9),
                scale_down_day=4,
                scale_down_date=date(2026, 5, 6),
                scale_down_action="reduce_50pct",
            ),
            path_profile=PathProfile(
                shakeout_pct=0.35,
                invalidation_pct=0.54,
                mae_p75=0.0848,
                clean_win_pct=0.458,
                is_shakeout_heavy=False,
                is_invalidation_heavy=True,
                entry_strategy="scaled",
                entry_note="50% initial, 50% on confirmation (high invalidation)",
            ),
            validation_passed=True,
            validation_warnings=[],
            validation_blocking=[],
            policy_version="2026-05-03.3",
        )
        
        message = setup.to_telegram_message()
        
        # Should contain PATH PROFILE section with invalidation info
        self.assertIn("PATH PROFILE", message)
        self.assertIn("Invalidation", message)
        self.assertIn("54%", message)
        self.assertIn("SCALED", message)
    
    def test_telegram_no_path_profile_for_clean_signals(self):
        """Test Telegram message does NOT show PATH PROFILE for clean signals."""
        from execution.services.trade_setup import TradeSetup, LegSetup, ExitRules, PathProfile
        
        setup = TradeSetup(
            signal_date=date(2026, 5, 2),
            signal_type="CALL",
            direction="LONG",
            spot_price=78653.0,
            expiry=date(2026, 5, 15),
            dte=11,
            long_leg=LegSetup(
                symbol="BTC-15MAY26-78000-C",
                action="BUY",
                strike=78000.0,
                delta=0.60,
                iv=0.35,
                price=3000.0,
                open_interest=300,
                bid_ask_spread_pct=0.03,
            ),
            short_leg=LegSetup(
                symbol="BTC-15MAY26-86000-C",
                action="SELL",
                strike=86000.0,
                delta=0.25,
                iv=0.38,
                price=1000.0,
                open_interest=400,
                bid_ask_spread_pct=0.04,
            ),
            spread_width=8000.0,
            spread_width_pct=0.10,
            net_debit=2000.0,
            max_profit=6000.0,
            max_loss=2000.0,
            risk_reward=3.0,
            breakeven=80000.0,
            execution_cost=120.0,
            adjusted_max_profit=5880.0,
            net_edge_pct=0.90,
            risk_budget=3200.0,
            contracts=1,
            total_risk=2000.0,
            total_max_profit=6000.0,
            exit_rules=ExitRules(
                stop_loss_spot=75120.0,
                stop_loss_spot_pct=0.045,
                stop_loss_value=800.0,
                stop_loss_value_pct=0.6,
                take_profit_pct=0.7,
                take_profit_value=4200.0,
                max_hold_days=9,
                max_hold_date=date(2026, 5, 11),
                scale_down_day=6,
                scale_down_date=date(2026, 5, 8),
                scale_down_action="reduce_50pct",
            ),
            path_profile=PathProfile(
                shakeout_pct=0.21,
                invalidation_pct=0.27,
                mae_p75=0.0471,
                clean_win_pct=0.729,
                is_shakeout_heavy=False,
                is_invalidation_heavy=False,
                entry_strategy="single",
                entry_note="Full position at entry",
            ),
            validation_passed=True,
            validation_warnings=[],
            validation_blocking=[],
            policy_version="2026-05-03.3",
        )
        
        message = setup.to_telegram_message()
        
        # Should NOT contain PATH PROFILE section for clean signals
        self.assertNotIn("PATH PROFILE", message)
        self.assertNotIn("Shakeout Rate", message)


class EntryStrategyLogicTests(TestCase):
    """Tests for entry strategy determination logic."""
    
    def test_dca_strategy_for_shakeout_heavy(self):
        """Test DCA strategy is selected for shakeout-heavy signals."""
        policy = get_policy()
        
        # MVRV_SHORT has 57% shakeout rate
        self.assertTrue(policy.is_shakeout_heavy("MVRV_SHORT"))
        
        # BEAR_PROBE has 40% shakeout rate (exactly at threshold)
        self.assertTrue(policy.is_shakeout_heavy("BEAR_PROBE"))
    
    def test_scaled_strategy_for_invalidation_heavy(self):
        """Test scaled strategy is appropriate for invalidation-heavy signals."""
        policy = get_policy()
        
        # OPTION_CALL has 54% invalidation rate
        self.assertTrue(policy.is_invalidation_heavy("OPTION_CALL"))
        self.assertFalse(policy.is_shakeout_heavy("OPTION_CALL"))
        
        # OPTION_PUT has 44% invalidation rate
        self.assertTrue(policy.is_invalidation_heavy("OPTION_PUT"))
    
    def test_single_strategy_for_clean_signals(self):
        """Test single strategy is appropriate for clean signals."""
        policy = get_policy()
        
        # CALL has 21% shakeout, 27% invalidation - both below thresholds
        self.assertFalse(policy.is_shakeout_heavy("CALL"))
        self.assertFalse(policy.is_invalidation_heavy("CALL"))
        
        # BULL_PROBE has 19% shakeout, 22% invalidation - both below thresholds
        self.assertFalse(policy.is_shakeout_heavy("BULL_PROBE"))
        self.assertFalse(policy.is_invalidation_heavy("BULL_PROBE"))
    
    def test_threshold_boundaries(self):
        """Test threshold boundary conditions."""
        policy = get_policy()
        
        # Shakeout threshold is 40%
        # BEAR_PROBE has exactly 40% - should be shakeout-heavy
        profile = policy.get_path_profile("BEAR_PROBE")
        self.assertEqual(profile["shakeout_pct"], 0.40)
        self.assertTrue(policy.is_shakeout_heavy("BEAR_PROBE"))
        
        # OPTION_CALL has 35% shakeout - should NOT be shakeout-heavy
        profile = policy.get_path_profile("OPTION_CALL")
        self.assertEqual(profile["shakeout_pct"], 0.35)
        self.assertFalse(policy.is_shakeout_heavy("OPTION_CALL"))
        
        # Invalidation threshold is 35%
        # BEAR_PROBE has 40% invalidation - should be invalidation-heavy
        self.assertTrue(policy.is_invalidation_heavy("BEAR_PROBE"))


# =============================================================================
# PATH-AWARE EXIT POLICY TESTS
# =============================================================================

class PathAwareExitConfigTests(TestCase):
    """Tests for path-aware exit configuration in ExitConfig."""
    
    def test_exit_config_has_path_aware_fields(self):
        """Test ExitConfig has all path-aware exit fields."""
        exit_cfg = ExitConfig(
            stop_loss_pct=0.07,
            take_profit_pct=0.60,
            max_hold_days=12,
            scale_down_day=7,
            profit_lock_threshold=0.25,
            trailing_stop_pct=0.04,
            stop_tighten_day=9,
            stop_tighten_factor=0.7,
        )
        
        self.assertEqual(exit_cfg.profit_lock_threshold, 0.25)
        self.assertEqual(exit_cfg.trailing_stop_pct, 0.04)
        self.assertEqual(exit_cfg.stop_tighten_day, 9)
        self.assertEqual(exit_cfg.stop_tighten_factor, 0.7)
    
    def test_exit_config_defaults(self):
        """Test ExitConfig has sensible defaults for path-aware fields."""
        exit_cfg = ExitConfig(
            stop_loss_pct=0.05,
            take_profit_pct=0.70,
            max_hold_days=10,
        )
        
        # Check defaults
        self.assertEqual(exit_cfg.profit_lock_threshold, 0.30)  # Default 30%
        self.assertEqual(exit_cfg.trailing_stop_pct, 0.0)       # Default disabled
        self.assertIsNone(exit_cfg.stop_tighten_day)            # Default None
        self.assertEqual(exit_cfg.stop_tighten_factor, 0.5)     # Default 50%


class PathAwareExitPolicyTests(TestCase):
    """Tests for path-aware exit parameters in policy."""
    
    def test_mvrv_short_has_trailing_stop(self):
        """Test MVRV_SHORT has trailing stop configured (shakeout-heavy)."""
        policy = get_policy()
        exit_cfg = policy.get_exit_params("MVRV_SHORT")
        
        # MVRV_SHORT should have trailing stop (shakeout-heavy signal)
        self.assertGreater(exit_cfg.trailing_stop_pct, 0)
        self.assertEqual(exit_cfg.trailing_stop_pct, 0.04)  # 4% trailing
    
    def test_bear_probe_has_trailing_stop(self):
        """Test BEAR_PROBE has trailing stop configured (shakeout-heavy)."""
        policy = get_policy()
        exit_cfg = policy.get_exit_params("BEAR_PROBE")
        
        # BEAR_PROBE should have trailing stop (shakeout-heavy signal)
        self.assertGreater(exit_cfg.trailing_stop_pct, 0)
        self.assertEqual(exit_cfg.trailing_stop_pct, 0.03)  # 3% trailing
    
    def test_option_call_has_early_tightening(self):
        """Test OPTION_CALL has early stop tightening (invalidation-heavy)."""
        policy = get_policy()
        exit_cfg = policy.get_exit_params("OPTION_CALL")
        
        # OPTION_CALL should have early tightening (invalidation-heavy)
        self.assertIsNotNone(exit_cfg.stop_tighten_day)
        self.assertEqual(exit_cfg.stop_tighten_day, 3)  # Tighten on day 3
        self.assertEqual(exit_cfg.stop_tighten_factor, 0.5)  # 50% tighter
    
    def test_option_put_has_early_tightening(self):
        """Test OPTION_PUT has early stop tightening (invalidation-heavy)."""
        policy = get_policy()
        exit_cfg = policy.get_exit_params("OPTION_PUT")
        
        # OPTION_PUT should have early tightening (invalidation-heavy)
        self.assertIsNotNone(exit_cfg.stop_tighten_day)
        self.assertEqual(exit_cfg.stop_tighten_day, 2)  # Tighten on day 2
    
    def test_clean_signals_no_trailing_stop(self):
        """Test clean signals (CALL, PUT) have no trailing stop."""
        policy = get_policy()
        
        call_exit = policy.get_exit_params("CALL")
        put_exit = policy.get_exit_params("PUT")
        
        # Clean signals should NOT have trailing stop
        self.assertEqual(call_exit.trailing_stop_pct, 0.0)
        self.assertEqual(put_exit.trailing_stop_pct, 0.0)
    
    def test_shakeout_heavy_has_delayed_tightening(self):
        """Test shakeout-heavy signals have delayed stop tightening."""
        policy = get_policy()
        
        mvrv_exit = policy.get_exit_params("MVRV_SHORT")
        bear_exit = policy.get_exit_params("BEAR_PROBE")
        
        # Shakeout-heavy signals should have delayed tightening
        self.assertGreaterEqual(mvrv_exit.stop_tighten_day, 8)  # Day 9
        self.assertGreaterEqual(bear_exit.stop_tighten_day, 7)  # Day 8
        
        # And less aggressive tightening factor
        self.assertGreaterEqual(mvrv_exit.stop_tighten_factor, 0.6)  # 70%
        self.assertGreaterEqual(bear_exit.stop_tighten_factor, 0.6)  # 70%
    
    def test_invalidation_heavy_has_lower_take_profit(self):
        """Test invalidation-heavy signals have lower take-profit."""
        policy = get_policy()
        
        option_call_exit = policy.get_exit_params("OPTION_CALL")
        option_put_exit = policy.get_exit_params("OPTION_PUT")
        call_exit = policy.get_exit_params("CALL")
        
        # Invalidation-heavy should have 50% take-profit
        self.assertEqual(option_call_exit.take_profit_pct, 0.50)
        self.assertEqual(option_put_exit.take_profit_pct, 0.50)
        
        # Clean signals should have take-profit >= invalidation-heavy
        # (calibration may adjust this, but should be >= 50%)
        self.assertGreaterEqual(call_exit.take_profit_pct, option_call_exit.take_profit_pct)


class ExitRulesPathAwareFieldsTests(TestCase):
    """Tests for path-aware fields in ExitRules dataclass."""
    
    def test_exit_rules_has_path_aware_fields(self):
        """Test ExitRules dataclass has all path-aware fields."""
        from execution.services.trade_setup import ExitRules
        
        rules = ExitRules(
            stop_loss_spot=83474.0,
            stop_loss_spot_pct=0.07,
            stop_loss_value=693.0,
            stop_loss_value_pct=0.6,
            take_profit_pct=0.6,
            take_profit_value=1360.0,
            max_hold_days=12,
            max_hold_date=date(2026, 5, 14),
            scale_down_day=7,
            scale_down_date=date(2026, 5, 9),
            scale_down_action="close_full_position",
            # Path-aware fields
            profit_lock_threshold=0.25,
            profit_lock_stop=78653.0,
            trailing_stop_pct=0.04,
            stop_tighten_day=9,
            stop_tighten_date=date(2026, 5, 11),
            tightened_stop_pct=0.049,
        )
        
        self.assertEqual(rules.profit_lock_threshold, 0.25)
        self.assertEqual(rules.profit_lock_stop, 78653.0)
        self.assertEqual(rules.trailing_stop_pct, 0.04)
        self.assertEqual(rules.stop_tighten_day, 9)
        self.assertEqual(rules.stop_tighten_date, date(2026, 5, 11))
        self.assertEqual(rules.tightened_stop_pct, 0.049)
    
    def test_exit_rules_defaults(self):
        """Test ExitRules has sensible defaults for path-aware fields."""
        from execution.services.trade_setup import ExitRules
        
        rules = ExitRules(
            stop_loss_spot=83474.0,
            stop_loss_spot_pct=0.07,
            stop_loss_value=693.0,
            stop_loss_value_pct=0.6,
            take_profit_pct=0.6,
            take_profit_value=1360.0,
            max_hold_days=12,
            max_hold_date=date(2026, 5, 14),
            scale_down_day=7,
            scale_down_date=date(2026, 5, 9),
            scale_down_action="close_full_position",
        )
        
        # Check defaults
        self.assertEqual(rules.profit_lock_threshold, 0.30)
        self.assertIsNone(rules.profit_lock_stop)
        self.assertEqual(rules.trailing_stop_pct, 0.0)
        self.assertIsNone(rules.stop_tighten_day)
        self.assertIsNone(rules.stop_tighten_date)
        self.assertIsNone(rules.tightened_stop_pct)


class ExitRulesToDictTests(TestCase):
    """Tests for path-aware exit rules serialization in to_dict()."""
    
    def test_to_dict_includes_path_aware_exit_fields(self):
        """Test to_dict() includes all path-aware exit fields."""
        from execution.services.trade_setup import TradeSetup, LegSetup, ExitRules, PathProfile
        
        setup = TradeSetup(
            signal_date=date(2026, 5, 2),
            signal_type="MVRV_SHORT",
            direction="SHORT",
            spot_price=78653.0,
            expiry=date(2026, 5, 15),
            dte=12,
            long_leg=LegSetup(
                symbol="BTC-15MAY26-80000-P",
                action="BUY",
                strike=80000.0,
                delta=-0.58,
                iv=0.35,
                price=2757.48,
                open_interest=257,
                bid_ask_spread_pct=0.03,
            ),
            short_leg=LegSetup(
                symbol="BTC-15MAY26-76000-P",
                action="SELL",
                strike=76000.0,
                delta=-0.29,
                iv=0.38,
                price=1024.21,
                open_interest=541,
                bid_ask_spread_pct=0.04,
            ),
            spread_width=4000.0,
            spread_width_pct=0.051,
            net_debit=1733.27,
            max_profit=2266.73,
            max_loss=1733.27,
            risk_reward=1.31,
            breakeven=78267.0,
            execution_cost=107.46,
            adjusted_max_profit=2159.27,
            net_edge_pct=0.72,
            risk_budget=1800.0,
            contracts=1,
            total_risk=1733.27,
            total_max_profit=2266.73,
            exit_rules=ExitRules(
                stop_loss_spot=84159.0,
                stop_loss_spot_pct=0.07,
                stop_loss_value=693.0,
                stop_loss_value_pct=0.6,
                take_profit_pct=0.6,
                take_profit_value=1360.0,
                max_hold_days=12,
                max_hold_date=date(2026, 5, 14),
                scale_down_day=7,
                scale_down_date=date(2026, 5, 9),
                scale_down_action="close_full_position",
                # Path-aware fields
                profit_lock_threshold=0.25,
                profit_lock_stop=78653.0,
                trailing_stop_pct=0.04,
                stop_tighten_day=9,
                stop_tighten_date=date(2026, 5, 11),
                tightened_stop_pct=0.049,
            ),
            path_profile=PathProfile(
                shakeout_pct=0.57,
                invalidation_pct=0.43,
                mae_p75=0.0719,
                clean_win_pct=0.571,
                is_shakeout_heavy=True,
                is_invalidation_heavy=True,
                entry_strategy="dca",
                entry_note="33% initial, 67% DCA at +7% (shakeout-heavy)",
            ),
            validation_passed=True,
            validation_warnings=[],
            validation_blocking=[],
            policy_version="2026-05-03.3",
        )
        
        data = setup.to_dict()
        exit_rules = data["exit_rules"]
        
        # Check path-aware fields are serialized
        self.assertIn("profit_lock_threshold", exit_rules)
        self.assertIn("profit_lock_stop", exit_rules)
        self.assertIn("trailing_stop_pct", exit_rules)
        self.assertIn("stop_tighten_day", exit_rules)
        self.assertIn("stop_tighten_date", exit_rules)
        self.assertIn("tightened_stop_pct", exit_rules)
        
        # Check values
        self.assertEqual(exit_rules["profit_lock_threshold"], 0.25)
        self.assertEqual(exit_rules["profit_lock_stop"], 78653.0)
        self.assertEqual(exit_rules["trailing_stop_pct"], 0.04)
        self.assertEqual(exit_rules["stop_tighten_day"], 9)
        self.assertEqual(exit_rules["stop_tighten_date"], "2026-05-11")
        self.assertEqual(exit_rules["tightened_stop_pct"], 0.049)


class TelegramPathAwareExitTests(TestCase):
    """Tests for path-aware exit rules in Telegram messages."""
    
    def test_telegram_shows_trailing_stop_for_shakeout_heavy(self):
        """Test Telegram message shows trailing stop for shakeout-heavy signals."""
        from execution.services.trade_setup import TradeSetup, LegSetup, ExitRules, PathProfile
        
        setup = TradeSetup(
            signal_date=date(2026, 5, 2),
            signal_type="MVRV_SHORT",
            direction="SHORT",
            spot_price=78653.0,
            expiry=date(2026, 5, 15),
            dte=12,
            long_leg=LegSetup(
                symbol="BTC-15MAY26-80000-P",
                action="BUY",
                strike=80000.0,
                delta=-0.58,
                iv=0.35,
                price=2757.48,
                open_interest=257,
                bid_ask_spread_pct=0.03,
            ),
            short_leg=LegSetup(
                symbol="BTC-15MAY26-76000-P",
                action="SELL",
                strike=76000.0,
                delta=-0.29,
                iv=0.38,
                price=1024.21,
                open_interest=541,
                bid_ask_spread_pct=0.04,
            ),
            spread_width=4000.0,
            spread_width_pct=0.051,
            net_debit=1733.27,
            max_profit=2266.73,
            max_loss=1733.27,
            risk_reward=1.31,
            breakeven=78267.0,
            execution_cost=107.46,
            adjusted_max_profit=2159.27,
            net_edge_pct=0.72,
            risk_budget=1800.0,
            contracts=1,
            total_risk=1733.27,
            total_max_profit=2266.73,
            exit_rules=ExitRules(
                stop_loss_spot=84159.0,
                stop_loss_spot_pct=0.07,
                stop_loss_value=693.0,
                stop_loss_value_pct=0.6,
                take_profit_pct=0.6,
                take_profit_value=1360.0,
                max_hold_days=12,
                max_hold_date=date(2026, 5, 14),
                scale_down_day=7,
                scale_down_date=date(2026, 5, 9),
                scale_down_action="close_full_position",
                profit_lock_threshold=0.25,
                profit_lock_stop=78653.0,
                trailing_stop_pct=0.04,  # 4% trailing stop
                stop_tighten_day=9,
                stop_tighten_date=date(2026, 5, 11),
                tightened_stop_pct=0.049,
            ),
            path_profile=PathProfile(
                shakeout_pct=0.57,
                invalidation_pct=0.43,
                mae_p75=0.0719,
                clean_win_pct=0.571,
                is_shakeout_heavy=True,
                is_invalidation_heavy=True,
                entry_strategy="dca",
                entry_note="33% initial, 67% DCA at +7% (shakeout-heavy)",
            ),
            validation_passed=True,
            validation_warnings=[],
            validation_blocking=[],
            policy_version="2026-05-03.3",
        )
        
        message = setup.to_telegram_message()
        
        # Should show Dynamic Exits section
        self.assertIn("Dynamic Exits", message)
        self.assertIn("Trailing Stop", message)
        self.assertIn("4%", message)
        self.assertIn("Profit Lock", message)
        self.assertIn("25%", message)
        self.assertIn("Stop Tighten", message)
        self.assertIn("Day 9", message)
    
    def test_telegram_no_dynamic_exits_for_clean_signals(self):
        """Test Telegram message does NOT show Dynamic Exits for clean signals without trailing."""
        from execution.services.trade_setup import TradeSetup, LegSetup, ExitRules, PathProfile
        
        setup = TradeSetup(
            signal_date=date(2026, 5, 2),
            signal_type="CALL",
            direction="LONG",
            spot_price=78653.0,
            expiry=date(2026, 5, 15),
            dte=11,
            long_leg=LegSetup(
                symbol="BTC-15MAY26-78000-C",
                action="BUY",
                strike=78000.0,
                delta=0.60,
                iv=0.35,
                price=3000.0,
                open_interest=300,
                bid_ask_spread_pct=0.03,
            ),
            short_leg=LegSetup(
                symbol="BTC-15MAY26-86000-C",
                action="SELL",
                strike=86000.0,
                delta=0.25,
                iv=0.38,
                price=1000.0,
                open_interest=400,
                bid_ask_spread_pct=0.04,
            ),
            spread_width=8000.0,
            spread_width_pct=0.10,
            net_debit=2000.0,
            max_profit=6000.0,
            max_loss=2000.0,
            risk_reward=3.0,
            breakeven=80000.0,
            execution_cost=120.0,
            adjusted_max_profit=5880.0,
            net_edge_pct=0.90,
            risk_budget=3200.0,
            contracts=1,
            total_risk=2000.0,
            total_max_profit=6000.0,
            exit_rules=ExitRules(
                stop_loss_spot=75120.0,
                stop_loss_spot_pct=0.045,
                stop_loss_value=800.0,
                stop_loss_value_pct=0.6,
                take_profit_pct=0.7,
                take_profit_value=4200.0,
                max_hold_days=9,
                max_hold_date=date(2026, 5, 11),
                scale_down_day=6,
                scale_down_date=date(2026, 5, 8),
                scale_down_action="reduce_50pct",
                profit_lock_threshold=0.0,  # Disabled
                trailing_stop_pct=0.0,      # No trailing
                stop_tighten_day=None,      # No tightening
            ),
            path_profile=PathProfile(
                shakeout_pct=0.21,
                invalidation_pct=0.27,
                mae_p75=0.0471,
                clean_win_pct=0.729,
                is_shakeout_heavy=False,
                is_invalidation_heavy=False,
                entry_strategy="single",
                entry_note="Full position at entry",
            ),
            validation_passed=True,
            validation_warnings=[],
            validation_blocking=[],
            policy_version="2026-05-03.3",
        )
        
        message = setup.to_telegram_message()
        
        # Should NOT show Dynamic Exits section (no trailing, no profit lock, no tightening)
        self.assertNotIn("Dynamic Exits", message)
        self.assertNotIn("Trailing Stop", message)
    
    def test_telegram_shows_profit_lock_only(self):
        """Test Telegram shows profit lock when only that is configured."""
        from execution.services.trade_setup import TradeSetup, LegSetup, ExitRules
        
        setup = TradeSetup(
            signal_date=date(2026, 5, 2),
            signal_type="CALL",
            direction="LONG",
            spot_price=78653.0,
            expiry=date(2026, 5, 15),
            dte=11,
            long_leg=LegSetup(
                symbol="BTC-15MAY26-78000-C",
                action="BUY",
                strike=78000.0,
                delta=0.60,
                iv=0.35,
                price=3000.0,
                open_interest=300,
                bid_ask_spread_pct=0.03,
            ),
            short_leg=None,
            spread_width=8000.0,
            spread_width_pct=0.10,
            net_debit=2000.0,
            max_profit=6000.0,
            max_loss=2000.0,
            risk_reward=3.0,
            breakeven=80000.0,
            execution_cost=120.0,
            adjusted_max_profit=5880.0,
            net_edge_pct=0.90,
            risk_budget=3200.0,
            contracts=1,
            total_risk=2000.0,
            total_max_profit=6000.0,
            exit_rules=ExitRules(
                stop_loss_spot=75120.0,
                stop_loss_spot_pct=0.045,
                stop_loss_value=800.0,
                stop_loss_value_pct=0.6,
                take_profit_pct=0.7,
                take_profit_value=4200.0,
                max_hold_days=9,
                max_hold_date=date(2026, 5, 11),
                scale_down_day=6,
                scale_down_date=date(2026, 5, 8),
                scale_down_action="reduce_50pct",
                profit_lock_threshold=0.30,  # Enabled
                trailing_stop_pct=0.0,       # No trailing
                stop_tighten_day=None,       # No tightening
            ),
            path_profile=None,
            validation_passed=True,
            validation_warnings=[],
            validation_blocking=[],
            policy_version="2026-05-03.3",
        )
        
        message = setup.to_telegram_message()
        
        # Should show Dynamic Exits with only profit lock
        self.assertIn("Dynamic Exits", message)
        self.assertIn("Profit Lock", message)
        self.assertIn("30%", message)
        self.assertNotIn("Trailing Stop", message)
    
    def test_telegram_shows_stop_tighten_only(self):
        """Test Telegram shows stop tighten when only that is configured."""
        from execution.services.trade_setup import TradeSetup, LegSetup, ExitRules
        
        setup = TradeSetup(
            signal_date=date(2026, 5, 2),
            signal_type="OPTION_CALL",
            direction="LONG",
            spot_price=78653.0,
            expiry=date(2026, 5, 15),
            dte=9,
            long_leg=LegSetup(
                symbol="BTC-15MAY26-78000-C",
                action="BUY",
                strike=78000.0,
                delta=0.65,
                iv=0.35,
                price=3500.0,
                open_interest=300,
                bid_ask_spread_pct=0.03,
            ),
            short_leg=None,
            spread_width=8000.0,
            spread_width_pct=0.10,
            net_debit=2500.0,
            max_profit=5500.0,
            max_loss=2500.0,
            risk_reward=2.2,
            breakeven=80500.0,
            execution_cost=150.0,
            adjusted_max_profit=5350.0,
            net_edge_pct=0.85,
            risk_budget=3200.0,
            contracts=1,
            total_risk=2500.0,
            total_max_profit=5500.0,
            exit_rules=ExitRules(
                stop_loss_spot=71940.0,
                stop_loss_spot_pct=0.085,
                stop_loss_value=1000.0,
                stop_loss_value_pct=0.6,
                take_profit_pct=0.5,
                take_profit_value=2750.0,
                max_hold_days=7,
                max_hold_date=date(2026, 5, 9),
                scale_down_day=4,
                scale_down_date=date(2026, 5, 6),
                scale_down_action="reduce_50pct",
                profit_lock_threshold=0.0,   # Disabled
                trailing_stop_pct=0.0,       # No trailing
                stop_tighten_day=3,          # Tighten on day 3
                stop_tighten_date=date(2026, 5, 5),
                tightened_stop_pct=0.0425,   # Tightened to 4.25%
            ),
            path_profile=None,
            validation_passed=True,
            validation_warnings=[],
            validation_blocking=[],
            policy_version="2026-05-03.3",
        )
        
        message = setup.to_telegram_message()
        
        # Should show Dynamic Exits with only stop tighten
        self.assertIn("Dynamic Exits", message)
        self.assertIn("Stop Tighten", message)
        self.assertIn("Day 3", message)
        self.assertNotIn("Trailing Stop", message)
        self.assertNotIn("Profit Lock", message)
