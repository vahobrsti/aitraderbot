"""
Deribit execution service.

Takes an EntryPlan from DeribitEntryEngine and executes it on Deribit,
creating all the Django model records (Intent, Orders, Events) along the way.

Handles:
- Multi-leg execution (spreads, condors)
- Order sequencing (legs placed in order, with fill checks)
- Partial fill handling
- Protection registration (polling-based for options)
- Dry-run mode for paper trading

Usage:
    from execution.services.deribit_executor import DeribitExecutor
    from execution.services.deribit_entry import DeribitEntryEngine

    engine = DeribitEntryEngine()
    plan = engine.plan_entry(signal, spot_price=95000)

    executor = DeribitExecutor(account)
    result = executor.execute(plan, signal, dry_run=False)
"""
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from django.db import transaction
from django.utils import timezone as dj_timezone

from execution.models import (
    ExchangeAccount, ExecutionIntent, Order, ExecutionEvent
)
from execution.exchanges.deribit import DeribitAdapter
from execution.exchanges.base import OrderRequest
from execution.services.risk import RiskManager
from execution.services.deribit_entry import EntryPlan, LegPlan, LegType

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of executing an EntryPlan."""
    success: bool
    intent: Optional[ExecutionIntent] = None
    orders: list[Order] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    dry_run: bool = False


class DeribitExecutor:
    """
    Executes EntryPlans on Deribit.

    Lifecycle:
    1. Create ExecutionIntent from plan
    2. Run risk checks
    3. Calculate quantities per leg
    4. Place orders sequentially (sell legs after buy legs for spreads)
    5. Register for polling-based exit management
    """

    def __init__(self, account: ExchangeAccount):
        if account.exchange != "deribit":
            raise ValueError(f"DeribitExecutor requires a Deribit account, got {account.exchange}")
        self.account = account
        self.adapter = self._create_adapter()
        self.risk_manager = RiskManager(account)

    def _create_adapter(self) -> DeribitAdapter:
        import os
        api_key = os.environ.get(self.account.api_key_env, "")
        api_secret = os.environ.get(self.account.api_secret_env, "")
        if not api_key or not api_secret:
            raise ValueError(
                f"Missing credentials: {self.account.api_key_env} / {self.account.api_secret_env}"
            )
        return DeribitAdapter(api_key, api_secret, self.account.is_testnet)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(
        self,
        plan: EntryPlan,
        signal,
        dry_run: bool = False,
    ) -> ExecutionResult:
        """
        Execute an EntryPlan on Deribit.

        Args:
            plan: The entry plan from DeribitEntryEngine
            signal: DailySignal that triggered this trade
            dry_run: If True, create intent but don't place orders
        """
        result = ExecutionResult(success=False, dry_run=dry_run)
        result.warnings = list(plan.warnings)

        try:
            # 1. Create intent
            intent = self._create_intent(plan, signal)
            result.intent = intent

            # 2. Risk check
            risk_result = self.risk_manager.check_intent(intent)
            if not risk_result.passed:
                intent.status = "rejected"
                intent.status_reason = risk_result.reason
                intent.save()
                self._log_event(intent, "risk_check_failed", {"reason": risk_result.reason})
                result.errors.append(f"Risk check failed: {risk_result.reason}")
                return result

            if risk_result.adjusted_notional:
                intent.target_notional_usd = risk_result.adjusted_notional

            intent.status = "approved"
            intent.approved_at = dj_timezone.now()
            intent.save()
            self._log_event(intent, "risk_check_passed", {})

            if dry_run:
                result.success = True
                self._log_event(intent, "dry_run", {
                    "legs": [self._leg_to_dict(leg) for leg in plan.legs],
                    "total_risk_usd": plan.total_risk_usd,
                    "rationale": plan.rationale,
                })
                return result

            # 3. Calculate quantities
            qty_per_leg = self._calculate_quantities(plan, intent)

            # 4. Place orders (buy legs first, then sell legs for spreads)
            buy_legs = [l for l in plan.legs if l.side == "buy"]
            sell_legs = [l for l in plan.legs if l.side == "sell"]

            # For iron condors, place all legs
            # For spreads, buy first then sell
            ordered_legs = buy_legs + sell_legs

            all_success = True
            for leg in ordered_legs:
                qty = qty_per_leg.get(leg.symbol, Decimal("0.1"))
                order = self._place_leg_order(intent, leg, qty)
                if order:
                    result.orders.append(order)
                    if order.status in ("rejected", "cancelled"):
                        all_success = False
                        result.errors.append(
                            f"Leg {leg.leg_type.value} ({leg.symbol}) failed: {order.error_message}"
                        )
                else:
                    all_success = False
                    result.errors.append(f"Failed to place order for {leg.symbol}")

            # 5. Update intent status
            if all_success and result.orders:
                filled_count = sum(1 for o in result.orders if o.status == "filled")
                if filled_count == len(result.orders):
                    intent.status = "entry_filled"
                    intent.completed_at = dj_timezone.now()
                else:
                    intent.status = "entry_submitted"
            elif result.orders:
                intent.status = "partial"
                intent.status_reason = "; ".join(result.errors)
            else:
                intent.status = "failed"
                intent.status_reason = "; ".join(result.errors)

            intent.save()

            # 6. Register for polling exits if filled
            if intent.status == "entry_filled":
                self._register_polling_exits(intent)

            result.success = intent.status in ("entry_filled", "entry_submitted", "partial")

        except Exception as e:
            logger.exception(f"Execution error: {e}")
            result.errors.append(str(e))
            if result.intent:
                result.intent.status = "failed"
                result.intent.status_reason = str(e)
                result.intent.save()

        return result

    # ------------------------------------------------------------------
    # Intent creation
    # ------------------------------------------------------------------

    @transaction.atomic
    def _create_intent(self, plan: EntryPlan, signal) -> ExecutionIntent:
        """Create ExecutionIntent from an EntryPlan."""
        # Determine option_type from primary leg
        primary_leg = plan.legs[0] if plan.legs else None
        option_type = primary_leg.option_type if primary_leg else ""

        # For spreads, store both symbols
        spread_long = ""
        spread_short = ""
        if plan.is_spread and not plan.is_condor:
            for leg in plan.legs:
                if leg.side == "buy":
                    spread_long = leg.symbol
                elif leg.side == "sell":
                    spread_short = leg.symbol

        # Primary symbol is the first leg
        target_symbol = primary_leg.symbol if primary_leg else ""

        intent = ExecutionIntent.objects.create(
            signal=signal,
            account=self.account,
            signal_date=signal.date,
            direction=plan.direction,
            instrument_type="option",
            option_type=option_type,
            target_symbol=target_symbol,
            target_notional_usd=Decimal(str(plan.total_risk_usd)),
            strike_price=primary_leg.strike if primary_leg else None,
            expiry_date=primary_leg.expiry if primary_leg else None,
            spread_long_symbol=spread_long,
            spread_short_symbol=spread_short,
            stop_loss_pct=Decimal(str(signal.stop_loss_pct)) if signal.stop_loss_pct else None,
            take_profit_pct=Decimal(str(signal.take_profit_pct)) if signal.take_profit_pct else None,
            max_hold_days=signal.max_hold_days,
            status="pending",
        )

        self._log_event(intent, "intent_created", {
            "trade_decision": plan.trade_decision,
            "direction": plan.direction,
            "legs": [self._leg_to_dict(leg) for leg in plan.legs],
            "total_risk_usd": plan.total_risk_usd,
            "naked_pct": plan.naked_pct,
            "spread_pct": plan.spread_pct,
            "spot_price": plan.spot_price,
            "iv_rank": plan.iv_rank,
            "rationale": plan.rationale,
            "warnings": plan.warnings,
        })

        return intent

    # ------------------------------------------------------------------
    # Quantity calculation
    # ------------------------------------------------------------------

    def _calculate_quantities(
        self, plan: EntryPlan, intent: ExecutionIntent
    ) -> dict[str, Decimal]:
        """
        Calculate order quantity per leg.

        Deribit options are quoted in BTC. Quantity = number of contracts.
        For BTC options, 1 contract = 1 BTC notional.
        Min trade amount is typically 0.1 BTC.
        """
        target_notional = float(intent.target_notional_usd or plan.total_risk_usd)
        qty_map: dict[str, Decimal] = {}

        if plan.is_condor:
            # Iron condor: all legs same quantity
            # Size based on max risk (wing width)
            # Deribit BTC options: qty in BTC, min 0.1
            condor_qty = max(0.1, target_notional / plan.spot_price)
            condor_qty = round(condor_qty, 1)  # Round to 0.1
            for leg in plan.legs:
                qty_map[leg.symbol] = Decimal(str(condor_qty))
        elif plan.is_spread:
            # Spread: both legs same quantity
            # Estimate cost from mid prices
            buy_leg = next((l for l in plan.legs if l.side == "buy"), None)
            sell_leg = next((l for l in plan.legs if l.side == "sell"), None)

            if buy_leg and buy_leg.mid_price and sell_leg and sell_leg.mid_price:
                net_debit_btc = float(buy_leg.mid_price - sell_leg.mid_price)
                if net_debit_btc > 0:
                    # Debit spread: size by debit cost
                    debit_usd = net_debit_btc * plan.spot_price
                    qty = target_notional / max(debit_usd, 1.0)
                else:
                    # Credit spread: size by max loss (width - credit)
                    qty = target_notional / plan.spot_price
            else:
                # Fallback: size by notional / spot
                qty = target_notional / plan.spot_price

            qty = max(0.1, round(qty, 1))
            for leg in plan.legs:
                qty_map[leg.symbol] = Decimal(str(qty))
        else:
            # Single leg
            leg = plan.legs[0]
            if leg.mid_price and float(leg.mid_price) > 0:
                premium_usd = float(leg.mid_price) * plan.spot_price
                qty = target_notional / max(premium_usd, 1.0)
            else:
                qty = target_notional / plan.spot_price

            qty = max(0.1, round(qty, 1))
            qty_map[leg.symbol] = Decimal(str(qty))

        return qty_map

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    def _place_leg_order(
        self,
        intent: ExecutionIntent,
        leg: LegPlan,
        qty: Decimal,
    ) -> Optional[Order]:
        """Place a single leg order on Deribit."""
        order = Order.objects.create(
            intent=intent,
            symbol=leg.symbol,
            side=leg.side,
            order_type="market",
            qty=qty,
            status="pending",
        )

        request = OrderRequest(
            symbol=leg.symbol,
            side=leg.side,
            order_type="market",
            qty=qty,
            client_order_id=order.client_order_id,
        )

        logger.info(
            f"Placing {leg.leg_type.value} order: {leg.side} {qty} {leg.symbol} "
            f"(delta={leg.delta}, IV={leg.iv})"
        )

        response = self.adapter.place_order(request)

        if response.success:
            order.exchange_order_id = response.exchange_order_id or ""
            order.status = response.status or "submitted"
            order.filled_qty = response.filled_qty or Decimal("0")
            order.avg_fill_price = response.avg_price
            order.submitted_at = dj_timezone.now()
            order.save()

            self._log_event(intent, "order_submitted", {
                "order_id": str(order.id),
                "exchange_order_id": response.exchange_order_id,
                "leg_type": leg.leg_type.value,
                "symbol": leg.symbol,
                "side": leg.side,
                "qty": str(qty),
                "delta": leg.delta,
                "iv": leg.iv,
                "mid_price": str(leg.mid_price) if leg.mid_price else None,
            }, order=order)

            # Poll for fill status
            self._sync_order(order)
            return order
        else:
            order.status = "rejected"
            order.error_code = response.error_code or ""
            order.error_message = response.error_message or ""
            order.save()

            self._log_event(intent, "order_rejected", {
                "leg_type": leg.leg_type.value,
                "symbol": leg.symbol,
                "error_code": response.error_code,
                "error_message": response.error_message,
            }, order=order)

            return order

    def _sync_order(self, order: Order) -> None:
        """Poll Deribit for order fill status."""
        if not order.exchange_order_id:
            return

        response = self.adapter.get_order(order.symbol, order.exchange_order_id)
        if response.success:
            order.status = response.status or order.status
            order.filled_qty = response.filled_qty or order.filled_qty
            order.avg_fill_price = response.avg_price or order.avg_fill_price
            order.save()

    # ------------------------------------------------------------------
    # Exit registration
    # ------------------------------------------------------------------

    def _register_polling_exits(self, intent: ExecutionIntent) -> None:
        """Register intent for polling-based exit management."""
        has_exit_params = (
            intent.stop_loss_pct
            or intent.take_profit_pct
            or intent.max_hold_days
        )

        if has_exit_params:
            intent.status = "protected"
            intent.exit_method = "polling"
            intent.protected_at = dj_timezone.now()
        else:
            intent.status = "unprotected"
            intent.exit_method = "none"
            logger.warning(f"Intent {intent.id} has no exit parameters")

        intent.save()

        self._log_event(intent, "polling_exit_registered", {
            "stop_loss_pct": str(intent.stop_loss_pct) if intent.stop_loss_pct else None,
            "take_profit_pct": str(intent.take_profit_pct) if intent.take_profit_pct else None,
            "max_hold_days": intent.max_hold_days,
            "exit_method": intent.exit_method,
        })

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log_event(
        self,
        intent: ExecutionIntent,
        event_type: str,
        payload: dict,
        order: Optional[Order] = None,
    ) -> ExecutionEvent:
        return ExecutionEvent.objects.create(
            intent=intent,
            order=order,
            event_type=event_type,
            payload=payload,
        )

    @staticmethod
    def _leg_to_dict(leg: LegPlan) -> dict:
        return {
            "symbol": leg.symbol,
            "side": leg.side,
            "option_type": leg.option_type,
            "strike": str(leg.strike),
            "expiry": str(leg.expiry),
            "leg_type": leg.leg_type.value,
            "delta": leg.delta,
            "iv": leg.iv,
            "mid_price": str(leg.mid_price) if leg.mid_price else None,
            "bid": str(leg.bid) if leg.bid else None,
            "ask": str(leg.ask) if leg.ask else None,
            "spread_pct": leg.spread_pct,
            "open_interest": str(leg.open_interest) if leg.open_interest else None,
            "dte": leg.dte,
        }
