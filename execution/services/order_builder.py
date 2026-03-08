"""
Order builder service.
Constructs single-leg and spread orders from signal guidance.
"""
import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from execution.exchanges.base import OrderRequest
from execution.services.instrument_selector import StrikeSelection, SpreadSelection

logger = logging.getLogger(__name__)


@dataclass
class OrderPlan:
    """A planned order (not yet submitted)."""
    symbol: str
    side: str  # 'buy' or 'sell'
    qty: Decimal
    order_type: str
    price: Optional[Decimal] = None
    reduce_only: bool = False
    leg_label: str = ''  # 'long_leg', 'short_leg', 'single'


@dataclass
class SpreadOrderPlan:
    """Plan for a spread (two legs)."""
    long_leg: OrderPlan
    short_leg: OrderPlan
    spread_type: str
    net_debit: Optional[Decimal] = None  # Estimated cost


class OrderBuilder:
    """
    Builds order plans from instrument selections.
    Handles: single options, vertical spreads, position sizing.
    """
    
    def __init__(self, account_max_usd: Decimal):
        self.account_max_usd = account_max_usd
    
    def build_single_option_order(
        self,
        selection: StrikeSelection,
        direction: str,
        target_notional: Decimal,
        mark_price: Optional[Decimal] = None,
        lot_size: Decimal = Decimal('0.01'),
        min_qty: Decimal = Decimal('0.01'),
    ) -> OrderPlan:
        """
        Build order for a single option.
        
        Args:
            selection: Selected instrument
            direction: 'long' or 'short'
            target_notional: Target USD exposure
            mark_price: Current option price (for sizing)
            lot_size: Minimum qty increment
            min_qty: Minimum order qty
        """
        side = 'buy' if direction == 'long' else 'sell'
        
        # Calculate quantity
        qty = self._calculate_qty(target_notional, mark_price, lot_size, min_qty)
        
        return OrderPlan(
            symbol=selection.symbol,
            side=side,
            qty=qty,
            order_type='market',
            leg_label='single',
        )
    
    def build_spread_orders(
        self,
        spread: SpreadSelection,
        target_notional: Decimal,
        long_mark: Optional[Decimal] = None,
        short_mark: Optional[Decimal] = None,
        lot_size: Decimal = Decimal('0.01'),
        min_qty: Decimal = Decimal('0.01'),
    ) -> SpreadOrderPlan:
        """
        Build orders for a vertical spread.
        
        For call spread (bullish): buy lower strike, sell higher strike
        For put spread (bearish): buy higher strike, sell lower strike
        """
        # Calculate quantity based on spread cost
        net_debit = None
        if long_mark and short_mark:
            net_debit = long_mark - short_mark
            if net_debit > 0:
                qty = self._calculate_qty(target_notional, net_debit, lot_size, min_qty)
            else:
                # Credit spread - size based on max loss (width - credit)
                qty = self._calculate_qty(target_notional, abs(net_debit), lot_size, min_qty)
        else:
            # Fallback sizing
            qty = self._calculate_qty(target_notional, None, lot_size, min_qty)
        
        long_order = OrderPlan(
            symbol=spread.long_leg.symbol,
            side='buy',
            qty=qty,
            order_type='market',
            leg_label='long_leg',
        )
        
        short_order = OrderPlan(
            symbol=spread.short_leg.symbol,
            side='sell',
            qty=qty,
            order_type='market',
            leg_label='short_leg',
        )
        
        return SpreadOrderPlan(
            long_leg=long_order,
            short_leg=short_order,
            spread_type=spread.spread_type,
            net_debit=net_debit,
        )
    
    def build_stop_loss_order(
        self,
        symbol: str,
        position_qty: Decimal,
        position_side: str,
        entry_price: Decimal,
        stop_loss_pct: Decimal,
    ) -> OrderPlan:
        """
        Build a stop-loss order for an existing position.
        
        Args:
            stop_loss_pct: e.g., 0.04 = 4% adverse move triggers exit
        """
        # For long position, stop is below entry
        # For short position, stop is above entry
        if position_side == 'long':
            trigger_price = entry_price * (1 - stop_loss_pct)
            close_side = 'sell'
        else:
            trigger_price = entry_price * (1 + stop_loss_pct)
            close_side = 'buy'
        
        return OrderPlan(
            symbol=symbol,
            side=close_side,
            qty=position_qty,
            order_type='stop_market',
            price=trigger_price,
            reduce_only=True,
            leg_label='stop_loss',
        )
    
    def build_take_profit_order(
        self,
        symbol: str,
        position_qty: Decimal,
        position_side: str,
        entry_price: Decimal,
        take_profit_pct: Decimal,
    ) -> OrderPlan:
        """
        Build a take-profit order for an existing position.
        
        Args:
            take_profit_pct: e.g., 0.70 = 70% gain triggers exit
        """
        if position_side == 'long':
            trigger_price = entry_price * (1 + take_profit_pct)
            close_side = 'sell'
        else:
            trigger_price = entry_price * (1 - take_profit_pct)
            close_side = 'buy'
        
        return OrderPlan(
            symbol=symbol,
            side=close_side,
            qty=position_qty,
            order_type='take_profit',
            price=trigger_price,
            reduce_only=True,
            leg_label='take_profit',
        )
    
    def build_scale_down_order(
        self,
        symbol: str,
        position_qty: Decimal,
        position_side: str,
        scale_pct: Decimal = Decimal('0.75'),
    ) -> OrderPlan:
        """
        Build order to scale down position (e.g., reduce to 25% = close 75%).
        """
        close_qty = position_qty * scale_pct
        close_side = 'sell' if position_side == 'long' else 'buy'
        
        return OrderPlan(
            symbol=symbol,
            side=close_side,
            qty=close_qty,
            order_type='market',
            reduce_only=True,
            leg_label='scale_down',
        )
    
    def build_full_close_order(
        self,
        symbol: str,
        position_qty: Decimal,
        position_side: str,
    ) -> OrderPlan:
        """Build order to fully close a position."""
        close_side = 'sell' if position_side == 'long' else 'buy'
        
        return OrderPlan(
            symbol=symbol,
            side=close_side,
            qty=position_qty,
            order_type='market',
            reduce_only=True,
            leg_label='full_close',
        )
    
    def to_order_request(self, plan: OrderPlan, client_order_id: str) -> OrderRequest:
        """Convert OrderPlan to exchange OrderRequest."""
        return OrderRequest(
            symbol=plan.symbol,
            side=plan.side,
            order_type=plan.order_type,
            qty=plan.qty,
            price=plan.price,
            trigger_price=plan.price if plan.order_type in ('stop_market', 'take_profit') else None,
            reduce_only=plan.reduce_only,
            client_order_id=client_order_id,
        )
    
    def _calculate_qty(
        self,
        target_notional: Decimal,
        price: Optional[Decimal],
        lot_size: Decimal,
        min_qty: Decimal,
    ) -> Decimal:
        """Calculate order quantity from notional and price."""
        if price and price > 0:
            qty = target_notional / price
        else:
            # Fallback: assume ~5% premium for ATM options
            estimated_premium = target_notional * Decimal('0.05')
            qty = target_notional / max(estimated_premium, Decimal('100'))
        
        # Round to lot size
        qty = (qty / lot_size).quantize(Decimal('1')) * lot_size
        qty = max(qty, min_qty)
        
        return qty
