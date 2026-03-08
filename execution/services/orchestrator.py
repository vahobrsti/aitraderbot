"""
Execution orchestrator service.
Handles the full lifecycle: DailySignal -> ExecutionIntent -> Risk checks -> Exchange -> Persist.
"""
import logging
import os
from decimal import Decimal
from typing import Optional
from datetime import date
from django.db import transaction
from django.utils import timezone

from execution.models import (
    ExchangeAccount, ExecutionIntent, Order, Fill, Position, ExecutionEvent
)
from execution.exchanges import BybitAdapter, DeribitAdapter
from execution.exchanges.base import OrderRequest, ExchangeAdapter
from .risk import RiskManager, RiskCheckResult

logger = logging.getLogger(__name__)


class ExecutionOrchestrator:
    """
    Main orchestration service for trade execution.
    Coordinates: signal interpretation, risk checks, order placement, status tracking.
    """
    
    def __init__(self, account: ExchangeAccount):
        self.account = account
        self.adapter = self._create_adapter()
        self.risk_manager = RiskManager(account)
    
    def _create_adapter(self) -> ExchangeAdapter:
        """Create appropriate exchange adapter based on account."""
        api_key = os.environ.get(self.account.api_key_env, '')
        api_secret = os.environ.get(self.account.api_secret_env, '')
        
        if not api_key or not api_secret:
            raise ValueError(
                f"Missing credentials: {self.account.api_key_env} or {self.account.api_secret_env}"
            )
        
        if self.account.exchange == 'bybit':
            return BybitAdapter(api_key, api_secret, self.account.is_testnet)
        elif self.account.exchange == 'deribit':
            return DeribitAdapter(api_key, api_secret, self.account.is_testnet)
        else:
            raise ValueError(f"Unsupported exchange: {self.account.exchange}")
    
    @transaction.atomic
    def create_intent_from_signal(self, signal) -> ExecutionIntent:
        """
        Create an ExecutionIntent from a DailySignal.
        Interprets signal fields into execution parameters.
        """
        # Determine direction from trade_decision
        decision = signal.trade_decision.upper()
        
        # Map all valid trade decisions
        decision_map = {
            'CALL': ('long', 'call'),
            'OPTION_CALL': ('long', 'call'),
            'PUT': ('short', 'put'),
            'OPTION_PUT': ('short', 'put'),
            'TACTICAL_PUT': ('long', 'put'),  # Tactical put is a hedge, direction depends on context
        }
        
        if decision not in decision_map:
            raise ValueError(f"Cannot execute signal with decision: {signal.trade_decision}")
        
        direction, option_type = decision_map[decision]
        
        # Calculate target notional from size multiplier
        base_notional = self.account.max_position_usd * Decimal('0.5')  # 50% of max as base
        size_mult = Decimal(str(signal.size_multiplier or 1.0))
        target_notional = base_notional * size_mult
        
        intent = ExecutionIntent.objects.create(
            signal=signal,
            account=self.account,
            signal_date=signal.date,
            direction=direction,
            instrument_type='option',
            option_type=option_type,
            target_notional_usd=target_notional,
            stop_loss_pct=Decimal(str(signal.stop_loss_pct)) if signal.stop_loss_pct else None,
            take_profit_pct=Decimal(str(signal.take_profit_pct)) if signal.take_profit_pct else None,
            max_hold_days=signal.max_hold_days,
            status='pending',
        )
        
        self._log_event(intent, 'intent_created', {
            'signal_date': str(signal.date),
            'trade_decision': signal.trade_decision,
            'direction': direction,
            'target_notional': str(target_notional),
        })
        
        return intent
    
    @transaction.atomic
    def process_intent(self, intent: ExecutionIntent) -> bool:
        """
        Process an execution intent through the full lifecycle.
        Returns True if execution was successful.
        """
        try:
            # Step 1: Risk checks
            intent.status = 'risk_check'
            intent.save()
            
            risk_result = self.risk_manager.check_intent(intent)
            
            if not risk_result.passed:
                intent.status = 'rejected'
                intent.status_reason = risk_result.reason
                intent.save()
                self._log_event(intent, 'risk_check_failed', {'reason': risk_result.reason})
                return False
            
            # Apply any adjustments from risk check
            if risk_result.adjusted_notional:
                intent.target_notional_usd = risk_result.adjusted_notional
            
            intent.status = 'approved'
            intent.approved_at = timezone.now()
            intent.save()
            self._log_event(intent, 'risk_check_passed', {})
            
            # Step 2: Find instrument
            symbol = self._select_instrument(intent)
            if not symbol:
                intent.status = 'failed'
                intent.status_reason = 'No suitable instrument found'
                intent.save()
                return False
            
            intent.target_symbol = symbol
            intent.status = 'executing'
            intent.save()
            
            # Step 3: Calculate quantity
            instrument = self.adapter.get_instrument(symbol)
            if not instrument:
                intent.status = 'failed'
                intent.status_reason = f'Instrument not found: {symbol}'
                intent.save()
                return False
            
            # Get current price for sizing
            # For options, use mark price or theoretical value
            qty = self._calculate_qty(intent, instrument)
            intent.target_qty = qty
            intent.save()
            
            # Step 4: Place order
            order = self._place_order(intent, symbol, qty)
            if not order:
                intent.status = 'failed'
                intent.status_reason = 'Order placement failed'
                intent.save()
                return False
            
            # Step 5: Update intent status based on order
            if order.status == 'filled':
                intent.status = 'filled'
                intent.completed_at = timezone.now()
            elif order.status in ('open', 'partial', 'submitted'):
                # Order accepted but not yet filled - this is success, not failure
                intent.status = 'executing' if order.status != 'partial' else 'partial'
            elif order.status in ('rejected', 'cancelled', 'expired'):
                intent.status = 'failed'
                intent.status_reason = f'Order {order.status}: {order.error_message}'
            # For 'pending' status, leave intent as 'executing' - will be updated by reconciliation
            
            intent.save()
            return intent.status not in ('failed', 'rejected')
            
        except Exception as e:
            logger.exception(f"Error processing intent {intent.id}: {e}")
            intent.status = 'failed'
            intent.status_reason = str(e)
            intent.save()
            self._log_event(intent, 'order_error', {'error': str(e)})
            return False
    
    def _select_instrument(self, intent: ExecutionIntent) -> Optional[str]:
        """
        Select the best instrument for the intent.
        For options: find appropriate strike and expiry.
        """
        if intent.instrument_type != 'option':
            # For perpetuals, use standard symbol
            return self.adapter.normalize_symbol('BTC-USDT-PERP')
        
        # Get available options
        instruments = self.adapter.get_instruments(
            instrument_type='option',
            underlying='BTC'
        )
        
        if not instruments:
            logger.warning("No option instruments available")
            return None
        
        # Filter by option type
        target_type = intent.option_type or ('call' if intent.direction == 'long' else 'put')
        options = [i for i in instruments if i.option_type == target_type]
        
        if not options:
            logger.warning(f"No {target_type} options available")
            return None
        
        # Select based on DTE and strike guidance from signal
        # Default: ATM option with 30-60 DTE
        target_dte = 45
        if intent.signal and intent.signal.dte_range:
            # Parse DTE range like "45-90d"
            try:
                dte_str = intent.signal.dte_range.replace('d', '').split('-')
                target_dte = int(dte_str[0])
            except (ValueError, IndexError):
                pass
        
        # Sort by expiry and find closest to target DTE
        from datetime import timedelta
        target_expiry = date.today() + timedelta(days=target_dte)
        
        options_with_expiry = [o for o in options if o.expiry]
        if not options_with_expiry:
            return options[0].symbol if options else None
        
        # Find closest expiry
        best_option = min(
            options_with_expiry,
            key=lambda o: abs((o.expiry.date() if hasattr(o.expiry, 'date') else o.expiry) - target_expiry)
        )
        
        return best_option.symbol
    
    def _calculate_qty(self, intent: ExecutionIntent, instrument) -> Decimal:
        """Calculate order quantity based on notional and instrument specs."""
        target_notional = intent.target_notional_usd or Decimal('1000')
        
        # Use lot size as minimum increment
        lot_size = instrument.lot_size or Decimal('0.1')
        min_qty = instrument.min_qty or Decimal('0.1')
        
        # Try to get actual mark price for better sizing
        mark_price = None
        try:
            # For options, get ticker/mark price
            if hasattr(self.adapter, 'session') and intent.target_symbol:
                if self.account.exchange == 'bybit':
                    ticker = self.adapter.session.get_tickers(
                        category='option',
                        symbol=intent.target_symbol
                    )
                    if ticker.get('retCode') == 0:
                        tickers = ticker.get('result', {}).get('list', [])
                        if tickers:
                            mark_price = Decimal(tickers[0].get('markPrice', '0'))
        except Exception as e:
            logger.warning(f"Could not fetch mark price: {e}")
        
        if mark_price and mark_price > 0:
            # Use actual mark price
            qty = target_notional / mark_price
        else:
            # Fallback: estimate based on underlying price and typical premium
            # Options typically trade at 1-10% of underlying for ATM
            # Use conservative 3% estimate
            estimated_premium = target_notional * Decimal('0.03')
            qty = target_notional / max(estimated_premium, Decimal('100'))
        
        # Round to lot size
        qty = (qty / lot_size).quantize(Decimal('1')) * lot_size
        qty = max(qty, min_qty)
        
        return qty
    
    def _place_order(self, intent: ExecutionIntent, symbol: str, qty: Decimal) -> Optional[Order]:
        """Place order on exchange and persist to database."""
        side = 'buy' if intent.direction == 'long' else 'sell'
        
        # Create order record first
        order = Order.objects.create(
            intent=intent,
            symbol=symbol,
            side=side,
            order_type='market',  # Start with market orders for simplicity
            qty=qty,
            status='pending',
        )
        
        # Place on exchange
        request = OrderRequest(
            symbol=symbol,
            side=side,
            order_type='market',
            qty=qty,
            client_order_id=order.client_order_id,
        )
        
        response = self.adapter.place_order(request)
        
        if response.success:
            order.exchange_order_id = response.exchange_order_id
            order.status = 'submitted'
            order.submitted_at = timezone.now()
            order.save()
            
            self._log_event(intent, 'order_submitted', {
                'order_id': str(order.id),
                'exchange_order_id': response.exchange_order_id,
                'symbol': symbol,
                'side': side,
                'qty': str(qty),
            }, order=order)
            
            # Poll for fill status
            self._sync_order_status(order)
            
            return order
        else:
            order.status = 'rejected'
            order.error_code = response.error_code
            order.error_message = response.error_message
            order.save()
            
            self._log_event(intent, 'order_rejected', {
                'error_code': response.error_code,
                'error_message': response.error_message,
            }, order=order)
            
            return None
    
    def _sync_order_status(self, order: Order) -> None:
        """Sync order status from exchange."""
        response = self.adapter.get_order(order.symbol, order.exchange_order_id)
        
        if response.success:
            order.status = response.status
            order.filled_qty = response.filled_qty
            order.avg_fill_price = response.avg_price
            order.save()
            
            if response.status == 'filled':
                self._log_event(order.intent, 'order_filled', {
                    'order_id': str(order.id),
                    'filled_qty': str(response.filled_qty),
                    'avg_price': str(response.avg_price),
                }, order=order)
    
    def sync_positions(self) -> list[Position]:
        """Sync all positions from exchange to database."""
        sync_result = self.adapter.get_positions()
        synced = []
        
        # Track which symbols we received from exchange
        active_symbols = set()
        
        for pos_info in sync_result.positions:
            active_symbols.add(pos_info.symbol)
            position, created = Position.objects.update_or_create(
                account=self.account,
                symbol=pos_info.symbol,
                defaults={
                    'side': pos_info.side,
                    'qty': pos_info.qty,
                    'entry_price': pos_info.entry_price,
                    'mark_price': pos_info.mark_price,
                    'liquidation_price': pos_info.liquidation_price,
                    'unrealized_pnl': pos_info.unrealized_pnl,
                    'realized_pnl': pos_info.realized_pnl,
                    'leverage': pos_info.leverage,
                    'option_type': pos_info.option_type or '',
                    'strike': pos_info.strike,
                    'expiry': pos_info.expiry.date() if pos_info.expiry else None,
                    'synced_at': timezone.now(),
                }
            )
            synced.append(position)
        
        # Only zero out positions if API call succeeded
        # This prevents wiping positions on transient API failures
        if sync_result.success:
            Position.objects.filter(
                account=self.account,
                qty__gt=0,
            ).exclude(
                symbol__in=active_symbols
            ).update(
                qty=Decimal('0'),
                side='none',
                unrealized_pnl=Decimal('0'),
                synced_at=timezone.now(),
            )
        else:
            logger.warning(
                f"Skipping position zero-out for {self.account.name} due to API error: {sync_result.error}"
            )
        
        return synced
    
    def _log_event(
        self, 
        intent: ExecutionIntent, 
        event_type: str, 
        payload: dict,
        order: Optional[Order] = None
    ) -> ExecutionEvent:
        """Log an execution event."""
        return ExecutionEvent.objects.create(
            intent=intent,
            order=order,
            event_type=event_type,
            payload=payload,
        )
    
    def reconcile(self) -> dict:
        """
        Reconciliation job: sync exchange state with database.
        Returns summary of changes.
        """
        summary = {
            'positions_synced': 0,
            'orders_updated': 0,
            'discrepancies': [],
        }
        
        # Sync positions
        positions = self.sync_positions()
        summary['positions_synced'] = len(positions)
        
        # Check open orders
        open_orders = Order.objects.filter(
            intent__account=self.account,
            status__in=['submitted', 'open', 'partial'],
        )
        
        for order in open_orders:
            self._sync_order_status(order)
            summary['orders_updated'] += 1
        
        # Log reconciliation event
        ExecutionEvent.objects.create(
            event_type='reconciliation',
            payload=summary,
        )
        
        return summary
