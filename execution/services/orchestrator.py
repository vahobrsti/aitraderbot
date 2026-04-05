"""
Execution orchestrator service.
Handles the full lifecycle: Signal -> Intent -> Entry -> Protection -> Exit.

V1 Scope:
- Single-leg options only (CALL/PUT/OPTION_CALL/OPTION_PUT)
- Entry order -> SL/TP placement -> Polling fallback
- No spreads
"""
import logging
import os
from decimal import Decimal
from typing import Optional
from datetime import date, timedelta
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
    
    Lifecycle:
    1. create_intent_from_signal() - Parse signal into intent
    2. process_intent() - Risk check -> Entry order -> Wait for fill
    3. protect_position() - Place SL/TP or register for polling
    4. sync_positions() / reconcile() - Keep state in sync
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
        V1: Single-leg options only.
        """
        decision = signal.trade_decision.upper()
        
        # V1: Only single-leg options
        decision_map = {
            'CALL': ('long', 'call'),
            'OPTION_CALL': ('long', 'call'),
            'PUT': ('short', 'put'),
            'OPTION_PUT': ('short', 'put'),
            'TACTICAL_PUT': ('long', 'put'),
        }
        
        if decision not in decision_map:
            raise ValueError(f"Cannot execute signal with decision: {signal.trade_decision}")
        
        direction, option_type = decision_map[decision]
        
        # Calculate target notional
        base_notional = self.account.max_position_usd * Decimal('0.5')
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
            'option_type': option_type,
            'target_notional': str(target_notional),
        })
        
        return intent
    
    @transaction.atomic
    def process_intent(self, intent: ExecutionIntent) -> bool:
        """
        Process intent through: Risk Check -> Entry Order -> Fill.
        After this, call protect_position() to place SL/TP.
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
            
            if risk_result.adjusted_notional:
                intent.target_notional_usd = risk_result.adjusted_notional
            
            intent.status = 'approved'
            intent.approved_at = timezone.now()
            intent.save()
            self._log_event(intent, 'risk_check_passed', {})
            
            # Step 2: Select instrument (V1: simple selection)
            symbol = self._select_instrument(intent)
            if not symbol:
                intent.status = 'failed'
                intent.status_reason = 'No suitable instrument found'
                intent.save()
                return False
            
            intent.target_symbol = symbol
            intent.save()
            
            # Step 3: Get instrument info and calculate qty
            instrument = self.adapter.get_instrument(symbol)
            if not instrument:
                intent.status = 'failed'
                intent.status_reason = f'Instrument not found: {symbol}'
                intent.save()
                return False
            
            qty = self._calculate_qty(intent, instrument)
            intent.target_qty = qty
            intent.save()
            
            # Step 4: Place entry order
            order = self._place_entry_order(intent, symbol, qty)
            if not order:
                intent.status = 'failed'
                intent.status_reason = 'Entry order placement failed'
                intent.save()
                return False
            
            intent.status = 'entry_submitted'
            intent.save()
            
            # Step 5: Sync order status (poll for fill)
            self._sync_order_status(order)
            
            if order.status == 'filled':
                intent.status = 'entry_filled'
                intent.completed_at = timezone.now()
                intent.save()
                self._log_event(intent, 'entry_filled', {
                    'symbol': symbol,
                    'qty': str(qty),
                    'avg_price': str(order.avg_fill_price),
                })
                return True
            elif order.status in ('open', 'partial', 'submitted', 'pending'):
                # Not filled yet - will be updated by reconciliation
                self._log_event(intent, 'entry_pending', {'order_status': order.status})
                return True
            else:
                intent.status = 'failed'
                intent.status_reason = f'Entry order {order.status}'
                intent.save()
                return False
                
        except Exception as e:
            logger.exception(f"Error processing intent {intent.id}: {e}")
            intent.status = 'failed'
            intent.status_reason = str(e)
            intent.save()
            self._log_event(intent, 'error', {'error': str(e)})
            return False
    
    def protect_position(self, intent: ExecutionIntent) -> bool:
        """
        Register position for polling-based exit management.
        Options don't support native SL/TP - manage_exits handles all exits.
        
        Returns True if registered successfully.
        """
        if intent.status != 'entry_filled':
            logger.warning(f"Cannot protect intent {intent.id} in status {intent.status}")
            return False
        
        # Check if we have exit parameters
        has_exit_params = (
            intent.stop_loss_pct or 
            intent.take_profit_pct or 
            intent.max_hold_days
        )
        
        if not has_exit_params:
            logger.warning(f"Intent {intent.id} has no exit parameters")
            intent.status = 'unprotected'
            intent.exit_method = 'none'
            intent.save()
            self._log_event(intent, 'no_exit_params', {
                'warning': 'Position has no SL/TP/time stop defined'
            })
            return False
        
        # Register for polling-based exits
        intent.status = 'protected'
        intent.exit_method = 'polling'
        intent.protected_at = timezone.now()
        intent.save()
        
        self._log_event(intent, 'polling_exit_registered', {
            'stop_loss_pct': str(intent.stop_loss_pct) if intent.stop_loss_pct else None,
            'take_profit_pct': str(intent.take_profit_pct) if intent.take_profit_pct else None,
            'max_hold_days': intent.max_hold_days,
            'note': 'Options use polling-based exits via manage_exits command',
        })
        
        logger.info(
            f"Intent {intent.id} registered for polling exits "
            f"(SL={intent.stop_loss_pct}, TP={intent.take_profit_pct}, "
            f"max_days={intent.max_hold_days})"
        )
        return True
    
    def _select_instrument(self, intent: ExecutionIntent) -> Optional[str]:
        """
        Select option instrument based on signal guidance.
        Options only - no perpetuals.
        """
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
        
        # Parse DTE from signal
        target_dte = 45
        if intent.signal and intent.signal.dte_range:
            try:
                dte_str = intent.signal.dte_range.replace('d', '').split('-')
                target_dte = int(dte_str[0])
            except (ValueError, IndexError):
                pass
        
        # Find closest to target DTE
        target_expiry = date.today() + timedelta(days=target_dte)
        options_with_expiry = [o for o in options if o.expiry]
        
        if not options_with_expiry:
            return options[0].symbol if options else None
        
        best = min(
            options_with_expiry,
            key=lambda o: abs((o.expiry.date() if hasattr(o.expiry, 'date') else o.expiry) - target_expiry)
        )
        
        return best.symbol
    
    def _calculate_qty(self, intent: ExecutionIntent, instrument) -> Decimal:
        """Calculate order quantity based on notional and instrument."""
        target_notional = intent.target_notional_usd or Decimal('1000')
        lot_size = instrument.lot_size or Decimal('0.1')
        min_qty = instrument.min_qty or Decimal('0.1')
        
        # Try to get mark price
        mark_price = None
        try:
            if self.account.exchange == 'bybit' and intent.target_symbol:
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
            qty = target_notional / mark_price
        else:
            # Fallback: estimate 5% premium
            estimated_premium = target_notional * Decimal('0.05')
            qty = target_notional / max(estimated_premium, Decimal('100'))
        
        qty = (qty / lot_size).quantize(Decimal('1')) * lot_size
        qty = max(qty, min_qty)
        
        return qty
    
    def _place_entry_order(self, intent: ExecutionIntent, symbol: str, qty: Decimal) -> Optional[Order]:
        """Place entry order on exchange."""
        side = 'buy' if intent.direction == 'long' else 'sell'
        
        order = Order.objects.create(
            intent=intent,
            symbol=symbol,
            side=side,
            order_type='market',
            qty=qty,
            status='pending',
        )
        
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
            
            self._log_event(intent, 'entry_order_submitted', {
                'order_id': str(order.id),
                'exchange_order_id': response.exchange_order_id,
                'symbol': symbol,
                'side': side,
                'qty': str(qty),
            }, order=order)
            
            return order
        else:
            order.status = 'rejected'
            order.error_code = response.error_code
            order.error_message = response.error_message
            order.save()
            
            self._log_event(intent, 'entry_order_rejected', {
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
    
    def sync_positions(self) -> list[Position]:
        """Sync all positions from exchange to database."""
        sync_result = self.adapter.get_positions()
        synced = []
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
        
        # Only zero out if API succeeded
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
                f"Skipping position zero-out for {self.account.name}: {sync_result.error}"
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
        Reconciliation job: sync exchange state, update intent statuses.
        """
        summary = {
            'positions_synced': 0,
            'orders_updated': 0,
            'intents_updated': 0,
            'unprotected_alerts': 0,
        }
        
        # Sync positions
        positions = self.sync_positions()
        summary['positions_synced'] = len(positions)
        
        # Check open orders
        open_orders = Order.objects.filter(
            intent__account=self.account,
            status__in=['submitted', 'open', 'partial', 'pending'],
        )
        
        for order in open_orders:
            self._sync_order_status(order)
            summary['orders_updated'] += 1
            
            # Update intent if order filled
            if order.status == 'filled' and order.intent.status == 'entry_submitted':
                order.intent.status = 'entry_filled'
                order.intent.completed_at = timezone.now()
                order.intent.save()
                summary['intents_updated'] += 1
        
        # Check for unprotected positions (alert condition)
        unprotected = ExecutionIntent.objects.filter(
            account=self.account,
            status__in=['entry_filled', 'unprotected'],
            protected_at__isnull=True,
        )
        summary['unprotected_alerts'] = unprotected.count()
        
        if summary['unprotected_alerts'] > 0:
            logger.warning(
                f"ALERT: {summary['unprotected_alerts']} unprotected position(s) "
                f"for account {self.account.name}"
            )
        
        ExecutionEvent.objects.create(
            event_type='reconciliation',
            payload=summary,
        )
        
        return summary
