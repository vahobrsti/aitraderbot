"""
Deribit API adapter.
Handles all Deribit-specific quirks: symbol formats, precision, order types, auth.
Uses direct HTTP requests (no official SDK).
"""
import hashlib
import hmac
import json
import logging
import time
from decimal import Decimal
from typing import Optional
from datetime import datetime
import requests

from .base import (
    ExchangeAdapter, OrderRequest, OrderResponse, 
    PositionInfo, PositionSyncResult, InstrumentInfo, AccountBalance
)

logger = logging.getLogger(__name__)


class DeribitAdapter(ExchangeAdapter):
    """
    Deribit API adapter.
    Supports: BTC/ETH Perpetuals, Futures, Options.
    """
    
    MAINNET_URL = 'https://www.deribit.com'
    TESTNET_URL = 'https://test.deribit.com'
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        super().__init__(api_key, api_secret, testnet)
        self.base_url = self.TESTNET_URL if testnet else self.MAINNET_URL
        self._access_token = None
        self._token_expiry = 0
    
    @property
    def exchange_name(self) -> str:
        return 'deribit'
    
    def _get_access_token(self) -> str:
        """Get or refresh access token."""
        if self._access_token and time.time() < self._token_expiry - 60:
            return self._access_token
        
        response = requests.get(
            f"{self.base_url}/api/v2/public/auth",
            params={
                'client_id': self.api_key,
                'client_secret': self.api_secret,
                'grant_type': 'client_credentials',
            }
        )
        
        data = response.json()
        if data.get('result'):
            self._access_token = data['result']['access_token']
            self._token_expiry = time.time() + data['result']['expires_in']
            return self._access_token
        
        raise Exception(f"Deribit auth failed: {data.get('error')}")
    
    def _request(self, method: str, endpoint: str, params: dict = None, private: bool = True) -> dict:
        """Make authenticated request to Deribit."""
        url = f"{self.base_url}/api/v2/{endpoint}"
        
        headers = {}
        if private:
            token = self._get_access_token()
            headers['Authorization'] = f'Bearer {token}'
        
        try:
            if method == 'GET':
                response = requests.get(url, params=params, headers=headers, timeout=30)
            else:
                response = requests.post(url, json=params, headers=headers, timeout=30)
            
            return response.json()
        except Exception as e:
            logger.exception(f"Deribit request error: {e}")
            return {'error': {'message': str(e)}}
    
    def _map_order_type(self, order_type: str) -> str:
        """Map internal order type to Deribit format."""
        mapping = {
            'market': 'market',
            'limit': 'limit',
            'stop_market': 'stop_market',
            'stop_limit': 'stop_limit',
            'take_profit': 'take_profit_market',
        }
        return mapping.get(order_type, 'market')
    
    def _map_status(self, deribit_status: str) -> str:
        """Map Deribit order status to internal format."""
        mapping = {
            'open': 'open',
            'filled': 'filled',
            'rejected': 'rejected',
            'cancelled': 'cancelled',
            'untriggered': 'pending',
            'triggered': 'open',
        }
        return mapping.get(deribit_status, 'pending')
    
    def place_order(self, request: OrderRequest) -> OrderResponse:
        """Place order on Deribit."""
        try:
            endpoint = 'private/buy' if request.side == 'buy' else 'private/sell'
            
            params = {
                'instrument_name': request.symbol,
                'amount': float(request.qty),
                'type': self._map_order_type(request.order_type),
            }
            
            if request.price and request.order_type in ('limit', 'stop_limit'):
                params['price'] = float(request.price)
            
            if request.trigger_price and request.order_type in ('stop_market', 'stop_limit', 'take_profit'):
                params['trigger'] = 'mark_price'
                params['trigger_price'] = float(request.trigger_price)
            
            if request.reduce_only:
                params['reduce_only'] = True
            
            if request.client_order_id:
                params['label'] = request.client_order_id
            
            params['time_in_force'] = request.time_in_force.lower()
            
            logger.info(f"Deribit place_order: {params}")
            response = self._request('GET', endpoint, params)
            
            if response.get('result'):
                order = response['result']['order']
                return OrderResponse(
                    success=True,
                    exchange_order_id=order.get('order_id'),
                    client_order_id=order.get('label'),
                    status=self._map_status(order.get('order_state')),
                    filled_qty=Decimal(str(order.get('filled_amount', 0))),
                    avg_price=Decimal(str(order.get('average_price'))) if order.get('average_price') else None,
                    raw_response=response,
                )
            else:
                error = response.get('error', {})
                return OrderResponse(
                    success=False,
                    error_code=str(error.get('code', 'UNKNOWN')),
                    error_message=error.get('message', 'Unknown error'),
                    raw_response=response,
                )
                
        except Exception as e:
            logger.exception(f"Deribit place_order error: {e}")
            return OrderResponse(
                success=False,
                error_code='EXCEPTION',
                error_message=str(e),
            )
    
    def cancel_order(self, symbol: str, order_id: str) -> OrderResponse:
        """Cancel order on Deribit."""
        try:
            response = self._request('GET', 'private/cancel', {'order_id': order_id})
            
            if response.get('result'):
                return OrderResponse(
                    success=True,
                    exchange_order_id=order_id,
                    status='cancelled',
                    raw_response=response,
                )
            else:
                error = response.get('error', {})
                return OrderResponse(
                    success=False,
                    error_code=str(error.get('code', 'UNKNOWN')),
                    error_message=error.get('message', 'Unknown error'),
                    raw_response=response,
                )
                
        except Exception as e:
            logger.exception(f"Deribit cancel_order error: {e}")
            return OrderResponse(
                success=False,
                error_code='EXCEPTION',
                error_message=str(e),
            )
    
    def get_order(self, symbol: str, order_id: str) -> OrderResponse:
        """Get order status from Deribit."""
        try:
            response = self._request('GET', 'private/get_order_state', {'order_id': order_id})
            
            if response.get('result'):
                order = response['result']
                return OrderResponse(
                    success=True,
                    exchange_order_id=order.get('order_id'),
                    client_order_id=order.get('label'),
                    status=self._map_status(order.get('order_state')),
                    filled_qty=Decimal(str(order.get('filled_amount', 0))),
                    avg_price=Decimal(str(order.get('average_price'))) if order.get('average_price') else None,
                    raw_response=response,
                )
            else:
                error = response.get('error', {})
                return OrderResponse(
                    success=False,
                    error_code=str(error.get('code', 'UNKNOWN')),
                    error_message=error.get('message', 'Unknown error'),
                    raw_response=response,
                )
                
        except Exception as e:
            logger.exception(f"Deribit get_order error: {e}")
            return OrderResponse(
                success=False,
                error_code='EXCEPTION',
                error_message=str(e),
            )
    
    def get_open_orders(self, symbol: Optional[str] = None) -> list[OrderResponse]:
        """Get open orders from Deribit."""
        results = []
        
        try:
            params = {}
            if symbol:
                params['instrument_name'] = symbol
            else:
                params['currency'] = 'BTC'  # Default to BTC
            
            response = self._request('GET', 'private/get_open_orders_by_currency', params)
            
            if response.get('result'):
                for order in response['result']:
                    results.append(OrderResponse(
                        success=True,
                        exchange_order_id=order.get('order_id'),
                        client_order_id=order.get('label'),
                        status=self._map_status(order.get('order_state')),
                        filled_qty=Decimal(str(order.get('filled_amount', 0))),
                        avg_price=Decimal(str(order.get('average_price'))) if order.get('average_price') else None,
                        raw_response=order,
                    ))
        except Exception as e:
            logger.exception(f"Deribit get_open_orders error: {e}")
        
        return results
    
    def get_positions(self, symbol: Optional[str] = None) -> PositionSyncResult:
        """Get positions from Deribit."""
        results = []
        had_error = False
        error_msg = None
        
        try:
            if symbol:
                response = self._request('GET', 'private/get_position', {'instrument_name': symbol})
                if response.get('error'):
                    had_error = True
                    error_msg = response['error'].get('message', 'Unknown error')
                positions = [response.get('result')] if response.get('result') else []
            else:
                response = self._request('GET', 'private/get_positions', {'currency': 'BTC'})
                if response.get('error'):
                    had_error = True
                    error_msg = response['error'].get('message', 'Unknown error')
                positions = response.get('result', [])
            
            for pos in positions:
                if not pos:
                    continue
                    
                size = Decimal(str(pos.get('size', 0)))
                if size == 0:
                    continue
                
                side = 'long' if size > 0 else 'short'
                
                # Parse option details from instrument name
                inst_name = pos.get('instrument_name', '')
                option_type = None
                strike = None
                expiry = None
                
                if '-C' in inst_name or '-P' in inst_name:
                    option_type = 'call' if '-C' in inst_name else 'put'
                    parts = inst_name.split('-')
                    if len(parts) >= 4:
                        strike = Decimal(parts[2])
                        try:
                            expiry = datetime.strptime(parts[1], '%d%b%y')
                        except ValueError:
                            pass
                
                results.append(PositionInfo(
                    symbol=inst_name,
                    side=side,
                    qty=abs(size),
                    entry_price=Decimal(str(pos.get('average_price', 0))),
                    mark_price=Decimal(str(pos.get('mark_price'))) if pos.get('mark_price') else None,
                    liquidation_price=Decimal(str(pos.get('estimated_liquidation_price'))) if pos.get('estimated_liquidation_price') else None,
                    unrealized_pnl=Decimal(str(pos.get('floating_profit_loss', 0))),
                    realized_pnl=Decimal(str(pos.get('realized_profit_loss', 0))),
                    leverage=Decimal(str(pos.get('leverage', 1))),
                    option_type=option_type,
                    strike=strike,
                    expiry=expiry,
                ))
        except Exception as e:
            had_error = True
            error_msg = str(e)
            logger.exception(f"Deribit get_positions error: {e}")
        
        return PositionSyncResult(
            success=not had_error,
            positions=results,
            error=error_msg,
        )
    
    def close_position(self, symbol: str, qty: Optional[Decimal] = None) -> OrderResponse:
        """Close position on Deribit."""
        try:
            params = {'instrument_name': symbol}
            if qty:
                params['amount'] = float(qty)
            
            response = self._request('GET', 'private/close_position', params)
            
            if response.get('result'):
                order = response['result'].get('order', {})
                return OrderResponse(
                    success=True,
                    exchange_order_id=order.get('order_id'),
                    status=self._map_status(order.get('order_state', 'filled')),
                    raw_response=response,
                )
            else:
                error = response.get('error', {})
                return OrderResponse(
                    success=False,
                    error_code=str(error.get('code', 'UNKNOWN')),
                    error_message=error.get('message', 'Unknown error'),
                    raw_response=response,
                )
                
        except Exception as e:
            logger.exception(f"Deribit close_position error: {e}")
            return OrderResponse(
                success=False,
                error_code='EXCEPTION',
                error_message=str(e),
            )
    
    def get_instruments(
        self, 
        instrument_type: str = 'option',
        underlying: Optional[str] = None
    ) -> list[InstrumentInfo]:
        """Get available instruments from Deribit."""
        results = []
        currency = underlying or 'BTC'
        
        kind_map = {
            'option': 'option',
            'perpetual': 'future',
            'future': 'future',
        }
        kind = kind_map.get(instrument_type, 'option')
        
        try:
            response = self._request(
                'GET', 
                'public/get_instruments',
                {'currency': currency, 'kind': kind, 'expired': False},
                private=False
            )
            
            if response.get('result'):
                for inst in response['result']:
                    info = InstrumentInfo(
                        symbol=inst.get('instrument_name'),
                        base_currency=inst.get('base_currency'),
                        quote_currency=inst.get('quote_currency', 'USD'),
                        instrument_type=instrument_type,
                        tick_size=Decimal(str(inst.get('tick_size', 0.0001))),
                        lot_size=Decimal(str(inst.get('contract_size', 1))),
                        min_qty=Decimal(str(inst.get('min_trade_amount', 0.1))),
                        max_qty=Decimal('10000'),  # Deribit doesn't expose max
                    )
                    
                    if kind == 'option':
                        info.option_type = inst.get('option_type')
                        info.strike = Decimal(str(inst.get('strike'))) if inst.get('strike') else None
                        info.underlying = currency
                        if inst.get('expiration_timestamp'):
                            info.expiry = datetime.fromtimestamp(inst['expiration_timestamp'] / 1000)
                    
                    results.append(info)
                    
        except Exception as e:
            logger.exception(f"Deribit get_instruments error: {e}")
        
        return results
    
    def get_instrument(self, symbol: str) -> Optional[InstrumentInfo]:
        """Get specific instrument info from Deribit."""
        try:
            response = self._request(
                'GET',
                'public/get_instrument',
                {'instrument_name': symbol},
                private=False
            )
            
            if response.get('result'):
                inst = response['result']
                kind = inst.get('kind', 'option')
                
                info = InstrumentInfo(
                    symbol=inst.get('instrument_name'),
                    base_currency=inst.get('base_currency'),
                    quote_currency=inst.get('quote_currency', 'USD'),
                    instrument_type='option' if kind == 'option' else 'perpetual',
                    tick_size=Decimal(str(inst.get('tick_size', 0.0001))),
                    lot_size=Decimal(str(inst.get('contract_size', 1))),
                    min_qty=Decimal(str(inst.get('min_trade_amount', 0.1))),
                    max_qty=Decimal('10000'),
                )
                
                if kind == 'option':
                    info.option_type = inst.get('option_type')
                    info.strike = Decimal(str(inst.get('strike'))) if inst.get('strike') else None
                    if inst.get('expiration_timestamp'):
                        info.expiry = datetime.fromtimestamp(inst['expiration_timestamp'] / 1000)
                
                return info
        except Exception as e:
            logger.exception(f"Deribit get_instrument error: {e}")
        
        return None
    
    def get_balance(self, currency: Optional[str] = None) -> list[AccountBalance]:
        """Get account balances from Deribit."""
        results = []
        currencies = [currency] if currency else ['BTC', 'ETH']
        
        for curr in currencies:
            try:
                response = self._request('GET', 'private/get_account_summary', {'currency': curr})
                
                if response.get('result'):
                    acc = response['result']
                    results.append(AccountBalance(
                        currency=curr,
                        total=Decimal(str(acc.get('equity', 0))),
                        available=Decimal(str(acc.get('available_funds', 0))),
                        used=Decimal(str(acc.get('initial_margin', 0))),
                        unrealized_pnl=Decimal(str(acc.get('futures_pl', 0))) + Decimal(str(acc.get('options_pl', 0))),
                    ))
            except Exception as e:
                logger.exception(f"Deribit get_balance error for {curr}: {e}")
        
        return results
    
    def normalize_symbol(self, symbol: str) -> str:
        """Convert internal symbol to Deribit format."""
        # Internal: BTC-USDT-PERP -> Deribit: BTC-PERPETUAL
        if '-PERP' in symbol:
            base = symbol.split('-')[0]
            return f"{base}-PERPETUAL"
        return symbol
    
    def denormalize_symbol(self, exchange_symbol: str) -> str:
        """Convert Deribit symbol to internal format."""
        # Deribit: BTC-PERPETUAL -> Internal: BTC-USD-PERP
        if '-PERPETUAL' in exchange_symbol:
            base = exchange_symbol.split('-')[0]
            return f"{base}-USD-PERP"
        return exchange_symbol
