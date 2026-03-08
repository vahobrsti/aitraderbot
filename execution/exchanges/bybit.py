"""
Bybit V5 API adapter.
Handles all Bybit-specific quirks: symbol formats, precision, order types, auth.
Uses pybit SDK for HTTP requests.
"""
import logging
from decimal import Decimal
from typing import Optional
from datetime import datetime

from .base import (
    ExchangeAdapter, OrderRequest, OrderResponse, 
    PositionInfo, PositionSyncResult, InstrumentInfo, AccountBalance
)

logger = logging.getLogger(__name__)


class BybitAdapter(ExchangeAdapter):
    """
    Bybit V5 Unified Trading API adapter.
    Supports: Spot, Linear (USDT Perp), Inverse, Options.
    """
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        super().__init__(api_key, api_secret, testnet)
        self._session = None
    
    @property
    def exchange_name(self) -> str:
        return 'bybit'
    
    @property
    def session(self):
        """Lazy-load pybit session."""
        if self._session is None:
            try:
                from pybit.unified_trading import HTTP
                self._session = HTTP(
                    testnet=self.testnet,
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                )
            except ImportError:
                raise ImportError("pybit is required. Install with: pip install pybit")
        return self._session
    
    def _get_category(self, symbol: str) -> str:
        """Determine Bybit category from symbol."""
        if '-C' in symbol or '-P' in symbol:
            return 'option'
        elif symbol.endswith('USDT') or symbol.endswith('PERP'):
            return 'linear'
        elif symbol.startswith('BTC') or symbol.startswith('ETH'):
            return 'inverse'
        return 'spot'
    
    def _map_order_type(self, order_type: str) -> str:
        """Map internal order type to Bybit format."""
        mapping = {
            'market': 'Market',
            'limit': 'Limit',
            'stop_market': 'Market',
            'stop_limit': 'Limit',
            'take_profit': 'Market',
        }
        return mapping.get(order_type, 'Market')
    
    def _map_tif(self, tif: str) -> str:
        """Map time-in-force to Bybit format."""
        mapping = {
            'GTC': 'GTC',
            'IOC': 'IOC',
            'FOK': 'FOK',
            'PostOnly': 'PostOnly',
        }
        return mapping.get(tif, 'GTC')
    
    def _map_status(self, bybit_status: str) -> str:
        """Map Bybit order status to internal format."""
        mapping = {
            'Created': 'pending',
            'New': 'open',
            'PartiallyFilled': 'partial',
            'Filled': 'filled',
            'Cancelled': 'cancelled',
            'Rejected': 'rejected',
            'Expired': 'expired',
            'PartiallyFilledCanceled': 'cancelled',
        }
        return mapping.get(bybit_status, 'pending')
    
    def place_order(self, request: OrderRequest) -> OrderResponse:
        """Place order on Bybit."""
        try:
            category = self._get_category(request.symbol)
            
            params = {
                'category': category,
                'symbol': request.symbol,
                'side': 'Buy' if request.side == 'buy' else 'Sell',
                'orderType': self._map_order_type(request.order_type),
                'qty': str(request.qty),
                'timeInForce': self._map_tif(request.time_in_force),
            }
            
            if request.price and request.order_type in ('limit', 'stop_limit'):
                params['price'] = str(request.price)
            
            if request.trigger_price and request.order_type in ('stop_market', 'stop_limit', 'take_profit'):
                params['triggerPrice'] = str(request.trigger_price)
                params['triggerDirection'] = 1 if request.side == 'buy' else 2
            
            if request.reduce_only:
                params['reduceOnly'] = True
            
            if request.client_order_id:
                params['orderLinkId'] = request.client_order_id
            
            logger.info(f"Bybit place_order: {params}")
            response = self.session.place_order(**params)
            
            if response.get('retCode') == 0:
                result = response.get('result', {})
                return OrderResponse(
                    success=True,
                    exchange_order_id=result.get('orderId'),
                    client_order_id=result.get('orderLinkId'),
                    status='submitted',
                    raw_response=response,
                )
            else:
                return OrderResponse(
                    success=False,
                    error_code=str(response.get('retCode')),
                    error_message=response.get('retMsg'),
                    raw_response=response,
                )
                
        except Exception as e:
            logger.exception(f"Bybit place_order error: {e}")
            return OrderResponse(
                success=False,
                error_code='EXCEPTION',
                error_message=str(e),
            )
    
    def cancel_order(self, symbol: str, order_id: str) -> OrderResponse:
        """Cancel order on Bybit."""
        try:
            category = self._get_category(symbol)
            params = {
                'category': category,
                'symbol': symbol,
            }
            
            # Determine if order_id is exchange ID or client ID
            if order_id.startswith('ai_'):
                params['orderLinkId'] = order_id
            else:
                params['orderId'] = order_id
            
            response = self.session.cancel_order(**params)
            
            if response.get('retCode') == 0:
                return OrderResponse(
                    success=True,
                    exchange_order_id=response.get('result', {}).get('orderId'),
                    status='cancelled',
                    raw_response=response,
                )
            else:
                return OrderResponse(
                    success=False,
                    error_code=str(response.get('retCode')),
                    error_message=response.get('retMsg'),
                    raw_response=response,
                )
                
        except Exception as e:
            logger.exception(f"Bybit cancel_order error: {e}")
            return OrderResponse(
                success=False,
                error_code='EXCEPTION',
                error_message=str(e),
            )
    
    def get_order(self, symbol: str, order_id: str) -> OrderResponse:
        """Get order status from Bybit."""
        try:
            category = self._get_category(symbol)
            params = {
                'category': category,
                'symbol': symbol,
            }
            
            if order_id.startswith('ai_'):
                params['orderLinkId'] = order_id
            else:
                params['orderId'] = order_id
            
            response = self.session.get_order_history(**params)
            
            if response.get('retCode') == 0:
                orders = response.get('result', {}).get('list', [])
                if orders:
                    order = orders[0]
                    return OrderResponse(
                        success=True,
                        exchange_order_id=order.get('orderId'),
                        client_order_id=order.get('orderLinkId'),
                        status=self._map_status(order.get('orderStatus')),
                        filled_qty=Decimal(order.get('cumExecQty', '0')),
                        avg_price=Decimal(order.get('avgPrice')) if order.get('avgPrice') else None,
                        raw_response=response,
                    )
                return OrderResponse(
                    success=False,
                    error_code='NOT_FOUND',
                    error_message='Order not found',
                )
            else:
                return OrderResponse(
                    success=False,
                    error_code=str(response.get('retCode')),
                    error_message=response.get('retMsg'),
                    raw_response=response,
                )
                
        except Exception as e:
            logger.exception(f"Bybit get_order error: {e}")
            return OrderResponse(
                success=False,
                error_code='EXCEPTION',
                error_message=str(e),
            )
    
    def get_open_orders(self, symbol: Optional[str] = None) -> list[OrderResponse]:
        """Get open orders from Bybit."""
        results = []
        categories = ['linear', 'option'] if not symbol else [self._get_category(symbol)]
        
        for category in categories:
            try:
                params = {'category': category}
                if symbol:
                    params['symbol'] = symbol
                
                response = self.session.get_open_orders(**params)
                
                if response.get('retCode') == 0:
                    for order in response.get('result', {}).get('list', []):
                        results.append(OrderResponse(
                            success=True,
                            exchange_order_id=order.get('orderId'),
                            client_order_id=order.get('orderLinkId'),
                            status=self._map_status(order.get('orderStatus')),
                            filled_qty=Decimal(order.get('cumExecQty', '0')),
                            avg_price=Decimal(order.get('avgPrice')) if order.get('avgPrice') else None,
                            raw_response=order,
                        ))
            except Exception as e:
                logger.exception(f"Bybit get_open_orders error for {category}: {e}")
        
        return results
    
    def get_positions(self, symbol: Optional[str] = None) -> PositionSyncResult:
        """Get positions from Bybit."""
        results = []
        had_error = False
        error_msg = None
        
        # Options are USDC-settled, perpetuals are USDT-settled
        category_settle_map = [
            ('linear', 'USDT'),
            ('option', 'USDC'),
        ]
        
        if symbol:
            category = self._get_category(symbol)
            settle = 'USDC' if category == 'option' else 'USDT'
            category_settle_map = [(category, settle)]
        
        for category, settle_coin in category_settle_map:
            try:
                params = {'category': category, 'settleCoin': settle_coin}
                if symbol:
                    params['symbol'] = symbol
                
                response = self.session.get_positions(**params)
                
                if response.get('retCode') == 0:
                    for pos in response.get('result', {}).get('list', []):
                        size = Decimal(pos.get('size', '0'))
                        if size == 0:
                            continue
                        
                        side = pos.get('side', '').lower()
                        if side == 'buy':
                            side = 'long'
                        elif side == 'sell':
                            side = 'short'
                        
                        results.append(PositionInfo(
                            symbol=pos.get('symbol'),
                            side=side,
                            qty=size,
                            entry_price=Decimal(pos.get('avgPrice', '0')),
                            mark_price=Decimal(pos.get('markPrice')) if pos.get('markPrice') else None,
                            liquidation_price=Decimal(pos.get('liqPrice')) if pos.get('liqPrice') else None,
                            unrealized_pnl=Decimal(pos.get('unrealisedPnl', '0')),
                            realized_pnl=Decimal(pos.get('cumRealisedPnl', '0')),
                            leverage=Decimal(pos.get('leverage', '1')),
                        ))
                else:
                    had_error = True
                    error_msg = f"Bybit API error: {response.get('retMsg')}"
                    logger.error(f"Bybit get_positions error for {category}: {error_msg}")
            except Exception as e:
                had_error = True
                error_msg = str(e)
                logger.exception(f"Bybit get_positions error for {category}: {e}")
        
        return PositionSyncResult(
            success=not had_error,
            positions=results,
            error=error_msg,
        )
    
    def close_position(self, symbol: str, qty: Optional[Decimal] = None) -> OrderResponse:
        """Close position on Bybit."""
        sync_result = self.get_positions(symbol)
        if not sync_result.success or not sync_result.positions:
            return OrderResponse(
                success=False,
                error_code='NO_POSITION',
                error_message=f'No position found for {symbol}' if sync_result.success else sync_result.error,
            )
        
        pos = sync_result.positions[0]
        close_qty = qty or pos.qty
        close_side = 'sell' if pos.side == 'long' else 'buy'
        
        return self.place_order(OrderRequest(
            symbol=symbol,
            side=close_side,
            order_type='market',
            qty=close_qty,
            reduce_only=True,
        ))
    
    def get_instruments(
        self, 
        instrument_type: str = 'option',
        underlying: Optional[str] = None
    ) -> list[InstrumentInfo]:
        """Get available instruments from Bybit."""
        results = []
        
        category_map = {
            'option': 'option',
            'perpetual': 'linear',
            'future': 'linear',
            'spot': 'spot',
        }
        category = category_map.get(instrument_type, 'linear')
        
        try:
            params = {'category': category}
            if underlying and category == 'option':
                params['baseCoin'] = underlying
            
            response = self.session.get_instruments_info(**params)
            
            if response.get('retCode') == 0:
                for inst in response.get('result', {}).get('list', []):
                    lot_filter = inst.get('lotSizeFilter', {})
                    price_filter = inst.get('priceFilter', {})
                    
                    info = InstrumentInfo(
                        symbol=inst.get('symbol'),
                        base_currency=inst.get('baseCoin', ''),
                        quote_currency=inst.get('quoteCoin', inst.get('settleCoin', '')),
                        instrument_type=instrument_type,
                        tick_size=Decimal(price_filter.get('tickSize', '0.01')),
                        lot_size=Decimal(lot_filter.get('qtyStep', '0.001')),
                        min_qty=Decimal(lot_filter.get('minOrderQty', '0.001')),
                        max_qty=Decimal(lot_filter.get('maxOrderQty', '100')),
                    )
                    
                    if category == 'option':
                        info.option_type = inst.get('optionsType', '').lower()
                        info.underlying = inst.get('baseCoin')
                        if inst.get('deliveryTime'):
                            info.expiry = datetime.fromtimestamp(int(inst['deliveryTime']) / 1000)
                    
                    results.append(info)
                    
        except Exception as e:
            logger.exception(f"Bybit get_instruments error: {e}")
        
        return results
    
    def get_instrument(self, symbol: str) -> Optional[InstrumentInfo]:
        """Get specific instrument info from Bybit."""
        category = self._get_category(symbol)
        
        try:
            response = self.session.get_instruments_info(
                category=category,
                symbol=symbol,
            )
            
            if response.get('retCode') == 0:
                instruments = response.get('result', {}).get('list', [])
                if instruments:
                    inst = instruments[0]
                    lot_filter = inst.get('lotSizeFilter', {})
                    price_filter = inst.get('priceFilter', {})
                    
                    return InstrumentInfo(
                        symbol=inst.get('symbol'),
                        base_currency=inst.get('baseCoin', ''),
                        quote_currency=inst.get('quoteCoin', inst.get('settleCoin', '')),
                        instrument_type='option' if category == 'option' else 'perpetual',
                        tick_size=Decimal(price_filter.get('tickSize', '0.01')),
                        lot_size=Decimal(lot_filter.get('qtyStep', '0.001')),
                        min_qty=Decimal(lot_filter.get('minOrderQty', '0.001')),
                        max_qty=Decimal(lot_filter.get('maxOrderQty', '100')),
                    )
        except Exception as e:
            logger.exception(f"Bybit get_instrument error: {e}")
        
        return None
    
    def get_balance(self, currency: Optional[str] = None) -> list[AccountBalance]:
        """Get account balances from Bybit."""
        results = []
        
        try:
            response = self.session.get_wallet_balance(accountType='UNIFIED')
            
            if response.get('retCode') == 0:
                for account in response.get('result', {}).get('list', []):
                    for coin in account.get('coin', []):
                        if currency and coin.get('coin') != currency:
                            continue
                        
                        results.append(AccountBalance(
                            currency=coin.get('coin'),
                            total=Decimal(coin.get('walletBalance', '0')),
                            available=Decimal(coin.get('availableToWithdraw', '0')),
                            used=Decimal(coin.get('locked', '0')),
                            unrealized_pnl=Decimal(coin.get('unrealisedPnl', '0')),
                        ))
        except Exception as e:
            logger.exception(f"Bybit get_balance error: {e}")
        
        return results
    
    def normalize_symbol(self, symbol: str) -> str:
        """Convert internal symbol to Bybit format."""
        # Internal: BTC-USDT-PERP -> Bybit: BTCUSDT
        # Internal: BTC-30JUN23-20000-C -> Bybit: BTC-30JUN23-20000-C (options same)
        if '-PERP' in symbol:
            return symbol.replace('-USDT-PERP', 'USDT').replace('-', '')
        return symbol
    
    def denormalize_symbol(self, exchange_symbol: str) -> str:
        """Convert Bybit symbol to internal format."""
        # Bybit: BTCUSDT -> Internal: BTC-USDT-PERP
        if exchange_symbol.endswith('USDT') and '-' not in exchange_symbol:
            base = exchange_symbol[:-4]
            return f"{base}-USDT-PERP"
        return exchange_symbol
