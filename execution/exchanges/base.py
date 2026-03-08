"""
Provider-agnostic exchange interface.
All exchange adapters must implement this interface.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional
from datetime import datetime


@dataclass
class OrderRequest:
    """Standardized order request across exchanges."""
    symbol: str
    side: str  # 'buy' or 'sell'
    order_type: str  # 'market', 'limit', 'stop_market', 'stop_limit'
    qty: Decimal
    price: Optional[Decimal] = None
    trigger_price: Optional[Decimal] = None
    reduce_only: bool = False
    client_order_id: Optional[str] = None
    time_in_force: str = 'GTC'  # GTC, IOC, FOK


@dataclass
class OrderResponse:
    """Standardized order response from exchanges."""
    success: bool
    exchange_order_id: Optional[str] = None
    client_order_id: Optional[str] = None
    status: Optional[str] = None
    filled_qty: Decimal = Decimal('0')
    avg_price: Optional[Decimal] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    raw_response: Optional[dict] = None


@dataclass
class PositionSyncResult:
    """Result of position sync from exchange."""
    success: bool
    positions: list['PositionInfo']
    error: Optional[str] = None


@dataclass
class PositionInfo:
    """Standardized position information."""
    symbol: str
    side: str  # 'long', 'short', 'none'
    qty: Decimal
    entry_price: Decimal
    mark_price: Optional[Decimal] = None
    liquidation_price: Optional[Decimal] = None
    unrealized_pnl: Decimal = Decimal('0')
    realized_pnl: Decimal = Decimal('0')
    leverage: Decimal = Decimal('1')
    margin_mode: str = 'cross'
    # Option-specific
    option_type: Optional[str] = None
    strike: Optional[Decimal] = None
    expiry: Optional[datetime] = None


@dataclass
class InstrumentInfo:
    """Standardized instrument/contract information."""
    symbol: str
    base_currency: str
    quote_currency: str
    instrument_type: str  # 'perpetual', 'future', 'option', 'spot'
    tick_size: Decimal
    lot_size: Decimal
    min_qty: Decimal
    max_qty: Decimal
    # Option-specific
    option_type: Optional[str] = None
    strike: Optional[Decimal] = None
    expiry: Optional[datetime] = None
    underlying: Optional[str] = None


@dataclass
class AccountBalance:
    """Standardized account balance."""
    currency: str
    total: Decimal
    available: Decimal
    used: Decimal
    unrealized_pnl: Decimal = Decimal('0')


class ExchangeAdapter(ABC):
    """
    Abstract base class for exchange adapters.
    Implementations handle exchange-specific quirks internally.
    """
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
    
    @property
    @abstractmethod
    def exchange_name(self) -> str:
        """Return exchange identifier."""
        pass
    
    # === Order Management ===
    
    @abstractmethod
    def place_order(self, request: OrderRequest) -> OrderResponse:
        """
        Place a new order on the exchange.
        Returns standardized OrderResponse.
        """
        pass
    
    @abstractmethod
    def cancel_order(self, symbol: str, order_id: str) -> OrderResponse:
        """
        Cancel an existing order.
        order_id can be exchange_order_id or client_order_id.
        """
        pass
    
    @abstractmethod
    def get_order(self, symbol: str, order_id: str) -> OrderResponse:
        """Get current status of an order."""
        pass
    
    @abstractmethod
    def get_open_orders(self, symbol: Optional[str] = None) -> list[OrderResponse]:
        """Get all open orders, optionally filtered by symbol."""
        pass
    
    # === Position Management ===
    
    @abstractmethod
    def get_positions(self, symbol: Optional[str] = None) -> 'PositionSyncResult':
        """
        Get current positions, optionally filtered by symbol.
        Returns PositionSyncResult with success flag to distinguish API failures from empty positions.
        """
        pass
    
    @abstractmethod
    def close_position(self, symbol: str, qty: Optional[Decimal] = None) -> OrderResponse:
        """
        Close a position. If qty is None, close entire position.
        Returns the closing order response.
        """
        pass
    
    # === Instrument Info ===
    
    @abstractmethod
    def get_instruments(
        self, 
        instrument_type: str = 'option',
        underlying: Optional[str] = None
    ) -> list[InstrumentInfo]:
        """
        Get available instruments.
        instrument_type: 'perpetual', 'future', 'option', 'spot'
        """
        pass
    
    @abstractmethod
    def get_instrument(self, symbol: str) -> Optional[InstrumentInfo]:
        """Get info for a specific instrument."""
        pass
    
    # === Account ===
    
    @abstractmethod
    def get_balance(self, currency: Optional[str] = None) -> list[AccountBalance]:
        """Get account balances."""
        pass
    
    # === Utility ===
    
    @abstractmethod
    def normalize_symbol(self, symbol: str) -> str:
        """
        Convert internal symbol format to exchange-specific format.
        E.g., 'BTC-USDT-PERP' -> 'BTCUSDT' (Bybit) or 'BTC-PERPETUAL' (Deribit)
        """
        pass
    
    @abstractmethod
    def denormalize_symbol(self, exchange_symbol: str) -> str:
        """Convert exchange symbol to internal format."""
        pass
    
    def health_check(self) -> bool:
        """Check if exchange connection is healthy."""
        try:
            self.get_balance()
            return True
        except Exception:
            return False
