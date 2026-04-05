"""
Base class for options data fetchers.

All exchange-specific fetchers inherit from this and implement
the abstract methods for their API.
"""
from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal
from typing import Optional


class OptionsFetcherBase(ABC):
    """
    Abstract base for options market data fetchers.
    
    Each exchange adapter must implement:
    - get_spot_price()
    - fetch_option_chain()
    - fetch_single_option()
    """
    
    @property
    @abstractmethod
    def exchange_name(self) -> str:
        """Return exchange identifier (e.g., 'bybit', 'deribit')."""
        pass
    
    @abstractmethod
    def get_spot_price(self, symbol: str = 'BTC') -> Optional[Decimal]:
        """Get current spot/index price for underlying."""
        pass
    
    @abstractmethod
    def fetch_option_chain(
        self,
        underlying: str = 'BTC',
        dte_min: int = 1,
        dte_max: int = 30,
        moneyness_range: tuple = (-0.20, 0.20),
    ) -> list[dict]:
        """
        Fetch all options within DTE and moneyness filters.
        
        Returns list of dicts with standardized keys:
        - timestamp, symbol, underlying, expiry, strike, option_type
        - spot_price, index_price
        - bid, ask, mid_price, mark_price, last_price
        - iv, delta, gamma, vega, theta
        - bid_size, ask_size, volume_24h, open_interest
        - dte, moneyness, spread_pct
        - exchange
        """
        pass
    
    @abstractmethod
    def fetch_single_option(self, symbol: str) -> Optional[dict]:
        """Fetch snapshot for a specific option symbol."""
        pass
    
    def safe_decimal(self, value) -> Optional[Decimal]:
        """Safely convert to Decimal. Zero is valid."""
        if value is None or value == '':
            return None
        try:
            return Decimal(str(value))
        except:
            return None
    
    def compute_derived(
        self,
        snapshot: dict,
        timestamp: datetime,
        spot_price: Decimal,
    ) -> dict:
        """Compute derived fields for a snapshot."""
        # DTE
        expiry = snapshot.get('expiry')
        if expiry and timestamp:
            delta_seconds = (expiry - timestamp).total_seconds()
            snapshot['dte'] = max(delta_seconds / 86400, 0)
        
        # Moneyness
        strike = snapshot.get('strike')
        if strike and spot_price and spot_price > 0:
            snapshot['moneyness'] = float((strike - spot_price) / spot_price)
        
        # Spread
        bid = snapshot.get('bid')
        ask = snapshot.get('ask')
        mid = snapshot.get('mid_price')
        if bid is not None and ask is not None and mid and mid > 0:
            snapshot['spread_pct'] = float((ask - bid) / mid)
        
        return snapshot
