"""
Deribit Options Data Fetcher

Collects option snapshots from Deribit API.
Deribit has better historical data and is the primary BTC options venue.

Usage:
    from datafeed.ingestion.deribit_options import DeribitOptionsFetcher
    
    fetcher = DeribitOptionsFetcher()
    snapshots = fetcher.fetch_option_chain(underlying='BTC', dte_range=(7, 30))
"""
import logging
import requests
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from .options_base import OptionsFetcherBase

logger = logging.getLogger(__name__)


class DeribitOptionsFetcher(OptionsFetcherBase):
    """
    Fetches option market data from Deribit API.
    
    Deribit API is public for market data (no auth needed).
    Auth provides higher rate limits and access to historical data.
    Base URL: https://www.deribit.com/api/v2/public/
    """
    
    BASE_URL = "https://www.deribit.com/api/v2"
    TEST_URL = "https://test.deribit.com/api/v2"
    
    def __init__(self, client_id: str = None, client_secret: str = None, testnet: bool = False):
        self.client_id = client_id
        self.client_secret = client_secret
        self.testnet = testnet
        self.base_url = self.TEST_URL if testnet else self.BASE_URL
        self._access_token = None
        self._token_expiry = None
    
    @property
    def exchange_name(self) -> str:
        return 'deribit'
    
    def _authenticate(self) -> Optional[str]:
        """Get access token for authenticated requests."""
        if not self.client_id or not self.client_secret:
            return None
        
        # Check if token still valid
        if self._access_token and self._token_expiry:
            if datetime.now(timezone.utc).timestamp() < self._token_expiry - 60:
                return self._access_token
        
        try:
            url = f"{self.base_url}/public/auth"
            params = {
                'grant_type': 'client_credentials',
                'client_id': self.client_id,
                'client_secret': self.client_secret,
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'result' in data:
                self._access_token = data['result']['access_token']
                self._token_expiry = datetime.now(timezone.utc).timestamp() + data['result']['expires_in']
                return self._access_token
        except Exception as e:
            logger.error(f"Deribit auth error: {e}")
        
        return None
    
    def _request(self, method: str, params: dict = None, private: bool = False) -> Optional[dict]:
        """Make API request."""
        try:
            prefix = 'private' if private else 'public'
            url = f"{self.base_url}/{prefix}/{method}"
            
            headers = {}
            if private or self.client_id:
                token = self._authenticate()
                if token:
                    headers['Authorization'] = f'Bearer {token}'
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'result' in data:
                return data['result']
            else:
                logger.error(f"Deribit API error: {data.get('error')}")
                return None
        except Exception as e:
            logger.error(f"Deribit request error: {e}")
            return None
    
    def get_spot_price(self, symbol: str = 'BTC') -> Optional[Decimal]:
        """Get current index price."""
        result = self._request('get_index_price', {'index_name': f'{symbol.lower()}_usd'})
        if result:
            return self.safe_decimal(result.get('index_price'))
        return None
    
    def get_instruments(self, underlying: str = 'BTC', kind: str = 'option') -> list[dict]:
        """Get all available instruments."""
        result = self._request('get_instruments', {
            'currency': underlying,
            'kind': kind,
            'expired': 'false',
        })
        return result or []
    
    def fetch_option_chain(
        self,
        underlying: str = 'BTC',
        dte_min: int = 1,
        dte_max: int = 30,
        moneyness_range: tuple = (-0.20, 0.20),
    ) -> list[dict]:
        """Fetch all options within DTE and moneyness filters."""
        now = datetime.now(timezone.utc)
        snapshots = []
        
        # Get spot price
        spot_price = self.get_spot_price(underlying)
        if not spot_price:
            logger.error("Could not fetch spot price")
            return []
        
        # Get all option instruments
        instruments = self.get_instruments(underlying, 'option')
        logger.info(f"Found {len(instruments)} {underlying} options on Deribit")
        
        # Filter by DTE first
        valid_instruments = []
        for inst in instruments:
            expiry_ts = inst.get('expiration_timestamp', 0) / 1000
            expiry = datetime.fromtimestamp(expiry_ts, tz=timezone.utc)
            dte = (expiry - now).total_seconds() / 86400
            
            if dte_min <= dte <= dte_max:
                inst['_expiry'] = expiry
                inst['_dte'] = dte
                valid_instruments.append(inst)
        
        logger.info(f"Filtered to {len(valid_instruments)} options by DTE ({dte_min}-{dte_max})")
        
        # Fetch ticker for each valid instrument
        for inst in valid_instruments:
            symbol = inst['instrument_name']
            ticker = self._request('ticker', {'instrument_name': symbol})
            
            if not ticker:
                continue
            
            snapshot = self._parse_ticker(ticker, inst, now, spot_price)
            if not snapshot:
                continue
            
            # Apply moneyness filter
            moneyness = snapshot.get('moneyness', 0)
            if moneyness < moneyness_range[0] or moneyness > moneyness_range[1]:
                continue
            
            snapshots.append(snapshot)
        
        logger.info(f"Final: {len(snapshots)} options after moneyness filter")
        return snapshots
    
    def fetch_single_option(self, symbol: str) -> Optional[dict]:
        """Fetch snapshot for a specific option symbol."""
        now = datetime.now(timezone.utc)
        
        # Extract underlying from symbol (e.g., BTC-26APR26-100000-C -> BTC)
        underlying = symbol.split('-')[0]
        spot_price = self.get_spot_price(underlying)
        
        if not spot_price:
            return None
        
        # Get instrument info
        instruments = self.get_instruments(underlying)
        inst = next((i for i in instruments if i['instrument_name'] == symbol), None)
        
        if not inst:
            logger.error(f"Instrument {symbol} not found")
            return None
        
        # Get ticker
        ticker = self._request('ticker', {'instrument_name': symbol})
        if not ticker:
            return None
        
        return self._parse_ticker(ticker, inst, now, spot_price)
    
    def _parse_ticker(
        self,
        ticker: dict,
        instrument: dict,
        timestamp: datetime,
        spot_price: Decimal,
    ) -> Optional[dict]:
        """Parse Deribit ticker into standardized snapshot dict."""
        try:
            symbol = ticker.get('instrument_name', '')
            
            # Parse from instrument info
            strike = self.safe_decimal(instrument.get('strike'))
            option_type = instrument.get('option_type', '').lower()  # 'call' or 'put'
            
            # Expiry
            expiry_ts = instrument.get('expiration_timestamp', 0) / 1000
            expiry = datetime.fromtimestamp(expiry_ts, tz=timezone.utc)
            
            # DTE
            dte = (expiry - timestamp).total_seconds() / 86400
            if dte < 0:
                return None
            
            # Moneyness
            moneyness = float((strike - spot_price) / spot_price) if strike and spot_price else 0
            
            # Prices (Deribit returns in BTC, convert to USD)
            index_price = self.safe_decimal(ticker.get('index_price'))
            usd_multiplier = float(index_price) if index_price else float(spot_price)
            
            bid = self.safe_decimal(ticker.get('best_bid_price'))
            ask = self.safe_decimal(ticker.get('best_ask_price'))
            mark = self.safe_decimal(ticker.get('mark_price'))
            last = self.safe_decimal(ticker.get('last_price'))
            
            # Convert BTC prices to USD
            if bid is not None:
                bid = Decimal(str(float(bid) * usd_multiplier))
            if ask is not None:
                ask = Decimal(str(float(ask) * usd_multiplier))
            if mark is not None:
                mark = Decimal(str(float(mark) * usd_multiplier))
            if last is not None:
                last = Decimal(str(float(last) * usd_multiplier))
            
            # Mid price
            mid = None
            if bid is not None and ask is not None:
                mid = (bid + ask) / 2
            
            # Spread
            spread_pct = None
            if bid is not None and ask is not None and mid and mid > 0:
                spread_pct = float((ask - bid) / mid)
            
            # IV (Deribit returns as decimal)
            iv = self.safe_decimal(ticker.get('mark_iv'))
            if iv is not None:
                iv = iv / 100  # Convert from percentage to decimal
            
            # Greeks
            greeks = ticker.get('greeks', {})
            delta = self.safe_decimal(greeks.get('delta'))
            gamma = self.safe_decimal(greeks.get('gamma'))
            vega = self.safe_decimal(greeks.get('vega'))
            theta = self.safe_decimal(greeks.get('theta'))
            
            # Liquidity
            bid_size = self.safe_decimal(ticker.get('best_bid_amount'))
            ask_size = self.safe_decimal(ticker.get('best_ask_amount'))
            volume_24h = self.safe_decimal(ticker.get('stats', {}).get('volume'))
            open_interest = self.safe_decimal(ticker.get('open_interest'))
            
            return {
                'timestamp': timestamp,
                'symbol': symbol,
                'underlying': symbol.split('-')[0],
                'expiry': expiry,
                'strike': strike,
                'option_type': option_type,
                'spot_price': spot_price,
                'index_price': index_price,
                'bid': bid,
                'ask': ask,
                'mid_price': mid,
                'mark_price': mark,
                'last_price': last,
                'iv': iv,
                'delta': delta,
                'gamma': gamma,
                'vega': vega,
                'theta': theta,
                'bid_size': bid_size,
                'ask_size': ask_size,
                'volume_24h': volume_24h,
                'open_interest': open_interest,
                'dte': dte,
                'moneyness': moneyness,
                'spread_pct': spread_pct,
                'exchange': 'deribit',
            }
            
        except Exception as e:
            logger.error(f"Error parsing Deribit ticker {ticker.get('instrument_name')}: {e}")
            return None
    
    def fetch_historical_trades(
        self,
        symbol: str,
        start_timestamp: int,
        end_timestamp: int,
        count: int = 1000,
    ) -> list[dict]:
        """
        Fetch historical trades for an instrument.
        
        Useful for backfilling and analyzing actual execution prices.
        """
        result = self._request('get_last_trades_by_instrument_and_time', {
            'instrument_name': symbol,
            'start_timestamp': start_timestamp,
            'end_timestamp': end_timestamp,
            'count': count,
            'sorting': 'asc',
        })
        return result.get('trades', []) if result else []
