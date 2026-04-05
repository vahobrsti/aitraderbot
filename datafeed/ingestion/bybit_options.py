"""
Bybit Options Data Fetcher

Collects option snapshots (price, IV, greeks, liquidity) from Bybit V5 API.
Designed for building training datasets for option PnL modeling.

Usage:
    from datafeed.ingestion.bybit_options import BybitOptionsFetcher
    
    fetcher = BybitOptionsFetcher()
    snapshots = fetcher.fetch_option_chain(underlying='BTC', dte_range=(7, 30))
"""
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

logger = logging.getLogger(__name__)


class BybitOptionsFetcher:
    """
    Fetches option market data from Bybit V5 API.
    
    Data collected per option:
    - Prices: bid, ask, mid, mark, last
    - IV: implied volatility
    - Greeks: delta, gamma, vega, theta
    - Liquidity: bid/ask size, volume, open interest
    - Underlying: spot price, index price
    """
    
    def __init__(self, api_key: str = None, api_secret: str = None, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self._session = None
    
    @property
    def session(self):
        """Lazy-load pybit session."""
        if self._session is None:
            try:
                from pybit.unified_trading import HTTP
                # Use auth if provided (higher rate limits), else public
                if self.api_key and self.api_secret:
                    self._session = HTTP(
                        testnet=self.testnet,
                        api_key=self.api_key,
                        api_secret=self.api_secret,
                    )
                else:
                    self._session = HTTP(testnet=self.testnet)
            except ImportError:
                raise ImportError("pybit required: pip install pybit")
        return self._session
    
    def get_spot_price(self, symbol: str = 'BTCUSDT') -> Optional[Decimal]:
        """Get current spot price."""
        try:
            response = self.session.get_tickers(category='spot', symbol=symbol)
            if response.get('retCode') == 0:
                tickers = response.get('result', {}).get('list', [])
                if tickers:
                    return Decimal(tickers[0].get('lastPrice', '0'))
        except Exception as e:
            logger.error(f"Error fetching spot price: {e}")
        return None
    
    def get_index_price(self, symbol: str = 'BTC') -> Optional[Decimal]:
        """Get BTC index price from options category."""
        try:
            # Bybit provides index price in option tickers
            response = self.session.get_tickers(category='option', baseCoin=symbol)
            if response.get('retCode') == 0:
                tickers = response.get('result', {}).get('list', [])
                if tickers:
                    # All options share same underlying index
                    return Decimal(tickers[0].get('underlyingPrice', '0'))
        except Exception as e:
            logger.error(f"Error fetching index price: {e}")
        return None
    
    def fetch_option_chain(
        self,
        underlying: str = 'BTC',
        dte_min: int = 1,
        dte_max: int = 30,
        moneyness_range: tuple = (-0.20, 0.20),
    ) -> list[dict]:
        """
        Fetch all options within DTE and moneyness filters.
        
        Args:
            underlying: Base coin (BTC, ETH)
            dte_min: Minimum days to expiry
            dte_max: Maximum days to expiry
            moneyness_range: (min, max) as fraction of spot, e.g., (-0.20, 0.20) = ±20%
        
        Returns:
            List of option snapshot dicts ready for OptionSnapshot model
        """
        now = datetime.now(timezone.utc)
        snapshots = []
        
        # Get current spot/index price
        spot_price = self.get_spot_price(f'{underlying}USDT')
        if not spot_price:
            logger.error("Could not fetch spot price")
            return []
        
        try:
            # Fetch all option tickers
            response = self.session.get_tickers(category='option', baseCoin=underlying)
            
            if response.get('retCode') != 0:
                logger.error(f"Bybit API error: {response.get('retMsg')}")
                return []
            
            tickers = response.get('result', {}).get('list', [])
            logger.info(f"Fetched {len(tickers)} option tickers for {underlying}")
            
            for ticker in tickers:
                snapshot = self._parse_ticker(ticker, now, spot_price)
                if not snapshot:
                    continue
                
                # Apply DTE filter
                dte = snapshot.get('dte', 0)
                if dte < dte_min or dte > dte_max:
                    continue
                
                # Apply moneyness filter
                moneyness = snapshot.get('moneyness', 0)
                if moneyness < moneyness_range[0] or moneyness > moneyness_range[1]:
                    continue
                
                snapshots.append(snapshot)
            
            logger.info(f"Filtered to {len(snapshots)} options (DTE {dte_min}-{dte_max}, moneyness {moneyness_range})")
            
        except Exception as e:
            logger.exception(f"Error fetching option chain: {e}")
        
        return snapshots
    
    def fetch_single_option(self, symbol: str) -> Optional[dict]:
        """Fetch snapshot for a specific option symbol."""
        now = datetime.now(timezone.utc)
        
        # Extract underlying from symbol (e.g., BTC-26APR26-100000-C -> BTC)
        underlying = symbol.split('-')[0]
        spot_price = self.get_spot_price(f'{underlying}USDT')
        
        if not spot_price:
            return None
        
        try:
            response = self.session.get_tickers(category='option', symbol=symbol)
            
            if response.get('retCode') == 0:
                tickers = response.get('result', {}).get('list', [])
                if tickers:
                    return self._parse_ticker(tickers[0], now, spot_price)
        except Exception as e:
            logger.error(f"Error fetching option {symbol}: {e}")
        
        return None
    
    def _parse_ticker(
        self, 
        ticker: dict, 
        timestamp: datetime, 
        spot_price: Decimal
    ) -> Optional[dict]:
        """Parse Bybit ticker response into snapshot dict."""
        try:
            symbol = ticker.get('symbol', '')
            
            # Parse symbol: BTC-26APR26-100000-C or BTC-26APR26-100000-C-USDT
            parts = symbol.split('-')
            if len(parts) < 4:
                return None
            
            underlying = parts[0]
            expiry_str = parts[1]
            strike = Decimal(parts[2])
            # Handle both C/P and C-USDT/P-USDT formats
            opt_part = parts[3]
            option_type = 'call' if opt_part.startswith('C') else 'put'
            
            # Parse expiry (format: 26APR26 = 26 Apr 2026)
            expiry = self._parse_expiry(expiry_str)
            if not expiry:
                return None
            
            # Calculate DTE
            dte = (expiry - timestamp).total_seconds() / 86400
            if dte < 0:
                return None
            
            # Calculate moneyness
            moneyness = float((strike - spot_price) / spot_price)
            
            # Extract prices
            bid = self._safe_decimal(ticker.get('bid1Price'))
            ask = self._safe_decimal(ticker.get('ask1Price'))
            mark = self._safe_decimal(ticker.get('markPrice'))
            last = self._safe_decimal(ticker.get('lastPrice'))
            index_price = self._safe_decimal(ticker.get('underlyingPrice'))
            
            # Calculate mid price
            mid = None
            if bid and ask:
                mid = (bid + ask) / 2
            
            # Calculate spread
            spread_pct = None
            if bid and ask and mid and mid > 0:
                spread_pct = float((ask - bid) / mid)
            
            # Extract IV (Bybit provides as decimal, e.g., 0.65 = 65%)
            iv = self._safe_decimal(ticker.get('markIv'))
            
            # Extract greeks
            delta = self._safe_decimal(ticker.get('delta'))
            gamma = self._safe_decimal(ticker.get('gamma'))
            vega = self._safe_decimal(ticker.get('vega'))
            theta = self._safe_decimal(ticker.get('theta'))
            
            # Liquidity
            bid_size = self._safe_decimal(ticker.get('bid1Size'))
            ask_size = self._safe_decimal(ticker.get('ask1Size'))
            volume_24h = self._safe_decimal(ticker.get('volume24h'))
            open_interest = self._safe_decimal(ticker.get('openInterest'))
            
            return {
                'timestamp': timestamp,
                'symbol': symbol,
                'underlying': underlying,
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
                'exchange': 'bybit',
            }
            
        except Exception as e:
            logger.error(f"Error parsing ticker {ticker.get('symbol')}: {e}")
            return None
    
    def _parse_expiry(self, expiry_str: str) -> Optional[datetime]:
        """Parse Bybit expiry format: 26APR26 or 7APR26 -> datetime."""
        try:
            # Format: D[D]MMMYY (e.g., 26APR26 or 7APR26)
            import re
            match = re.match(r'^(\d{1,2})([A-Z]{3})(\d{2})$', expiry_str.upper())
            if not match:
                return None
            
            day = int(match.group(1))
            month_str = match.group(2)
            year = 2000 + int(match.group(3))
            
            months = {
                'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4,
                'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8,
                'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
            }
            month = months.get(month_str)
            if not month:
                return None
            
            # Bybit options expire at 08:00 UTC
            return datetime(year, month, day, 8, 0, 0, tzinfo=timezone.utc)
            
        except Exception as e:
            logger.error(f"Error parsing expiry {expiry_str}: {e}")
            return None
    
    def _safe_decimal(self, value) -> Optional[Decimal]:
        """Safely convert to Decimal. Zero is a valid value."""
        if value is None or value == '':
            return None
        try:
            return Decimal(str(value))
        except:
            return None
    
    def fetch_historical_klines(
        self,
        symbol: str = 'BTCUSDT',
        interval: str = '60',  # 1 hour
        limit: int = 200,
    ) -> list[dict]:
        """
        Fetch historical OHLCV for underlying.
        
        Args:
            symbol: Trading pair
            interval: Kline interval (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)
            limit: Number of candles (max 200)
        
        Returns:
            List of OHLCV dicts
        """
        try:
            response = self.session.get_kline(
                category='spot',
                symbol=symbol,
                interval=interval,
                limit=limit,
            )
            
            if response.get('retCode') != 0:
                logger.error(f"Kline API error: {response.get('retMsg')}")
                return []
            
            klines = []
            for k in response.get('result', {}).get('list', []):
                # Bybit returns: [startTime, open, high, low, close, volume, turnover]
                klines.append({
                    'timestamp': datetime.fromtimestamp(int(k[0]) / 1000, tz=timezone.utc),
                    'open': Decimal(k[1]),
                    'high': Decimal(k[2]),
                    'low': Decimal(k[3]),
                    'close': Decimal(k[4]),
                    'volume': Decimal(k[5]),
                })
            
            # Bybit returns newest first, reverse for chronological order
            return list(reversed(klines))
            
        except Exception as e:
            logger.exception(f"Error fetching klines: {e}")
            return []
