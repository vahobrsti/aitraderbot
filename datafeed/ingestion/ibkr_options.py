"""
IBKR (Interactive Brokers) IBIT Options Data Fetcher.

Collects option snapshots for IBIT (iShares Bitcoin ETF) via the IBKR API,
mirroring the DeribitOptionsFetcher interface so snapshots land in the same
OptionSnapshot table (tagged exchange='ibkr').

Client library: ib_async (the maintained successor to the archived ib_insync).
It is imported lazily inside connect() so importing this module never requires
ib_async to be installed — the pure parsing/formatting helpers stay usable and
unit-testable without a live IB Gateway/TWS connection.

Runtime requirement: a running IB Gateway or TWS reachable at
IBKR_HOST:IBKR_PORT with client id IBKR_CLIENT_ID. Market data is read-only
(no order permissions needed). Live option greeks require market-data
subscriptions for both IBIT and its options.

Environment variables:
    IBKR_HOST        default 127.0.0.1
    IBKR_PORT        default 4002 (Gateway paper); 4001 live, 7497 TWS paper
    IBKR_CLIENT_ID   default 7

Usage:
    from datafeed.ingestion.ibkr_options import IBKROptionsFetcher

    fetcher = IBKROptionsFetcher()
    snapshots = fetcher.fetch_option_chain(underlying='IBIT', dte_min=7, dte_max=45)
"""
import logging
import os
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from .options_base import OptionsFetcherBase

logger = logging.getLogger(__name__)

# Standard US equity-option contract multiplier (confirmed for IBIT via IBKR).
IBIT_MULTIPLIER = 100

# IBKR market-data type codes -> label. reqMarketDataType / ticker.marketDataType.
MARKET_DATA_TYPE = {
    1: "live",
    2: "frozen",
    3: "delayed",
    4: "delayed-frozen",
}


class IBKROptionsFetcher(OptionsFetcherBase):
    """Fetches IBIT option market data from IBKR via ib_async."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        client_id: Optional[int] = None,
        exchange: str = "SMART",
        trading_class: str = "IBIT",
        connect_timeout: float = 15.0,
    ):
        self.host = host or os.environ.get("IBKR_HOST", "127.0.0.1")
        self.port = int(port or os.environ.get("IBKR_PORT", 4002))
        self.client_id = int(client_id or os.environ.get("IBKR_CLIENT_ID", 7))
        self.exchange = exchange
        self.trading_class = trading_class
        self.connect_timeout = connect_timeout
        self._ib = None

    @property
    def exchange_name(self) -> str:
        return "ibkr"

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------
    def connect(self):
        """Establish a read-only connection to IB Gateway/TWS."""
        if self._ib is not None and getattr(self._ib, "isConnected", lambda: False)():
            return self._ib

        try:
            from ib_async import IB
        except ImportError as e:
            raise ImportError(
                "ib_async is required for live IBKR data collection. "
                "Install with `pip install ib_async`."
            ) from e

        ib = IB()
        ib.connect(
            self.host,
            self.port,
            clientId=self.client_id,
            readonly=True,
            timeout=self.connect_timeout,
        )
        self._ib = ib
        logger.info(
            "Connected to IBKR at %s:%s (clientId=%s)",
            self.host, self.port, self.client_id,
        )
        return ib

    def disconnect(self):
        if self._ib is not None:
            try:
                self._ib.disconnect()
            finally:
                self._ib = None

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------
    def _underlying_contract(self):
        from ib_async import Stock
        stock = Stock(self.trading_class, self.exchange, "USD")
        self._ib.qualifyContracts(stock)
        return stock

    def get_spot_price(self, symbol: str = "IBIT") -> Optional[Decimal]:
        """Get IBIT spot (last/close) price."""
        self.connect()
        stock = self._underlying_contract()
        ticker = self._ib.reqMktData(stock, "", snapshot=True)
        self._ib.sleep(2)
        price = _first_valid(ticker.last, ticker.close, ticker.marketPrice())
        return self.safe_decimal(price)

    def fetch_option_chain(
        self,
        underlying: str = "IBIT",
        dte_min: int = 7,
        dte_max: int = 45,
        moneyness_range: tuple = (-0.20, 0.20),
    ) -> list[dict]:
        """
        Discover the IBIT chain, filter by DTE and moneyness, then request market
        data only for the surviving contracts.

        Discover-then-filter is deliberate: IBKR per-contract market-data requests
        are slow and line-limited, so we never request data for the full chain.
        """
        self.connect()
        now = datetime.now(timezone.utc)

        stock = self._underlying_contract()
        spot = self.get_spot_price(underlying)
        if not spot:
            logger.error("Could not fetch IBIT spot price")
            return []
        spot_f = float(spot)

        params = self._ib.reqSecDefOptParams(
            stock.symbol, "", stock.secType, stock.conId
        )
        chain = _select_chain(params, self.exchange, self.trading_class)
        if chain is None:
            logger.error("No matching IBIT option parameters for exchange=%s class=%s",
                         self.exchange, self.trading_class)
            return []

        lo = spot_f * (1.0 + moneyness_range[0])
        hi = spot_f * (1.0 + moneyness_range[1])
        expiries = _filter_expiries(chain.expirations, now, dte_min, dte_max)
        strikes = [s for s in chain.strikes if lo <= s <= hi]

        if not expiries or not strikes:
            logger.warning(
                "IBIT chain filter produced no contracts (expiries=%d, strikes=%d)",
                len(expiries), len(strikes),
            )
            return []

        from ib_async import Option
        contracts = []
        for expiry in expiries:
            for strike in strikes:
                for right in ("P", "C"):
                    contracts.append(Option(
                        self.trading_class, expiry, strike, right, self.exchange,
                        tradingClass=self.trading_class, multiplier=str(IBIT_MULTIPLIER),
                    ))
        qualified = self._ib.qualifyContracts(*contracts)
        logger.info("Requesting market data for %d IBIT contracts", len(qualified))

        # Model greeks + open interest generic ticks (100=option volume,
        # 101=option open interest). Streaming request, read after a short wait,
        # then cancel — snapshots don't carry generic ticks.
        tickers = [self._ib.reqMktData(c, "100,101", snapshot=False) for c in qualified]
        self._ib.sleep(4)

        snapshots = []
        for c, t in zip(qualified, tickers):
            snap = self._ticker_to_snapshot(c, t, now, spot)
            self._ib.cancelMktData(c)
            if snap is not None:
                snapshots.append(snap)

        logger.info("Collected %d IBIT option snapshots", len(snapshots))
        return snapshots

    def fetch_single_option(self, symbol: str) -> Optional[dict]:
        """Not required for the publish pipeline; chain fetch is the entry point."""
        raise NotImplementedError(
            "IBKROptionsFetcher fetches by chain; single-symbol fetch is not supported."
        )

    def _ticker_to_snapshot(self, contract, ticker, timestamp, spot_price) -> Optional[dict]:
        """Adapt a live ib_async Ticker into a standard snapshot dict."""
        greeks = getattr(ticker, "modelGreeks", None)
        md_type = MARKET_DATA_TYPE.get(getattr(ticker, "marketDataType", None))
        return build_snapshot(
            symbol=format_ibit_symbol(contract.lastTradeDateOrContractMonth,
                                      contract.strike, contract.right),
            underlying="IBIT",
            expiry=_parse_ib_expiry(contract.lastTradeDateOrContractMonth),
            strike=contract.strike,
            right=contract.right,
            spot_price=spot_price,
            timestamp=timestamp,
            bid=_pos(getattr(ticker, "bid", None)),
            ask=_pos(getattr(ticker, "ask", None)),
            last=_pos(getattr(ticker, "last", None)),
            mark=_pos(getattr(ticker, "markPrice", None)),
            iv=getattr(greeks, "impliedVol", None) if greeks else None,
            delta=getattr(greeks, "delta", None) if greeks else None,
            gamma=getattr(greeks, "gamma", None) if greeks else None,
            vega=getattr(greeks, "vega", None) if greeks else None,
            theta=getattr(greeks, "theta", None) if greeks else None,
            bid_size=getattr(ticker, "bidSize", None),
            ask_size=getattr(ticker, "askSize", None),
            volume=getattr(ticker, "volume", None),
            open_interest=_option_oi(ticker, contract.right),
            market_data_type=md_type,
        )


# ======================================================================
# Pure helpers (no ib_async dependency — unit-testable)
# ======================================================================

def _first_valid(*vals):
    """Return the first value that is a real, positive number."""
    for v in vals:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f == f and f > 0:  # not NaN, positive
            return f
    return None


def _pos(v):
    """Coerce to float if it's a valid positive number, else None (IB uses NaN/-1)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f < 0:  # NaN or negative sentinel
        return None
    return f


def _option_oi(ticker, right: str):
    """Pick the open interest matching the option right."""
    if right == "C":
        return getattr(ticker, "callOpenInterest", None)
    return getattr(ticker, "putOpenInterest", None)


def _parse_ib_expiry(yyyymmdd: str) -> datetime:
    """IB expiry 'YYYYMMDD' -> tz-aware UTC datetime at 00:00."""
    return datetime.strptime(yyyymmdd, "%Y%m%d").replace(tzinfo=timezone.utc)


def format_ibit_symbol(yyyymmdd: str, strike: float, right: str) -> str:
    """Readable IBIT option symbol, e.g. IBIT-18SEP26-35-P."""
    exp = datetime.strptime(yyyymmdd, "%Y%m%d")
    strike_str = f"{strike:g}"
    return f"IBIT-{exp.strftime('%d%b%y').upper()}-{strike_str}-{right}"


def _select_chain(params, exchange: str, trading_class: str):
    """Pick the option-parameter set matching our exchange and trading class."""
    if not params:
        return None
    for p in params:
        if p.exchange == exchange and p.tradingClass == trading_class:
            return p
    # Fall back to the trading-class match on any exchange.
    for p in params:
        if p.tradingClass == trading_class:
            return p
    return params[0]


def _filter_expiries(expirations, now: datetime, dte_min: int, dte_max: int) -> list[str]:
    """Keep expiries whose DTE (from now) is within [dte_min, dte_max]."""
    kept = []
    for exp in sorted(expirations):
        try:
            exp_dt = _parse_ib_expiry(exp)
        except ValueError:
            continue
        dte = (exp_dt - now).total_seconds() / 86400
        if dte_min <= dte <= dte_max:
            kept.append(exp)
    return kept


def build_snapshot(
    symbol: str,
    underlying: str,
    expiry: datetime,
    strike: float,
    right: str,
    spot_price,
    timestamp: datetime,
    bid=None,
    ask=None,
    last=None,
    mark=None,
    iv=None,
    delta=None,
    gamma=None,
    vega=None,
    theta=None,
    bid_size=None,
    ask_size=None,
    volume=None,
    open_interest=None,
    market_data_type: Optional[str] = None,
) -> dict:
    """
    Build a standardized snapshot dict in the same shape the collector persists,
    matching DeribitOptionsFetcher output. USD-quoted (no BTC->USD conversion).
    """
    def _dec(v):
        if v is None:
            return None
        try:
            return Decimal(str(v))
        except Exception:
            return None

    strike_d = _dec(strike)
    spot_d = _dec(spot_price)
    bid_d, ask_d = _dec(bid), _dec(ask)

    mid = None
    if bid_d is not None and ask_d is not None:
        mid = (bid_d + ask_d) / 2

    spread_pct = None
    if bid_d is not None and ask_d is not None and mid and mid > 0:
        spread_pct = float((ask_d - bid_d) / mid)

    dte = (expiry - timestamp).total_seconds() / 86400 if expiry and timestamp else None
    moneyness = None
    if strike_d is not None and spot_d and spot_d > 0:
        moneyness = float((strike_d - spot_d) / spot_d)

    return {
        "timestamp": timestamp,
        "symbol": symbol,
        "underlying": underlying,
        "expiry": expiry,
        "strike": strike_d,
        "option_type": "call" if right == "C" else "put",
        "spot_price": spot_d,
        "index_price": spot_d,
        "bid": bid_d,
        "ask": ask_d,
        "mid_price": mid,
        "mark_price": _dec(mark),
        "last_price": _dec(last),
        "iv": _dec(iv),
        "delta": _dec(delta),
        "gamma": _dec(gamma),
        "vega": _dec(vega),
        "theta": _dec(theta),
        "bid_size": _dec(bid_size),
        "ask_size": _dec(ask_size),
        "volume_24h": _dec(volume),
        "open_interest": _dec(open_interest),
        "dte": dte,
        "moneyness": moneyness,
        "spread_pct": spread_pct,
        "exchange": "ibkr",
        "market_data_type": market_data_type,
    }
