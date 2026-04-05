"""
Instrument selector service.
Handles strike selection (ATM/ITM/OTM) and expiry selection based on signal guidance.
"""
import logging
from dataclasses import dataclass
from decimal import Decimal
from datetime import date, timedelta
from typing import Optional

from execution.exchanges.base import ExchangeAdapter, InstrumentInfo

logger = logging.getLogger(__name__)


@dataclass
class StrikeSelection:
    """Result of strike selection."""
    symbol: str
    strike: Decimal
    expiry: date
    option_type: str  # 'call' or 'put'
    moneyness: str  # 'atm', 'itm', 'otm'
    dte: int
    rationale: str


@dataclass
class SpreadSelection:
    """Result of spread selection (two legs)."""
    long_leg: StrikeSelection
    short_leg: StrikeSelection
    spread_type: str  # 'call_spread', 'put_spread'
    width_pct: Decimal
    max_profit: Optional[Decimal] = None
    max_loss: Optional[Decimal] = None


class InstrumentSelector:
    """
    Selects optimal instruments based on signal guidance.
    Handles: ATM/ITM/OTM selection, DTE targeting, spread construction.
    """
    
    def __init__(self, adapter: ExchangeAdapter):
        self.adapter = adapter
    
    def select_single_option(
        self,
        underlying: str,
        option_type: str,
        strike_guidance: str,
        dte_range: str,
        underlying_price: Decimal,
    ) -> Optional[StrikeSelection]:
        """
        Select a single option based on signal guidance.
        
        Args:
            underlying: e.g., 'BTC'
            option_type: 'call' or 'put'
            strike_guidance: 'atm', 'slight_otm', 'otm', 'itm'
            dte_range: e.g., '45-90d' or '30-60d'
            underlying_price: Current price of underlying
        """
        # Parse DTE range
        target_dte, max_dte = self._parse_dte_range(dte_range)
        
        # Get available options
        instruments = self.adapter.get_instruments(
            instrument_type='option',
            underlying=underlying
        )
        
        if not instruments:
            logger.warning(f"No options available for {underlying}")
            return None
        
        # Filter by option type
        options = [i for i in instruments if i.option_type == option_type]
        if not options:
            logger.warning(f"No {option_type} options available")
            return None
        
        # Filter by expiry (within DTE range)
        today = date.today()
        min_expiry = today + timedelta(days=target_dte)
        max_expiry = today + timedelta(days=max_dte)
        
        valid_options = []
        for opt in options:
            if not opt.expiry:
                continue
            exp_date = opt.expiry.date() if hasattr(opt.expiry, 'date') else opt.expiry
            if min_expiry <= exp_date <= max_expiry:
                valid_options.append(opt)
        
        if not valid_options:
            # Fallback: find closest to target DTE
            logger.info(f"No options in DTE range {target_dte}-{max_dte}, finding closest")
            valid_options = [o for o in options if o.expiry]
            if not valid_options:
                return None
        
        # Select strike based on guidance
        target_strike = self._calculate_target_strike(
            underlying_price, option_type, strike_guidance
        )
        
        # Find best match
        best = self._find_best_option(valid_options, target_strike, target_dte, today)
        if not best:
            return None
        
        exp_date = best.expiry.date() if hasattr(best.expiry, 'date') else best.expiry
        dte = (exp_date - today).days
        
        return StrikeSelection(
            symbol=best.symbol,
            strike=best.strike or Decimal('0'),
            expiry=exp_date,
            option_type=option_type,
            moneyness=self._classify_moneyness(best.strike, underlying_price, option_type),
            dte=dte,
            rationale=f"Selected {strike_guidance} {option_type} with {dte} DTE",
        )
    
    def select_spread(
        self,
        underlying: str,
        spread_type: str,
        strike_guidance: str,
        dte_range: str,
        width_pct: Decimal,
        underlying_price: Decimal,
    ) -> Optional[SpreadSelection]:
        """
        Select a vertical spread (call spread or put spread).
        
        Args:
            spread_type: 'call_spread' or 'put_spread'
            width_pct: Spread width as percentage (e.g., 0.10 = 10%)
        """
        option_type = 'call' if spread_type == 'call_spread' else 'put'
        
        # Get long leg first
        long_leg = self.select_single_option(
            underlying, option_type, strike_guidance, dte_range, underlying_price
        )
        if not long_leg:
            return None
        
        # Calculate short leg strike
        width_amount = underlying_price * width_pct
        
        if spread_type == 'call_spread':
            # Bull call spread: buy lower strike, sell higher strike
            short_strike = long_leg.strike + width_amount
        else:
            # Bear put spread: buy higher strike, sell lower strike
            short_strike = long_leg.strike - width_amount
        
        # Find short leg option
        instruments = self.adapter.get_instruments(
            instrument_type='option',
            underlying=underlying
        )
        
        # Filter for same expiry and option type
        candidates = [
            i for i in instruments
            if i.option_type == option_type
            and i.expiry
            and (i.expiry.date() if hasattr(i.expiry, 'date') else i.expiry) == long_leg.expiry
        ]
        
        if not candidates:
            logger.warning("No candidates for short leg")
            return None
        
        # Find closest to target short strike
        short_opt = min(
            candidates,
            key=lambda o: abs((o.strike or Decimal('0')) - short_strike)
        )
        
        short_leg = StrikeSelection(
            symbol=short_opt.symbol,
            strike=short_opt.strike or Decimal('0'),
            expiry=long_leg.expiry,
            option_type=option_type,
            moneyness=self._classify_moneyness(short_opt.strike, underlying_price, option_type),
            dte=long_leg.dte,
            rationale=f"Short leg for {spread_type}",
        )
        
        # Calculate max profit/loss
        actual_width = abs(long_leg.strike - short_leg.strike)
        
        return SpreadSelection(
            long_leg=long_leg,
            short_leg=short_leg,
            spread_type=spread_type,
            width_pct=actual_width / underlying_price if underlying_price else Decimal('0'),
        )
    
    def _parse_dte_range(self, dte_range: str) -> tuple[int, int]:
        """Parse DTE range string like '45-90d' into (min, max)."""
        if not dte_range:
            return 45, 90  # Default
        
        try:
            cleaned = dte_range.lower().replace('d', '').strip()
            if '-' in cleaned:
                parts = cleaned.split('-')
                return int(parts[0]), int(parts[1])
            else:
                dte = int(cleaned)
                return dte, dte + 30
        except (ValueError, IndexError):
            return 45, 90
    
    def _calculate_target_strike(
        self,
        underlying_price: Decimal,
        option_type: str,
        guidance: str,
    ) -> Decimal:
        """Calculate target strike based on guidance."""
        if not guidance or guidance == 'atm':
            return underlying_price
        
        # OTM/ITM offsets
        offset_pct = Decimal('0.05')  # 5% default
        if 'slight' in guidance:
            offset_pct = Decimal('0.02')  # 2% for slight
        
        if guidance in ('otm', 'slight_otm'):
            if option_type == 'call':
                return underlying_price * (1 + offset_pct)
            else:
                return underlying_price * (1 - offset_pct)
        elif guidance in ('itm', 'slight_itm'):
            if option_type == 'call':
                return underlying_price * (1 - offset_pct)
            else:
                return underlying_price * (1 + offset_pct)
        
        return underlying_price
    
    def _find_best_option(
        self,
        options: list[InstrumentInfo],
        target_strike: Decimal,
        target_dte: int,
        today: date,
    ) -> Optional[InstrumentInfo]:
        """Find best option balancing strike and DTE targets."""
        if not options:
            return None
        
        def score(opt: InstrumentInfo) -> float:
            strike_diff = abs((opt.strike or Decimal('0')) - target_strike)
            strike_score = float(strike_diff / target_strike) if target_strike else 0
            
            exp_date = opt.expiry.date() if hasattr(opt.expiry, 'date') else opt.expiry
            dte = (exp_date - today).days
            dte_score = abs(dte - target_dte) / 30  # Normalize by ~1 month
            
            # Weight strike more heavily than DTE
            return strike_score * 2 + dte_score
        
        return min(options, key=score)
    
    def _classify_moneyness(
        self,
        strike: Optional[Decimal],
        underlying_price: Decimal,
        option_type: str,
    ) -> str:
        """Classify option as ATM/ITM/OTM."""
        if not strike or not underlying_price:
            return 'unknown'
        
        pct_diff = (strike - underlying_price) / underlying_price
        
        if abs(pct_diff) < Decimal('0.02'):
            return 'atm'
        
        if option_type == 'call':
            return 'otm' if pct_diff > 0 else 'itm'
        else:
            return 'itm' if pct_diff > 0 else 'otm'
