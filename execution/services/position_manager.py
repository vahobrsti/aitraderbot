"""
Position manager service.
Monitors positions and determines when exit rules trigger.
"""
import logging
from dataclasses import dataclass
from decimal import Decimal
from datetime import date, timedelta
from typing import Optional
from enum import Enum

from django.utils import timezone

from execution.models import Position, ExecutionIntent

logger = logging.getLogger(__name__)


class ExitReason(Enum):
    """Reasons for exiting a position."""
    STOP_LOSS = 'stop_loss'
    TAKE_PROFIT = 'take_profit'
    SCALE_DOWN = 'scale_down'
    TIME_STOP = 'time_stop'
    EXPIRY_APPROACHING = 'expiry_approaching'
    MANUAL = 'manual'
    SIGNAL_REVERSAL = 'signal_reversal'


@dataclass
class ExitSignal:
    """Signal to exit or reduce a position."""
    position: Position
    reason: ExitReason
    action: str  # 'close', 'scale_down'
    close_pct: Decimal  # 1.0 = full close, 0.75 = close 75%
    urgency: str  # 'immediate', 'end_of_day', 'next_session'
    notes: str


class PositionManager:
    """
    Monitors positions and checks exit rules.
    Rules checked:
    - Stop loss (price-based)
    - Take profit (price-based)
    - Scale down (time-based)
    - Time stop (max hold days)
    - Expiry approaching (for options)
    """
    
    def __init__(self, min_dte_before_close: int = 3):
        """
        Args:
            min_dte_before_close: Close options this many days before expiry
        """
        self.min_dte_before_close = min_dte_before_close
    
    def check_position(self, position: Position) -> Optional[ExitSignal]:
        """
        Check all exit rules for a position.
        Returns ExitSignal if any rule triggers, None otherwise.
        """
        # Get the intent that opened this position (if any)
        intent = self._find_opening_intent(position)
        
        # Check rules in priority order
        checks = [
            self._check_stop_loss,
            self._check_take_profit,
            self._check_time_stop,
            self._check_scale_down,
            self._check_expiry,
        ]
        
        for check in checks:
            signal = check(position, intent)
            if signal:
                return signal
        
        return None
    
    def check_all_positions(self, account) -> list[ExitSignal]:
        """Check all open positions for an account."""
        positions = Position.objects.filter(
            account=account,
            qty__gt=0,
        )
        
        signals = []
        for pos in positions:
            signal = self.check_position(pos)
            if signal:
                signals.append(signal)
        
        return signals
    
    def _find_opening_intent(self, position: Position) -> Optional[ExecutionIntent]:
        """Find the ExecutionIntent that opened this position."""
        # Look for filled intent with matching symbol
        intent = ExecutionIntent.objects.filter(
            account=position.account,
            target_symbol=position.symbol,
            status='filled',
        ).order_by('-completed_at').first()
        
        return intent
    
    def _check_stop_loss(
        self, position: Position, intent: Optional[ExecutionIntent]
    ) -> Optional[ExitSignal]:
        """Check if stop loss has been hit."""
        if not intent or not intent.stop_loss_pct:
            return None
        
        if not position.entry_price or not position.mark_price:
            return None
        
        stop_pct = intent.stop_loss_pct
        
        if position.side == 'long':
            # Long position: stop if price dropped by stop_pct
            stop_price = position.entry_price * (1 - stop_pct)
            triggered = position.mark_price <= stop_price
        else:
            # Short position: stop if price rose by stop_pct
            stop_price = position.entry_price * (1 + stop_pct)
            triggered = position.mark_price >= stop_price
        
        if triggered:
            pnl_pct = self._calculate_pnl_pct(position)
            return ExitSignal(
                position=position,
                reason=ExitReason.STOP_LOSS,
                action='close',
                close_pct=Decimal('1.0'),
                urgency='immediate',
                notes=f"Stop loss triggered at {pnl_pct:.1%} loss (threshold: {stop_pct:.1%})",
            )
        
        return None
    
    def _check_take_profit(
        self, position: Position, intent: Optional[ExecutionIntent]
    ) -> Optional[ExitSignal]:
        """Check if take profit has been hit."""
        if not intent or not intent.take_profit_pct:
            return None
        
        if not position.entry_price or not position.mark_price:
            return None
        
        tp_pct = intent.take_profit_pct
        pnl_pct = self._calculate_pnl_pct(position)
        
        if pnl_pct >= tp_pct:
            return ExitSignal(
                position=position,
                reason=ExitReason.TAKE_PROFIT,
                action='close',
                close_pct=Decimal('1.0'),
                urgency='immediate',
                notes=f"Take profit triggered at {pnl_pct:.1%} gain (threshold: {tp_pct:.1%})",
            )
        
        return None
    
    def _check_time_stop(
        self, position: Position, intent: Optional[ExecutionIntent]
    ) -> Optional[ExitSignal]:
        """Check if max hold days exceeded."""
        if not intent or not intent.max_hold_days:
            return None
        
        if not intent.completed_at:
            return None
        
        entry_date = intent.completed_at.date()
        days_held = (date.today() - entry_date).days
        
        if days_held >= intent.max_hold_days:
            return ExitSignal(
                position=position,
                reason=ExitReason.TIME_STOP,
                action='close',
                close_pct=Decimal('1.0'),
                urgency='end_of_day',
                notes=f"Time stop: held {days_held} days (max: {intent.max_hold_days})",
            )
        
        return None
    
    def _check_scale_down(
        self, position: Position, intent: Optional[ExecutionIntent]
    ) -> Optional[ExitSignal]:
        """Check if scale-down day reached."""
        if not intent:
            return None
        
        # Get scale_down_day from signal if available
        scale_down_day = None
        if intent.signal and hasattr(intent.signal, 'scale_down_day'):
            scale_down_day = intent.signal.scale_down_day
        
        if not scale_down_day or not intent.completed_at:
            return None
        
        entry_date = intent.completed_at.date()
        days_held = (date.today() - entry_date).days
        
        # Check if we've reached scale-down day but haven't scaled yet
        # We track this by checking if position is still at full size
        if days_held >= scale_down_day:
            # Check if already scaled (would need to track this)
            # For now, trigger scale-down if at full size
            return ExitSignal(
                position=position,
                reason=ExitReason.SCALE_DOWN,
                action='scale_down',
                close_pct=Decimal('0.75'),  # Close 75%, keep 25%
                urgency='end_of_day',
                notes=f"Scale-down day {scale_down_day} reached (held {days_held} days)",
            )
        
        return None
    
    def _check_expiry(
        self, position: Position, intent: Optional[ExecutionIntent]
    ) -> Optional[ExitSignal]:
        """Check if option expiry is approaching."""
        if not position.expiry:
            return None
        
        days_to_expiry = (position.expiry - date.today()).days
        
        if days_to_expiry <= self.min_dte_before_close:
            return ExitSignal(
                position=position,
                reason=ExitReason.EXPIRY_APPROACHING,
                action='close',
                close_pct=Decimal('1.0'),
                urgency='immediate' if days_to_expiry <= 1 else 'end_of_day',
                notes=f"Option expires in {days_to_expiry} days",
            )
        
        return None
    
    def _calculate_pnl_pct(self, position: Position) -> Decimal:
        """Calculate P&L percentage for position."""
        if not position.entry_price or not position.mark_price:
            return Decimal('0')
        
        if position.entry_price == 0:
            return Decimal('0')
        
        if position.side == 'long':
            return (position.mark_price - position.entry_price) / position.entry_price
        else:
            return (position.entry_price - position.mark_price) / position.entry_price
