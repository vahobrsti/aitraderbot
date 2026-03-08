"""
Risk management service.
Validates execution intents against account limits and market conditions.
"""
import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional
from datetime import date, timedelta
from django.db.models import Sum
from django.utils import timezone

logger = logging.getLogger(__name__)


@dataclass
class RiskCheckResult:
    """Result of risk validation."""
    passed: bool
    reason: str = ''
    adjusted_qty: Optional[Decimal] = None
    adjusted_notional: Optional[Decimal] = None


class RiskManager:
    """
    Validates execution intents against risk limits.
    Checks: position limits, daily loss limits, duplicate prevention.
    """
    
    def __init__(self, account):
        self.account = account
    
    def check_intent(self, intent) -> RiskCheckResult:
        """
        Run all risk checks on an execution intent.
        Returns RiskCheckResult with pass/fail and reason.
        """
        checks = [
            self._check_account_active,
            self._check_duplicate_intent,
            self._check_position_limit,
            self._check_daily_loss_limit,
            self._check_open_positions,
        ]
        
        for check in checks:
            result = check(intent)
            if not result.passed:
                logger.warning(f"Risk check failed: {result.reason}")
                return result
        
        return RiskCheckResult(passed=True, reason='All checks passed')
    
    def _check_account_active(self, intent) -> RiskCheckResult:
        """Verify account is active."""
        if not self.account.is_active:
            return RiskCheckResult(
                passed=False,
                reason=f"Account {self.account.name} is not active"
            )
        return RiskCheckResult(passed=True)
    
    def _check_duplicate_intent(self, intent) -> RiskCheckResult:
        """Check for duplicate execution intents."""
        from execution.models import ExecutionIntent
        
        existing = ExecutionIntent.objects.filter(
            idempotency_key=intent.idempotency_key
        ).exclude(pk=intent.pk).exclude(
            status__in=['cancelled', 'failed', 'rejected']
        ).exists()
        
        if existing:
            return RiskCheckResult(
                passed=False,
                reason=f"Duplicate intent exists for {intent.idempotency_key}"
            )
        return RiskCheckResult(passed=True)
    
    def _check_position_limit(self, intent) -> RiskCheckResult:
        """Check if intent exceeds position size limit."""
        target_notional = intent.target_notional_usd or Decimal('0')
        max_position = self.account.max_position_usd
        
        if target_notional > max_position:
            # Adjust down to limit
            return RiskCheckResult(
                passed=True,
                reason=f"Adjusted notional from {target_notional} to {max_position}",
                adjusted_notional=max_position,
            )
        return RiskCheckResult(passed=True)
    
    def _check_daily_loss_limit(self, intent) -> RiskCheckResult:
        """Check if daily loss limit has been reached."""
        from execution.models import ExecutionIntent, Position
        
        today = date.today()
        
        # Sum realized losses from today's closed intents
        daily_loss = ExecutionIntent.objects.filter(
            account=self.account,
            completed_at__date=today,
            status='filled',
        ).aggregate(
            total_loss=Sum('positions__realized_pnl')
        )['total_loss'] or Decimal('0')
        
        # Add unrealized losses from open positions
        unrealized = Position.objects.filter(
            account=self.account,
            qty__gt=0,
        ).aggregate(
            total=Sum('unrealized_pnl')
        )['total'] or Decimal('0')
        
        total_loss = min(daily_loss, Decimal('0')) + min(unrealized, Decimal('0'))
        
        if abs(total_loss) >= self.account.max_daily_loss_usd:
            return RiskCheckResult(
                passed=False,
                reason=f"Daily loss limit reached: {abs(total_loss)} >= {self.account.max_daily_loss_usd}"
            )
        return RiskCheckResult(passed=True)
    
    def _check_open_positions(self, intent) -> RiskCheckResult:
        """Check for conflicting open positions."""
        from execution.models import Position
        
        # Check if we already have a position in the opposite direction
        opposite_side = 'short' if intent.direction == 'long' else 'long'
        
        conflicting = Position.objects.filter(
            account=self.account,
            side=opposite_side,
            qty__gt=0,
        ).exists()
        
        if conflicting:
            return RiskCheckResult(
                passed=False,
                reason=f"Conflicting {opposite_side} position exists"
            )
        return RiskCheckResult(passed=True)
    
    def calculate_position_size(
        self,
        intent,
        current_price: Decimal,
        volatility: Optional[Decimal] = None
    ) -> Decimal:
        """
        Calculate appropriate position size based on risk parameters.
        Uses Kelly criterion or fixed fractional sizing.
        """
        max_notional = min(
            intent.target_notional_usd or self.account.max_position_usd,
            self.account.max_position_usd
        )
        
        # Apply size multiplier from signal if available
        if hasattr(intent.signal, 'size_multiplier'):
            max_notional *= Decimal(str(intent.signal.size_multiplier))
        
        # Convert to quantity
        if current_price > 0:
            qty = max_notional / current_price
        else:
            qty = Decimal('0')
        
        return qty.quantize(Decimal('0.0001'))
