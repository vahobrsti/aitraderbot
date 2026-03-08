"""
Management command to check for unprotected positions and alert.
Run frequently (every minute) to catch positions without SL/TP.

Exit code 1 if unprotected positions found (for monitoring integration).
"""
import logging
from datetime import timedelta
from django.core.management.base import BaseCommand
from django.utils import timezone

from execution.models import ExchangeAccount, ExecutionIntent, Position

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Check for unprotected positions (no SL/TP) and alert'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--account',
            type=str,
            help='Account name to check (default: all active accounts)',
        )
        parser.add_argument(
            '--max-age-minutes',
            type=int,
            default=5,
            help='Alert if unprotected for longer than this (default: 5)',
        )
    
    def handle(self, *args, **options):
        account_name = options.get('account')
        max_age = options.get('max_age_minutes', 5)
        
        accounts = ExchangeAccount.objects.filter(is_active=True)
        if account_name:
            accounts = accounts.filter(name=account_name)
        
        cutoff = timezone.now() - timedelta(minutes=max_age)
        
        # Find unprotected intents
        unprotected = ExecutionIntent.objects.filter(
            account__in=accounts,
            status__in=['entry_filled', 'unprotected'],
            protected_at__isnull=True,
            completed_at__lt=cutoff,  # Entry filled more than max_age ago
        )
        
        if not unprotected.exists():
            self.stdout.write(self.style.SUCCESS("All positions protected"))
            return
        
        # Alert!
        self.stdout.write(self.style.ERROR(
            f"ALERT: {unprotected.count()} UNPROTECTED POSITION(S)"
        ))
        
        for intent in unprotected:
            age_minutes = (timezone.now() - intent.completed_at).total_seconds() / 60
            self.stdout.write(self.style.ERROR(
                f"  - {intent.target_symbol} ({intent.direction}) "
                f"unprotected for {age_minutes:.0f} minutes"
            ))
            
            # Check if position still exists
            position = Position.objects.filter(
                account=intent.account,
                symbol=intent.target_symbol,
                qty__gt=0,
            ).first()
            
            if position:
                self.stdout.write(f"    Position: {position.qty} @ {position.entry_price}")
                self.stdout.write(f"    Mark: {position.mark_price}, PnL: {position.unrealized_pnl}")
            else:
                self.stdout.write("    Position may have been closed")
        
        # Exit with error code for monitoring
        exit(1)
