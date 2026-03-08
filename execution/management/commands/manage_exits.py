"""
Management command to check and execute exit rules for open positions.
This is the FALLBACK/GUARDIAN for positions using polling-based exits.

Run every 5 minutes via cron:
  */5 * * * * python manage.py manage_exits

Checks:
- Stop loss (price-based)
- Take profit (price-based)  
- Time stop (max_hold_days)
- Scale down (scale_down_day)
- Expiry approaching (DTE <= 3)
"""
import logging
from decimal import Decimal
from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils import timezone

from execution.models import ExchangeAccount, Position, Order, ExecutionIntent, ExecutionEvent
from execution.services.orchestrator import ExecutionOrchestrator
from execution.services.position_manager import PositionManager, ExitSignal, ExitReason
from execution.services.order_builder import OrderBuilder
from execution.exchanges.base import OrderRequest

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Check exit rules and close positions (polling-based exit management)'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--account',
            type=str,
            help='Account name to check (default: all active accounts)',
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be done without executing',
        )
        parser.add_argument(
            '--reason',
            type=str,
            choices=['stop_loss', 'take_profit', 'time_stop', 'scale_down', 'expiry'],
            help='Only process specific exit reason',
        )
    
    def handle(self, *args, **options):
        dry_run = options.get('dry_run', False)
        account_name = options.get('account')
        reason_filter = options.get('reason')
        
        if dry_run:
            self.stdout.write(self.style.WARNING('DRY RUN - no orders will be placed'))
        
        # Get accounts to check
        accounts = ExchangeAccount.objects.filter(is_active=True)
        if account_name:
            accounts = accounts.filter(name=account_name)
        
        if not accounts.exists():
            self.stdout.write(self.style.ERROR('No active accounts found'))
            return
        
        total_exits = 0
        total_errors = 0
        
        for account in accounts:
            self.stdout.write(f"\nChecking account: {account.name} ({account.exchange})")
            
            try:
                exits, errors = self._process_account(
                    account, dry_run, reason_filter
                )
                total_exits += exits
                total_errors += errors
            except Exception as e:
                logger.exception(f"Error processing account {account.name}: {e}")
                self.stdout.write(self.style.ERROR(f"  Error: {e}"))
                total_errors += 1
        
        # Summary
        self.stdout.write('')
        if total_exits > 0:
            self.stdout.write(self.style.SUCCESS(f"Processed {total_exits} exit(s)"))
        else:
            self.stdout.write("No exits triggered")
        
        if total_errors > 0:
            self.stdout.write(self.style.ERROR(f"{total_errors} error(s) occurred"))
    
    def _process_account(
        self, account: ExchangeAccount, dry_run: bool, reason_filter: str
    ) -> tuple[int, int]:
        """Process exit rules for one account."""
        exits = 0
        errors = 0
        
        # Initialize services
        orchestrator = ExecutionOrchestrator(account)
        position_manager = PositionManager()
        order_builder = OrderBuilder(account.max_position_usd)
        
        # Sync positions first
        self.stdout.write("  Syncing positions...")
        orchestrator.sync_positions()
        
        # Get intents that need polling-based exit management
        # These are positions with exit_method='polling' or legacy 'protected' status
        polling_intents = ExecutionIntent.objects.filter(
            account=account,
            status__in=['protected', 'entry_filled', 'unprotected'],
        ).exclude(
            exit_method='native'  # Skip native exits - exchange handles them
        )
        
        self.stdout.write(f"  Found {polling_intents.count()} position(s) to monitor")
        
        # Check each position
        exit_signals = []
        for intent in polling_intents:
            position = Position.objects.filter(
                account=account,
                symbol=intent.target_symbol,
                qty__gt=0,
            ).first()
            
            if not position:
                # Position closed externally
                if intent.status != 'closed':
                    intent.status = 'closed'
                    intent.status_reason = 'Position closed externally'
                    intent.save()
                    self.stdout.write(f"  {intent.target_symbol}: Position closed externally")
                continue
            
            signal = position_manager.check_position(position)
            if signal:
                exit_signals.append((intent, signal))
        
        if not exit_signals:
            self.stdout.write("  No exit signals triggered")
            return 0, 0
        
        # Filter by reason if specified
        if reason_filter:
            target_reason = ExitReason(reason_filter)
            exit_signals = [(i, s) for i, s in exit_signals if s.reason == target_reason]
        
        self.stdout.write(f"  {len(exit_signals)} exit signal(s) to process")
        
        for intent, signal in exit_signals:
            try:
                success = self._execute_exit(
                    intent, signal, orchestrator, order_builder, dry_run
                )
                if success:
                    exits += 1
                else:
                    errors += 1
            except Exception as e:
                logger.exception(f"Error executing exit for {signal.position.symbol}: {e}")
                self.stdout.write(self.style.ERROR(f"    Error: {e}"))
                errors += 1
        
        return exits, errors
    
    def _execute_exit(
        self,
        intent: ExecutionIntent,
        signal: ExitSignal,
        orchestrator: ExecutionOrchestrator,
        order_builder: OrderBuilder,
        dry_run: bool,
    ) -> bool:
        """Execute a single exit signal."""
        position = signal.position
        
        self.stdout.write(
            f"  {signal.reason.value}: {position.symbol} "
            f"({position.side} {position.qty}) - {signal.notes}"
        )
        
        if dry_run:
            self.stdout.write(self.style.WARNING(
                f"    [DRY RUN] Would {signal.action} {signal.close_pct:.0%}"
            ))
            return True
        
        # Build the exit order
        if signal.action == 'close' or signal.close_pct >= Decimal('1.0'):
            order_plan = order_builder.build_full_close_order(
                position.symbol, position.qty, position.side
            )
        else:
            order_plan = order_builder.build_scale_down_order(
                position.symbol, position.qty, position.side, signal.close_pct
            )
        
        # Generate client order ID
        client_id = f"exit_{signal.reason.value}_{intent.id}"[:32]
        
        # Place the order
        request = order_builder.to_order_request(order_plan, client_id)
        response = orchestrator.adapter.place_order(request)
        
        if response.success:
            self.stdout.write(self.style.SUCCESS(
                f"    Order placed: {response.exchange_order_id}"
            ))
            
            # Update intent status
            intent.status = 'exit_triggered'
            intent.status_reason = f"{signal.reason.value}: {signal.notes}"
            intent.save()
            
            # Log event
            ExecutionEvent.objects.create(
                intent=intent,
                event_type='exit_triggered',
                payload={
                    'position_id': position.id,
                    'symbol': position.symbol,
                    'reason': signal.reason.value,
                    'action': signal.action,
                    'close_pct': str(signal.close_pct),
                    'notes': signal.notes,
                    'order_id': response.exchange_order_id,
                },
            )
            
            return True
        else:
            self.stdout.write(self.style.ERROR(
                f"    Order failed: {response.error_message}"
            ))
            return False
