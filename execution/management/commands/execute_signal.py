"""
Management command to execute a signal.
Creates ExecutionIntent from DailySignal and processes through exchange.

Usage:
    python manage.py execute_signal --date 2024-01-15 --account bybit-main
    python manage.py execute_signal --latest --account bybit-main --dry-run
"""
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone
from datetime import date, timedelta
import logging

from signals.models import DailySignal
from execution.models import ExchangeAccount, ExecutionIntent
from execution.services import ExecutionOrchestrator

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Execute a trading signal through the configured exchange'

    def add_arguments(self, parser):
        parser.add_argument(
            '--date',
            type=str,
            help='Signal date to execute (YYYY-MM-DD)',
        )
        parser.add_argument(
            '--latest',
            action='store_true',
            help='Execute the latest signal',
        )
        parser.add_argument(
            '--type',
            type=str,
            default=None,
            help='Trade decision type (CALL, PUT, OPTION_CALL, OPTION_PUT, TACTICAL_PUT). For IRON_CONDOR/MVRV_SHORT use execute_deribit.',
        )
        parser.add_argument(
            '--account',
            type=str,
            required=True,
            help='Exchange account name to use',
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Simulate execution without placing orders',
        )
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force execution even if intent already exists',
        )

    def handle(self, *args, **options):
        # Get signal
        signal_type_filter = options.get('type')
        if signal_type_filter:
            signal_type_filter = signal_type_filter.upper()

        if options['latest']:
            # Find latest active signal date first (mirrors notify_signal logic)
            latest_active = DailySignal.active().order_by('-date').first()
            if not latest_active:
                raise CommandError('No active signals found')
            latest_date = latest_active.date

            # Check if latest date has a veto (blocks execution of older trades)
            veto_row = DailySignal.active().filter(
                date=latest_date, trade_decision="NO_TRADE"
            ).first()
            if veto_row:
                no_trade_reasons = veto_row.no_trade_reasons or []
                if "OVERLAY_VETO" in no_trade_reasons:
                    raise CommandError(
                        f"Latest date {latest_date} has an active veto (OVERLAY_VETO). "
                        f"Cannot execute stale trades. Use --date to override."
                    )

            # Now select tradeable signal for latest date
            qs = DailySignal.tradeable().filter(date=latest_date)
            if signal_type_filter:
                signal = qs.filter(trade_decision=signal_type_filter).first()
            else:
                signal = DailySignal.pick_highest_priority(qs)
            if not signal:
                raise CommandError(f'No tradeable signals found for latest date {latest_date}')
        elif options['date']:
            try:
                signal_date = date.fromisoformat(options['date'])
                qs = DailySignal.tradeable().filter(date=signal_date)
                if signal_type_filter:
                    signal = qs.filter(trade_decision=signal_type_filter).first()
                else:
                    signal = DailySignal.pick_highest_priority(qs)
                if not signal:
                    raise CommandError(f"No tradeable signal found for date {options['date']}")
            except ValueError:
                raise CommandError(f"Invalid date format: {options['date']}")
        else:
            raise CommandError('Must specify --date or --latest')

        self.stdout.write(f"Signal: {signal.date} | {signal.trade_decision} | {signal.fusion_state}")

        # Check if tradeable by this command
        valid_decisions = ('CALL', 'PUT', 'OPTION_CALL', 'OPTION_PUT', 'TACTICAL_PUT')
        if signal.trade_decision.upper() not in valid_decisions:
            raise CommandError(
                f"Signal is {signal.trade_decision}, not supported by execute_signal. "
                f"Use 'execute_deribit' for IRON_CONDOR/MVRV_SHORT."
            )

        # Get account
        try:
            account = ExchangeAccount.objects.get(name=options['account'])
        except ExchangeAccount.DoesNotExist:
            available = list(ExchangeAccount.objects.values_list('name', flat=True))
            raise CommandError(f"Account '{options['account']}' not found. Available: {available}")

        self.stdout.write(f"Account: {account.name} ({account.exchange})")

        # Check for existing intent
        existing = ExecutionIntent.objects.filter(
            signal=signal,
            account=account,
        ).exclude(status__in=['cancelled', 'failed', 'rejected']).first()

        if existing and not options['force']:
            self.stdout.write(self.style.WARNING(
                f"Intent already exists: {existing.id} ({existing.status}). Use --force to override."
            ))
            return

        if options['dry_run']:
            self.stdout.write(self.style.SUCCESS('DRY RUN - Would create intent:'))
            self.stdout.write(f"  Direction: {'long' if signal.trade_decision in ('CALL', 'TACTICAL_PUT') else 'short'}")
            self.stdout.write(f"  Option type: {signal.trade_decision.lower()}")
            self.stdout.write(f"  Size multiplier: {signal.size_multiplier}")
            self.stdout.write(f"  Stop loss: {signal.stop_loss_pct}")
            return

        # Create and process intent
        try:
            orchestrator = ExecutionOrchestrator(account)
            
            self.stdout.write('Creating execution intent...')
            intent = orchestrator.create_intent_from_signal(signal)
            self.stdout.write(f"Intent created: {intent.id}")

            self.stdout.write('Processing intent...')
            success = orchestrator.process_intent(intent)

            intent.refresh_from_db()
            
            if success:
                self.stdout.write(self.style.SUCCESS(
                    f"Execution successful: {intent.status}"
                ))
                if intent.target_symbol:
                    self.stdout.write(f"  Symbol: {intent.target_symbol}")
                if intent.target_qty:
                    self.stdout.write(f"  Quantity: {intent.target_qty}")
            else:
                self.stdout.write(self.style.ERROR(
                    f"Execution failed: {intent.status} - {intent.status_reason}"
                ))

        except Exception as e:
            logger.exception(f"Execution error: {e}")
            raise CommandError(f"Execution failed: {e}")
