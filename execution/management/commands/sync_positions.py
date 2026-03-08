"""
Management command to sync positions from exchange.
Run periodically via cron or scheduler.

Usage:
    python manage.py sync_positions --account bybit-main
    python manage.py sync_positions --all
"""
from django.core.management.base import BaseCommand, CommandError
import logging

from execution.models import ExchangeAccount
from execution.services import ExecutionOrchestrator

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Sync positions from exchange to database'

    def add_arguments(self, parser):
        parser.add_argument('--account', type=str, help='Account name to sync')
        parser.add_argument('--all', action='store_true', help='Sync all active accounts')

    def handle(self, *args, **options):
        if options['all']:
            accounts = ExchangeAccount.objects.filter(is_active=True)
        elif options['account']:
            try:
                accounts = [ExchangeAccount.objects.get(name=options['account'])]
            except ExchangeAccount.DoesNotExist:
                raise CommandError(f"Account '{options['account']}' not found")
        else:
            raise CommandError('Must specify --account or --all')

        for account in accounts:
            self.stdout.write(f"Syncing {account.name}...")
            try:
                orchestrator = ExecutionOrchestrator(account)
                positions = orchestrator.sync_positions()
                self.stdout.write(self.style.SUCCESS(f"  Synced {len(positions)} positions"))
            except Exception as e:
                logger.exception(f"Sync error for {account.name}: {e}")
                self.stdout.write(self.style.ERROR(f"  Error: {e}"))
