"""
Reconciliation job: sync exchange state with database.
Run periodically to catch missed fills, position changes, etc.

Usage:
    python manage.py reconcile --account bybit-main
    python manage.py reconcile --all
"""
from django.core.management.base import BaseCommand, CommandError
import logging

from execution.models import ExchangeAccount
from execution.services import ExecutionOrchestrator

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Reconcile exchange state with database'

    def add_arguments(self, parser):
        parser.add_argument('--account', type=str, help='Account name')
        parser.add_argument('--all', action='store_true', help='All active accounts')

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
            self.stdout.write(f"Reconciling {account.name}...")
            try:
                orchestrator = ExecutionOrchestrator(account)
                summary = orchestrator.reconcile()
                self.stdout.write(self.style.SUCCESS(
                    f"  Positions: {summary['positions_synced']}, "
                    f"Orders: {summary['orders_updated']}"
                ))
            except Exception as e:
                logger.exception(f"Reconcile error: {e}")
                self.stdout.write(self.style.ERROR(f"  Error: {e}"))
