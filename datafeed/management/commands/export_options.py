"""
Export Option Snapshots to CSV for Model Training

Exports collected option data with computed features for ML modeling.

Usage:
    # Export all data
    python manage.py export_options
    
    # Export specific date range
    python manage.py export_options --start 2026-01-01 --end 2026-04-01
    
    # Export specific underlying
    python manage.py export_options --underlying BTC
    
    # Custom output file
    python manage.py export_options --output my_options.csv
"""
import csv
from datetime import datetime
from django.core.management.base import BaseCommand
from django.utils import timezone

from datafeed.models import OptionSnapshot


class Command(BaseCommand):
    help = "Export option snapshots to CSV for model training"
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--output',
            default='option_snapshots.csv',
            help='Output CSV file path',
        )
        parser.add_argument(
            '--start',
            help='Start date (YYYY-MM-DD)',
        )
        parser.add_argument(
            '--end',
            help='End date (YYYY-MM-DD)',
        )
        parser.add_argument(
            '--underlying',
            default='BTC',
            help='Filter by underlying (default: BTC)',
        )
        parser.add_argument(
            '--dte-min',
            type=float,
            help='Minimum DTE filter',
        )
        parser.add_argument(
            '--dte-max',
            type=float,
            help='Maximum DTE filter',
        )
    
    def handle(self, *args, **options):
        qs = OptionSnapshot.objects.all()
        
        # Apply filters
        if options['underlying']:
            qs = qs.filter(underlying=options['underlying'])
        
        if options['start']:
            start = datetime.strptime(options['start'], '%Y-%m-%d')
            start = timezone.make_aware(start)
            qs = qs.filter(timestamp__gte=start)
        
        if options['end']:
            end = datetime.strptime(options['end'], '%Y-%m-%d')
            # Include entire end date (up to 23:59:59)
            end = end.replace(hour=23, minute=59, second=59)
            end = timezone.make_aware(end)
            qs = qs.filter(timestamp__lte=end)
        
        if options['dte_min']:
            qs = qs.filter(dte__gte=options['dte_min'])
        
        if options['dte_max']:
            qs = qs.filter(dte__lte=options['dte_max'])
        
        qs = qs.order_by('timestamp', 'symbol')
        
        count = qs.count()
        if count == 0:
            self.stdout.write(self.style.WARNING("No data to export"))
            return
        
        self.stdout.write(f"Exporting {count} snapshots...")
        
        # Define columns
        columns = [
            # Identifiers
            'timestamp',
            'symbol',
            'underlying',
            'expiry',
            'strike',
            'option_type',
            
            # Underlying
            'spot_price',
            'index_price',
            
            # Option prices
            'bid',
            'ask',
            'mid_price',
            'mark_price',
            'last_price',
            
            # IV
            'iv',
            
            # Greeks
            'delta',
            'gamma',
            'vega',
            'theta',
            
            # Liquidity
            'bid_size',
            'ask_size',
            'volume_24h',
            'open_interest',
            
            # Derived
            'dte',
            'moneyness',
            'spread_pct',
            
            # Metadata
            'exchange',
        ]
        
        output_path = options['output']
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            
            for snap in qs.iterator():
                row = [
                    snap.timestamp.isoformat(),
                    snap.symbol,
                    snap.underlying,
                    snap.expiry.isoformat() if snap.expiry else '',
                    float(snap.strike) if snap.strike is not None else '',
                    snap.option_type,
                    float(snap.spot_price) if snap.spot_price is not None else '',
                    float(snap.index_price) if snap.index_price is not None else '',
                    float(snap.bid) if snap.bid is not None else '',
                    float(snap.ask) if snap.ask is not None else '',
                    float(snap.mid_price) if snap.mid_price is not None else '',
                    float(snap.mark_price) if snap.mark_price is not None else '',
                    float(snap.last_price) if snap.last_price is not None else '',
                    float(snap.iv) if snap.iv is not None else '',
                    float(snap.delta) if snap.delta is not None else '',
                    float(snap.gamma) if snap.gamma is not None else '',
                    float(snap.vega) if snap.vega is not None else '',
                    float(snap.theta) if snap.theta is not None else '',
                    float(snap.bid_size) if snap.bid_size is not None else '',
                    float(snap.ask_size) if snap.ask_size is not None else '',
                    float(snap.volume_24h) if snap.volume_24h is not None else '',
                    float(snap.open_interest) if snap.open_interest is not None else '',
                    snap.dte if snap.dte is not None else '',
                    snap.moneyness if snap.moneyness is not None else '',
                    snap.spread_pct if snap.spread_pct is not None else '',
                    snap.exchange,
                ]
                writer.writerow(row)
        
        self.stdout.write(self.style.SUCCESS(f"Exported to {output_path}"))
