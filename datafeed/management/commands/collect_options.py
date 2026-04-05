"""
Collect Option Snapshots from Exchanges

Fetches option chain data and stores in OptionSnapshot table.
Supports Bybit and Deribit. Run hourly via cron for continuous data collection.

Usage:
    # Collect from Bybit (default)
    python manage.py collect_options
    
    # Collect from Deribit
    python manage.py collect_options --exchange deribit
    
    # Collect from both
    python manage.py collect_options --exchange all
    
    # Custom filters
    python manage.py collect_options --dte-min 3 --dte-max 21 --moneyness 0.15
    
    # Single option
    python manage.py collect_options --symbol BTC-26APR26-100000-C
    
    # Dry run (don't save)
    python manage.py collect_options --dry-run
"""
from django.core.management.base import BaseCommand

from datafeed.models import OptionSnapshot


def get_fetcher(exchange: str, testnet: bool = False):
    """Factory to get the appropriate fetcher."""
    if exchange == 'bybit':
        from datafeed.ingestion.bybit_options import BybitOptionsFetcher
        return BybitOptionsFetcher(testnet=testnet)
    elif exchange == 'deribit':
        from datafeed.ingestion.deribit_options import DeribitOptionsFetcher
        return DeribitOptionsFetcher(testnet=testnet)
    else:
        raise ValueError(f"Unknown exchange: {exchange}")


class Command(BaseCommand):
    help = "Collect option snapshots from exchanges"
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--exchange',
            default='bybit',
            choices=['bybit', 'deribit', 'all'],
            help='Exchange to collect from (default: bybit)',
        )
        parser.add_argument(
            '--underlying',
            default='BTC',
            help='Underlying asset (default: BTC)',
        )
        parser.add_argument(
            '--dte-min',
            type=int,
            default=3,
            help='Minimum days to expiry (default: 3)',
        )
        parser.add_argument(
            '--dte-max',
            type=int,
            default=30,
            help='Maximum days to expiry (default: 30)',
        )
        parser.add_argument(
            '--moneyness',
            type=float,
            default=0.20,
            help='Moneyness range as fraction (default: 0.20 = ±20%%)',
        )
        parser.add_argument(
            '--symbol',
            help='Fetch single option by symbol',
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Print data without saving',
        )
        parser.add_argument(
            '--testnet',
            action='store_true',
            help='Use testnet',
        )
    
    def handle(self, *args, **options):
        exchanges = ['bybit', 'deribit'] if options['exchange'] == 'all' else [options['exchange']]
        
        for exchange in exchanges:
            self.stdout.write(f"\n{'='*60}")
            self.stdout.write(f"Exchange: {exchange.upper()}")
            self.stdout.write(f"{'='*60}")
            
            try:
                fetcher = get_fetcher(exchange, options['testnet'])
                
                if options['symbol']:
                    self._fetch_single(fetcher, options['symbol'], options['dry_run'])
                else:
                    self._fetch_chain(fetcher, options)
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Error with {exchange}: {e}"))
    
    def _fetch_single(self, fetcher, symbol: str, dry_run: bool):
        self.stdout.write(f"Fetching {symbol}...")
        
        snapshot = fetcher.fetch_single_option(symbol)
        if not snapshot:
            self.stdout.write(self.style.ERROR(f"Could not fetch {symbol}"))
            return
        
        self._print_snapshot(snapshot)
        
        if not dry_run:
            self._save_snapshot(snapshot)
            self.stdout.write(self.style.SUCCESS("Saved"))
    
    def _fetch_chain(self, fetcher, options):
        underlying = options['underlying']
        dte_min = options['dte_min']
        dte_max = options['dte_max']
        moneyness = options['moneyness']
        dry_run = options['dry_run']
        
        self.stdout.write(f"\nCollecting {underlying} options")
        self.stdout.write(f"  DTE: {dte_min}-{dte_max}")
        self.stdout.write(f"  Moneyness: ±{moneyness*100:.0f}%")
        self.stdout.write("")
        
        snapshots = fetcher.fetch_option_chain(
            underlying=underlying,
            dte_min=dte_min,
            dte_max=dte_max,
            moneyness_range=(-moneyness, moneyness),
        )
        
        if not snapshots:
            self.stdout.write(self.style.WARNING("No options found"))
            return
        
        # Group by expiry for display
        by_expiry = {}
        for s in snapshots:
            exp = s['expiry'].strftime('%d%b%y').upper()
            by_expiry.setdefault(exp, []).append(s)
        
        self.stdout.write(f"{'Symbol':<32} {'Type':>4} {'Strike':>10} {'DTE':>5} {'Bid':>12} {'Ask':>12} {'IV':>7} {'Delta':>7} {'Spread':>7}")
        self.stdout.write("-" * 120)
        
        saved = 0
        for expiry in sorted(by_expiry.keys()):
            opts = sorted(by_expiry[expiry], key=lambda x: (x['strike'], x['option_type']))
            for s in opts:
                self._print_snapshot_row(s)
                
                if not dry_run:
                    self._save_snapshot(s)
                    saved += 1
        
        self.stdout.write("")
        if dry_run:
            self.stdout.write(f"Dry run: {len(snapshots)} snapshots (not saved)")
        else:
            self.stdout.write(self.style.SUCCESS(f"Saved {saved} snapshots"))
    
    def _print_snapshot(self, s: dict):
        """Print detailed single snapshot."""
        self.stdout.write(f"\n{s['symbol']}")
        self.stdout.write(f"  Exchange: {s['exchange']}")
        self.stdout.write(f"  Timestamp: {s['timestamp']}")
        self.stdout.write(f"  Expiry: {s['expiry']} (DTE: {s['dte']:.1f})")
        self.stdout.write(f"  Strike: ${s['strike']:,.0f}")
        self.stdout.write(f"  Type: {s['option_type']}")
        self.stdout.write(f"  Spot: ${s['spot_price']:,.2f}")
        self.stdout.write(f"  Moneyness: {s['moneyness']*100:+.1f}%")
        self.stdout.write(f"  Bid: {self._fmt_price(s['bid'])}")
        self.stdout.write(f"  Ask: {self._fmt_price(s['ask'])}")
        self.stdout.write(f"  Mid: {self._fmt_price(s['mid_price'])}")
        self.stdout.write(f"  Mark: {self._fmt_price(s['mark_price'])}")
        self.stdout.write(f"  IV: {s['iv']*100:.1f}%" if s['iv'] is not None else "  IV: N/A")
        self.stdout.write(f"  Delta: {s['delta']:.4f}" if s['delta'] is not None else "  Delta: N/A")
        self.stdout.write(f"  Gamma: {s['gamma']}" if s['gamma'] is not None else "  Gamma: N/A")
        self.stdout.write(f"  Vega: {s['vega']}" if s['vega'] is not None else "  Vega: N/A")
        self.stdout.write(f"  Theta: {s['theta']}" if s['theta'] is not None else "  Theta: N/A")
        self.stdout.write(f"  Spread: {s['spread_pct']*100:.2f}%" if s['spread_pct'] is not None else "  Spread: N/A")
        self.stdout.write(f"  Volume 24h: {s['volume_24h']}" if s['volume_24h'] is not None else "  Volume 24h: N/A")
        self.stdout.write(f"  Open Interest: {s['open_interest']}" if s['open_interest'] is not None else "  Open Interest: N/A")
    
    def _print_snapshot_row(self, s: dict):
        """Print snapshot as table row."""
        bid_str = f"${float(s['bid']):,.2f}" if s['bid'] is not None else "N/A"
        ask_str = f"${float(s['ask']):,.2f}" if s['ask'] is not None else "N/A"
        iv_str = f"{float(s['iv'])*100:.1f}%" if s['iv'] is not None else "N/A"
        delta_str = f"{float(s['delta']):.3f}" if s['delta'] is not None else "N/A"
        spread_str = f"{s['spread_pct']*100:.1f}%" if s['spread_pct'] is not None else "N/A"
        
        self.stdout.write(
            f"{s['symbol']:<32} {s['option_type']:>4} ${float(s['strike']):>9,.0f} "
            f"{s['dte']:>5.1f} {bid_str:>12} {ask_str:>12} {iv_str:>7} {delta_str:>7} {spread_str:>7}"
        )
    
    def _fmt_price(self, val) -> str:
        """Format price for display."""
        if val is None:
            return "N/A"
        return f"${float(val):,.4f}"
    
    def _save_snapshot(self, s: dict):
        """Save snapshot to database."""
        OptionSnapshot.objects.update_or_create(
            timestamp=s['timestamp'],
            symbol=s['symbol'],
            exchange=s['exchange'],
            defaults={
                'underlying': s['underlying'],
                'expiry': s['expiry'],
                'strike': s['strike'],
                'option_type': s['option_type'],
                'spot_price': s['spot_price'],
                'index_price': s['index_price'],
                'bid': s['bid'],
                'ask': s['ask'],
                'mid_price': s['mid_price'],
                'mark_price': s['mark_price'],
                'last_price': s['last_price'],
                'iv': s['iv'],
                'delta': s['delta'],
                'gamma': s['gamma'],
                'vega': s['vega'],
                'theta': s['theta'],
                'bid_size': s['bid_size'],
                'ask_size': s['ask_size'],
                'volume_24h': s['volume_24h'],
                'open_interest': s['open_interest'],
                'dte': s['dte'],
                'moneyness': s['moneyness'],
                'spread_pct': s['spread_pct'],
            }
        )
