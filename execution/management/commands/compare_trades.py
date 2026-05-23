"""
Compare manual trades vs engine-suggested trades.

Records actual trades and computes what the engine would have suggested,
then compares P&L outcomes.

Usage:
    # Record a manual trade
    python manage.py compare_trades record \\
        --signal-date 2026-04-17 \\
        --long-symbol BTC-1MAY26-72000-P --long-entry 750 --long-exit 1245 \\
        --short-symbol BTC-1MAY26-77000-P --short-entry 2380 --short-exit 3300 \\
        --qty 0.01

    # Show comparison for a signal
    python manage.py compare_trades show --signal-date 2026-04-17

    # List all comparisons
    python manage.py compare_trades list
"""
import json
from datetime import date, datetime, timezone
from decimal import Decimal

from django.core.management.base import BaseCommand, CommandError
from django.db import models

from signals.models import DailySignal
from datafeed.models import RawDailyData, OptionSnapshot, OptionTrade
from execution.services.deribit_entry import DeribitEntryEngine


class TradeComparison(models.Model):
    """Stores comparison between manual and engine trades."""
    
    class Meta:
        app_label = 'execution'
        managed = False  # We'll use JSON storage instead


class Command(BaseCommand):
    help = "Compare manual trades vs engine-suggested trades"

    def add_arguments(self, parser):
        subparsers = parser.add_subparsers(dest='action', help='Action to perform')
        
        # Record subcommand
        record_parser = subparsers.add_parser('record', help='Record a manual trade')
        record_parser.add_argument('--signal-date', type=str, required=True)
        record_parser.add_argument('--type', type=str, default=None, help='Trade decision type (e.g., IRON_CONDOR)')
        record_parser.add_argument('--long-symbol', type=str, help='Long leg symbol')
        record_parser.add_argument('--long-entry', type=float, help='Long leg entry price (USD)')
        record_parser.add_argument('--long-exit', type=float, help='Long leg exit price (USD)')
        record_parser.add_argument('--short-symbol', type=str, help='Short leg symbol (for spreads)')
        record_parser.add_argument('--short-entry', type=float, help='Short leg entry price (USD)')
        record_parser.add_argument('--short-exit', type=float, help='Short leg exit price (USD)')
        record_parser.add_argument('--qty', type=float, default=0.01, help='Position size in BTC')
        record_parser.add_argument('--notes', type=str, default='', help='Trade notes')
        
        # Show subcommand
        show_parser = subparsers.add_parser('show', help='Show comparison for a signal')
        show_parser.add_argument('--signal-date', type=str, required=True)
        show_parser.add_argument('--type', type=str, default=None, help='Trade decision type (e.g., IRON_CONDOR)')
        
        # List subcommand
        subparsers.add_parser('list', help='List all recorded comparisons')

    def handle(self, *args, **options):
        action = options.get('action')
        
        if action == 'record':
            self._record_trade(options)
        elif action == 'show':
            self._show_comparison(options)
        elif action == 'list':
            self._list_comparisons()
        else:
            self.stdout.write("Use: compare_trades [record|show|list]")

    def _record_trade(self, options):
        """Record a manual trade and compute engine comparison."""
        signal_date = date.fromisoformat(options['signal_date'])
        
        # Load signal by type or highest priority
        signal_type_filter = options.get('type')
        candidates = DailySignal.tradeable().filter(date=signal_date)
        if signal_type_filter:
            signal = candidates.filter(trade_decision=signal_type_filter.upper()).first()
        else:
            signal = DailySignal.pick_highest_priority(candidates)
        if not signal:
            raise CommandError(f"No tradeable signal for {signal_date}")
        
        # Load spot price
        try:
            raw = RawDailyData.objects.get(date=signal_date)
            spot = float(raw.btc_close)
        except RawDailyData.DoesNotExist:
            raise CommandError(f"No raw data for {signal_date}")
        
        qty = Decimal(str(options['qty']))
        
        # Calculate manual trade P&L
        manual_pnl = Decimal('0')
        manual_legs = []
        
        if options.get('long_symbol'):
            long_entry = Decimal(str(options['long_entry']))
            long_exit = Decimal(str(options['long_exit']))
            long_pnl = (long_exit - long_entry) * qty
            manual_pnl += long_pnl
            manual_legs.append({
                'symbol': options['long_symbol'],
                'side': 'buy',
                'entry': float(long_entry),
                'exit': float(long_exit),
                'pnl': float(long_pnl),
            })
            self.stdout.write(f"Long leg: {options['long_symbol']}")
            self.stdout.write(f"  Entry: ${long_entry:,.2f} -> Exit: ${long_exit:,.2f}")
            self.stdout.write(f"  P&L: ${long_pnl:,.2f}")
        
        if options.get('short_symbol'):
            short_entry = Decimal(str(options['short_entry']))
            short_exit = Decimal(str(options['short_exit']))
            # Short leg: profit when price drops
            short_pnl = (short_entry - short_exit) * qty
            manual_pnl += short_pnl
            manual_legs.append({
                'symbol': options['short_symbol'],
                'side': 'sell',
                'entry': float(short_entry),
                'exit': float(short_exit),
                'pnl': float(short_pnl),
            })
            self.stdout.write(f"Short leg: {options['short_symbol']}")
            self.stdout.write(f"  Entry: ${short_entry:,.2f} -> Exit: ${short_exit:,.2f}")
            self.stdout.write(f"  P&L: ${short_pnl:,.2f}")
        
        self.stdout.write(f"\nManual Trade Total P&L: ${manual_pnl:,.2f}")
        
        # Generate engine suggestion
        engine = DeribitEntryEngine(snapshot_staleness_hours=720)  # 30 days for historical
        plan = engine.plan_entry(signal, account_size_usd=100_000, spot_price=spot)
        
        engine_suggestion = None
        if plan:
            engine_suggestion = {
                'decision': plan.trade_decision,
                'direction': plan.direction,
                'spot': plan.spot_price,
                'risk_budget': plan.total_risk_usd,
                'legs': [
                    {
                        'symbol': leg.symbol,
                        'side': leg.side,
                        'strike': float(leg.strike),
                        'delta': leg.delta,
                        'iv': leg.iv,
                        'mid_price': float(leg.mid_price) if leg.mid_price else None,
                        'dte': leg.dte,
                    }
                    for leg in plan.legs
                ],
                'rationale': plan.rationale,
            }
            self.stdout.write(f"\nEngine would have suggested:")
            self.stdout.write(f"  {plan.trade_decision} @ spot ${spot:,.0f}")
            for leg in plan.legs:
                self.stdout.write(f"  {leg.side.upper()} {leg.symbol} (delta={leg.delta}, mid=${float(leg.mid_price) if leg.mid_price else 0:,.0f})")
        else:
            self.stdout.write("\nEngine: No plan generated")
        
        # Store comparison
        comparison = {
            'signal_date': str(signal_date),
            'trade_decision': signal.trade_decision,
            'signal_type': signal.trade_decision,
            'fusion_state': signal.fusion_state,
            'spot': spot,
            'manual': {
                'legs': manual_legs,
                'total_pnl': float(manual_pnl),
                'qty': float(qty),
                'notes': options.get('notes', ''),
            },
            'engine': engine_suggestion,
            'recorded_at': datetime.now(timezone.utc).isoformat(),
        }
        
        self._save_comparison(comparison)
        self.stdout.write(self.style.SUCCESS(f"\nComparison saved for {signal_date}"))

    def _show_comparison(self, options):
        """Show comparison for a specific signal date."""
        signal_date = options['signal_date']
        signal_type = options.get('type')
        comparisons = self._load_comparisons()
        
        found = [c for c in comparisons if c['signal_date'] == signal_date]
        if signal_type:
            found = [c for c in found if c.get('trade_decision') == signal_type.upper()]
        if not found:
            self.stdout.write(f"No comparison recorded for {signal_date}")
            return
        
        for comp in found:
            self._print_comparison(comp)

    def _list_comparisons(self):
        """List all recorded comparisons."""
        comparisons = self._load_comparisons()
        
        if not comparisons:
            self.stdout.write("No comparisons recorded yet.")
            return
        
        self.stdout.write(f"\n{'Date':<12} | {'Signal':<15} | {'Manual P&L':>12} | {'Engine':>20}")
        self.stdout.write("-" * 70)
        
        total_manual = 0
        for comp in sorted(comparisons, key=lambda x: x['signal_date']):
            manual_pnl = comp['manual']['total_pnl']
            total_manual += manual_pnl
            engine_str = comp['engine']['decision'] if comp['engine'] else 'No plan'
            
            pnl_style = self.style.SUCCESS if manual_pnl > 0 else self.style.ERROR
            self.stdout.write(
                f"{comp['signal_date']:<12} | {comp['signal_type']:<15} | "
                f"{pnl_style(f'${manual_pnl:>10,.2f}')} | {engine_str:>20}"
            )
        
        self.stdout.write("-" * 70)
        total_style = self.style.SUCCESS if total_manual > 0 else self.style.ERROR
        self.stdout.write(f"{'TOTAL':<12} | {'':<15} | {total_style(f'${total_manual:>10,.2f}')}")

    def _print_comparison(self, comp):
        """Print detailed comparison."""
        self.stdout.write(f"\n{'=' * 70}")
        self.stdout.write(f"Signal: {comp['signal_date']} | {comp['signal_type']} | {comp['fusion_state']}")
        self.stdout.write(f"Spot: ${comp['spot']:,.0f}")
        self.stdout.write(f"{'=' * 70}")
        
        self.stdout.write(f"\n--- MANUAL TRADE ---")
        for leg in comp['manual']['legs']:
            self.stdout.write(f"  {leg['side'].upper()} {leg['symbol']}")
            self.stdout.write(f"    Entry: ${leg['entry']:,.2f} -> Exit: ${leg['exit']:,.2f}")
            pnl_style = self.style.SUCCESS if leg['pnl'] > 0 else self.style.ERROR
            pnl_str = f"${leg['pnl']:,.2f}"
            self.stdout.write(f"    P&L: {pnl_style(pnl_str)}")
        
        total_style = self.style.SUCCESS if comp['manual']['total_pnl'] > 0 else self.style.ERROR
        total_str = f"${comp['manual']['total_pnl']:,.2f}"
        self.stdout.write(f"\n  TOTAL P&L: {total_style(total_str)}")
        
        if comp['engine']:
            self.stdout.write(f"\n--- ENGINE SUGGESTION ---")
            self.stdout.write(f"  Decision: {comp['engine']['decision']}")
            self.stdout.write(f"  Risk budget: ${comp['engine']['risk_budget']:,.0f}")
            for leg in comp['engine']['legs']:
                mid = f"${leg['mid_price']:,.0f}" if leg['mid_price'] else 'n/a'
                self.stdout.write(f"  {leg['side'].upper()} {leg['symbol']} (delta={leg['delta']:.2f}, mid={mid})")
            self.stdout.write(f"  Rationale: {comp['engine']['rationale']}")
        else:
            self.stdout.write(f"\n--- ENGINE: No plan generated ---")

    def _load_comparisons(self) -> list:
        """Load comparisons from JSON file."""
        import os
        path = 'data/trade_comparisons.json'
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return []

    def _save_comparison(self, comparison: dict):
        """Save comparison to JSON file, keyed by (signal_date, trade_decision)."""
        import os
        path = 'data/trade_comparisons.json'
        os.makedirs('data', exist_ok=True)
        
        comparisons = self._load_comparisons()
        
        # Update existing or append — keyed by both date and trade_decision
        existing_idx = next(
            (i for i, c in enumerate(comparisons)
             if c['signal_date'] == comparison['signal_date']
             and c.get('trade_decision') == comparison.get('trade_decision')),
            None
        )
        if existing_idx is not None:
            comparisons[existing_idx] = comparison
        else:
            comparisons.append(comparison)
        
        with open(path, 'w') as f:
            json.dump(comparisons, f, indent=2)
