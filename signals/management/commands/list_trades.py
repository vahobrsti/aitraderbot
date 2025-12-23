# signals/management/commands/list_trades.py
"""
List all trade opportunities with dates and direction.
Combines:
- Option A: Core fusion trades (LONG, PRIMARY_SHORT from overlays)
- Tactical Puts: Inside bull regimes
"""

from django.core.management.base import BaseCommand
from pathlib import Path
import pandas as pd

from signals.fusion import fuse_signals, add_fusion_features, MarketState, FusionResult, Confidence
from signals.overlays import apply_overlays, get_size_multiplier
from signals.tactical_puts import tactical_put_inside_bull


class Command(BaseCommand):
    help = "List all trade opportunities for a given year"

    def add_arguments(self, parser):
        parser.add_argument(
            "--csv",
            type=str,
            default="features_14d_5pct.csv",
            help="Input features CSV",
        )
        parser.add_argument(
            "--year",
            type=int,
            default=None,
            help="Filter to specific year (e.g., 2024)",
        )

    def handle(self, *args, **options):
        csv_path = Path(options["csv"])
        year = options.get("year")

        if not csv_path.exists():
            self.stderr.write(f"CSV not found: {csv_path}")
            return

        # Load and prepare data
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        df = add_fusion_features(df)
        
        if year:
            df = df.loc[f'{year}-01-01':f'{year}-12-31']
            self.stdout.write(f"\nFiltered to {year}: {len(df)} days\n")
        
        # Define state categories
        long_states = {MarketState.STRONG_BULLISH, MarketState.EARLY_RECOVERY, MarketState.MOMENTUM_CONTINUATION, MarketState.BULL_PROBE}
        short_states = {MarketState.DISTRIBUTION_RISK, MarketState.BEAR_CONTINUATION, MarketState.BEAR_PROBE}
        
        # Cooldown tracking (collapse clusters into single events)
        last_long_date = None
        last_short_date = None
        last_tactical_date = None
        
        # Cooldown days: suppress consecutive trades to avoid clustering
        core_cooldown_days = 7     # LONG, PRIMARY_SHORT
        probe_cooldown_days = 5    # BULL_PROBE, BEAR_PROBE
        tactical_cooldown_days = 7 # TACTICAL_PUT
        
        all_trades = []
        
        for i, (date, row) in enumerate(df.iterrows()):
            result = fuse_signals(row)
            overlay = apply_overlays(result, row)
            size_mult = get_size_multiplier(overlay)
            date_str = date.strftime('%Y-%m-%d')
            
            is_bull_probe = result.state == MarketState.BULL_PROBE
            is_bear_probe = result.state == MarketState.BEAR_PROBE
            is_long_state = result.state in long_states and not is_bull_probe
            is_short_state = result.state in short_states and not is_bear_probe
            
            # Gate probes by score (avoid -1 shorts / weak longs)
            if is_bull_probe and result.score < 2:
                continue  # skip weak bull probes
            if is_bear_probe and result.score > -2:
                continue  # skip weak bear probes
            
            # Cooldown checks
            # Core LONG trades
            if is_long_state and last_long_date is not None:
                if (date - last_long_date).days < core_cooldown_days:
                    continue  # skip clustered longs
            
            # Core SHORT trades
            if is_short_state and last_short_date is not None:
                if (date - last_short_date).days < core_cooldown_days:
                    continue  # skip clustered shorts
            
            # Probe cooldowns (existing)
            if is_bull_probe and last_long_date is not None:
                if (date - last_long_date).days < probe_cooldown_days:
                    continue
            if is_bear_probe and last_short_date is not None:
                if (date - last_short_date).days < probe_cooldown_days:
                    continue
            
            # Score-based probe sizing (instead of flat 0.5)
            if is_bull_probe:
                if result.score >= 4:
                    size_mult = min(size_mult, 0.60)
                elif result.score == 3:
                    size_mult = min(size_mult, 0.50)
                else:  # score == 2
                    size_mult = min(size_mult, 0.35)
            
            if is_bear_probe:
                if result.score <= -4:
                    size_mult = min(size_mult, 0.60)
                elif result.score == -3:
                    size_mult = min(size_mult, 0.50)
                else:  # score == -2
                    size_mult = min(size_mult, 0.35)
            
            # === OPTION A: Core Fusion Trades ===
            
            # LONG trades
            if result.state in long_states and size_mult > 0:
                if overlay.long_veto_strength < 2:  # simplified veto check
                    edge_label = ""
                    if overlay.long_edge_active:
                        edge_label = " +EDGE" if overlay.edge_strength == 2 else " +edge"
                    trade_type = 'BULL_PROBE' if is_bull_probe else 'LONG'
                    all_trades.append({
                        'date': date_str,
                        'type': trade_type,
                        'direction': 'LONG',
                        'state': result.state.value,
                        'size': size_mult,
                        'confidence': result.confidence.value,
                        'notes': f"score={result.score:+d}{edge_label}",
                    })
                    # Update cooldown tracker
                    last_long_date = date
            
            # SHORT trades (from short states)
            if result.state in short_states and size_mult > 0:
                if overlay.short_veto_strength < 2:  # simplified veto check
                    trade_type = 'BEAR_PROBE' if is_bear_probe else 'PRIMARY_SHORT'
                    all_trades.append({
                        'date': date_str,
                        'type': trade_type,
                        'direction': 'SHORT',
                        'state': result.state.value,
                        'size': size_mult,
                        'confidence': result.confidence.value,
                        'notes': f"score={result.score:+d}",
                    })
                    # Update cooldown tracker
                    last_short_date = date
            
            # === TACTICAL PUTS (inside bull regimes) ===
            # Apply cooldown to tactical puts too
            if result.state in long_states:
                # Skip if within cooldown
                if last_tactical_date is not None and (date - last_tactical_date).days < tactical_cooldown_days:
                    continue
                    
                tactical = tactical_put_inside_bull(result, row)
                if tactical.active:
                    str_label = "FULL" if tactical.strength == 2 else "PARTIAL"
                    all_trades.append({
                        'date': date_str,
                        'type': 'TACTICAL_PUT',
                        'direction': 'PUT',
                        'state': result.state.value,
                        'size': tactical.size_mult,
                        'confidence': 'tactical',
                        'notes': f"{str_label} (hedge inside bull)",
                    })
                    last_tactical_date = date
        
        # Sort by date
        all_trades.sort(key=lambda x: x['date'])
        
        # Print summary
        self.stdout.write("=" * 90)
        self.stdout.write("ALL TRADE OPPORTUNITIES")
        self.stdout.write("=" * 90)
        
        # Count by type
        long_count = sum(1 for t in all_trades if t['type'] == 'LONG')
        bull_probe_count = sum(1 for t in all_trades if t['type'] == 'BULL_PROBE')
        short_count = sum(1 for t in all_trades if t['type'] == 'PRIMARY_SHORT')
        bear_probe_count = sum(1 for t in all_trades if t['type'] == 'BEAR_PROBE')
        tactical_puts = sum(1 for t in all_trades if t['type'] == 'TACTICAL_PUT')
        
        self.stdout.write(f"\nLONG:                {long_count:3d}")
        self.stdout.write(f"BULL_PROBE (0.5x):   {bull_probe_count:3d}")
        self.stdout.write(f"PRIMARY_SHORT:       {short_count:3d}")
        self.stdout.write(f"BEAR_PROBE (0.5x):   {bear_probe_count:3d}")
        self.stdout.write(f"TACTICAL_PUT:        {tactical_puts:3d}")
        self.stdout.write(f"─────────────────────────")
        self.stdout.write(f"TOTAL:               {len(all_trades):3d}")
        
        # Print all trades
        self.stdout.write("\n" + "-" * 90)
        self.stdout.write(f"{'Date':12s} | {'Type':14s} | {'Dir':8s} | {'State':20s} | {'Size':5s} | Notes")
        self.stdout.write("-" * 90)
        
        for t in all_trades:
            dir_emoji = "🟢" if t['direction'] == 'LONG' else "🔴"
            self.stdout.write(
                f"{t['date']:12s} | {t['type']:14s} | {dir_emoji} {t['direction']:5s} | "
                f"{t['state']:20s} | {t['size']:.2f}  | {t['notes']}"
            )
        
        self.stdout.write("-" * 90)
        self.stdout.write(f"\nTotal trades: {len(all_trades)}")
        self.stdout.write("Done.\n")
