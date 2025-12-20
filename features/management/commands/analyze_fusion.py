# features/management/commands/analyze_fusion.py
"""
Django command to analyze signal fusion on feature data.
Shows market state distribution, confidence levels, and sample trade signals.
"""

from django.core.management.base import BaseCommand
from pathlib import Path
import pandas as pd

from features.signals.fusion import (
    fuse_signals, MarketState, Confidence, fuse_dataframe
)
from features.signals.options import (
    get_strategy, format_recommendation, generate_trade_signal
)
from features.signals.overlays import apply_long_overlays


class Command(BaseCommand):
    help = "Analyze signal fusion on feature CSV"

    def add_arguments(self, parser):
        parser.add_argument(
            "--csv",
            type=str,
            default="features_14d_5pct.csv",
            help="Input features CSV",
        )
        parser.add_argument(
            "--latest",
            type=int,
            default=10,
            help="Number of latest rows to show signals for",
        )
        parser.add_argument(
            "--direction",
            type=str,
            choices=["long", "short", "all"],
            default=None,
            help="Filter by direction: 'long', 'short', or 'all'. Shows last N setups after overlay filter.",
        )

    def handle(self, *args, **options):
        csv_path = Path(options["csv"])
        latest_n = options["latest"]
        direction = options.get("direction")

        if not csv_path.exists():
            self.stderr.write(f"CSV not found: {csv_path}")
            return

        # Load features
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        self.stdout.write(f"\nLoaded {len(df)} rows from {csv_path}\n")
        
        # If direction specified, show filtered setups and exit
        if direction:
            self._show_direction_setups(df, direction, latest_n)
            return

        # === MARKET STATE DISTRIBUTION ===
        self.stdout.write("=" * 60)
        self.stdout.write("MARKET STATE DISTRIBUTION")
        self.stdout.write("=" * 60)

        state_counts = {s.value: 0 for s in MarketState}
        confidence_counts = {c.value: 0 for c in Confidence}
        score_sum = 0
        
        # Overlay stats with strength levels
        long_signals = 0
        edge_full = 0
        edge_partial = 0
        long_veto_strong = 0
        long_veto_moderate = 0
        short_signals = 0
        short_veto_hard = 0
        short_veto_soft = 0
        
        # Post-overlay trade counts
        long_trades_after = 0
        short_trades_after = 0

        for idx, row in df.iterrows():
            result = fuse_signals(row)
            state_counts[result.state.value] += 1
            confidence_counts[result.confidence.value] += 1
            score_sum += result.score
            
            # Track overlay effects
            overlay = apply_long_overlays(result, row)
            
            if result.state in {MarketState.STRONG_BULLISH, MarketState.EARLY_RECOVERY, MarketState.MOMENTUM_CONTINUATION}:
                long_signals += 1
                if overlay.edge_strength == 2:
                    edge_full += 1
                elif overlay.edge_strength == 1:
                    edge_partial += 1
                if overlay.long_veto_strength == 2:
                    long_veto_strong += 1
                elif overlay.long_veto_strength == 1:
                    long_veto_moderate += 1
                
                # Count surviving trades (edge or no strong veto)
                if overlay.long_veto_strength < 2:
                    long_trades_after += 1
            
            if result.state in {MarketState.DISTRIBUTION_RISK, MarketState.BEAR_CONTINUATION}:
                short_signals += 1
                if overlay.short_veto_strength == 2:
                    short_veto_hard += 1
                elif overlay.short_veto_strength == 1:
                    short_veto_soft += 1
                
                # Count surviving trades (no hard veto)
                if overlay.short_veto_strength < 2:
                    short_trades_after += 1

        total = len(df)
        
        self.stdout.write("\nMarket States:")
        for state, count in sorted(state_counts.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            bar = "█" * int(pct / 2)
            self.stdout.write(f"  {state:25s} {count:5d} ({pct:5.1f}%) {bar}")

        self.stdout.write("\nConfidence Levels:")
        for conf, count in sorted(confidence_counts.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            self.stdout.write(f"  {conf:10s} {count:5d} ({pct:5.1f}%)")

        self.stdout.write(f"\nAverage Fusion Score: {score_sum / total:.2f}")
        
        # Detailed overlay stats
        self.stdout.write("\n" + "=" * 60)
        self.stdout.write("OVERLAY STATS")
        self.stdout.write("=" * 60)
        
        self.stdout.write(f"\nLONG signals: {long_signals}")
        if long_signals > 0:
            self.stdout.write(f"  Edge FULL:      {edge_full:4d} ({edge_full/long_signals*100:.1f}%)")
            self.stdout.write(f"  Edge PARTIAL:   {edge_partial:4d} ({edge_partial/long_signals*100:.1f}%)")
            self.stdout.write(f"  Veto STRONG:    {long_veto_strong:4d} ({long_veto_strong/long_signals*100:.1f}%)")
            self.stdout.write(f"  Veto MODERATE:  {long_veto_moderate:4d} ({long_veto_moderate/long_signals*100:.1f}%)")
        
        self.stdout.write(f"\nSHORT signals: {short_signals}")
        if short_signals > 0:
            self.stdout.write(f"  Veto HARD:      {short_veto_hard:4d} ({short_veto_hard/short_signals*100:.1f}%)")
            self.stdout.write(f"  Veto SOFT:      {short_veto_soft:4d} ({short_veto_soft/short_signals*100:.1f}%)")
        
        # Post-overlay frequency
        total_before = long_signals + short_signals
        total_after = long_trades_after + short_trades_after
        years = len(df) / 365
        
        self.stdout.write(f"\n--- POST-OVERLAY TRADE FREQUENCY ---")
        self.stdout.write(f"Before overlay: {total_before} trades ({total_before/years:.1f}/year, {total_before/(len(df)/30):.1f}/month)")
        self.stdout.write(f"After overlay:  {total_after} trades ({total_after/years:.1f}/year, {total_after/(len(df)/30):.1f}/month)")
        self.stdout.write(f"  Long:  {long_trades_after} | Short: {short_trades_after}")
        
        # === SHORT VETOED SIGNALS ===
        self.stdout.write("\n" + "=" * 60)
        self.stdout.write("LAST 10 SHORT-VETOED SIGNALS")
        self.stdout.write("=" * 60)
        
        short_vetoed_list = []
        for idx, row in df.iterrows():
            result = fuse_signals(row)
            if result.state in {MarketState.DISTRIBUTION_RISK, MarketState.BEAR_CONTINUATION}:
                overlay = apply_long_overlays(result, row)
                if overlay.short_veto_active:
                    date_str = str(idx)[:10] if hasattr(idx, 'strftime') else str(idx)
                    short_vetoed_list.append({
                        'date': date_str,
                        'state': result.state.value,
                        'score': result.score,
                        'reason': overlay.reason,
                    })
        
        self.stdout.write(f"\nTotal short-vetoed: {len(short_vetoed_list)}")
        for sig in short_vetoed_list[-10:]:
            self.stdout.write(f"  {sig['date']} | {sig['state']:20s} | score: {sig['score']:+d} | {sig['reason']}")

        # === ALL TRADE SIGNALS (NOT NO_TRADE) ===
        self.stdout.write("\n" + "=" * 60)
        self.stdout.write("ALL TRADE SIGNALS (Excluding no_trade)")
        self.stdout.write("=" * 60)
        
        trade_signals = []
        for idx, row in df.iterrows():
            result = fuse_signals(row)
            if result.state != MarketState.NO_TRADE:
                date_str = str(idx)[:10] if hasattr(idx, 'strftime') else str(idx)
                trade_signals.append({
                    'date': date_str,
                    'state': result.state.value,
                    'confidence': result.confidence.value,
                    'score': result.score,
                })
        
        self.stdout.write(f"\nTotal trade signals: {len(trade_signals)}")
        
        for sig in trade_signals[-50:]:  # Show last 50 trade signals
            dir_emoji = "🟢" if sig['state'] in ['strong_bullish', 'early_recovery', 'momentum'] else "🔴"
            self.stdout.write(f"  {sig['date']} {dir_emoji} {sig['state']:20s} | {sig['confidence']:6s} | score: {sig['score']:+d}")
        
        if len(trade_signals) > 50:
            self.stdout.write(f"  ... (showing last 50 of {len(trade_signals)})")

        # === LATEST SIGNALS ===
        self.stdout.write("\n" + "=" * 60)
        self.stdout.write(f"LATEST {latest_n} ROWS (Any State)")
        self.stdout.write("=" * 60)

        latest_rows = df.tail(latest_n)
        
        for idx, row in latest_rows.iterrows():
            date_str = str(idx)[:10] if hasattr(idx, 'strftime') else str(idx)
            signal = generate_trade_signal(row.to_dict(), date_str)
            
            # Direction emoji
            dir_emoji = "🟢" if signal.direction == 'long' else \
                       "🔴" if signal.direction == 'short' else "⚪"
            
            # Confidence indicator
            conf_stars = "★★★" if signal.confidence == Confidence.HIGH else \
                        "★★☆" if signal.confidence == Confidence.MEDIUM else "★☆☆"
            
            self.stdout.write(f"\n{date_str} {dir_emoji} {signal.market_state.value}")
            self.stdout.write(f"  Confidence: {conf_stars} ({signal.confidence.value}) | Score: {signal.fusion_score}")
            self.stdout.write(f"  Structures: {', '.join([s.value for s in signal.structures])}")
            self.stdout.write(f"  DTE: {signal.min_dte}-{signal.max_dte}d | Size: {signal.position_size_pct*100:.1f}%")
            if signal.whale_campaign:
                self.stdout.write(f"  🐋 Whale Campaign Active!")

        # === STRATEGY SUMMARY ===
        self.stdout.write("\n" + "=" * 60)
        self.stdout.write("STRATEGY GUIDE BY STATE")
        self.stdout.write("=" * 60)

        for state in MarketState:
            strategy = get_strategy(state)
            if strategy.primary_structures[0].value == "no_trade":
                continue
            
            self.stdout.write(f"\n{state.value.upper()}:")
            self.stdout.write(f"  Primary: {', '.join([s.value for s in strategy.primary_structures])}")
            self.stdout.write(f"  Strike: {strategy.strike_guidance.value} | DTE: {strategy.dte}")
            self.stdout.write(f"  Sizing: H={strategy.sizing.high_confidence*100:.0f}% M={strategy.sizing.medium_confidence*100:.0f}% L={strategy.sizing.low_confidence*100:.0f}%")

        self.stdout.write("\nDone.\n")
    
    def _show_direction_setups(self, df, direction: str, n: int):
        """Show last N setups filtered by direction after overlay filter."""
        from features.signals.overlays import apply_overlays
        
        long_states = {MarketState.STRONG_BULLISH, MarketState.EARLY_RECOVERY, MarketState.MOMENTUM_CONTINUATION}
        short_states = {MarketState.DISTRIBUTION_RISK, MarketState.BEAR_CONTINUATION}
        
        setups = []
        
        for idx, row in df.iterrows():
            result = fuse_signals(row)
            overlay = apply_overlays(result, row)
            date_str = str(idx)[:10]
            
            # Filter by direction
            if direction == "long" and result.state in long_states:
                # Keep if edge boosted OR not vetoed
                if overlay.long_edge_active or not overlay.long_veto_active:
                    overlay_status = "Clean"
                    if overlay.long_edge_active:
                        overlay_status = f"EDGE: {overlay.reason.replace('LONG EDGE: ', '')}"
                    setups.append({
                        'date': date_str,
                        'state': result.state.value,
                        'score': result.score,
                        'confidence': result.confidence.value,
                        'overlay': overlay_status,
                    })
            
            elif direction == "short" and result.state in short_states:
                # Keep if not hard vetoed
                if not overlay.short_veto_active or 'SOFT' in overlay.reason:
                    overlay_status = "Clean"
                    if overlay.short_veto_active:
                        overlay_status = overlay.reason.replace('SHORT VETO: ', '')
                    setups.append({
                        'date': date_str,
                        'state': result.state.value,
                        'score': result.score,
                        'confidence': result.confidence.value,
                        'overlay': overlay_status,
                    })
            
            elif direction == "all" and result.state != MarketState.NO_TRADE:
                # Apply all filters
                is_long = result.state in long_states
                is_short = result.state in short_states
                
                if is_long and (overlay.long_edge_active or not overlay.long_veto_active):
                    overlay_status = "Clean" if not overlay.long_edge_active else f"EDGE"
                    setups.append({
                        'date': date_str,
                        'state': result.state.value,
                        'score': result.score,
                        'confidence': result.confidence.value,
                        'overlay': overlay_status,
                    })
                elif is_short and (not overlay.short_veto_active or 'SOFT' in overlay.reason):
                    overlay_status = "Clean" if not overlay.short_veto_active else "SOFT VETO"
                    setups.append({
                        'date': date_str,
                        'state': result.state.value,
                        'score': result.score,
                        'confidence': result.confidence.value,
                        'overlay': overlay_status,
                    })
        
        # Print results
        dir_emoji = "🟢" if direction == "long" else "🔴" if direction == "short" else "⚪"
        self.stdout.write(f"\n{dir_emoji} LAST {n} {direction.upper()} SETUPS (After Overlay Filter)")
        self.stdout.write("=" * 80)
        self.stdout.write(f"\nTotal {direction} setups: {len(setups)}")
        self.stdout.write("")
        
        for sig in setups[-n:]:
            self.stdout.write(f"  {sig['date']} | {sig['state']:20s} | score: {sig['score']:+d} | {sig['confidence']:6s} | {sig['overlay']}")
        
        self.stdout.write("\nDone.\n")
