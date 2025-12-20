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

    def handle(self, *args, **options):
        csv_path = Path(options["csv"])
        latest_n = options["latest"]

        if not csv_path.exists():
            self.stderr.write(f"CSV not found: {csv_path}")
            return

        # Load features
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        self.stdout.write(f"\nLoaded {len(df)} rows from {csv_path}\n")

        # === MARKET STATE DISTRIBUTION ===
        self.stdout.write("=" * 60)
        self.stdout.write("MARKET STATE DISTRIBUTION")
        self.stdout.write("=" * 60)

        state_counts = {s.value: 0 for s in MarketState}
        confidence_counts = {c.value: 0 for c in Confidence}
        score_sum = 0

        for idx, row in df.iterrows():
            result = fuse_signals(row)
            state_counts[result.state.value] += 1
            confidence_counts[result.confidence.value] += 1
            score_sum += result.score

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

        # === LATEST SIGNALS ===
        self.stdout.write("\n" + "=" * 60)
        self.stdout.write(f"LATEST {latest_n} TRADE SIGNALS")
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
