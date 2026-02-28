# signals/management/commands/analyze_fusion.py
"""
Django command to analyze signal fusion on feature data.
Shows market state distribution, confidence levels, and sample trade signals.
Includes tactical puts analysis for puts inside bull regimes.
"""

from django.core.management.base import BaseCommand
from pathlib import Path
import pandas as pd

from signals.fusion import (
    fuse_signals, MarketState, Confidence, fuse_dataframe, FusionResult
)
from signals.options import (
    get_strategy, format_recommendation, generate_trade_signal
)
from signals.overlays import apply_long_overlays, apply_overlays, get_size_multiplier
from signals.tactical_puts import tactical_put_inside_bull, TacticalPutStrategy


class Command(BaseCommand):
    help = "Analyze signal fusion: market state distribution, overlay stats, tactical puts, and trade signals"

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
        parser.add_argument(
            "--year",
            type=int,
            default=None,
            help="Filter to a specific year (e.g., 2024)",
        )
        parser.add_argument(
            "--explain",
            action="store_true",
            help="Show detailed per-horizon breakdown explaining fusion score",
        )
        parser.add_argument(
            "--date",
            type=str,
            default=None,
            help="Target date for --explain mode (YYYY-MM-DD), defaults to latest",
        )
        parser.add_argument(
            "--research",
            action="store_true",
            help="Run full research pipeline (bucket stats, modeling, validation)",
        )
        parser.add_argument(
            "--stats",
            type=str,
            choices=["metric", "combo", "state", "score"],
            default=None,
            help="Show stats: metric (per-bucket), combo (pairwise), state, or score",
        )

    def handle(self, *args, **options):
        csv_path = Path(options["csv"])
        latest_n = options["latest"]
        direction = options.get("direction")
        year = options.get("year")

        if not csv_path.exists():
            self.stderr.write(f"CSV not found: {csv_path}")
            return

        # Load features
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        self.stdout.write(f"\nLoaded {len(df)} rows from {csv_path}\n")
        
        # Filter by year if specified
        if year:
            df = df.loc[f'{year}-01-01':f'{year}-12-31']
            self.stdout.write(f"Filtered to {year}: {len(df)} rows\n")
        
        if len(df) == 0:
            self.stderr.write("No rows after filtering. Nothing to analyze.\n")
            return
        
        # If explain mode, show detailed breakdown and exit
        if options.get("explain"):
            self._show_explain(df, options.get("date"))
            return
        
        # Research pipeline mode
        if options.get("research"):
            self._show_research(df)
            return
        
        # Stats mode
        if options.get("stats"):
            self._show_stats(df, options["stats"])
            return
        
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
        short_edge_full = 0
        short_edge_partial = 0
        
        # Tactical put stats
        tactical_put_full = 0
        tactical_put_partial = 0
        
        # Post-overlay trade counts
        long_trades_after = 0
        short_trades_after = 0
        tactical_put_count = 0
        option_call_count = 0
        option_put_count = 0

        for idx, row in df.iterrows():
            result = fuse_signals(row)
            state_counts[result.state.value] += 1
            confidence_counts[result.confidence.value] += 1
            score_sum += result.score
            
            # Track overlay effects
            overlay = apply_long_overlays(result, row)
            
            if result.state in {MarketState.STRONG_BULLISH, MarketState.EARLY_RECOVERY, MarketState.MOMENTUM_CONTINUATION, MarketState.BULL_PROBE}:
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
                
                # Check for tactical puts
                tactical_result = tactical_put_inside_bull(result, row)
                if tactical_result.active:
                    tactical_put_count += 1
                    if tactical_result.strength == 2:
                        tactical_put_full += 1
                    else:
                        tactical_put_partial += 1
            
            if result.state in {MarketState.DISTRIBUTION_RISK, MarketState.BEAR_CONTINUATION, MarketState.BEAR_PROBE}:
                short_signals += 1
                if overlay.short_veto_strength == 2:
                    short_veto_hard += 1
                elif overlay.short_veto_strength == 1:
                    short_veto_soft += 1
                if overlay.short_edge_strength == 2:
                    short_edge_full += 1
                elif overlay.short_edge_strength == 1:
                    short_edge_partial += 1
                
                # Count surviving trades (no hard veto)
                if overlay.short_veto_strength < 2:
                    short_trades_after += 1
            
            # Track option signals (fire on any fusion state)
            if int(row.get('signal_option_call', 0)) == 1:
                option_call_count += 1
            if int(row.get('signal_option_put', 0)) == 1:
                option_put_count += 1

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
            self.stdout.write(f"  Edge FULL:      {short_edge_full:4d} ({short_edge_full/short_signals*100:.1f}%)")
            self.stdout.write(f"  Edge PARTIAL:   {short_edge_partial:4d} ({short_edge_partial/short_signals*100:.1f}%)")
            self.stdout.write(f"  Veto HARD:      {short_veto_hard:4d} ({short_veto_hard/short_signals*100:.1f}%)")
            self.stdout.write(f"  Veto SOFT:      {short_veto_soft:4d} ({short_veto_soft/short_signals*100:.1f}%)")
        
        # Tactical puts stats
        self.stdout.write(f"\nTACTICAL PUTS (inside bull): {tactical_put_count}")
        if long_signals > 0:
            self.stdout.write(f"  FULL:           {tactical_put_full:4d} ({tactical_put_full/long_signals*100:.1f}%)")
            self.stdout.write(f"  PARTIAL:        {tactical_put_partial:4d} ({tactical_put_partial/long_signals*100:.1f}%)")
        
        # Post-overlay frequency
        total_before = long_signals + short_signals
        total_after = long_trades_after + short_trades_after
        total_short_opps = short_trades_after + tactical_put_count
        years = len(df) / 365
        
        self.stdout.write(f"\n--- POST-OVERLAY TRADE FREQUENCY ---")
        self.stdout.write(f"Before overlay: {total_before} trades ({total_before/years:.1f}/year, {total_before/(len(df)/30):.1f}/month)")
        self.stdout.write(f"After overlay:  {total_after} trades ({total_after/years:.1f}/year, {total_after/(len(df)/30):.1f}/month)")
        self.stdout.write(f"  Long:  {long_trades_after} | Short: {short_trades_after}")
        self.stdout.write(f"\n--- TOTAL SHORT OPPORTUNITIES ---")
        self.stdout.write(f"Fusion shorts passed overlay: {short_trades_after}")
        self.stdout.write(f"Tactical puts in bull regime: {tactical_put_count}")
        self.stdout.write(f"TOTAL SHORT OPPS: {total_short_opps} ({total_short_opps/years:.1f}/year)")
        
        self.stdout.write(f"\n--- OPTION SIGNALS (Rule-based) ---")
        self.stdout.write(f"OPTION_CALL (MVRV cheap + fear):  {option_call_count} ({option_call_count/years:.1f}/year)")
        self.stdout.write(f"OPTION_PUT  (MVRV hot + greed):   {option_put_count} ({option_put_count/years:.1f}/year)")
        
        # === TACTICAL PUTS LIST ===
        self.stdout.write("\n" + "=" * 60)
        self.stdout.write("TACTICAL PUTS (Puts Inside Bull Regimes)")
        self.stdout.write("=" * 60)
        
        tactical_put_list = []
        for idx, row in df.iterrows():
            result = fuse_signals(row)
            if result.state in {MarketState.STRONG_BULLISH, MarketState.EARLY_RECOVERY, MarketState.MOMENTUM_CONTINUATION, MarketState.BULL_PROBE}:
                tactical_result = tactical_put_inside_bull(result, row)
                if tactical_result.active:
                    date_str = str(idx)[:10] if hasattr(idx, 'strftime') else str(idx)
                    tactical_put_list.append({
                        'date': date_str,
                        'state': result.state.value,
                        'strength': tactical_result.strength,
                        'size': tactical_result.size_mult,
                        'reason': tactical_result.reason,
                    })
        
        self.stdout.write(f"\nTotal tactical puts: {len(tactical_put_list)}")
        for sig in tactical_put_list[-15:]:
            str_label = "FULL" if sig['strength'] == 2 else "PARTIAL"
            self.stdout.write(f"  {sig['date']} | {sig['state']:15s} | {str_label:7s} | size={sig['size']:.2f} | {sig['reason'][:45]}")
        
        # === SHORT VETOED SIGNALS ===
        self.stdout.write("\n" + "=" * 60)
        self.stdout.write("LAST 10 SHORT-VETOED SIGNALS")
        self.stdout.write("=" * 60)
        
        short_vetoed_list = []
        for idx, row in df.iterrows():
            result = fuse_signals(row)
            if result.state in {MarketState.DISTRIBUTION_RISK, MarketState.BEAR_CONTINUATION, MarketState.BEAR_PROBE}:
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
            date_str = str(idx)[:10] if hasattr(idx, 'strftime') else str(idx)
            
            if result.state != MarketState.NO_TRADE:
                trade_signals.append({
                    'date': date_str,
                    'state': result.state.value,
                    'confidence': result.confidence.value,
                    'score': result.score,
                    'type': 'fusion',
                })
            
            # Option signals (tracked independently)
            if int(row.get('signal_option_call', 0)) == 1:
                trade_signals.append({
                    'date': date_str,
                    'state': result.state.value,
                    'confidence': result.confidence.value,
                    'score': result.score,
                    'type': 'OPTION_CALL',
                })
            if int(row.get('signal_option_put', 0)) == 1:
                trade_signals.append({
                    'date': date_str,
                    'state': result.state.value,
                    'confidence': result.confidence.value,
                    'score': result.score,
                    'type': 'OPTION_PUT',
                })
        
        self.stdout.write(f"\nTotal trade signals: {len(trade_signals)}")
        
        for sig in trade_signals[-50:]:
            if sig['type'] == 'OPTION_CALL':
                dir_emoji = "📗"
            elif sig['type'] == 'OPTION_PUT':
                dir_emoji = "📕"
            elif sig['state'] in ['strong_bullish', 'early_recovery', 'momentum', 'bull_probe']:
                dir_emoji = "🟢"
            else:
                dir_emoji = "🔴"
            type_label = f"[{sig['type']:12s}]" if sig['type'] != 'fusion' else " " * 14
            self.stdout.write(f"  {sig['date']} {dir_emoji} {type_label} {sig['state']:20s} | {sig['confidence']:6s} | score: {sig['score']:+d}")
        
        if len(trade_signals) > 50:
            self.stdout.write(f"  ... (showing last 50 of {len(trade_signals)})")

        # === HIT RATE BY STATE ===
        self.stdout.write("\n" + "=" * 60)
        self.stdout.write("HIT RATE BY STATE (Price Validation)")
        self.stdout.write("=" * 60)
        
        long_states_set = {MarketState.STRONG_BULLISH, MarketState.EARLY_RECOVERY, MarketState.MOMENTUM_CONTINUATION, MarketState.BULL_PROBE}
        short_states_set = {MarketState.DISTRIBUTION_RISK, MarketState.BEAR_CONTINUATION, MarketState.BEAR_PROBE}
        
        has_labels = 'label_good_move_long' in df.columns and 'label_good_move_short' in df.columns
        if has_labels:
            state_hits = {}  # {state_value: {'count': N, 'hits': N}}
            for idx, row in df.iterrows():
                result = fuse_signals(row)
                sv = result.state.value
                if sv not in state_hits:
                    state_hits[sv] = {'count': 0, 'long_hits': 0, 'short_hits': 0}
                state_hits[sv]['count'] += 1
                if result.state in long_states_set:
                    state_hits[sv]['long_hits'] += int(row.get('label_good_move_long', 0))
                elif result.state in short_states_set:
                    state_hits[sv]['short_hits'] += int(row.get('label_good_move_short', 0))
            
            self.stdout.write(f"\n{'State':25s} | {'Hit Rate':>10s} | {'Hits':>5s} | {'Count':>5s} | {'Dir':>5s}")
            self.stdout.write("-" * 65)
            for state in MarketState:
                sv = state.value
                if sv not in state_hits or state == MarketState.NO_TRADE:
                    continue
                info = state_hits[sv]
                count = info['count']
                if state in long_states_set:
                    hits = info['long_hits']
                    direction_label = 'long'
                elif state in short_states_set:
                    hits = info['short_hits']
                    direction_label = 'short'
                else:
                    continue
                hit_rate = hits / count * 100 if count > 0 else 0
                bar = "█" * int(hit_rate / 5)
                self.stdout.write(f"  {sv:23s} | {hit_rate:8.1f}%  | {hits:5d} | {count:5d} | {direction_label:>5s} {bar}")
        else:
            self.stdout.write("\n(label_good_move_long/short columns not found — skipping hit rate)")

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
            
            # Check for tactical put
            result = fuse_signals(row)
            tactical_result = tactical_put_inside_bull(result, row)
            if tactical_result.active:
                str_label = "FULL" if tactical_result.strength == 2 else "PARTIAL"
                self.stdout.write(f"  🔻 TACTICAL PUT ({str_label}): size={tactical_result.size_mult:.2f}")
            
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

    # ── Research pipeline methods (use signals.research services) ──
    def _show_research(self, df: pd.DataFrame):
        """Run full research pipeline: stats, modeling, validation."""
        from signals.research.fusion_table import build_research_table
        from signals.research.stats import compute_bucket_stats, compute_state_stats
        from signals.research.modeling import run_model_comparison
        from signals.research.reporting import (
            generate_weighting_proposal,
            validate_monotonicity,
            validate_state_stability,
        )

        self.stdout.write("\n" + "=" * 70)
        self.stdout.write("RESEARCH PIPELINE")
        self.stdout.write("=" * 70)

        # 1. Build research table
        self.stdout.write("\n[1/5] Building research table...")
        rt = build_research_table(df)
        self.stdout.write(f"  → {len(rt)} rows, {len(rt.columns)} columns")
        self.stdout.write(f"  → Date range: {rt.index.min()} → {rt.index.max()}")

        # 2. Bucket stats
        self.stdout.write("\n[2/5] Computing bucket stats...")
        for metric in ("mdia_bucket", "whale_bucket", "mvrv_ls_bucket"):
            stats = compute_bucket_stats(rt, metric)
            label = metric.replace("_bucket", "").upper()
            self.stdout.write(f"\n  {label}:")
            self.stdout.write(f"  {'Bucket':25s} | {'N':>5s} | {'Long HR':>8s} | {'Short HR':>8s} | {'Ret14d':>8s} | {'Flag':>4s}")
            self.stdout.write("  " + "-" * 65)
            for _, row in stats.iterrows():
                bucket = row.get(metric, "?")
                flag = "⚠️" if row.get("_flagged", False) else ""
                self.stdout.write(
                    f"  {bucket:25s} | {int(row['n']):5d} | "
                    f"{row.get('long_hit_rate', 0)*100:7.1f}% | "
                    f"{row.get('short_hit_rate', 0)*100:7.1f}% | "
                    f"{row.get('ret_14d_median', 0)*100:7.2f}% | {flag}"
                )

        # 3. Modeling
        self.stdout.write("\n[3/5] Running model comparison (logistic regression)...")
        for target in ("label_good_move_long", "label_good_move_short"):
            result = run_model_comparison(rt, target_col=target, n_splits=5)
            short_label = "LONG" if "long" in target else "SHORT"
            self.stdout.write(f"\n  TARGET: {short_label}")
            self.stdout.write(f"  {'Model':30s} | {'AUC':>6s} | {'LogLoss':>8s}")
            self.stdout.write("  " + "-" * 50)
            for mr in sorted(result["model_results"], key=lambda x: -x["mean_auc"]):
                self.stdout.write(
                    f"  {mr['model_name']:30s} | {mr['mean_auc']:.4f} | {mr['mean_logloss']:.4f}"
                )

            summary = result["summary"]
            self.stdout.write(f"\n  Best standalone:   {summary['best_standalone']}")
            self.stdout.write(f"  Best incremental:  {summary['best_incremental']}")
            if summary["confirmer_metrics"]:
                self.stdout.write(f"  Confirmer metrics: {', '.join(summary['confirmer_metrics'])}")

            self.stdout.write(f"\n  Permutation importance:")
            for m, imp in sorted(result["feature_importance"].items(), key=lambda x: -x[1]):
                self.stdout.write(f"    {m:10s}: {imp:+.4f} AUC drop")

        # 4. Monotonicity
        self.stdout.write("\n[4/5] Score monotonicity validation...")
        mono = validate_monotonicity(rt)
        self.stdout.write(f"\n  {'Score':>6s} | {'N':>5s} | {'Long HR':>8s} | {'Short HR':>8s} | {'Ret14d':>8s}")
        self.stdout.write("  " + "-" * 50)
        for _, row in mono.iterrows():
            flag = " ⚠️" if row.get("_flagged", False) else ""
            self.stdout.write(
                f"  {int(row['score']):6d} | {int(row['n']):5d} | "
                f"{row.get('long_hit_rate', 0)*100:7.1f}% | "
                f"{row.get('short_hit_rate', 0)*100:7.1f}% | "
                f"{row.get('ret_14d_median', 0)*100:7.2f}%{flag}"
            )
        if "monotonic_long" in mono.columns:
            self.stdout.write(f"\n  Monotonic (long):  {'✓' if mono['monotonic_long'].iloc[0] else '✗'}")
        if "monotonic_short" in mono.columns:
            self.stdout.write(f"  Monotonic (short): {'✓' if mono['monotonic_short'].iloc[0] else '✗'}")

        # 5. State stability
        self.stdout.write("\n[5/5] State stability by year...")
        stability = validate_state_stability(rt)
        current_state = None
        for _, row in stability.iterrows():
            st = row["fusion_state"]
            if st != current_state:
                self.stdout.write(f"\n  {st.upper()}")
                current_state = st
            flag = " ⚠️" if row.get("_flagged", False) else ""
            self.stdout.write(
                f"    {int(row['year'])} | n={int(row['n']):4d} | "
                f"long={row.get('long_hit_rate', 0)*100:5.1f}% | "
                f"short={row.get('short_hit_rate', 0)*100:5.1f}% | "
                f"ret14d={row.get('ret_14d_median', 0)*100:6.2f}%{flag}"
            )

        self.stdout.write("\n\nDone.\n")

    def _show_stats(self, df: pd.DataFrame, stats_type: str):
        """Show standalone stats using the research service layer."""
        from signals.research.fusion_table import build_research_table
        from signals.research.stats import (
            compute_bucket_stats,
            compute_combo_stats,
            compute_state_stats,
            compute_score_stats,
        )

        self.stdout.write("\nBuilding research table...")
        rt = build_research_table(df)
        self.stdout.write(f"  → {len(rt)} rows\n")

        if stats_type == "metric":
            for metric in ("mdia_bucket", "whale_bucket", "mvrv_ls_bucket"):
                stats = compute_bucket_stats(rt, metric)
                label = metric.replace("_bucket", "").upper()
                self.stdout.write(f"\n{'=' * 60}")
                self.stdout.write(f"{label} BUCKET STATS")
                self.stdout.write(f"{'=' * 60}")
                self._print_stats_table(stats, metric)

        elif stats_type == "combo":
            from signals.research.constants import BUCKET_COLS
            pairs = [
                (BUCKET_COLS[0], BUCKET_COLS[1]),
                (BUCKET_COLS[0], BUCKET_COLS[2]),
                (BUCKET_COLS[1], BUCKET_COLS[2]),
            ]
            for col_a, col_b in pairs:
                stats = compute_combo_stats(rt, [col_a, col_b])
                label = f"{col_a} × {col_b}".upper()
                self.stdout.write(f"\n{'=' * 60}")
                self.stdout.write(f"{label}")
                self.stdout.write(f"{'=' * 60}")
                self._print_stats_table(stats, col_a)

        elif stats_type == "state":
            stats = compute_state_stats(rt)
            self.stdout.write(f"\n{'=' * 60}")
            self.stdout.write("FUSION STATE STATS")
            self.stdout.write(f"{'=' * 60}")
            self._print_stats_table(stats, "fusion_state")

        elif stats_type == "score":
            stats = compute_score_stats(rt)
            self.stdout.write(f"\n{'=' * 60}")
            self.stdout.write("FUSION SCORE STATS")
            self.stdout.write(f"{'=' * 60}")
            self._print_stats_table(stats, "fusion_score")

        self.stdout.write("\nDone.\n")

    def _print_stats_table(self, stats: 'pd.DataFrame', group_col: str):
        """Pretty-print a stats DataFrame."""
        self.stdout.write(f"\n{'Group':30s} | {'N':>5s} | {'Long HR':>8s} | {'Short HR':>8s} | {'Ret14d med':>10s} | {'MFE14d':>8s} | {'MAE14d':>8s}")
        self.stdout.write("-" * 95)
        for _, row in stats.iterrows():
            group = str(row.get(group_col, "?"))
            flag = " ⚠️" if row.get("_flagged", False) else ""
            self.stdout.write(
                f"{group:30s} | {int(row['n']):5d} | "
                f"{row.get('long_hit_rate', 0)*100:7.1f}% | "
                f"{row.get('short_hit_rate', 0)*100:7.1f}% | "
                f"{row.get('ret_14d_median', 0)*100:9.2f}% | "
                f"{row.get('mfe_14d_median', 0)*100:7.2f}% | "
                f"{row.get('mae_14d_median', 0)*100:7.2f}%{flag}"
            )

    def _show_direction_setups(self, df, direction: str, n: int):
        """Show last N setups filtered by direction after overlay filter."""
        from signals.overlays import apply_overlays
        
        long_states = {MarketState.STRONG_BULLISH, MarketState.EARLY_RECOVERY, MarketState.MOMENTUM_CONTINUATION, MarketState.BULL_PROBE}
        short_states = {MarketState.DISTRIBUTION_RISK, MarketState.BEAR_CONTINUATION, MarketState.BEAR_PROBE}
        
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
                        'type': 'long',
                    })
                
                # Also check for tactical puts
                tactical_result = tactical_put_inside_bull(result, row)
                if tactical_result.active:
                    setups.append({
                        'date': date_str,
                        'state': result.state.value,
                        'score': result.score,
                        'confidence': result.confidence.value,
                        'overlay': f"TACTICAL PUT: {tactical_result.reason[:30]}",
                        'type': 'tactical_put',
                    })
                
                # Option call signals
                if int(row.get('signal_option_call', 0)) == 1:
                    setups.append({
                        'date': date_str,
                        'state': result.state.value,
                        'score': result.score,
                        'confidence': result.confidence.value,
                        'overlay': 'MVRV cheap + fear',
                        'type': 'OPTION_CALL',
                    })
            
            elif direction == "short" and result.state in short_states:
                # PRIMARY_SHORT: Keep if not hard vetoed
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
                        'type': 'PRIMARY_SHORT',
                    })
            
            elif direction == "short" and result.state in long_states:
                # TACTICAL_PUT: Check for tactical puts in bull states
                tactical_result = tactical_put_inside_bull(result, row)
                if tactical_result.active:
                    str_label = "FULL" if tactical_result.strength == 2 else "PARTIAL"
                    setups.append({
                        'date': date_str,
                        'state': result.state.value,
                        'score': result.score,
                        'confidence': result.confidence.value,
                        'overlay': f"{str_label}: {tactical_result.reason[tactical_result.reason.find('MVRV'):][:35]}",
                        'type': 'TACTICAL_PUT',
                    })
            
            if direction == "short":
                # Option put signals (independent of fusion state)
                if int(row.get('signal_option_put', 0)) == 1:
                    setups.append({
                        'date': date_str,
                        'state': result.state.value,
                        'score': result.score,
                        'confidence': result.confidence.value,
                        'overlay': 'MVRV hot + greed',
                        'type': 'OPTION_PUT',
                    })
            
            elif direction == "all":
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
                        'type': 'LONG',
                    })
                    
                    # Also check tactical puts
                    tactical_result = tactical_put_inside_bull(result, row)
                    if tactical_result.active:
                        str_label = "FULL" if tactical_result.strength == 2 else "PARTIAL"
                        setups.append({
                            'date': date_str,
                            'state': result.state.value,
                            'score': result.score,
                            'confidence': result.confidence.value,
                            'overlay': f"{str_label}",
                            'type': 'TACTICAL_PUT',
                        })
                        
                elif is_short and (not overlay.short_veto_active or 'SOFT' in overlay.reason):
                    overlay_status = "Clean" if not overlay.short_veto_active else "SOFT VETO"
                    setups.append({
                        'date': date_str,
                        'state': result.state.value,
                        'score': result.score,
                        'confidence': result.confidence.value,
                        'overlay': overlay_status,
                        'type': 'PRIMARY_SHORT',
                    })
                
                # Option signals (tracked for all states)
                if int(row.get('signal_option_call', 0)) == 1:
                    setups.append({
                        'date': date_str,
                        'state': result.state.value,
                        'score': result.score,
                        'confidence': result.confidence.value,
                        'overlay': 'MVRV cheap + fear',
                        'type': 'OPTION_CALL',
                    })
                if int(row.get('signal_option_put', 0)) == 1:
                    setups.append({
                        'date': date_str,
                        'state': result.state.value,
                        'score': result.score,
                        'confidence': result.confidence.value,
                        'overlay': 'MVRV hot + greed',
                        'type': 'OPTION_PUT',
                    })
        
        # Count by type
        primary_short_count = sum(1 for s in setups if s['type'] == 'PRIMARY_SHORT')
        tactical_put_count = sum(1 for s in setups if s['type'] == 'TACTICAL_PUT')
        long_count = sum(1 for s in setups if s['type'] in ('LONG', 'long'))
        option_call_count = sum(1 for s in setups if s['type'] == 'OPTION_CALL')
        option_put_count = sum(1 for s in setups if s['type'] == 'OPTION_PUT')
        
        # Print results
        dir_emoji = "🟢" if direction == "long" else "🔴" if direction == "short" else "⚪"
        self.stdout.write(f"\n{dir_emoji} LAST {n} {direction.upper()} SETUPS (After Overlay Filter)")
        self.stdout.write("=" * 80)
        
        if direction == "short":
            self.stdout.write(f"\nPRIMARY_SHORT (from short states): {primary_short_count}")
            self.stdout.write(f"TACTICAL_PUT (from bull states):   {tactical_put_count}")
            self.stdout.write(f"OPTION_PUT (rule-based):           {option_put_count}")
            self.stdout.write(f"TOTAL SHORT-RELATED ACTIONS:       {len(setups)}")
        elif direction == "long":
            self.stdout.write(f"\nTotal long setups: {long_count}")
            if tactical_put_count > 0:
                self.stdout.write(f"(also {tactical_put_count} tactical puts shown)")
            if option_call_count > 0:
                self.stdout.write(f"OPTION_CALL (rule-based): {option_call_count}")
        else:
            self.stdout.write(f"\nTotal setups: {len(setups)}")
            self.stdout.write(f"  LONG: {long_count} | PRIMARY_SHORT: {primary_short_count} | TACTICAL_PUT: {tactical_put_count}")
            self.stdout.write(f"  OPTION_CALL: {option_call_count} | OPTION_PUT: {option_put_count}")
        
        self.stdout.write("")
        
        for sig in setups[-n:]:
            type_label = f"[{sig['type']:12s}]"
            self.stdout.write(f"  {sig['date']} | {type_label} | {sig['state']:20s} | score: {sig['score']:+d} | {sig['confidence']:6s} | {sig['overlay']}")
        
        self.stdout.write("\nDone.\n")

    def _show_explain(self, df: pd.DataFrame, target_date: str = None):
        """Show detailed per-horizon breakdown explaining fusion score for a specific date."""
        from datetime import datetime
        
        # Get target row
        if target_date:
            try:
                target_dt = datetime.strptime(target_date, "%Y-%m-%d")
                target_idx = target_dt.strftime("%Y-%m-%d")
                if target_idx not in df.index.astype(str).tolist():
                    # Try finding by date
                    matching = [idx for idx in df.index if str(idx)[:10] == target_idx]
                    if matching:
                        row = df.loc[matching[0]]
                        date_str = str(matching[0])[:10]
                    else:
                        self.stderr.write(f"Date {target_date} not found in data")
                        return
                else:
                    row = df.loc[target_idx]
                    date_str = target_idx
            except ValueError:
                self.stderr.write(f"Invalid date format: {target_date}. Use YYYY-MM-DD")
                return
        else:
            row = df.iloc[-1]
            date_str = str(df.index[-1])[:10]
        
        # Compute fusion result
        result = fuse_signals(row)
        score, components = result.score, result.components
        
        # Helper function to format bucket value
        def fmt_bucket(val):
            if val > 0:
                return f"+{int(val)}"
            return f"{int(val):2d}"
        
        def bucket_label(val):
            if val > 0:
                return "accumulation" if val == 1 else "strong_accum"
            elif val < 0:
                return "distribution" if val == -1 else "strong_distrib"
            return "neutral"
        
        def trend_label(val):
            if val > 0:
                return "rising"
            elif val < 0:
                return "falling"
            return "flat"
        
        # === HEADER ===
        self.stdout.write("\n" + "═" * 70)
        self.stdout.write(f"FUSION EXPLANATION: {date_str} | {result.state.value.upper()}")
        self.stdout.write("═" * 70)
        
        # === OVERALL SCORE ===
        self.stdout.write(f"\nCONFIDENCE SCORE: {score:+d} → {result.confidence.value.upper()}")
        
        c = components
        mdia_label = 'strong_inflow' if c.get('mdia_strong') else 'inflow' if c.get('mdia_inflow') else 'aging' if c.get('mdia_aging') else 'neutral/distrib'
        whale_label = 'strong_distrib' if c.get('whale_distrib_strong') else 'distrib' if c.get('whale_distrib') else 'sponsored' if c.get('whale_sponsored') else 'mixed' if c.get('whale_mixed') else 'neutral'
        mvrv_label = 'bullish' if c.get('mvrv_macro_bullish') else 'bearish' if c.get('mvrv_macro_bearish') else 'neutral'
        
        self.stdout.write(f"├── MDIA:     ({mdia_label})")
        self.stdout.write(f"├── Whale:    ({whale_label})")
        self.stdout.write(f"└── MVRV-LS:  ({mvrv_label})")
        
        # === SHORT SOURCE (if applicable) ===
        if result.short_source:
            self.stdout.write(f"\n📊 SHORT SOURCE: {result.short_source.upper()}")
            # (Score-based short fallback was removed in v1 redesign)
        
        # === MDIA BREAKDOWN ===
        self.stdout.write("\n" + "─" * 70)
        self.stdout.write("MDIA BREAKDOWN (Capital Flow / Aging)")
        self.stdout.write("─" * 70)
        
        # Get MDIA buckets if available
        mdia_horizons = [1, 2, 4, 7]
        mdia_buckets = []
        for h in mdia_horizons:
            bucket_col = f'mdia_bucket_{h}d'
            z_col = f'mdia_slope_z_{h}d'
            if bucket_col in row:
                bucket = int(row.get(bucket_col, 0))
                z = row.get(z_col, 0)
                z_str = f"z={z:6.2f}" if not pd.isna(z) else "z=  N/A"
                label = "inflow" if bucket <= -1 else ("aging" if bucket >= 1 else "neutral")
                self.stdout.write(f"Bucket {h}d: {fmt_bucket(bucket):>3} ({label:8}) {z_str}")
                mdia_buckets.append(bucket)
        
        # Breadth calculation
        if mdia_buckets:
            inflow_count = sum(1 for b in mdia_buckets if b <= -1)
            self.stdout.write(f"\nBreadth: {inflow_count} of {len(mdia_buckets)} horizons showing inflow")
            
            # Current regime determination
            strong_inflow = row.get('mdia_regime_strong_inflow', 0) == 1
            inflow = row.get('mdia_regime_inflow', 0) == 1
            aging = row.get('mdia_regime_aging', 0) == 1
            
            if strong_inflow:
                self.stdout.write("→ Current Regime: STRONG_INFLOW")
            elif inflow:
                self.stdout.write("→ Current Regime: INFLOW")
            elif aging:
                self.stdout.write("→ Current Regime: AGING (trend-unfriendly)")
            else:
                self.stdout.write("→ Current Regime: NEUTRAL/DISTRIB")
        else:
            self.stdout.write("(MDIA bucket features not found in data)")
        
        # === WHALE BREAKDOWN ===
        self.stdout.write("\n" + "─" * 70)
        self.stdout.write("WHALE BREAKDOWN (Smart Money Intent)")
        self.stdout.write("─" * 70)
        mvrv_val = row.get('mvrv_60d', 0.0)
        self.stdout.write(f"Context: MVRV-60d = {mvrv_val:.2f} (used for profit-taking logic)")
        
        whale_horizons = [1, 2, 4, 7]
        
        for group, group_label in [('mega', 'MEGA WHALES (100-10k BTC)'), ('small', 'SMALL WHALES (1-100 BTC)')]:
            self.stdout.write(f"\n{group_label}")
            
            buckets = []
            for h in whale_horizons:
                bucket_col = f'whale_{group}_bucket_{h}d'
                z_col = f'whale_{group}_delta_z_{h}d'
                if bucket_col in row:
                    bucket = int(row.get(bucket_col, 0))
                    z = row.get(z_col, 0)
                    z_str = f"z={z:6.2f}" if not pd.isna(z) else "z=  N/A"
                    prefix = "├──" if h != whale_horizons[-1] else "└──"
                    self.stdout.write(f"{prefix} {h}d: {fmt_bucket(bucket):>3} ({bucket_label(bucket):12}) {z_str}")
                    buckets.append(bucket)
            
            if buckets:
                accum_count = sum(1 for b in buckets if b > 0)
                distrib_count = sum(1 for b in buckets if b < 0)
                any_accum = (accum_count >= 2) and (distrib_count == 0)
                any_distrib = (distrib_count >= 2) and (accum_count == 0)
                self.stdout.write(f"    accum_count={accum_count}, distrib_count={distrib_count} → any_accum={any_accum}")
        
        # Cross-group regime
        self.stdout.write(f"\n→ Cross-Group Regime: {whale_label.upper()}")
        
        # === MVRV-LS BREAKDOWN ===
        self.stdout.write("\n" + "─" * 70)
        self.stdout.write("MVRV-LS BREAKDOWN (Macro Terrain)")
        self.stdout.write("─" * 70)
        
        # Level info
        level = row.get('mvrv_ls_level', 0)
        pct = row.get('mvrv_ls_roll_pct_365d', 0.5)
        z_365 = row.get('mvrv_ls_z_score_365d', 0.0)
        level_labels = {
            -2: 'capitulation',
            -1: 'late_bear/recovery',
             0: 'neutral',
             1: 'expansion',
             2: 'overheated'
        }
        mvrv_val = row.get('mvrv_60d', 0.0)
        self.stdout.write(f"Level: {int(level)} ({level_labels.get(int(level), 'unknown')}) | Z-365d: {z_365:.2f} | Percentile: {pct*100:.0f}%")
        
        # Trend buckets
        self.stdout.write("\nTrend Buckets:")
        mvrv_horizons = [2, 4, 7, 14]
        rising_count = 0
        falling_count = 0
        
        for h in mvrv_horizons:
            trend_col = f'mvrv_ls_trend_{h}d'
            z_col = f'mvrv_ls_delta_z_{h}d'
            if trend_col in row:
                trend = int(row.get(trend_col, 0))
                z = row.get(z_col, 0)
                z_str = f"z={z:6.2f}" if not pd.isna(z) else "z=  N/A"
                prefix = "├──" if h != mvrv_horizons[-1] else "└──"
                self.stdout.write(f"{prefix} {h:2d}d: {fmt_bucket(trend):>3} ({trend_label(trend):8}) {z_str}")
                if trend > 0:
                    rising_count += 1
                elif trend < 0:
                    falling_count += 1
        
        self.stdout.write(f"    rising={rising_count}, falling={falling_count}")
        
        # Current regime
        self.stdout.write(f"\n→ Current Macro Regime: {mvrv_label.upper()}")
        
        # === CLASSIFICATION TRACE ===
        self.stdout.write("\n" + "─" * 70)
        self.stdout.write("CLASSIFICATION TRACE (Empirical Hierarchy)")
        self.stdout.write("─" * 70)
        
        mdia_strong = c.get('mdia_strong', 0) == 1
        mdia_inflow = c.get('mdia_inflow', 0) == 1
        mdia_non_inflow = not mdia_inflow
        
        whale_sponsored = c.get('whale_sponsored', 0) == 1
        whale_mixed = c.get('whale_mixed', 0) == 1
        whale_distrib = c.get('whale_distrib', 0) == 1
        whale_distrib_strong = c.get('whale_distrib_strong', 0) == 1
        
        mvrv_macro_bullish = c.get('mvrv_macro_bullish', 0) == 1
        mvrv_recovery = row.get('mvrv_ls_regime_call_confirm_recovery', 0) == 1
        mvrv_macro_neutral = c.get('mvrv_macro_neutral', 0) == 1
        mvrv_put_or_bear = row.get('mvrv_ls_regime_put_confirm', 0) == 1 or row.get('mvrv_ls_regime_bear_continuation', 0) == 1
        
        rules = [
            ("STRONG_BULLISH", mdia_strong and whale_sponsored and mvrv_macro_bullish,
             f"mdia_strong={mdia_strong}, whale_sponsored={whale_sponsored}, macro_bullish={mvrv_macro_bullish}"),
            ("EARLY_RECOVERY", mdia_inflow and whale_sponsored and mvrv_recovery,
             f"mdia_inflow={mdia_inflow}, whale_sponsored={whale_sponsored}, mvrv_recovery={mvrv_recovery}"),
            ("BEAR_CONTINUATION", mdia_non_inflow and whale_distrib and mvrv_put_or_bear,
             f"not_mdia_inflow={mdia_non_inflow}, whale_distrib={whale_distrib}, mvrv_put/bear={mvrv_put_or_bear}"),
            ("BEAR_PROBE", mdia_non_inflow and whale_distrib_strong and mvrv_macro_neutral,
             f"not_mdia_inflow={mdia_non_inflow}, whale_distrib_strong={whale_distrib_strong}, macro_neutral={mvrv_macro_neutral}"),
            ("DISTRIBUTION_RISK", mdia_non_inflow and whale_distrib and not mvrv_macro_bullish,
             f"not_mdia_inflow={mdia_non_inflow}, whale_distrib={whale_distrib}, not_macro_bullish={not mvrv_macro_bullish}"),
            ("MOMENTUM_CONTINUATION", mdia_inflow and (whale_sponsored or whale_mixed) and mvrv_macro_bullish,
             f"mdia_inflow={mdia_inflow}, whale_sponsored/mixed={(whale_sponsored or whale_mixed)}, macro_bullish={mvrv_macro_bullish}"),
            ("BULL_PROBE", mdia_inflow and whale_sponsored and mvrv_macro_neutral,
             f"mdia_inflow={mdia_inflow}, whale_sponsored={whale_sponsored}, macro_neutral={mvrv_macro_neutral}"),
        ]
        
        matched = None
        for rule_name, condition, details in rules:
            status = "✓" if condition else "✗"
            self.stdout.write(f"{status} {rule_name:25s} {details}")
            if condition and matched is None:
                matched = rule_name
        
        if matched:
            self.stdout.write(f"\n→ Matched: {matched}")
        else:
            self.stdout.write(f"\n→ Fallback: NO_TRADE (no conditions matched)")
        
        # === OPTION SIGNALS ===
        signal_call = int(row.get('signal_option_call', 0))
        signal_put = int(row.get('signal_option_put', 0))
        
        if signal_call or signal_put:
            self.stdout.write("\n" + "─" * 70)
            self.stdout.write("OPTION SIGNALS (Rule-based, from interactions.py)")
            self.stdout.write("─" * 70)
            
            call_emoji = "📗 FIRED" if signal_call else "  off"
            put_emoji = "📕 FIRED" if signal_put else "  off"
            self.stdout.write(f"signal_option_call: {call_emoji}  (MVRV cheap 2+ flags + sentiment fear)")
            self.stdout.write(f"signal_option_put:  {put_emoji}  (MVRV overheated + sentiment greed + whale distrib)")
            
            # Show underlying values
            sent = row.get('sentiment_norm', 0)
            mvrv_flags = f"{int(row.get('mvrv_comp_undervalued_90d', 0))}/{int(row.get('mvrv_comp_new_low_180d', 0))}/{int(row.get('mvrv_comp_near_bottom_any', 0))}"
            mvrv_pct = row.get('mvrv_60d_pct', 0)
            self.stdout.write(f"  sentiment_norm: {sent:+.2f} | MVRV flags: {mvrv_flags} | 60d_pct: {mvrv_pct:.2f}")
        
        self.stdout.write("\n" + "═" * 70)
        self.stdout.write("Done.\n")

