# signals/management/commands/dev_signal_cycle.py
"""
Development command: full signal lifecycle in one shot.

Generates signal → persists → builds trade setup → sends Telegram notification.
Works with historical dates. If option chain data doesn't exist for the date,
the setup section is left empty with a warning (signal + notification still fire).

Usage:
    # Full cycle for a specific date
    python manage.py dev_signal_cycle --date 2026-06-10 --notify

    # Dry run (show what would happen, don't persist or notify)
    python manage.py dev_signal_cycle --date 2026-06-10 --dry-run

    # Force a specific signal type (bypass fusion engine decision)
    python manage.py dev_signal_cycle --date 2026-06-10 --force-signal CALL --notify

    # Open paper trade from the setup (requires option chain data)
    python manage.py dev_signal_cycle --date 2026-06-10 --notify --paper-trade

    # Skip setup entirely (just signal + notification)
    python manage.py dev_signal_cycle --date 2026-06-10 --notify --no-setup
"""
from datetime import datetime, date, timezone
from decimal import Decimal

from django.core.management.base import BaseCommand

from signals.services import SignalService


SIGNAL_TYPES = [
    "CALL", "PUT", "OPTION_CALL", "OPTION_PUT", "TACTICAL_PUT",
    "BULL_PROBE", "BEAR_PROBE", "MVRV_SHORT", "IRON_CONDOR",
    "BULL_PUT_SPREAD", "BEAR_CALL_SPREAD",
]


class Command(BaseCommand):
    help = "Dev workflow: generate signal → persist → build setup → notify Telegram → open paper trade"

    def add_arguments(self, parser):
        parser.add_argument(
            "--date",
            type=str,
            required=True,
            help="Target date (YYYY-MM-DD)",
        )
        parser.add_argument(
            "--force-signal",
            type=str,
            choices=SIGNAL_TYPES,
            default=None,
            dest="force_signal",
            help="Force a specific trade decision (bypasses fusion priority waterfall)",
        )
        parser.add_argument(
            "--notify",
            action="store_true",
            help="Send Telegram notification",
        )
        parser.add_argument(
            "--paper-trade",
            action="store_true",
            dest="paper_trade",
            help="Open a paper trade using the setup (requires option chain data)",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            dest="dry_run",
            help="Show what would happen without persisting, notifying, or trading",
        )
        parser.add_argument(
            "--no-setup",
            action="store_true",
            dest="no_setup",
            help="Skip trade setup generation (just signal + notification)",
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Print detailed output",
        )
        parser.add_argument(
            "--long_model",
            type=str,
            default="models/long_model.joblib",
            help="Path to long model",
        )
        parser.add_argument(
            "--short_model",
            type=str,
            default="models/short_model.joblib",
            help="Path to short model",
        )

    def handle(self, *args, **options):
        target_date = datetime.strptime(options["date"], "%Y-%m-%d").date()
        dry_run = options["dry_run"]
        force_signal = options.get("force_signal")
        notify = options["notify"]
        paper_trade = options["paper_trade"]
        no_setup = options["no_setup"]
        verbose = options["verbose"]

        if dry_run:
            self.stdout.write(self.style.WARNING("=== DRY RUN MODE ===\n"))

        # ─── STEP 1: Generate Signal ───────────────────────────────────────
        self.stdout.write(self.style.MIGRATE_HEADING("STEP 1: Generate Signal"))

        service = SignalService(
            long_model_path=options["long_model"],
            short_model_path=options["short_model"],
        )

        try:
            results = service.generate_all_signals(target_date)
        except ValueError as e:
            self.stderr.write(self.style.ERROR(f"Signal generation failed: {e}"))
            return

        # Pick the target result
        if force_signal:
            # Find matching result, or use the first one but override decision
            target_result = None
            for r in results:
                if r.trade_decision == force_signal:
                    target_result = r
                    break

            if target_result is None:
                # Force: take the primary result and override its decision
                target_result = results[0]
                self.stdout.write(self.style.WARNING(
                    f"  Fusion produced '{target_result.trade_decision}', "
                    f"forcing to '{force_signal}'"
                ))
                target_result = self._override_decision(target_result, force_signal)
        else:
            # Pick highest priority tradeable signal
            tradeable = [r for r in results if r.trade_decision != "NO_TRADE"]
            target_result = tradeable[0] if tradeable else results[0]

        self._print_signal_summary(target_result)

        # ─── STEP 2: Persist Signal ───────────────────────────────────────
        self.stdout.write(self.style.MIGRATE_HEADING("\nSTEP 2: Persist Signal"))

        if dry_run:
            self.stdout.write("  [skipped — dry run]")
            signal = None
        else:
            signal, created = service.persist_signal(target_result)
            status = "CREATED" if created else "UPDATED"
            self.stdout.write(self.style.SUCCESS(
                f"  ✓ DailySignal {status}: {signal.date} | {signal.trade_decision}"
            ))

        # ─── STEP 3: Build Trade Setup ────────────────────────────────────
        self.stdout.write(self.style.MIGRATE_HEADING("\nSTEP 3: Build Trade Setup"))

        setup = None
        setup_available = False

        if no_setup:
            self.stdout.write("  [skipped — --no-setup flag]")
        elif target_result.trade_decision == "NO_TRADE":
            self.stdout.write("  [skipped — NO_TRADE signal]")
        elif target_result.trade_decision in ("BULL_PUT_SPREAD", "BEAR_CALL_SPREAD"):
            # Income spreads are produced by the income gate, not TradeSetupBuilder.
            # The risk-tiered setups are already attached to the SignalResult.
            income_setups = target_result.income_spread_setups or []
            if income_setups:
                setup_available = True
                self._print_income_setups(target_result)
                # TradeSetupSnapshot persistence is handled by notifier.send_from_model
            else:
                self.stdout.write(self.style.WARNING(
                    "  ⚠ Income gate produced no tradable spreads for this date.\n"
                    "    (regime may be eligible, but no chain setup passed the filters)"
                ))
        else:
            setup = self._build_setup(target_date, target_result.trade_decision, verbose)
            if setup:
                setup_available = True
                self._print_setup_summary(setup)
                # Persist setup to DB
                if not dry_run and signal:
                    try:
                        setup.save_to_db(signal=signal)
                        self.stdout.write(self.style.SUCCESS(
                            "  ✓ TradeSetupSnapshot saved"
                        ))
                    except Exception as e:
                        self.stdout.write(self.style.WARNING(
                            f"  ⚠ Setup save failed: {e}"
                        ))
            else:
                self.stdout.write(self.style.WARNING(
                    "  ⚠ No option chain data available for this date.\n"
                    "    Setup section is empty. Run 'manage.py collect_options' "
                    "    to populate option snapshots first."
                ))

        # ─── STEP 4: Send Telegram Notification ───────────────────────────
        self.stdout.write(self.style.MIGRATE_HEADING("\nSTEP 4: Telegram Notification"))

        if not notify:
            self.stdout.write("  [skipped — --notify not set]")
        elif dry_run:
            self.stdout.write("  [skipped — dry run]")
            if signal is None:
                self.stdout.write("  Would send notification for:")
                self.stdout.write(f"    {target_result.trade_decision} | {target_result.fusion_state}")
        elif target_result.trade_decision == "NO_TRADE":
            self.stdout.write("  [skipped — NO_TRADE signal]")
        else:
            self._send_notification(signal, setup_available)

        # ─── STEP 5: Open Paper Trade ─────────────────────────────────────
        self.stdout.write(self.style.MIGRATE_HEADING("\nSTEP 5: Paper Trade"))

        if not paper_trade:
            self.stdout.write("  [skipped — --paper-trade not set]")
        elif dry_run:
            self.stdout.write("  [skipped — dry run]")
        elif target_result.trade_decision in ("BULL_PUT_SPREAD", "BEAR_CALL_SPREAD"):
            self.stdout.write(self.style.WARNING(
                "  ⚠ Paper trade not supported for income spreads via this command."
            ))
        elif not setup_available:
            self.stdout.write(self.style.WARNING(
                "  ⚠ Cannot open paper trade without option chain data.\n"
                "    Run 'manage.py collect_options --exchange deribit' first."
            ))
        else:
            self._open_paper_trade(target_result, setup, target_date)

        # ─── Summary ──────────────────────────────────────────────────────
        self.stdout.write(self.style.MIGRATE_HEADING("\n" + "=" * 60))
        self.stdout.write(f"  Date:     {target_date}")
        self.stdout.write(f"  Signal:   {target_result.trade_decision}")
        self.stdout.write(f"  State:    {target_result.fusion_state}")
        self.stdout.write(f"  Size:     {target_result.effective_size:.2f}")
        self.stdout.write(f"  Setup:    {'✓' if setup_available else '✗ (no option data)'}")
        self.stdout.write(f"  Notify:   {'✓' if notify and not dry_run and target_result.trade_decision != 'NO_TRADE' else '✗'}")
        self.stdout.write(f"  Paper:    {'✓' if paper_trade and setup_available and not dry_run else '✗'}")
        self.stdout.write("=" * 60 + "\n")

    def _override_decision(self, result, forced_decision: str):
        """
        Override the trade_decision on a SignalResult.
        Keeps all other fields (fusion state, ML probs, etc.) intact.
        """
        from signals.services import SignalResult

        # Recompute effective size based on forced decision
        from signals.services import (
            OPTION_SIGNAL_SIZE_MULT, MVRV_SHORT_SIZE_MULT,
        )
        if forced_decision in ("OPTION_CALL", "OPTION_PUT"):
            effective_size = OPTION_SIGNAL_SIZE_MULT
        elif forced_decision == "MVRV_SHORT":
            effective_size = MVRV_SHORT_SIZE_MULT
        elif forced_decision == "IRON_CONDOR":
            effective_size = 0.50
        elif forced_decision == "NO_TRADE":
            effective_size = 0.0
        else:
            effective_size = result.size_multiplier

        # Get strategy summary for the forced decision
        from signals.options import get_decision_strategy_summary, DECISION_STRATEGY_MAP
        from execution.services.policy import get_policy

        policy = get_policy()
        policy_exit = policy.get_exit_params(forced_decision)

        if forced_decision in DECISION_STRATEGY_MAP:
            strategy_summary = get_decision_strategy_summary(forced_decision)
        else:
            strategy_summary = {
                "primary_structures": result.option_structures,
                "strike_guidance": result.strike_guidance,
                "dte_range": result.dte_range,
                "rationale": result.strategy_rationale,
                "stop_loss": result.stop_loss,
                "stop_loss_pct": result.stop_loss_pct,
                "scale_down_day": result.scale_down_day,
                "max_hold_days": result.max_hold_days,
                "spread_width_pct": result.spread_width_pct,
                "take_profit_pct": result.take_profit_pct,
            }

        if policy_exit is not None:
            strategy_summary["stop_loss_pct"] = policy_exit.stop_loss_pct
            strategy_summary["take_profit_pct"] = policy_exit.take_profit_pct
            strategy_summary["max_hold_days"] = policy_exit.max_hold_days
            strategy_summary["scale_down_day"] = policy_exit.scale_down_day
        strategy_summary["spread_width_pct"] = policy.get_spread_width(forced_decision)
        dte_cfg = policy.get_dte_target(forced_decision)
        strategy_summary["dte_range"] = f"{dte_cfg.min_dte}-{dte_cfg.max_dte}d"

        return SignalResult(
            date=result.date,
            p_long=result.p_long,
            p_short=result.p_short,
            signal_option_call=result.signal_option_call,
            signal_option_put=result.signal_option_put,
            fusion_state=result.fusion_state,
            fusion_confidence=result.fusion_confidence,
            fusion_score=result.fusion_score,
            overlay_reason=result.overlay_reason,
            size_multiplier=effective_size,
            dte_multiplier=result.dte_multiplier,
            tactical_put_active=result.tactical_put_active,
            tactical_put_strategy=result.tactical_put_strategy,
            tactical_put_size=result.tactical_put_size,
            trade_decision=forced_decision,
            trade_notes=f"[FORCED] Original: {result.trade_decision}",
            option_structures=strategy_summary.get("primary_structures", ""),
            strike_guidance=strategy_summary.get("strike_guidance", ""),
            dte_range=strategy_summary.get("dte_range", ""),
            strategy_rationale=strategy_summary.get("rationale", ""),
            stop_loss=strategy_summary.get("stop_loss", ""),
            stop_loss_pct=strategy_summary.get("stop_loss_pct"),
            scale_down_day=strategy_summary.get("scale_down_day"),
            max_hold_days=strategy_summary.get("max_hold_days"),
            spread_width_pct=strategy_summary.get("spread_width_pct"),
            take_profit_pct=strategy_summary.get("take_profit_pct"),
            no_trade_reasons=[],
            decision_trace=[f"FORCED: {forced_decision}"],
            score_components=result.score_components,
            effective_size=effective_size,
            decision_version=result.decision_version,
            model_versions=result.model_versions,
            short_source=result.short_source,
            condor_score=result.condor_score,
            condor_eligible=result.condor_eligible,
            condor_veto_reasons=result.condor_veto_reasons,
            condor_score_components=result.condor_score_components,
            condor_short_call=result.condor_short_call,
            condor_short_put=result.condor_short_put,
            condor_cost_basis=result.condor_cost_basis,
            condor_strike_meta=result.condor_strike_meta,
            income_spread_setups=result.income_spread_setups,
            income_spread_score=result.income_spread_score,
            income_spread_eligible=result.income_spread_eligible,
            income_spread_veto_reasons=result.income_spread_veto_reasons,
        )

    def _print_signal_summary(self, result):
        """Print compact signal summary."""
        self.stdout.write(f"  Date:       {result.date}")
        self.stdout.write(f"  Decision:   {result.trade_decision}")
        self.stdout.write(f"  State:      {result.fusion_state} (score={result.fusion_score:+d})")
        self.stdout.write(f"  Confidence: {result.fusion_confidence}")
        self.stdout.write(f"  Size:       {result.effective_size:.2f}")
        self.stdout.write(f"  ML:         p_long={result.p_long:.3f}, p_short={result.p_short:.3f}")
        if result.trade_notes:
            self.stdout.write(f"  Notes:      {result.trade_notes}")

    def _build_setup(self, signal_date: date, signal_type: str, verbose: bool):
        """Build trade setup from option chain data. Returns None if no data."""
        try:
            from execution.services.trade_setup import TradeSetupBuilder
            builder = TradeSetupBuilder()
            setup = builder.build_setup(signal_date, signal_type=signal_type)
            return setup
        except Exception as e:
            if verbose:
                self.stdout.write(self.style.WARNING(f"  Setup build error: {e}"))
            return None

    def _print_setup_summary(self, setup):
        """Print compact setup summary."""
        self.stdout.write(f"  Direction:  {setup.direction}")
        self.stdout.write(f"  Expiry:     {setup.expiry} (DTE: {setup.dte})")
        self.stdout.write(f"  Long Leg:   {setup.long_leg.symbol} @ ${setup.long_leg.price:,.2f} (Δ{setup.long_leg.delta:.3f})")
        self.stdout.write(f"  Short Leg:  {setup.short_leg.symbol} @ ${setup.short_leg.price:,.2f} (Δ{setup.short_leg.delta:.3f})")
        self.stdout.write(f"  Width:      ${setup.spread_width:,.0f} ({setup.spread_width_pct*100:.1f}%)")
        self.stdout.write(f"  Net Debit:  ${setup.net_debit:,.2f}")
        self.stdout.write(f"  Max Profit: ${setup.max_profit:,.2f}")
        self.stdout.write(f"  Max Loss:   ${setup.max_loss:,.2f}")
        self.stdout.write(f"  R:R:        1:{setup.risk_reward:.2f}")
        self.stdout.write(f"  Contracts:  {setup.contracts}")
        if not setup.validation_passed:
            self.stdout.write(self.style.ERROR(
                f"  ⚠ Validation: FAILED ({', '.join(setup.validation_blocking)})"
            ))
        elif setup.validation_warnings:
            self.stdout.write(self.style.WARNING(
                f"  ⚠ Warnings: {', '.join(setup.validation_warnings)}"
            ))

    def _print_income_setups(self, result):
        """Print income spread risk-tiered setups (sourced from the income gate)."""
        is_put = result.trade_decision == "BULL_PUT_SPREAD"
        short_label = "Short Put " if is_put else "Short Call"
        long_label = "Long Put  " if is_put else "Long Call "
        self.stdout.write(f"  Income gate score: {result.income_spread_score:.0f}/100")
        self.stdout.write(f"  Tradable setups:   {len(result.income_spread_setups)}")
        tier_labels = {"low": "🟢 LOW RISK", "medium": "🟡 MEDIUM RISK", "high": "🔴 HIGH RISK"}
        for s in result.income_spread_setups:
            tier = tier_labels.get(s.get("risk_tier"), str(s.get("risk_tier")).upper())
            delta = s.get("short_delta", 0)
            pop = (1 - delta) * 100
            self.stdout.write(f"\n  {tier} (Δ{delta:.2f}, ~{pop:.0f}% POP)")
            self.stdout.write(f"    {short_label}: ${s['short_strike']:,.0f} ({s['otm_pct']*100:.1f}% OTM)")
            self.stdout.write(f"    {long_label}: ${s['long_strike']:,.0f}")
            self.stdout.write(f"    Width:      ${s['spread_width']:,.0f}")
            self.stdout.write(f"    Credit:     ${s['credit']:,.2f} ({s['credit_width_pct']*100:.1f}% of width)")
            self.stdout.write(f"    Max Loss:   ${s['max_loss']:,.2f}")
            self.stdout.write(f"    DTE:        {s['dte']}d | R:R: 1:{s['risk_reward']:.2f}")

    def _send_notification(self, signal, setup_available: bool):
        """Send Telegram notification."""
        try:
            from notifications.notifier import TelegramNotifier
            notifier = TelegramNotifier()
            success = notifier.send_from_model(signal, include_setup=setup_available)

            if success:
                self.stdout.write(self.style.SUCCESS(
                    f"  ✓ Telegram sent: {signal.trade_decision}"
                ))
            else:
                self.stdout.write(self.style.WARNING(
                    "  ✗ Telegram send returned False (NO_TRADE or error)"
                ))
        except ValueError as e:
            self.stdout.write(self.style.ERROR(
                f"  ✗ Telegram not configured: {e}"
            ))
        except Exception as e:
            self.stdout.write(self.style.ERROR(
                f"  ✗ Telegram error: {e}"
            ))

    def _open_paper_trade(self, result, setup, signal_date: date):
        """Open a paper trade using the setup's primary leg."""
        try:
            from datafeed.services.trade_tracker import TradeTracker

            tracker = TradeTracker()

            # Use the long leg as entry (buying the spread)
            entry_price = Decimal(str(setup.net_debit))
            entry_spot = Decimal(str(setup.spot_price))
            symbol = setup.long_leg.symbol
            qty = Decimal(str(setup.contracts))

            # Entry IV from long leg
            entry_iv = Decimal(str(setup.long_leg.iv)) if setup.long_leg.iv else None

            # Use midday of signal date as entry timestamp
            entry_ts = datetime.combine(
                signal_date, datetime.min.time().replace(hour=12)
            ).replace(tzinfo=timezone.utc)

            trade = tracker.open_trade(
                signal_type=result.trade_decision,
                direction="LONG" if setup.direction in ("LONG", "NEUTRAL") else "SHORT",
                symbol=symbol,
                qty=qty,
                entry_price=entry_price,
                entry_spot=entry_spot,
                entry_iv=entry_iv,
                entry_timestamp=entry_ts,
                is_paper=True,
                exchange="deribit",
            )

            self.stdout.write(self.style.SUCCESS(
                f"  ✓ Paper trade opened: {trade.trade_id}\n"
                f"    Symbol: {symbol}\n"
                f"    Entry:  ${float(entry_price):,.2f} × {qty} = ${float(trade.notional):,.2f}\n"
                f"    Spot:   ${float(entry_spot):,.0f}"
            ))
        except Exception as e:
            self.stdout.write(self.style.ERROR(
                f"  ✗ Paper trade error: {e}"
            ))
