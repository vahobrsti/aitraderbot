"""
Tests for IBIT wheel short-leg selection and its Telegram formatting.

Fixture quotes are real IBIT SEP-18-'26 bid/ask and open interest pulled from
IBKR (spot ~36.68); deltas are assigned by moneyness to exercise the tier bands.
"""
from datetime import datetime, timezone

import pandas as pd
from django.test import TestCase

from signals.wheel import (
    IBIT_MULTIPLIER,
    select_wheel_legs,
    selection_hash,
    wheel_side_for_decision,
)

SPOT = 36.68
_TS = datetime(2026, 8, 7, 16, 0, tzinfo=timezone.utc)
_EXPIRY = pd.Timestamp("2026-09-18", tz="UTC")


def _row(strike, option_type, delta, bid, ask):
    mid = (bid + ask) / 2
    return {
        "symbol": f"IBIT-18SEP26-{strike:g}-{'P' if option_type == 'put' else 'C'}",
        "exchange": "ibkr",
        "timestamp": _TS,
        "expiry": _EXPIRY,
        "strike": strike,
        "option_type": option_type,
        "delta": delta,
        "bid": bid,
        "ask": ask,
        "spread_pct": (ask - bid) / mid,
        "dte": 42.0,
        "spot_price": SPOT,
    }


def _chain():
    return pd.DataFrame([
        # Puts (real quotes; deltas by moneyness -> low/med/high tiers)
        _row(33, "put", -0.15, 0.53, 0.55),   # 10.0% OTM -> low
        _row(34, "put", -0.22, 0.73, 0.76),   # 7.3% OTM  -> medium
        _row(35, "put", -0.30, 1.01, 1.04),   # 4.6% OTM  -> high
        # Calls
        _row(38, "call", 0.33, 1.14, 1.16),   # 3.6% OTM  -> below 4% min-OTM, filtered out
        _row(39, "call", 0.25, 0.80, 0.83),   # 6.3% OTM  -> medium
        _row(40, "call", 0.18, 0.56, 0.59),   # 9.0% OTM  -> low
    ])


class WheelSideMappingTests(TestCase):
    def test_decision_mapping(self):
        self.assertEqual(wheel_side_for_decision("BULL_PUT_SPREAD"), "put")
        self.assertEqual(wheel_side_for_decision("BEAR_CALL_SPREAD"), "call")
        self.assertIsNone(wheel_side_for_decision("IRON_CONDOR"))
        self.assertIsNone(wheel_side_for_decision(""))


class SelectWheelLegsPutTests(TestCase):
    def setUp(self):
        self.legs = select_wheel_legs(_chain(), "put", SPOT, dte_mode="income")

    def test_three_tiers_selected(self):
        tiers = {leg.risk_tier for leg in self.legs}
        self.assertEqual(tiers, {"low", "medium", "high"})

    def test_strikes_map_to_tiers(self):
        by_tier = {leg.risk_tier: leg for leg in self.legs}
        self.assertEqual(by_tier["low"].strike, 33)
        self.assertEqual(by_tier["medium"].strike, 34)
        self.assertEqual(by_tier["high"].strike, 35)

    def test_positions_are_cash_secured_puts(self):
        for leg in self.legs:
            self.assertEqual(leg.position, "cash_secured_put")
            self.assertEqual(leg.side, "put")

    def test_credit_is_short_bid_and_premium_scaled(self):
        high = next(l for l in self.legs if l.risk_tier == "high")
        self.assertAlmostEqual(high.credit, 1.01)  # short leg bid
        self.assertAlmostEqual(high.premium_usd, 1.01 * IBIT_MULTIPLIER)

    def test_cash_reserve_computed(self):
        high = next(l for l in self.legs if l.risk_tier == "high")
        # strike*100 - premium = 3500 - 101 = 3399
        self.assertAlmostEqual(high.cash_reserve_usd, 3399.0)

    def test_expiry_and_dte_present(self):
        leg = self.legs[0]
        self.assertEqual(leg.expiry, "2026-09-18")
        self.assertEqual(leg.dte, 42)


class SelectWheelLegsCallTests(TestCase):
    def setUp(self):
        self.legs = select_wheel_legs(_chain(), "call", SPOT, dte_mode="income")

    def test_near_money_call_filtered_by_min_otm(self):
        # 38 call is only 3.6% OTM (< 4% min) -> should not appear
        strikes = {leg.strike for leg in self.legs}
        self.assertNotIn(38, strikes)

    def test_covered_calls_have_no_cash_reserve(self):
        for leg in self.legs:
            self.assertEqual(leg.position, "covered_call")
            self.assertIsNone(leg.cash_reserve_usd)

    def test_tiers_present(self):
        by_tier = {leg.risk_tier: leg for leg in self.legs}
        self.assertIn("low", by_tier)      # 40 call
        self.assertIn("medium", by_tier)   # 39 call
        self.assertEqual(by_tier["low"].strike, 40)
        self.assertEqual(by_tier["medium"].strike, 39)


class SelectWheelLegsEdgeTests(TestCase):
    def test_empty_chain_returns_empty(self):
        self.assertEqual(select_wheel_legs(pd.DataFrame(), "put", SPOT), [])

    def test_invalid_side_raises(self):
        with self.assertRaises(ValueError):
            select_wheel_legs(_chain(), "straddle", SPOT)

    def test_dte_out_of_window_yields_nothing(self):
        # income window is 21-45d; a 5d chain should select nothing
        df = _chain()
        df["dte"] = 5.0
        self.assertEqual(select_wheel_legs(df, "put", SPOT, dte_mode="income"), [])


class SelectionHashTests(TestCase):
    def test_hash_is_stable_and_order_independent(self):
        legs = select_wheel_legs(_chain(), "put", SPOT, dte_mode="income")
        h1 = selection_hash(legs)
        h2 = selection_hash(list(reversed(legs)))
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 64)

    def test_hash_changes_when_selection_changes(self):
        full = select_wheel_legs(_chain(), "put", SPOT, dte_mode="income")
        fewer = full[:-1]  # drop a tier -> different selection
        self.assertNotEqual(selection_hash(full), selection_hash(fewer))

    def test_empty_selection_has_hash(self):
        self.assertEqual(len(selection_hash([])), 64)


class LegToDictTests(TestCase):
    def test_leg_serializes_to_json_dict(self):
        from signals.wheel import leg_to_dict
        legs = select_wheel_legs(_chain(), "put", SPOT, dte_mode="income")
        d = leg_to_dict(legs[0])
        self.assertEqual(
            set(d),
            {"risk_tier", "side", "position", "strike", "delta", "credit",
             "premium_usd", "otm_pct", "dte", "bid", "ask", "spread_pct",
             "expiry", "symbol", "cash_reserve_usd"},
        )
        self.assertEqual(d["position"], "cash_secured_put")


# ======================================================================
# compute_ibit_wheel command (persist to IbitWheelSetup)
# ======================================================================
from datetime import timedelta

from django.core.management import call_command
from django.utils import timezone

from datafeed.models import OptionSnapshot
from signals.models import DailySignal, IbitWheelSetup


def _make_income_signal(target_date, decision="BULL_PUT_SPREAD"):
    return DailySignal.objects.create(
        date=target_date,
        p_long=0.5, p_short=0.5,
        signal_option_call=0, signal_option_put=0,
        fusion_state="range", fusion_confidence="MEDIUM", fusion_score=0,
        trade_decision=decision,
        income_spread_score=75.0, income_spread_eligible=True,
        income_spread_setups=[],
    )


def _make_ibkr_put(strike, delta, bid, ask):
    from decimal import Decimal
    now = timezone.now()
    expiry = now + timedelta(days=30)
    return OptionSnapshot.objects.create(
        timestamp=now, symbol=f"IBIT-PUT-{strike}", underlying="IBIT",
        expiry=expiry, strike=Decimal(str(strike)), option_type="put",
        spot_price=Decimal(str(SPOT)),
        bid=Decimal(str(bid)), ask=Decimal(str(ask)),
        mid_price=(Decimal(str(bid)) + Decimal(str(ask))) / 2,
        delta=Decimal(str(delta)), exchange="ibkr",
    )


class ComputeIbitWheelCommandTests(TestCase):
    def setUp(self):
        self.date = timezone.now().date()
        _make_income_signal(self.date)
        self.low = _make_ibkr_put(33, -0.15, 0.53, 0.55)
        self.med = _make_ibkr_put(34, -0.22, 0.73, 0.76)
        self.high = _make_ibkr_put(35, -0.30, 1.01, 1.04)

    def test_saves_setup_with_legs(self):
        call_command("compute_ibit_wheel", latest=True)
        setup = IbitWheelSetup.objects.get(
            signal_date=self.date, trade_decision="BULL_PUT_SPREAD"
        )
        self.assertEqual(setup.side, "put")
        self.assertEqual(setup.position, "cash_secured_put")
        self.assertAlmostEqual(setup.spot_price, SPOT)
        self.assertEqual(len(setup.legs), 3)
        self.assertEqual({l["risk_tier"] for l in setup.legs}, {"low", "medium", "high"})

    def test_unchanged_selection_is_not_rewritten(self):
        call_command("compute_ibit_wheel", latest=True)
        setup = IbitWheelSetup.objects.get(signal_date=self.date, trade_decision="BULL_PUT_SPREAD")
        first_updated = setup.updated_at
        call_command("compute_ibit_wheel", latest=True)
        setup.refresh_from_db()
        self.assertEqual(setup.updated_at, first_updated)

    def test_changed_selection_updates_setup(self):
        call_command("compute_ibit_wheel", latest=True)
        setup = IbitWheelSetup.objects.get(signal_date=self.date, trade_decision="BULL_PUT_SPREAD")
        old_hash = setup.selection_hash
        self.high.delete()  # drop the high tier -> selection changes
        call_command("compute_ibit_wheel", latest=True)
        setup.refresh_from_db()
        self.assertNotEqual(setup.selection_hash, old_hash)
        self.assertEqual(len(setup.legs), 2)

    def test_dry_run_saves_nothing(self):
        call_command("compute_ibit_wheel", latest=True, dry_run=True)
        self.assertEqual(IbitWheelSetup.objects.count(), 0)
