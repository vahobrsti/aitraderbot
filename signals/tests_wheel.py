"""
Tests for IBIT wheel short-leg selection and its Telegram formatting.

Fixture quotes are real IBIT SEP-18-'26 bid/ask and open interest pulled from
IBKR (spot ~36.68); deltas are assigned by moneyness to exercise the tier bands.
"""
from datetime import datetime, timezone

import pandas as pd
from django.test import TestCase

from signals.wheel import select_wheel_legs, wheel_side_for_decision, IBIT_MULTIPLIER

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


class WheelTelegramFormatTests(TestCase):
    def _notifier(self):
        from notifications.notifier import TelegramNotifier
        return TelegramNotifier(bot_token="test", chat_id="test")

    def test_put_message_contains_key_fields(self):
        legs = select_wheel_legs(_chain(), "put", SPOT, dte_mode="income")
        msg = self._notifier()._format_ibit_wheel_message(
            "2026-08-07", "put", SPOT, legs, market_data_type="live"
        )
        self.assertIn("IBIT WHEEL", msg)
        self.assertIn("Cash-Secured Put", msg)
        self.assertIn("Cash Reserve", msg)
        self.assertIn("$36.68", msg)
        self.assertIn("2026-09-18", msg)

    def test_frozen_data_caveat_shown(self):
        legs = select_wheel_legs(_chain(), "put", SPOT, dte_mode="income")
        msg = self._notifier()._format_ibit_wheel_message(
            "2026-08-07", "put", SPOT, legs, market_data_type="frozen"
        )
        self.assertIn("indicative only", msg)

    def test_call_message_shows_backing(self):
        legs = select_wheel_legs(_chain(), "call", SPOT, dte_mode="income")
        msg = self._notifier()._format_ibit_wheel_message(
            "2026-08-07", "call", SPOT, legs
        )
        self.assertIn("Covered Call", msg)
        self.assertIn("100 shares", msg)

    def test_no_legs_message(self):
        msg = self._notifier()._format_ibit_wheel_message(
            "2026-08-07", "put", SPOT, [], market_data_type="live"
        )
        self.assertIn("No IBIT contract", msg)
