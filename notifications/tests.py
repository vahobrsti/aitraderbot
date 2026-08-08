"""
Tests for publish_ibit_wheel publication deduplication.

The command runs on an hourly market-hours cron; without dedup the same wheel
alert would be re-sent every run. These tests confirm it sends once, skips an
identical repeat, and re-sends when the selection changes or --force is used.
"""
from datetime import timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

from django.core.management import call_command
from django.test import TestCase
from django.utils import timezone

from datafeed.models import OptionSnapshot
from notifications.models import WheelPublication
from signals.models import DailySignal

SPOT = Decimal("36.68")


def _make_signal(target_date):
    return DailySignal.objects.create(
        date=target_date,
        p_long=0.5,
        p_short=0.5,
        signal_option_call=0,
        signal_option_put=0,
        fusion_state="range",
        fusion_confidence="MEDIUM",
        fusion_score=0,
        trade_decision="BULL_PUT_SPREAD",
        income_spread_score=75.0,
        income_spread_eligible=True,
        income_spread_setups=[],
    )


def _make_put_snapshot(strike, delta, bid, ask):
    now = timezone.now()
    expiry = now + timedelta(days=30)  # income window (21-45d)
    return OptionSnapshot.objects.create(
        timestamp=now,
        symbol=f"IBIT-PUT-{strike}",
        underlying="IBIT",
        expiry=expiry,
        strike=Decimal(str(strike)),
        option_type="put",
        spot_price=SPOT,
        bid=Decimal(str(bid)),
        ask=Decimal(str(ask)),
        mid_price=(Decimal(str(bid)) + Decimal(str(ask))) / 2,
        delta=Decimal(str(delta)),
        exchange="ibkr",
    )


class PublishIbitWheelDedupTests(TestCase):
    def setUp(self):
        self.date = timezone.now().date()
        _make_signal(self.date)
        # Three puts mapping to low/medium/high tiers (spot 36.68).
        self.low = _make_put_snapshot(33, -0.15, 0.53, 0.55)
        self.med = _make_put_snapshot(34, -0.22, 0.73, 0.76)
        self.high = _make_put_snapshot(35, -0.30, 1.01, 1.04)

    def _run(self, **kwargs):
        """Run the command with a mocked notifier; return the mock instance."""
        with patch("notifications.notifier.TelegramNotifier") as cls:
            inst = MagicMock()
            inst.send_ibit_wheel.return_value = True
            cls.return_value = inst
            call_command("publish_ibit_wheel", latest=True, **kwargs)
        return inst

    def test_first_run_sends_and_records(self):
        inst = self._run()
        self.assertEqual(inst.send_ibit_wheel.call_count, 1)
        self.assertEqual(
            WheelPublication.objects.filter(
                signal_date=self.date, trade_decision="BULL_PUT_SPREAD"
            ).count(),
            1,
        )

    def test_second_identical_run_is_skipped(self):
        self._run()
        inst2 = self._run()
        inst2.send_ibit_wheel.assert_not_called()
        # Still exactly one publication row.
        self.assertEqual(WheelPublication.objects.count(), 1)

    def test_changed_selection_is_resent(self):
        self._run()
        # Change the selection: drop the high-tier contract -> different hash.
        self.high.delete()
        inst2 = self._run()
        self.assertEqual(inst2.send_ibit_wheel.call_count, 1)

    def test_force_resends_identical_selection(self):
        self._run()
        inst2 = self._run(force=True)
        self.assertEqual(inst2.send_ibit_wheel.call_count, 1)

    def test_failed_send_is_not_recorded(self):
        with patch("notifications.notifier.TelegramNotifier") as cls:
            inst = MagicMock()
            inst.send_ibit_wheel.return_value = False  # send failed
            cls.return_value = inst
            call_command("publish_ibit_wheel", latest=True)
        # No publication recorded -> next run will retry.
        self.assertEqual(WheelPublication.objects.count(), 0)
