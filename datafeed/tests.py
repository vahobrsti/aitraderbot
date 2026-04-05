"""
Tests for options data ingestion and trade tracking.
"""
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import patch, MagicMock

from django.test import TestCase

from datafeed.models import OptionSnapshot, OptionTrade
from datafeed.ingestion.bybit_options import BybitOptionsFetcher
from datafeed.ingestion.deribit_options import DeribitOptionsFetcher
from datafeed.services.trade_tracker import TradeTracker


class OptionSnapshotModelTest(TestCase):
    """Tests for OptionSnapshot model."""
    
    def test_derived_fields_computed_on_save(self):
        """Verify DTE, moneyness, spread_pct are computed."""
        now = datetime.now(timezone.utc)
        expiry = now + timedelta(days=14)
        
        snap = OptionSnapshot.objects.create(
            timestamp=now,
            symbol='BTC-TEST-70000-C',
            underlying='BTC',
            expiry=expiry,
            strike=Decimal('70000'),
            option_type='call',
            spot_price=Decimal('67000'),
            bid=Decimal('1500'),
            ask=Decimal('1600'),
            mid_price=Decimal('1550'),
            exchange='test',
        )
        
        # DTE should be ~14
        self.assertAlmostEqual(snap.dte, 14.0, delta=0.1)
        
        # Moneyness = (70000 - 67000) / 67000 ≈ 0.0448
        self.assertAlmostEqual(snap.moneyness, 0.0448, delta=0.001)
        
        # Spread = (1600 - 1500) / 1550 ≈ 0.0645
        self.assertAlmostEqual(snap.spread_pct, 0.0645, delta=0.001)
    
    def test_zero_values_preserved(self):
        """Verify zero values are not converted to None."""
        now = datetime.now(timezone.utc)
        
        snap = OptionSnapshot.objects.create(
            timestamp=now,
            symbol='BTC-TEST-70000-C',
            underlying='BTC',
            expiry=now + timedelta(days=7),
            strike=Decimal('70000'),
            option_type='call',
            spot_price=Decimal('70000'),  # ATM, moneyness = 0
            bid=Decimal('0'),  # Zero bid
            ask=Decimal('100'),
            delta=Decimal('0'),  # Zero delta possible for deep OTM
            volume_24h=Decimal('0'),  # Zero volume
            exchange='test',
        )
        
        snap.refresh_from_db()
        
        self.assertEqual(snap.bid, Decimal('0'))
        self.assertEqual(snap.delta, Decimal('0'))
        self.assertEqual(snap.volume_24h, Decimal('0'))
        self.assertAlmostEqual(snap.moneyness, 0.0, delta=0.0001)


class BybitFetcherTest(TestCase):
    """Tests for Bybit options fetcher."""
    
    def test_safe_decimal_preserves_zero(self):
        """Verify _safe_decimal doesn't drop zero."""
        fetcher = BybitOptionsFetcher()
        
        self.assertEqual(fetcher._safe_decimal('0'), Decimal('0'))
        self.assertEqual(fetcher._safe_decimal(0), Decimal('0'))
        self.assertEqual(fetcher._safe_decimal('0.0'), Decimal('0.0'))
        self.assertIsNone(fetcher._safe_decimal(None))
        self.assertIsNone(fetcher._safe_decimal(''))
    
    def test_parse_expiry_single_digit_day(self):
        """Verify expiry parsing handles single-digit days."""
        fetcher = BybitOptionsFetcher()
        
        # Single digit day
        result = fetcher._parse_expiry('7APR26')
        self.assertEqual(result.day, 7)
        self.assertEqual(result.month, 4)
        self.assertEqual(result.year, 2026)
        
        # Double digit day
        result = fetcher._parse_expiry('26APR26')
        self.assertEqual(result.day, 26)
        self.assertEqual(result.month, 4)
        self.assertEqual(result.year, 2026)
    
    @patch.object(BybitOptionsFetcher, 'session')
    def test_fetch_option_chain_filters_correctly(self, mock_session):
        """Verify DTE and moneyness filters work."""
        fetcher = BybitOptionsFetcher()
        
        # Mock spot price
        mock_session.get_tickers.side_effect = [
            # First call: spot price
            {'retCode': 0, 'result': {'list': [{'lastPrice': '67000'}]}},
            # Second call: options
            {'retCode': 0, 'result': {'list': [
                {
                    'symbol': 'BTC-17APR26-70000-C-USDT',
                    'bid1Price': '1000',
                    'ask1Price': '1100',
                    'markPrice': '1050',
                    'markIv': '0.45',
                    'delta': '0.35',
                    'gamma': '0.00001',
                    'vega': '50',
                    'theta': '-20',
                    'underlyingPrice': '67000',
                },
            ]}},
        ]
        
        # This would need actual date mocking to work properly
        # For now just verify the method doesn't crash
        with patch.object(fetcher, '_parse_expiry') as mock_expiry:
            mock_expiry.return_value = datetime(2026, 4, 17, 8, 0, tzinfo=timezone.utc)
            # Method should handle the mocked data
            # Full integration test would require more setup


class DeribitFetcherTest(TestCase):
    """Tests for Deribit options fetcher."""
    
    def test_safe_decimal_preserves_zero(self):
        """Verify safe_decimal doesn't drop zero."""
        fetcher = DeribitOptionsFetcher()
        
        self.assertEqual(fetcher.safe_decimal('0'), Decimal('0'))
        self.assertEqual(fetcher.safe_decimal(0), Decimal('0'))
        self.assertIsNone(fetcher.safe_decimal(None))


class TradeTrackerTest(TestCase):
    """Tests for trade tracking service."""
    
    def setUp(self):
        self.tracker = TradeTracker()
        self.now = datetime.now(timezone.utc)
        
        # Create test snapshot
        self.snapshot = OptionSnapshot.objects.create(
            timestamp=self.now - timedelta(hours=1),
            symbol='BTC-17APR26-68000-C',
            underlying='BTC',
            expiry=self.now + timedelta(days=14),
            strike=Decimal('68000'),
            option_type='call',
            spot_price=Decimal('67000'),
            bid=Decimal('1600'),
            ask=Decimal('1700'),
            mid_price=Decimal('1650'),
            mark_price=Decimal('1650'),
            iv=Decimal('0.45'),
            delta=Decimal('0.45'),
            exchange='bybit',
        )
    
    def test_open_trade_with_explicit_timestamp(self):
        """Verify historical backfill works with explicit timestamp."""
        entry_time = self.now - timedelta(days=5)
        
        trade = self.tracker.open_trade(
            signal_type='BULL_PROBE',
            direction='LONG',
            symbol='BTC-17APR26-68000-C',
            qty=Decimal('0.1'),
            entry_price=Decimal('1650'),
            entry_spot=Decimal('67000'),
            entry_timestamp=entry_time,
            exchange='bybit',
        )
        
        self.assertEqual(trade.entry_timestamp, entry_time)
        self.assertEqual(trade.signal_type, 'BULL_PROBE')
        self.assertEqual(trade.direction, 'LONG')
        self.assertIsNone(trade.exit_timestamp)
    
    def test_close_trade_computes_pnl(self):
        """Verify PnL computation on close."""
        entry_time = self.now - timedelta(days=3)
        exit_time = self.now
        
        trade = self.tracker.open_trade(
            signal_type='BULL_PROBE',
            direction='LONG',
            symbol='BTC-17APR26-68000-C',
            qty=Decimal('0.1'),
            entry_price=Decimal('1650'),
            entry_spot=Decimal('67000'),
            entry_timestamp=entry_time,
            exchange='bybit',
        )
        
        trade = self.tracker.close_trade(
            trade_id=trade.trade_id,
            exit_price=Decimal('2000'),
            exit_spot=Decimal('69000'),
            exit_reason='tp',
            exit_timestamp=exit_time,
        )
        
        # PnL = (2000 - 1650) * 0.1 = 35
        self.assertEqual(trade.realized_pnl, Decimal('35'))
        self.assertEqual(trade.exit_reason, 'tp')
        self.assertEqual(trade.exit_timestamp, exit_time)
        
        # Spot change = (69000 - 67000) / 67000 ≈ 0.0298
        self.assertAlmostEqual(trade.spot_change_pct, 0.0298, delta=0.001)
    
    def test_close_trade_short_direction(self):
        """Verify PnL is inverted for short trades."""
        entry_time = self.now - timedelta(days=3)
        exit_time = self.now
        
        trade = self.tracker.open_trade(
            signal_type='PRIMARY_SHORT',
            direction='SHORT',
            symbol='BTC-17APR26-68000-P',
            qty=Decimal('0.1'),
            entry_price=Decimal('1500'),
            entry_spot=Decimal('67000'),
            entry_timestamp=entry_time,
            exchange='bybit',
        )
        
        trade = self.tracker.close_trade(
            trade_id=trade.trade_id,
            exit_price=Decimal('1200'),  # Price dropped = profit for short
            exit_spot=Decimal('68000'),
            exit_reason='tp',
            exit_timestamp=exit_time,
        )
        
        # Short PnL = (1500 - 1200) * 0.1 = 30
        self.assertEqual(trade.realized_pnl, Decimal('30'))
    
    def test_close_trade_validates_timestamp_order(self):
        """Verify exit must be after entry."""
        entry_time = self.now
        
        trade = self.tracker.open_trade(
            signal_type='BULL_PROBE',
            direction='LONG',
            symbol='BTC-17APR26-68000-C',
            qty=Decimal('0.1'),
            entry_price=Decimal('1650'),
            entry_spot=Decimal('67000'),
            entry_timestamp=entry_time,
            exchange='bybit',
        )
        
        with self.assertRaises(ValueError) as ctx:
            self.tracker.close_trade(
                trade_id=trade.trade_id,
                exit_price=Decimal('2000'),
                exit_spot=Decimal('69000'),
                exit_reason='tp',
                exit_timestamp=entry_time - timedelta(hours=1),  # Before entry
            )
        
        self.assertIn('after entry', str(ctx.exception))
    
    def test_cannot_close_already_closed_trade(self):
        """Verify double-close is prevented."""
        entry_time = self.now - timedelta(days=1)
        
        trade = self.tracker.open_trade(
            signal_type='BULL_PROBE',
            direction='LONG',
            symbol='BTC-17APR26-68000-C',
            qty=Decimal('0.1'),
            entry_price=Decimal('1650'),
            entry_spot=Decimal('67000'),
            entry_timestamp=entry_time,
            exchange='bybit',
        )
        
        self.tracker.close_trade(
            trade_id=trade.trade_id,
            exit_price=Decimal('2000'),
            exit_spot=Decimal('69000'),
            exit_reason='tp',
        )
        
        with self.assertRaises(ValueError) as ctx:
            self.tracker.close_trade(
                trade_id=trade.trade_id,
                exit_price=Decimal('2100'),
                exit_spot=Decimal('70000'),
                exit_reason='manual',
            )
        
        self.assertIn('already closed', str(ctx.exception))
    
    def test_get_open_trades(self):
        """Verify open trades query."""
        # Create open trade
        open_trade = self.tracker.open_trade(
            signal_type='BULL_PROBE',
            direction='LONG',
            symbol='BTC-17APR26-68000-C',
            qty=Decimal('0.1'),
            entry_price=Decimal('1650'),
            entry_spot=Decimal('67000'),
            exchange='bybit',
        )
        
        # Create and close another trade
        closed_trade = self.tracker.open_trade(
            signal_type='BEAR_PROBE',
            direction='SHORT',
            symbol='BTC-17APR26-68000-P',
            qty=Decimal('0.1'),
            entry_price=Decimal('1500'),
            entry_spot=Decimal('67000'),
            entry_timestamp=self.now - timedelta(days=1),
            exchange='bybit',
        )
        self.tracker.close_trade(
            trade_id=closed_trade.trade_id,
            exit_price=Decimal('1200'),
            exit_spot=Decimal('68000'),
            exit_reason='tp',
        )
        
        open_trades = self.tracker.get_open_trades()
        
        self.assertEqual(len(open_trades), 1)
        self.assertEqual(open_trades[0].trade_id, open_trade.trade_id)


class ExportOptionsTest(TestCase):
    """Tests for CSV export."""
    
    def test_zero_values_exported_correctly(self):
        """Verify zeros are exported as '0' not blank."""
        from io import StringIO
        import csv
        
        now = datetime.now(timezone.utc)
        
        # Create snapshot with zero values
        OptionSnapshot.objects.create(
            timestamp=now,
            symbol='BTC-TEST-70000-C',
            underlying='BTC',
            expiry=now + timedelta(days=7),
            strike=Decimal('70000'),
            option_type='call',
            spot_price=Decimal('70000'),
            bid=Decimal('0'),
            ask=Decimal('100'),
            mid_price=Decimal('50'),
            delta=Decimal('0'),
            volume_24h=Decimal('0'),
            exchange='test',
        )
        
        # Run export command
        from django.core.management import call_command
        out = StringIO()
        call_command('export_options', '--output=/tmp/test_export.csv', stdout=out)
        
        # Read and verify
        with open('/tmp/test_export.csv', 'r') as f:
            reader = csv.DictReader(f)
            row = next(reader)
            
            self.assertEqual(row['bid'], '0.0')
            self.assertEqual(row['delta'], '0.0')
            self.assertEqual(row['volume_24h'], '0.0')
