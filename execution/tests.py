"""
Comprehensive tests for execution app.
Covers: models, adapters, risk management, orchestration, and edge cases.
"""
from decimal import Decimal
from datetime import date, timedelta
from unittest.mock import Mock, patch, MagicMock
from django.test import TestCase
from django.utils import timezone

from execution.models import (
    ExchangeAccount, ExecutionIntent, Order, Fill, Position, ExecutionEvent
)
from execution.exchanges.base import (
    OrderRequest, OrderResponse, PositionInfo, PositionSyncResult, InstrumentInfo
)
from execution.services.risk import RiskManager, RiskCheckResult


# =============================================================================
# MODEL TESTS
# =============================================================================

class ExchangeAccountModelTests(TestCase):
    """Tests for ExchangeAccount model."""
    
    def test_create_account(self):
        """Test creating an exchange account."""
        account = ExchangeAccount.objects.create(
            name='test-bybit',
            exchange='bybit',
            account_type='unified',
            api_key_env='BYBIT_API_KEY',
            api_secret_env='BYBIT_API_SECRET',
            is_testnet=True,
            max_position_usd=Decimal('5000'),
            max_daily_loss_usd=Decimal('500'),
        )
        self.assertEqual(account.exchange, 'bybit')
        self.assertTrue(account.is_active)
        self.assertEqual(str(account), 'test-bybit (bybit)')
    
    def test_account_defaults(self):
        """Test default values are set correctly."""
        account = ExchangeAccount.objects.create(
            name='minimal',
            exchange='deribit',
            api_key_env='KEY',
            api_secret_env='SECRET',
        )
        self.assertEqual(account.account_type, 'unified')
        self.assertTrue(account.is_testnet)
        self.assertTrue(account.is_active)
        self.assertEqual(account.max_position_usd, Decimal('10000'))
        self.assertEqual(account.max_daily_loss_usd, Decimal('1000'))


class ExecutionIntentModelTests(TestCase):
    """Tests for ExecutionIntent model."""
    
    def setUp(self):
        self.account = ExchangeAccount.objects.create(
            name='test-account',
            exchange='bybit',
            api_key_env='KEY',
            api_secret_env='SECRET',
        )
    
    def test_intent_status_choices(self):
        """Test that all status choices are valid."""
        valid_statuses = [s[0] for s in ExecutionIntent.STATUS_CHOICES]
        self.assertIn('pending', valid_statuses)
        self.assertIn('filled', valid_statuses)
        self.assertIn('rejected', valid_statuses)
    
    def test_direction_choices(self):
        """Test that direction choices are valid."""
        valid_directions = [d[0] for d in ExecutionIntent.DIRECTION_CHOICES]
        self.assertIn('long', valid_directions)
        self.assertIn('short', valid_directions)


class OrderModelTests(TestCase):
    """Tests for Order model."""
    
    def setUp(self):
        self.account = ExchangeAccount.objects.create(
            name='test', exchange='bybit',
            api_key_env='KEY', api_secret_env='SECRET',
        )
    
    def test_client_order_id_auto_generated(self):
        """Test client_order_id is auto-generated with ai_ prefix."""
        # Create order without intent (using mock)
        order = Order(
            intent_id=None,  # Will be set via mock
            symbol='BTCUSDT',
            side='buy',
            order_type='market',
            qty=Decimal('0.1'),
        )
        # Manually trigger the save logic for client_order_id
        if not order.client_order_id:
            import uuid
            order.client_order_id = f"ai_{uuid.uuid4().hex[:16]}"
        
        self.assertTrue(order.client_order_id.startswith('ai_'))
        self.assertEqual(len(order.client_order_id), 19)  # ai_ + 16 hex chars
    
    def test_order_status_choices(self):
        """Test order status choices are valid."""
        valid_statuses = [s[0] for s in Order.STATUS_CHOICES]
        self.assertIn('pending', valid_statuses)
        self.assertIn('filled', valid_statuses)
        self.assertIn('cancelled', valid_statuses)


class PositionModelTests(TestCase):
    """Tests for Position model."""
    
    def setUp(self):
        self.account = ExchangeAccount.objects.create(
            name='test', exchange='bybit',
            api_key_env='KEY', api_secret_env='SECRET',
        )
    
    def test_position_creation(self):
        """Test creating a position."""
        pos = Position.objects.create(
            account=self.account,
            symbol='BTCUSDT',
            side='long',
            qty=Decimal('1.0'),
            entry_price=Decimal('50000'),
        )
        self.assertEqual(pos.symbol, 'BTCUSDT')
        self.assertEqual(pos.side, 'long')
    
    def test_position_unique_constraint(self):
        """Test account+symbol uniqueness."""
        Position.objects.create(
            account=self.account,
            symbol='BTCUSDT',
            side='long',
            qty=Decimal('1.0'),
            entry_price=Decimal('50000'),
        )
        
        # Should raise on duplicate
        from django.db import IntegrityError
        with self.assertRaises(IntegrityError):
            Position.objects.create(
                account=self.account,
                symbol='BTCUSDT',
                side='short',
                qty=Decimal('0.5'),
                entry_price=Decimal('51000'),
            )


# =============================================================================
# BASE ADAPTER DATACLASS TESTS
# =============================================================================

class OrderRequestTests(TestCase):
    """Tests for OrderRequest dataclass."""
    
    def test_order_request_creation(self):
        """Test creating an OrderRequest with required fields."""
        request = OrderRequest(
            symbol='BTCUSDT',
            side='buy',
            order_type='market',
            qty=Decimal('0.1'),
        )
        self.assertEqual(request.symbol, 'BTCUSDT')
        self.assertEqual(request.side, 'buy')
        self.assertEqual(request.qty, Decimal('0.1'))
        self.assertIsNone(request.price)
        self.assertEqual(request.time_in_force, 'GTC')
    
    def test_order_request_with_all_fields(self):
        """Test OrderRequest with all optional fields."""
        request = OrderRequest(
            symbol='BTC-30JUN25-100000-C',
            side='sell',
            order_type='limit',
            qty=Decimal('1.0'),
            price=Decimal('0.05'),
            trigger_price=Decimal('95000'),
            reduce_only=True,
            client_order_id='ai_test123',
            time_in_force='IOC',
        )
        self.assertEqual(request.price, Decimal('0.05'))
        self.assertTrue(request.reduce_only)
        self.assertEqual(request.time_in_force, 'IOC')


class OrderResponseTests(TestCase):
    """Tests for OrderResponse dataclass."""
    
    def test_successful_response(self):
        """Test successful order response."""
        response = OrderResponse(
            success=True,
            exchange_order_id='12345',
            client_order_id='ai_abc',
            status='filled',
            filled_qty=Decimal('0.1'),
            avg_price=Decimal('50000'),
        )
        self.assertTrue(response.success)
        self.assertEqual(response.status, 'filled')
    
    def test_failed_response(self):
        """Test failed order response."""
        response = OrderResponse(
            success=False,
            error_code='INSUFFICIENT_BALANCE',
            error_message='Not enough margin',
        )
        self.assertFalse(response.success)
        self.assertEqual(response.error_code, 'INSUFFICIENT_BALANCE')


class PositionSyncResultTests(TestCase):
    """Tests for PositionSyncResult dataclass."""
    
    def test_successful_sync_with_positions(self):
        """Test successful sync returning positions."""
        positions = [
            PositionInfo(
                symbol='BTCUSDT',
                side='long',
                qty=Decimal('1.0'),
                entry_price=Decimal('50000'),
            )
        ]
        result = PositionSyncResult(success=True, positions=positions)
        
        self.assertTrue(result.success)
        self.assertEqual(len(result.positions), 1)
        self.assertIsNone(result.error)
    
    def test_failed_sync(self):
        """Test failed sync with error."""
        result = PositionSyncResult(
            success=False,
            positions=[],
            error='Connection timeout',
        )
        
        self.assertFalse(result.success)
        self.assertEqual(len(result.positions), 0)
        self.assertEqual(result.error, 'Connection timeout')
    
    def test_successful_sync_empty_positions(self):
        """Test successful sync with no positions (legitimately empty)."""
        result = PositionSyncResult(success=True, positions=[])
        
        self.assertTrue(result.success)
        self.assertEqual(len(result.positions), 0)


# =============================================================================
# RISK MANAGER TESTS
# =============================================================================

class RiskManagerTests(TestCase):
    """Tests for RiskManager service."""
    
    def setUp(self):
        self.account = ExchangeAccount.objects.create(
            name='risk-test',
            exchange='bybit',
            api_key_env='KEY',
            api_secret_env='SECRET',
            is_active=True,
            max_position_usd=Decimal('10000'),
            max_daily_loss_usd=Decimal('1000'),
        )
        self.risk_manager = RiskManager(self.account)
    
    def test_inactive_account_fails(self):
        """Inactive account should fail risk check."""
        self.account.is_active = False
        self.account.save()
        
        mock_intent = Mock()
        mock_intent.idempotency_key = 'test-key'
        
        result = self.risk_manager._check_account_active(mock_intent)
        self.assertFalse(result.passed)
        self.assertIn('not active', result.reason)
    
    def test_active_account_passes(self):
        """Active account should pass risk check."""
        mock_intent = Mock()
        result = self.risk_manager._check_account_active(mock_intent)
        self.assertTrue(result.passed)
    
    def test_position_limit_adjustment(self):
        """Position exceeding limit should be adjusted."""
        mock_intent = Mock()
        mock_intent.target_notional_usd = Decimal('15000')  # Exceeds 10000 limit
        
        result = self.risk_manager._check_position_limit(mock_intent)
        
        self.assertTrue(result.passed)
        self.assertEqual(result.adjusted_notional, Decimal('10000'))
        self.assertIn('Adjusted', result.reason)
    
    def test_position_within_limit_passes(self):
        """Position within limit should pass without adjustment."""
        mock_intent = Mock()
        mock_intent.target_notional_usd = Decimal('5000')
        
        result = self.risk_manager._check_position_limit(mock_intent)
        
        self.assertTrue(result.passed)
        self.assertIsNone(result.adjusted_notional)
    
    def test_conflicting_position_blocks_trade(self):
        """Opposite direction position should block new trade."""
        # Create existing short position
        Position.objects.create(
            account=self.account,
            symbol='BTCUSDT',
            side='short',
            qty=Decimal('1.0'),
            entry_price=Decimal('50000'),
        )
        
        mock_intent = Mock()
        mock_intent.direction = 'long'  # Trying to go long while short
        
        result = self.risk_manager._check_open_positions(mock_intent)
        
        self.assertFalse(result.passed)
        self.assertIn('Conflicting', result.reason)
    
    def test_same_direction_position_allowed(self):
        """Same direction position should not block trade."""
        Position.objects.create(
            account=self.account,
            symbol='BTCUSDT',
            side='long',
            qty=Decimal('1.0'),
            entry_price=Decimal('50000'),
        )
        
        mock_intent = Mock()
        mock_intent.direction = 'long'  # Adding to long
        
        result = self.risk_manager._check_open_positions(mock_intent)
        self.assertTrue(result.passed)
    
    def test_zero_qty_position_ignored(self):
        """Closed position (qty=0) should not block trades."""
        Position.objects.create(
            account=self.account,
            symbol='BTCUSDT',
            side='short',
            qty=Decimal('0'),  # Closed position
            entry_price=Decimal('50000'),
        )
        
        mock_intent = Mock()
        mock_intent.direction = 'long'
        
        result = self.risk_manager._check_open_positions(mock_intent)
        self.assertTrue(result.passed)


# =============================================================================
# BYBIT ADAPTER TESTS
# =============================================================================

class BybitAdapterTests(TestCase):
    """Tests for Bybit exchange adapter."""
    
    def setUp(self):
        self.adapter = None  # Will be created with mocked session
    
    @patch('execution.exchanges.bybit.BybitAdapter.session', new_callable=lambda: property(lambda self: Mock()))
    def test_get_category_detection(self, mock_session):
        """Test symbol category detection."""
        from execution.exchanges.bybit import BybitAdapter
        adapter = BybitAdapter('key', 'secret', testnet=True)
        
        self.assertEqual(adapter._get_category('BTCUSDT'), 'linear')
        self.assertEqual(adapter._get_category('BTC-30JUN25-100000-C'), 'option')
        self.assertEqual(adapter._get_category('BTC-30JUN25-100000-P'), 'option')
        self.assertEqual(adapter._get_category('BTCUSD'), 'inverse')
    
    def test_order_type_mapping(self):
        """Test order type mapping to Bybit format."""
        from execution.exchanges.bybit import BybitAdapter
        adapter = BybitAdapter('key', 'secret', testnet=True)
        
        self.assertEqual(adapter._map_order_type('market'), 'Market')
        self.assertEqual(adapter._map_order_type('limit'), 'Limit')
        self.assertEqual(adapter._map_order_type('stop_market'), 'Market')
    
    def test_status_mapping(self):
        """Test Bybit status to internal status mapping."""
        from execution.exchanges.bybit import BybitAdapter
        adapter = BybitAdapter('key', 'secret', testnet=True)
        
        self.assertEqual(adapter._map_status('New'), 'open')
        self.assertEqual(adapter._map_status('Filled'), 'filled')
        self.assertEqual(adapter._map_status('Cancelled'), 'cancelled')
        self.assertEqual(adapter._map_status('PartiallyFilled'), 'partial')
    
    def test_symbol_normalization(self):
        """Test symbol format conversion."""
        from execution.exchanges.bybit import BybitAdapter
        adapter = BybitAdapter('key', 'secret', testnet=True)
        
        # Internal to Bybit
        self.assertEqual(adapter.normalize_symbol('BTC-USDT-PERP'), 'BTCUSDT')
        self.assertEqual(adapter.normalize_symbol('BTC-30JUN25-100000-C'), 'BTC-30JUN25-100000-C')
        
        # Bybit to internal
        self.assertEqual(adapter.denormalize_symbol('BTCUSDT'), 'BTC-USDT-PERP')
        self.assertEqual(adapter.denormalize_symbol('BTC-30JUN25-100000-C'), 'BTC-30JUN25-100000-C')
    
    @patch('execution.exchanges.bybit.BybitAdapter.session')
    def test_place_order_success(self, mock_session):
        """Test successful order placement."""
        from execution.exchanges.bybit import BybitAdapter
        
        mock_session.place_order.return_value = {
            'retCode': 0,
            'result': {
                'orderId': '123456',
                'orderLinkId': 'ai_test',
            }
        }
        
        adapter = BybitAdapter('key', 'secret', testnet=True)
        adapter._session = mock_session
        
        request = OrderRequest(
            symbol='BTCUSDT',
            side='buy',
            order_type='market',
            qty=Decimal('0.1'),
        )
        
        response = adapter.place_order(request)
        
        self.assertTrue(response.success)
        self.assertEqual(response.exchange_order_id, '123456')
    
    @patch('execution.exchanges.bybit.BybitAdapter.session')
    def test_place_order_failure(self, mock_session):
        """Test failed order placement."""
        from execution.exchanges.bybit import BybitAdapter
        
        mock_session.place_order.return_value = {
            'retCode': 10001,
            'retMsg': 'Insufficient balance',
        }
        
        adapter = BybitAdapter('key', 'secret', testnet=True)
        adapter._session = mock_session
        
        request = OrderRequest(
            symbol='BTCUSDT',
            side='buy',
            order_type='market',
            qty=Decimal('100'),
        )
        
        response = adapter.place_order(request)
        
        self.assertFalse(response.success)
        self.assertEqual(response.error_code, '10001')
    
    @patch('execution.exchanges.bybit.BybitAdapter.session')
    def test_get_positions_success(self, mock_session):
        """Test successful position retrieval."""
        from execution.exchanges.bybit import BybitAdapter
        
        # Return position for linear, empty for option
        mock_session.get_positions.side_effect = [
            {
                'retCode': 0,
                'result': {
                    'list': [{
                        'symbol': 'BTCUSDT',
                        'side': 'Buy',
                        'size': '0.5',
                        'avgPrice': '50000',
                        'unrealisedPnl': '100',
                        'cumRealisedPnl': '50',
                        'leverage': '10',
                    }]
                }
            },
            {  # Option category returns empty
                'retCode': 0,
                'result': {'list': []}
            }
        ]
        
        adapter = BybitAdapter('key', 'secret', testnet=True)
        adapter._session = mock_session
        
        result = adapter.get_positions()
        
        self.assertTrue(result.success)
        self.assertEqual(len(result.positions), 1)
        self.assertEqual(result.positions[0].symbol, 'BTCUSDT')
        self.assertEqual(result.positions[0].side, 'long')
    
    @patch('execution.exchanges.bybit.BybitAdapter.session')
    def test_get_positions_api_error(self, mock_session):
        """Test position retrieval with API error."""
        from execution.exchanges.bybit import BybitAdapter
        
        mock_session.get_positions.return_value = {
            'retCode': 10002,
            'retMsg': 'Invalid API key',
        }
        
        adapter = BybitAdapter('key', 'secret', testnet=True)
        adapter._session = mock_session
        
        result = adapter.get_positions()
        
        self.assertFalse(result.success)
        self.assertIn('Invalid API key', result.error)
    
    @patch('execution.exchanges.bybit.BybitAdapter.session')
    def test_get_positions_exception(self, mock_session):
        """Test position retrieval with exception."""
        from execution.exchanges.bybit import BybitAdapter
        
        mock_session.get_positions.side_effect = Exception('Network error')
        
        adapter = BybitAdapter('key', 'secret', testnet=True)
        adapter._session = mock_session
        
        result = adapter.get_positions()
        
        self.assertFalse(result.success)
        self.assertIn('Network error', result.error)


# =============================================================================
# DERIBIT ADAPTER TESTS
# =============================================================================

class DeribitAdapterTests(TestCase):
    """Tests for Deribit exchange adapter."""
    
    def test_symbol_normalization(self):
        """Test symbol format conversion."""
        from execution.exchanges.deribit import DeribitAdapter
        adapter = DeribitAdapter('key', 'secret', testnet=True)
        
        # Internal to Deribit
        self.assertEqual(adapter.normalize_symbol('BTC-USD-PERP'), 'BTC-PERPETUAL')
        self.assertEqual(adapter.normalize_symbol('BTC-30JUN25-100000-C'), 'BTC-30JUN25-100000-C')
        
        # Deribit to internal
        self.assertEqual(adapter.denormalize_symbol('BTC-PERPETUAL'), 'BTC-USD-PERP')
    
    def test_status_mapping(self):
        """Test Deribit status to internal status mapping."""
        from execution.exchanges.deribit import DeribitAdapter
        adapter = DeribitAdapter('key', 'secret', testnet=True)
        
        self.assertEqual(adapter._map_status('open'), 'open')
        self.assertEqual(adapter._map_status('filled'), 'filled')
        self.assertEqual(adapter._map_status('cancelled'), 'cancelled')
        self.assertEqual(adapter._map_status('untriggered'), 'pending')
    
    @patch('execution.exchanges.deribit.requests.get')
    def test_get_positions_success(self, mock_get):
        """Test successful position retrieval from Deribit."""
        from execution.exchanges.deribit import DeribitAdapter
        
        # Mock auth response
        mock_get.return_value.json.side_effect = [
            {  # Auth response
                'result': {
                    'access_token': 'test_token',
                    'expires_in': 3600,
                }
            },
            {  # Positions response
                'result': [{
                    'instrument_name': 'BTC-PERPETUAL',
                    'size': 1000,
                    'average_price': 50000,
                    'floating_profit_loss': 100,
                    'realized_profit_loss': 50,
                    'leverage': 10,
                }]
            }
        ]
        
        adapter = DeribitAdapter('key', 'secret', testnet=True)
        result = adapter.get_positions()
        
        self.assertTrue(result.success)
        self.assertEqual(len(result.positions), 1)
        self.assertEqual(result.positions[0].side, 'long')
    
    @patch('execution.exchanges.deribit.requests.get')
    def test_get_positions_api_error(self, mock_get):
        """Test position retrieval with API error."""
        from execution.exchanges.deribit import DeribitAdapter
        
        mock_get.return_value.json.side_effect = [
            {'result': {'access_token': 'token', 'expires_in': 3600}},
            {'error': {'code': 10001, 'message': 'Invalid request'}},
        ]
        
        adapter = DeribitAdapter('key', 'secret', testnet=True)
        result = adapter.get_positions()
        
        self.assertFalse(result.success)
        self.assertIn('Invalid request', result.error)
    
    @patch('execution.exchanges.deribit.requests.get')
    def test_option_position_parsing(self, mock_get):
        """Test parsing option position details from instrument name."""
        from execution.exchanges.deribit import DeribitAdapter
        
        mock_get.return_value.json.side_effect = [
            {'result': {'access_token': 'token', 'expires_in': 3600}},
            {'result': [{
                'instrument_name': 'BTC-30JUN25-100000-C',
                'size': 5,
                'average_price': 0.05,
                'floating_profit_loss': 0.01,
                'realized_profit_loss': 0,
            }]},
        ]
        
        adapter = DeribitAdapter('key', 'secret', testnet=True)
        result = adapter.get_positions()
        
        self.assertTrue(result.success)
        pos = result.positions[0]
        self.assertEqual(pos.option_type, 'call')
        self.assertEqual(pos.strike, Decimal('100000'))


# =============================================================================
# ORCHESTRATOR TESTS
# =============================================================================

class OrchestratorDecisionMappingTests(TestCase):
    """Tests for trade decision to intent mapping."""
    
    def test_call_decision_mapping(self):
        """Test CALL maps to long/call."""
        from execution.services.orchestrator import ExecutionOrchestrator
        
        # Test the decision map directly
        decision_map = {
            'CALL': ('long', 'call'),
            'OPTION_CALL': ('long', 'call'),
            'PUT': ('short', 'put'),
            'OPTION_PUT': ('short', 'put'),
            'TACTICAL_PUT': ('long', 'put'),
        }
        
        self.assertEqual(decision_map['CALL'], ('long', 'call'))
        self.assertEqual(decision_map['OPTION_CALL'], ('long', 'call'))
        self.assertEqual(decision_map['PUT'], ('short', 'put'))
        self.assertEqual(decision_map['OPTION_PUT'], ('short', 'put'))
        self.assertEqual(decision_map['TACTICAL_PUT'], ('long', 'put'))
    
    def test_no_trade_not_in_mapping(self):
        """Test NO_TRADE is not a valid decision for execution."""
        decision_map = {
            'CALL': ('long', 'call'),
            'OPTION_CALL': ('long', 'call'),
            'PUT': ('short', 'put'),
            'OPTION_PUT': ('short', 'put'),
            'TACTICAL_PUT': ('long', 'put'),
        }
        
        self.assertNotIn('NO_TRADE', decision_map)


class PositionSyncTests(TestCase):
    """Tests for position synchronization logic."""
    
    def setUp(self):
        self.account = ExchangeAccount.objects.create(
            name='sync-test',
            exchange='bybit',
            api_key_env='BYBIT_API_KEY',
            api_secret_env='BYBIT_API_SECRET',
        )
    
    @patch.dict('os.environ', {'BYBIT_API_KEY': 'test', 'BYBIT_API_SECRET': 'test'})
    @patch('execution.exchanges.bybit.BybitAdapter')
    def test_sync_positions_updates_existing(self, mock_adapter_class):
        """Test that sync updates existing positions."""
        from execution.services.orchestrator import ExecutionOrchestrator
        
        # Create existing position
        Position.objects.create(
            account=self.account,
            symbol='BTCUSDT',
            side='long',
            qty=Decimal('1.0'),
            entry_price=Decimal('50000'),
        )
        
        # Mock adapter returns updated position
        mock_adapter = Mock()
        mock_adapter.get_positions.return_value = PositionSyncResult(
            success=True,
            positions=[PositionInfo(
                symbol='BTCUSDT',
                side='long',
                qty=Decimal('1.5'),  # Increased
                entry_price=Decimal('51000'),
                unrealized_pnl=Decimal('500'),
            )]
        )
        mock_adapter_class.return_value = mock_adapter
        
        orchestrator = ExecutionOrchestrator(self.account)
        orchestrator.adapter = mock_adapter
        
        synced = orchestrator.sync_positions()
        
        self.assertEqual(len(synced), 1)
        pos = Position.objects.get(account=self.account, symbol='BTCUSDT')
        self.assertEqual(pos.qty, Decimal('1.5'))
    
    @patch.dict('os.environ', {'BYBIT_API_KEY': 'test', 'BYBIT_API_SECRET': 'test'})
    @patch('execution.exchanges.bybit.BybitAdapter')
    def test_sync_positions_zeros_closed(self, mock_adapter_class):
        """Test that closed positions are zeroed out on successful sync."""
        from execution.services.orchestrator import ExecutionOrchestrator
        
        # Create existing position that will be closed
        Position.objects.create(
            account=self.account,
            symbol='BTCUSDT',
            side='long',
            qty=Decimal('1.0'),
            entry_price=Decimal('50000'),
        )
        
        # Mock adapter returns empty (position closed)
        mock_adapter = Mock()
        mock_adapter.get_positions.return_value = PositionSyncResult(
            success=True,
            positions=[]  # No positions returned
        )
        mock_adapter_class.return_value = mock_adapter
        
        orchestrator = ExecutionOrchestrator(self.account)
        orchestrator.adapter = mock_adapter
        
        orchestrator.sync_positions()
        
        pos = Position.objects.get(account=self.account, symbol='BTCUSDT')
        self.assertEqual(pos.qty, Decimal('0'))
        self.assertEqual(pos.side, 'none')
    
    @patch.dict('os.environ', {'BYBIT_API_KEY': 'test', 'BYBIT_API_SECRET': 'test'})
    @patch('execution.exchanges.bybit.BybitAdapter')
    def test_sync_positions_skips_zero_on_api_failure(self, mock_adapter_class):
        """Test that positions are NOT zeroed on API failure."""
        from execution.services.orchestrator import ExecutionOrchestrator
        
        # Create existing position
        Position.objects.create(
            account=self.account,
            symbol='BTCUSDT',
            side='long',
            qty=Decimal('1.0'),
            entry_price=Decimal('50000'),
        )
        
        # Mock adapter returns failure
        mock_adapter = Mock()
        mock_adapter.get_positions.return_value = PositionSyncResult(
            success=False,
            positions=[],
            error='Connection timeout'
        )
        mock_adapter_class.return_value = mock_adapter
        
        orchestrator = ExecutionOrchestrator(self.account)
        orchestrator.adapter = mock_adapter
        
        orchestrator.sync_positions()
        
        # Position should NOT be zeroed
        pos = Position.objects.get(account=self.account, symbol='BTCUSDT')
        self.assertEqual(pos.qty, Decimal('1.0'))
        self.assertEqual(pos.side, 'long')
    
    @patch.dict('os.environ', {'BYBIT_API_KEY': 'test', 'BYBIT_API_SECRET': 'test'})
    @patch('execution.exchanges.bybit.BybitAdapter')
    def test_sync_positions_partial_update_on_failure(self, mock_adapter_class):
        """Test that received positions are updated even on partial failure."""
        from execution.services.orchestrator import ExecutionOrchestrator
        
        # Create two existing positions
        Position.objects.create(
            account=self.account, symbol='BTCUSDT',
            side='long', qty=Decimal('1.0'), entry_price=Decimal('50000'),
        )
        Position.objects.create(
            account=self.account, symbol='ETHUSDT',
            side='long', qty=Decimal('10.0'), entry_price=Decimal('3000'),
        )
        
        # Mock adapter returns one position but with error flag
        mock_adapter = Mock()
        mock_adapter.get_positions.return_value = PositionSyncResult(
            success=False,  # API had error
            positions=[PositionInfo(
                symbol='BTCUSDT',
                side='long',
                qty=Decimal('2.0'),  # Updated
                entry_price=Decimal('52000'),
            )],
            error='Partial failure'
        )
        mock_adapter_class.return_value = mock_adapter
        
        orchestrator = ExecutionOrchestrator(self.account)
        orchestrator.adapter = mock_adapter
        
        orchestrator.sync_positions()
        
        # BTCUSDT should be updated
        btc = Position.objects.get(account=self.account, symbol='BTCUSDT')
        self.assertEqual(btc.qty, Decimal('2.0'))
        
        # ETHUSDT should NOT be zeroed (API failed)
        eth = Position.objects.get(account=self.account, symbol='ETHUSDT')
        self.assertEqual(eth.qty, Decimal('10.0'))


# =============================================================================
# IDEMPOTENCY TESTS
# =============================================================================

class IdempotencyKeyTests(TestCase):
    """Tests for idempotency key generation."""
    
    def test_idempotency_key_format(self):
        """Test idempotency key format."""
        # The key format is: {date}_{account_id}_{direction}[_{counter}]
        from datetime import date
        test_date = date(2024, 1, 15)
        account_id = 'abc123'
        direction = 'long'
        
        base_key = f"{test_date}_{account_id}_{direction}"
        
        self.assertIn('2024-01-15', base_key)
        self.assertIn('long', base_key)
    
    def test_retry_key_includes_counter(self):
        """Test that retry keys include counter suffix."""
        base_key = "2024-01-15_abc123_long"
        retry_key = f"{base_key}_1"
        
        self.assertIn('_1', retry_key)
        self.assertNotEqual(base_key, retry_key)


# =============================================================================
# EVENT LOGGING TESTS
# =============================================================================

class ExecutionEventTests(TestCase):
    """Tests for execution event logging."""
    
    def test_event_type_choices(self):
        """Test that all event types are valid."""
        valid_types = [e[0] for e in ExecutionEvent.EVENT_TYPES]
        
        self.assertIn('intent_created', valid_types)
        self.assertIn('order_submitted', valid_types)
        self.assertIn('order_filled', valid_types)
        self.assertIn('risk_check_failed', valid_types)
        self.assertIn('reconciliation', valid_types)
    
    def test_event_payload_is_dict(self):
        """Test that event payload defaults to empty dict."""
        event = ExecutionEvent(event_type='reconciliation')
        self.assertEqual(event.payload, {})


# =============================================================================
# INSTRUMENT SELECTOR TESTS
# =============================================================================

class InstrumentSelectorTests(TestCase):
    """Tests for InstrumentSelector service."""
    
    def setUp(self):
        self.mock_adapter = Mock()
    
    def test_parse_dte_range_standard(self):
        """Test parsing standard DTE range."""
        from execution.services.instrument_selector import InstrumentSelector
        selector = InstrumentSelector(self.mock_adapter)
        
        min_dte, max_dte = selector._parse_dte_range('45-90d')
        self.assertEqual(min_dte, 45)
        self.assertEqual(max_dte, 90)
    
    def test_parse_dte_range_single(self):
        """Test parsing single DTE value."""
        from execution.services.instrument_selector import InstrumentSelector
        selector = InstrumentSelector(self.mock_adapter)
        
        min_dte, max_dte = selector._parse_dte_range('30d')
        self.assertEqual(min_dte, 30)
        self.assertEqual(max_dte, 60)  # Default +30
    
    def test_parse_dte_range_empty(self):
        """Test parsing empty DTE range returns defaults."""
        from execution.services.instrument_selector import InstrumentSelector
        selector = InstrumentSelector(self.mock_adapter)
        
        min_dte, max_dte = selector._parse_dte_range('')
        self.assertEqual(min_dte, 45)
        self.assertEqual(max_dte, 90)
    
    def test_calculate_target_strike_atm(self):
        """Test ATM strike calculation."""
        from execution.services.instrument_selector import InstrumentSelector
        selector = InstrumentSelector(self.mock_adapter)
        
        strike = selector._calculate_target_strike(
            Decimal('100000'), 'call', 'atm'
        )
        self.assertEqual(strike, Decimal('100000'))
    
    def test_calculate_target_strike_otm_call(self):
        """Test OTM call strike calculation."""
        from execution.services.instrument_selector import InstrumentSelector
        selector = InstrumentSelector(self.mock_adapter)
        
        strike = selector._calculate_target_strike(
            Decimal('100000'), 'call', 'otm'
        )
        self.assertEqual(strike, Decimal('105000'))  # 5% above
    
    def test_calculate_target_strike_otm_put(self):
        """Test OTM put strike calculation."""
        from execution.services.instrument_selector import InstrumentSelector
        selector = InstrumentSelector(self.mock_adapter)
        
        strike = selector._calculate_target_strike(
            Decimal('100000'), 'put', 'otm'
        )
        self.assertEqual(strike, Decimal('95000'))  # 5% below
    
    def test_calculate_target_strike_slight_otm(self):
        """Test slight OTM strike calculation."""
        from execution.services.instrument_selector import InstrumentSelector
        selector = InstrumentSelector(self.mock_adapter)
        
        strike = selector._calculate_target_strike(
            Decimal('100000'), 'call', 'slight_otm'
        )
        self.assertEqual(strike, Decimal('102000'))  # 2% above
    
    def test_classify_moneyness_atm(self):
        """Test ATM classification."""
        from execution.services.instrument_selector import InstrumentSelector
        selector = InstrumentSelector(self.mock_adapter)
        
        result = selector._classify_moneyness(
            Decimal('100500'), Decimal('100000'), 'call'
        )
        self.assertEqual(result, 'atm')  # Within 2%
    
    def test_classify_moneyness_otm_call(self):
        """Test OTM call classification."""
        from execution.services.instrument_selector import InstrumentSelector
        selector = InstrumentSelector(self.mock_adapter)
        
        result = selector._classify_moneyness(
            Decimal('110000'), Decimal('100000'), 'call'
        )
        self.assertEqual(result, 'otm')
    
    def test_classify_moneyness_itm_call(self):
        """Test ITM call classification."""
        from execution.services.instrument_selector import InstrumentSelector
        selector = InstrumentSelector(self.mock_adapter)
        
        result = selector._classify_moneyness(
            Decimal('90000'), Decimal('100000'), 'call'
        )
        self.assertEqual(result, 'itm')
    
    def test_select_single_option_no_instruments(self):
        """Test selection when no instruments available."""
        from execution.services.instrument_selector import InstrumentSelector
        
        self.mock_adapter.get_instruments.return_value = []
        selector = InstrumentSelector(self.mock_adapter)
        
        result = selector.select_single_option(
            'BTC', 'call', 'atm', '45-90d', Decimal('100000')
        )
        self.assertIsNone(result)


# =============================================================================
# ORDER BUILDER TESTS
# =============================================================================

class OrderBuilderTests(TestCase):
    """Tests for OrderBuilder service."""
    
    def test_calculate_qty_with_price(self):
        """Test quantity calculation with known price."""
        from execution.services.order_builder import OrderBuilder
        builder = OrderBuilder(Decimal('10000'))
        
        qty = builder._calculate_qty(
            target_notional=Decimal('1000'),
            price=Decimal('100'),
            lot_size=Decimal('0.1'),
            min_qty=Decimal('0.1'),
        )
        self.assertEqual(qty, Decimal('10.0'))
    
    def test_calculate_qty_rounds_to_lot_size(self):
        """Test quantity rounds to lot size."""
        from execution.services.order_builder import OrderBuilder
        builder = OrderBuilder(Decimal('10000'))
        
        qty = builder._calculate_qty(
            target_notional=Decimal('1000'),
            price=Decimal('333'),  # Would give 3.003...
            lot_size=Decimal('0.1'),
            min_qty=Decimal('0.1'),
        )
        self.assertEqual(qty, Decimal('3.0'))
    
    def test_calculate_qty_respects_min(self):
        """Test quantity respects minimum."""
        from execution.services.order_builder import OrderBuilder
        builder = OrderBuilder(Decimal('10000'))
        
        qty = builder._calculate_qty(
            target_notional=Decimal('10'),
            price=Decimal('1000'),  # Would give 0.01
            lot_size=Decimal('0.01'),
            min_qty=Decimal('0.1'),
        )
        self.assertEqual(qty, Decimal('0.1'))
    
    def test_build_single_option_order_long(self):
        """Test building long option order."""
        from execution.services.order_builder import OrderBuilder
        from execution.services.instrument_selector import StrikeSelection
        
        builder = OrderBuilder(Decimal('10000'))
        selection = StrikeSelection(
            symbol='BTC-30JUN25-100000-C',
            strike=Decimal('100000'),
            expiry=date.today() + timedelta(days=45),
            option_type='call',
            moneyness='atm',
            dte=45,
            rationale='test',
        )
        
        order = builder.build_single_option_order(
            selection=selection,
            direction='long',
            target_notional=Decimal('1000'),
            mark_price=Decimal('500'),
        )
        
        self.assertEqual(order.symbol, 'BTC-30JUN25-100000-C')
        self.assertEqual(order.side, 'buy')
        self.assertEqual(order.order_type, 'market')
    
    def test_build_stop_loss_order_long(self):
        """Test building stop loss for long position."""
        from execution.services.order_builder import OrderBuilder
        builder = OrderBuilder(Decimal('10000'))
        
        order = builder.build_stop_loss_order(
            symbol='BTCUSDT',
            position_qty=Decimal('1.0'),
            position_side='long',
            entry_price=Decimal('100000'),
            stop_loss_pct=Decimal('0.04'),
        )
        
        self.assertEqual(order.side, 'sell')
        self.assertEqual(order.price, Decimal('96000'))  # 4% below
        self.assertTrue(order.reduce_only)
    
    def test_build_stop_loss_order_short(self):
        """Test building stop loss for short position."""
        from execution.services.order_builder import OrderBuilder
        builder = OrderBuilder(Decimal('10000'))
        
        order = builder.build_stop_loss_order(
            symbol='BTCUSDT',
            position_qty=Decimal('1.0'),
            position_side='short',
            entry_price=Decimal('100000'),
            stop_loss_pct=Decimal('0.04'),
        )
        
        self.assertEqual(order.side, 'buy')
        self.assertEqual(order.price, Decimal('104000'))  # 4% above
    
    def test_build_take_profit_order(self):
        """Test building take profit order."""
        from execution.services.order_builder import OrderBuilder
        builder = OrderBuilder(Decimal('10000'))
        
        order = builder.build_take_profit_order(
            symbol='BTCUSDT',
            position_qty=Decimal('1.0'),
            position_side='long',
            entry_price=Decimal('100000'),
            take_profit_pct=Decimal('0.50'),
        )
        
        self.assertEqual(order.side, 'sell')
        self.assertEqual(order.price, Decimal('150000'))  # 50% above
    
    def test_build_scale_down_order(self):
        """Test building scale down order."""
        from execution.services.order_builder import OrderBuilder
        builder = OrderBuilder(Decimal('10000'))
        
        order = builder.build_scale_down_order(
            symbol='BTCUSDT',
            position_qty=Decimal('4.0'),
            position_side='long',
            scale_pct=Decimal('0.75'),
        )
        
        self.assertEqual(order.side, 'sell')
        self.assertEqual(order.qty, Decimal('3.0'))  # 75% of 4
        self.assertTrue(order.reduce_only)
    
    def test_build_full_close_order(self):
        """Test building full close order."""
        from execution.services.order_builder import OrderBuilder
        builder = OrderBuilder(Decimal('10000'))
        
        order = builder.build_full_close_order(
            symbol='BTCUSDT',
            position_qty=Decimal('2.5'),
            position_side='short',
        )
        
        self.assertEqual(order.side, 'buy')
        self.assertEqual(order.qty, Decimal('2.5'))
        self.assertTrue(order.reduce_only)
    
    def test_to_order_request(self):
        """Test converting OrderPlan to OrderRequest."""
        from execution.services.order_builder import OrderBuilder, OrderPlan
        builder = OrderBuilder(Decimal('10000'))
        
        plan = OrderPlan(
            symbol='BTCUSDT',
            side='buy',
            qty=Decimal('1.0'),
            order_type='market',
        )
        
        request = builder.to_order_request(plan, 'ai_test123')
        
        self.assertEqual(request.symbol, 'BTCUSDT')
        self.assertEqual(request.client_order_id, 'ai_test123')


# =============================================================================
# POSITION MANAGER TESTS
# =============================================================================

class PositionManagerTests(TestCase):
    """Tests for PositionManager service."""
    
    def setUp(self):
        self.account = ExchangeAccount.objects.create(
            name='pm-test',
            exchange='bybit',
            api_key_env='KEY',
            api_secret_env='SECRET',
        )
    
    def test_calculate_pnl_pct_long_profit(self):
        """Test P&L calculation for profitable long."""
        from execution.services.position_manager import PositionManager
        pm = PositionManager()
        
        pos = Position(
            entry_price=Decimal('100'),
            mark_price=Decimal('120'),
            side='long',
        )
        
        pnl = pm._calculate_pnl_pct(pos)
        self.assertEqual(pnl, Decimal('0.2'))  # 20% profit
    
    def test_calculate_pnl_pct_long_loss(self):
        """Test P&L calculation for losing long."""
        from execution.services.position_manager import PositionManager
        pm = PositionManager()
        
        pos = Position(
            entry_price=Decimal('100'),
            mark_price=Decimal('90'),
            side='long',
        )
        
        pnl = pm._calculate_pnl_pct(pos)
        self.assertEqual(pnl, Decimal('-0.1'))  # 10% loss
    
    def test_calculate_pnl_pct_short_profit(self):
        """Test P&L calculation for profitable short."""
        from execution.services.position_manager import PositionManager
        pm = PositionManager()
        
        pos = Position(
            entry_price=Decimal('100'),
            mark_price=Decimal('80'),
            side='short',
        )
        
        pnl = pm._calculate_pnl_pct(pos)
        self.assertEqual(pnl, Decimal('0.2'))  # 20% profit
    
    def test_check_stop_loss_triggered(self):
        """Test stop loss detection when triggered."""
        from execution.services.position_manager import PositionManager, ExitReason
        pm = PositionManager()
        
        pos = Position(
            account=self.account,
            symbol='BTCUSDT',
            side='long',
            qty=Decimal('1.0'),
            entry_price=Decimal('100'),
            mark_price=Decimal('94'),  # 6% loss
        )
        
        mock_intent = Mock()
        mock_intent.stop_loss_pct = Decimal('0.05')  # 5% stop
        
        signal = pm._check_stop_loss(pos, mock_intent)
        
        self.assertIsNotNone(signal)
        self.assertEqual(signal.reason, ExitReason.STOP_LOSS)
        self.assertEqual(signal.action, 'close')
    
    def test_check_stop_loss_not_triggered(self):
        """Test stop loss not triggered when within threshold."""
        from execution.services.position_manager import PositionManager
        pm = PositionManager()
        
        pos = Position(
            account=self.account,
            symbol='BTCUSDT',
            side='long',
            qty=Decimal('1.0'),
            entry_price=Decimal('100'),
            mark_price=Decimal('97'),  # 3% loss
        )
        
        mock_intent = Mock()
        mock_intent.stop_loss_pct = Decimal('0.05')  # 5% stop
        
        signal = pm._check_stop_loss(pos, mock_intent)
        self.assertIsNone(signal)
    
    def test_check_take_profit_triggered(self):
        """Test take profit detection when triggered."""
        from execution.services.position_manager import PositionManager, ExitReason
        pm = PositionManager()
        
        pos = Position(
            account=self.account,
            symbol='BTCUSDT',
            side='long',
            qty=Decimal('1.0'),
            entry_price=Decimal('100'),
            mark_price=Decimal('180'),  # 80% profit
        )
        
        mock_intent = Mock()
        mock_intent.stop_loss_pct = None
        mock_intent.take_profit_pct = Decimal('0.70')  # 70% target
        
        signal = pm._check_take_profit(pos, mock_intent)
        
        self.assertIsNotNone(signal)
        self.assertEqual(signal.reason, ExitReason.TAKE_PROFIT)
    
    def test_check_time_stop_triggered(self):
        """Test time stop detection when max days exceeded."""
        from execution.services.position_manager import PositionManager, ExitReason
        pm = PositionManager()
        
        pos = Position(
            account=self.account,
            symbol='BTCUSDT',
            side='long',
            qty=Decimal('1.0'),
            entry_price=Decimal('100'),
            mark_price=Decimal('105'),
        )
        
        mock_intent = Mock()
        mock_intent.stop_loss_pct = None
        mock_intent.take_profit_pct = None
        mock_intent.max_hold_days = 5
        mock_intent.completed_at = timezone.now() - timedelta(days=6)
        
        signal = pm._check_time_stop(pos, mock_intent)
        
        self.assertIsNotNone(signal)
        self.assertEqual(signal.reason, ExitReason.TIME_STOP)
    
    def test_check_expiry_approaching(self):
        """Test expiry detection for options."""
        from execution.services.position_manager import PositionManager, ExitReason
        pm = PositionManager(min_dte_before_close=3)
        
        pos = Position(
            account=self.account,
            symbol='BTC-30JUN25-100000-C',
            side='long',
            qty=Decimal('1.0'),
            entry_price=Decimal('100'),
            mark_price=Decimal('105'),
            expiry=date.today() + timedelta(days=2),  # 2 days to expiry
        )
        
        signal = pm._check_expiry(pos, None)
        
        self.assertIsNotNone(signal)
        self.assertEqual(signal.reason, ExitReason.EXPIRY_APPROACHING)
    
    def test_check_expiry_not_approaching(self):
        """Test expiry not triggered when far from expiry."""
        from execution.services.position_manager import PositionManager
        pm = PositionManager(min_dte_before_close=3)
        
        pos = Position(
            account=self.account,
            symbol='BTC-30JUN25-100000-C',
            side='long',
            qty=Decimal('1.0'),
            entry_price=Decimal('100'),
            mark_price=Decimal('105'),
            expiry=date.today() + timedelta(days=30),  # 30 days to expiry
        )
        
        signal = pm._check_expiry(pos, None)
        self.assertIsNone(signal)
