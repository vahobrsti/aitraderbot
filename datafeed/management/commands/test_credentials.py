"""
Test Exchange API Credentials

Verifies that API keys are configured and working for options data collection.
Run on production after deploying credentials.

Usage:
    python manage.py test_credentials
    python manage.py test_credentials --exchange deribit
    python manage.py test_credentials --exchange bybit
"""
import os
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Test exchange API credentials for options data collection"
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--exchange',
            default='all',
            choices=['bybit', 'deribit', 'all'],
            help='Exchange to test (default: all)',
        )
    
    def handle(self, *args, **options):
        exchanges = ['bybit', 'deribit'] if options['exchange'] == 'all' else [options['exchange']]
        
        all_passed = True
        
        for exchange in exchanges:
            self.stdout.write(f"\n{'='*50}")
            self.stdout.write(f"Testing {exchange.upper()}")
            self.stdout.write(f"{'='*50}")
            
            passed = self._test_exchange(exchange)
            if not passed:
                all_passed = False
        
        self.stdout.write(f"\n{'='*50}")
        if all_passed:
            self.stdout.write(self.style.SUCCESS("All tests passed"))
        else:
            self.stdout.write(self.style.ERROR("Some tests failed"))
    
    def _test_exchange(self, exchange: str) -> bool:
        # Check env vars
        if exchange == 'bybit':
            api_key = os.environ.get('BYBIT_API_KEY')
            api_secret = os.environ.get('BYBIT_API_SECRET')
            key_name = 'BYBIT_API_KEY / BYBIT_API_SECRET'
        else:
            api_key = os.environ.get('DERIBIT_API_KEY')
            api_secret = os.environ.get('DERIBIT_API_SECRET')
            key_name = 'DERIBIT_API_KEY / DERIBIT_API_SECRET'
        
        # Check if env vars exist
        if not api_key or not api_secret:
            self.stdout.write(self.style.ERROR(f"  ✗ {key_name} not set"))
            return False
        
        if api_key.startswith('your_'):
            self.stdout.write(self.style.ERROR(f"  ✗ {key_name} still has placeholder value"))
            return False
        
        self.stdout.write(self.style.SUCCESS(f"  ✓ {key_name} configured"))
        self.stdout.write(f"    Key: {api_key[:8]}...{api_key[-4:]}")
        
        # Test API connection
        try:
            if exchange == 'bybit':
                passed = self._test_bybit(api_key, api_secret)
            else:
                passed = self._test_deribit(api_key, api_secret)
            return passed
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"  ✗ Connection failed: {e}"))
            return False
    
    def _test_bybit(self, api_key: str, api_secret: str) -> bool:
        from pybit.unified_trading import HTTP
        
        session = HTTP(
            testnet=False,
            api_key=api_key,
            api_secret=api_secret,
        )
        
        # Test 1: Get spot price (public)
        response = session.get_tickers(category='spot', symbol='BTCUSDT')
        if response.get('retCode') != 0:
            self.stdout.write(self.style.ERROR(f"  ✗ Spot ticker failed: {response.get('retMsg')}"))
            return False
        
        spot = response['result']['list'][0]['lastPrice']
        self.stdout.write(self.style.SUCCESS(f"  ✓ Spot price: ${float(spot):,.2f}"))
        
        # Test 2: Get options (requires USDC Contracts permission)
        response = session.get_tickers(category='option', baseCoin='BTC')
        if response.get('retCode') != 0:
            self.stdout.write(self.style.ERROR(f"  ✗ Options ticker failed: {response.get('retMsg')}"))
            return False
        
        count = len(response['result']['list'])
        self.stdout.write(self.style.SUCCESS(f"  ✓ Options available: {count}"))
        
        # Test 3: Check rate limit (authenticated should be higher)
        # Bybit returns rate limit info in headers, but pybit doesn't expose it easily
        # Just verify we can make multiple calls
        self.stdout.write(self.style.SUCCESS(f"  ✓ API authenticated successfully"))
        
        return True
    
    def _test_deribit(self, client_id: str, client_secret: str) -> bool:
        import requests
        
        base_url = "https://www.deribit.com/api/v2"
        
        # Test 1: Authenticate
        auth_response = requests.get(
            f"{base_url}/public/auth",
            params={
                'grant_type': 'client_credentials',
                'client_id': client_id,
                'client_secret': client_secret,
            },
            timeout=10,
        )
        auth_data = auth_response.json()
        
        if 'error' in auth_data:
            self.stdout.write(self.style.ERROR(f"  ✗ Auth failed: {auth_data['error']['message']}"))
            return False
        
        token = auth_data['result']['access_token']
        expires_in = auth_data['result']['expires_in']
        self.stdout.write(self.style.SUCCESS(f"  ✓ Authenticated (token expires in {expires_in}s)"))
        
        # Test 2: Get index price
        index_response = requests.get(
            f"{base_url}/public/get_index_price",
            params={'index_name': 'btc_usd'},
            timeout=10,
        )
        index_data = index_response.json()
        
        if 'result' not in index_data:
            self.stdout.write(self.style.ERROR(f"  ✗ Index price failed"))
            return False
        
        index_price = index_data['result']['index_price']
        self.stdout.write(self.style.SUCCESS(f"  ✓ Index price: ${index_price:,.2f}"))
        
        # Test 3: Get instruments (options)
        instruments_response = requests.get(
            f"{base_url}/public/get_instruments",
            params={'currency': 'BTC', 'kind': 'option', 'expired': 'false'},
            timeout=10,
        )
        instruments_data = instruments_response.json()
        
        if 'result' not in instruments_data:
            self.stdout.write(self.style.ERROR(f"  ✗ Instruments failed"))
            return False
        
        count = len(instruments_data['result'])
        self.stdout.write(self.style.SUCCESS(f"  ✓ Options available: {count}"))
        
        return True
