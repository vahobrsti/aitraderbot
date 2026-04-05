"""
Utility functions for execution app.
"""
import os
from typing import Optional
from execution.models import ExchangeAccount
from execution.exchanges import BybitAdapter, DeribitAdapter
from execution.exchanges.base import ExchangeAdapter


def get_adapter(account: ExchangeAccount) -> ExchangeAdapter:
    """
    Create exchange adapter from account configuration.
    Loads credentials from environment variables.
    """
    api_key = os.environ.get(account.api_key_env, '')
    api_secret = os.environ.get(account.api_secret_env, '')
    
    if not api_key or not api_secret:
        raise ValueError(f"Missing credentials for {account.name}")
    
    adapters = {
        'bybit': BybitAdapter,
        'deribit': DeribitAdapter,
    }
    
    adapter_class = adapters.get(account.exchange)
    if not adapter_class:
        raise ValueError(f"Unsupported exchange: {account.exchange}")
    
    return adapter_class(api_key, api_secret, account.is_testnet)


def get_default_account(exchange: str = 'bybit') -> Optional[ExchangeAccount]:
    """Get the default active account for an exchange."""
    return ExchangeAccount.objects.filter(
        exchange=exchange,
        is_active=True,
    ).first()
