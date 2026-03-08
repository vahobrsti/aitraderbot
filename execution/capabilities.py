"""
Exchange capability matrix for OPTIONS.
Neither Bybit nor Deribit support native SL/TP for options.
All exit management is polling-based via manage_exits command.
"""

# Options do NOT support native conditional orders on any exchange
# Exit management is 100% polling-based

OPTION_EXIT_METHOD = 'polling'

def supports_native_exits(exchange: str, instrument_type: str) -> bool:
    """
    Check if exchange supports native SL/TP for this instrument.
    For options: always False.
    """
    if instrument_type == 'option':
        return False
    # We only trade options, but keep this for future
    return False


def get_exit_method(exchange: str, instrument_type: str) -> str:
    """Get exit method for instrument type. Always 'polling' for options."""
    return 'polling'
