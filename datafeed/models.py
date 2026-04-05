from django.db import models


class RawDailyData(models.Model):
    """
    Canonical raw daily BTC data:
    one row per date, columns match what you have in Google Sheets.
    """
    date = models.DateField(unique=True)

    # On-chain metrics
    mdia = models.FloatField(null=True, blank=True)  # mean_dollar_invested_age

    mvrv_long_short_diff_usd = models.FloatField(null=True, blank=True)

    mvrv_usd_1d = models.FloatField(null=True, blank=True)
    mvrv_usd_7d = models.FloatField(null=True, blank=True)
    mvrv_usd_30d = models.FloatField(null=True, blank=True)
    mvrv_usd_60d = models.FloatField(null=True, blank=True)
    mvrv_usd_90d = models.FloatField(null=True, blank=True)
    mvrv_usd_180d = models.FloatField(null=True, blank=True)
    mvrv_usd_365d = models.FloatField(null=True, blank=True)

    # Holder buckets
    btc_holders_1_10 = models.FloatField(null=True, blank=True)
    btc_holders_10_100 = models.FloatField(null=True, blank=True)
    btc_holders_100_1k = models.FloatField(null=True, blank=True)
    btc_holders_1k_10k = models.FloatField(null=True, blank=True)

    # Exchange Funds Flow
    exchange_flow_balance = models.FloatField(null=True, blank=True)  # inflows - outflows

    # Sentiment
    sentiment_weighted_total = models.FloatField(null=True, blank=True)

    # BTC OHLC (daily)
    btc_open = models.FloatField(null=True, blank=True)
    btc_high = models.FloatField(null=True, blank=True)
    btc_low = models.FloatField(null=True, blank=True)
    btc_close = models.FloatField(null=True, blank=True)

    # BTC price (from btc_price sheet)
    btc_price_mean = models.FloatField(null=True, blank=True)

    # Housekeeping
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "raw_daily_data"
        ordering = ["date"]

    def __str__(self):
        return f"{self.date}"


class OptionSnapshot(models.Model):
    """
    Point-in-time snapshot of an option contract.
    Captures price, IV, greeks, and liquidity data for modeling.
    
    Granularity: hourly or on-demand snapshots.
    """
    # Identifiers
    timestamp = models.DateTimeField(db_index=True)
    symbol = models.CharField(max_length=64, db_index=True)  # e.g., BTC-26APR26-100000-C
    underlying = models.CharField(max_length=16, default='BTC')  # BTC, ETH
    
    # Contract specs
    expiry = models.DateTimeField()
    strike = models.DecimalField(max_digits=12, decimal_places=2)
    option_type = models.CharField(max_length=4)  # 'call' or 'put'
    
    # Underlying price at snapshot
    spot_price = models.DecimalField(max_digits=12, decimal_places=2)
    index_price = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    
    # Option prices
    bid = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True)
    ask = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True)
    mid_price = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True)
    mark_price = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True)
    last_price = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True)
    
    # Implied volatility
    iv = models.DecimalField(max_digits=8, decimal_places=4, null=True, blank=True)  # e.g., 0.65 = 65%
    
    # Greeks (if available from exchange)
    delta = models.DecimalField(max_digits=8, decimal_places=6, null=True, blank=True)
    gamma = models.DecimalField(max_digits=12, decimal_places=8, null=True, blank=True)
    vega = models.DecimalField(max_digits=12, decimal_places=6, null=True, blank=True)
    theta = models.DecimalField(max_digits=12, decimal_places=6, null=True, blank=True)
    
    # Liquidity
    bid_size = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True)
    ask_size = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True)
    volume_24h = models.DecimalField(max_digits=16, decimal_places=4, null=True, blank=True)
    open_interest = models.DecimalField(max_digits=16, decimal_places=4, null=True, blank=True)
    
    # Derived (computed on save)
    dte = models.FloatField(null=True, blank=True)  # days to expiry
    moneyness = models.FloatField(null=True, blank=True)  # (strike - spot) / spot
    spread_pct = models.FloatField(null=True, blank=True)  # (ask - bid) / mid
    
    # Metadata
    exchange = models.CharField(max_length=16, default='bybit')
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = "option_snapshot"
        indexes = [
            models.Index(fields=['timestamp', 'symbol']),
            models.Index(fields=['underlying', 'expiry', 'strike']),
            models.Index(fields=['timestamp', 'underlying', 'dte']),
        ]
        unique_together = [['timestamp', 'symbol', 'exchange']]
    
    def save(self, *args, **kwargs):
        # Compute derived fields
        if self.expiry and self.timestamp:
            delta_seconds = (self.expiry - self.timestamp).total_seconds()
            self.dte = max(delta_seconds / 86400, 0)
        
        if self.strike is not None and self.spot_price is not None and self.spot_price > 0:
            self.moneyness = float((self.strike - self.spot_price) / self.spot_price)
        
        if self.bid is not None and self.ask is not None and self.mid_price is not None and self.mid_price > 0:
            self.spread_pct = float((self.ask - self.bid) / self.mid_price)
        
        super().save(*args, **kwargs)
    
    def __str__(self):
        return f"{self.symbol} @ {self.timestamp}"


class OptionTrade(models.Model):
    """
    Records actual option trades for PnL tracking and model training.
    Links entry snapshot to exit snapshot for full trade lifecycle.
    """
    # Trade identification
    trade_id = models.CharField(max_length=64, unique=True)
    signal_type = models.CharField(max_length=32, db_index=True)  # BULL_PROBE, PRIMARY_SHORT, etc.
    direction = models.CharField(max_length=8)  # LONG or SHORT
    
    # Entry
    entry_timestamp = models.DateTimeField(db_index=True)
    entry_snapshot = models.ForeignKey(
        OptionSnapshot, 
        on_delete=models.SET_NULL, 
        null=True, 
        related_name='entry_trades'
    )
    entry_price = models.DecimalField(max_digits=12, decimal_places=4)
    entry_spot = models.DecimalField(max_digits=12, decimal_places=2)
    entry_iv = models.DecimalField(max_digits=8, decimal_places=4, null=True, blank=True)
    
    # Position
    symbol = models.CharField(max_length=64)
    qty = models.DecimalField(max_digits=12, decimal_places=4)
    notional = models.DecimalField(max_digits=16, decimal_places=2)  # USD value at entry
    
    # Exit (filled when trade closes)
    exit_timestamp = models.DateTimeField(null=True, blank=True)
    exit_snapshot = models.ForeignKey(
        OptionSnapshot, 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True,
        related_name='exit_trades'
    )
    exit_price = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True)
    exit_spot = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    exit_iv = models.DecimalField(max_digits=8, decimal_places=4, null=True, blank=True)
    exit_reason = models.CharField(max_length=32, null=True, blank=True)  # tp, stop, time_stop, manual
    
    # PnL
    realized_pnl = models.DecimalField(max_digits=16, decimal_places=2, null=True, blank=True)
    pnl_pct = models.FloatField(null=True, blank=True)
    
    # Path stats (computed from snapshots during hold period)
    max_favorable_excursion = models.FloatField(null=True, blank=True)  # best unrealized gain %
    max_adverse_excursion = models.FloatField(null=True, blank=True)  # worst unrealized loss %
    iv_change = models.FloatField(null=True, blank=True)  # exit_iv - entry_iv
    spot_change_pct = models.FloatField(null=True, blank=True)  # (exit_spot - entry_spot) / entry_spot
    
    # Metadata
    is_paper = models.BooleanField(default=True)
    exchange = models.CharField(max_length=16, default='bybit')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = "option_trade"
        indexes = [
            models.Index(fields=['entry_timestamp']),
            models.Index(fields=['signal_type', 'entry_timestamp']),
        ]
    
    def __str__(self):
        status = "OPEN" if self.exit_timestamp is None else "CLOSED"
        return f"{self.trade_id} {self.signal_type} {status}"
