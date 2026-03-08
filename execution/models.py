"""
Execution domain models for exchange integration.
Tracks accounts, intents, orders, fills, positions, and events.
"""
from django.db import models
from django.utils import timezone
import uuid


class ExchangeAccount(models.Model):
    """
    Exchange account credentials and configuration.
    Supports multiple accounts per exchange (e.g., testnet vs mainnet).
    """
    EXCHANGE_CHOICES = [
        ('bybit', 'Bybit'),
        ('deribit', 'Deribit'),
    ]
    ACCOUNT_TYPE_CHOICES = [
        ('unified', 'Unified Trading'),
        ('classic', 'Classic'),
        ('portfolio', 'Portfolio Margin'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100, help_text="Friendly name for this account")
    exchange = models.CharField(max_length=20, choices=EXCHANGE_CHOICES)
    account_type = models.CharField(max_length=20, choices=ACCOUNT_TYPE_CHOICES, default='unified')
    
    # Credentials stored encrypted in env or secrets manager - these are references
    api_key_env = models.CharField(
        max_length=100, 
        help_text="Environment variable name for API key (e.g., BYBIT_API_KEY)"
    )
    api_secret_env = models.CharField(
        max_length=100,
        help_text="Environment variable name for API secret"
    )
    
    is_testnet = models.BooleanField(default=True, help_text="Use testnet endpoints")
    is_active = models.BooleanField(default=True)
    
    # Risk limits
    max_position_usd = models.DecimalField(
        max_digits=12, decimal_places=2, default=10000,
        help_text="Maximum position size in USD"
    )
    max_daily_loss_usd = models.DecimalField(
        max_digits=12, decimal_places=2, default=1000,
        help_text="Daily loss limit in USD"
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'exchange_account'
        verbose_name = 'Exchange Account'
        verbose_name_plural = 'Exchange Accounts'

    def __str__(self):
        return f"{self.name} ({self.exchange})"


class ExecutionIntent(models.Model):
    """
    Represents the intent to execute a trade based on a DailySignal.
    Tracks the lifecycle from signal -> risk check -> execution -> completion.
    """
    STATUS_CHOICES = [
        ('pending', 'Pending'),           # Created, awaiting processing
        ('risk_check', 'Risk Check'),     # Undergoing risk validation
        ('approved', 'Approved'),         # Passed risk checks
        ('rejected', 'Rejected'),         # Failed risk checks
        ('executing', 'Executing'),       # Orders being placed
        ('partial', 'Partially Filled'),  # Some orders filled
        ('filled', 'Filled'),             # All orders filled
        ('cancelled', 'Cancelled'),       # Manually cancelled
        ('failed', 'Failed'),             # Execution failed
    ]
    DIRECTION_CHOICES = [
        ('long', 'Long'),
        ('short', 'Short'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    signal = models.ForeignKey(
        'signals.DailySignal', 
        on_delete=models.PROTECT,
        related_name='execution_intents'
    )
    account = models.ForeignKey(
        ExchangeAccount,
        on_delete=models.PROTECT,
        related_name='intents'
    )
    
    # Intent details
    signal_date = models.DateField(db_index=True)
    direction = models.CharField(max_length=10, choices=DIRECTION_CHOICES)
    instrument_type = models.CharField(
        max_length=20,
        default='option',
        help_text="option, perpetual, future"
    )
    
    # Target parameters from signal
    target_symbol = models.CharField(max_length=50, blank=True)
    target_qty = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True)
    target_notional_usd = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    
    # Option-specific
    option_type = models.CharField(max_length=10, blank=True, help_text="call or put")
    strike_price = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    expiry_date = models.DateField(null=True, blank=True)
    
    # Risk parameters
    stop_loss_pct = models.DecimalField(max_digits=5, decimal_places=4, null=True, blank=True)
    take_profit_pct = models.DecimalField(max_digits=5, decimal_places=4, null=True, blank=True)
    max_hold_days = models.IntegerField(null=True, blank=True)
    
    # Status tracking
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    status_reason = models.TextField(blank=True)
    
    # Idempotency
    idempotency_key = models.CharField(
        max_length=100, 
        unique=True,
        help_text="Unique key to prevent duplicate executions"
    )
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    approved_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = 'execution_intent'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['signal_date', 'status']),
            models.Index(fields=['account', 'status']),
        ]

    def __str__(self):
        return f"{self.signal_date} | {self.direction} | {self.status}"

    def save(self, *args, **kwargs):
        if not self.idempotency_key:
            # Include a counter for retries on same day/direction
            base_key = f"{self.signal_date}_{self.account_id}_{self.direction}"
            existing_count = ExecutionIntent.objects.filter(
                idempotency_key__startswith=base_key
            ).count()
            if existing_count > 0:
                self.idempotency_key = f"{base_key}_{existing_count}"
            else:
                self.idempotency_key = base_key
        super().save(*args, **kwargs)


class Order(models.Model):
    """
    Individual order placed on an exchange.
    One ExecutionIntent may spawn multiple orders (legs, retries).
    """
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('submitted', 'Submitted'),
        ('open', 'Open'),
        ('partial', 'Partially Filled'),
        ('filled', 'Filled'),
        ('cancelled', 'Cancelled'),
        ('rejected', 'Rejected'),
        ('expired', 'Expired'),
    ]
    SIDE_CHOICES = [
        ('buy', 'Buy'),
        ('sell', 'Sell'),
    ]
    ORDER_TYPE_CHOICES = [
        ('market', 'Market'),
        ('limit', 'Limit'),
        ('stop_market', 'Stop Market'),
        ('stop_limit', 'Stop Limit'),
        ('take_profit', 'Take Profit'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    intent = models.ForeignKey(
        ExecutionIntent,
        on_delete=models.CASCADE,
        related_name='orders'
    )
    
    # Exchange identifiers
    exchange_order_id = models.CharField(max_length=100, blank=True, db_index=True)
    client_order_id = models.CharField(max_length=100, unique=True)
    
    # Order details
    symbol = models.CharField(max_length=50)
    side = models.CharField(max_length=10, choices=SIDE_CHOICES)
    order_type = models.CharField(max_length=20, choices=ORDER_TYPE_CHOICES)
    qty = models.DecimalField(max_digits=18, decimal_places=8)
    price = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True)
    
    # Stop/TP parameters
    trigger_price = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True)
    reduce_only = models.BooleanField(default=False)
    
    # Status
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    filled_qty = models.DecimalField(max_digits=18, decimal_places=8, default=0)
    avg_fill_price = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True)
    
    # Error tracking
    error_code = models.CharField(max_length=50, blank=True)
    error_message = models.TextField(blank=True)
    retry_count = models.IntegerField(default=0)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    submitted_at = models.DateTimeField(null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'execution_order'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['exchange_order_id']),
            models.Index(fields=['status', 'created_at']),
        ]

    def __str__(self):
        return f"{self.symbol} {self.side} {self.qty} @ {self.price or 'MKT'}"

    def save(self, *args, **kwargs):
        if not self.client_order_id:
            self.client_order_id = f"ai_{uuid.uuid4().hex[:16]}"
        super().save(*args, **kwargs)


class Fill(models.Model):
    """
    Individual fill/execution for an order.
    One order may have multiple partial fills.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    order = models.ForeignKey(Order, on_delete=models.CASCADE, related_name='fills')
    
    exchange_fill_id = models.CharField(max_length=100, unique=True)
    qty = models.DecimalField(max_digits=18, decimal_places=8)
    price = models.DecimalField(max_digits=18, decimal_places=8)
    fee = models.DecimalField(max_digits=18, decimal_places=8, default=0)
    fee_currency = models.CharField(max_length=10, default='USDT')
    
    filled_at = models.DateTimeField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'execution_fill'
        ordering = ['-filled_at']

    def __str__(self):
        return f"{self.qty} @ {self.price}"


class Position(models.Model):
    """
    Current position state synced from exchange.
    Updated by reconciliation jobs.
    """
    SIDE_CHOICES = [
        ('long', 'Long'),
        ('short', 'Short'),
        ('none', 'None'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    account = models.ForeignKey(
        ExchangeAccount,
        on_delete=models.CASCADE,
        related_name='positions'
    )
    intent = models.ForeignKey(
        ExecutionIntent,
        on_delete=models.SET_NULL,
        null=True, blank=True,
        related_name='positions'
    )
    
    symbol = models.CharField(max_length=50, db_index=True)
    side = models.CharField(max_length=10, choices=SIDE_CHOICES)
    qty = models.DecimalField(max_digits=18, decimal_places=8)
    entry_price = models.DecimalField(max_digits=18, decimal_places=8)
    mark_price = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True)
    liquidation_price = models.DecimalField(max_digits=18, decimal_places=8, null=True, blank=True)
    
    unrealized_pnl = models.DecimalField(max_digits=18, decimal_places=8, default=0)
    realized_pnl = models.DecimalField(max_digits=18, decimal_places=8, default=0)
    
    leverage = models.DecimalField(max_digits=5, decimal_places=2, default=1)
    margin_mode = models.CharField(max_length=20, default='cross')
    
    # For options
    option_type = models.CharField(max_length=10, blank=True)
    strike = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    expiry = models.DateField(null=True, blank=True)
    
    synced_at = models.DateTimeField(default=timezone.now)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'execution_position'
        unique_together = ['account', 'symbol']
        ordering = ['-synced_at']

    def __str__(self):
        return f"{self.symbol} {self.side} {self.qty}"


class ExecutionEvent(models.Model):
    """
    Audit log for all execution-related events.
    Enables debugging, reconciliation, and compliance.
    """
    EVENT_TYPES = [
        ('intent_created', 'Intent Created'),
        ('risk_check_passed', 'Risk Check Passed'),
        ('risk_check_failed', 'Risk Check Failed'),
        ('order_submitted', 'Order Submitted'),
        ('order_filled', 'Order Filled'),
        ('order_cancelled', 'Order Cancelled'),
        ('order_rejected', 'Order Rejected'),
        ('order_error', 'Order Error'),
        ('position_opened', 'Position Opened'),
        ('position_closed', 'Position Closed'),
        ('stop_triggered', 'Stop Triggered'),
        ('reconciliation', 'Reconciliation'),
        ('manual_override', 'Manual Override'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    intent = models.ForeignKey(
        ExecutionIntent,
        on_delete=models.CASCADE,
        related_name='events',
        null=True, blank=True
    )
    order = models.ForeignKey(
        Order,
        on_delete=models.CASCADE,
        related_name='events',
        null=True, blank=True
    )
    
    event_type = models.CharField(max_length=30, choices=EVENT_TYPES)
    payload = models.JSONField(default=dict, help_text="Event-specific data")
    
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'execution_event'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['event_type', 'created_at']),
        ]

    def __str__(self):
        return f"{self.event_type} @ {self.created_at}"
