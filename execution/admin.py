from django.contrib import admin
from .models import (
    ExchangeAccount, ExecutionIntent, Order, Fill, Position, ExecutionEvent
)


@admin.register(ExchangeAccount)
class ExchangeAccountAdmin(admin.ModelAdmin):
    list_display = ['name', 'exchange', 'account_type', 'is_active', 'is_testnet', 'created_at']
    list_filter = ['exchange', 'account_type', 'is_active', 'is_testnet']
    search_fields = ['name']


@admin.register(ExecutionIntent)
class ExecutionIntentAdmin(admin.ModelAdmin):
    list_display = ['id', 'signal_date', 'direction', 'status', 'account', 'created_at']
    list_filter = ['status', 'direction', 'account__exchange']
    date_hierarchy = 'signal_date'
    raw_id_fields = ['signal', 'account']


@admin.register(Order)
class OrderAdmin(admin.ModelAdmin):
    list_display = ['id', 'intent', 'exchange_order_id', 'symbol', 'side', 'order_type', 'status', 'qty', 'price']
    list_filter = ['status', 'side', 'order_type', 'intent__account__exchange']
    search_fields = ['exchange_order_id', 'symbol']
    raw_id_fields = ['intent']


@admin.register(Fill)
class FillAdmin(admin.ModelAdmin):
    list_display = ['id', 'order', 'exchange_fill_id', 'qty', 'price', 'fee', 'filled_at']
    list_filter = ['order__intent__account__exchange']
    raw_id_fields = ['order']


@admin.register(Position)
class PositionAdmin(admin.ModelAdmin):
    list_display = ['id', 'account', 'symbol', 'side', 'qty', 'entry_price', 'unrealized_pnl', 'synced_at']
    list_filter = ['side', 'account__exchange']
    search_fields = ['symbol']
    raw_id_fields = ['account', 'intent']


@admin.register(ExecutionEvent)
class ExecutionEventAdmin(admin.ModelAdmin):
    list_display = ['id', 'intent', 'event_type', 'created_at']
    list_filter = ['event_type']
    raw_id_fields = ['intent', 'order']
