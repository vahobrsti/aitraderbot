"""
REST API serializers.
Centralizes all API serialization logic.
"""
from rest_framework import serializers

from signals.models import DailySignal


class DailySignalSerializer(serializers.ModelSerializer):
    """
    Full serializer for DailySignal model.
    Returns all signal data for API consumers.
    """
    
    class Meta:
        model = DailySignal
        fields = [
            'date',
            'p_long',
            'p_short',
            'signal_option_call',
            'signal_option_put',
            'fusion_state',
            'fusion_confidence',
            'fusion_score',
            'overlay_reason',
            'size_multiplier',
            'dte_multiplier',
            'tactical_put_active',
            'tactical_put_strategy',
            'tactical_put_size',
            'trade_decision',
            'trade_notes',
            'option_structures',
            'strike_guidance',
            'dte_range',
            'strategy_rationale',
            'created_at',
            'updated_at',
        ]
        read_only_fields = fields


class DailySignalSummarySerializer(serializers.ModelSerializer):
    """
    Lightweight serializer for signal lists.
    Returns only key fields for quick scanning.
    """
    
    class Meta:
        model = DailySignal
        fields = [
            'date',
            'p_long',
            'p_short',
            'fusion_state',
            'fusion_score',
            'trade_decision',
        ]
        read_only_fields = fields
