from django.db import models


class DailySignal(models.Model):
    """
    Stores daily ML predictions and fusion engine outputs.
    One row per date, capturing the complete signal state for that day.
    """
    # Primary key
    date = models.DateField(unique=True, db_index=True)

    # ML Model Outputs
    p_long = models.FloatField(help_text="ML probability for long position")
    p_short = models.FloatField(help_text="ML probability for short position")
    signal_option_call = models.IntegerField(help_text="Call signal (0/1)")
    signal_option_put = models.IntegerField(help_text="Put signal (0/1)")

    # Fusion Engine Outputs
    fusion_state = models.CharField(
        max_length=50,
        help_text="Market state from fusion engine (e.g., STRONG_BULLISH)"
    )
    fusion_confidence = models.CharField(
        max_length=20,
        help_text="Confidence level (HIGH/MEDIUM/LOW)"
    )
    fusion_score = models.IntegerField(help_text="Numeric fusion score")
    short_source = models.CharField(
        max_length=10,
        blank=True,
        default="",
        help_text="Source of short signal: rule or score (empty for non-short states)"
    )

    # Overlay Outputs
    overlay_reason = models.TextField(
        blank=True,
        default="",
        help_text="Explanation from overlay logic"
    )
    size_multiplier = models.FloatField(
        default=1.0,
        help_text="Position size multiplier from overlay"
    )
    dte_multiplier = models.FloatField(
        default=1.0,
        help_text="DTE adjustment multiplier"
    )

    # Tactical Put
    tactical_put_active = models.BooleanField(
        default=False,
        help_text="Whether tactical put was triggered"
    )
    tactical_put_strategy = models.CharField(
        max_length=50,
        blank=True,
        default="",
        help_text="Strategy type if tactical put active"
    )
    tactical_put_size = models.FloatField(
        default=0.0,
        help_text="Tactical put sizing"
    )

    # Final Decision
    trade_decision = models.CharField(
        max_length=30,
        help_text="Final trade decision (CALL/PUT/TACTICAL_PUT/NO_TRADE)"
    )
    trade_notes = models.TextField(
        blank=True,
        default="",
        help_text="Additional notes about the trade decision"
    )
    
    # NO_TRADE Diagnostics
    no_trade_reasons = models.JSONField(
        default=list,
        blank=True,
        help_text="List of reason codes for NO_TRADE (e.g., FUSION_STATE_NO_TRADE, CONFIDENCE_TOO_LOW)"
    )
    decision_trace = models.JSONField(
        default=list,
        blank=True,
        help_text="Step-by-step decision gates (fusion → overlay → ml → final)"
    )
    score_components = models.JSONField(
        default=dict,
        blank=True,
        help_text="Breakdown of score: {mdia: {score: 2, label: 'strong_inflow'}, ...}"
    )
    effective_size = models.FloatField(
        default=0.0,
        help_text="Final computed position size (0 on NO_TRADE)"
    )
    
    # Versioning for reproducibility
    decision_version = models.CharField(
        max_length=30,
        blank=True,
        default="",
        help_text="Version of decision logic (e.g., 2025-12-28.1)"
    )
    model_versions = models.JSONField(
        default=dict,
        blank=True,
        help_text="Model versions used: {long: 'filename', short: 'filename'}"
    )

    # Option Strategy (from options.py)
    option_structures = models.CharField(
        max_length=200,
        blank=True,
        default="",
        help_text="Recommended option structures (e.g., long_call, call_spread)"
    )
    strike_guidance = models.CharField(
        max_length=30,
        blank=True,
        default="",
        help_text="Strike selection guidance (e.g., atm, slight_otm)"
    )
    dte_range = models.CharField(
        max_length=30,
        blank=True,
        default="",
        help_text="Days to expiration range (e.g., 45-90d)"
    )
    strategy_rationale = models.TextField(
        blank=True,
        default="",
        help_text="Human-readable strategy rationale"
    )

    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "daily_signal"
        ordering = ["-date"]
        verbose_name = "Daily Signal"
        verbose_name_plural = "Daily Signals"

    def __str__(self):
        return f"{self.date} | {self.fusion_state} | {self.trade_decision}"
