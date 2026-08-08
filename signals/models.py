from django.db import models


class DailySignal(models.Model):
    """
    Stores ML predictions and fusion engine outputs.
    One row per (date, trade_decision) pair. Multiple trade types can
    coexist on the same day (e.g., MVRV_SHORT + IRON_CONDOR).
    Hourly re-evaluation updates the existing row if the decision hasn't changed.
    """
    # Primary temporal key
    date = models.DateField(db_index=True)

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
        max_length=200,
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
    stop_loss = models.CharField(
        max_length=200,
        blank=True,
        default="",
        help_text="Stop loss guidance (e.g., 4.0% stop | scale to 25% on day 5 | hard cut day 6)"
    )
    
    # Numeric execution fields for exchange integration (Bybit/Deribit)
    stop_loss_pct = models.FloatField(
        null=True,
        blank=True,
        help_text="Stop loss as decimal (e.g., 0.04 = 4% adverse move triggers exit)"
    )
    scale_down_day = models.IntegerField(
        null=True,
        blank=True,
        help_text="Day to reduce position to 25% (time-based scale-down)"
    )
    max_hold_days = models.IntegerField(
        null=True,
        blank=True,
        help_text="Hard time stop: close all remaining on this day"
    )
    spread_width_pct = models.FloatField(
        null=True,
        blank=True,
        help_text="Spread width as decimal (e.g., 0.10 = 10% width)"
    )
    take_profit_pct = models.FloatField(
        null=True,
        blank=True,
        help_text="Take profit target as decimal (e.g., 0.70 = 70% of max spread value)"
    )

    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(
        default=True,
        db_index=True,
        help_text="False = manually deactivated by operator. Currently always True (signals are final once fired)."
    )

    # Iron Condor Gate
    condor_score = models.FloatField(
        default=0.0,
        help_text="Range score (0-100) for iron condor eligibility"
    )
    condor_eligible = models.BooleanField(
        default=False,
        help_text="Whether condor gate passed (score >= threshold, no vetoes)"
    )
    condor_veto_reasons = models.JSONField(
        default=list,
        blank=True,
        help_text="Hard veto reasons blocking condor entry"
    )
    condor_score_components = models.JSONField(
        default=dict,
        blank=True,
        help_text="Breakdown of range score components"
    )

    # Iron Condor MVRV-based strike levels
    condor_short_call = models.FloatField(
        null=True,
        blank=True,
        help_text="Short call strike = max(spot*1.10, CB*1.12)"
    )
    condor_short_put = models.FloatField(
        null=True,
        blank=True,
        help_text="Short put strike = min(spot*0.90, CB*0.92)"
    )
    condor_cost_basis = models.FloatField(
        null=True,
        blank=True,
        help_text="MVRV-60d implied cost basis (spot / mvrv_60d)"
    )
    condor_strike_meta = models.JSONField(
        default=dict,
        blank=True,
        help_text="Strike computation metadata: sources, distances, mvrv_60d"
    )

    # Income Spread (Bull Put / Bear Call) risk-tiered setups
    income_spread_setups = models.JSONField(
        default=list,
        blank=True,
        help_text="List of risk-tiered spread setups [{risk_tier, short_strike, long_strike, credit, spread_width, dte, max_loss, short_delta, credit_width_pct, otm_pct, risk_reward}]"
    )
    income_spread_score = models.FloatField(
        default=0.0,
        help_text="Income gate score (0-100)"
    )
    income_spread_eligible = models.BooleanField(
        default=False,
        help_text="Whether income gate passed (score >= threshold, no vetoes, chain valid)"
    )
    income_spread_veto_reasons = models.JSONField(
        default=list,
        blank=True,
        help_text="Veto reasons blocking income spread entry"
    )

    class Meta:
        db_table = "daily_signal"
        ordering = ["-date", "-updated_at"]
        verbose_name = "Daily Signal"
        verbose_name_plural = "Daily Signals"
        constraints = [
            models.UniqueConstraint(
                fields=["date", "trade_decision"],
                name="unique_date_trade_decision",
            )
        ]

    # Trade decision priority (lower number = higher priority).
    # Used for deterministic selection when multiple signals exist on the same day.
    DECISION_PRIORITY = {
        "CALL": 1,
        "PUT": 2,
        "TACTICAL_PUT": 3,
        "OPTION_CALL": 4,
        "OPTION_PUT": 5,
        "MVRV_SHORT": 6,
        "IRON_CONDOR": 7,
        "BULL_PUT_SPREAD": 8,
        "BEAR_CALL_SPREAD": 9,
        "NO_TRADE": 99,
    }

    @property
    def priority(self) -> int:
        return self.DECISION_PRIORITY.get(self.trade_decision, 50)

    @classmethod
    def active(cls):
        """Return queryset of active (non-stale) signals."""
        return cls.objects.filter(is_active=True)

    @classmethod
    def tradeable(cls):
        """Return queryset of active, non-NO_TRADE signals."""
        return cls.objects.filter(is_active=True).exclude(trade_decision="NO_TRADE")

    @classmethod
    def pick_highest_priority(cls, queryset):
        """
        From a queryset of signals, return the one with highest trading priority.
        Only considers active signals. Deterministic: uses DECISION_PRIORITY,
        then falls back to updated_at.
        """
        signals = list(queryset.filter(is_active=True) if hasattr(queryset, 'filter') else [s for s in queryset if s.is_active])
        if not signals:
            return None
        signals.sort(key=lambda s: (cls.DECISION_PRIORITY.get(s.trade_decision, 50), -(s.updated_at.timestamp() if s.updated_at else 0)))
        return signals[0]

    def __str__(self):
        return f"{self.date} | {self.fusion_state} | {self.trade_decision}"


class IbitWheelSetup(models.Model):
    """
    Persisted IBIT wheel selection for an income signal.

    The income-gate signals (BULL_PUT_SPREAD / BEAR_CALL_SPREAD) are run against
    the IBIT option chain to pick single short-leg wheel positions (cash-secured
    put / covered call). Rather than pushing these to Telegram, we store the
    selection here and expose it via the API for on-demand retrieval.

    One row per (signal_date, trade_decision); recomputed intraday and updated in
    place. ``selection_hash`` lets the compute job skip a write when nothing
    changed, so ``updated_at`` reflects the last time the selection actually moved.
    """
    signal_date = models.DateField(db_index=True)
    trade_decision = models.CharField(
        max_length=30, help_text="BULL_PUT_SPREAD or BEAR_CALL_SPREAD"
    )
    side = models.CharField(max_length=4, help_text="put or call")
    position = models.CharField(
        max_length=20, help_text="cash_secured_put or covered_call"
    )
    spot_price = models.FloatField(help_text="IBIT spot used for selection")
    dte_mode = models.CharField(max_length=10, default="income")

    # List of selected legs (one per risk tier). Each leg dict carries
    # risk_tier, strike, delta, credit, premium_usd, otm_pct, dte, expiry,
    # symbol, bid, ask, spread_pct, cash_reserve_usd.
    legs = models.JSONField(default=list)

    # sha256 of the selected legs — used to detect intraday changes.
    selection_hash = models.CharField(max_length=64)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "ibit_wheel_setup"
        ordering = ["-signal_date", "trade_decision"]
        verbose_name = "IBIT Wheel Setup"
        verbose_name_plural = "IBIT Wheel Setups"
        constraints = [
            models.UniqueConstraint(
                fields=["signal_date", "trade_decision"],
                name="unique_ibit_wheel_per_signal",
            )
        ]

    def __str__(self):
        return f"{self.signal_date} | {self.trade_decision} | {len(self.legs)} legs"
