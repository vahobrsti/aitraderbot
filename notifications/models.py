from django.db import models


class WheelPublication(models.Model):
    """
    Deduplication marker for IBIT wheel Telegram alerts.

    The publish_ibit_wheel command runs on an hourly market-hours cron, so
    without a marker the same wheel selection would be re-sent every hour. We
    record the last-published selection per (signal_date, trade_decision) and
    skip when an identical selection was already sent — but re-publish when the
    selection changes intraday (the hash differs).
    """
    signal_date = models.DateField(db_index=True)
    trade_decision = models.CharField(max_length=30)
    # sha256 of the selected legs (tier/side/strike/expiry). Identical hash =>
    # nothing changed => skip re-publishing.
    selection_hash = models.CharField(max_length=64)

    sent_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "wheel_publication"
        verbose_name = "Wheel Publication"
        verbose_name_plural = "Wheel Publications"
        constraints = [
            models.UniqueConstraint(
                fields=["signal_date", "trade_decision"],
                name="unique_wheel_publication_per_signal",
            )
        ]

    def __str__(self):
        return f"{self.signal_date} | {self.trade_decision} | {self.selection_hash[:8]}"
