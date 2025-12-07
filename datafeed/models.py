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

