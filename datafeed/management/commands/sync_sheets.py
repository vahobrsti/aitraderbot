# datafeed/management/commands/sync_sheets.py

import os
from email.utils import specialsre

from django.core.management.base import BaseCommand
from django.db import transaction

from datafeed.ingestion.google_sheets import (
    get_spreadsheet,
    read_two_column_sheet,
    parse_date,
    parse_float,
)
from datafeed.models import RawDailyData


# Simple 2-column sheets: Date, Value
SHEET_FIELD_MAP = {
    "mean_dollar_invested_age": "mdia",
    "mvrv_usd_1d": "mvrv_usd_1d",
    "mvrv_usd_7d": "mvrv_usd_7d",
    "mvrv_usd_30d": "mvrv_usd_30d",
    "mvrv_usd_60d": "mvrv_usd_60d",
    "mvrv_usd_90d": "mvrv_usd_90d",
    "mvrv_usd_180d": "mvrv_usd_180d",
    "mvrv_usd_365d": "mvrv_usd_365d",
    "mvrv_long_short_diff_usd": "mvrv_long_short_diff_usd",
    "BTC_holders_1_10": "btc_holders_1_10",
    "BTC_holders_10_100": "btc_holders_10_100",
    "BTC_holders_100_1k": "btc_holders_100_1k",
    "BTC_holders_1k_10k": "btc_holders_1k_10k",
    "sentiment_weighted_total": "sentiment_weighted_total",
    "btc_price":"btc_price_mean"
}


class Command(BaseCommand):
    help = "Sync BTC Santiment Google Sheets into raw_daily_data table"

    def add_arguments(self, parser):
        parser.add_argument(
            "--sheet-id",
            dest="sheet_id",
            default=None,
            help="Google Sheet ID (overrides GSPREAD_SHEET_ID env var if provided)",
        )

    def handle(self, *args, **options):
        sheet_id = options["sheet_id"] or os.environ.get("GSPREAD_SHEET_ID")
        if not sheet_id:
            raise RuntimeError(
                "No sheet ID provided. Use --sheet-id or set GSPREAD_SHEET_ID in .env"
            )

        spreadsheet = get_spreadsheet(sheet_id)
        # 1) Merge all series into dict[date -> {field: value, ...}]
        merged: dict = {}

        # A) 2-column sheets (Date, Value)
        for sheet_name, field_name in SHEET_FIELD_MAP.items():
            self.stdout.write(f"Reading sheet: {sheet_name} -> {field_name}")
            series = read_two_column_sheet(spreadsheet, sheet_name)
            for dt, value in series.items():
                merged.setdefault(dt, {})
                merged[dt][field_name] = value

        # B) btc_OHLC sheet (assumed columns: Date, Open, High, Low, Close)
        try:
            self.stdout.write("Reading sheet: btc_OHLC")
            ws = spreadsheet.worksheet("btc_OHLC")
            rows = ws.get_all_records()
            for row in rows:
                date_str = row.get("Date")
                if not date_str:
                    continue
                dt = parse_date(date_str)
                if not dt:
                    continue
                merged.setdefault(dt, {})
                merged[dt]["btc_open"] = parse_float(row.get("Open Price USD"))
                merged[dt]["btc_high"] = parse_float(row.get("High Price USD"))
                merged[dt]["btc_low"] = parse_float(row.get("Low Price USD"))
                merged[dt]["btc_close"] = parse_float(row.get("Close Price USD"))
        except Exception as e:
            self.stdout.write(self.style.WARNING(f"btc_OHLC sheet skipped: {e}"))

        # C) btc_price sheet (assumed: Date, mean_price, normalised_price)
        try:
            self.stdout.write("Reading sheet: btc_price")
            ws = spreadsheet.worksheet("btc_price")
            rows = ws.get_all_records()
            for row in rows:
                date_str = row.get("Date")
                if not date_str:
                    continue
                dt = parse_date(date_str)
                if not dt:
                    continue
                merged.setdefault(dt, {})
                merged[dt]["btc_price_mean"] = parse_float(
                    row.get("Value")
                )
        except Exception as e:
            self.stdout.write(self.style.WARNING(f"btc_price sheet skipped: {e}"))

        self.stdout.write(f"Total dates merged: {len(merged)}")

        # 2) Upsert into RawDailyData
        with transaction.atomic():
            self._sync_raw_daily_data(merged)

        self.stdout.write(self.style.SUCCESS("Sync complete."))

    def _sync_raw_daily_data(self, merged: dict):
        """
        merged: dict[date -> {field_name: value, ...}]
        """
        created = 0
        updated = 0

        for dt, values in merged.items():
            obj, is_created = RawDailyData.objects.update_or_create(
                date=dt,
                defaults=values,
            )
            if is_created:
                created += 1
            else:
                updated += 1

        self.stdout.write(
            f"Created {created} rows, updated {updated} rows in raw_daily_data."
        )
