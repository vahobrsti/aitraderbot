# datafeed/management/commands/sync_sheets.py

import os

from django.core.management.base import BaseCommand
from django.db import transaction

from datafeed.ingestion.google_sheets import (
    get_spreadsheet,
    read_two_column_sheet,
    parse_date,
    parse_float,
    verify_sheet_freshness,
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
        parser.add_argument(
            "--skip-freshness-check",
            dest="skip_freshness_check",
            action="store_true",
            default=False,
            help="Skip the freshness verification check (for testing only)",
        )
        parser.add_argument(
            "--control-sheet",
            dest="control_sheet",
            default="control",
            help="Name of the control sheet for freshness check (default: control)",
        )
        parser.add_argument(
            "--control-cell",
            dest="control_cell",
            default="A1",
            help="Cell address to check for freshness (default: A1)",
        )
        parser.add_argument(
            "--sync-days",
            dest="sync_days",
            type=int,
            default=14,
            help="Only sync dates within this many days of today (default: 14, use 0 for all)",
        )
        parser.add_argument(
            "--full-sync",
            dest="full_sync",
            action="store_true",
            default=False,
            help="Force full sync of all dates (same as --sync-days 0)",
        )

    def handle(self, *args, **options):
        sheet_id = options["sheet_id"] or os.environ.get("GSPREAD_SHEET_ID")
        if not sheet_id:
            raise RuntimeError(
                "No sheet ID provided. Use --sheet-id or set GSPREAD_SHEET_ID in .env"
            )

        # Step 2: Verify sheet freshness (unless bypassed)
        if not options["skip_freshness_check"]:
            self._verify_freshness(
                sheet_id,
                options["control_sheet"],
                options["control_cell"],
            )
        else:
            self.stdout.write(
                self.style.WARNING("Skipping freshness check (--skip-freshness-check)")
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
        sync_days = 0 if options["full_sync"] else options["sync_days"]
        with transaction.atomic():
            self._sync_raw_daily_data(merged, sync_days)

        self.stdout.write(self.style.SUCCESS("Sync complete."))

    def _sync_raw_daily_data(self, merged: dict, sync_days: int):
        """
        merged: dict[date -> {field_name: value, ...}]
        sync_days: only update records within this many days of today (0 = all)
        """
        import datetime
        
        today = datetime.date.today()
        created = 0
        updated = 0
        skipped = 0

        for dt, values in merged.items():
            # Skip old dates if sync_days is set (> 0)
            if sync_days > 0:
                days_ago = (today - dt).days
                if days_ago > sync_days:
                    skipped += 1
                    continue
            
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
        if skipped > 0:
            self.stdout.write(
                f"Skipped {skipped} unchanged historical rows (older than {sync_days} days)."
            )

    def _verify_freshness(self, sheet_id: str, control_sheet: str, control_cell: str):
        """
        Verify that the sheet is fresh before syncing.
        Aborts with an error if the control cell doesn't contain today's date.
        """
        self.stdout.write(
            f"Checking freshness: {control_sheet}!{control_cell}"
        )

        try:
            is_fresh, cell_value, expected = verify_sheet_freshness(
                sheet_id=sheet_id,
                control_sheet=control_sheet,
                control_cell=control_cell,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to verify sheet freshness: {e}\n"
                f"Hint: Make sure the '{control_sheet}' sheet exists with cell {control_cell}."
            )

        if not is_fresh:
            raise RuntimeError(
                f"Sheet is stale! {control_sheet}!{control_cell} = \"{cell_value}\" "
                f"but today is \"{expected}\"\n"
                f"Aborting sync to prevent ingesting stale data.\n"
                f"Run 'python manage.py refresh_sheet' first, or use --skip-freshness-check to bypass."
            )

        self.stdout.write(
            self.style.SUCCESS(f"Sheet is fresh: {control_sheet}!{control_cell} = {cell_value}")
        )
