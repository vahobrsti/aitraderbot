# datafeed/management/commands/refresh_mvrv_hourly.py
"""
Hourly refresh: force sheet recalculation and save mvrv_usd_60d to CSV.

Cron example (every hour):
0 * * * * cd /var/www/app && /var/www/app/venv/bin/python manage.py refresh_mvrv_hourly >> /var/www/app/logs/cron.log 2>&1
"""

import os
import datetime
from django.core.management.base import BaseCommand
from datafeed.ingestion.google_sheets import (
    get_spreadsheet,
    force_refresh_control_cell,
    get_write_client,
)


class Command(BaseCommand):
    help = "Force sheet refresh and save mvrv_usd_60d to CSV with hour,value columns"

    def add_arguments(self, parser):
        parser.add_argument(
            "--sheet-id",
            dest="sheet_id",
            default=None,
            help="Google Sheet ID (overrides GSPREAD_SHEET_ID env var)",
        )
        parser.add_argument(
            "--output",
            dest="output",
            default="mvrv_usd_60d_hourly.csv",
            help="Output CSV file path (default: mvrv_usd_60d_hourly.csv)",
        )
        parser.add_argument(
            "--skip-refresh",
            dest="skip_refresh",
            action="store_true",
            default=False,
            help="Skip the sheet refresh step (just read and save)",
        )

    def handle(self, *args, **options):
        sheet_id = options["sheet_id"] or os.environ.get("GSPREAD_SHEET_ID")
        if not sheet_id:
            raise RuntimeError(
                "No sheet ID provided. Use --sheet-id or set GSPREAD_SHEET_ID in .env"
            )

        output_path = options["output"]

        # Step 1: Force refresh (toggle control cell to trigger recalc)
        if not options["skip_refresh"]:
            self.stdout.write("Forcing sheet refresh...")
            new_value = self._force_refresh_with_toggle(sheet_id)
            self.stdout.write(
                self.style.SUCCESS(f"Refresh triggered: wrote '{new_value}' to control!A1")
            )

        # Step 2: Read mvrv_usd_60d sheet
        self.stdout.write("Reading mvrv_usd_60d sheet...")
        spreadsheet = get_spreadsheet(sheet_id)
        
        try:
            ws = spreadsheet.worksheet("mvrv_usd_60d")
            rows = ws.get_all_records()
        except Exception as e:
            raise RuntimeError(f"Failed to read mvrv_usd_60d sheet: {e}")

        # Step 3: Get current hour and latest value
        now = datetime.datetime.utcnow()
        hour_str = now.strftime("%Y-%m-%d %H:00")

        # Get the latest value (last row with a value)
        latest_value = None
        for row in reversed(rows):
            value = row.get("Value") or row.get("value")
            if value not in (None, "", "NaN"):
                try:
                    latest_value = float(value)
                    break
                except (TypeError, ValueError):
                    continue

        if latest_value is None:
            raise RuntimeError("No valid value found in mvrv_usd_60d sheet")

        self.stdout.write(f"Latest value: {latest_value}")

        # Step 4: Append to CSV (create with header if doesn't exist)
        file_exists = os.path.exists(output_path)
        
        with open(output_path, "a") as f:
            if not file_exists:
                f.write("hour,value\n")
            f.write(f"{hour_str},{latest_value}\n")

        self.stdout.write(
            self.style.SUCCESS(f"Saved: {hour_str},{latest_value} -> {output_path}")
        )

    def _force_refresh_with_toggle(self, sheet_id: str) -> str:
        """
        Toggle between DATE() formula and string to force recalculation.
        This guarantees a cell change even if called multiple times per second.
        """
        client = get_write_client()
        spreadsheet = client.open_by_key(sheet_id)
        worksheet = spreadsheet.worksheet("control")
        
        # Read current value to decide what to write
        current = worksheet.acell("A1").value or ""
        today = datetime.datetime.utcnow().date()
        
        if current.startswith("="):
            # Currently a formula, write plain string
            new_value = today.strftime("%Y-%m-%d")
        else:
            # Currently a string, write DATE() formula
            new_value = f"=DATE({today.year},{today.month},{today.day})"
        
        worksheet.update_acell("A1", new_value)
        return new_value
