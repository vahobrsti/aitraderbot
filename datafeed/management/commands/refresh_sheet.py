# datafeed/management/commands/refresh_sheet.py
"""
Force Google Sheet refresh by writing today's UTC date to the control cell.

This triggers SANsheets recalculation without opening a browser.
Key properties:
- No formula rewriting
- No UI interaction
- Deterministic trigger
- Idempotent (writing today's date again is harmless)
"""

import os
from django.core.management.base import BaseCommand
from datafeed.ingestion.google_sheets import force_refresh_control_cell


class Command(BaseCommand):
    help = "Force Google Sheet refresh by writing today's UTC date to control cell"

    def add_arguments(self, parser):
        parser.add_argument(
            "--sheet-id",
            dest="sheet_id",
            default=None,
            help="Google Sheet ID (overrides GSPREAD_SHEET_ID env var if provided)",
        )
        parser.add_argument(
            "--control-sheet",
            dest="control_sheet",
            default="control",
            help="Name of the control sheet (default: control)",
        )
        parser.add_argument(
            "--control-cell",
            dest="control_cell",
            default="A1",
            help="Cell address to write to (default: A1)",
        )

    def handle(self, *args, **options):
        sheet_id = options["sheet_id"] or os.environ.get("GSPREAD_SHEET_ID")
        if not sheet_id:
            raise RuntimeError(
                "No sheet ID provided. Use --sheet-id or set GSPREAD_SHEET_ID in .env"
            )

        control_sheet = options["control_sheet"]
        control_cell = options["control_cell"]

        self.stdout.write(
            f"Forcing refresh: writing today's date to {control_sheet}!{control_cell}"
        )

        try:
            date_written = force_refresh_control_cell(
                sheet_id=sheet_id,
                control_sheet=control_sheet,
                control_cell=control_cell,
            )
            self.stdout.write(
                self.style.SUCCESS(
                    f"Sheet refresh triggered. Wrote '{date_written}' to {control_sheet}!{control_cell}"
                )
            )
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f"Failed to refresh sheet: {e}")
            )
            raise
