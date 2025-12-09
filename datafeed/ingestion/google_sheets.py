from dotenv import load_dotenv
from django.conf import settings
import os
from pathlib import Path
import datetime
import gspread
from google.oauth2.service_account import Credentials
from typing import Dict, Optional

BASE_DIR = settings.BASE_DIR

load_dotenv(".env")

SHEET_ID = os.getenv("GSPREAD_SHEET_ID")
DEFAULT_CREDS_PATH = BASE_DIR / "credentials" / "gsheets-service-account.json"
CREDS_FILE = Path(os.environ.get("GSPREAD_CREDS_FILE", str(DEFAULT_CREDS_PATH)))
SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]


def get_client():
    """
    Authorize a Google Sheets client using the service account JSON.
    """
    if not CREDS_FILE.exists():
        raise FileNotFoundError(
            f"Google Sheets credentials file not found: {CREDS_FILE}"
        )

    creds = Credentials.from_service_account_file(
        str(CREDS_FILE),
        scopes=SCOPES,
    )
    return gspread.authorize(creds)


def get_spreadsheet(sheet_id: Optional[str] = None):

    """
    Open the spreadsheet by ID.
    If sheet_id is None, read it from GSPREAD_SHEET_ID env var.
    """
    if sheet_id is None:
        sheet_id = os.environ.get("GSPREAD_SHEET_ID")

    if not sheet_id:
        raise RuntimeError(
            "No Google Sheet ID provided. "
            "Pass sheet_id explicitly or set GSPREAD_SHEET_ID in the environment."
        )

    client = get_client()
    return client.open_by_key(sheet_id)


def parse_date(date_str: str):
    """
    Parse dates in 'YYYY-MM-DD' format (e.g. '2016-01-01').
    """
    if not date_str:
        return None
    return datetime.datetime.strptime(date_str.strip(), "%Y-%m-%d").date()


def parse_float(value):
    """
    Convert sheet value to float or None.
    """
    if value in (None, "", "NaN"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_two_column_sheet(spreadsheet, sheet_name: str, date_col="Date", value_col="Value"):
    """
    For sheets with exactly two columns: Date, Value.
    Returns: dict[date -> float]
    """
    try:
        ws = spreadsheet.worksheet(sheet_name)
    except gspread.WorksheetNotFound:
        return {}

    rows = ws.get_all_records()  # Optional[str] = None
    result: Dict[datetime.date, Optional[float]] = {}

    for row in rows:
        date_str = row.get(date_col)
        if date_str is None:
            date_str = row.get(date_col.lower())

        value = row.get(value_col)
        if value is None:
            value = row.get(value_col.lower())

        if not date_str:
            continue

        dt = parse_date(date_str)
        if not dt:
            continue

        result[dt] = parse_float(value)

    return result
