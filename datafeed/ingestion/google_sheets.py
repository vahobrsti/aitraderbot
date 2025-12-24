from dotenv import load_dotenv
from django.conf import settings
import os
from pathlib import Path
import datetime
import json
import gspread
from google.oauth2.service_account import Credentials
from typing import Dict, Optional

BASE_DIR = settings.BASE_DIR

load_dotenv(".env")

SHEET_ID = os.getenv("GSPREAD_SHEET_ID")

# For production: set GSPREAD_CREDS_JSON with the full JSON content
# For local dev: use GSPREAD_CREDS_FILE or the default path
DEFAULT_CREDS_PATH = BASE_DIR / "credentials" / "gsheets-service-account.json"
CREDS_FILE = Path(os.environ.get("GSPREAD_CREDS_FILE", str(DEFAULT_CREDS_PATH)))

# Scopes for readonly access (data ingestion)
SCOPES_READONLY = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
# Scopes for read/write access (refresh control cell)
SCOPES_READWRITE = ["https://www.googleapis.com/auth/spreadsheets"]

# Keep SCOPES for backwards compatibility
SCOPES = SCOPES_READONLY


def _get_credentials(scopes: list[str]) -> Credentials:
    """
    Load Google credentials from environment or file.
    
    Priority:
    1. GSPREAD_CREDS_JSON env var (JSON string) - for production/cloud
    2. GSPREAD_CREDS_FILE env var or default path - for local development
    """
    # Option 1: JSON content in environment variable (production)
    creds_json = os.environ.get("GSPREAD_CREDS_JSON")
    if creds_json:
        try:
            creds_info = json.loads(creds_json)
            return Credentials.from_service_account_info(creds_info, scopes=scopes)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in GSPREAD_CREDS_JSON: {e}")
    
    # Option 2: File path (local development)
    if not CREDS_FILE.exists():
        raise FileNotFoundError(
            f"Google Sheets credentials not found. Either:\n"
            f"  1. Set GSPREAD_CREDS_JSON env var with JSON content (production), or\n"
            f"  2. Place credentials file at: {CREDS_FILE}"
        )
    
    return Credentials.from_service_account_file(str(CREDS_FILE), scopes=scopes)


def get_client(readonly: bool = True):
    """
    Authorize a Google Sheets client using service account credentials.
    
    Credentials are loaded from:
    1. GSPREAD_CREDS_JSON env var (JSON string) - for production/cloud
    2. GSPREAD_CREDS_FILE env var or default path - for local development
    
    Args:
        readonly: If True, use readonly scope. If False, use read/write scope.
    """
    scopes = SCOPES_READONLY if readonly else SCOPES_READWRITE
    creds = _get_credentials(scopes)
    return gspread.authorize(creds)


def get_write_client():
    """
    Get a Google Sheets client with write permissions.
    Used for forcing sheet refresh by writing to control cell.
    """
    return get_client(readonly=False)


def force_refresh_control_cell(
    sheet_id: str,
    control_sheet: str = "control",
    control_cell: str = "A1",
) -> str:
    """
    Force the Google Sheet to recalculate by writing today's UTC date
    to the control cell.
    
    Args:
        sheet_id: The Google Sheet ID
        control_sheet: Name of the control sheet (default: "control")
        control_cell: Cell address to write to (default: "A1")
    
    Returns:
        The date string that was written (YYYY-MM-DD format)
    
    This creates a real edit which forces recalculation of any formulas
    that depend on this cell.
    """
    client = get_write_client()
    spreadsheet = client.open_by_key(sheet_id)
    worksheet = spreadsheet.worksheet(control_sheet)
    
    today_utc = datetime.datetime.utcnow().date().strftime("%Y-%m-%d")
    worksheet.update_acell(control_cell, today_utc)
    
    return today_utc


def verify_sheet_freshness(
    sheet_id: str,
    control_sheet: str = "control",
    control_cell: str = "A1",
) -> tuple[bool, str, str]:
    """
    Verify that the sheet is fresh by checking the control cell value.
    
    Args:
        sheet_id: The Google Sheet ID
        control_sheet: Name of the control sheet (default: "control")
        control_cell: Cell address to read (default: "A1")
    
    Returns:
        Tuple of (is_fresh, cell_value, expected_value)
        - is_fresh: True if the control cell contains today's UTC date
        - cell_value: The actual value in the control cell
        - expected_value: Today's UTC date (what we expected)
    """
    client = get_client(readonly=True)
    spreadsheet = client.open_by_key(sheet_id)
    worksheet = spreadsheet.worksheet(control_sheet)
    
    cell_value = worksheet.acell(control_cell).value or ""
    today_utc = datetime.datetime.utcnow().date().strftime("%Y-%m-%d")
    
    is_fresh = cell_value.strip() == today_utc
    return is_fresh, cell_value.strip(), today_utc


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
