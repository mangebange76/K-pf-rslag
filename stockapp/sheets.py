# stockapp/sheets.py
# -*- coding: utf-8 -*-
"""
Enkla hjälpare för Google Sheets:
- get_spreadsheet()  -> Spreadsheet-objektet
- get_ws(name, ...)  -> Worksheet (skapar om saknas)
- ensure_ws(name, header=...) -> Worksheet med ev. rubrikrad
"""

from __future__ import annotations
from typing import List, Optional

import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

from .config import SHEET_URL, SHEET_NAME  # SHEET_NAME används bara som default ibland
from .utils import with_backoff


# ---------------------------------------------------------------------
# Autentisering & klient
# ---------------------------------------------------------------------
def _gspread_client() -> gspread.Client:
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(
        st.secrets["GOOGLE_CREDENTIALS"],
        scopes=scope,
    )
    return gspread.authorize(creds)


def get_spreadsheet() -> gspread.Spreadsheet:
    """
    Öppnar kalkylarket via URL i secrets.
    """
    client = _gspread_client()
    return client.open_by_url(SHEET_URL if SHEET_URL else st.secrets["SHEET_URL"])


# ---------------------------------------------------------------------
# Worksheet-access
# ---------------------------------------------------------------------
def get_ws(name: str, create_if_missing: bool = True) -> gspread.Worksheet:
    """
    Hämtar ett worksheet. Skapar om det saknas (default).
    """
    ss = get_spreadsheet()
    try:
        return ss.worksheet(name)
    except Exception:
        if not create_if_missing:
            raise
        # Skapa enkelt blad om det saknas
        with_backoff(ss.add_worksheet, title=name, rows=1000, cols=50)
        return ss.worksheet(name)


def ensure_ws(
    name: str,
    header: Optional[List[str]] = None,
    rows: int = 1000,
    cols: int = 50,
) -> gspread.Worksheet:
    """
    Säkerställer att ett worksheet finns. Om header anges och bladet var nyss skapat/tomt,
    skrivs headern in som första rad.
    """
    ss = get_spreadsheet()

    # Finns redan?
    try:
        ws = ss.worksheet(name)
        # Om header efterfrågas men saknas (tomt blad), skriv den
        if header:
            try:
                existing = ws.get_all_values()
                if not existing:
                    with_backoff(ws.update, [header])
            except Exception:
                pass
        return ws
    except Exception:
        # Skapa
        with_backoff(ss.add_worksheet, title=name, rows=max(10, rows), cols=max(5, cols))
        ws = ss.worksheet(name)
        if header:
            with_backoff(ws.update, [header])
        return ws
