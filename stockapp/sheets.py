# stockapp/sheets.py
# -*- coding: utf-8 -*-
"""
Enkla hjälpare för Google Sheets:
- get_spreadsheet()           -> Spreadsheet-objektet (via st.secrets["SHEET_URL"])
- get_ws(name, create=True)   -> Worksheet med angivet namn (skapar om saknas)
- ensure_ws(name, header=...) -> Samma som get_ws men kan skriva rubrikrad om bladet är tomt

Krav i st.secrets:
- GOOGLE_CREDENTIALS : service account JSON (dict)
- SHEET_URL          : fullständig URL till ditt Google Sheet
"""

from __future__ import annotations
from typing import List, Optional

import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

from .utils import with_backoff


# ---------------------------------------------------------------------
# Autentisering & klient
# ---------------------------------------------------------------------
def _gspread_client() -> gspread.Client:
    creds_info = st.secrets.get("GOOGLE_CREDENTIALS")
    if not creds_info:
        raise RuntimeError("GOOGLE_CREDENTIALS saknas i st.secrets.")
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(creds_info, scopes=scope)
    return gspread.authorize(creds)


def get_spreadsheet() -> gspread.Spreadsheet:
    """
    Öppnar kalkylarket via URL i st.secrets["SHEET_URL"].
    """
    url = st.secrets.get("SHEET_URL")
    if not url:
        raise RuntimeError("SHEET_URL saknas i st.secrets.")
    client = _gspread_client()
    return client.open_by_url(url)


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
        with_backoff(ss.add_worksheet, title=name, rows=1000, cols=50)
        return ss.worksheet(name)


def ensure_ws(
    name: str,
    header: Optional[List[str]] = None,
    rows: int = 1000,
    cols: int = 50,
) -> gspread.Worksheet:
    """
    Säkerställer att ett worksheet finns. Om header anges och bladet är tomt,
    skrivs headern in som första rad.
    """
    ss = get_spreadsheet()
    try:
        ws = ss.worksheet(name)
        if header:
            try:
                existing = ws.get_all_values()
                if not existing:
                    with_backoff(ws.update, [header])
            except Exception:
                pass
        return ws
    except Exception:
        with_backoff(ss.add_worksheet, title=name, rows=max(10, rows), cols=max(5, cols))
        ws = ss.worksheet(name)
        if header:
            with_backoff(ws.update, [header])
        return ws
