# stockapp/sheets.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Iterable, List, Optional

import pandas as pd
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

from .config import SHEET_URL, SHEET_NAME, RATES_SHEET_NAME
from .utils import with_backoff


# ------------------------------------------------------------
# GSpread-klient & Spreadsheet (cache: en gång per session)
# ------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _gspread_client() -> gspread.Client:
    """Auktorisera gspread-klient från Streamlit secrets."""
    creds_dict = st.secrets.get("GOOGLE_CREDENTIALS")
    if not creds_dict:
        raise RuntimeError("Saknar GOOGLE_CREDENTIALS i st.secrets.")
    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
    return gspread.authorize(creds)


@st.cache_resource(show_spinner=False)
def get_spreadsheet() -> gspread.Spreadsheet:
    """Öppna Spreadsheet via URL (cache: en gång per session)."""
    if not SHEET_URL:
        raise RuntimeError("SHEET_URL saknas i config/secrets.")
    client = _gspread_client()
    return client.open_by_url(SHEET_URL)


# ------------------------------------------------------------
# Arbetsbladshjälpare
# ------------------------------------------------------------
def get_ws(title: Optional[str] = None, rows: int = 1000, cols: int = 60) -> gspread.Worksheet:
    """
    Hämta arbetsblad med givet namn (skapa om saknas).
    Defaulttitel = SHEET_NAME.
    """
    ss = get_spreadsheet()
    title = title or SHEET_NAME
    try:
        return ss.worksheet(title)
    except Exception:
        # skapa nytt blad om det saknas
        with_backoff(ss.add_worksheet, title=title, rows=rows, cols=cols)
        ws = ss.worksheet(title)
        return ws


def get_rates_ws(rows: int = 200, cols: int = 10) -> gspread.Worksheet:
    """Arbetsbladet för valutakurser (skapas om saknas)."""
    return get_ws(RATES_SHEET_NAME, rows=rows, cols=cols)


def list_worksheets() -> List[str]:
    """Lista alla bladnamn (bra för felsökning)."""
    ss = get_spreadsheet()
    return [ws.title for ws in ss.worksheets()]


# ------------------------------------------------------------
# Läs/skriv-utility för DataFrame
# ------------------------------------------------------------
def ws_read_df(ws: gspread.Worksheet) -> pd.DataFrame:
    """
    Läs hela bladet som DataFrame via get_all_records().
    Tomt blad -> tom DataFrame.
    """
    try:
        records = with_backoff(ws.get_all_records)
        if not records:
            # kontrollera om det finns header-rad men utan data
            header = with_backoff(ws.row_values, 1)
            return pd.DataFrame(columns=header) if header else pd.DataFrame()
        return pd.DataFrame.from_records(records)
    except Exception as e:
        raise RuntimeError(f"Misslyckades läsa blad '{ws.title}': {e}") from e


def ws_write_df(ws: gspread.Worksheet, df: pd.DataFrame) -> None:
    """
    Skriv en DataFrame till bladet:
    - Rensar bladet
    - Skriver header + värden
    """
    try:
        # Konvertera NaN -> tom sträng för ett stabilt 2D-array
        body = [list(df.columns)]
        if not df.empty:
            values = df.astype(object).where(pd.notna(df), "").values.tolist()
            body.extend(values)

        with_backoff(ws.clear)
        if body:
            with_backoff(ws.update, body)
    except Exception as e:
        raise RuntimeError(f"Misslyckades skriva till blad '{ws.title}': {e}") from e


def ws_write_rows(ws: gspread.Worksheet, rows: Iterable[Iterable]) -> None:
    """
    Skriv godtycklig 2D-array (inkl. header i första raden).
    Rensar bladet innan skrivning.
    """
    matrix = [list(r) for r in rows]
    with_backoff(ws.clear)
    if matrix:
        with_backoff(ws.update, matrix)


def ensure_headers(ws: gspread.Worksheet, headers: List[str]) -> None:
    """
    Säkerställ att första raden exakt matchar headers.
    Om bladet är tomt, skriv bara headers.
    Om annan header redan finns, skrivs de över.
    """
    try:
        first_row = with_backoff(ws.row_values, 1)
    except Exception:
        first_row = []
    if first_row != headers:
        with_backoff(ws.clear)
        with_backoff(ws.update, [headers])
