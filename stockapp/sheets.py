# -*- coding: utf-8 -*-
"""
stockapp.sheets
----------------
Tunn wrapper runt gspread för att läsa/skriva Google Sheet.
Fristående: ingen import från stockapp.utils – har egen _with_backoff.
"""

from __future__ import annotations
from typing import List, Optional
import time
import random

import streamlit as st
import pandas as pd

import gspread
from google.oauth2.service_account import Credentials

from .config import SHEET_URL, SHEET_NAME, MAX_ROWS_WRITE


# ---------------------------------------------------------------------
# Backoff med jitter (fristående)
# ---------------------------------------------------------------------
def _with_backoff(fn, *args, **kwargs):
    """Kör fn(*args, **kwargs) med exponentiell backoff & jitter."""
    delay = 0.8
    for attempt in range(6):  # 6 försök totalt
        try:
            return fn(*args, **kwargs)
        except Exception:
            if attempt >= 5:
                raise
            time.sleep(delay + random.random() * 0.25)
            delay *= 1.7


# ---------------------------------------------------------------------
# gspread-klient & blad
# ---------------------------------------------------------------------
def _gspread_client() -> gspread.Client:
    if not st.secrets.get("GOOGLE_CREDENTIALS"):
        raise RuntimeError("GOOGLE_CREDENTIALS saknas i st.secrets.")
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
    return gspread.authorize(creds)


def _spreadsheet() -> gspread.Spreadsheet:
    if not SHEET_URL:
        raise RuntimeError("SHEET_URL saknas (lägg in i st.secrets eller config).")
    client = _gspread_client()
    return _with_backoff(client.open_by_url, SHEET_URL)


def get_ws(name: Optional[str] = None) -> gspread.Worksheet:
    """
    Hämta (eller skapa) ett worksheet.
    """
    ss = _spreadsheet()
    title = name or SHEET_NAME or "Data"
    try:
        return _with_backoff(ss.worksheet, title)
    except Exception:
        # Skapa om det inte finns
        _with_backoff(ss.add_worksheet, title=title, rows=2000, cols=50)
        return _with_backoff(ss.worksheet, title)


# ---------------------------------------------------------------------
# Läs/skriv utilities
# ---------------------------------------------------------------------
def ensure_headers(ws: gspread.Worksheet, headers: List[str]) -> None:
    """
    Säkerställ att första raden innehåller headers (exakt ordning).
    Skapar/ersätter endast om raden saknas eller avviker i längd.
    """
    try:
        vals = _with_backoff(ws.get_values, "1:1") or []
        first = vals[0] if vals else []
    except Exception:
        first = []

    # Om tomt eller olika antal kolumner → skriv headers
    if not first or len([h for h in first if h is not None and str(h).strip() != ""]) != len(headers):
        body = [headers]
        _with_backoff(ws.clear)
        _with_backoff(ws.update, body)


def ws_read_df(
    ws: gspread.Worksheet,
    expected_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Läs hela bladet som DataFrame. Första raden = headers.
    Om bladet är tomt returneras tom DF (ev. med expected_cols).
    """
    values = _with_backoff(ws.get_all_values) or []
    if not values:
        return pd.DataFrame(columns=expected_cols or [])

    header = [str(h).strip() for h in (values[0] if values else [])]
    rows = values[1:] if len(values) > 1 else []

    if not header:
        # Fallback: använd expected_cols som header om möjligt
        header = list(expected_cols or [])

    # Normalisera längder
    width = len(header)
    norm_rows = []
    for r in rows:
        r2 = list(r)[:width]
        if len(r2) < width:
            r2 = r2 + [""] * (width - len(r2))
        norm_rows.append(r2)

    df = pd.DataFrame(norm_rows, columns=header)
    return df


def ws_write_df(
    ws: gspread.Worksheet,
    df: pd.DataFrame,
    enforce_headers: bool = True,
    expected_cols: Optional[List[str]] = None,
) -> None:
    """
    Skriv hela DF till bladet (overwrite):
    - Första rad = headers
    - Resterande = data som strängar
    """
    if df is None:
        df = pd.DataFrame()

    if expected_cols:
        # säkerställ kolumnernas ordning/kompletthet
        for c in expected_cols:
            if c not in df.columns:
                df[c] = ""
        df = df[expected_cols]

    headers = [str(c) for c in df.columns.tolist()]
    if enforce_headers:
        ensure_headers(ws, headers)

    # Konvertera till str för Sheets
    data_rows = [[("" if pd.isna(v) else str(v)) for v in row] for row in df.values.tolist()]
    body = [headers] + data_rows

    # Guard mot för stora skrivningar
    if MAX_ROWS_WRITE and len(body) > MAX_ROWS_WRITE:
        raise RuntimeError(f"För många rader ({len(body)} > MAX_ROWS_WRITE={MAX_ROWS_WRITE}).")

    _with_backoff(ws.clear)
    _with_backoff(ws.update, body)
