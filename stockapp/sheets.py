# -*- coding: utf-8 -*-
"""
Google Sheets-hjälpare:
- get_ws(title)  → worksheet (robust med fallback till flik som har rubriken 'Ticker')
- ensure_headers(ws, headers)
- ws_read_df(ws, header_row=1) -> DataFrame
- ws_write_df(ws, df, include_header=True, clear=True)
"""

from __future__ import annotations
from typing import List, Optional

import pandas as pd
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

from .config import SHEET_URL
from .utils import with_backoff


# ---------------------------------------------------------------------
# Autentisering / klient
# ---------------------------------------------------------------------
def _gspread_client():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
    return gspread.authorize(credentials)

def _open_spreadsheet():
    if not SHEET_URL:
        raise RuntimeError("SHEET_URL saknas i secrets.")
    client = _gspread_client()
    return client.open_by_url(SHEET_URL)

# ---------------------------------------------------------------------
# Worksheet-funktioner
# ---------------------------------------------------------------------
def _has_ticker_header(ws) -> bool:
    try:
        header = with_backoff(ws.row_values, 1) or []
        return any(str(h).strip().lower() == "ticker" for h in header)
    except Exception:
        return False

def get_ws(title: str):
    """Öppna flik med angivet title. Fallback: första flik som har rubriken 'Ticker' i rad 1."""
    ss = _open_spreadsheet()

    # 1) Försök exakt titel
    try:
        return ss.worksheet(title)
    except Exception:
        pass

    # 2) Fallback: leta flik med 'Ticker' i första raden
    try:
        for ws in ss.worksheets():
            if _has_ticker_header(ws):
                return ws
    except Exception:
        pass

    # 3) Sista utvägen: skapa fliken
    ws = ss.add_worksheet(title=title, rows=2000, cols=100)
    return ws

def ensure_headers(ws, headers: List[str]) -> None:
    """Säkerställ att rad 1 innehåller önskade headers (ersätter om det skiljer)."""
    cur = with_backoff(ws.row_values, 1) or []
    # jämför kort — om identiskt, gör inget
    if len(cur) >= len(headers) and all(str(cur[i]).strip() == str(headers[i]).strip() for i in range(len(headers))):
        return
    body = [headers]
    with_backoff(ws.clear)
    with_backoff(ws.update, body)

def ws_read_df(ws, header_row: int = 1) -> pd.DataFrame:
    """Läs hela fliken som DataFrame. Antar att header ligger på `header_row`."""
    values: List[List[str]] = with_backoff(ws.get_all_values) or []
    if not values:
        return pd.DataFrame()

    # normalisera längd per rad (till max kolumnantal)
    max_len = max(len(r) for r in values) if values else 0
    values = [r + [""] * (max_len - len(r)) for r in values]

    # hämta header
    if header_row < 1 or header_row > len(values):
        header_row = 1
    header = [str(x).strip() for x in values[header_row - 1]]

    # data efter header
    rows = values[header_row:]
    # drop helt tomma rader
    rows = [r for r in rows if any(str(c).strip() != "" for c in r)]

    df = pd.DataFrame(rows, columns=header)
    return df

def ws_write_df(ws, df: pd.DataFrame, include_header: bool = True, clear: bool = True) -> None:
    """Skriv en DataFrame till fliken (ersätter befintligt innehåll)."""
    if df is None:
        return
    data = []
    if include_header:
        data.append(list(df.columns))
    for _, r in df.iterrows():
        data.append([None if pd.isna(v) else v for v in r.tolist()])

    if clear:
        with_backoff(ws.clear)
    with_backoff(ws.update, data)
