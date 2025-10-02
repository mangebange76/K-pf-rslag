# -*- coding: utf-8 -*-
"""
Robusta Google Sheets-hjälpare:
- get_ws() öppnar rätt flik även om SHEET_NAME inte matchar (heuristik)
- ws_read_df(ws) läser värden utan att skriva något
- ws_write_df/ws_replace_all ersätter hela innehållet säkert
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import re

import pandas as pd
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

from .config import SHEET_URL, SHEET_NAME
from .utils import with_backoff


# ------------------------------------------------------------
# GSpread-klient
# ------------------------------------------------------------
def _client() -> gspread.Client:
    creds = Credentials.from_service_account_info(
        st.secrets["GOOGLE_CREDENTIALS"],
        scopes=[
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ],
    )
    return gspread.authorize(creds)


def _open_spreadsheet():
    if not SHEET_URL:
        raise RuntimeError("SHEET_URL saknas i secrets/config.")
    cli = _client()
    return with_backoff(cli.open_by_url, SHEET_URL)


# ------------------------------------------------------------
# Välj bästa fliken
# ------------------------------------------------------------
_HEADER_CANDIDATES = {"ticker", "bolagsnamn", "valuta", "sektor"}

def _header_score(first_row: List[str]) -> int:
    vals = [str(x).strip().lower() for x in first_row]
    s = 0
    for key in _HEADER_CANDIDATES:
        if key in vals:
            s += 1
    # extra poäng om "ticker" är första kolumnen
    if len(vals) > 0 and vals[0] == "ticker":
        s += 2
    return s


def _find_best_worksheet(ss) -> gspread.models.Worksheet:
    # 1) Försök exakt SHEET_NAME
    try:
        if SHEET_NAME:
            return with_backoff(ss.worksheet, SHEET_NAME)
    except Exception:
        pass

    # 2) Heuristik: välj fliken vars första rad matchar våra rubriker
    best_ws = None
    best_score = -1
    for ws in with_backoff(ss.worksheets):
        try:
            row1 = with_backoff(ws.row_values, 1) or []
        except Exception:
            row1 = []
        score = _header_score(row1)
        if score > best_score:
            best_score = score
            best_ws = ws

    if best_ws:
        if SHEET_NAME and best_ws.title != SHEET_NAME:
            st.info(f"ℹ️ Hittade ingen flik '{SHEET_NAME}'. Använder '{best_ws.title}'.")
        return best_ws

    # 3) Fallback: första fliken
    return with_backoff(ss.get_worksheet, 0)


def get_ws() -> gspread.models.Worksheet:
    ss = _open_spreadsheet()
    return _find_best_worksheet(ss)


# ------------------------------------------------------------
# Läs/skriv
# ------------------------------------------------------------
def ws_read_df(ws) -> pd.DataFrame:
    """Läs hela fliken till DataFrame (första raden = header)."""
    values = with_backoff(ws.get_all_values)
    if not values:
        return pd.DataFrame()
    header = [str(x).strip() for x in values[0]]
    rows = values[1:]
    # ta bort helt tomma rader
    rows = [r for r in rows if any(str(x).strip() for x in r)]
    df = pd.DataFrame(rows, columns=header) if rows else pd.DataFrame(columns=header)
    return df


def ws_replace_all(ws, df: pd.DataFrame):
    """Ersätter hela innehållet i fliken med df (inkl. header)."""
    if df is None:
        df = pd.DataFrame()
    # Gör strängar och fyll NaN
    out = df.copy()
    out = out.fillna("")
    body = [list(out.columns)] + out.astype(str).values.tolist()
    with_backoff(ws.clear)
    with_backoff(ws.update, body)


def ws_write_df(ws, df: pd.DataFrame):
    """Alias för ws_replace_all – kvar för bakåtkompabilitet."""
    ws_replace_all(ws, df)
