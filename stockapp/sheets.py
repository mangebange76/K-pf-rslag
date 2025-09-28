# stockapp/sheets.py
# -*- coding: utf-8 -*-
"""
Google Sheets-koppling & util-funktioner för hela appen.

- get_gspread_client()                  -> autentisering via st.secrets["GOOGLE_CREDENTIALS"]
- get_spreadsheet()                     -> öppnar via st.secrets["SHEET_URL"] eller config.SHEET_URL
- get_ws(title, create=False, rows=2_000, cols=200)
- read_table(title) -> DataFrame
- write_table(title, df, header=True, clear=True, order_cols=None)
- read_portfolio_df()                   -> läser huvudarket (SHEET_NAME) och garanterar FINAL_COLS
- save_portfolio_df(df, do_snapshot=False, snapshot_title=None)
- ensure_header(ws, header)

Notera: Modulen importerar inte 'storage' (cirkulär risk). Andra moduler bör använda
read_portfolio_df/save_portfolio_df i stället för att tala direkt med gspread.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import streamlit as st
import pandas as pd
import numpy as np

import gspread
from google.oauth2.service_account import Credentials

from .config import (
    SHEET_URL as CFG_SHEET_URL,
    SHEET_NAME,
    FINAL_COLS,
)
from .utils import (
    with_backoff,
    ensure_final_cols,
    norm_ticker,
    AppUserError,
)


# ---------------------------------------------------------------------
# Autentisering & grundkoppling
# ---------------------------------------------------------------------
def get_gspread_client() -> gspread.Client:
    """Skapar en gspread-klient från service account i st.secrets."""
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    try:
        creds = Credentials.from_service_account_info(
            st.secrets["GOOGLE_CREDENTIALS"], scopes=scope
        )
    except KeyError as exc:
        raise AppUserError("Saknar GOOGLE_CREDENTIALS i st.secrets.") from exc
    return gspread.authorize(creds)


def get_spreadsheet() -> gspread.Spreadsheet:
    """Öppnar kalkylarket via URL i secrets (fallback config)."""
    client = get_gspread_client()
    sheet_url = st.secrets.get("SHEET_URL", CFG_SHEET_URL)
    if not sheet_url:
        raise AppUserError("SHEET_URL saknas (i st.secrets eller config).")
    return with_backoff(client.open_by_url, sheet_url)


def get_ws(title: str, create: bool = False, rows: int = 2000, cols: int = 200) -> gspread.Worksheet:
    """Hämtar ett arbetsblad. Skapar vid behov om create=True."""
    ss = get_spreadsheet()
    try:
        return with_backoff(ss.worksheet, title)
    except Exception:
        if not create:
            raise
        # skapa nytt ark
        with_backoff(ss.add_worksheet, title=title, rows=rows, cols=cols)
        return with_backoff(ss.worksheet, title)


# ---------------------------------------------------------------------
# Hjälp för tabell I/O
# ---------------------------------------------------------------------
def ensure_header(ws: gspread.Worksheet, header: Sequence[str]) -> None:
    """Säkerställ att översta raden motsvarar headern."""
    # Rensa inte här – skriv bara in rubrikerna.
    with_backoff(ws.update, [list(header)], **{"range_name": "A1"})


def _values_from_df(df: pd.DataFrame, order_cols: Optional[Sequence[str]] = None) -> List[List[Any]]:
    """Konvertera DataFrame till 2D-array (inkl. rubriker i första raden)."""
    work = df.copy()
    if order_cols:
        # lägg till saknade kolumner som tomma
        for c in order_cols:
            if c not in work.columns:
                work[c] = np.nan
        work = work[list(order_cols)]
    else:
        # behåll befintlig ordning
        pass

    # Gör om NaN till tom sträng för Sheets
    work = work.replace({np.nan: ""})
    vals: List[List[Any]] = [list(work.columns)]
    vals.extend(work.astype(object).values.tolist())
    return vals


def read_table(title: str) -> pd.DataFrame:
    """Läser ett arbetsblad som tabell (rubriker i första raden)."""
    ws = get_ws(title, create=True)
    records: List[Dict[str, Any]] = with_backoff(ws.get_all_records)
    df = pd.DataFrame.from_records(records)
    return df


def write_table(
    title: str,
    df: pd.DataFrame,
    header: bool = True,
    clear: bool = True,
    order_cols: Optional[Sequence[str]] = None,
) -> None:
    """
    Skriver en hel tabell till ett arbetsblad.
    - header=True => första raden är rubriker
    - clear=True  => ws.clear() före skrivning
    - order_cols  => kolumnordning; saknade kolumner fylls ut tomma
    """
    ws = get_ws(title, create=True)
    if clear:
        with_backoff(ws.clear)
    values = _values_from_df(df, order_cols=order_cols)
    with_backoff(ws.update, values)


# ---------------------------------------------------------------------
# Portfölj I/O (huvudark)
# ---------------------------------------------------------------------
def read_portfolio_df() -> pd.DataFrame:
    """
    Läser huvudarket (SHEET_NAME) och ser till att:
    - DataFrame alltid innehåller alla FINAL_COLS
    - Ticker normaliseras (upper/trim)
    """
    df = read_table(SHEET_NAME)
    if df.empty:
        # init tom tabell med FINAL_COLS
        df = pd.DataFrame(columns=list(FINAL_COLS))
    # säkerställ alla slutkolumner och defaults
    df = ensure_final_cols(df)
    if "Ticker" in df.columns:
        df["Ticker"] = df["Ticker"].map(norm_ticker)
    return df


def save_portfolio_df(
    df: pd.DataFrame,
    do_snapshot: bool = False,
    snapshot_title: Optional[str] = None,
) -> None:
    """
    Sparar hela portföljtabellen:
    - Default kolumnordning: FINAL_COLS först, sedan ev. extra kolumner (i befintlig ordning).
    - do_snapshot=True skriver även en snapshot till eget blad (timestamp i namn).
    """
    if "Ticker" in df.columns:
        df["Ticker"] = df["Ticker"].map(norm_ticker)

    # Ordna kolumner: FINAL_COLS först
    cols: List[str] = list(FINAL_COLS)
    for c in df.columns:
        if c not in cols:
            cols.append(c)

    # snapshot?
    if do_snapshot:
        if not snapshot_title:
            # t.ex. "snapshot_2025-09-28T20:21:00Z"
            from .utils import ts_str
            snapshot_title = f"snapshot_{ts_str()}"
        write_table(snapshot_title, df[cols], header=True, clear=True, order_cols=cols)

    # skriv huvudarket
    write_table(SHEET_NAME, df[cols], header=True, clear=True, order_cols=cols)
