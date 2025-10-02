# -*- coding: utf-8 -*-
"""
Robust Google Sheets-åtkomst:
- get_spreadsheet()
- get_ws(preferred_name=None)  -> worksheet med smart fallback
- ws_read_df(ws)               -> DataFrame (header autodetekteras)
- ws_write_df(ws, df)          -> skriver tabellen
- ensure_headers(ws, headers)  -> säkerställer rubrikrad

Den här modulen har inga beroenden på övriga stockapp-moduler
förutom utils.with_backoff (ofarligt, ingen cirkel).
"""

from __future__ import annotations
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import gspread
from google.oauth2.service_account import Credentials

from .utils import with_backoff


# ------------------------------------------------------------
# GSpread klient
# ------------------------------------------------------------
def _gspread_client():
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds_info = st.secrets.get("GOOGLE_CREDENTIALS", None)
    if not creds_info:
        raise RuntimeError("Saknar GOOGLE_CREDENTIALS i secrets.")
    credentials = Credentials.from_service_account_info(creds_info, scopes=scope)
    return gspread.authorize(credentials)


def get_spreadsheet():
    url = st.secrets.get("SHEET_URL", "").strip()
    if not url:
        raise RuntimeError("SHEET_URL saknas i secrets.")
    client = _gspread_client()
    return with_backoff(client.open_by_url, url)


# ------------------------------------------------------------
# Worksheet-val med fallback
# ------------------------------------------------------------
def _has_ticker_header(values: List[List[str]]) -> bool:
    """Returnerar True om någon rad innehåller en cell som (case-insensitive) == 'ticker'."""
    for row in values[:10]:  # kolla bara de första ~10 raderna
        for cell in row:
            if str(cell).strip().lower() == "ticker":
                return True
    return False


def get_ws(preferred_name: Optional[str] = None):
    """
    Försök öppna fliken med 'preferred_name'. Om den inte finns:
    - Välj första fliken som har en rubrikrad där 'Ticker' förekommer.
    - Som sista utväg: ta workbook.sheet1.
    Skriver en liten info-text i Streamlit om fallback används.
    """
    ss = get_spreadsheet()
    # 1) försök preferred
    if preferred_name:
        try:
            return with_backoff(ss.worksheet, preferred_name)
        except Exception:
            pass

    # 2) leta flik med 'Ticker' i header
    try:
        for ws in ss.worksheets():
            vals = with_backoff(ws.get_all_values)
            if _has_ticker_header(vals):
                st.caption(f"🔎 Läser från flik: **{ws.title}** (fallback)")
                return ws
    except Exception:
        pass

    # 3) sheet1 som sista fallback
    ws = ss.sheet1
    st.caption(f"🔎 Läser från flik: **{ws.title}** (sheet1 fallback)")
    return ws


# ------------------------------------------------------------
# Läs/skriv
# ------------------------------------------------------------
def _find_header_row(values: List[List[str]]) -> int:
    """
    Hitta index (0-baserat) för rubrikrad.
    Heuristik: första rad som innehåller 'Ticker' (case-insensitive),
    annars 0 om det finns några värden, annars -1.
    """
    for i, row in enumerate(values[:20]):
        row_stripped = [str(c).strip().lower() for c in row]
        if "ticker" in row_stripped:
            return i
    return 0 if values else -1


def ws_read_df(ws) -> pd.DataFrame:
    """
    Läser hela arket till en DataFrame:
    - autodetekterar rubrikraden
    - trimmar whitespace i rubriker & värden
    - ersätter tomma strängar med NaN
    - droppar rader utan 'Ticker'
    """
    values: List[List[str]] = with_backoff(ws.get_all_values) or []
    if not values:
        return pd.DataFrame()

    hidx = _find_header_row(values)
    if hidx < 0 or hidx >= len(values):
        return pd.DataFrame()

    headers = [str(c).strip() for c in values[hidx]]
    body = values[hidx + 1 :]

    # klipp bort helt tomma rader i slutet
    body = [row for row in body if any(str(c).strip() for c in row)]

    # gör raderna lika långa som headers
    width = len(headers)
    norm_rows = []
    for r in body:
        if len(r) < width:
            r = r + [""] * (width - len(r))
        elif len(r) > width:
            r = r[:width]
        norm_rows.append([str(c).strip() for c in r])

    df = pd.DataFrame(norm_rows, columns=headers)

    # tomma -> NaN
    df = df.replace({"": np.nan, "nan": np.nan, "NaN": np.nan})

    # droppa rader utan ticker
    if "Ticker" in df.columns:
        df["__tkr__"] = df["Ticker"].astype(str).str.strip()
        df = df[df["__tkr__"] != ""].drop(columns=["__tkr__"])
    return df.reset_index(drop=True)


def ws_write_df(ws, df: pd.DataFrame) -> None:
    """
    Skriver rubriker + värden till ws (ersätter befintligt innehåll).
    """
    if df is None or df.empty:
        with_backoff(ws.clear)
        return
    # konvertera NaN -> tom sträng
    out = df.copy()
    out = out.fillna("")
    values = [list(out.columns)] + out.astype(str).values.tolist()
    with_backoff(ws.clear)
    with_backoff(ws.update, values)


def ensure_headers(ws, headers: List[str]) -> None:
    """
    Om arket är tomt (eller topp-raden inte ser ut som rubriker), skriv rubriker.
    Är rubrikerna redan där görs inget.
    """
    vals = with_backoff(ws.get_all_values)
    if not vals:
        with_backoff(ws.update, [headers])
        return

    first = [str(c).strip() for c in vals[0]]
    # om första rad inte innehåller 'Ticker' → betraktas inte som korrekt header
    if "Ticker" not in first:
        with_backoff(ws.clear)
        with_backoff(ws.update, [headers])
