# -*- coding: utf-8 -*-
"""
stockapp.sheets
---------------
Tunn wrapper runt gspread för att läsa/skriva DataFrame med robusta
fallbacks: öppnar kalkylblad från URL i secrets, försöker hitta flik
på namn (annars första fliken). Bygger DataFrame från första raden
som headers och resten som rader.
"""

from __future__ import annotations
from typing import List, Optional

import pandas as pd
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

from .config import SHEET_URL, SHEET_NAME


# ---------------------------------------------------------------------
# gspread-klient
# ---------------------------------------------------------------------
def _gspread_client():
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
    return gspread.authorize(creds)


def _open_spreadsheet():
    if not SHEET_URL:
        raise RuntimeError("SHEET_URL saknas i secrets.")
    client = _gspread_client()
    return client.open_by_url(SHEET_URL)


# ---------------------------------------------------------------------
# Worksheet helpers
# ---------------------------------------------------------------------
def get_ws(name: Optional[str] = None):
    """
    Hämta worksheet med namn 'name'. Om det inte finns, fall back till första fliken.
    """
    ss = _open_spreadsheet()
    if not name:
        return ss.sheet1
    try:
        return ss.worksheet(name)
    except Exception:
        # fallback: första fliken
        return ss.sheet1


def ws_read_df(ws) -> pd.DataFrame:
    """
    Läs alla celler, första raden = headers, resten = data.
    Rader som är HELT tomma tas bort, annars behålls de.
    """
    values = ws.get_all_values() or []
    if not values:
        return pd.DataFrame()

    headers = [str(h).strip() for h in values[0]]
    rows = values[1:]

    # behåll alla rader som inte är helt tomma
    kept = [r for r in rows if any((str(c).strip() != "" for c in r))]

    # pad korta rader till samma längd som headers
    width = len(headers)
    norm_rows = [ (r + [""] * (width - len(r)))[:width] for r in kept ]

    df = pd.DataFrame(norm_rows, columns=headers)
    return df


def ws_write_df(ws, df: pd.DataFrame) -> None:
    """
    Skriv DataFrame till arket: headers på rad 1, sedan värden.
    """
    if df is None:
        return

    # konvertera till str för att undvika typfel i gspread
    out = df.copy()
    out = out.fillna("")

    # bygg 2D-array: headers + rows
    headers = list(out.columns)
    rows = out.astype(str).values.tolist()

    ws.clear()
    ws.update([headers] + rows)
