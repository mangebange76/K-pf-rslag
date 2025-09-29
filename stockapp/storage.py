# stockapp/storage.py
# -*- coding: utf-8 -*-
"""
Google Sheets-lagring:
- hamta_data()        -> DataFrame med säkerställt schema
- spara_data(df, ...) -> skriver hela DF till huvudbladet
- backup_snapshot_sheet(df, ...) -> skapar snapshot-flik
- get_spreadsheet() / skapa_koppling() -> interna hjälpmetoder

Krav i st.secrets:
- SHEET_URL
- GOOGLE_CREDENTIALS (service account JSON)
"""

from __future__ import annotations
from typing import List, Tuple

import pandas as pd
import numpy as np
import streamlit as st

# Google Sheets
import gspread
from google.oauth2.service_account import Credentials

from .config import FINAL_COLS, SHEET_NAME
from .utils import ensure_schema, with_backoff, now_stamp, dedupe_tickers


# ---------------------------------------------------------------------
# GSpread-klient
# ---------------------------------------------------------------------
def _gspread_client():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
    return gspread.authorize(credentials)

def get_spreadsheet():
    client = _gspread_client()
    return client.open_by_url(st.secrets["SHEET_URL"])

def skapa_koppling(sheet_name: str = SHEET_NAME):
    return get_spreadsheet().worksheet(sheet_name)


# ---------------------------------------------------------------------
# Läs / Spara / Snapshot
# ---------------------------------------------------------------------
def hamta_data() -> pd.DataFrame:
    """
    Läser alla rader från huvudbladet och säkerställer schema/typer.
    Returnerar alltid en DF med FINAL_COLS (saknade fylls).
    """
    try:
        sheet = skapa_koppling()
        rows = with_backoff(sheet.get_all_records)  # List[dict]
        df = pd.DataFrame(rows)
    except Exception:
        # Om bladet saknas eller är tomt – bygg minimal DF
        df = pd.DataFrame({c: [] for c in FINAL_COLS})

    # Säkerställ schema + typer
    df = ensure_schema(df)

    # Dedupe tickers i läs-ögonblicket (vi skriver inte tillbaka automatiskt här)
    df, _dups = dedupe_tickers(df)
    return df


def _safe_cell_value(v) -> str:
    """
    Konverterar cellvärde till str för gspread.update.
    - NaN/NaT -> ""
    - float -> str(v)
    - annars str(v)
    """
    if v is None:
        return ""
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return ""
    s = str(v)
    if s.lower() in ("nan", "nat", "none"):
        return ""
    return s


def spara_data(df: pd.DataFrame, do_snapshot: bool = False) -> None:
    """
    Skriver HELA DataFrame till huvudbladet.
    Säkerhet:
      - Avbryter om df är None/tom eller saknar 'Ticker'.
      - Skapar snapshot-flik först om do_snapshot=True.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        st.error("spara_data: DF saknas – ingen skrivning.")
        return
    if "Ticker" not in df.columns:
        st.error("spara_data: 'Ticker'-kolumn saknas – ingen skrivning.")
        return
    if df.empty:
        st.error("spara_data: DF är tom – ingen skrivning (skydd mot wipe).")
        return

    # Säkerställ schema och ordning på kolumner
    df = ensure_schema(df)
    df = df[[c for c in FINAL_COLS if c in df.columns]].copy()

    # Ta bort dubbletter (behåll första)
    df, dups = dedupe_tickers(df)
    if dups:
        st.warning(f"Dubbletter borttagna vid sparning: {sorted(set(dups))}")

    # Snapshot före skrivning om begärt
    if do_snapshot:
        try:
            backup_snapshot_sheet(df, base_sheet_name=SHEET_NAME)
        except Exception as e:
            st.warning(f"Kunde inte skapa snapshot före skrivning: {e}")

    # Skriv
    sheet = skapa_koppling()
    with_backoff(sheet.clear)

    header = list(df.columns)
    values = df.astype(object).where(pd.notna(df), None).values.tolist()
    # Konvertera till str för gspread
    rows_str = [[_safe_cell_value(x) for x in row] for row in values]

    with_backoff(sheet.update, [header] + rows_str)
    st.success(f"Sparat {len(df)} rader till '{SHEET_NAME}' ({now_stamp()}).")


def backup_snapshot_sheet(df: pd.DataFrame, base_sheet_name: str = SHEET_NAME) -> None:
    """
    Skapar snapshot-flik 'Snapshot-YYYYMMDD' och fyller med DF.
    """
    ss = get_spreadsheet()
    snap_name = f"Snapshot-{now_stamp()}"
    try:
        ss.add_worksheet(title=snap_name, rows=max(1000, len(df) + 10), cols=max(50, len(df.columns) + 2))
        ws = ss.worksheet(snap_name)
        with_backoff(ws.clear)
        header = list(df.columns)
        values = df.astype(object).where(pd.notna(df), None).values.tolist()
        rows_str = [[_safe_cell_value(x) for x in row] for row in values]
        with_backoff(ws.update, [header] + rows_str)
        st.success(f"Snapshot skapad: {snap_name}")
    except Exception as e:
        st.warning(f"Misslyckades skapa snapshot-flik: {e}")
