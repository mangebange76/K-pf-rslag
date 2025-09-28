# -*- coding: utf-8 -*-
from __future__ import annotations

import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

from .utils import with_backoff
from .config import FINAL_COLS, SHEET_NAME

# ---- Google Sheets-koppling --------------------------------------------------

SHEET_URL = st.secrets["SHEET_URL"]  # måste vara satt i Secrets
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

def _get_spreadsheet():
    return client.open_by_url(SHEET_URL)

def _worksheet():
    return _get_spreadsheet().worksheet(SHEET_NAME)

def hamta_data() -> pd.DataFrame:
    """
    Läser hela bladet. Returnerar tom DF med FINAL_COLS om bladet är tomt eller saknas.
    """
    try:
        ws = _worksheet()
        rows = with_backoff(ws.get_all_records)
        df = pd.DataFrame(rows)
        # Om helt tomt, returnera skeleton
        if df is None or df.empty:
            return pd.DataFrame({c: [] for c in FINAL_COLS})
        return df
    except Exception:
        # Skapa tom struktur om bladet saknas/inte nås
        return pd.DataFrame({c: [] for c in FINAL_COLS})

def _backup_snapshot_sheet(df: pd.DataFrame):
    """
    Skapar en snapshot-flik i samma Google Sheet: 'Snapshot-YYYYMMDD-HHMMSS'
    och fyller den med hela df. Kräver endast Sheets (inte Drive).
    """
    from datetime import datetime as _dt
    ss = _get_spreadsheet()
    snap_name = f"Snapshot-{_dt.now().strftime('%Y%m%d-%H%M%S')}"
    try:
        ss.add_worksheet(title=snap_name, rows=max(1000, len(df)+10), cols=max(50, len(df.columns)+2))
        ws = ss.worksheet(snap_name)
        with_backoff(ws.clear)
        body = [df.columns.values.tolist()] + df.astype(str).values.tolist()
        with_backoff(ws.update, body)
        st.sidebar.success(f"Snapshot skapad: {snap_name}")
    except Exception as e:
        st.sidebar.warning(f"Misslyckades skapa snapshot-flik: {e}")

def spara_data(df: pd.DataFrame, do_snapshot: bool = False):
    """
    Skriv hela DataFrame till huvudbladet. Optionellt: snapshot före skrivning.
    Säker: gör ingenting om df är tomt (för att undvika oavsiktlig wipe).
    """
    if df is None or df.empty:
        st.warning("Hoppar över skrivning: tom DataFrame (skydd mot wipe).")
        return
    try:
        if do_snapshot:
            try:
                _backup_snapshot_sheet(df)
            except Exception as e:
                st.warning(f"Kunde inte skapa snapshot före skrivning: {e}")
        ws = _worksheet()
        with_backoff(ws.clear)
        body = [df.columns.values.tolist()] + df.astype(str).values.tolist()
        with_backoff(ws.update, body)
    except Exception as e:
        st.error(f"Kunde inte spara till Google Sheets: {e}")
