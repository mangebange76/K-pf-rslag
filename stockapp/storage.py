# stockapp/storage.py
# -*- coding: utf-8 -*-

from typing import List
import time
import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

# === Konfiguration via secrets ===
SHEET_URL: str = st.secrets["SHEET_URL"]
SHEET_NAME: str = st.secrets.get("SHEET_NAME", "Blad1")

_scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
_credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=_scope)
_client = gspread.authorize(_credentials)

def with_backoff(func, *args, **kwargs):
    """Liten backoff-hjälpare för att mildra 429/kvotfel (Sheets kvoter)."""
    delays = [0, 0.5, 1.0, 2.0, 3.5]
    last_err = None
    for d in delays:
        if d:
            time.sleep(d)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_err = e
    if last_err:
        raise last_err

def _get_spreadsheet():
    return _client.open_by_url(SHEET_URL)

def _worksheet():
    return _get_spreadsheet().worksheet(SHEET_NAME)

@st.cache_data(show_spinner=False)
def hamta_data() -> pd.DataFrame:
    """
    Läser alla rader från huvudbladet (SHEET_NAME) och returnerar som DataFrame.
    Cachas för att minska API-kostnad.
    """
    try:
        ws = _worksheet()
        rows: List[dict] = with_backoff(ws.get_all_records)
        if not rows:
            # Helt tomt eller endast header – returnera tom DF
            return pd.DataFrame()
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"Kunde inte läsa data från Google Sheets: {e}")
        return pd.DataFrame()

def spara_data(df: pd.DataFrame, do_snapshot: bool = False) -> None:
    """
    Skriv hela DataFrame till huvudbladet. Optionellt: skapa snapshot-flik först.
    """
    if df is None:
        return
    try:
        if do_snapshot:
            try:
                backup_snapshot_sheet(df, base_sheet_name=SHEET_NAME)
            except Exception as ee:
                st.warning(f"Kunde inte skapa snapshot före skrivning: {ee}")

        ws = _worksheet()
        # Töm bladet och skriv om allt (header + värden)
        with_backoff(ws.clear)
        values = [df.columns.tolist()] + df.astype(str).values.tolist()
        with_backoff(ws.update, values)
    except Exception as e:
        st.error(f"Kunde inte spara till Google Sheets: {e}")

def backup_snapshot_sheet(df: pd.DataFrame, base_sheet_name: str = SHEET_NAME) -> None:
    """
    Skapar en snapshot-flik i samma Google Sheet: 'Snapshot-YYYYMMDD-HHMMSS'
    och fyller den med hela df. Kräver endast Sheets (inte Drive).
    """
    try:
        ss = _get_spreadsheet()
        ts = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
        snap_name = f"Snapshot-{ts}"
        ss.add_worksheet(title=snap_name, rows=max(1000, len(df) + 10), cols=max(50, len(df.columns) + 2))
        ws = ss.worksheet(snap_name)
        with_backoff(ws.clear)
        values = [df.columns.tolist()] + df.astype(str).values.tolist()
        with_backoff(ws.update, values)
        st.success(f"Snapshot skapad: {snap_name}")
    except Exception as e:
        st.warning(f"Misslyckades skapa snapshot-flik: {e}")
