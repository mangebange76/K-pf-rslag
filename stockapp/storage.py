# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

from .utils import with_backoff
from .config import FINAL_COLS, STANDARD_VALUTAKURSER

SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"
RATES_SHEET_NAME = "Valutakurser"

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

def _book():
    return client.open_by_url(SHEET_URL)

def _ensure_rates_sheet():
    ss = _book()
    try:
        return ss.worksheet(RATES_SHEET_NAME)
    except Exception:
        ss.add_worksheet(title=RATES_SHEET_NAME, rows=10, cols=5)
        ws = ss.worksheet(RATES_SHEET_NAME)
        with_backoff(ws.update, [["Valuta","Kurs"]])
        return ws

def read_dataframe() -> pd.DataFrame:
    ws = _book().worksheet(SHEET_NAME)
    data = with_backoff(ws.get_all_records)
    df = pd.DataFrame(data)
    if df.empty:
        df = pd.DataFrame({c: [] for c in FINAL_COLS})
    return df

def write_dataframe(df: pd.DataFrame):
    ws = _book().worksheet(SHEET_NAME)
    with_backoff(ws.clear)
    with_backoff(ws.update, [df.columns.values.tolist()] + df.astype(str).values.tolist())

def backup_snapshot(df: pd.DataFrame):
    try:
        ss = _book()
        name = "Snapshot"
        # enkel lösning: ersätt/skriv en Snapshot-flik (istället för att skapa många nya)
        try:
            ws = ss.worksheet(name)
        except Exception:
            ss.add_worksheet(title=name, rows=max(1000, len(df)+10), cols=max(50, len(df.columns)+2))
            ws = ss.worksheet(name)
        with_backoff(ws.clear)
        with_backoff(ws.update, [df.columns.values.tolist()] + df.astype(str).values.tolist())
    except Exception as e:
        st.warning(f"Kunde inte skriva snapshot: {e}")

@st.cache_data(show_spinner=False)
def las_sparade_valutakurser_cached(nonce: int):
    ws = _ensure_rates_sheet()
    rows = with_backoff(ws.get_all_records)
    out = {}
    for r in rows:
        cur = str(r.get("Valuta", "")).upper().strip()
        val = str(r.get("Kurs", "")).replace(",", ".").strip()
        try:
            out[cur] = float(val)
        except:
            pass
    return out

def las_sparade_valutakurser() -> dict:
    return las_sparade_valutakurser_cached(st.session_state.get("rates_reload", 0))

def spara_valutakurser(rates: dict):
    ws = _ensure_rates_sheet()
    body = [["Valuta","Kurs"]]
    for k in ["USD","NOK","CAD","EUR","SEK"]:
        v = rates.get(k, STANDARD_VALUTAKURSER.get(k, 1.0))
        body.append([k, str(v)])
    with_backoff(ws.clear)
    with_backoff(ws.update, body)
