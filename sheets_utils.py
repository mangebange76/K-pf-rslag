# sheets_utils.py
import streamlit as st
import pandas as pd
import gspread
import time
from typing import Tuple
from datetime import datetime
from google.oauth2.service_account import Credentials

# Tidsstämplar (Stockholm)
try:
    import pytz
    TZ_STHLM = pytz.timezone("Europe/Stockholm")
    def now_stamp() -> str: return datetime.now(TZ_STHLM).strftime("%Y-%m-%d %H:%M")
    def today_stamp() -> str: return datetime.now(TZ_STHLM).strftime("%Y-%m-%d")
except Exception:
    def now_stamp() -> str: return datetime.now().strftime("%Y-%m-%d %H:%M")
    def today_stamp() -> str: return datetime.now().strftime("%Y-%m-%d")

SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"
RATES_SHEET_NAME = "Valutakurser"
STANDARD_VALUTAKURSER = {"USD": 9.75, "NOK": 0.95, "CAD": 7.05, "EUR": 11.18, "SEK": 1.0}

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

def _with_backoff(func, *args, **kwargs):
    delays = [0, .5, 1, 2]
    last_err = None
    for d in delays:
        if d: time.sleep(d)
        try: return func(*args, **kwargs)
        except Exception as e: last_err = e
    raise last_err

def get_spreadsheet():
    return client.open_by_url(SHEET_URL)

def _main_ws():
    return get_spreadsheet().worksheet(SHEET_NAME)

def skapa_rates_sheet_if_missing():
    ss = get_spreadsheet()
    try: return ss.worksheet(RATES_SHEET_NAME)
    except Exception:
        ss.add_worksheet(title=RATES_SHEET_NAME, rows=20, cols=5)
        ws = ss.worksheet(RATES_SHEET_NAME)
        ws.update([["Valuta","Kurs"]])
        return ws

def hamta_data() -> pd.DataFrame:
    ws = _main_ws()
    data = _with_backoff(ws.get_all_records)
    return pd.DataFrame(data)

def spara_data(df: pd.DataFrame):
    ws = _main_ws()
    _with_backoff(ws.clear)
    _with_backoff(ws.update, [df.columns.values.tolist()] + df.astype(str).values.tolist())

# Snapshot
SNAP_PREFIX = "SNAP_"
def skapa_snapshot_om_saknas(df: pd.DataFrame) -> Tuple[bool, str]:
    try:
        ss = get_spreadsheet()
        namn = f"{SNAP_PREFIX}{today_stamp()}"
        if df is None or df.empty:
            df = pd.DataFrame({"_empty": []})
        if any(ws.title == namn for ws in ss.worksheets()):
            return (False, f"Snapshot {namn} finns redan.")
        ws = ss.add_worksheet(title=namn, rows=max(2, len(df)+4), cols=max(2, len(df.columns)+4))
        ws.update('A1', [[f"Snapshot skapad: {now_stamp()}"], [f"Källa: {SHEET_NAME}"]])
        body = [list(df.columns)] + df.astype(str).values.tolist()
        ws.update('A3', body)
        return (True, f"Snapshot skapad: {namn}")
    except Exception as e:
        return (False, f"Kunde inte skapa snapshot ({e}).")

# Valutakurser
@st.cache_data(show_spinner=False)
def las_sparade_valutakurser_cached(nonce: int):
    ws = skapa_rates_sheet_if_missing()
    rows = _with_backoff(ws.get_all_records)
    out = {}
    for r in rows:
        cur = str(r.get("Valuta","")).upper().strip()
        val = str(r.get("Kurs","")).replace(",", ".").strip()
        try: out[cur] = float(val)
        except: pass
    return out

def las_sparade_valutakurser() -> dict:
    return las_sparade_valutakurser_cached(st.session_state.get("rates_reload", 0))

def spara_valutakurser(rates: dict):
    ws = skapa_rates_sheet_if_missing()
    body = [["Valuta","Kurs"]]
    for k in ["USD","NOK","CAD","EUR","SEK"]:
        body.append([k, str(rates.get(k, STANDARD_VALUTAKURSER.get(k, 1.0)))])
    _with_backoff(ws.clear)
    _with_backoff(ws.update, body)

def hamta_valutakurs(valuta: str, user_rates: dict) -> float:
    if not valuta: return 1.0
    return user_rates.get(valuta.upper(), STANDARD_VALUTAKURSER.get(valuta.upper(), 1.0))
