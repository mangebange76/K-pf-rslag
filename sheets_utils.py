# sheets_utils.py
from __future__ import annotations
import time
from datetime import datetime
import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

# --- Lokal Stockholm-tid om pytz finns (annars systemtid) ---
try:
    import pytz
    TZ_STHLM = pytz.timezone("Europe/Stockholm")
    def now_stamp() -> str:
        return datetime.now(TZ_STHLM).strftime("%Y-%m-%d %H:%M")
except Exception:
    def now_stamp() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M")

# --- Google Sheets-koppling ---
SHEET_URL = st.secrets["SHEET_URL"]
MAIN_SHEET_NAME = "Blad1"
RATES_SHEET_NAME = "Valutakurser"
LOGS_SHEET_NAME = "LOGS"
PROPOSALS_SHEET_NAME = "FÖRSLAG"

scope = ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
_client = gspread.authorize(credentials)

def _with_backoff(func, *args, **kwargs):
    delays = [0, 0.5, 1.0, 2.0]
    last_err = None
    for d in delays:
        if d: time.sleep(d)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_err = e
    raise last_err

def _get_spreadsheet():
    return _client.open_by_url(SHEET_URL)

def _get_or_create_ws(title: str, rows: int = 100, cols: int = 30):
    ss = _get_spreadsheet()
    try:
        return ss.worksheet(title)
    except Exception:
        ss.add_worksheet(title=title, rows=rows, cols=cols)
        return ss.worksheet(title)

# --- Valutakurser: standardvärden & CRUD ---
STANDARD_VALUTAKURSER = {"USD": 9.75, "NOK": 0.95, "CAD": 7.05, "EUR": 11.18, "SEK": 1.0}

@st.cache_data(show_spinner=False)
def las_sparade_valutakurser_cached(nonce: int):
    ws = _get_or_create_ws(RATES_SHEET_NAME, rows=10, cols=5)
    rows = _with_backoff(ws.get_all_records)
    out = {}
    for r in rows:
        cur = str(r.get("Valuta","")).upper().strip()
        val = str(r.get("Kurs","")).replace(",", ".").strip()
        try:
            out[cur] = float(val)
        except:
            pass
    return out

def las_sparade_valutakurser() -> dict:
    return las_sparade_valutakurser_cached(st.session_state.get("rates_reload", 0)) or STANDARD_VALUTAKURSER.copy()

def spara_valutakurser(rates: dict):
    ws = _get_or_create_ws(RATES_SHEET_NAME, rows=10, cols=5)
    body = [["Valuta","Kurs"]]
    for k in ["USD","NOK","CAD","EUR","SEK"]:
        v = rates.get(k, STANDARD_VALUTAKURSER.get(k, 1.0))
        body.append([k, str(v)])
    _with_backoff(ws.clear)
    _with_backoff(ws.update, body)

def hamta_valutakurs(valuta: str, user_rates: dict) -> float:
    if not valuta:
        return 1.0
    return user_rates.get(valuta.upper(), STANDARD_VALUTAKURSER.get(valuta.upper(), 1.0))

# --- Data (huvudblad) ---
def _ensure_main_sheet_and_header():
    ws = _get_or_create_ws(MAIN_SHEET_NAME, rows=2000, cols=40)
    try:
        vals = _with_backoff(ws.get_all_values)
        if not vals:
            return ws
        if vals and vals[0]:
            return ws
    except Exception:
        pass
    return ws

def hamta_data() -> pd.DataFrame:
    ws = _ensure_main_sheet_and_header()
    recs = _with_backoff(ws.get_all_records)
    return pd.DataFrame(recs)

def spara_data(df: pd.DataFrame):
    ws = _get_or_create_ws(MAIN_SHEET_NAME, rows=max(100, len(df)+10), cols=max(40, len(df.columns)+2))
    _with_backoff(ws.clear)
    _with_backoff(ws.update, [df.columns.values.tolist()] + df.astype(str).values.tolist())

# --- Snapshot vid uppstart ---
def _today_tag() -> str:
    return datetime.now().strftime("%Y-%m-%d")

def skapa_snapshot_om_saknas(df: pd.DataFrame):
    ss = _get_spreadsheet()
    tag = _today_tag()
    existing = [ws.title for ws in ss.worksheets()]
    has_today = any(t.startswith(f"SNAP_{tag}") for t in existing)
    if has_today:
        return False, "Snapshot för idag finns redan."
    title = f"SNAP_{tag}_{datetime.now().strftime('%H%M')}"
    ss.add_worksheet(title=title, rows=max(100, len(df)+10), cols=max(40, len(df.columns)+2))
    ws = ss.worksheet(title)
    _with_backoff(ws.update, [df.columns.values.tolist()] + df.astype(str).values.tolist())
    return True, f"Snapshot skapad: {title}"

# --- LOGG & FÖRSLAG till Sheets ---
def save_logs_to_sheet(logs: list[dict]) -> int:
    ws = _get_or_create_ws(LOGS_SHEET_NAME, rows=max(200, len(logs)+10), cols=12)
    # Hämta befintligt för att lägga till längst ned
    vals = _with_backoff(ws.get_all_values)
    header = ["ts","ticker","summary","ps_json"]
    if not vals:
        _with_backoff(ws.update, [header])
        vals = [header]
    rows = []
    for r in logs:
        rows.append([
            r.get("ts",""),
            r.get("ticker",""),
            r.get("summary",""),
            json_dumps_safe(r.get("ps", {}))
        ])
    _with_backoff(ws.append_rows, rows)
    return len(rows)

def save_proposals_to_sheet(df: pd.DataFrame):
    ws = _get_or_create_ws(PROPOSALS_SHEET_NAME, rows=max(100, len(df)+10), cols=max(20, len(df.columns)+2))
    _with_backoff(ws.clear)
    _with_backoff(ws.update, [df.columns.values.tolist()] + df.astype(str).values.tolist())

def json_dumps_safe(obj) -> str:
    try:
        import json
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)
