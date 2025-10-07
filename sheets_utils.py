# sheets_utils.py
import streamlit as st
import pandas as pd
import gspread
import time
from datetime import datetime
from google.oauth2.service_account import Credentials

# ---- Tidsstämplar ----
try:
    import pytz
    TZ_STHLM = pytz.timezone("Europe/Stockholm")
    def now_stamp(): return datetime.now(TZ_STHLM).strftime("%Y-%m-%d %H:%M")
    def now_ymd(): return datetime.now(TZ_STHLM).strftime("%Y-%m-%d")
    def now_compact(): return datetime.now(TZ_STHLM).strftime("%Y%m%d_%H%M")
except Exception:
    def now_stamp(): return datetime.now().strftime("%Y-%m-%d %H:%M")
    def now_ymd(): return datetime.now().strftime("%Y-%m-%d")
    def now_compact(): return datetime.now().strftime("%Y%m%d_%H%M")

# ---- Secrets / kopplingar ----
SHEET_URL = st.secrets["SHEET_URL"]
MAIN_SHEET_NAME = "Blad1"
RATES_SHEET_NAME = "Valutakurser"

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

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

def get_spreadsheet():
    return client.open_by_url(SHEET_URL)

# ---- Rates-ark ----
def _ensure_rates_sheet():
    ss = get_spreadsheet()
    try:
        return ss.worksheet(RATES_SHEET_NAME)
    except Exception:
        ss.add_worksheet(title=RATES_SHEET_NAME, rows=10, cols=5)
        ws = ss.worksheet(RATES_SHEET_NAME)
        ws.update("A1", [["Valuta","Kurs"]])
        return ws

@st.cache_data(show_spinner=False)
def las_sparade_valutakurser():
    ws = _ensure_rates_sheet()
    rows = _with_backoff(ws.get_all_records)
    out = {}
    for r in rows:
        cur = str(r.get("Valuta","")).upper().strip()
        val = str(r.get("Kurs","")).replace(",",".").strip()
        try: out[cur] = float(val)
        except: pass
    # defaults
    out.setdefault("USD", 9.75)
    out.setdefault("NOK", 0.95)
    out.setdefault("CAD", 7.05)
    out.setdefault("EUR", 11.18)
    out.setdefault("SEK", 1.0)
    return out

def spara_valutakurser(rates: dict):
    ws = _ensure_rates_sheet()
    body = [["Valuta","Kurs"]]
    for k in ["USD","NOK","CAD","EUR","SEK"]:
        v = rates.get(k, las_sparade_valutakurser().get(k, 1.0))
        body.append([k, str(v)])
    _with_backoff(ws.clear)
    _with_backoff(ws.update, body)

def hamta_valutakurs(valuta: str, user_rates: dict) -> float:
    if not valuta: return 1.0
    return float(user_rates.get(str(valuta).upper(), 1.0))

# ---- Läs/skriv huvuddata ----
def _ensure_main_sheet():
    ss = get_spreadsheet()
    try:
        return ss.worksheet(MAIN_SHEET_NAME)
    except Exception:
        ss.add_worksheet(title=MAIN_SHEET_NAME, rows=2000, cols=60)
        return ss.worksheet(MAIN_SHEET_NAME)

def hamta_data() -> pd.DataFrame:
    ws = _ensure_main_sheet()
    rows = _with_backoff(ws.get_all_records)
    if not rows: return pd.DataFrame()
    return pd.DataFrame(rows)

def spara_data(df: pd.DataFrame):
    ws = _ensure_main_sheet()
    _with_backoff(ws.clear)
    _with_backoff(ws.update, [df.columns.values.tolist()] + df.astype(str).values.tolist())

# ---- Snapshot ----
def skapa_snapshot_om_saknas(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Skapar SNAP_YYYY-MM-DD om det inte redan finns ett snapshot för dagens datum.
    Returnerar (ok, meddelande).
    """
    try:
        ss = get_spreadsheet()
        existing = [w.title for w in ss.worksheets()]
        today_tag = f"SNAP_{now_ymd()}"
        if any(t.startswith(today_tag) for t in existing):
            return False, f"Snapshot finns redan: {today_tag}"
        title = f"{today_tag}_{now_compact().split('_')[1]}"
        ss.add_worksheet(title=title, rows=max(2000, len(df)+10), cols=max(60, len(df.columns)+2))
        ws = ss.worksheet(title)
        _with_backoff(ws.update, [df.columns.values.tolist()] + df.astype(str).values.tolist())
        return True, f"Snapshot skapad: {title}"
    except Exception as e:
        return False, f"Kunde inte skapa snapshot: {e}"
