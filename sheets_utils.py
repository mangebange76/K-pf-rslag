# sheets_utils.py
from __future__ import annotations
import time
import pandas as pd
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

# -------------------- Konfig --------------------
SHEET_URL = st.secrets.get("SHEET_URL", "")
SHEET_NAME = st.secrets.get("SHEET_NAME", "Blad1")
RATES_SHEET_NAME = st.secrets.get("RATES_SHEET_NAME", "Valutakurser")

DEFAULT_RATES = {"USD": 9.75, "EUR": 11.18, "NOK": 0.95, "CAD": 7.05, "SEK": 1.0}

# Minimalt kolumnschema (inkl. nya källa-/TS-fält och datum per PSQ)
MAIN_COLS = [
    "Ticker","Bolagsnamn","Utestående aktier","Antal aktier",
    "Valuta","Aktuell kurs","Årlig utdelning","CAGR 5 år (%)",
    "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt",
    "P/S Q1 datum","P/S Q2 datum","P/S Q3 datum","P/S Q4 datum",
    "Källa P/S","Källa P/S Q1","Källa P/S Q2","Källa P/S Q3","Källa P/S Q4",
    "Källa Aktuell kurs","Källa Utestående aktier","TS P/S","TS Utestående aktier",
    "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
    "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
    "Senast manuellt uppdaterad","Senast auto uppdaterad",
]

# -------------------- Google Sheets klient --------------------
def _client():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds_info = st.secrets.get("GOOGLE_CREDENTIALS", {})
    if not creds_info:
        raise RuntimeError("Saknar GOOGLE_CREDENTIALS i secrets.")
    credentials = Credentials.from_service_account_info(creds_info, scopes=scope)
    return gspread.authorize(credentials)

def _with_backoff(func, *args, **kwargs):
    delays = [0, 0.5, 1, 2]
    last = None
    for d in delays:
        if d: time.sleep(d)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last = e
    raise last

def _spreadsheet():
    if not SHEET_URL:
        raise RuntimeError("Saknar SHEET_URL i secrets.")
    return _with_backoff(_client().open_by_url, SHEET_URL)

def _main_ws():
    ss = _spreadsheet()
    try:
        return ss.worksheet(SHEET_NAME)
    except Exception:
        ws = ss.add_worksheet(title=SHEET_NAME, rows=2000, cols=len(MAIN_COLS)+5)
        _with_backoff(ws.update, [MAIN_COLS])
        return ws

def _rates_ws():
    ss = _spreadsheet()
    try:
        return ss.worksheet(RATES_SHEET_NAME)
    except Exception:
        ws = ss.add_worksheet(title=RATES_SHEET_NAME, rows=20, cols=3)
        _with_backoff(ws.update, [["Valuta","Kurs"]]+[[k, DEFAULT_RATES[k]] for k in ["USD","EUR","NOK","CAD","SEK"]])
        return ws

# -------------------- Huvud-DF till/från Sheets --------------------
def hamta_data() -> pd.DataFrame:
    ws = _main_ws()
    rows = _with_backoff(ws.get_all_records)
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame({c: [] for c in MAIN_COLS})
    # säkerställ kolumnerna
    for c in MAIN_COLS:
        if c not in df.columns:
            df[c] = "" if c in ["Ticker","Bolagsnamn","Valuta","Källa P/S","Källa P/S Q1","Källa P/S Q2","Källa P/S Q3","Källa P/S Q4","Källa Aktuell kurs","Källa Utestående aktier","P/S Q1 datum","P/S Q2 datum","P/S Q3 datum","P/S Q4 datum","Senast manuellt uppdaterad","Senast auto uppdaterad","TS P/S","TS Utestående aktier"] else 0.0
    # typning
    num_cols = ["Utestående aktier","Antal aktier","Aktuell kurs","Årlig utdelning","CAGR 5 år (%)",
                "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt",
                "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
                "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in ["Ticker","Bolagsnamn","Valuta","Källa P/S","Källa P/S Q1","Källa P/S Q2","Källa P/S Q3","Källa P/S Q4",
              "Källa Aktuell kurs","Källa Utestående aktier","P/S Q1 datum","P/S Q2 datum","P/S Q3 datum","P/S Q4 datum",
              "Senast manuellt uppdaterad","Senast auto uppdaterad","TS P/S","TS Utestående aktier"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df[MAIN_COLS]

def spara_data(df: pd.DataFrame) -> None:
    ws = _main_ws()
    # se till att alla kolumner finns och i rätt ordning
    for c in MAIN_COLS:
        if c not in df.columns:
            df[c] = ""
    df = df[MAIN_COLS]
    _with_backoff(ws.clear)
    values = [df.columns.tolist()] + df.astype(str).values.tolist()
    _with_backoff(ws.update, values)

# -------------------- Valutakurser --------------------
@st.cache_data(show_spinner=False)
def las_sparade_valutakurser() -> dict:
    ws = _rates_ws()
    rows = _with_backoff(ws.get_all_records)
    out = {"SEK": 1.0}
    for r in rows:
        cur = str(r.get("Valuta","")).upper().strip()
        val = str(r.get("Kurs","")).replace(",", ".").strip()
        try:
            out[cur] = float(val)
        except Exception:
            pass
    # defaults om saknas
    for k,v in DEFAULT_RATES.items():
        out.setdefault(k, v)
    return out

def spara_valutakurser(rates: dict) -> None:
    ws = _rates_ws()
    body = [["Valuta","Kurs"]]
    for k in ["USD","NOK","CAD","EUR","SEK"]:
        body.append([k, float(rates.get(k, DEFAULT_RATES.get(k, 1.0)))])
    _with_backoff(ws.clear)
    _with_backoff(ws.update, body)
    st.cache_data.clear()  # så sidebar läser om

def hamta_valutakurs(valuta: str, user_rates: dict | None = None) -> float:
    if not valuta:
        return 1.0
    v = str(valuta).upper().strip()
    if user_rates and v in user_rates:
        return float(user_rates[v])
    saved = las_sparade_valutakurser()
    return float(saved.get(v, DEFAULT_RATES.get(v, 1.0)))
