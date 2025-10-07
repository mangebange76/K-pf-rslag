# sheets_utils.py
from __future__ import annotations
import time
from datetime import datetime
import pandas as pd
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

# -------------------- Tid/zon --------------------
try:
    import pytz
    TZ_STHLM = pytz.timezone("Europe/Stockholm")
    def now_stamp() -> str:
        return datetime.now(TZ_STHLM).strftime("%Y-%m-%d")
except Exception:
    def now_stamp() -> str:
        return datetime.now().strftime("%Y-%m-%d")

# -------------------- Konfig --------------------
SHEET_URL = st.secrets.get("SHEET_URL", "")
SHEET_NAME = st.secrets.get("SHEET_NAME", "Blad1")
RATES_SHEET_NAME = st.secrets.get("RATES_SHEET_NAME", "Valutakurser")

DEFAULT_RATES = {"USD": 9.75, "EUR": 11.18, "NOK": 0.95, "CAD": 7.05, "SEK": 1.0}

# Viktigt: matchar app.py + nya metadatafält som vyerna använder
FINAL_COLS = [
    # Bas
    "Ticker", "Bolagsnamn", "Valuta", "Aktuell kurs", "Årlig utdelning",
    "Utestående aktier", "Antal aktier",
    # P/S & kvartal
    "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "P/S Q1 datum", "P/S Q2 datum", "P/S Q3 datum", "P/S Q4 datum",
    "Källa Aktuell kurs", "Källa Utestående aktier", "Källa P/S", "Källa P/S Q1", "Källa P/S Q2", "Källa P/S Q3", "Källa P/S Q4",
    # Omsättning & riktkurser
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
    # Derivat/övrigt
    "CAGR 5 år (%)", "P/S-snitt",
    # Tidsstämplar & meta
    "Senast manuellt uppdaterad", "Senast auto uppdaterad",
    "TS P/S", "TS Utestående aktier", "TS Omsättning",
]

_STR_COLS = {
    "Ticker","Bolagsnamn","Valuta",
    "P/S Q1 datum","P/S Q2 datum","P/S Q3 datum","P/S Q4 datum",
    "Källa Aktuell kurs","Källa Utestående aktier","Källa P/S","Källa P/S Q1","Källa P/S Q2","Källa P/S Q3","Källa P/S Q4",
    "Senast manuellt uppdaterad","Senast auto uppdaterad",
    "TS P/S","TS Utestående aktier","TS Omsättning",
}
_NUM_COLS = {
    "Utestående aktier","Antal aktier",
    "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt",
    "Aktuell kurs","Årlig utdelning","CAGR 5 år (%)",
    "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
    "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
}

# -------------------- Google Sheets-klient --------------------
def _client():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds_info = st.secrets.get("GOOGLE_CREDENTIALS", {})
    if not creds_info:
        raise RuntimeError("Saknar GOOGLE_CREDENTIALS i secrets.")
    credentials = Credentials.from_service_account_info(creds_info, scopes=scope)
    return gspread.authorize(credentials)

def _with_backoff(func, *args, **kwargs):
    delays = [0, 0.5, 1.0, 2.0]
    last = None
    for d in delays:
        if d:
            time.sleep(d)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last = e
    raise last

def _spreadsheet():
    if not SHEET_URL:
        raise RuntimeError("Saknar SHEET_URL i secrets.")
    return _with_backoff(_client().open_by_url, SHEET_URL)

def _worksheet(name: str):
    ss = _spreadsheet()
    try:
        return ss.worksheet(name)
    except Exception:
        ws = ss.add_worksheet(title=name, rows=5000, cols=len(FINAL_COLS) + 5)
        _with_backoff(ws.update, [FINAL_COLS])
        return ws

def _rates_ws():
    ss = _spreadsheet()
    try:
        return ss.worksheet(RATES_SHEET_NAME)
    except Exception:
        ws = ss.add_worksheet(title=RATES_SHEET_NAME, rows=20, cols=3)
        _with_backoff(ws.update, [["Valuta","Kurs"]] + [[k, DEFAULT_RATES[k]] for k in ["USD","EUR","NOK","CAD","SEK"]])
        return ws

# -------------------- IO: huvudblad --------------------
def hamta_data() -> pd.DataFrame:
    ws = _worksheet(SHEET_NAME)
    rows = _with_backoff(ws.get_all_records)
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame({c: [] for c in FINAL_COLS})
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)
    # håll ordningen
    for c in FINAL_COLS:
        if c not in df.columns:
            df[c] = "" if c in _STR_COLS else 0.0
    return df[FINAL_COLS]

def spara_data(df: pd.DataFrame) -> None:
    ws = _worksheet(SHEET_NAME)
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)
    for c in FINAL_COLS:
        if c not in df.columns:
            df[c] = "" if c in _STR_COLS else 0.0
    df = df[FINAL_COLS]
    _with_backoff(ws.clear)
    _with_backoff(ws.update, [df.columns.tolist()] + df.astype(str).values.tolist())

# -------------------- Schemahjälpare --------------------
def säkerställ_kolumner(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for kol in FINAL_COLS:
        if kol not in df.columns:
            df[kol] = "" if kol in _STR_COLS else 0.0
    return df

def konvertera_typer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in _NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in _STR_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str)
    # Heuristik: om Utestående aktier råkat vara i absoluta tal, konvertera till miljoner
    if "Utestående aktier" in df.columns:
        df["Utestående aktier"] = df["Utestående aktier"].apply(lambda x: float(x)/1e6 if float(x) > 1e9 else float(x))
    return df

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
    st.cache_data.clear()  # reset cache så sidopanelen läser om

def hamta_valutakurs(valuta: str, user_rates: dict | None = None) -> float:
    if not valuta:
        return 1.0
    v = str(valuta).upper().strip()
    if user_rates and v in user_rates:
        return float(user_rates[v])
    saved = las_sparade_valutakurser()
    return float(saved.get(v, DEFAULT_RATES.get(v, 1.0)))

# -------------------- Snapshot-funktion --------------------
def skapa_snapshot_om_saknas(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Skapar ett nytt blad 'SNAP_YYYY-MM-DD' om det inte redan finns.
    Returnerar (skapades?, meddelande).
    """
    today = now_stamp()
    snap_name = f"SNAP_{today}"
    ss = _spreadsheet()
    try:
        ss.worksheet(snap_name)
        return False, f"Snapshot {snap_name} finns redan."
    except Exception:
        pass

    # skapa nytt blad och skriv DF
    ws = ss.add_worksheet(title=snap_name, rows= max(1000, len(df) + 10), cols= max(20, len(FINAL_COLS) + 2))
    # säkerställ schema innan skrivning
    df2 = säkerställ_kolumner(df)
    df2 = konvertera_typer(df2)
    for c in FINAL_COLS:
        if c not in df2.columns:
            df2[c] = "" if c in _STR_COLS else 0.0
    df2 = df2[FINAL_COLS]

    _with_backoff(ws.update, [df2.columns.tolist()] + df2.astype(str).values.tolist())
    return True, f"Skapade snapshot {snap_name}."
