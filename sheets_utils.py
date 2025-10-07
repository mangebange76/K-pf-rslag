# sheets_utils.py
from __future__ import annotations
import time
import pandas as pd
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

# --------------------------------------------------------------------------------------
# Konfig
# --------------------------------------------------------------------------------------
SHEET_URL = st.secrets.get("SHEET_URL", "")
SHEET_NAME = st.secrets.get("SHEET_NAME", "Blad1")
RATES_SHEET_NAME = st.secrets.get("RATES_SHEET_NAME", "Valutakurser")

DEFAULT_RATES = {"USD": 9.75, "EUR": 11.18, "NOK": 0.95, "CAD": 7.05, "SEK": 1.0}

# Ursprungs-kolumner + nya meta/källfält som vyerna använder
FINAL_COLS = [
    # Nyckel & bas
    "Ticker", "Bolagsnamn", "Utestående aktier",
    # P/S
    "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "P/S-snitt",
    # P/S metadata
    "P/S Q1 datum", "P/S Q2 datum", "P/S Q3 datum", "P/S Q4 datum",
    "Källa P/S", "Källa P/S Q1", "Källa P/S Q2", "Källa P/S Q3", "Källa P/S Q4",
    "TS P/S",
    # Omsättning & riktkurser
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
    # Portfölj/övrigt
    "Antal aktier", "Valuta", "Årlig utdelning", "Aktuell kurs",
    "CAGR 5 år (%)",
    # Källor/tidsstämplar övrigt
    "Källa Aktuell kurs", "Källa Utestående aktier", "TS Utestående aktier",
    # Manuella datum
    "Senast manuellt uppdaterad", "Senast auto uppdaterad",
]

# --------------------------------------------------------------------------------------
# Google Sheets-klient
# --------------------------------------------------------------------------------------
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
        ws = ss.add_worksheet(title=SHEET_NAME, rows=5000, cols=len(FINAL_COLS) + 5)
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

# --------------------------------------------------------------------------------------
# Läs/skriv huvud-DF
# --------------------------------------------------------------------------------------
def hamta_data() -> pd.DataFrame:
    ws = _main_ws()
    rows = _with_backoff(ws.get_all_records)
    df = pd.DataFrame(rows)

    # Tom grund-DF om bladet är tomt
    if df.empty:
        return pd.DataFrame({c: [] for c in FINAL_COLS})

    # Säkerställ kolumner och typer
    df = säkerställ_kolumner(df)
    df = migrera_gamla_riktkurskolumner(df)
    df = konvertera_typer(df)
    # Sätt kolumnordning
    for c in FINAL_COLS:
        if c not in df.columns:
            df[c] = "" if c in _str_cols() else 0.0
    return df[FINAL_COLS]

def spara_data(df: pd.DataFrame) -> None:
    ws = _main_ws()
    # Gör DF kompatibel: säkerställ kolumner och ordning
    df = säkerställ_kolumner(df)
    df = migrera_gamla_riktkurskolumner(df)
    df = konvertera_typer(df)
    for c in FINAL_COLS:
        if c not in df.columns:
            df[c] = "" if c in _str_cols() else 0.0
    df = df[FINAL_COLS]
    _with_backoff(ws.clear)
    values = [df.columns.tolist()] + df.astype(str).values.tolist()
    _with_backoff(ws.update, values)

# --------------------------------------------------------------------------------------
# Valutakurser
# --------------------------------------------------------------------------------------
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
    # Defaults om saknas
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
    st.cache_data.clear()  # läs-cache för kurser

def hamta_valutakurs(valuta: str, user_rates: dict | None = None) -> float:
    if not valuta:
        return 1.0
    v = str(valuta).upper().strip()
    if user_rates and v in user_rates:
        return float(user_rates[v])
    saved = las_sparade_valutakurser()
    return float(saved.get(v, DEFAULT_RATES.get(v, 1.0)))

# --------------------------------------------------------------------------------------
# Schemahjälpare (för kompatibilitet med din gamla app.py)
# --------------------------------------------------------------------------------------
def _num_cols():
    return [
        "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
        "Antal aktier", "Årlig utdelning", "Aktuell kurs", "CAGR 5 år (%)", "P/S-snitt"
    ]

def _str_cols():
    return [
        "Ticker","Bolagsnamn","Valuta","Senast manuellt uppdaterad","Senast auto uppdaterad",
        "P/S Q1 datum","P/S Q2 datum","P/S Q3 datum","P/S Q4 datum",
        "Källa P/S","Källa P/S Q1","Källa P/S Q2","Källa P/S Q3","Källa P/S Q4",
        "Källa Aktuell kurs","Källa Utestående aktier","TS P/S","TS Utestående aktier",
    ]

def säkerställ_kolumner(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # lägg till nya kolumner vid behov
    for kol in FINAL_COLS:
        if kol not in df.columns:
            df[kol] = "" if kol in _str_cols() else 0.0
    # extra: om gamla kolumner saknas i FINAL_COLS, behåll dem (vi tappas inte bort)
    for c in df.columns:
        if c not in FINAL_COLS and c not in _str_cols() and c not in _num_cols():
            # lämna kvar – vi skriver ändå bara FINAL_COLS vid spara_data
            pass
    return df

def migrera_gamla_riktkurskolumner(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mapping = {
        "Riktkurs 2026": "Riktkurs om 1 år",
        "Riktkurs 2027": "Riktkurs om 2 år",
        "Riktkurs 2028": "Riktkurs om 3 år",
        "Riktkurs om idag": "Riktkurs idag",
    }
    for old, new in mapping.items():
        if old in df.columns:
            if new not in df.columns:
                df[new] = 0.0
            new_vals = pd.to_numeric(df[new], errors="coerce").fillna(0.0)
            old_vals = pd.to_numeric(df[old], errors="coerce").fillna(0.0)
            mask = (new_vals == 0.0) & (old_vals > 0.0)
            df.loc[mask, new] = old_vals[mask]
            df = df.drop(columns=[old])
    return df

def konvertera_typer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in _num_cols():
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in _str_cols():
        if c in df.columns:
            df[c] = df[c].astype(str)
    # special: se till att "Utestående aktier" är i miljoner (om någon rad råkat vara i absolut)
    # heuristik: om ett värde > 1e9 → anta absolut och konvertera till miljoner
    if "Utestående aktier" in df.columns:
        df["Utestående aktier"] = df["Utestående aktier"].apply(lambda x: float(x)/1e6 if float(x) > 1e9 else float(x))
    return df
