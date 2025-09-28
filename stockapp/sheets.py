# stockapp/sheets.py
# -*- coding: utf-8 -*-
"""
Google Sheets-kopplingar (gspread) med:
- backoff/retries
- säker läs/skriv (skriver aldrig ut en tom DF, och rör inte om inga data)
- snapshot-stöd
- valutablad (läsa/spara)
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple, Optional

import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

from .config import (
    FINAL_COLS,
    STANDARD_VALUTAKURSER,
    säkerställ_kolumner,
    konvertera_typer,
)

# ---------------------------------------------------------------------------
# Konfiguration från secrets (med rimliga defaults)
# ---------------------------------------------------------------------------

SHEET_URL: str = st.secrets.get("SHEET_URL", "")
SHEET_NAME: str = st.secrets.get("SHEET_NAME", "Blad1")
RATES_SHEET_NAME: str = st.secrets.get("RATES_SHEET_NAME", "Valutakurser")

SCOPES = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]

# Klient initieras lazily och cachas i sessionen
def _get_client() -> gspread.Client:
    key = "_gs_client"
    if key in st.session_state and st.session_state[key] is not None:
        return st.session_state[key]
    creds_dict = st.secrets.get("GOOGLE_CREDENTIALS", None)
    if not creds_dict:
        raise RuntimeError("Saknar GOOGLE_CREDENTIALS i st.secrets.")
    credentials = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
    client = gspread.authorize(credentials)
    st.session_state[key] = client
    return client

# ---------------------------------------------------------------------------
# Hjälpare: backoff runt gspread-anrop
# ---------------------------------------------------------------------------

def _with_backoff(func, *args, **kwargs):
    """
    Enkel backoff vid intermittent fel/kvoter (429/5xx). Prova några gånger.
    Returnerar func(*args, **kwargs) eller re-raisar sista felet.
    """
    delays = [0.0, 0.4, 0.8, 1.6, 3.0]
    last_err = None
    for d in delays:
        if d > 0:
            time.sleep(d)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_err = e
            # fortsätt försöka
            continue
    # Om vi hamnar här – ge upp
    raise last_err

# ---------------------------------------------------------------------------
# Spreadsheet / Worksheet helpers
# ---------------------------------------------------------------------------

def get_spreadsheet():
    if not SHEET_URL:
        raise RuntimeError("SHEET_URL saknas i st.secrets.")
    client = _get_client()
    return _with_backoff(client.open_by_url, SHEET_URL)

def get_ws(name: str):
    ss = get_spreadsheet()
    try:
        return _with_backoff(ss.worksheet, name)
    except Exception:
        # skapa nytt blad om det saknas
        _with_backoff(ss.add_worksheet, title=name, rows=1000, cols=50)
        return _with_backoff(ss.worksheet, name)

def get_main_ws():
    return get_ws(SHEET_NAME)

def get_rates_ws():
    return get_ws(RATES_SHEET_NAME)

# ---------------------------------------------------------------------------
# Läs / Spara huvudblad (DataFrame)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def read_df() -> pd.DataFrame:
    """
    Läser huvudbladet som DataFrame. Säkrar schema & typer.
    Returnerar alltid en DF med FINAL_COLS (kan vara tom).
    """
    ws = get_main_ws()
    try:
        rows: List[Dict[str, str]] = _with_backoff(ws.get_all_records)
    except Exception:
        rows = []
    df = pd.DataFrame(rows)

    # Säkerställ schema + typer
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)
    return df

def _df_to_body(df: pd.DataFrame) -> List[List[str]]:
    """
    Bygger gspread-kompatibel body (lista av rader) från DF.
    Alla värden görs strängar (Sheets sparar ändå som strängar).
    """
    headers = df.columns.tolist()
    values = df.astype(str).values.tolist()
    return [headers] + values

def _safe_to_write(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    En uppsättning säkerhetsregler för att undvika att råka tömma databasen:
    - DF får inte vara None eller tom på rader
    - Måste innehålla minst kolumnen "Ticker"
    - Minst 1 rad måste ha Ticker ifylld
    """
    if df is None:
        return False, "DataFrame är None."
    if df.empty:
        return False, "DataFrame är tom (0 rader)."
    if "Ticker" not in df.columns:
        return False, "Kolumnen 'Ticker' saknas."
    if df["Ticker"].astype(str).str.strip().eq("").all():
        return False, "Inga tickers – vägrar skriva."
    return True, ""

def backup_snapshot(df: pd.DataFrame, prefix: str = "Snapshot") -> Optional[str]:
    """
    Skapar en snapshot-flik i samma Google Sheet: 'Snapshot-YYYYmmdd-HHMMSS'.
    Returnerar namnet på bladet vid lyckad snapshot, annars None.
    Misslyckanden ger endast Streamlit-varning – sparning stoppas inte.
    """
    from datetime import datetime as _dt
    ss = get_spreadsheet()
    snap_name = f"{prefix}-{_dt.now().strftime('%Y%m%d-%H%M%S')}"
    try:
        ws = _with_backoff(ss.add_worksheet, title=snap_name, rows=max(1000, len(df)+10), cols=max(50, len(df.columns)+2))
        body = _df_to_body(df)
        _with_backoff(ws.clear)
        _with_backoff(ws.update, body)
        st.sidebar.success(f"Snapshot skapad: {snap_name}")
        return snap_name
    except Exception as e:
        st.sidebar.warning(f"Kunde inte skapa snapshot ({snap_name}): {e}")
        return None

def save_df(df: pd.DataFrame, do_snapshot: bool = False) -> bool:
    """
    Skriv HELA DataFrame till huvudbladet.
    Säker: skriver aldrig om DF inte passerar _safe_to_write.
    Skapar valfri snapshot före skrivning.
    Returnerar True vid lyckad skrivning.
    """
    ok, reason = _safe_to_write(df)
    if not ok:
        st.warning(f"Skrivning avbröts: {reason}")
        return False

    # säkerställ schema/typer precis innan skrivning
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)

    if do_snapshot:
        backup_snapshot(df, prefix="Snapshot")

    ws = get_main_ws()
    body = _df_to_body(df)

    # Viktigt: clear + update i en transaktionell sekvens med backoff.
    # Om update skulle fallera, är det risk att bladet är tomt – därför gör vi:
    # 1) Försök update direkt (gspread overwrite) – fungerar endast om dimensionerna räcker.
    # 2) Om det misslyckas: clear → update.
    try:
        _with_backoff(ws.update, body)
        return True
    except Exception:
        pass

    # Plan B: clear + update
    try:
        _with_backoff(ws.clear)
        _with_backoff(ws.update, body)
        return True
    except Exception as e:
        st.error(f"Misslyckades skriva till Google Sheets: {e}")
        return False

# ---------------------------------------------------------------------------
# Valutablad: Läs/Spara
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def read_saved_rates() -> Dict[str, float]:
    """
    Läser valutabladet (Valutakurser) och returnerar en dict {CCY: rate}.
    Saknade/ogiltiga värden ignoreras. Har alltid SEK:1.0.
    """
    ws = get_rates_ws()
    try:
        rows: List[Dict[str, str]] = _with_backoff(ws.get_all_records)
    except Exception:
        rows = []

    out: Dict[str, float] = {}
    for r in rows:
        cur = str(r.get("Valuta", "")).upper().strip()
        val = str(r.get("Kurs", "")).replace(",", ".").strip()
        if not cur:
            continue
        try:
            f = float(val)
            out[cur] = f
        except Exception:
            continue

    # se till att baseline finns alltid
    out.setdefault("SEK", 1.0)
    return out

def save_rates(rates: Dict[str, float]) -> bool:
    """
    Sparar valutakurser (ordningen: USD, NOK, CAD, EUR, SEK) till bladet.
    """
    ws = get_rates_ws()
    body = [["Valuta", "Kurs"]]
    for k in ["USD", "NOK", "CAD", "EUR", "SEK"]:
        v = rates.get(k, STANDARD_VALUTAKURSER.get(k, 1.0))
        try:
            v = float(v)
        except Exception:
            v = STANDARD_VALUTAKURSER.get(k, 1.0)
        body.append([k, str(v)])

    try:
        _with_backoff(ws.clear)
        _with_backoff(ws.update, body)
        # töm cache så sidopanelen ser nya värden direkt
        read_saved_rates.clear()  # type: ignore[attr-defined]
        return True
    except Exception as e:
        st.warning(f"Kunde inte spara valutakurser: {e}")
        return False
