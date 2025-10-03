# stockapp/rates.py
from __future__ import annotations

import time
from typing import Dict, List

import requests
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

DEFAULT_RATES: Dict[str, float] = {
    "USD": 10.00,
    "NOK": 1.00,
    "CAD": 7.50,
    "EUR": 11.00,
    "SEK": 1.00,
}

RATES_SHEET_NAME = st.secrets.get("RATES_WORKSHEET_NAME", "Valutakurser")


# ---------------- Google Sheets koppling ----------------
def _gs_client() -> gspread.Client:
    creds = st.secrets.get("GOOGLE_CREDENTIALS")
    if not creds:
        raise RuntimeError("Hittade inga service account-uppgifter (GOOGLE_CREDENTIALS) i secrets.")
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    credentials = Credentials.from_service_account_info(creds, scopes=scope)
    return gspread.authorize(credentials)

def _open_sheet():
    url = st.secrets.get("SHEET_URL")
    if not url:
        raise RuntimeError("Hittade ingen SHEET_URL i secrets.")
    return _gs_client().open_by_url(url)

def _get_or_create_rates_ws():
    ss = _open_sheet()
    try:
        ws = ss.worksheet(RATES_SHEET_NAME)
    except gspread.WorksheetNotFound:
        ws = ss.add_worksheet(title=RATES_SHEET_NAME, rows=20, cols=5)
        ws.update([["Valuta", "Kurs"]])
    return ws


# ---------------- Läs / spara till blad ----------------
def read_rates() -> Dict[str, float]:
    """Läser valutakurser från bladet 'Valutakurser' (Valuta|Kurs)."""
    try:
        ws = _get_or_create_rates_ws()
        rows = ws.get_all_records()
        out: Dict[str, float] = {}
        for r in rows:
            cur = str(r.get("Valuta", "")).upper().strip()
            val_raw = str(r.get("Kurs", "")).replace(",", ".").strip()
            try:
                val = float(val_raw)
            except Exception:
                continue
            if cur:
                out[cur] = val
        # komplettera standarder om någon saknas
        for k, v in DEFAULT_RATES.items():
            out.setdefault(k, v)
        return out
    except Exception:
        # Fallback till defaults om bladet saknas eller annat fel
        return dict(DEFAULT_RATES)

def save_rates(rates: Dict[str, float]) -> None:
    """Sparar till bladet 'Valutakurser' i ordning USD,NOK,CAD,EUR,SEK."""
    ws = _get_or_create_rates_ws()
    order = ["USD", "NOK", "CAD", "EUR", "SEK"]
    body: List[List[str]] = [["Valuta", "Kurs"]]
    for k in order:
        v = float(rates.get(k, DEFAULT_RATES.get(k, 1.0)))
        body.append([k, f"{v}"])
    ws.clear()
    ws.update(body)


# ---------------- Hämta live-kurser (utan nyckel) ----------------
def fetch_live_rates() -> Dict[str, float]:
    """
    Hämtar live-kurser via exchangerate.host (gratis, ingen API-nyckel).
    Vi frågar efter SEK-baserade kurser och inverterar för att få VALUTA->SEK.

      GET https://api.exchangerate.host/latest?base=SEK&symbols=USD,NOK,CAD,EUR

    Om svaret är t.ex. USD=0.091 (dvs 1 SEK = 0.091 USD),
    blir USD->SEK = 1 / 0.091.
    """
    url = "https://api.exchangerate.host/latest"
    params = {"base": "SEK", "symbols": "USD,NOK,CAD,EUR"}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    rates = data.get("rates", {}) or {}

    out = dict(DEFAULT_RATES)
    for cur in ["USD", "NOK", "CAD", "EUR"]:
        v = float(rates.get(cur, 0.0))
        if v > 0:
            out[cur] = round(1.0 / v, 6)  # VALUTA -> SEK
    out["SEK"] = 1.0
    return out
