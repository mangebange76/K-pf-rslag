from __future__ import annotations

import time
from typing import Dict

import pandas as pd
import streamlit as st

from stockapp.sheets import get_ws

# === Standardkurser (fallback) ===
DEFAULT_RATES: Dict[str, float] = {
    "USD": 10.0,
    "NOK": 1.0,
    "CAD": 7.5,
    "EUR": 11.0,
    "SEK": 1.0,
}

# Var sparar vi kurserna?
RATES_SHEET_NAME = st.secrets.get("RATES_WORKSHEET_NAME", "Valutakurser")

# ---------------- Hjälpare ----------------
def _with_backoff(func, *args, **kwargs):
    """Exponential backoff för API 429/5xx."""
    delays = [0.0, 0.5, 1.0, 2.0, 4.0]
    last_err = None
    for d in delays:
        if d:
            time.sleep(d)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            msg = str(e).lower()
            # backoff främst för rate limit/transienta fel
            if any(x in msg for x in ["429", "quota", "rate limit", "backendError".lower(), "timed out", "deadline"]):
                last_err = e
                continue
            raise
    if last_err:
        raise last_err

def _normalize_rates(raw: Dict[str, float]) -> Dict[str, float]:
    out = {}
    for k in ["USD", "NOK", "CAD", "EUR", "SEK"]:
        try:
            out[k] = float(raw.get(k, DEFAULT_RATES[k]))
        except Exception:
            out[k] = DEFAULT_RATES[k]
    # säkerställ SEK = 1.0
    out["SEK"] = 1.0
    return out

# ---------------- Läs & spara till Google Sheets ----------------
@st.cache_data(ttl=3600, show_spinner=False)  # cachea i 1h för att undvika 429 på read
def read_rates() -> Dict[str, float]:
    """
    Läser valutakurser från bladet 'Valutakurser' (eller namn i secrets:RATES_WORKSHEET_NAME).
    Format: två kolumner "Valuta" | "Kurs"
    Returnerar dict som alltid innehåller USD/NOK/CAD/EUR/SEK.
    """
    try:
        ws = get_ws(worksheet_name=RATES_SHEET_NAME)  # kan skapa bladet om det saknas
        # Läs allt i ett anrop
        rows = _with_backoff(ws.get_all_records)
        if not rows:
            # Om bladet finns men tomt – initiera rubriker
            _with_backoff(ws.update, [["Valuta", "Kurs"]])
            return DEFAULT_RATES.copy()

        out = {}
        for r in rows:
            cur = str(r.get("Valuta", "")).upper().strip()
            val = str(r.get("Kurs", "")).replace(",", ".").strip()
            try:
                out[cur] = float(val)
            except Exception:
                pass

        return _normalize_rates(out)
    except Exception:
        # På fel – returnera default (hellre än att spräcka appen)
        return DEFAULT_RATES.copy()

def save_rates(rates: Dict[str, float]) -> None:
    """
    Skriver kurs-tabellen i ETT batch-anrop för att minimera read-tryck.
    Raderar ej cache selektivt => vi kör st.cache_data.clear() för enkelhet.
    """
    data = _normalize_rates(rates)
    body = [["Valuta", "Kurs"]]
    for k in ["USD", "NOK", "CAD", "EUR", "SEK"]:
        body.append([k, float(data[k])])

    ws = get_ws(worksheet_name=RATES_SHEET_NAME)
    # Gör gärna i två snabba writes (clear + update). Vissa klienter gör read internt,
    # men detta är ändå minimalt.
    _with_backoff(ws.clear)
    _with_backoff(ws.update, body)

    # Invalidera cache så nästa read hämtar färskt, men undviker spamming
    try:
        st.cache_data.clear()
    except Exception:
        pass

# ---------------- Live-kurser (utan extra API-nycklar) ----------------
def _get_fx_yahoo(symbol: str) -> float:
    """
    Hämtar FX via yfinance (symbol som 'USDSEK=X').
    Returnerar float eller raise.
    """
    import yfinance as yf

    t = yf.Ticker(symbol)
    price = None
    # försök snabb väg
    try:
        info = t.fast_info
        price = info.get("lastPrice", None)
    except Exception:
        price = None
    # fallback: history 1d
    if price is None:
        try:
            h = t.history(period="1d")
            if not h.empty and "Close" in h:
                price = float(h["Close"].iloc[-1])
        except Exception:
            price = None
    if price is None:
        # sista försök regular info
        try:
            info2 = t.info or {}
            price = info2.get("regularMarketPrice", None)
        except Exception:
            price = None
    if price is None:
        raise RuntimeError(f"Kunde inte hämta kurs för {symbol}")
    return float(price)

def fetch_live_rates() -> Dict[str, float]:
    """
    Hämtar live USD/NOK/CAD/EUR mot SEK via Yahoo Finance FX.
    Returnerar dict med "_source" för visning.
    """
    pairs = {
        "USD": "USDSEK=X",
        "NOK": "NOKSEK=X",
        "CAD": "CADSEK=X",
        "EUR": "EURSEK=X",
    }
    out = {}
    for k, sym in pairs.items():
        # var snäll mot rate-limits vid upprepade klick
        time.sleep(0.15)
        out[k] = _get_fx_yahoo(sym)
    out["SEK"] = 1.0
    out["_source"] = "Yahoo Finance FX"
    return out
