# stockapp/rates.py
from __future__ import annotations

import time
from typing import Dict, Any, List

import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None  # type: ignore

from .sheets import get_spreadsheet

# -------------------------------------------------
# Konstanter
# -------------------------------------------------
RATES_SHEET_NAME = "Valutakurser"

DEFAULT_RATES: Dict[str, float] = {
    "USD": 9.75,
    "NOK": 0.95,
    "CAD": 7.05,
    "EUR": 11.18,
    "SEK": 1.0,
}

FX_PAIRS = {
    "USD": "USDSEK=X",
    "NOK": "NOKSEK=X",
    "CAD": "CADSEK=X",
    "EUR": "EURSEK=X",
}

# -------------------------------------------------
# Backoff-hjälpare (egen – ingen import från sheets)
# -------------------------------------------------
def _with_backoff(func, *args, **kwargs):
    delays = [0.0, 0.5, 1.0, 2.0, 3.0]
    last = None
    for d in delays:
        if d:
            time.sleep(d)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last = e
    if last:
        raise last
    raise RuntimeError("Okänt fel i _with_backoff")

# -------------------------------------------------
# Intern helpers
# -------------------------------------------------
def _get_or_create_rates_ws():
    ss = get_spreadsheet()
    try:
        return _with_backoff(ss.worksheet, RATES_SHEET_NAME)
    except Exception:
        # Skapa och initiera
        ws = _with_backoff(ss.add_worksheet, title=RATES_SHEET_NAME, rows=10, cols=5)
        body = [["Valuta", "Kurs"]]
        for k in ["USD", "NOK", "CAD", "EUR", "SEK"]:
            body.append([k, str(DEFAULT_RATES[k])])
        _with_backoff(ws.update, body)
        return ws

def _parse_float(x: Any) -> float:
    try:
        if isinstance(x, str):
            return float(x.replace(",", ".").strip())
        return float(x)
    except Exception:
        return 0.0

def _yf_last_fx(pair: str) -> float:
    if yf is None or not pair:
        return 0.0
    try:
        t = yf.Ticker(pair)
        # 1) Försök realtidsfält
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}
        px = info.get("regularMarketPrice")
        if px is not None and float(px) > 0:
            return float(px)
        # 2) Fallback: senaste close
        hist = t.history(period="5d")
        if not hist.empty and "Close" in hist.columns:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return 0.0

# -------------------------------------------------
# Publika API:n som app.py använder
# -------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def read_rates() -> Dict[str, float]:
    """
    Läser in valutakurser från bladet 'Valutakurser' (Valuta|Kurs).
    Returnerar alltid ett dict med minst DEFAULT_RATES.
    """
    ws = _get_or_create_rates_ws()
    rows = _with_backoff(ws.get_all_records)
    out: Dict[str, float] = {}
    if isinstance(rows, list):
        for r in rows:
            cur = str(r.get("Valuta", "")).upper().strip()
            val = _parse_float(r.get("Kurs", ""))
            if cur:
                out[cur] = val
    # Fyll upp saknade med standard
    for k, v in DEFAULT_RATES.items():
        out.setdefault(k, v)
    return out

def save_rates(rates: Dict[str, float]) -> None:
    """
    Sparar kurser (USD/NOK/CAD/EUR/SEK) till bladet.
    Tömmer cache så sidopanelen får in de nya värdena direkt.
    """
    ws = _get_or_create_rates_ws()
    body: List[List[str]] = [["Valuta", "Kurs"]]
    for k in ["USD", "NOK", "CAD", "EUR", "SEK"]:
        v = _parse_float(rates.get(k, DEFAULT_RATES[k]))
        body.append([k, f"{v}"])
    _with_backoff(ws.clear)
    _with_backoff(ws.update, body)
    # Bust cache
    try:
        read_rates.clear()  # type: ignore[attr-defined]
    except Exception:
        pass

def fetch_live_rates() -> Dict[str, float]:
    """
    Hämtar livekurser (bäst tillgängliga) via yfinance:
      USDSEK=X, NOKSEK=X, CADSEK=X, EURSEK=X
    Faller tillbaka till sparade eller DEFAULT om inget hittas.
    Returnerar ett fullständigt rates-dict.
    """
    current = read_rates().copy()
    updated = {}
    for k, pair in FX_PAIRS.items():
        val = _yf_last_fx(pair)
        if val > 0:
            updated[k] = val
    if updated:
        # slå ihop med befintliga
        current.update(updated)
    return current

def repair_rates_sheet() -> bool:
    """
    Säkerställer att bladet finns och har korrekt header.
    Om något är skrotat, skriv default-tabell.
    """
    ws = _get_or_create_rates_ws()
    try:
        rows = _with_backoff(ws.get_all_values)()
        header_ok = bool(rows and rows[0] and rows[0][:2] == ["Valuta", "Kurs"])
        if not header_ok:
            body = [["Valuta", "Kurs"]]
            for k in ["USD", "NOK", "CAD", "EUR", "SEK"]:
                body.append([k, str(DEFAULT_RATES[k])])
            _with_backoff(ws.clear)
            _with_backoff(ws.update, body)
        return True
    except Exception:
        # sista utväg: skriv default helt
        body = [["Valuta", "Kurs"]]
        for k in ["USD", "NOK", "CAD", "EUR", "SEK"]:
            body.append([k, str(DEFAULT_RATES[k])])
        _with_backoff(ws.clear)
        _with_backoff(ws.update, body)
        return True
