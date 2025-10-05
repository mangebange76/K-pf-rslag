# stockapp/rates.py
from __future__ import annotations

from typing import Dict

import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None

from .sheets import get_spreadsheet, with_backoff

RATES_SHEET_NAME = "Valutakurser"

# Startvärden om inget finns
DEFAULT_RATES: Dict[str, float] = {
    "USD": 10.00,
    "NOK": 1.00,
    "CAD": 7.50,
    "EUR": 11.00,
    "SEK": 1.0,
}

def _get_or_create_rates_ws():
    ss = get_spreadsheet()
    try:
        ws = with_backoff(ss.worksheet, RATES_SHEET_NAME)
        return ws
    except Exception:
        ws = with_backoff(ss.add_worksheet, title=RATES_SHEET_NAME, rows=10, cols=3)
        with_backoff(ws.update, [["Valuta", "Kurs"]])
        # seed med defaults
        body = [["Valuta","Kurs"]]
        for k in ["USD","NOK","CAD","EUR","SEK"]:
            body.append([k, str(DEFAULT_RATES[k])])
        with_backoff(ws.clear)
        with_backoff(ws.update, body)
        return ws

def read_rates() -> Dict[str, float]:
    ws = _get_or_create_rates_ws()
    rows = with_backoff(ws.get_all_records)()  # [{'Valuta': 'USD', 'Kurs': '10.12'}, ...]
    out: Dict[str, float] = {}
    for r in rows:
        cur = str(r.get("Valuta", "")).upper().strip()
        val = str(r.get("Kurs", "")).replace(",", ".").strip()
        try:
            out[cur] = float(val)
        except Exception:
            pass
    # Fyll upp med defaults om något saknas
    for k, v in DEFAULT_RATES.items():
        out.setdefault(k, v)
    return out

def save_rates(rates: Dict[str, float]):
    ws = _get_or_create_rates_ws()
    body = [["Valuta","Kurs"]]
    for k in ["USD","NOK","CAD","EUR","SEK"]:
        v = rates.get(k, DEFAULT_RATES.get(k, 1.0))
        try:
            v = float(v)
        except Exception:
            v = DEFAULT_RATES.get(k, 1.0)
        body.append([k, str(v)])
    with_backoff(ws.clear)
    with_backoff(ws.update, body)

def fetch_live_rates() -> Dict[str, float]:
    """
    Hämtar livekurser via Yahoo Finance FX-symboler:
      USDSEK=X, NOKSEK=X, CADSEK=X, EURSEK=X
    Faller tillbaka till sparade/DEFAULT om något saknas.
    """
    out = read_rates()  # start med sparade/defaults
    if yf is None:
        return out
    symbols = {
        "USD": "USDSEK=X",
        "NOK": "NOKSEK=X",
        "CAD": "CADSEK=X",
        "EUR": "EURSEK=X",
    }
    for k, sym in symbols.items():
        try:
            t = yf.Ticker(sym)
            info = {}
            try:
                info = t.info or {}
            except Exception:
                info = {}
            px = info.get("regularMarketPrice", None)
            if px is None:
                h = t.history(period="1d")
                if hasattr(h, "empty") and (not h.empty) and ("Close" in h):
                    px = float(h["Close"].iloc[-1])
            if px is not None and float(px) > 0:
                out[k] = float(px)
        except Exception:
            # ignorera och behåll sparad/default
            pass
    out["SEK"] = 1.0
    return out

def repair_rates_sheet():
    """Återskapar rubriker och säkerställer att alla valutor finns."""
    ws = _get_or_create_rates_ws()
    # läs befintligt
    current = read_rates()
    body = [["Valuta","Kurs"]]
    for k in ["USD","NOK","CAD","EUR","SEK"]:
        body.append([k, str(current.get(k, DEFAULT_RATES[k]))])
    with_backoff(ws.clear)
    with_backoff(ws.update, body)
