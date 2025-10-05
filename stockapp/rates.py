from __future__ import annotations

import time
import pandas as pd
import streamlit as st
import yfinance as yf

from .sheets import _spreadsheet, _with_backoff

RATES_SHEET_NAME = "Valutakurser"

DEFAULT_RATES = {"USD": 9.75, "NOK": 0.95, "CAD": 7.05, "EUR": 11.18, "SEK": 1.0}

def _ws_rates(create: bool = True):
    ss = _spreadsheet()
    try:
        return _with_backoff(ss.worksheet, RATES_SHEET_NAME)
    except Exception:
        if not create:
            raise
        _with_backoff(ss.add_worksheet, title=RATES_SHEET_NAME, rows=10, cols=5)
        ws = _with_backoff(ss.worksheet, RATES_SHEET_NAME)
        _with_backoff(ws.update, [["Valuta","Kurs"]])
        return ws

def read_rates() -> dict:
    ws = _ws_rates(create=True)
    rows = _with_backoff(ws.get_all_records)  # [{'Valuta':'USD','Kurs':'9.74'}, ...]
    out = {}
    for r in rows:
        cur = str(r.get("Valuta","")).upper().strip()
        val = str(r.get("Kurs","")).replace(",", ".").strip()
        try:
            out[cur] = float(val)
        except Exception:
            pass
    # fyll defaultar för saknade
    for k, v in DEFAULT_RATES.items():
        out.setdefault(k, v)
    return out

def save_rates(rates: dict):
    ws = _ws_rates(create=True)
    body = [["Valuta","Kurs"]]
    for k in ["USD","NOK","CAD","EUR","SEK"]:
        body.append([k, str(rates.get(k, DEFAULT_RATES.get(k, 1.0)))])
    _with_backoff(ws.clear)
    _with_backoff(ws.update, body)

def fetch_live_rates() -> dict:
    # Hämtar via Yahoo FX-par; robust med liten backoff
    pairs = {"USD": "USDSEK=X", "NOK": "NOKSEK=X", "CAD": "CADSEK=X", "EUR": "EURSEK=X"}
    out = {"SEK": 1.0}
    for k, y in pairs.items():
        try:
            t = yf.Ticker(y)
            px = None
            try:
                px = t.fast_info.last_price
            except Exception:
                pass
            if not px:
                hist = t.history(period="5d")
                if not hist.empty and "Close" in hist:
                    px = float(hist["Close"].dropna().iloc[-1])
            if px:
                out[k] = float(px)
        except Exception:
            pass
        time.sleep(0.25)
    # Fyll ev. luckor
    for k, v in DEFAULT_RATES.items():
        out.setdefault(k, v)
    return out

def repair_rates_sheet():
    """Säkerställ att bladet finns och har rubriker."""
    ws = _ws_rates(create=True)
    vals = _with_backoff(ws.get_all_values)()
    if not vals:
        _with_backoff(ws.update, [["Valuta","Kurs"]])
