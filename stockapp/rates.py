"""
Valutakurser: läs/spara mot ett separat blad samt hämta live från Yahoo Finance.
Exponerar: read_rates, save_rates, fetch_live_rates, repair_rates_sheet, DEFAULT_RATES
Kräver stockapp.sheets.get_spreadsheet()
"""

from __future__ import annotations

import time
from typing import Dict, List

import pandas as pd
import streamlit as st

from stockapp.sheets import get_spreadsheet, _with_backoff

# yfinance används för livekurser
try:
    import yfinance as yf
except Exception:
    yf = None

RATES_SHEET_NAME = st.secrets.get("RATES_SHEET_NAME", "Valutakurser")

DEFAULT_RATES: Dict[str, float] = {
    "USD": 9.50,
    "NOK": 1.00,
    "CAD": 7.50,
    "EUR": 11.00,
    "SEK": 1.00,
}


def _ensure_rates_ws():
    ss = get_spreadsheet()
    try:
        ws = _with_backoff(ss.worksheet, RATES_SHEET_NAME)
    except Exception:
        _with_backoff(ss.add_worksheet, title=RATES_SHEET_NAME, rows=20, cols=5)
        ws = _with_backoff(ss.worksheet, RATES_SHEET_NAME)
        _with_backoff(ws.update, "A1", [["Valuta", "Kurs"]])
    return ws


def read_rates() -> Dict[str, float]:
    try:
        ws = _ensure_rates_ws()
        rows = _with_backoff(ws.get_all_records)  # [{'Valuta':'USD','Kurs':'9.75'}, ...]
        out: Dict[str, float] = {}
        for r in rows:
            cur = str(r.get("Valuta", "")).upper().strip()
            val = str(r.get("Kurs", "")).replace(" ", "").replace(",", ".").strip()
            if not cur:
                continue
            try:
                out[cur] = float(val)
            except Exception:
                pass
        # lägg in standarder för saknade
        for k, v in DEFAULT_RATES.items():
            out.setdefault(k, float(v))
        return out
    except Exception:
        return DEFAULT_RATES.copy()


def save_rates(rates: Dict[str, float]):
    ws = _ensure_rates_ws()
    body = [["Valuta", "Kurs"]]
    for k in ["USD", "NOK", "CAD", "EUR", "SEK"]:
        v = float(rates.get(k, DEFAULT_RATES.get(k, 1.0)))
        body.append([k, str(v)])
    _with_backoff(ws.clear)
    _with_backoff(ws.update, "A1", body)


def fetch_live_rates() -> Dict[str, float]:
    """
    Hämtar USD/NOK/CAD/EUR → SEK från Yahoo:
      USDSEK=X, NOKSEK=X, CADSEK=X, EURSEK=X
    """
    if yf is None:
        raise RuntimeError("yfinance saknas i miljön.")

    pairs = {
        "USD": "USDSEK=X",
        "NOK": "NOKSEK=X",
        "CAD": "CADSEK=X",
        "EUR": "EURSEK=X",
    }

    out: Dict[str, float] = {"SEK": 1.0}
    for cur, symbol in pairs.items():
        price = None
        try:
            t = yf.Ticker(symbol)
            try:
                fi = getattr(t, "fast_info", None) or {}
                price = fi.get("lastPrice", None)
            except Exception:
                price = None
            if price is None:
                h = t.history(period="1d")
                if not h.empty and "Close" in h:
                    price = float(h["Close"].iloc[-1])
        except Exception:
            price = None
        if price is None:
            # behåll default om misslyckas
            price = DEFAULT_RATES.get(cur, 1.0)
        out[cur] = float(price)
        time.sleep(0.2)  # snäll throttling
    return out


def repair_rates_sheet():
    ws = _ensure_rates_ws()
    values = _with_backoff(ws.get_all_values)
    if not values:
        _with_backoff(ws.update, "A1", [["Valuta", "Kurs"]])
        return

    header = [h.strip().lower() for h in values[0]]
    need_header = False
    if "valuta" not in header or "kurs" not in header:
        need_header = True

    if need_header:
        # skriv om hela bladet: header + bevarade rader om möjligt
        rows = values[1:]
        fixed = [["Valuta", "Kurs"]]
        # försök hitta två kolumner i gamla rader
        for r in rows:
            if not r:
                continue
            cur = (r[0] if len(r) >= 1 else "").strip() or ""
            val = (r[1] if len(r) >= 2 else "").strip() or ""
            if cur:
                fixed.append([cur, val])
        _with_backoff(ws.clear)
        _with_backoff(ws.update, "A1", fixed)
