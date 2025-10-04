# stockapp/rates.py
from __future__ import annotations

import time
from typing import Dict

import streamlit as st

from stockapp.sheets import get_spreadsheet, _with_backoff

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
        _with_backoff(ws.update, "A1", [["Valuta", "Kurs"]], value_input_option="RAW")
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
                v = float(val)
                # normalisera uppenbart fel (tex 9369999886 pga lokal-tolkning)
                if v > 1000:  # ingen rimlig SEK-kurs för USD/NOK/CAD/EUR
                    # prova att skala ner om det ser ut som tusentalssammanslagning
                    while v > 1000:
                        v /= 1000.0
                out[cur] = float(v)
            except Exception:
                pass
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
        body.append([k, f"{v:.6f}"])
    _with_backoff(ws.clear)
    # RAW så att 9.37 förblir "9.37" i cellen oavsett lokal
    _with_backoff(ws.update, "A1", body, value_input_option="RAW")


def fetch_live_rates() -> Dict[str, float]:
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
            price = DEFAULT_RATES.get(cur, 1.0)
        out[cur] = float(price)
        time.sleep(0.2)
    return out


def repair_rates_sheet():
    ws = _ensure_rates_ws()
    values = _with_backoff(ws.get_all_values)
    if not values:
        _with_backoff(ws.update, "A1", [["Valuta", "Kurs"]], value_input_option="RAW")
        return
    header = [h.strip().lower() for h in values[0]]
    if "valuta" not in header or "kurs" not in header:
        rows = values[1:]
        fixed = [["Valuta", "Kurs"]]
        for r in rows:
            cur = (r[0] if len(r) >= 1 else "").strip()
            val = (r[1] if len(r) >= 2 else "").strip()
            if cur:
                fixed.append([cur, val])
        _with_backoff(ws.clear)
        _with_backoff(ws.update, "A1", fixed, value_input_option="RAW")
