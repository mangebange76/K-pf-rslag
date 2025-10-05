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

from .sheets import get_spreadsheet, _with_backoff

RATES_SHEET_NAME = "Valutakurser"

DEFAULT_RATES: Dict[str, float] = {
    "USD": 9.75,
    "NOK": 0.95,
    "CAD": 7.05,
    "EUR": 11.18,
    "SEK": 1.0,
}

FX = {"USD": "USDSEK=X", "NOK": "NOKSEK=X", "CAD": "CADSEK=X", "EUR": "EURSEK=X"}


def _get_or_create_rates_ws():
    ss = get_spreadsheet()
    try:
        return _with_backoff(ss.worksheet, RATES_SHEET_NAME)
    except Exception:
        ws = _with_backoff(ss.add_worksheet, title=RATES_SHEET_NAME, rows=20, cols=5)
        body = [["Valuta", "Kurs"]]
        for k in ["USD", "NOK", "CAD", "EUR", "SEK"]:
            body.append([k, str(DEFAULT_RATES[k])])
        _with_backoff(ws.update, body)
        return ws


def _to_float(x) -> float:
    try:
        if isinstance(x, str):
            return float(x.replace(",", ".").strip())
        return float(x)
    except Exception:
        return 0.0


@st.cache_data(ttl=600, show_spinner=False)
def read_rates() -> Dict[str, float]:
    ws = _get_or_create_rates_ws()
    rows = _with_backoff(ws.get_all_records)
    out: Dict[str, float] = {}
    if isinstance(rows, list):
        for r in rows:
            cur = str(r.get("Valuta", "")).upper().strip()
            val = _to_float(r.get("Kurs", ""))
            if cur:
                out[cur] = val
    for k, v in DEFAULT_RATES.items():
        out.setdefault(k, v)
    return out


def save_rates(rates: Dict[str, float]) -> None:
    ws = _get_or_create_rates_ws()
    body = [["Valuta", "Kurs"]]
    for k in ["USD", "NOK", "CAD", "EUR", "SEK"]:
        body.append([k, str(_to_float(rates.get(k, DEFAULT_RATES[k])))])
    _with_backoff(ws.clear)
    _with_backoff(ws.update, body)
    try:
        read_rates.clear()  # cache bust
    except Exception:
        pass


def _yf_last(pair: str) -> float:
    if yf is None:
        return 0.0
    try:
        t = yf.Ticker(pair)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}
        px = info.get("regularMarketPrice")
        if px is not None and _to_float(px) > 0:
            return float(px)
        hist = t.history(period="5d")
        if not hist.empty and "Close" in hist.columns:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return 0.0


def fetch_live_rates() -> Dict[str, float]:
    cur = read_rates().copy()
    updated = {}
    for k, sym in FX.items():
        val = _yf_last(sym)
        if val > 0:
            updated[k] = val
    if updated:
        cur.update(updated)
    return cur


def repair_rates_sheet() -> bool:
    ws = _get_or_create_rates_ws()
    try:
        rows = _with_backoff(ws.get_all_values)()
        if not rows or rows[0][:2] != ["Valuta", "Kurs"]:
            body = [["Valuta", "Kurs"]]
            for k in ["USD", "NOK", "CAD", "EUR", "SEK"]:
                body.append([k, str(DEFAULT_RATES[k])])
            _with_backoff(ws.clear)
            _with_backoff(ws.update, body)
        return True
    except Exception:
        body = [["Valuta", "Kurs"]]
        for k in ["USD", "NOK", "CAD", "EUR", "SEK"]:
            body.append([k, str(DEFAULT_RATES[k])])
        _with_backoff(ws.clear)
        _with_backoff(ws.update, body)
        return True
