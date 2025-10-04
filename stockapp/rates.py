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

RATES_SHEET_NAME = st.secrets.get("RATES_WORKSHEET_NAME", "Valutakurser")


# ---------------- Hjälpare ----------------
def _with_backoff(func, *args, **kwargs):
    delays = [0.0, 0.5, 1.0, 2.0, 4.0]
    last = None
    for d in delays:
        if d:
            time.sleep(d)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            msg = str(e).lower()
            if any(x in msg for x in ["429", "quota", "rate limit", "backenderror", "deadline", "timed out"]):
                last = e
                continue
            raise
    if last:
        raise last


def _to_float_any(v) -> float:
    """Robust parser som hanterar t.ex. '9.369999886,000000' (svensk stil)."""
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(" ", "")
    if not s:
        return 0.0
    if "," in s and "." in s:
        # svensk stil: punkter används ibland som tusentalsavskiljare, komma = decimal
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "")
            s = s.replace(",", ".")
    else:
        s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return 0.0


def _normalize_rates(raw: Dict[str, float]) -> Dict[str, float]:
    out = {}
    for k in ["USD", "NOK", "CAD", "EUR", "SEK"]:
        out[k] = float(_to_float_any(raw.get(k, DEFAULT_RATES[k])))
    out["SEK"] = 1.0
    return out


# ---------------- Läs & spara till Google Sheets ----------------
@st.cache_data(ttl=3600, show_spinner=False)  # cachea i 1h
def read_rates() -> Dict[str, float]:
    """Läser kurser från bladet 'Valutakurser' -> dict."""
    try:
        ws = get_ws(worksheet_name=RATES_SHEET_NAME)
        rows = _with_backoff(ws.get_all_records)
        if not rows:
            _with_backoff(ws.update, [["Valuta", "Kurs"]])
            return DEFAULT_RATES.copy()

        out = {}
        for r in rows:
            cur = str(r.get("Valuta", "")).upper().strip()
            val = _to_float_any(r.get("Kurs", ""))
            if cur:
                out[cur] = val
        return _normalize_rates(out)
    except Exception:
        return DEFAULT_RATES.copy()


def save_rates(rates: Dict[str, float]) -> None:
    """Skriver rates i ett svep (minimerar READ)."""
    data = _normalize_rates(rates)
    body = [["Valuta", "Kurs"]]
    for k in ["USD", "NOK", "CAD", "EUR", "SEK"]:
        body.append([k, float(data[k])])

    ws = get_ws(worksheet_name=RATES_SHEET_NAME)
    _with_backoff(ws.clear)
    _with_backoff(ws.update, body)

    try:
        st.cache_data.clear()
    except Exception:
        pass


def repair_rates_sheet() -> None:
    """
    Läser dagens värden, normaliserar dem och skriver tillbaka – botar '9369999886,000000'-problemet.
    """
    ws = get_ws(worksheet_name=RATES_SHEET_NAME)
    rows = _with_backoff(ws.get_all_records)
    tmp = {}
    for r in rows:
        cur = str(r.get("Valuta", "")).upper().strip()
        val = _to_float_any(r.get("Kurs", ""))
        if cur:
            tmp[cur] = val
    save_rates(tmp or DEFAULT_RATES)


# ---------------- Live-kurser via Yahoo Finance ----------------
def _get_fx_yahoo(symbol: str) -> float:
    import yfinance as yf

    t = yf.Ticker(symbol)
    price = None
    try:
        fi = getattr(t, "fast_info", None) or {}
        price = fi.get("lastPrice", None)
    except Exception:
        price = None
    if price is None:
        try:
            h = t.history(period="1d")
            if not h.empty and "Close" in h:
                price = float(h["Close"].iloc[-1])
        except Exception:
            price = None
    if price is None:
        try:
            info = t.info or {}
            price = info.get("regularMarketPrice", None)
        except Exception:
            price = None
    if price is None:
        raise RuntimeError(f"Kunde inte hämta kurs för {symbol}")
    return float(price)


def fetch_live_rates() -> Dict[str, float]:
    pairs = {
        "USD": "USDSEK=X",
        "NOK": "NOKSEK=X",
        "CAD": "CADSEK=X",
        "EUR": "EURSEK=X",
    }
    out = {}
    for k, sym in pairs.items():
        time.sleep(0.15)
        out[k] = _get_fx_yahoo(sym)
    out["SEK"] = 1.0
    out["_source"] = "Yahoo Finance FX"
    return out
