# stockapp/rates.py
import pandas as pd
import streamlit as st
import yfinance as yf
from .sheets import _get_or_create_worksheet, with_backoff

RATES_SHEET = "Valutakurser"
DEFAULT_RATES = {"USD": 10.0, "NOK": 1.0, "CAD": 7.0, "EUR": 11.0, "SEK": 1.0}

def repair_rates_sheet():
    ws = _get_or_create_worksheet(RATES_SHEET)
    rows = with_backoff(ws.get_all_values)
    if not rows:
        with_backoff(ws.update, [["Valuta","Kurs"],["USD","10.00"],["NOK","1.00"],["CAD","7.00"],["EUR","11.00"],["SEK","1.00"]])
        return
    header = rows[0]
    if not header or "Valuta" not in header or "Kurs" not in header:
        with_backoff(ws.clear)
        with_backoff(ws.update, [["Valuta","Kurs"],["USD","10.00"],["NOK","1.00"],["CAD","7.00"],["EUR","11.00"],["SEK","1.00"]])

def read_rates() -> dict:
    ws = _get_or_create_worksheet(RATES_SHEET)
    rows = with_backoff(ws.get_all_values)
    if not rows: return DEFAULT_RATES.copy()
    header = rows[0]
    data = rows[1:]
    idx_val = {h:i for i,h in enumerate(header)}
    out = DEFAULT_RATES.copy()
    for r in data:
        try:
            cur = str(r[idx_val.get("Valuta",0)]).strip().upper()
            val = str(r[idx_val.get("Kurs",1)]).strip().replace(",", ".")
            if cur:
                out[cur] = float(val)
        except Exception:
            pass
    return out

def save_rates(d: dict):
    ws = _get_or_create_worksheet(RATES_SHEET)
    body = [["Valuta","Kurs"]]
    for k in ["USD","NOK","CAD","EUR","SEK"]:
        v = d.get(k, DEFAULT_RATES.get(k, 1.0))
        body.append([k, str(v)])
    with_backoff(ws.clear)
    with_backoff(ws.update, body)

def fetch_live_rates() -> dict:
    # Via Yahoo FX (kan fallera — då lämnar vi orört)
    pairs = {"USD":"USDSEK=X", "EUR":"EURSEK=X", "NOK":"NOKSEK=X", "CAD":"CADSEK=X"}
    out = read_rates()
    for k, sym in pairs.items():
        try:
            t = yf.Ticker(sym)
            px = t.info.get("regularMarketPrice")
            if not px:
                h = t.history(period="1d")
                if not h.empty and "Close" in h:
                    px = float(h["Close"].iloc[-1])
            if px:
                out[k] = float(px)
        except Exception:
            pass
    out["SEK"] = 1.0
    return out
