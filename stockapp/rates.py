from __future__ import annotations
from typing import Dict
import requests
import streamlit as st
import pandas as pd
from .sheets import ws_read_df, ws_write_df

RATES_SHEET = "Valutakurser"
DEFAULT_RATES: Dict[str, float] = {"USD": 10.0, "NOK": 1.0, "CAD": 7.5, "EUR": 11.0, "SEK": 1.0}

def _ensure_rates_sheet():
    try:
        df = ws_read_df(RATES_SHEET)
        if df.empty or "Valuta" not in df.columns or "Kurs" not in df.columns:
            base = pd.DataFrame([["Valuta","Kurs"]], columns=["Valuta","Kurs"]).iloc[0:0]
            ws_write_df(RATES_SHEET, pd.DataFrame([["Valuta","Kurs"]], columns=["Valuta","Kurs"]).iloc[0:0])
            ws_write_df(RATES_SHEET, pd.DataFrame([["USD", DEFAULT_RATES["USD"]],
                                                   ["NOK", DEFAULT_RATES["NOK"]],
                                                   ["CAD", DEFAULT_RATES["CAD"]],
                                                   ["EUR", DEFAULT_RATES["EUR"]],
                                                   ["SEK", 1.0]], columns=["Valuta","Kurs"]))
    except Exception:
        pass

@st.cache_data(ttl=600, show_spinner=False)
def read_rates() -> Dict[str, float]:
    _ensure_rates_sheet()
    try:
        df = ws_read_df(RATES_SHEET)
        if df.empty: return DEFAULT_RATES.copy()
        out: Dict[str, float] = {}
        for _, r in df.iterrows():
            cur = str(r.get("Valuta","")).upper().strip()
            val = str(r.get("Kurs","")).strip().replace(",",".")
            try:
                out[cur] = float(val)
            except Exception:
                continue
        for k,v in DEFAULT_RATES.items():
            out.setdefault(k, v)
        return out
    except Exception:
        return DEFAULT_RATES.copy()

def save_rates(d: Dict[str, float]):
    rows = []
    for k in ["USD","NOK","CAD","EUR","SEK"]:
        rows.append([k, float(d.get(k, DEFAULT_RATES.get(k,1.0)))])
    ws_write_df(RATES_SHEET, pd.DataFrame(rows, columns=["Valuta","Kurs"]))

def fetch_live_rates() -> Dict[str, float]:
    """
    Hämtar SEK-kurser från exchangerate.host.
    """
    url = "https://api.exchangerate.host/latest?base=SEK&symbols=USD,NOK,CAD,EUR,SEK"
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()
        rates = data.get("rates", {})
        # vi vill ha X→SEK, men base=SEK returnerar SEK→X; invertera
        out = {}
        for c in ["USD","NOK","CAD","EUR","SEK"]:
            v = float(rates.get(c, 0.0))
            out[c] = (1.0 / v) if v > 0 else DEFAULT_RATES.get(c, 1.0)
        out["SEK"] = 1.0
        return out
    except Exception:
        return DEFAULT_RATES.copy()
