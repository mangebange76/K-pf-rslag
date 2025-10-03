# stockapp/rates.py
from __future__ import annotations

from typing import Dict
import pandas as pd
import streamlit as st

from .sheets import get_ws, ws_read_df, ws_write_df

RATES_SHEET_NAME = st.secrets.get("RATES_SHEET_NAME", "Valutakurser")

# Standardvärden (kan ändras här eller i bladet)
DEFAULT_RATES: Dict[str, float] = {
    "USD": 9.75,
    "NOK": 0.95,
    "CAD": 7.05,
    "EUR": 11.18,
    "SEK": 1.0,
}

def _ensure_rates_sheet():
    ws = get_ws(worksheet_name=RATES_SHEET_NAME)
    df = ws_read_df(ws)
    if df.empty:
        body = [["Valuta", "Kurs"]]
        for k in ["USD", "NOK", "CAD", "EUR", "SEK"]:
            body.append([k, str(DEFAULT_RATES.get(k, 1.0))])
        ws_write_df(ws, pd.DataFrame(body[1:], columns=body[0]))
    return ws

@st.cache_data(show_spinner=False)
def read_rates_cached(nonce: int) -> Dict[str, float]:
    ws = _ensure_rates_sheet()
    df = ws_read_df(ws)
    out: Dict[str, float] = {}
    for _, r in df.iterrows():
        cur = str(r.get("Valuta", "")).upper().strip()
        val = str(r.get("Kurs", "")).replace(",", ".").strip()
        try:
            out[cur] = float(val)
        except Exception:
            pass
    # alltid åtminstone default
    for k, v in DEFAULT_RATES.items():
        out.setdefault(k, v)
    return out

def read_rates() -> Dict[str, float]:
    return read_rates_cached(st.session_state.get("rates_reload", 0))

def save_rates(rates: Dict[str, float]) -> None:
    ws = _ensure_rates_sheet()
    body = [["Valuta", "Kurs"]]
    for k in ["USD", "NOK", "CAD", "EUR", "SEK"]:
        v = rates.get(k, DEFAULT_RATES.get(k, 1.0))
        body.append([k, str(v)])
    ws_write_df(ws, pd.DataFrame(body[1:], columns=body[0]))
