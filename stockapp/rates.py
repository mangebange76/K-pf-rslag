# stockapp/rates.py
from __future__ import annotations

import math
import time
from typing import Dict, Tuple

import pandas as pd
import streamlit as st

# HTTP och sista fallback via yfinance
import requests

try:
    import yfinance as yf  # används bara som sista fallback
except Exception:
    yf = None  # type: ignore

from .sheets import get_ws, ws_read_df, save_dataframe

# ----------------------------- Konstanter -----------------------------

DEFAULT_RATES: Dict[str, float] = {
    "USD": 10.00,
    "NOK": 1.00,
    "CAD": 7.50,
    "EUR": 11.00,
    "SEK": 1.00,
}

RATES_SHEET_NAME = st.secrets.get("RATES_SHEET_NAME", "Valutakurser")

CURRENCIES = ("USD", "NOK", "CAD", "EUR")  # alltid till SEK


# ----------------------------- Hjälpare ------------------------------

def _round6(x: float) -> float:
    try:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return 0.0
        return float(f"{float(x):.6f}")
    except Exception:
        return 0.0


def _normalize_rates(d: Dict[str, float]) -> Dict[str, float]:
    out = {k: _round6(float(v)) for k, v in d.items() if k}
    out["SEK"] = 1.0
    return out


# --------------------------- Läs/Spara i Sheet -----------------------

def _ensure_rates_sheet() -> None:
    """Skapar bladet 'Valutakurser' om det saknas."""
    try:
        ws = get_ws(worksheet_name=RATES_SHEET_NAME)
        df = ws_read_df(ws)
        if df is None or df.empty:
            save_rates(DEFAULT_RATES)
    except Exception:
        # Skapa nytt blad med defaultvärden
        save_rates(DEFAULT_RATES)


def read_rates() -> Dict[str, float]:
    """Läser kurser från Google Sheet. Skapar bladet vid behov."""
    try:
        _ensure_rates_sheet()
        ws = get_ws(worksheet_name=RATES_SHEET_NAME)
        df = ws_read_df(ws)
        rates: Dict[str, float] = {}
        if isinstance(df, pd.DataFrame) and not df.empty:
            # Stöd både "Valuta/Kurs" och eventuellt "Currency/Rate"
            val_col = "Valuta" if "Valuta" in df.columns else ("Currency" if "Currency" in df.columns else None)
            kurs_col = "Kurs" if "Kurs" in df.columns else ("Rate" if "Rate" in df.columns else None)
            if val_col and kurs_col:
                for _, r in df.iterrows():
                    k = str(r.get(val_col, "")).upper().strip()
                    v = str(r.get(kurs_col, "")).replace(",", ".").strip()
                    try:
                        rates[k] = float(v)
                    except Exception:
                        pass
        # Fyll upp med default om något saknas
        for c in (*CURRENCIES, "SEK"):
            rates.setdefault(c, DEFAULT_RATES[c])
        return _normalize_rates(rates)
    except Exception:
        return DEFAULT_RATES.copy()


def save_rates(rates: Dict[str, float]) -> None:
    """Sparar kurser till Google Sheet."""
    try:
        norm = _normalize_rates(rates)
        df = pd.DataFrame(
            {"Valuta": ["USD", "NOK", "CAD", "EUR", "SEK"],
             "Kurs":   [norm["USD"], norm["NOK"], norm["CAD"], norm["EUR"], 1.0]}
        )
        save_dataframe(df, worksheet_name=RATES_SHEET_NAME)
    except Exception as e:
        # Vi låter appen visa ev. fel i sidopanelen; här tyst fail
        raise e


# ----------------------------- Live-källor ---------------------------

def _fetch_frankfurter() -> Tuple[Dict[str, float], str]:
    """
    Frankfurter (ECB-data). Ingen nyckel behövs.
    Exempel: https://api.frankfurter.app/latest?from=USD&to=SEK
    """
    base_url = "https://api.frankfurter.app/latest"
    out: Dict[str, float] = {}
    for cur in CURRENCIES:
        params = {"from": cur, "to": "SEK"}
        r = requests.get(base_url, params=params, timeout=10)
        r.raise_for_status()
        js = r.json()
        rate = float(js.get("rates", {}).get("SEK", 0.0))
        if rate <= 0:
            raise ValueError(f"Frankfurter gav 0 för {cur}/SEK")
        out[cur] = rate
    return _normalize_rates(out), "frankfurter.app"


def _fetch_erapi() -> Tuple[Dict[str, float], str]:
    """
    ER-API (gratis, ingen nyckel). Ex: https://open.er-api.com/v6/latest/USD
    """
    base_url = "https://open.er-api.com/v6/latest/"
    out: Dict[str, float] = {}
    for cur in CURRENCIES:
        r = requests.get(base_url + cur, timeout=10)
        r.raise_for_status()
        js = r.json()
        if js.get("result") != "success":
            raise ValueError(f"ER-API fel för {cur}")
        rate = float(js.get("rates", {}).get("SEK", 0.0))
        if rate <= 0:
            raise ValueError(f"ER-API gav 0 för {cur}/SEK")
        out[cur] = rate
    return _normalize_rates(out), "open.er-api.com"


def _fetch_yfinance() -> Tuple[Dict[str, float], str]:
    """
    Sista fallback via yfinance på valutapar: USDSEK=X, NOKSEK=X, CADSEK=X, EURSEK=X
    Hämtar senaste stängning om realtidspris saknas.
    """
    if yf is None:
        raise RuntimeError("yfinance ej installerat")

    pairs = {
        "USD": "USDSEK=X",
        "NOK": "NOKSEK=X",
        "CAD": "CADSEK=X",
        "EUR": "EURSEK=X",
    }
    out: Dict[str, float] = {}
    for cur, ticker in pairs.items():
        t = yf.Ticker(ticker)
        price = None
        try:
            info = t.fast_info if hasattr(t, "fast_info") else {}
            price = float(getattr(info, "last_price", None) or info.get("last_price") or 0.0)
        except Exception:
            price = None
        if not price:
            hist = t.history(period="5d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
        if not price or price <= 0:
            raise ValueError(f"yfinance pris saknas för {ticker}")
        out[cur] = price
        time.sleep(0.2)  # snäll paus
    return _normalize_rates(out), "yfinance"
    

# ----------------------------- Publik API ----------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_live_rates() -> Dict[str, float]:
    """
    Försök i ordning: Frankfurter → ER-API → yfinance.
    Cache: 1 h. Returnerar dict med USD/NOK/CAD/EUR/SEK.
    Kastar Exception om alla källor faller.
    """
    errors = []
    for fn in (_fetch_frankfurter, _fetch_erapi, _fetch_yfinance):
        try:
            data, src = fn()
            # Lägg källa i session för debug/info
            st.session_state["_rates_source"] = src
            return data
        except Exception as e:
            errors.append(f"{fn.__name__}: {e}")
            continue
    raise RuntimeError("Kunde inte hämta livekurser från någon källa:\n" + "\n".join(errors))
