# stockapp/rates.py
# -*- coding: utf-8 -*-
"""
Valutafunktioner:
- las_sparade_valutakurser()        -> Dict[str, float]
- spara_valutakurser(rates)         -> None
- hamta_valutakurser_auto()         -> (rates: Dict[str, float], misses: List[str], provider: str)
- hamta_valutakurs(valuta, user_rates) -> float

Källor (fallback-ordning): FMP -> Frankfurter -> exchangerate.host.
Lagring i separat blad (RATES_SHEET_NAME) i samma Google Sheet.
"""

from __future__ import annotations
from typing import Dict, List, Tuple

import requests
import pandas as pd
import streamlit as st

from .config import STANDARD_VALUTAKURSER, RATES_SHEET_NAME
from .sheets import get_ws, ws_read_df, ws_write_df, ensure_headers


# ---------------------------------------------------------------------
# Hjälpare (lokalt)
# ---------------------------------------------------------------------
_CCYS = ["USD", "EUR", "CAD", "NOK", "SEK"]  # SEK behövs för komplett mapping


def _empty_rates() -> Dict[str, float]:
    return {k: float(STANDARD_VALUTAKURSER.get(k, 1.0)) for k in _CCYS}


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    # Säkerställ headrar
    df = df.copy()
    if not {"Valuta", "Kurs"}.issubset(df.columns):
        # Försök autodetektera första två kolumner
        cols = list(df.columns)
        if len(cols) >= 2:
            df = df.rename(columns={cols[0]: "Valuta", cols[1]: "Kurs"})
        else:
            df = pd.DataFrame(columns=["Valuta", "Kurs"])
    return df[["Valuta", "Kurs"]]


# ---------------------------------------------------------------------
# Publikt API
# ---------------------------------------------------------------------
def las_sparade_valutakurser() -> Dict[str, float]:
    """
    Läser valutor från bladet RATES_SHEET_NAME.
    Fyller ut med STANDARD_VALUTAKURSER om något saknas.
    """
    try:
        ws = get_ws(RATES_SHEET_NAME)
        df = ws_read_df(ws)
        df = _normalize_df(df)
        out: Dict[str, float] = {}
        for _, r in df.iterrows():
            cur = str(r.get("Valuta", "")).strip().upper()
            try:
                val = float(str(r.get("Kurs", "")).replace(",", "."))
            except Exception:
                continue
            if cur:
                out[cur] = val

        # fyll luckor
        for k in _CCYS:
            out.setdefault(k, float(STANDARD_VALUTAKURSER.get(k, 1.0)))
        return out
    except Exception:
        # Vid fel, återgå till standard
        return _empty_rates()


def spara_valutakurser(rates: Dict[str, float]) -> None:
    """
    Skriver valutakurser i tabellform till bladet RATES_SHEET_NAME.
    Ordning: USD, EUR, CAD, NOK, SEK
    """
    rows = []
    for k in ["USD", "EUR", "CAD", "NOK", "SEK"]:
        v = float(rates.get(k, STANDARD_VALUTAKURSER.get(k, 1.0)))
        rows.append({"Valuta": k, "Kurs": v})

    df = pd.DataFrame(rows, columns=["Valuta", "Kurs"])
    ws = get_ws(RATES_SHEET_NAME, rows=max(50, len(df) + 5), cols=5)
    ensure_headers(ws, ["Valuta", "Kurs"])
    ws_write_df(ws, df)


def hamta_valutakurser_auto() -> Tuple[Dict[str, float], List[str], str]:
    """
    Försök hämta USD/EUR/CAD/NOK -> SEK via externa källor.
    Ordning: 1) FMP (om API-nyckel finns)  2) Frankfurter  3) exchangerate.host
    Returnerar: (rates, misses, provider)
    """
    rates: Dict[str, float] = {}
    misses: List[str] = []
    provider = "okänd"

    # 1) FMP
    fmp_key = st.secrets.get("FMP_API_KEY", "")
    if fmp_key:
        provider = "FMP"
        base = st.secrets.get("FMP_BASE", "https://financialmodelingprep.com")
        for base_ccy in ["USD", "EUR", "CAD", "NOK"]:
            pair = f"{base_ccy}SEK"
            try:
                url = f"{base}/api/v3/fx/{pair}"
                r = requests.get(url, params={"apikey": fmp_key}, timeout=15)
                if r.status_code == 200:
                    j = r.json() or {}
                    px = j.get("price")
                    if px is not None and float(px) > 0:
                        rates[base_ccy] = float(px)
                    else:
                        misses.append(pair)
                else:
                    misses.append(f"{pair} (HTTP {r.status_code})")
            except Exception:
                misses.append(pair)

    # 2) Frankfurter (ECB)
    if len(rates) < 4:
        provider = "Frankfurter"
        for base_ccy in ["USD", "EUR", "CAD", "NOK"]:
            if base_ccy in rates:
                continue
            try:
                r2 = requests.get(
                    "https://api.frankfurter.app/latest",
                    params={"from": base_ccy, "to": "SEK"},
                    timeout=12,
                )
                if r2.status_code == 200:
                    v = (r2.json() or {}).get("rates", {}).get("SEK")
                    if v:
                        rates[base_ccy] = float(v)
                else:
                    misses.append(f"{base_ccy} (HTTP {r2.status_code})")
            except Exception:
                misses.append(base_ccy)

    # 3) exchangerate.host
    if len(rates) < 4:
        provider = "exchangerate.host"
        for base_ccy in ["USD", "EUR", "CAD", "NOK"]:
            if base_ccy in rates:
                continue
            try:
                r3 = requests.get(
                    "https://api.exchangerate.host/latest",
                    params={"base": base_ccy, "symbols": "SEK"},
                    timeout=12,
                )
                if r3.status_code == 200:
                    v = (r3.json() or {}).get("rates", {}).get("SEK")
                    if v:
                        rates[base_ccy] = float(v)
                else:
                    misses.append(f"{base_ccy} (HTTP {r3.status_code})")
            except Exception:
                misses.append(base_ccy)

    # fyll luckor från sparade/standard
    saved = las_sparade_valutakurser()
    for k in _CCYS:
        if k not in rates:
            rates[k] = float(saved.get(k, STANDARD_VALUTAKURSER.get(k, 1.0)))

    return rates, misses, provider


def hamta_valutakurs(valuta: str, user_rates: Dict[str, float]) -> float:
    """
    Hämta enskild valutakurs (-> SEK) från user_rates eller STANDARD_VALUTAKURSER.
    """
    if not valuta:
        return 1.0
    v = str(valuta).upper().strip()
    return float(user_rates.get(v, STANDARD_VALUTAKURSER.get(v, 1.0)))
