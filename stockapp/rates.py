# stockapp/rates.py
# -*- coding: utf-8 -*-
"""
Valutafunktioner:
- las_sparade_valutakurser()
- spara_valutakurser(rates)
- hamta_valutakurser_auto()    (FMP -> Frankfurter -> exchangerate.host)
- hamta_valutakurs(valuta, user_rates)
Fristående: öppnar Google Sheet själv (ingen import från storage).
"""

from __future__ import annotations
from typing import Dict, Tuple, List

import requests
import streamlit as st

# Google Sheets
import gspread
from google.oauth2.service_account import Credentials

from .config import STANDARD_VALUTAKURSER, RATES_SHEET_NAME  # t.ex. "Valutakurser"
from .utils import with_backoff


# ---------------------------------------------------------------------
# Lokal Sheets-koppling (fristående från storage)
# ---------------------------------------------------------------------
def _gspread_client():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
    return gspread.authorize(credentials)

def _get_spreadsheet():
    client = _gspread_client()
    return client.open_by_url(st.secrets["SHEET_URL"])

def _get_rates_ws():
    """
    Hämtar/Skapar arbetsbladet för valutakurser.
    """
    ss = _get_spreadsheet()
    try:
        return ss.worksheet(RATES_SHEET_NAME)
    except Exception:
        # Skapa bladet om det saknas
        ss.add_worksheet(title=RATES_SHEET_NAME, rows=50, cols=5)
        ws = ss.worksheet(RATES_SHEET_NAME)
        with_backoff(ws.update, [["Valuta", "Kurs"]])
        return ws


# ---------------------------------------------------------------------
# Publika API-funktioner
# ---------------------------------------------------------------------
def las_sparade_valutakurser() -> Dict[str, float]:
    """
    Läser sparade kurser (Valutakurser-arket). Faller tillbaka på STANDARD_VALUTAKURSER.
    Returnerar dict som t.ex. {"USD": 10.55, "EUR": 11.20, ...}
    """
    try:
        ws = _get_rates_ws()
        rows = with_backoff(ws.get_all_records)  # [{'Valuta':'USD','Kurs':'10.23'}, ...]
        out: Dict[str, float] = {}
        for r in rows:
            cur = str(r.get("Valuta", "")).upper().strip()
            val = str(r.get("Kurs", "")).replace(",", ".").strip()
            try:
                out[cur] = float(val)
            except Exception:
                pass
        # Fyll luckor med STANDARD
        for k, v in STANDARD_VALUTAKURSER.items():
            out.setdefault(k, float(v))
        return out
    except Exception:
        # Helt fel? Ta standard
        return {k: float(v) for k, v in STANDARD_VALUTAKURSER.items()}

def spara_valutakurser(rates: Dict[str, float]) -> None:
    """
    Skriver in rad-ordning: USD, NOK, CAD, EUR, SEK
    """
    ws = _get_rates_ws()
    body = [["Valuta", "Kurs"]]
    for k in ["USD", "NOK", "CAD", "EUR", "SEK"]:
        v = rates.get(k, STANDARD_VALUTAKURSER.get(k, 1.0))
        body.append([k, str(v)])
    with_backoff(ws.clear)
    with_backoff(ws.update, body)

def hamta_valutakurser_auto() -> Tuple[Dict[str, float], List[str], str]:
    """
    Försöker hämta USD/EUR/CAD/NOK -> SEK.
    Ordning: 1) FMP (om API-nyckel finns) -> 2) Frankfurter -> 3) exchangerate.host
    Returnerar (rates, misses, provider)
    """
    misses: List[str] = []
    rates: Dict[str, float] = {}
    provider: str = "okänd"

    # 1) FMP
    fmp_key = st.secrets.get("FMP_API_KEY", "")
    if fmp_key:
        try:
            base = st.secrets.get("FMP_BASE", "https://financialmodelingprep.com")
            def _pair(pair: str):
                url = f"{base}/api/v3/fx/{pair}"
                r = requests.get(url, params={"apikey": fmp_key}, timeout=15)
                if r.status_code != 200:
                    return None, r.status_code
                j = r.json() or {}
                return (float(j.get("price")) if j.get("price") is not None else None, 200)

            provider = "FMP"
            for pair in ("USDSEK", "NOKSEK", "CADSEK", "EURSEK"):
                v, sc = _pair(pair)
                if v and v > 0:
                    base_ccy = pair[:3]
                    rates[base_ccy] = float(v)
                else:
                    misses.append(f"{pair} (HTTP {sc if sc else '??'})")
        except Exception:
            pass

    # 2) Frankfurter (ECB)
    if len(rates) < 4:
        provider = "Frankfurter"
        for base_ccy in ("USD", "EUR", "CAD", "NOK"):
            try:
                r2 = requests.get("https://api.frankfurter.app/latest",
                                  params={"from": base_ccy, "to": "SEK"}, timeout=12)
                if r2.status_code == 200:
                    rr = r2.json() or {}
                    v = (rr.get("rates") or {}).get("SEK")
                    if v:
                        rates[base_ccy] = float(v)
            except Exception:
                pass

    # 3) exchangerate.host
    if len(rates) < 4:
        provider = "exchangerate.host"
        for base_ccy in ("USD", "EUR", "CAD", "NOK"):
            try:
                r3 = requests.get("https://api.exchangerate.host/latest",
                                  params={"base": base_ccy, "symbols": "SEK"}, timeout=12)
                if r3.status_code == 200:
                    v = (r3.json() or {}).get("rates", {}).get("SEK")
                    if v:
                        rates[base_ccy] = float(v)
            except Exception:
                pass

    # Fyll luckor med sparade/standard
    saved = las_sparade_valutakurser()
    for base_ccy in ("USD", "EUR", "CAD", "NOK", "SEK"):
        if base_ccy not in rates:
            rates[base_ccy] = float(saved.get(base_ccy, STANDARD_VALUTAKURSER.get(base_ccy, 1.0)))

    return rates, misses, provider

def hamta_valutakurs(valuta: str, user_rates: Dict[str, float]) -> float:
    """
    Hämtar kurs från user_rates (annars STANDARD_VALUTAKURSER).
    """
    if not valuta:
        return 1.0
    v = str(valuta).upper()
    return float(user_rates.get(v, STANDARD_VALUTAKURSER.get(v, 1.0)))
