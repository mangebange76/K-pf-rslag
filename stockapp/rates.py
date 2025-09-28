# -*- coding: utf-8 -*-
from __future__ import annotations

import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import requests

from .utils import with_backoff

# Egen flik för växelkurser
RATES_SHEET_NAME = "Valutakurser"

# Samma auth som i storage.py (secrets krävs)
SHEET_URL = st.secrets["SHEET_URL"]
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

STANDARD_VALUTAKURSER = {
    "USD": 9.75,
    "NOK": 0.95,
    "CAD": 7.05,
    "EUR": 11.18,
    "SEK": 1.0,
}

def _get_spreadsheet():
    return client.open_by_url(SHEET_URL)

def _rates_ws():
    ss = _get_spreadsheet()
    try:
        return ss.worksheet(RATES_SHEET_NAME)
    except Exception:
        ss.add_worksheet(title=RATES_SHEET_NAME, rows=10, cols=5)
        ws = ss.worksheet(RATES_SHEET_NAME)
        with_backoff(ws.update, [["Valuta","Kurs"]])
        return ws

def las_sparade_valutakurser() -> dict:
    try:
        ws = _rates_ws()
        rows = with_backoff(ws.get_all_records)
        out = {}
        for r in rows:
            cur = str(r.get("Valuta","")).upper().strip()
            val = str(r.get("Kurs","")).replace(",", ".").strip()
            try:
                out[cur] = float(val)
            except:
                pass
        return out
    except Exception:
        return {}

def spara_valutakurser(rates: dict):
    ws = _rates_ws()
    body = [["Valuta","Kurs"]]
    for k in ["USD","NOK","CAD","EUR","SEK"]:
        v = rates.get(k, STANDARD_VALUTAKURSER.get(k, 1.0))
        body.append([k, str(v)])
    with_backoff(ws.clear)
    with_backoff(ws.update, body)

def hamta_valutakurs(valuta: str, user_rates: dict) -> float:
    if not valuta:
        return 1.0
    return float(user_rates.get(valuta.upper(), STANDARD_VALUTAKURSER.get(valuta.upper(), 1.0)))

def hamta_valutakurser_auto():
    """
    Försöker i tur och ordning: Frankfurter (ECB) → exchangerate.host.
    Returnerar (rates, misses, provider).
    """
    misses = []
    rates = {}
    provider = None

    # Frankfurter
    provider = "Frankfurter"
    for base_ccy in ("USD","EUR","CAD","NOK"):
        try:
            r2 = requests.get("https://api.frankfurter.app/latest",
                              params={"from": base_ccy, "to": "SEK"}, timeout=12)
            if r2.status_code == 200:
                rr = r2.json() or {}
                v = (rr.get("rates") or {}).get("SEK")
                if v:
                    rates[base_ccy] = float(v)
                else:
                    misses.append(f"{base_ccy}SEK (Frankfurter)")
            else:
                misses.append(f"{base_ccy}SEK (HTTP {r2.status_code})")
        except Exception:
            misses.append(f"{base_ccy}SEK (Frankfurter fel)")

    # exchangerate.host (fyll luckor)
    if len(rates) < 4:
        provider = "exchangerate.host"
        for base_ccy in ("USD","EUR","CAD","NOK"):
            if base_ccy in rates: 
                continue
            try:
                r = requests.get("https://api.exchangerate.host/latest",
                                 params={"base": base_ccy, "symbols": "SEK"}, timeout=12)
                if r.status_code == 200:
                    v = (r.json() or {}).get("rates", {}).get("SEK")
                    if v:
                        rates[base_ccy] = float(v)
                    else:
                        misses.append(f"{base_ccy}SEK (exchangerate.host)")
                else:
                    misses.append(f"{base_ccy}SEK (HTTP {r.status_code})")
            except Exception:
                misses.append(f"{base_ccy}SEK (exchangerate.host fel)")

    # Fyll ev. luckor med sparat/standard
    saved = las_sparade_valutakurser()
    for base_ccy in ("USD","EUR","CAD","NOK"):
        if base_ccy not in rates:
            rates[base_ccy] = float(saved.get(base_ccy, STANDARD_VALUTAKURSER.get(base_ccy, 1.0)))

    return rates, misses, (provider or "okänd")
