# stockapp/rates.py
# -*- coding: utf-8 -*-
"""
Valutafunktioner:
- las_sparade_valutakurser()
- spara_valutakurser(rates)
- hamta_valutakurser_auto()    (FMP -> Frankfurter -> exchangerate.host)
- hamta_valutakurs(valuta, user_rates)

Robust hantering av Google Sheets-bladet för valutakurser:
- Hittar blad case-insensitivt (trim)
- Skapar blad om det saknas (med header)
- Backoff på alla API-anrop
- Faller tillbaka till STANDARD_VALUTAKURSER och visar varning istället för att krascha
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

def _normalize_title(s: str) -> str:
    return (s or "").strip().lower()

def _find_ws_loose(ss: gspread.Spreadsheet, title: str):
    """Hitta worksheet genom case-insensitive + trim-match."""
    want = _normalize_title(title)
    for ws in with_backoff(ss.worksheets):
        if _normalize_title(ws.title) == want:
            return ws
    return None

def _ensure_header(ws: gspread.Worksheet):
    """Säkerställ rubriker i valutabladet."""
    try:
        rows = with_backoff(ws.get_all_values)
        if not rows or not rows[0] or rows[0][:2] != ["Valuta", "Kurs"]:
            with_backoff(ws.clear)
            with_backoff(ws.update, [["Valuta", "Kurs"]])
    except Exception:
        # Sista utväg: försök skriva header ändå.
        try:
            with_backoff(ws.update, [["Valuta", "Kurs"]])
        except Exception:
            pass

def _get_rates_ws():
    """
    Försöker öppna (eller skapa) valutabladet.
    Returnerar worksheet-objekt eller None vid fel (utan att krascha appen).
    """
    title = RATES_SHEET_NAME or "Valutakurser"
    try:
        ss = _get_spreadsheet()
    except Exception as e:
        st.warning(f"⚠️ Kunde inte öppna Google Sheet (kontrollera SHEET_URL & behörighet): {e}")
        return None

    # 1) Direkt exakt match
    try:
        return with_backoff(ss.worksheet, title)
    except Exception:
        pass

    # 2) Lös match (case-insensitiv + trim)
    try:
        ws = _find_ws_loose(ss, title)
        if ws:
            _ensure_header(ws)
            return ws
    except Exception:
        pass

    # 3) Skapa bladet
    try:
        ws = with_backoff(ss.add_worksheet, title=title, rows=50, cols=5)
        _ensure_header(ws)
        return ws
    except Exception:
        # Kan hända om det redan finns ett blad med samma namn (race condition)
        try:
            ws = with_backoff(ss.worksheet, title)
            _ensure_header(ws)
            return ws
        except Exception as e:
            st.warning(f"⚠️ Kunde inte öppna eller skapa bladet '{title}': {e}")
            return None


# ---------------------------------------------------------------------
# Publika API-funktioner
# ---------------------------------------------------------------------
def las_sparade_valutakurser() -> Dict[str, float]:
    """
    Läser sparade kurser (Valutakurser-arket). Faller tillbaka på STANDARD_VALUTAKURSER.
    Returnerar dict som t.ex. {"USD": 10.55, "EUR": 11.20, ...}
    """
    ws = _get_rates_ws()
    if ws is None:
        # Falla tillbaka – visa tydlig info men krascha inte
        st.warning("Använder standardkurser då valutabladet inte kunde öppnas.")
        return {k: float(v) for k, v in STANDARD_VALUTAKURSER.items()}

    try:
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
    except Exception as e:
        st.warning(f"Kunde inte läsa valutakurser från bladet – använder standard. ({e})")
        return {k: float(v) for k, v in STANDARD_VALUTAKURSER.items()}

def spara_valutakurser(rates: Dict[str, float]) -> None:
    """
    Skriver in rad-ordning: USD, NOK, CAD, EUR, SEK
    """
    ws = _get_rates_ws()
    if ws is None:
        st.warning("Kunde inte spara kurser – valutabladet öppnades/skapades inte.")
        return

    body = [["Valuta", "Kurs"]]
    for k in ["USD", "NOK", "CAD", "EUR", "SEK"]:
        v = rates.get(k, STANDARD_VALUTAKURSER.get(k, 1.0))
        body.append([k, str(v)])

    try:
        with_backoff(ws.clear)
        with_backoff(ws.update, body)
        st.success("Valutakurser sparade.")
    except Exception as e:
        st.warning(f"Kunde inte spara valutakurser: {e}")

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
                r2 = requests.get(
                    "https://api.frankfurter.app/latest",
                    params={"from": base_ccy, "to": "SEK"},
                    timeout=12,
                )
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
                r3 = requests.get(
                    "https://api.exchangerate.host/latest",
                    params={"base": base_ccy, "symbols": "SEK"},
                    timeout=12,
                )
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
