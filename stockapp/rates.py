# -*- coding: utf-8 -*-
"""
Valutafunktioner (helt fristående från sheets.py):
- las_sparade_valutakurser()        -> dict
- spara_valutakurser(rates: dict)   -> None
- hamta_valutakurser_auto()         -> (rates: dict, misses: list[str], provider: str)
- hamta_valutakurs(valuta, user_rates) -> float

Läser/skriver till fliken RATES_SHEET_NAME i samma Google Sheet som portföljen,
men hanterar rubrikerna själv ("Valuta","Kurs").
"""

from __future__ import annotations
from typing import Dict, List, Tuple

import requests
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

from .config import SHEET_URL, RATES_SHEET_NAME, STANDARD_VALUTAKURSER
from .utils import with_backoff


# ---------------------------------------------------------------------
# GSpread-klient (fristående)
# ---------------------------------------------------------------------
def _client() -> gspread.Client:
    creds = Credentials.from_service_account_info(
        st.secrets["GOOGLE_CREDENTIALS"],
        scopes=[
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ],
    )
    return gspread.authorize(creds)


def _open_spreadsheet():
    if not SHEET_URL:
        raise RuntimeError("SHEET_URL saknas i secrets/config.")
    cli = _client()
    return with_backoff(cli.open_by_url, SHEET_URL)


def _get_rates_ws():
    """Hämta eller skapa valutafliken och säkerställ rubriker."""
    ss = _open_spreadsheet()
    ws = None
    try:
        ws = with_backoff(ss.worksheet, RATES_SHEET_NAME)
    except Exception:
        # Skapa om den saknas
        with_backoff(ss.add_worksheet, title=RATES_SHEET_NAME, rows=50, cols=5)
        ws = with_backoff(ss.worksheet, RATES_SHEET_NAME)

    # Säkerställ rubriker
    try:
        row1 = with_backoff(ws.row_values, 1) or []
    except Exception:
        row1 = []
    wanted = ["Valuta", "Kurs"]
    if [c.strip() for c in row1] != wanted:
        # skriv enbart rubriker om fel/inget finns
        with_backoff(ws.clear)
        with_backoff(ws.update, [wanted])

    return ws


# ---------------------------------------------------------------------
# Publika API-funktioner
# ---------------------------------------------------------------------
def las_sparade_valutakurser() -> Dict[str, float]:
    """
    Läser sparade kurser från valutafliken.
    Fyller luckor med STANDARD_VALUTAKURSER.
    """
    out: Dict[str, float] = {}
    try:
        ws = _get_rates_ws()
        rows = with_backoff(ws.get_all_values)
        if rows and len(rows) >= 2:
            for r in rows[1:]:
                if not r or len(r) < 2:
                    continue
                cur = str(r[0]).strip().upper()
                val = str(r[1]).strip().replace(",", ".")
                if not cur:
                    continue
                try:
                    out[cur] = float(val)
                except Exception:
                    pass
    except Exception as e:
        st.info(f"ℹ️ Kunde inte läsa valutabladet: {e}")

    # Fyll luckor
    for k, v in STANDARD_VALUTAKURSER.items():
        out.setdefault(k, float(v))
    # SEK ska alltid vara 1.0
    out["SEK"] = 1.0
    return out


def spara_valutakurser(rates: Dict[str, float]) -> None:
    """
    Skriver valutor i fast ordning: USD, EUR, CAD, NOK, SEK.
    Ersätter hela bladet.
    """
    ws = _get_rates_ws()
    order = ["USD", "EUR", "CAD", "NOK", "SEK"]
    body = [["Valuta", "Kurs"]]
    for k in order:
        v = rates.get(k, STANDARD_VALUTAKURSER.get(k, 1.0))
        try:
            body.append([k, float(v)])
        except Exception:
            body.append([k, v])

    with_backoff(ws.clear)
    with_backoff(ws.update, body)
    st.toast("💱 Valutakurser sparade.")


def hamta_valutakurser_auto() -> Tuple[Dict[str, float], List[str], str]:
    """
    Försöker hämta USD/EUR/CAD/NOK → SEK.
    Ordning: 1) FMP (om nyckel finns) 2) Frankfurter 3) exchangerate.host
    Returnerar (rates, misses, provider).
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
            for pair in ("USDSEK", "EURSEK", "CADSEK", "NOKSEK"):
                v, sc = _pair(pair)
                if v and v > 0:
                    rates[pair[:3]] = float(v)
                else:
                    misses.append(f"{pair} (HTTP {sc if sc else '??'})")
        except Exception:
            # gå vidare tyst
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
                    v = (r2.json() or {}).get("rates", {}).get("SEK")
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
    for k in ("USD", "EUR", "CAD", "NOK", "SEK"):
        if k not in rates:
            rates[k] = float(saved.get(k, STANDARD_VALUTAKURSER.get(k, 1.0)))

    # SEK alltid 1.0
    rates["SEK"] = 1.0

    return rates, misses, provider


def hamta_valutakurs(valuta: str, user_rates: Dict[str, float]) -> float:
    """Returnerar kurs till SEK för given valuta-kod."""
    if not valuta:
        return 1.0
    v = str(valuta).upper().strip()
    if v == "SEK":
        return 1.0
    return float(user_rates.get(v, STANDARD_VALUTAKURSER.get(v, 1.0)))
