# stockapp/rates.py
# -*- coding: utf-8 -*-
"""
Valutahantering:

Publika funktioner:
- las_sparade_valutakurser() -> Dict[str, float]
- spara_valutakurser(rates: Dict[str, float]) -> None
- hamta_valutakurser_auto() -> Tuple[Dict[str, float], List[str], str]
- hamta_valutakurs(valuta: str, user_rates: Dict[str, float]) -> float

Implementation:
- Lagring i ett separat blad (RATES_SHEET_NAME) i samma Google Sheet.
- Hämtning i ordning: FMP (om API-nyckel) → Frankfurter (ECB) → exchangerate.host.
- Fyller alltid luckor med sparade värden eller STANDARD_VALUTAKURSER.
"""

from __future__ import annotations
from typing import Dict, Tuple, List

import streamlit as st
import requests

from .config import STANDARD_VALUTAKURSER, RATES_SHEET_NAME
from .sheets import get_ws
from .utils import with_backoff, safe_float


# ---------------------------------------------------------------------
# Internt: hämta/skapande av rate-worksheet
# ---------------------------------------------------------------------
def _get_rates_ws():
    """
    Hämtar/Skapar arbetsbladet för valutakurser.
    """
    try:
        return get_ws(RATES_SHEET_NAME, create=True, rows=50, cols=5)
    except Exception as e:  # pragma: no cover
        # Ge tydligare fel i UI
        raise RuntimeError(f"Kunde inte öppna/skapa bladet '{RATES_SHEET_NAME}': {e}") from e


# ---------------------------------------------------------------------
# Publika API-funktioner
# ---------------------------------------------------------------------
def las_sparade_valutakurser() -> Dict[str, float]:
    """
    Läser sparade kurser från RATES_SHEET_NAME.
    Returnerar dict {"USD": 10.55, "EUR": 11.20, "NOK": 1.05, "CAD": 7.25, "SEK": 1.0}
    Luckor fylls från STANDARD_VALUTAKURSER.
    """
    out: Dict[str, float] = {}
    try:
        ws = _get_rates_ws()
        rows = with_backoff(ws.get_all_records)  # [{'Valuta':'USD','Kurs':'10.23'}, ...]
        for r in rows:
            cur = str(r.get("Valuta", "")).upper().strip()
            val = safe_float(r.get("Kurs", ""), default=None)
            if cur and val is not None:
                out[cur] = float(val)
    except Exception:
        # ignore och gå vidare med defaults
        pass

    # Fyll luckor med standard (och tvinga SEK=1.0)
    for k, v in STANDARD_VALUTAKURSER.items():
        out.setdefault(k, float(v))
    out["SEK"] = 1.0
    return out


def spara_valutakurser(rates: Dict[str, float]) -> None:
    """
    Skriver en liten tabell till RATES_SHEET_NAME i ordning: USD, NOK, CAD, EUR, SEK.
    """
    ws = _get_rates_ws()
    body = [["Valuta", "Kurs"]]
    order = ["USD", "NOK", "CAD", "EUR", "SEK"]
    for k in order:
        v = safe_float(rates.get(k, STANDARD_VALUTAKURSER.get(k, 1.0)), default=1.0)
        # SEK ska alltid vara 1.0
        if k == "SEK":
            v = 1.0
        body.append([k, str(v)])
    with_backoff(ws.clear)
    with_backoff(ws.update, body)


def hamta_valutakurser_auto() -> Tuple[Dict[str, float], List[str], str]:
    """
    Försöker hämta USD/EUR/CAD/NOK → SEK.
    Ordning: 1) FMP (om API-nyckel) → 2) Frankfurter → 3) exchangerate.host
    Returnerar (rates, misses, provider) där provider är namnet på sista källan som gav värden.
    """
    misses: List[str] = []
    rates: Dict[str, float] = {}
    provider: str = "okänd"

    # 1) Financial Modeling Prep
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
                px = j.get("price")
                return (float(px) if px is not None else None, 200)

            provider = "FMP"
            for pair in ("USDSEK", "NOKSEK", "CADSEK", "EURSEK"):
                v, sc = _pair(pair)
                if v and v > 0:
                    base_ccy = pair[:3]
                    rates[base_ccy] = float(v)
                else:
                    misses.append(f"{pair} (HTTP {sc if sc else '??'})")
        except Exception:
            # fortsätt
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

    # SEK ska vara 1.0
    rates["SEK"] = 1.0

    return rates, misses, provider


def hamta_valutakurs(valuta: str, user_rates: Dict[str, float]) -> float:
    """
    Hämtar kursen från user_rates eller STANDARD_VALUTAKURSER.
    Tom/None valuta => 1.0.
    """
    if not valuta:
        return 1.0
    code = str(valuta).upper().strip()
    return float(user_rates.get(code, STANDARD_VALUTAKURSER.get(code, 1.0)))
