# -*- coding: utf-8 -*-
"""
stockapp.fetchers.sec
---------------------
Hämtar utvalda fält från SEC/EDGAR Company Facts API.

Offentliga funktioner:
- get_all(ticker: str) -> dict

Returnerar ett dict med nycklar där data fanns, t.ex.:
{
  "Kassa (M)": 1234.5,
  "Utestående aktier (milj.)": 2500.0,
  "Net debt / EBITDA": 1.8
}

Kräver nätverkstillgång och en giltig SEC_USER_AGENT i st.secrets.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, List
import time
import math
import requests
import streamlit as st

# --------------------------------------------
# SEC headers & enkel backoff
# --------------------------------------------
def _sec_headers() -> Dict[str, str]:
    # SEC kräver tydlig UA med kontaktuppgift
    ua = st.secrets.get("SEC_USER_AGENT", "").strip()
    if not ua:
        # Fallback – funkar ofta, men sätt gärna SEC_USER_AGENT i secrets!
        ua = "StockApp/1.0 (contact: please-set-SEC_USER_AGENT-in-secrets@example.com)"
    return {
        "User-Agent": ua,
        "Accept": "application/json",
    }

def _get_json(url: str, params: Optional[Dict[str, Any]] = None, tries: int = 3, sleep_s: float = 0.8) -> Optional[Dict[str, Any]]:
    headers = _sec_headers()
    for i in range(tries):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=20)
            if r.status_code == 200:
                return r.json()
            # 429/403 etc – vänta och försök igen
            time.sleep(sleep_s * (i + 1))
        except Exception:
            time.sleep(sleep_s * (i + 1))
    return None

# --------------------------------------------
# Ticker → CIK
# --------------------------------------------
_TICKER_MAP_CACHE: Dict[str, str] = {}

def _load_ticker_map() -> None:
    global _TICKER_MAP_CACHE
    if _TICKER_MAP_CACHE:
        return
    j = _get_json("https://www.sec.gov/files/company_tickers.json")
    if not j:
        return
    # Kan komma som { "0": {...}, "1": {...} } eller som lista
    mapping: Dict[str, str] = {}
    if isinstance(j, dict) and "0" in j:
        for _, row in j.items():
            t = str(row.get("ticker", "")).upper().strip()
            cik = str(row.get("cik_str", "")).strip()
            if t and cik:
                mapping[t] = cik
    elif isinstance(j, list):
        for row in j:
            t = str(row.get("ticker", "")).upper().strip()
            cik = str(row.get("cik_str", "")).strip()
            if t and cik:
                mapping[t] = cik
    _TICKER_MAP_CACHE = mapping

def _ticker_to_cik(ticker: str) -> Optional[str]:
    _load_ticker_map()
    t = str(ticker).upper().strip()
    cik = _TICKER_MAP_CACHE.get(t, "")
    if not cik:
        return None
    # SEC kräver 10-siffrig CIK i URL
    return cik.zfill(10)

# --------------------------------------------
# Company Facts helpers
# --------------------------------------------
def _company_facts(cik10: str) -> Optional[Dict[str, Any]]:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"
    return _get_json(url)

def _latest_fact(facts: Dict[str, Any], taxonomy: str, tag: str, prefer_units: Optional[List[str]] = None) -> Optional[Tuple[float, str]]:
    """
    Returnerar (värde, unit) för senaste datapunkt som hittas för (taxonomy:tag).
    Om prefer_units anges, försöker välja en av dessa.
    """
    try:
        data = (facts.get("facts") or {}).get(taxonomy, {}).get(tag, {})
        units = data.get("units") or {}
        if not units:
            return None
        # välj unit
        keys = list(units.keys())
        unit_key = None
        if prefer_units:
            for u in prefer_units:
                if u in units:
                    unit_key = u
                    break
        if unit_key is None:
            unit_key = keys[0]
        arr = units.get(unit_key, [])
        if not arr:
            return None
        # sortera på 'end' datum om finns, annars ta sista
        def _end_ts(x):
            e = x.get("end") or x.get("fy") or ""
            return e
        arr_sorted = sorted(arr, key=_end_ts)
        val = arr_sorted[-1].get("val", None)
        if val is None:
            return None
        # val kan vara str/float/int
        try:
            v = float(val)
        except Exception:
            return None
        return v, unit_key
    except Exception:
        return None

def _to_millions(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    try:
        return round(float(x) / 1e6, 3)
    except Exception:
        return None

# --------------------------------------------
# Publik funktion
# --------------------------------------------
def get_all(ticker: str) -> Dict[str, Any]:
    """
    Hämtar ett litet urval nycklar från SEC Company Facts:
    - "Kassa (M)" (us-gaap:CashAndCashEquivalentsAtCarryingValue)
      fallback: us-gaap:CashAndCashEquivalentsPeriodEnd
    - "Utestående aktier (milj.)" (us-gaap:CommonStockSharesOutstanding)
    - "Net debt / EBITDA" om (us-gaap:NetDebtToEBITDA) finns
    Kan utökas senare med fler taggar.
    """
    out: Dict[str, Any] = {}

    cik10 = _ticker_to_cik(ticker)
    if not cik10:
        return out

    facts = _company_facts(cik10)
    if not facts:
        return out

    # Kassa
    cash = None
    for tag in ["CashAndCashEquivalentsAtCarryingValue", "CashAndCashEquivalentsPeriodEnd"]:
        r = _latest_fact(facts, "us-gaap", tag, prefer_units=["USD"])
        if r:
            cash = _to_millions(r[0])
            break
    if cash is not None and cash > 0:
        out["Kassa (M)"] = cash

    # Utestående aktier
    shares = _latest_fact(facts, "us-gaap", "CommonStockSharesOutstanding", prefer_units=["shares"])
    if shares and shares[0] > 0:
        out["Utestående aktier (milj.)"] = round(float(shares[0]) / 1e6, 3)

    # Net debt / EBITDA – direkt tag om finns
    nde = _latest_fact(facts, "us-gaap", "NetDebtToEBITDA", prefer_units=None)
    if nde and nde[0] is not None and not math.isnan(float(nde[0])):
        out["Net debt / EBITDA"] = round(float(nde[0]), 3)

    return out
