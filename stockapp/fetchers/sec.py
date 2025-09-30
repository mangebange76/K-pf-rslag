# -*- coding: utf-8 -*-
"""
SEC fetcher – kompletterar främst US-bolag:
- Hämtar companyfacts (XBRL) och plockar ut:
  - Senaste 'CommonStockSharesOutstanding'
  - Revenue TTM (grovt summerad över 4 kvartal)
Returnerar (data, facts, log).
Kräver st.secrets["SEC_EMAIL"] för User-Agent.
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
import re
import time
import requests
import streamlit as st

UA = f"StockApp/1.0 (+{st.secrets.get('SEC_EMAIL','unknown@example.com')})"

def _get(url: str, timeout: int = 20):
    headers = {"User-Agent": UA, "Accept-Encoding": "gzip, deflate"}
    r = requests.get(url, timeout=timeout, headers=headers)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} for {url}")
    return r.json() or {}

def _to_cik(ticker: str) -> Optional[str]:
    """
    Hämta CIK via SECs statiska lista (en ganska stor JSON).
    Vi cachear hårt i runtime via st.session_state för att slippa upprepa hämtning.
    """
    t = ticker.upper().replace(".", "-")
    cache_key = "_sec_ticker_map"
    if cache_key not in st.session_state:
        try:
            js = _get("https://www.sec.gov/files/company_tickers.json")
            # format: { "0":{"cik_str":..., "ticker":"A", "title":"Agilent..."}, ... }
            mp = {}
            for _, v in js.items():
                tic = str(v.get("ticker","")).upper()
                cik = str(v.get("cik_str","")).zfill(10)
                if tic:
                    mp[tic] = cik
            st.session_state[cache_key] = mp
        except Exception:
            st.session_state[cache_key] = {}
    return st.session_state[cache_key].get(t)

def _pick_us_gaap(js: Dict[str, Any], tag: str) -> Optional[List[Dict[str, Any]]]:
    """
    Plocka en lista med fakta-poster för us-gaap:<tag>.
    """
    facts = js.get("facts", {}).get("us-gaap", {})
    node = facts.get(tag)
    if not node:
        return None
    return node.get("units", {}).get("USD") or node.get("units", {}).get("shares")

def fetch_sec(ticker: str) -> Tuple[Dict[str, Any], Dict[str, str], List[str]]:
    data: Dict[str, Any]  = {}
    facts: Dict[str, str] = {}
    log:  List[str]       = []

    if not st.secrets.get("SEC_EMAIL"):
        log.append("SEC: Ingen SEC_EMAIL i secrets – hoppar över.")
        return data, facts, log

    t0 = time.time()

    # Bara relevanta för US-tickers (heuristik: saknar .ST, .OL, .TO etc)
    if "." in ticker and not ticker.upper().endswith(".US"):
        log.append("SEC: Ticker verkar inte vara US – hoppar över.")
        return data, facts, log

    try:
        cik = _to_cik(ticker)
        if not cik:
            log.append("SEC: Hittade ingen CIK för tickern.")
            return data, facts, log

        comp = _get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json")

        # Shares outstanding (senaste)
        shares_list = _pick_us_gaap(comp, "CommonStockSharesOutstanding")
        if shares_list:
            # välj senaste 10-Q/10-K
            shares_list = sorted(shares_list, key=lambda x: x.get("end", "") or "", reverse=True)
            for it in shares_list:
                val = it.get("val")
                if isinstance(val, (int, float)) and val > 0:
                    data["Utestående aktier (milj.) – SEC"] = float(val) / 1e6
                    facts["Utestående aktier (milj.) – SEC"] = "SEC/companyfacts"
                    break

        # Revenue TTM (summerat 4 senaste kvartal med 'Revenues' eller 'SalesRevenueNet')
        for tag in ("Revenues", "SalesRevenueNet"):
            arr = _pick_us_gaap(comp, tag)
            if not arr:
                continue
            # ta kvartalsposter
            q = [a for a in arr if str(a.get("fp","")).upper() in ("Q1","Q2","Q3","Q4")]
            q = sorted(q, key=lambda x: x.get("end",""), reverse=True)
            # summera 4 senaste kvartal
            if len(q) >= 4:
                ttm = 0.0
                cnt = 0
                for it in q[:4]:
                    v = it.get("val")
                    if isinstance(v, (int, float)):
                        ttm += float(v); cnt += 1
                if cnt == 4 and ttm > 0:
                    data["Omsättning TTM (SEC)"] = float(ttm)
                    facts["Omsättning TTM (SEC)"] = "SEC/companyfacts"
                    break
    except Exception as e:
        log.append(f"SEC: {e}")

    log.append(f"SEC klart på {time.time() - t0:.2f}s")
    return data, facts, log
