# -*- coding: utf-8 -*-
import requests, streamlit as st
from functools import lru_cache

FMP_BASE = st.secrets.get("FMP_BASE", "https://financialmodelingprep.com")
FMP_KEY  = st.secrets.get("FMP_API_KEY", "")

def _fmp_get(path: str, params=None):
    params = (params or {}).copy()
    if FMP_KEY:
        params["apikey"] = FMP_KEY
    url = f"{FMP_BASE}/{path}"
    try:
        r = requests.get(url, params=params, timeout=20)
        if r.status_code == 200:
            return r.json(), 200
        return None, r.status_code
    except Exception:
        return None, 0

def fmp_ratios_ttm(ticker: str) -> float:
    if not FMP_KEY: return 0.0
    j, sc = _fmp_get(f"api/v3/ratios-ttm/{ticker}")
    if isinstance(j, list) and j:
        v = j[0].get("priceToSalesTTM") or j[0].get("priceToSalesRatioTTM")
        try:
            return float(v or 0.0)
        except Exception:
            return 0.0
    return 0.0

def fmp_ratios_quarterly(ticker: str) -> dict:
    """
    Returnerar {1: ps_Q1, 2: ps_Q2, 3: ps_Q3, 4: ps_Q4} om tillgängligt.
    """
    out = {}
    if not FMP_KEY: return out
    j, sc = _fmp_get(f"api/v3/ratios/{ticker}", {"period": "quarter", "limit": 4})
    if isinstance(j, list) and j:
        for i, row in enumerate(j[:4], start=1):
            v = row.get("priceToSalesRatio")
            try:
                if v: out[i] = float(v)
            except Exception:
                pass
    return out

@lru_cache(maxsize=256)
def fx_rate_cached(base: str, quote: str) -> float:
    base = (base or "").upper(); quote = (quote or "").upper()
    if not base or not quote or base == quote:
        return 1.0
    try:
        r = requests.get("https://api.frankfurter.app/latest", params={"from": base, "to": quote}, timeout=12)
        if r.status_code == 200:
            return float((r.json() or {}).get("rates", {}).get(quote, 0.0) or 0.0)
    except Exception:
        pass
    try:
        r = requests.get("https://api.exchangerate.host/latest", params={"base": base, "symbols": quote}, timeout=12)
        if r.status_code == 200:
            return float((r.json() or {}).get("rates", {}).get(quote, 0.0) or 0.0)
    except Exception:
        pass
    return 0.0
