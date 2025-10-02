# -*- coding: utf-8 -*-
"""
stockapp.fetchers.fmp
---------------------
FinancialModelingPrep (FMP) - kostnadsfri nivå med API-nyckel.

Returnerar ett dict med källnycklar som orkestratorn kan plocka upp:
- price
- marketCap
- sharesOutstanding
- evEbitdaTTM
- dividendYield
- fcfYield
- debtToEquity
- pb
- psTTM
- sector
- industry
- currency
(+ underlag: freeCashFlowTTM, dividendsPaidTTM, revenueTTM, equity)

OBS:
- P/S per kvartal lämnas tomma här (psQ1..Q4) – bättre från Yahoo/SEC.
- FMP kräver FMP_API_KEY i st.secrets. Bas-URL är defaultad.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, List
import time
import math

import requests
import streamlit as st

# -------------------------------------------------------------------
# Konfiguration & headers
# -------------------------------------------------------------------
FMP_API_KEY: str = st.secrets.get("FMP_API_KEY", "").strip()
FMP_BASE: str = st.secrets.get("FMP_BASE", "https://financialmodelingprep.com").rstrip("/")

_HEADERS = {
    "User-Agent": st.secrets.get("FMP_USER_AGENT", "stockapp/1.0 (Streamlit)"),
    "Accept": "application/json",
}

def _have_key() -> bool:
    return bool(FMP_API_KEY)

# -------------------------------------------------------------------
# Hjälpare
# -------------------------------------------------------------------
def _to_float(x, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None or (isinstance(x, str) and not x.strip()):
            return default
        v = float(x)
        if math.isnan(v):
            return default
        return v
    except Exception:
        return default

def _get_json(path: str, params: Optional[dict] = None, max_retries: int = 3, timeout: int = 20) -> Tuple[Optional[dict | list], int]:
    """
    Robust GET mot FMP. Returnerar (json, status_code).
    Hanterar 429/5xx med exponential backoff.
    """
    if params is None:
        params = {}
    params = dict(params)
    if FMP_API_KEY:
        params["apikey"] = FMP_API_KEY

    url = f"{FMP_BASE}{path}"
    delay = 1.0
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, headers=_HEADERS, timeout=timeout)
            sc = r.status_code
            if sc == 200:
                # vissa endpoints ger [] – betrakta som tomt men OK
                try:
                    return r.json(), sc
                except Exception:
                    return None, sc
            # rate limit / servers
            if sc in (429, 500, 502, 503, 504):
                time.sleep(delay)
                delay = min(delay * 2.0, 8.0)
                continue
            # andra fel – ge upp
            return None, sc
        except Exception:
            time.sleep(delay)
            delay = min(delay * 2.0, 8.0)
    return None, 0

@st.cache_data(ttl=1800, show_spinner=False)
def _quote(symbol: str) -> Tuple[Optional[dict], int]:
    # /api/v3/quote/AAPL
    j, sc = _get_json(f"/api/v3/quote/{symbol.upper()}")
    if isinstance(j, list) and j:
        return j[0], sc
    return None, sc

@st.cache_data(ttl=1800, show_spinner=False)
def _profile(symbol: str) -> Tuple[Optional[dict], int]:
    # /api/v3/profile/AAPL
    j, sc = _get_json(f"/api/v3/profile/{symbol.upper()}")
    if isinstance(j, list) and j:
        return j[0], sc
    return None, sc

@st.cache_data(ttl=1800, show_spinner=False)
def _key_metrics_ttm(symbol: str) -> Tuple[Optional[dict], int]:
    # /api/v3/key-metrics-ttm/AAPL
    j, sc = _get_json(f"/api/v3/key-metrics-ttm/{symbol.upper()}")
    if isinstance(j, list) and j:
        return j[0], sc
    return None, sc

@st.cache_data(ttl=1800, show_spinner=False)
def _income_q(symbol: str, limit: int = 4) -> Tuple[Optional[List[dict]], int]:
    # /api/v3/income-statement/AAPL?period=quarter&limit=4
    j, sc = _get_json(f"/api/v3/income-statement/{symbol.upper()}", params={"period": "quarter", "limit": limit})
    if isinstance(j, list) and j:
        return j, sc
    return None, sc

@st.cache_data(ttl=1800, show_spinner=False)
def _cashflow_q(symbol: str, limit: int = 4) -> Tuple[Optional[List[dict]], int]:
    # /api/v3/cash-flow-statement/AAPL?period=quarter&limit=4
    j, sc = _get_json(f"/api/v3/cash-flow-statement/{symbol.upper()}", params={"period": "quarter", "limit": limit})
    if isinstance(j, list) and j:
        return j, sc
    return None, sc

# -------------------------------------------------------------------
# Huvud: härled fält
# -------------------------------------------------------------------
def get_all_fields(symbol: str) -> Dict[str, Any]:
    """
    Returnerar ett dict med FMP-baserade fält.
    Kräver FMP_API_KEY. Om nyckel saknas returneras {}.
    """
    out: Dict[str, Any] = {}
    sym = (symbol or "").upper().strip()
    if not sym or not _have_key():
        return out

    # 1) quote: price, marketCap, sharesOutstanding, etc.
    q, sc_q = _quote(sym)
    if q:
        pr = _to_float(q.get("price"))
        mc = _to_float(q.get("marketCap"))
        so = _to_float(q.get("sharesOutstanding"))
        if pr is not None:
            out["price"] = pr
        if mc is not None:
            out["marketCap"] = mc
        if so is not None:
            out["sharesOutstanding"] = so

        # vissa kvoter kan finnas här också
        ps_ttm = _to_float(q.get("priceToSalesRatioTTM"))
        if ps_ttm is not None:
            out["psTTM"] = ps_ttm

    # 2) profile: sector, industry, currency, dividendYield
    p, sc_p = _profile(sym)
    if p:
        sector = p.get("sector") or p.get("sectorName") or ""
        industry = p.get("industry") or p.get("industryTitle") or ""
        currency = p.get("currency") or "USD"
        if sector:
            out["sector"] = str(sector)
        if industry:
            out["industry"] = str(industry)
        if currency:
            out["currency"] = str(currency)

        dy = _to_float(p.get("lastDiv"), None)
        # profile.lastDiv är belopp; prova också dividendYieldTTM om finns
        divY = _to_float(p.get("dividendYieldTTM"), None)
        if divY is not None:
            out["dividendYield"] = float(divY * 100.0 if divY < 1 else divY)  # i %
        elif dy is not None and pr:
            out["dividendYield"] = float(dy / pr * 100.0)

        pb = _to_float(p.get("priceToBookRatioTTM"), None)
        if pb is not None:
            out["pb"] = pb

    # 3) key-metrics-ttm: EV/EBITDA (ttm), debtToEquity, payout/FCF if finns, osv.
    km, sc_km = _key_metrics_ttm(sym)
    if km:
        ev_eb = _to_float(km.get("enterpriseValueOverEBITDATTM")) or _to_float(km.get("evToEbitdaTTM"))
        if ev_eb is not None and ev_eb > 0:
            out["evEbitdaTTM"] = ev_eb

        dte = _to_float(km.get("debtToEquityTTM"))
        if dte is not None:
            out["debtToEquity"] = dte

        ps_ttm2 = _to_float(km.get("priceToSalesRatioTTM"))
        if ps_ttm2 is not None and "psTTM" not in out:
            out["psTTM"] = ps_ttm2

        pb2 = _to_float(km.get("priceToBookRatioTTM"))
        if pb2 is not None and "pb" not in out:
            out["pb"] = pb2

        # vissa konton kan finnas för FCF-yield direkt
        fcf_y = _to_float(km.get("freeCashFlowYieldTTM"))
        if fcf_y is not None:
            out["fcfYield"] = float(fcf_y * 100.0 if fcf_y < 1 else fcf_y)

    # 4) Cash flow (kvartal) – räkna FCF TTM & utdelningspayout/FCF om möjligt
    cf, sc_cf = _cashflow_q(sym, limit=4)
    fcf_ttm = None
    div_paid_ttm = None
    if cf:
        # FMP anger freeCashFlow, dividendsPaid (vanligen negativt tal)
        fcf_vals = [_to_float(row.get("freeCashFlow")) for row in cf]
        fcf_vals = [v for v in fcf_vals if v is not None]
        if fcf_vals:
            fcf_ttm = float(sum(fcf_vals))
            out["freeCashFlowTTM"] = fcf_ttm

        div_vals = [_to_float(row.get("dividendsPaid")) for row in cf]
        div_vals = [abs(v) for v in div_vals if v is not None]
        if div_vals:
            div_paid_ttm = float(sum(div_vals))
            out["dividendsPaidTTM"] = div_paid_ttm

    # 5) Income (kvartal) – revenueTTM för ev. sanity-checks
    inc, sc_inc = _income_q(sym, limit=4)
    if inc:
        rev_vals = [_to_float(row.get("revenue")) for row in inc]
        rev_vals = [v for v in rev_vals if v is not None]
        if rev_vals:
            out["revenueTTM"] = float(sum(rev_vals))

    # 6) Härled FCF Yield om vi har market cap och FCF TTM
    if "marketCap" in out and out["marketCap"] and fcf_ttm and fcf_ttm != 0:
        out["fcfYield"] = float((fcf_ttm / float(out["marketCap"])) * 100.0)

    # 7) Härled payout/FCF (%) om möjligt
    if fcf_ttm and fcf_ttm > 0 and div_paid_ttm is not None:
        out["payoutFCF"] = float(div_paid_ttm / fcf_ttm * 100.0)

    # 8) Rensa bort icke-satta fält
    clean: Dict[str, Any] = {}
    for k, v in out.items():
        if v is None:
            continue
        clean[k] = v

    return clean
