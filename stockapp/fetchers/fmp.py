# -*- coding: utf-8 -*-
"""
stockapp.fetchers.fmp
---------------------
Hämtar nyckeltal från Financial Modeling Prep.

Publik funktion:
- get_all(ticker: str) -> dict

Behöver:
- st.secrets["FMP_API_KEY"] (obligatorisk)
Valfritt:
- st.secrets["FMP_BASE"] (default https://financialmodelingprep.com/api/v3)
"""

from __future__ import annotations
from typing import Any, Dict, Optional, List
import time
import math
import requests
import streamlit as st


def _base() -> str:
    b = str(st.secrets.get("FMP_BASE", "")).strip()
    return b or "https://financialmodelingprep.com/api/v3"


def _api_key() -> str:
    return str(st.secrets.get("FMP_API_KEY", "")).strip()


def _get(url: str, params: Optional[Dict[str, Any]] = None, tries: int = 3, sleep_s: float = 0.7):
    if params is None:
        params = {}
    ak = _api_key()
    if not ak:
        return None
    params = {**params, "apikey": ak}
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code == 200:
                return r.json()
            time.sleep(sleep_s * (i + 1))
        except Exception:
            time.sleep(sleep_s * (i + 1))
    return None


def _to_float(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _to_millions(x) -> Optional[float]:
    v = _to_float(x, None)
    if v is None:
        return None
    return round(v / 1e6, 3)


def _pick_latest(arr, *keys):
    """Plocka första nyckel som finns i dict/list-svar."""
    if not arr:
        return None
    d = arr[0] if isinstance(arr, list) else arr
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def get_all(ticker: str) -> Dict[str, Any]:
    """
    Returnerar ett dict med de fält vi hittar för tickern.
    Mappar till våra svenska kolumnnamn.
    """
    out: Dict[str, Any] = {}
    t = str(ticker or "").upper().strip()
    if not t:
        return out
    if not _api_key():
        # ingen API-nyckel => inget att göra
        return out

    base = _base()

    # --------- PROFILE (namn, sektor, industri, valuta)
    prof = _get(f"{base}/profile/{t}")
    if isinstance(prof, list) and prof:
        p0 = prof[0]
        name = p0.get("companyName") or p0.get("company-name")
        if name:
            out["Bolagsnamn"] = name
        sector = p0.get("sector") or p0.get("sectorName")
        if sector:
            out["Sektor"] = sector
        industry = p0.get("industry") or p0.get("industryName")
        if industry:
            out["Industri"] = industry
        ccy = p0.get("currency")
        if ccy:
            out["Valuta"] = ccy

    # --------- QUOTE (kurs, market cap, P/S ttm, P/B, utdelning)
    quote = _get(f"{base}/quote/{t}")
    if isinstance(quote, list) and quote:
        q0 = quote[0]
        price = _to_float(q0.get("price"))
        if price:
            out["Kurs"] = price
        mcap = _to_float(q0.get("marketCap"))
        if mcap:
            out["Market Cap"] = mcap  # i bolagets valuta
        ps_ttm = _to_float(q0.get("priceToSalesTrailing12Months"))
        if ps_ttm is not None:
            out["P/S"] = ps_ttm
        pb = _to_float(q0.get("priceToBook"))
        if pb is not None:
            out["P/B"] = pb
        dy = _to_float(q0.get("trailingAnnualDividendYield"))
        if dy is not None and dy > 0:
            out["Dividend yield (%)"] = round(dy * 100.0, 2)

        # vissa svar har sharesOutstanding här
        sh = _to_float(q0.get("sharesOutstanding"))
        if sh and sh > 0:
            out["Utestående aktier (milj.)"] = round(sh / 1e6, 3)

    # --------- RATIOS TTM (marginaler, ROE, D/E)
    ratios = _get(f"{base}/ratios-ttm/{t}")
    if isinstance(ratios, list) and ratios:
        r0 = ratios[0]
        gm = _to_float(r0.get("grossProfitMarginTTM"))
        if gm is not None:
            out["Gross margin (%)"] = round(gm * 100.0, 2)
        om = _to_float(r0.get("operatingProfitMarginTTM"))
        if om is not None:
            out["Operating margin (%)"] = round(om * 100.0, 2)
        nm = _to_float(r0.get("netProfitMarginTTM"))
        if nm is not None:
            out["Net margin (%)"] = round(nm * 100.0, 2)
        roe = _to_float(r0.get("returnOnEquityTTM"))
        if roe is not None:
            out["ROE (%)"] = round(roe * 100.0, 2)
        de = _to_float(r0.get("debtEquityRatioTTM"))
        if de is not None:
            out["Debt/Equity"] = de

        dy2 = _to_float(r0.get("dividendYieldTTM"))
        if dy2 is not None and dy2 > 0:
            out["Dividend yield (%)"] = round(dy2 * 100.0, 2)

        payout = _to_float(r0.get("payoutRatioTTM"))
        # payoutRatioTTM är ofta på EPS-basis – vi visar den som % om den är rimlig
        if payout is not None and payout >= 0:
            out["Dividend payout (FCF) (%)"] = round(payout * 100.0, 2)

    # --------- KEY METRICS TTM (EV/EBITDA, FCF Yield, P/B ibland)
    km = _get(f"{base}/key-metrics-ttm/{t}")
    if isinstance(km, list) and km:
        k0 = km[0]
        ev_eb = _to_float(k0.get("enterpriseValueOverEBITDATTM") or k0.get("enterpriseValueOverEBITDA"))
        if ev_eb is not None and ev_eb > 0:
            out["EV/EBITDA (ttm)"] = ev_eb
        fcfy = _to_float(k0.get("freeCashFlowYieldTTM"))
        if fcfy is not None:
            out["FCF Yield (%)"] = round(fcfy * 100.0, 2)
        pb2 = _to_float(k0.get("priceToBookRatioTTM"))
        if pb2 is not None:
            out["P/B"] = pb2

    # --------- Income statement (senaste års omsättning)
    inc = _get(f"{base}/income-statement/{t}", params={"limit": 1})
    if isinstance(inc, list) and inc:
        rev = _to_float(inc[0].get("revenue"))
        if rev is not None:
            out["Omsättning i år (M)"] = _to_millions(rev)

    # --------- Balance sheet (cash, shares) – ibland bättre källor än quote
    bs = _get(f"{base}/balance-sheet-statement/{t}", params={"limit": 1})
    if isinstance(bs, list) and bs:
        cash = _to_float(bs[0].get("cashAndCashEquivalents"))
        if cash is not None:
            out["Kassa (M)"] = _to_millions(cash)
        sh2 = _to_float(bs[0].get("commonStockSharesOutstanding"))
        if (sh2 is not None) and sh2 > 0:
            out["Utestående aktier (milj.)"] = round(sh2 / 1e6, 3)

    # --------- Cash flow (Free Cash Flow) – för ev. härledd FCF Yield (om market cap finns)
    cf = _get(f"{base}/cash-flow-statement/{t}", params={"limit": 1})
    if isinstance(cf, list) and cf:
        fcf = _to_float(cf[0].get("freeCashFlow"))
        if fcf is not None:
            out["FCF (M)"] = _to_millions(fcf)
            if "Market Cap" in out and out["Market Cap"]:
                try:
                    fcfy_calc = (fcf / float(out["Market Cap"])) * 100.0
                    # Om vi saknade FCF Yield från metrics kan vi härleda en grov
                    if "FCF Yield (%)" not in out and not math.isnan(fcfy_calc):
                        out["FCF Yield (%)"] = round(fcfy_calc, 2)
                except Exception:
                    pass

    # Filtrera bort None/NaN
    clean: Dict[str, Any] = {}
    for k, v in out.items():
        try:
            if v is None:
                continue
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                continue
            clean[k] = v
        except Exception:
            continue

    return clean
