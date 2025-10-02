# stockapp/fetchers/yahoo.py
# -*- coding: utf-8 -*-
"""
Yahoo-hämtare (utan yfinance; använder officiella web-endpoints med requests).

Publikt API:
    get_live_price(ticker) -> Optional[float]
    fetch_ticker(ticker)   -> Dict[str, Any]

Returnerar normaliserade nycklar (om tillgängliga):
    symbol                  -> "NVDA"
    name                    -> "NVIDIA Corporation"
    currency                -> "USD"
    price                   -> float
    market_cap              -> float  (i basvaluta)
    shares_outstanding      -> float  (antal)
    annual_dividend         -> float  (per aktie, basvaluta)
    dividend_yield_pct      -> float  (%)
    ev_ebitda               -> float
    gross_margin_pct        -> float
    net_margin_pct          -> float
    debt_to_equity          -> float
    sector                  -> str
    industry                -> str

OBS:
 - Yahoo throttlar ibland – vi har enkel backoff/retry.
 - Vissa fält saknas ofta; vi fyller bara det vi säkert hittar.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

import math
import time
import requests


# ------------------------------------------------------------
# Hjälpare
# ------------------------------------------------------------
def _to_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        try:
            v = float(str(x).replace(",", "."))
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        except Exception:
            return None


def _req_json(url: str, params: Dict[str, Any] | None = None, tries: int = 3, sleep_s: float = 0.6) -> Optional[Dict[str, Any]]:
    """GET JSON med enkel retry/backoff."""
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params or {}, timeout=20)
            last = r
            if r.status_code == 200:
                return r.json()
            # 4xx/5xx -> försök igen lätt
            time.sleep(sleep_s * (i + 1))
        except Exception:
            time.sleep(sleep_s * (i + 1))
    # ge upp
    return None


# ------------------------------------------------------------
# Källor
# ------------------------------------------------------------
def _yahoo_quote(ticker: str) -> Optional[Dict[str, Any]]:
    """
    quote/v7 endpoint: pris, marketCap, currency, utdelnings-yield m.m.
    """
    url = "https://query1.finance.yahoo.com/v7/finance/quote"
    js = _req_json(url, params={"symbols": ticker})
    if not js:
        return None
    try:
        res = (js.get("quoteResponse") or {}).get("result") or []
        return res[0] if res else None
    except Exception:
        return None


def _yahoo_quote_summary(ticker: str) -> Optional[Dict[str, Any]]:
    """
    quoteSummary/v10 endpoint med flera modules:
     - assetProfile (sector/industry)
     - summaryDetail (dividendRate/dividendYield)
     - financialData (margins, EV/EBITDA)
     - defaultKeyStatistics (sharesOutstanding, debtToEquity)
     - price (longName/shortName)
    """
    modules = ",".join([
        "price",
        "assetProfile",
        "summaryDetail",
        "financialData",
        "defaultKeyStatistics",
    ])
    url = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/" + ticker
    js = _req_json(url, params={"modules": modules})
    if not js:
        return None
    try:
        res = (js.get("quoteSummary") or {}).get("result") or []
        return res[0] if res else None
    except Exception:
        return None


# ------------------------------------------------------------
# Publikt API
# ------------------------------------------------------------
def get_live_price(ticker: str) -> Optional[float]:
    """
    Snabb prisfunktion: returnerar regularMarketPrice om möjligt.
    """
    q = _yahoo_quote(ticker)
    if not q:
        return None
    p = _to_float(q.get("regularMarketPrice"))
    return p


def fetch_ticker(ticker: str) -> Dict[str, Any]:
    """
    Samlar ihop nycklar från Yahoo quote + quoteSummary.
    """
    tkr = str(ticker).strip()
    out: Dict[str, Any] = {"symbol": tkr}

    # 1) Basdata/pris/marketcap
    q = _yahoo_quote(tkr)
    if q:
        out["currency"] = q.get("currency") or out.get("currency")
        p = _to_float(q.get("regularMarketPrice"))
        if p is not None:
            out["price"] = p
        mc = _to_float(q.get("marketCap"))
        if mc is not None:
            out["market_cap"] = mc
        so = _to_float(q.get("sharesOutstanding"))
        if so is not None:
            out["shares_outstanding"] = so

        # Förekommer ibland här:
        dy = _to_float(q.get("trailingAnnualDividendYield"))
        if dy is not None:
            out["dividend_yield_pct"] = float(dy) * 100.0
        dr = _to_float(q.get("trailingAnnualDividendRate"))
        if dr is not None:
            out["annual_dividend"] = dr

        nm = q.get("longName") or q.get("shortName")
        if nm:
            out["name"] = nm

    # 2) Fördjupning
    qs = _yahoo_quote_summary(tkr)
    if qs:
        # price -> longName/shortName som fallback-namn
        pr = qs.get("price") or {}
        nm2 = pr.get("longName") or pr.get("shortName")
        if nm2 and not out.get("name"):
            out["name"] = nm2
        if (pr.get("currency") and not out.get("currency")):
            out["currency"] = pr.get("currency")

        # assetProfile -> sector/industry
        ap = qs.get("assetProfile") or {}
        sec = ap.get("sector")
        ind = ap.get("industry")
        if sec:
            out["sector"] = sec
        if ind:
            out["industry"] = ind

        # summaryDetail -> dividendRate/dividendYield
        sd = qs.get("summaryDetail") or {}
        dr2 = _to_float(_raw(sd, "dividendRate"))
        if dr2 is not None:
            out["annual_dividend"] = dr2
        dy2 = _to_float(_raw(sd, "dividendYield"))
        if dy2 is not None:
            out["dividend_yield_pct"] = float(dy2) * 100.0

        # financialData -> margins, EV/EBITDA
        fd = qs.get("financialData") or {}
        gm = _to_float(_raw(fd, "grossMargins"))
        if gm is not None:
            out["gross_margin_pct"] = float(gm) * 100.0
        pm = _to_float(_raw(fd, "profitMargins"))
        if pm is not None:
            out["net_margin_pct"] = float(pm) * 100.0
        e2e = _to_float(_raw(fd, "enterpriseToEbitda"))
        if e2e is not None:
            out["ev_ebitda"] = e2e

        # defaultKeyStatistics -> sharesOutstanding, debtToEquity
        ks = qs.get("defaultKeyStatistics") or {}
        so2 = _to_float(_raw(ks, "sharesOutstanding"))
        if so2 is not None:
            out["shares_outstanding"] = so2
        dte = _to_float(_raw(ks, "debtToEquity"))
        if dte is not None:
            out["debt_to_equity"] = dte

    # Valuta default till USD om okänd
    if not out.get("currency"):
        out["currency"] = "USD"

    return out


# ------------------------------------------------------------
# Små hjälpare för Yahoo:s "raw"-fält (kan vara dict med 'raw'/'fmt')
# ------------------------------------------------------------
def _raw(obj: Dict[str, Any], key: str):
    """
    Yahoo returnerar ofta {"raw": 123, "fmt": "123.00"} – den här plockar "raw" om finnes.
    """
    if key not in obj:
        return None
    v = obj.get(key)
    if isinstance(v, dict) and "raw" in v:
        return v.get("raw")
    return v
