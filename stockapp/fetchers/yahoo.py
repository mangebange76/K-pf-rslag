# -*- coding: utf-8 -*-
"""
stockapp.fetchers.yahoo
-----------------------
Robusta Yahoo-funktioner (utan externa paket) via quoteSummary-API:t.

Publikt API:
- get_live_price(ticker) -> float | None
- get_all(ticker) -> Dict[str, Any]
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import requests
import math
import time

YF_BASE = "https://query1.finance.yahoo.com/v10/finance/quoteSummary"

# ---- Hjälpare ---------------------------------------------------------

def _yf_get_json(ticker: str, modules: List[str], timeout: float = 12.0) -> Dict[str, Any]:
    """
    Hämtar quoteSummary-json för givna modules.
    Returnerar {} om fel.
    """
    params = {"modules": ",".join(modules)}
    url = f"{YF_BASE}/{ticker}"
    try:
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code != 200:
            return {}
        j = r.json() or {}
        res = (j.get("quoteSummary") or {}).get("result")
        if not res or not isinstance(res, list):
            return {}
        return res[0] or {}
    except Exception:
        return {}

def _safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(p)
        if cur is None:
            return default
    return cur

def _safe_float(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

def _as_pct(val) -> Optional[float]:
    v = _safe_float(val, None)
    if v is None:
        return None
    # Yahoo anger ofta t.ex. dividendYield som 0.0123 (=1.23%)
    if 0 < v < 1.0:
        return round(v * 100.0, 2)
    return round(v, 2)

def _fmt_millions(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x) / 1e6
    except Exception:
        return None

# ---- Publika funktioner -----------------------------------------------

def get_live_price(ticker: str) -> Optional[float]:
    """
    Snabbt pris. Returnerar None om det inte fanns.
    """
    data = _yf_get_json(ticker, ["price"])
    p = _safe_get(data, ["price", "regularMarketPrice", "raw"], None)
    if p is None:
        # prova fallback (ibland ligger den i "regularMarketOpen" när stängt)
        p = _safe_get(data, ["price", "regularMarketPreviousClose", "raw"], None)
    return _safe_float(p, None)

def _get_quarter_revenues(ticker: str, limit: int = 6) -> List[Tuple[str, float]]:
    """
    Hämtar kvartalsintäkter [(date, revenue), ...] från Yahoo.
    Returnerar max 'limit' senaste.
    """
    data = _yf_get_json(ticker, ["incomeStatementHistoryQuarterly"])
    arr = _safe_get(data, ["incomeStatementHistoryQuarterly", "incomeStatementHistory"], [])
    out: List[Tuple[str, float]] = []
    if isinstance(arr, list):
        for it in arr[:limit]:
            try:
                end_date = _safe_get(it, ["endDate", "fmt"], None) or _safe_get(it, ["endDate", "raw"], None)
                rev = _safe_get(it, ["totalRevenue", "raw"], None)
                revf = _safe_float(rev, None)
                if end_date and revf is not None:
                    out.append((str(end_date), float(revf)))
            except Exception:
                pass
    return out

def get_all(ticker: str) -> Dict[str, Any]:
    """
    Komplett hämtning från Yahoo (price, key stats, profile, margins m.m.).
    Returnerar ett dict med svenska kolumnnamn när möjligt.
    """
    # Försök i ett par "omgångar" om nät strular
    modules = [
        "price",
        "summaryDetail",
        "defaultKeyStatistics",
        "financialData",
        "assetProfile",
        "quoteType",
    ]
    data: Dict[str, Any] = {}
    for attempt in range(2):
        data = _yf_get_json(ticker, modules)
        if data:
            break
        time.sleep(0.8)

    out: Dict[str, Any] = {}
    logs: List[str] = []

    # Bas
    price = _safe_get(data, ["price"], {}) or {}
    ks = _safe_get(data, ["defaultKeyStatistics"], {}) or {}
    sd = _safe_get(data, ["summaryDetail"], {}) or {}
    fin = _safe_get(data, ["financialData"], {}) or {}
    prof = _safe_get(data, ["assetProfile"], {}) or {}
    qtype = _safe_get(data, ["quoteType"], {}) or {}

    # Namn, valuta
    namn = _safe_get(price, ["longName"], None) or _safe_get(price, ["shortName"], None)
    if namn:
        out["Bolagsnamn"] = str(namn)
    val = _safe_get(price, ["currency"], None)
    if val:
        out["Valuta"] = str(val)

    # Kurs
    kurs = _safe_get(price, ["regularMarketPrice", "raw"], None)
    if kurs is None:
        kurs = _safe_get(price, ["regularMarketPreviousClose", "raw"], None)
    if kurs is not None:
        out["Kurs"] = _safe_float(kurs, None)

    # Market Cap
    mcap = _safe_get(price, ["marketCap", "raw"], None)
    if mcap is None:
        mcap = _safe_get(ks, ["marketCap", "raw"], None)
    if mcap is not None:
        out["Market Cap"] = _safe_float(mcap, None)

    # Utestående aktier (milj.)
    sh = _safe_get(ks, ["sharesOutstanding", "raw"], None)
    if sh is not None:
        out["Utestående aktier (milj.)"] = _fmt_millions(_safe_float(sh, None))
    else:
        # fallback: mcap/price
        if out.get("Market Cap") and out.get("Kurs"):
            try:
                out["Utestående aktier (milj.)"] = float(out["Market Cap"]) / float(out["Kurs"]) / 1e6
            except Exception:
                pass

    # Marginaler & multiplar
    ps_ttm = _safe_get(sd, ["priceToSalesTrailing12Months", "raw"], None)
    if ps_ttm is not None:
        out["P/S"] = _safe_float(ps_ttm, None)

    ev_ebitda = _safe_get(ks, ["enterpriseToEbitda", "raw"], None)
    if ev_ebitda is not None:
        out["EV/EBITDA (ttm)"] = _safe_float(ev_ebitda, None)

    pb = _safe_get(ks, ["priceToBook", "raw"], None)
    if pb is not None:
        out["P/B"] = _safe_float(pb, None)

    gm = _safe_get(fin, ["grossMargins", "raw"], None)
    if gm is not None:
        out["Gross margin (%)"] = _as_pct(gm)

    opm = _safe_get(fin, ["operatingMargins", "raw"], None)
    if opm is not None:
        out["Operating margin (%)"] = _as_pct(opm)

    nm = _safe_get(fin, ["profitMargins", "raw"], None)
    if nm is not None:
        out["Net margin (%)"] = _as_pct(nm)

    dy = _safe_get(sd, ["dividendYield", "raw"], None)
    if dy is not None:
        out["Dividend yield (%)"] = _as_pct(dy)

    # Sektor/Industri
    sector = _safe_get(prof, ["sector"], None)
    if sector:
        out["Sektor"] = str(sector)
    industry = _safe_get(prof, ["industry"], None)
    if industry:
        out["Industri"] = str(industry)

    # Kvartalsintäkter (för att kunna fylla P/S Q1..Q4 rudimentärt)
    # OBS: Detta är "per kvartal" revenue; om du vill P/S per kvartal kan man
    # dela MarketCap med just den kvartalets revenue (inte helt standardiserat).
    try:
        revs = _get_quarter_revenues(ticker, limit=4)  # [(date, rev), ...], senaste först
        if revs and out.get("Market Cap"):
            m = float(out["Market Cap"])
            # Sortera senaste först → Q1 = senaste
            # Vi returnerar P/S Q1..Q4 på denna approximativa metod
            if len(revs) > 0 and revs[0][1] > 0:
                out["P/S Q1"] = float(m / float(revs[0][1]))
            if len(revs) > 1 and revs[1][1] > 0:
                out["P/S Q2"] = float(m / float(revs[1][1]))
            if len(revs) > 2 and revs[2][1] > 0:
                out["P/S Q3"] = float(m / float(revs[2][1]))
            if len(revs) > 3 and revs[3][1] > 0:
                out["P/S Q4"] = float(m / float(revs[3][1]))
    except Exception:
        pass

    # Liten logg
    out["__yahoo_fields__"] = len([k for k in out.keys() if not k.startswith("__")])

    return out
