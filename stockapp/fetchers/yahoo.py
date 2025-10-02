# -*- coding: utf-8 -*-
"""
Yahoo-fetcher med säkra fallbacks.
Ger både livepris (get_live_price) och ett snapshot av grundnyckeltal (get_snapshot).
Kräver yfinance (installeras i Streamlit Cloud-miljön).
"""

from __future__ import annotations
from typing import Dict, Optional, Any

import math
import pandas as pd

try:
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover
    yf = None  # type: ignore


def _sf(x, default=None):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return default
        return float(x)
    except Exception:
        return default


def _pct100(x):
    """yfinance returnerar många marginaler som andel 0..1 → konvertera till %."""
    v = _sf(x, None)
    if v is None:
        return None
    return float(v) * 100.0


def get_live_price(ticker: str) -> Optional[float]:
    """Hämta snabbast möjliga pris från Yahoo."""
    if yf is None:
        return None
    try:
        tk = yf.Ticker(str(ticker))
        # fast_info är snabb, fall tillbaka till info om saknas
        fi = getattr(tk, "fast_info", {}) or {}
        price = fi.get("last_price") or fi.get("regular_market_price")
        if not price:
            info = tk.info or {}
            price = info.get("regularMarketPrice") or info.get("currentPrice")
        return _sf(price, None)
    except Exception:
        return None


def get_snapshot(ticker: str) -> Dict[str, Any]:
    """
    Hämtar ett "bästa ansträngning"-paket med fält vi använder i appen.
    Returnerar dict (kan vara delmängd om Yahoo saknar värden):
      - Kurs, Valuta, Bolagsnamn, Sektor, Industri
      - Market Cap, Utestående aktier (milj.)
      - P/S, EV/EBITDA (ttm), P/B
      - Gross/Operating/Net margin (%), ROE (%)
      - Dividend yield (%), Dividend payout (FCF) (%)  [payout → payoutRatio om finns]
    """
    out: Dict[str, Any] = {}
    if yf is None:
        return out

    tk = yf.Ticker(str(ticker))

    # --- Pris / valuta / namn / sektor / industri ---------------------------
    try:
        fi = getattr(tk, "fast_info", {}) or {}
        info = tk.info or {}

        price = fi.get("last_price") or fi.get("regular_market_price") or info.get("regularMarketPrice") or info.get("currentPrice")
        currency = fi.get("currency") or info.get("currency")
        name = info.get("shortName") or info.get("longName") or info.get("symbol") or str(ticker)
        sector = info.get("sector")
        industry = info.get("industry")

        if price:
            out["Kurs"] = _sf(price, None)
        if currency:
            out["Valuta"] = str(currency)
        if name:
            out["Bolagsnamn"] = str(name)
        if sector:
            out["Sektor"] = str(sector)
        if industry:
            out["Industri"] = str(industry)
    except Exception:
        pass

    # --- Storlek / aktier ----------------------------------------------------
    try:
        mcap = fi.get("market_cap") if "fi" in locals() else None
        if not mcap:
            mcap = info.get("marketCap")
        if mcap:
            out["Market Cap"] = _sf(mcap, None)

        shares = info.get("sharesOutstanding")  # absolut antal
        if shares:
            out["Utestående aktier (milj.)"] = float(shares) / 1e6
    except Exception:
        pass

    # --- Multiplar -----------------------------------------------------------
    try:
        out["P/S"] = _sf(info.get("priceToSalesTrailing12Months"), None)
    except Exception:
        pass

    try:
        ev_eb = info.get("enterpriseToEbitda") or info.get("evToEbitda")
        out["EV/EBITDA (ttm)"] = _sf(ev_eb, None)
    except Exception:
        pass

    try:
        out["P/B"] = _sf(info.get("priceToBook"), None)
    except Exception:
        pass

    # --- Marginaler / ROE ----------------------------------------------------
    try:
        out["Gross margin (%)"] = _pct100(info.get("grossMargins"))
    except Exception:
        pass
    try:
        out["Operating margin (%)"] = _pct100(info.get("operatingMargins"))
    except Exception:
        pass
    try:
        out["Net margin (%)"] = _pct100(info.get("profitMargins"))
    except Exception:
        pass
    try:
        out["ROE (%)"] = _pct100(info.get("returnOnEquity"))
    except Exception:
        pass

    # --- Utdelning -----------------------------------------------------------
    try:
        out["Dividend yield (%)"] = _pct100(info.get("dividendYield"))
    except Exception:
        pass
    try:
        pr = info.get("payoutRatio")
        if pr is not None:
            out["Dividend payout (FCF) (%)"] = _pct100(pr)
    except Exception:
        pass

    return {k: v for k, v in out.items() if v is not None}
