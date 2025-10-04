# stockapp/fetchers/yahoo.py
from __future__ import annotations

import math
from typing import Dict, Any

import numpy as np
import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None  # type: ignore


def _safe_float(x) -> float:
    try:
        f = float(x)
        if math.isnan(f) or math.isinf(f):
            return 0.0
        return f
    except Exception:
        return 0.0


def _first_nonzero(*vals) -> float:
    for v in vals:
        f = _safe_float(v)
        if f != 0.0:
            return f
    return 0.0


def _cagr5_from_annual(fin_df: pd.DataFrame) -> float:
    """
    Försök beräkna CAGR ~5 år från årlig 'Total Revenue'.
    Vi använder minsta möjliga 2 punkter om färre än 5 finns.
    """
    try:
        if not isinstance(fin_df, pd.DataFrame) or fin_df.empty:
            return 0.0
        # Identifiera rätt rad
        idx = None
        for cand in ["Total Revenue", "TotalRevenue", "Total revenue"]:
            if cand in fin_df.index:
                idx = cand
                break
        if idx is None:
            return 0.0
        series = fin_df.loc[idx].dropna()
        if len(series) < 2:
            return 0.0
        # kronologiskt
        series = series.sort_index()
        start = float(series.iloc[0])
        end   = float(series.iloc[-1])
        years = max(1, len(series) - 1)
        if start <= 0:
            return 0.0
        cagr = (end / start) ** (1.0 / years) - 1.0
        return round(cagr * 100.0, 2)
    except Exception:
        return 0.0


def _revenue_ttm_from_quarterlies(q_df: pd.DataFrame) -> float:
    """
    TTM: summera senaste fyra 'Total Revenue' från kvartalsdf om möjligt.
    """
    try:
        if not isinstance(q_df, pd.DataFrame) or q_df.empty:
            return 0.0
        idx = None
        for cand in ["Total Revenue", "TotalRevenue", "Total revenue"]:
            if cand in q_df.index:
                idx = cand
                break
        if idx is None:
            return 0.0
        series = q_df.loc[idx].dropna()
        if series.empty:
            return 0.0
        series = series.sort_index(ascending=False)  # nyast först
        if len(series) >= 4:
            return float(series.iloc[:4].sum())
        return float(series.sum())
    except Exception:
        return 0.0


@st.cache_data(ttl=600, show_spinner=False)
def get_all(ticker: str) -> Dict[str, Any]:
    """
    Hämtar ett robust paket av nycklar från Yahoo (via yfinance). Cache: 10 min.

    Returnerar (alla värden i aktiens noteringsvaluta):
      price, currency, market_cap, shares_outstanding, name, sector, industry,
      ps_ttm, pb, ev_ebitda,
      dividend_rate, dividend_yield_pct, payout_ratio_pct,
      book_value_per_share,
      gross_margins_pct, operating_margins_pct, profit_margins_pct,
      enterprise_value, ebitda,
      revenue_ttm, revenue_growth_pct, cagr5_pct
    """
    if yf is None:
        raise RuntimeError("yfinance är inte installerat.")

    tkr = ticker.upper().strip()
    t = yf.Ticker(tkr)

    # -------- Grundinfo / fast_info --------
    info: Dict[str, Any] = {}
    try:
        info = t.info or {}
    except Exception:
        info = {}

    fast: Dict[str, Any] = {}
    try:
        fast_raw = getattr(t, "fast_info", {})  # kan vara dict eller objekt
        if isinstance(fast_raw, dict):
            fast = fast_raw
        else:
            # extrahera några fält om det är ett objekt
            fast = {
                "last_price": getattr(fast_raw, "last_price", None),
                "currency": getattr(fast_raw, "currency", None),
                "market_cap": getattr(fast_raw, "market_cap", None),
                "shares_outstanding": getattr(fast_raw, "shares_outstanding", None),
            }
    except Exception:
        fast = {}

    price       = _first_nonzero(fast.get("last_price"), info.get("regularMarketPrice"))
    currency    = fast.get("currency") or info.get("currency") or ""
    market_cap  = _first_nonzero(fast.get("market_cap"), info.get("marketCap"))
    shares_out  = _first_nonzero(fast.get("shares_outstanding"), info.get("sharesOutstanding"))

    name    = info.get("shortName") or info.get("longName") or tkr
    sector  = info.get("sector") or ""
    industry= info.get("industry") or ""

    # -------- Marginaler (procent) --------
    gross_margins_pct     = _safe_float(info.get("grossMargins")) * 100.0
    operating_margins_pct = _safe_float(info.get("operatingMargins")) * 100.0
    profit_margins_pct    = _safe_float(info.get("profitMargins")) * 100.0

    # -------- Utdelning --------
    dividend_rate      = _safe_float(info.get("dividendRate"))
    dividend_yield_pct = _safe_float(info.get("dividendYield")) * 100.0
    payout_ratio_pct   = _safe_float(info.get("payoutRatio")) * 100.0

    # -------- Värdering --------
    pb           = _safe_float(info.get("priceToBook"))
    ev_to_ebitda = _safe_float(info.get("enterpriseToEbitda"))
    enterprise_value = _safe_float(info.get("enterpriseValue"))
    ebitda       = _safe_float(info.get("ebitda"))
    bvps         = _safe_float(info.get("bookValue"))

    # -------- Tillväxt --------
    revenue_growth_pct = _safe_float(info.get("revenueGrowth")) * 100.0

    # -------- Finansiella tabeller (kan saknas i vissa miljöer) --------
    annual_fin = pd.DataFrame()
    quarterly_fin = pd.DataFrame()
    try:
        # yfinance 0.2.x
        annual_fin = getattr(t, "financials", pd.DataFrame())
        quarterly_fin = getattr(t, "quarterly_financials", pd.DataFrame())
    except Exception:
        pass

    cagr5_pct = _cagr5_from_annual(annual_fin)
    revenue_ttm = _revenue_ttm_from_quarterlies(quarterly_fin)
    if revenue_ttm <= 0:
        # fallback: totalRevenue (annual) från info
        revenue_ttm = _safe_float(info.get("totalRevenue"))

    ps_ttm = 0.0
    if market_cap > 0 and revenue_ttm > 0:
        ps_ttm = market_cap / revenue_ttm

    return {
        "price": price,
        "currency": currency,
        "market_cap": market_cap,
        "shares_outstanding": shares_out,
        "name": name,
        "sector": sector,
        "industry": industry,

        "ps_ttm": ps_ttm,
        "pb": pb,
        "ev_ebitda": ev_to_ebitda,

        "dividend_rate": dividend_rate,
        "dividend_yield_pct": dividend_yield_pct,
        "payout_ratio_pct": payout_ratio_pct,

        "book_value_per_share": bvps,

        "gross_margins_pct": gross_margins_pct,
        "operating_margins_pct": operating_margins_pct,
        "profit_margins_pct": profit_margins_pct,

        "enterprise_value": enterprise_value,
        "ebitda": ebitda,

        "revenue_ttm": revenue_ttm,
        "revenue_growth_pct": revenue_growth_pct,
        "cagr5_pct": cagr5_pct,
    }
