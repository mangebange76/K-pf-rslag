# stockapp/fetchers/yahoo.py
from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, Any

import numpy as np
import pandas as pd
import streamlit as st

# yfinance används som primär källa
try:
    import yfinance as yf
except Exception as e:
    yf = None  # hanteras i get_all

TTL_SECS = 600  # 10 min cache


def _safe_float(x) -> float:
    try:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return 0.0
        return float(x)
    except Exception:
        return 0.0


def _cagr5_from_financials(tkr: "yf.Ticker") -> float:
    """
    CAGR på 'Total Revenue' utifrån årsdata i yfinance (5 år om möjligt).
    Returnerar procent (t.ex. 12.3).
    """
    try:
        df = getattr(tkr, "financials", None)
        if isinstance(df, pd.DataFrame) and not df.empty and "Total Revenue" in df.index:
            series = df.loc["Total Revenue"].dropna().sort_index()
        else:
            df = getattr(tkr, "income_stmt", None)
            if isinstance(df, pd.DataFrame) and not df.empty and "Total Revenue" in df.index:
                series = df.loc["Total Revenue"].dropna().sort_index()
            else:
                return 0.0

        if len(series) < 2:
            return 0.0
        start = float(series.iloc[0])
        end = float(series.iloc[-1])
        years = max(1, len(series) - 1)
        if start <= 0:
            return 0.0
        cagr = (end / start) ** (1.0 / years) - 1.0
        return round(cagr * 100.0, 2)
    except Exception:
        return 0.0


def _revenue_ttm_and_growth(tkr: "yf.Ticker") -> tuple[float, float]:
    """
    Returnerar (revenue_ttm, growth_pct). Båda i företagets rapporteringsvaluta.
    Hämtas via quarterly financials (summa 4 kvartal) med YoY-growth mot föregående 4.
    """
    try:
        q = getattr(tkr, "quarterly_financials", None)
        if not (isinstance(q, pd.DataFrame) and not q.empty and "Total Revenue" in q.index):
            return 0.0, 0.0

        s = q.loc["Total Revenue"].dropna()
        if len(s) < 4:
            return 0.0, 0.0

        s = s.sort_index()
        last4 = float(s.iloc[-4:].sum())
        prev4 = float(s.iloc[-8:-4].sum()) if len(s) >= 8 else 0.0
        growth = ((last4 - prev4) / prev4 * 100.0) if prev4 > 0 else 0.0
        return last4, round(growth, 2)
    except Exception:
        return 0.0, 0.0


def _book_value_per_share(info: dict, shares_out: float) -> float:
    # yfinance ger ibland 'bookValue' = BVPS direkt
    bvps = _safe_float(info.get("bookValue"))
    if bvps > 0:
        return bvps
    # annars: totalStockholderEquity / shares
    tse = _safe_float(info.get("totalStockholderEquity"))
    if tse > 0 and shares_out > 0:
        return tse / shares_out
    return 0.0


def _ev_ebitda(info: dict) -> tuple[float, float, float]:
    ev = _safe_float(info.get("enterpriseValue"))
    ebitda = _safe_float(info.get("ebitda"))
    ratio = (ev / ebitda) if (ev > 0 and ebitda > 0) else 0.0
    return ev, ebitda, ratio


def _price(info: dict, tkr_obj: "yf.Ticker") -> float:
    p = _safe_float(info.get("regularMarketPrice"))
    if p > 0:
        return p
    # fallback – sista stängning
    try:
        h = tkr_obj.history(period="1d")
        if not h.empty and "Close" in h:
            return float(h["Close"].iloc[-1])
    except Exception:
        pass
    return 0.0


def _ps_ttm(mcap: float, revenue_ttm: float) -> float:
    return (mcap / revenue_ttm) if (mcap > 0 and revenue_ttm > 0) else 0.0


def _pb(price: float, bvps: float, info: dict) -> float:
    p2b = _safe_float(info.get("priceToBook"))
    if p2b > 0:
        return p2b
    return (price / bvps) if (price > 0 and bvps > 0) else 0.0


def _payout_ratio_pct(info: dict) -> float:
    pr = info.get("payoutRatio", None)
    if pr is None:
        return 0.0
    # ibland i [0..1], ibland redan i %
    return round(float(pr) * 100.0, 2) if pr <= 1 else round(float(pr), 2)


def _dividend_rate_and_yield(info: dict) -> tuple[float, float]:
    rate = _safe_float(info.get("dividendRate"))
    yld = _safe_float(info.get("dividendYield"))  # ofta [0..1]
    if 0 < yld <= 1:
        yld *= 100.0
    return rate, round(yld, 2)


def _margins(info: dict) -> tuple[float, float, float]:
    gm = _safe_float(info.get("grossMargins"))
    om = _safe_float(info.get("operatingMargins"))
    nm = _safe_float(info.get("profitMargins"))
    # konvertera ev. [0..1] till %
    gm = gm * 100.0 if 0 < gm < 1 else gm
    om = om * 100.0 if 0 < om < 1 else om
    nm = nm * 100.0 if 0 < nm < 1 else nm
    return round(gm, 2), round(om, 2), round(nm, 2)


@st.cache_data(ttl=TTL_SECS, show_spinner=False)
def get_all(ticker: str) -> Dict[str, Any]:
    """
    Returnerar en dict med alla nycklar appen använder från Yahoo.
    Nycklar:
      name, currency, price, market_cap, shares_outstanding,
      ps_ttm, pb, ev_ebitda, enterprise_value, ebitda,
      dividend_rate, dividend_yield_pct, payout_ratio_pct,
      revenue_ttm, revenue_growth_pct, book_value_per_share,
      gross_margins_pct, operating_margins_pct, profit_margins_pct,
      cagr5_pct
    """
    out: Dict[str, Any] = {}

    if yf is None:
        return out  # yfinance saknas i miljön

    try:
        t = yf.Ticker(ticker)
        try:
            info = t.fast_info or {}
        except Exception:
            info = {}

        # fast_info har begränsat innehåll – komplettera med .info (kan vara tung)
        try:
            full = t.info or {}
        except Exception:
            full = {}

        # kombinera – full tar företräde om värde finns
        data = {**info, **full}

        out["name"] = data.get("shortName") or data.get("longName") or ""
        out["currency"] = (data.get("currency") or "").upper()

        price = _price(data, t)
        out["price"] = price

        mcap = _safe_float(data.get("marketCap"))
        out["market_cap"] = mcap

        shares = _safe_float(data.get("sharesOutstanding"))
        out["shares_outstanding"] = shares

        # TTM revenue & growth
        rev_ttm, rev_growth = _revenue_ttm_and_growth(t)
        out["revenue_ttm"] = rev_ttm
        out["revenue_growth_pct"] = rev_growth

        # BVPS + PB
        bvps = _book_value_per_share(data, shares)
        out["book_value_per_share"] = bvps
        out["pb"] = round(_pb(price, bvps, data), 4)

        # PS TTM
        out["ps_ttm"] = round(_ps_ttm(mcap, rev_ttm), 4)

        # EV/EBITDA
        ev, ebitda, ratio = _ev_ebitda(data)
        out["enterprise_value"] = ev
        out["ebitda"] = ebitda
        out["ev_ebitda"] = round(ratio, 4) if ratio else 0.0

        # Utdelning + payout
        rate, yld = _dividend_rate_and_yield(data)
        out["dividend_rate"] = rate
        out["dividend_yield_pct"] = yld
        out["payout_ratio_pct"] = _payout_ratio_pct(data)

        # Marginaler
        gm, om, nm = _margins(data)
        out["gross_margins_pct"] = gm
        out["operating_margins_pct"] = om
        out["profit_margins_pct"] = nm

        # CAGR 5 år (omsättning)
        out["cagr5_pct"] = _cagr5_from_financials(t)

    except Exception:
        # returnera vad vi har; tom dict är också OK (appen hanterar)
        pass

    return out
