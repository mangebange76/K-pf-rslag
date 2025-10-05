from __future__ import annotations
from typing import Dict, Any, Iterable, Optional
import math
import pandas as pd
import numpy as np
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None  # type: ignore

def _f(x) -> float:
    try:
        if x is None or (isinstance(x,float) and math.isnan(x)): return 0.0
        return float(x)
    except Exception:
        return 0.0

def _sum_dividends_ttm(t: "yf.Ticker") -> float:
    try:
        d = t.dividends
        if d is None or d.empty: return 0.0
        cutoff = pd.Timestamp.utcnow().tz_localize("UTC") - pd.Timedelta(days=365)
        s = d[d.index >= cutoff]
        return float(s.sum()) if not s.empty else 0.0
    except Exception:
        return 0.0

def _cagr5_from_revenue(t: "yf.Ticker") -> float:
    for attr in ("income_stmt","financials"):
        try:
            df = getattr(t, attr, None)
            if isinstance(df, pd.DataFrame) and not df.empty and "Total Revenue" in df.index:
                s = df.loc["Total Revenue"].dropna().sort_index()
                if len(s) >= 2:
                    start = float(s.iloc[0]); end = float(s.iloc[-1]); years = max(1, len(s)-1)
                    if start > 0: return ((end/start)**(1/years) - 1.0)*100.0
        except Exception:
            pass
    return 0.0

@st.cache_data(ttl=600, show_spinner=False)
def get_all(ticker: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "name":"", "currency":"", "price":0.0, "sector":"", "industry":"",
        "dividend_rate":0.0, "dividend_yield_pct":0.0, "payout_ratio_pct":0.0,
        "ps_ttm":0.0, "pb":0.0, "shares_outstanding":0.0, "market_cap":0.0,
        "book_value_per_share":0.0, "cagr5_pct":0.0
    }
    if yf is None or not ticker: return out
    t = yf.Ticker(ticker)
    info = {}
    try: info = t.info or {}
    except Exception: info = {}
    out["name"]     = str(info.get("shortName") or info.get("longName") or "")
    out["currency"] = str(info.get("currency") or "USD").upper()
    out["sector"]   = str(info.get("sector") or "")
    out["price"]    = _f(info.get("regularMarketPrice")) or \
                      (lambda h: float(h["Close"].iloc[-1]) if isinstance(h,pd.DataFrame) and not h.empty else 0.0)(t.history(period="5d"))
    out["dividend_rate"] = _f(info.get("dividendRate")) or _sum_dividends_ttm(t)
    if out["price"]>0 and out["dividend_rate"]>0:
        out["dividend_yield_pct"] = (out["dividend_rate"]/out["price"])*100.0
    # payout approx via EPS fallback
    eps = _f(info.get("trailingEps"))
    if eps>0 and out["dividend_rate"]>0:
        out["payout_ratio_pct"] = (out["dividend_rate"]/eps)*100.0
    else:
        pr = _f(info.get("payoutRatio"))*100.0
        if pr>0: out["payout_ratio_pct"] = pr
    out["ps_ttm"] = _f(info.get("priceToSalesTrailing12Months"))
    out["pb"]     = _f(info.get("priceToBook"))
    out["shares_outstanding"] = _f(info.get("sharesOutstanding"))
    out["market_cap"]         = _f(info.get("marketCap"))
    out["book_value_per_share"] = _f(info.get("bookValue"))
    out["cagr5_pct"] = _cagr5_from_revenue(t)
    # Fallback PS via marketcap/revenue om bokfört saknas – hoppar för enkelhet
    return out

@st.cache_data(ttl=600, show_spinner=False)
def get_quick(ticker: str) -> Dict[str, Any]:
    """Snabb: name, currency, price, dividend_rate, CAGR(5y)."""
    d = get_all(ticker)
    return {
        "Bolagsnamn": d["name"],
        "Valuta": d["currency"],
        "Aktuell kurs": d["price"],
        "Årlig utdelning": d["dividend_rate"],
        "CAGR 5 år (%)": d["cagr5_pct"],
    }
