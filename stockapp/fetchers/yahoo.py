# stockapp/fetchers/yahoo.py
from __future__ import annotations

import math
import datetime as dt
from typing import Dict, Any, Optional, Iterable

import numpy as np
import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None  # type: ignore


def _safe_float(x) -> float:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return 0.0
        return float(x)
    except Exception:
        return 0.0


def _get_price(t: "yf.Ticker") -> float:
    price = 0.0
    try:
        info = t.info or {}
        price = _safe_float(info.get("regularMarketPrice"))
    except Exception:
        pass
    if price > 0:
        return price
    # Fallback: senaste close
    try:
        h = t.history(period="5d")
        if not h.empty and "Close" in h.columns:
            return float(h["Close"].iloc[-1])
    except Exception:
        pass
    return 0.0


def _sum_dividends_ttm(t: "yf.Ticker") -> float:
    """Utdelning per aktie TTM (summa senaste 365 dagar)."""
    try:
        div = t.dividends
        if div is None or div.empty:
            return 0.0
        cutoff = pd.Timestamp.utcnow().tz_localize("UTC") - pd.Timedelta(days=365)
        s = div[div.index >= cutoff]
        return float(s.sum()) if not s.empty else 0.0
    except Exception:
        return 0.0


def _get_cashflow_df(t: "yf.Ticker") -> pd.DataFrame:
    """
    Returnera kvartalsvis kassaflöde om möjligt (ger bäst TTM),
    annars årlig. Normalisera kolumnnamn till lower().
    """
    for attr in ("quarterly_cashflow", "cashflow"):
        try:
            df = getattr(t, attr, None)
            if isinstance(df, pd.DataFrame) and not df.empty:
                # YF returnerar transponerat (rad=post, kol=perioder)
                df2 = df.copy()
                df2.index = [str(i).strip().lower() for i in df2.index]
                return df2
        except Exception:
            pass
    return pd.DataFrame()


def _pick_row(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[pd.Series]:
    if df.empty:
        return None
    idx = df.index
    for name in candidates:
        key = name.strip().lower()
        # exakt
        if key in idx:
            return df.loc[key]
        # fuzzy contain
        hits = [i for i in idx if key in i]
        if hits:
            return df.loc[hits[0]]
    return None


def _compute_fcf_ttm(t: "yf.Ticker") -> float:
    """
    FCF (TTM) ≈ sum(Free Cash Flow, senaste 4 kv) om finns,
    annars sum(Operating CF) + sum(Capex) (Capex är normalt < 0).
    Faller tillbaka till årlig om kvartal saknas.
    """
    df = _get_cashflow_df(t)
    if df.empty:
        return 0.0

    # 1) Försök direkt Free Cash Flow
    free_cf_row = _pick_row(df, ["free cash flow", "freecashflow"])
    if free_cf_row is not None:
        vals = [ _safe_float(v) for v in free_cf_row.dropna().values[:4] ]
        if vals:
            return float(np.nansum(vals))

    # 2) Annars OCF + Capex
    ocf_row  = _pick_row(df, [
        "operating cash flow",
        "total cash from operating activities",
        "net cash provided by operating activities",
    ])
    capex_row = _pick_row(df, [
        "capital expenditure",
        "capital expenditures",
        "payments to acquire property plant and equipment",
    ])

    ocf_vals  = [ _safe_float(v) for v in (ocf_row.dropna().values[:4]  if ocf_row  is not None else []) ]
    capex_vals= [ _safe_float(v) for v in (capex_row.dropna().values[:4] if capex_row is not None else []) ]

    if ocf_vals and capex_vals:
        return float(np.nansum(ocf_vals) + np.nansum(capex_vals))

    # 3) Fallback: sista tillgängliga periodens FCF/OCF−Capex
    if ocf_row is not None and capex_row is not None:
        return _safe_float(ocf_row.iloc[0]) + _safe_float(capex_row.iloc[0])

    return 0.0


def _revenue_ttm_and_growth(t: "yf.Ticker") -> tuple[float, float]:
    """
    Försök hämta Revenue TTM och YoY-tillväxt i procent.
    Revenue TTM tas i första hand från info['totalRevenue'] (ttm),
    annars summering från 'income_stmt'/'financials' som fallback.
    """
    rev_ttm = 0.0
    growth_pct = 0.0

    # 1) info
    info = {}
    try:
        info = t.info or {}
    except Exception:
        info = {}
    rev_ttm = _safe_float(info.get("totalRevenue"))

    # 2) Growth (decimal → %)
    growth_pct = _safe_float(info.get("revenueGrowth")) * 100.0

    # 3) Fallback på rev_ttm via income_stmt / financials
    if rev_ttm <= 0:
        for attr in ("income_stmt", "financials"):
            try:
                df = getattr(t, attr, None)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    if "Total Revenue" in df.index:
                        s = df.loc["Total Revenue"].dropna()
                        if not s.empty:
                            # TTM ≈ senaste årsrad (grovt fallback)
                            rev_ttm = float(s.iloc[0])
                            break
            except Exception:
                pass

    # 4) Fallback growth på senaste två års rader
    if growth_pct == 0.0:
        for attr in ("income_stmt", "financials"):
            try:
                df = getattr(t, attr, None)
                if isinstance(df, pd.DataFrame) and not df.empty and "Total Revenue" in df.index:
                    s = df.loc["Total Revenue"].dropna()
                    if len(s) >= 2:
                        latest = float(s.iloc[0])
                        prev   = float(s.iloc[1])
                        if prev > 0:
                            growth_pct = (latest/prev - 1.0) * 100.0
                            break
            except Exception:
                pass

    return rev_ttm, growth_pct


def _cagr5_from_revenue(t: "yf.Ticker") -> float:
    """
    Grov CAGR över ~5 år från 'Total Revenue' i annuala statements.
    """
    for attr in ("income_stmt", "financials"):
        try:
            df = getattr(t, attr, None)
            if isinstance(df, pd.DataFrame) and not df.empty and "Total Revenue" in df.index:
                s = df.loc["Total Revenue"].dropna().sort_index()
                if len(s) >= 2:
                    start = float(s.iloc[0])
                    end   = float(s.iloc[-1])
                    years = max(1, len(s) - 1)
                    if start > 0:
                        return ((end/start) ** (1.0/years) - 1.0) * 100.0
        except Exception:
            pass
    return 0.0


@st.cache_data(ttl=600, show_spinner=False)
def get_all(ticker: str) -> Dict[str, Any]:
    """
    Hämtar ett brett urval nyckeltal från Yahoo (yfinance) och
    beräknar FCF_TTM, payout_fcf_pct, FCF-yield, m.m.

    Returnerar nycklar som appen förväntar sig.
    """
    out: Dict[str, Any] = {
        "name": "", "currency": "", "price": 0.0, "sector": "", "industry": "",
        "shares_outstanding": 0.0, "market_cap": 0.0,
        "ps_ttm": 0.0, "pb": 0.0, "ev_ebitda": 0.0,
        "dividend_rate": 0.0, "dividend_yield_pct": 0.0, "payout_ratio_pct": 0.0,
        "book_value_per_share": 0.0,
        "gross_margins_pct": 0.0, "operating_margins_pct": 0.0, "profit_margins_pct": 0.0,
        "revenue_ttm": 0.0, "revenue_growth_pct": 0.0,
        "enterprise_value": 0.0, "ebitda": 0.0,
        "cagr5_pct": 0.0,
        # NYTT – FCF-relaterat
        "fcf_ttm": 0.0,
        "fcf_yield_pct": 0.0,
        "dividends_ttm_ps": 0.0,
        "payout_fcf_pct": 0.0,
    }

    if yf is None or not ticker:
        return out

    tkr = ticker.strip().upper()
    t = yf.Ticker(tkr)

    # -- Basinfo/price
    info = {}
    try:
        info = t.info or {}
    except Exception:
        info = {}

    out["name"]     = str(info.get("shortName") or info.get("longName") or "")
    out["currency"] = str(info.get("currency") or "USD").upper()
    out["sector"]   = str(info.get("sector") or "")
    out["industry"] = str(info.get("industry") or "")

    out["price"] = _get_price(t)
    out["market_cap"] = _safe_float(info.get("marketCap"))
    out["shares_outstanding"] = _safe_float(info.get("sharesOutstanding"))

    # Multiplar
    out["ps_ttm"] = _safe_float(info.get("priceToSalesTrailing12Months"))
    out["pb"]     = _safe_float(info.get("priceToBook"))
    out["ev_ebitda"] = _safe_float(info.get("enterpriseToEbitda"))

    # EV/EBITDA råfält
    out["enterprise_value"] = _safe_float(info.get("enterpriseValue"))
    out["ebitda"]           = _safe_float(info.get("ebitda"))

    # Marginaler (decimal → %)
    out["gross_margins_pct"]     = _safe_float(info.get("grossMargins"))     * 100.0
    out["operating_margins_pct"] = _safe_float(info.get("operatingMargins")) * 100.0
    out["profit_margins_pct"]    = _safe_float(info.get("profitMargins"))    * 100.0

    # Book value / share
    out["book_value_per_share"] = _safe_float(info.get("bookValue"))

    # Revenue TTM + growth
    rev_ttm, growth_pct = _revenue_ttm_and_growth(t)
    out["revenue_ttm"]        = rev_ttm
    out["revenue_growth_pct"] = growth_pct

    # CAGR 5 år
    out["cagr5_pct"] = _cagr5_from_revenue(t)

    # Utdelning
    # Primärt: dividendRate från info (årlig). Komplettera med TTM per aktie.
    div_rate_info = _safe_float(info.get("dividendRate"))
    divs_ttm_ps = _sum_dividends_ttm(t)
    out["dividends_ttm_ps"] = divs_ttm_ps
    out["dividend_rate"] = div_rate_info if div_rate_info > 0 else divs_ttm_ps

    if out["price"] > 0 and out["dividend_rate"] > 0:
        out["dividend_yield_pct"] = (out["dividend_rate"] / out["price"]) * 100.0

    # EPS-payout (fallback)
    trailing_eps = _safe_float(info.get("trailingEps"))
    if trailing_eps > 0 and out["dividends_ttm_ps"] > 0:
        out["payout_ratio_pct"] = (out["dividends_ttm_ps"] / trailing_eps) * 100.0
    else:
        # info kan innehålla redan beräknad payoutRatio (decimal)
        pr = _safe_float(info.get("payoutRatio")) * 100.0
        if pr > 0:
            out["payout_ratio_pct"] = pr

    # FCF (TTM) och FCF-baserade mått
    fcf_ttm = _compute_fcf_ttm(t)
    out["fcf_ttm"] = fcf_ttm

    if out["market_cap"] > 0 and fcf_ttm != 0:
        out["fcf_yield_pct"] = (fcf_ttm / out["market_cap"]) * 100.0

    # Payout (FCF) i procent, baserat på per-aktie
    sh_out = out["shares_outstanding"]
    if sh_out > 0 and fcf_ttm != 0:
        fcf_ps = fcf_ttm / sh_out
        if fcf_ps != 0:
            out["payout_fcf_pct"] = (out["dividends_ttm_ps"] / fcf_ps) * 100.0

    # Fallback-beräkningar om ps/pb saknas
    if (out["ps_ttm"] == 0.0) and (out["market_cap"] > 0) and (out["revenue_ttm"] > 0):
        out["ps_ttm"] = out["market_cap"] / out["revenue_ttm"]

    # Fallback EV/EBITDA
    if (out["ev_ebitda"] == 0.0) and (out["enterprise_value"] > 0) and (out["ebitda"] > 0):
        out["ev_ebitda"] = out["enterprise_value"] / out["ebitda"]

    return out
