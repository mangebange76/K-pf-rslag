# stockapp/fetchers/yahoo.py
from __future__ import annotations

import math
from typing import Dict, Any

import pandas as pd
import yfinance as yf


def _f(x, mult: float | None = None) -> float:
    """Safe float; valfritt multiplicera (t.ex. för procent→%)"""
    try:
        v = float(x)
        if mult is not None:
            v *= mult
        if math.isnan(v) or math.isinf(v):
            return 0.0
        return v
    except Exception:
        return 0.0


def _cagr_from_financials(tkr: yf.Ticker) -> float:
    """Fall-back CAGR (5y) från årsredovisade intäkter om det behövs."""
    try:
        df_fin = getattr(tkr, "financials", None)
        if isinstance(df_fin, pd.DataFrame) and not df_fin.empty and "Total Revenue" in df_fin.index:
            series = df_fin.loc["Total Revenue"].dropna()
        else:
            # yfinance har ibland .income_stmt
            df_is = getattr(tkr, "income_stmt", None)
            if isinstance(df_is, pd.DataFrame) and not df_is.empty and "Total Revenue" in df_is.index:
                series = df_is.loc["Total Revenue"].dropna()
            else:
                return 0.0
        if series.empty or len(series) < 2:
            return 0.0
        series = series.sort_index()
        start = float(series.iloc[0]); end = float(series.iloc[-1])
        years = max(1, len(series) - 1)
        if start <= 0:
            return 0.0
        cagr = (end / start) ** (1.0 / years) - 1.0
        return round(cagr * 100.0, 2)
    except Exception:
        return 0.0


def get_all(ticker: str) -> Dict[str, Any]:
    """
    Hämtar ett brett paket nyckeltal från Yahoo.
    Nycklar (alla floats om inget annat nämns):
      name (str), price, currency (str),
      shares_outstanding, market_cap,
      ps_ttm, pb, enterprise_value, ebitda, ev_ebitda,
      dividend_rate, dividend_yield_pct, payout_ratio_pct,
      revenue_ttm, revenue_growth_pct,
      gross_margins_pct, operating_margins_pct, profit_margins_pct,
      book_value_per_share,
      cagr5_pct
    """
    out: Dict[str, Any] = {
        "name": "",
        "price": 0.0,
        "currency": "USD",
        "shares_outstanding": 0.0,
        "market_cap": 0.0,
        "ps_ttm": 0.0,
        "pb": 0.0,
        "enterprise_value": 0.0,
        "ebitda": 0.0,
        "ev_ebitda": 0.0,
        "dividend_rate": 0.0,
        "dividend_yield_pct": 0.0,
        "payout_ratio_pct": 0.0,
        "revenue_ttm": 0.0,
        "revenue_growth_pct": 0.0,
        "gross_margins_pct": 0.0,
        "operating_margins_pct": 0.0,
        "profit_margins_pct": 0.0,
        "book_value_per_share": 0.0,
        "cagr5_pct": 0.0,
    }

    try:
        t = yf.Ticker(ticker)
        info: Dict[str, Any] = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}

        # Bas
        pris = info.get("regularMarketPrice")
        if pris is None:
            h = t.history(period="1d")
            if not h.empty and "Close" in h:
                pris = float(h["Close"].iloc[-1])

        out["price"] = _f(pris)
        if info.get("currency"):
            out["currency"] = str(info.get("currency")).upper()
        namn = info.get("shortName") or info.get("longName") or ""
        if namn:
            out["name"] = str(namn)

        # Storlek/aktier
        out["shares_outstanding"] = _f(info.get("sharesOutstanding"))
        out["market_cap"] = _f(info.get("marketCap"))

        # Multiplar
        out["ps_ttm"] = _f(info.get("priceToSalesTrailing12Months"))
        out["pb"] = _f(info.get("priceToBook"))

        # EV / EBITDA
        ev = _f(info.get("enterpriseValue"))
        ebitda = _f(info.get("ebitda"))
        out["enterprise_value"] = ev
        out["ebitda"] = ebitda
        out["ev_ebitda"] = (ev / ebitda) if (ebitda and ebitda != 0) else 0.0

        # Utdelning
        out["dividend_rate"] = _f(info.get("dividendRate"))
        out["dividend_yield_pct"] = _f(info.get("dividendYield"), 100.0)  # 0.023 -> 2.3
        out["payout_ratio_pct"] = _f(info.get("payoutRatio"), 100.0)      # 0.35 -> 35

        # Tillväxt/lönsamhet
        out["revenue_ttm"] = _f(info.get("totalRevenue") or info.get("revenueTTM"))
        out["revenue_growth_pct"] = _f(info.get("revenueGrowth"), 100.0)
        out["gross_margins_pct"] = _f(info.get("grossMargins"), 100.0)
        out["operating_margins_pct"] = _f(info.get("operatingMargins"), 100.0)
        out["profit_margins_pct"] = _f(info.get("profitMargins"), 100.0)
        out["book_value_per_share"] = _f(info.get("bookValue"))

        # Kompletterande kvalitetsmått
        out["cagr5_pct"] = _cagr_from_financials(t)

    except Exception:
        # Låt out vara tomma defaults
        pass

    return out
