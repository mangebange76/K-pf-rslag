# stockapp/fetchers/yahoo.py
# -*- coding: utf-8 -*-
"""
Yahoo Finance-hämtare (via yfinance om möjligt).

Publikt API:
    get_live_price(ticker) -> Optional[float]
    fetch_ticker(ticker)   -> Dict[str, Any]  (fält mappade för orchestratorn)

Returnerade nycklar (om tillgängliga):
    currency                -> "USD" / "SEK" / ...
    price                   -> float
    shares_outstanding      -> float (antal aktier)
    market_cap              -> float (i bolagets valuta)
    ps_ttm                  -> float (TTM P/S)
    sector                  -> str
    industry                -> str
    gross_margin            -> float (%)  [0..100]
    net_margin              -> float (%)  [0..100]
    debt_to_equity          -> float (kvot)
    ev_ebitda               -> float
    fcf_m                   -> float (miljoner, valuta = company currency)
    cash_m                  -> float (miljoner)
    runway_quarters         -> float (≈ hur många kvartal kassan räcker vid negativ FCF)
    dividend_yield_pct      -> float (%)
    payout_ratio_cf_pct     -> float (%)  [oftast ej tillgänglig via Yahoo -> lämnas tomt]
"""

from __future__ import annotations
from typing import Any, Dict, Optional

import math
import numpy as np

# yfinance är frivilligt – modul kör utan om det saknas
try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None  # type: ignore


# -----------------------------
# Hjälp
# -----------------------------
def _to_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).replace(",", ".")
        v = float(s)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _safe_info_get(info: Dict[str, Any], *keys, factor: float = 1.0) -> Optional[float]:
    for k in keys:
        if k in info and info[k] is not None:
            v = _to_float(info[k])
            if v is not None:
                return v * factor
    return None


def _get_cashflow_free_cf(t: "yf.Ticker") -> Optional[float]:
    """
    Försök plocka 'Free Cash Flow' (senaste års-värde) ur cashflow-tabellen.
    Returnerar belopp (i samma valuta som bolaget) – inte i miljoner.
    """
    try:
        # yfinance har lite olika attribut beroende på version:
        # Pröva annual_cashflow först (nyare API):
        cf = getattr(t, "cashflow", None)
        if cf is None or cf.empty:
            cf = getattr(t, "annual_cashflow", None)
        if cf is None or cf.empty:
            return None

        # Rader kan heta olika, normalisera:
        # Vanliga nycklar: "Free Cash Flow" / "FreeCashFlow" / "freeCashFlow"
        candidates = [
            "Free Cash Flow",
            "FreeCashFlow",
            "freeCashFlow",
        ]
        # Ibland är index inte strängar – konvertera:
        idx = [str(i) for i in cf.index]
        cf.index = idx

        row_key = None
        for c in candidates:
            if c in cf.index:
                row_key = c
                break
        if row_key is None:
            # ibland ligger värden i 'Operating Cash Flow' minus 'Capital Expenditures'
            if "Operating Cash Flow" in cf.index and "Capital Expenditure" in cf.index:
                ocf = _to_float(cf.loc["Operating Cash Flow"].iloc[0])
                capex = _to_float(cf.loc["Capital Expenditure"].iloc[0])
                if ocf is not None and capex is not None:
                    return ocf - capex
            return None

        val = _to_float(cf.loc[row_key].iloc[0])
        return val
    except Exception:
        return None


def _get_total_cash(t: "yf.Ticker") -> Optional[float]:
    """
    Hämtar totalCash från balansräkning (annual balance sheet).
    Returnerar belopp (inte i miljoner).
    """
    try:
        bs = getattr(t, "balance_sheet", None)
        if bs is None or bs.empty:
            bs = getattr(t, "annual_balance_sheet", None)
        if bs is None or bs.empty:
            return None
        idx = [str(i) for i in bs.index]
        bs.index = idx
        # Vanliga nycklar: "Cash And Cash Equivalents", "CashAndCashEquivalents"
        for key in ["Cash And Cash Equivalents", "CashAndCashEquivalents", "cashAndCashEquivalents"]:
            if key in bs.index:
                return _to_float(bs.loc[key].iloc[0])
        # fallback: "Total Cash" förekommer ibland
        for key in ["Total Cash", "totalCash"]:
            if key in bs.index:
                return _to_float(bs.loc[key].iloc[0])
        return None
    except Exception:
        return None


# -----------------------------
# Publika funktioner
# -----------------------------
def get_live_price(ticker: str) -> Optional[float]:
    """
    Snabbkurs med yfinance. Fallback via history() om fast_info saknas.
    """
    if yf is None:
        return None
    try:
        t = yf.Ticker(ticker)
        # fast_info är snabbast
        fi = getattr(t, "fast_info", {}) or {}
        for k in ("last_price", "lastPrice", "regularMarketPrice", "last"):
            v = fi.get(k)
            v = _to_float(v)
            if v is not None:
                return v

        # fallback: hämta senaste close
        hist = t.history(period="1d", interval="1m")
        if hist is not None and not hist.empty:
            price = hist["Close"].dropna()
            if not price.empty:
                return _to_float(price.iloc[-1])
        # ytterligare fallback: 1d daglig close
        hist = t.history(period="1d")
        if hist is not None and not hist.empty:
            return _to_float(hist["Close"].dropna().iloc[-1])
        return None
    except Exception:
        return None


def fetch_ticker(ticker: str) -> Dict[str, Any]:
    """
    Hämtar ett robust paket nyckeltal för en ticker via yfinance.
    Returnerar ett dict med fält som orchestratorn förväntar sig (se modul-docstring).
    Fält som inte går att få fram utelämnas hellre än att sättas till 0.
    """
    out: Dict[str, Any] = {}

    if yf is None:
        return out

    try:
        t = yf.Ticker(ticker)

        # -- pris & valuta
        price = get_live_price(ticker)
        if price is not None:
            out["price"] = float(price)

        # fast_info först (snabbt), info som fallback
        fi = getattr(t, "fast_info", {}) or {}
        try:
            info = t.get_info() or {}
        except Exception:
            # vissa endpoints kan kasta – ignorera och kör med tom dict
            info = {}

        # valuta
        curr = None
        for key in ("currency", "financialCurrency"):
            if key in fi and fi[key]:
                curr = fi[key]
                break
            if key in info and info[key]:
                curr = info[key]
                break
        if curr:
            out["currency"] = str(curr).upper()

        # market cap
        mcap = None
        # fast_info kan ha "market_cap" i nyare yfinance
        for key in ("market_cap", "marketCap"):
            v = fi.get(key)
            if v is None and key in info:
                v = info.get(key)
            mcap = _to_float(v)
            if mcap is not None:
                break
        if mcap is not None:
            out["market_cap"] = float(mcap)

        # antal aktier
        shares = None
        for key in ("shares", "sharesOutstanding"):
            v = fi.get(key)
            if v is None:
                v = info.get(key)
            shares = _to_float(v)
            if shares is not None:
                break
        if shares is not None and shares > 0:
            out["shares_outstanding"] = float(shares)

        # P/S TTM
        ps_ttm = None
        for key in ("priceToSalesTrailing12Months", "priceToSalesTTM", "p2s"):
            v = info.get(key)
            ps_ttm = _to_float(v)
            if ps_ttm is not None:
                break
        if ps_ttm is not None and ps_ttm > 0:
            out["ps_ttm"] = float(ps_ttm)

        # sektor / industri
        sector = info.get("sector") or info.get("sectorDisp")
        industry = info.get("industry") or info.get("industryDisp")
        if sector:
            out["sector"] = str(sector)
        if industry:
            out["industry"] = str(industry)

        # marginaler (procent)
        gm = _safe_info_get(info, "grossMargins", factor=100.0)
        if gm is not None:
            out["gross_margin"] = float(gm)
        nm = _safe_info_get(info, "profitMargins", factor=100.0)
        if nm is not None:
            out["net_margin"] = float(nm)

        # Debt/Equity (Yahoo: debtToEquity ibland i procent – ofta redan kvot * 100 eller ren kvot)
        de = _safe_info_get(info, "debtToEquity")
        if de is not None:
            out["debt_to_equity"] = float(de)

        # EV/EBITDA
        ev_ebitda = _safe_info_get(info, "enterpriseToEbitda")
        if ev_ebitda is not None:
            out["ev_ebitda"] = float(ev_ebitda)

        # Utdelning & yield
        # dividendYield är oftast kvot (0.02 → 2%)
        dy = _safe_info_get(info, "dividendYield", factor=100.0)
        if dy is None:
            # försök räkna själv via trailingAnnualDividendRate / price
            div_rate = _safe_info_get(info, "trailingAnnualDividendRate")
            if div_rate is not None and price and price > 0:
                dy = (div_rate / price) * 100.0
        if dy is not None:
            out["dividend_yield_pct"] = float(dy)

        # Payout ratio via kassaflöde finns sällan hos Yahoo – lämnas tomt.
        # Vill vi fylla på senare från andra källor (FMP), låt orchestratorn göra det.
        # out["payout_ratio_cf_pct"] = ...

        # Kassaflöde & kassa
        fcf = _get_cashflow_free_cf(t)
        if fcf is not None:
            out["fcf_m"] = float(fcf) / 1e6  # miljoner
        total_cash = _get_total_cash(t)
        if total_cash is not None:
            out["cash_m"] = float(total_cash) / 1e6  # miljoner

        # Runway (kvartal) – grovt antagande: om FCF < 0 → bränns årligen, dela till kvartal.
        if total_cash is not None and fcf is not None and fcf < 0:
            cash_m = float(total_cash) / 1e6
            burn_m_per_year = abs(float(fcf)) / 1e6
            if burn_m_per_year > 0:
                out["runway_quarters"] = float(4.0 * (cash_m / burn_m_per_year))

        return out
    except Exception:
        # Misslyckas vi helt – returnera det vi har (kan vara tomt dict)
        return out
