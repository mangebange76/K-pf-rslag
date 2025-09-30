# -*- coding: utf-8 -*-
"""
Robusta Yahoo-hämtare via yfinance (med försiktiga fallbacks).
Returnerar både "platta" fält och 4 senaste TTM-fönster för P/S.
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple
import math
import datetime as dt

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None


def _try(x, default=None, fn=float):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return default
        return fn(x)
    except Exception:
        return default


def _safe_pct(x):
    if x is None:
        return None
    try:
        return float(x) * 100.0 if abs(x) < 1.0 else float(x)
    except Exception:
        return None


def _closest_price(hist: pd.DataFrame, when: pd.Timestamp) -> float | None:
    if hist is None or hist.empty:
        return None
    # välj närmaste handelsdag <= when, annars närmast efter
    sidx = hist.index.get_indexer([when], method="pad")
    if sidx[0] == -1:
        sidx = hist.index.get_indexer([when], method="nearest")
    try:
        return float(hist.iloc[sidx[0]]["Close"])
    except Exception:
        try:
            return float(hist.iloc[sidx[0]]["Adj Close"])
        except Exception:
            return None


def _quarterly_revenue_series(t: "yf.Ticker") -> pd.Series | None:
    # i nyare yfinance:
    # t.quarterly_financials (rows), eller t.quarterly_income_stmt
    for getter in ("quarterly_financials", "quarterly_income_stmt", "quarterly_income_statement"):
        try:
            q = getattr(t, getter)
            if callable(q):
                q = q()
            if isinstance(q, pd.DataFrame) and not q.empty:
                # hitta "Total Revenue"
                for key in ("Total Revenue", "TotalRevenue", "Revenue", "TotalRevenueUSD"):
                    if key in q.index:
                        s = q.loc[key]
                        s = s.dropna()
                        if isinstance(s, pd.Series) and not s.empty:
                            s = s.sort_index()
                            return s
        except Exception:
            pass
    return None


def _balance_sheet_last(t: "yf.Ticker") -> pd.DataFrame | None:
    for getter in ("balance_sheet", "quarterly_balance_sheet"):
        try:
            bs = getattr(t, getter)
            if callable(bs):
                bs = bs()
            if isinstance(bs, pd.DataFrame) and not bs.empty:
                bs = bs.dropna(axis=1, how="all")
                if not bs.empty:
                    # ta sista kolumnen
                    return bs.iloc[:, [0]] if bs.shape[1] == 1 else bs.iloc[:, [-1]]
        except Exception:
            pass
    return None


def fetch_ticker_yahoo(ticker: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Hämtar nyckeltal + 4 senaste TTM-fönster (date, mcap, ps).
    Returnerar: (data_dict, ps_windows)
    """
    data: Dict[str, Any] = {}
    ps_windows: List[Dict[str, Any]] = []

    if yf is None:
        # yfinance saknas – returnera tomt
        data["Namn"] = None
        data["Sektor"] = None
        return data, ps_windows

    t = yf.Ticker(ticker)

    # ---- Basinfo
    info = {}
    try:
        info = t.fast_info or {}
    except Exception:
        pass

    # Fallback: .info (lite tyngre)
    if not info:
        try:
            info = t.info or {}
        except Exception:
            info = {}

    long_info = {}
    try:
        long_info = t.info or {}
    except Exception:
        long_info = {}

    # Namn / sektor / valuta
    name = long_info.get("longName") or long_info.get("shortName")
    sector = long_info.get("sector")
    currency = long_info.get("currency") or info.get("currency") or "USD"

    data["Namn"] = name
    data["Sektor"] = sector
    data["Valuta"] = currency

    # Market cap & shares
    mcap = _try(info.get("market_cap") or long_info.get("marketCap"))
    shares = _try(info.get("shares") or info.get("shares_outstanding") or long_info.get("sharesOutstanding"))
    if shares and not mcap:
        # om mcap saknas men vi kan räkna via pris
        last = _try(info.get("last_price") or info.get("currentPrice") or long_info.get("currentPrice"))
        if last and last > 0:
            mcap = float(shares) * float(last)

    data["Market Cap"] = mcap
    data["Utestående aktier (milj.)"] = _try(shares, fn=float, default=None)
    if data["Utestående aktier (milj.)"] is not None:
        data["Utestående aktier (milj.)"] = data["Utestående aktier (milj.)"] / 1e6

    # P/S nu (TTM)
    ps_now = _try(long_info.get("priceToSalesTrailing12Months") or info.get("price_to_sales"))
    data["P/S"] = ps_now

    # Marginaler, värderingar, skuld
    data["P/B"] = _try(long_info.get("priceToBook") or info.get("price_to_book"))
    data["Gross margin (%)"] = _safe_pct(long_info.get("grossMargins"))
    data["Operating margin (%)"] = _safe_pct(long_info.get("operatingMargins"))
    data["Net margin (%)"] = _safe_pct(long_info.get("profitMargins"))

    # FCF & utdelning
    fcf = _try(long_info.get("freeCashflow"))
    data["FCF (TTM)"] = fcf
    if mcap and fcf:
        data["FCF Yield (%)"] = float(fcf) / float(mcap) * 100.0
    else:
        data["FCF Yield (%)"] = None
    data["Dividend yield (%)"] = _safe_pct(long_info.get("dividendYield"))

    # Nettoskuld/EBITDA och Debt/Equity
    net_debt_ebitda = None
    debt_equity = None
    # Försök via balansräkning
    bs = _balance_sheet_last(t)
    if isinstance(bs, pd.DataFrame) and not bs.empty:
        total_debt = _try(bs.loc["Total Debt"].squeeze()) if "Total Debt" in bs.index else None
        total_equity = _try(bs.loc["Stockholders Equity"].squeeze()) if "Stockholders Equity" in bs.index else None
        total_cash = _try(bs.loc["Cash"].squeeze()) if "Cash" in bs.index else None
        ebitda = _try(long_info.get("ebitda"))
        if total_debt and total_cash is not None and ebitda and ebitda != 0:
            net_debt_ebitda = (float(total_debt) - float(total_cash)) / float(ebitda)
        if total_debt and total_equity and total_equity != 0:
            debt_equity = float(total_debt) / float(total_equity)
    data["Net debt / EBITDA"] = _try(net_debt_ebitda)
    data["Debt/Equity"] = _try(debt_equity)

    # ---- 4 senaste TTM-fönster för P/S
    # bygg på kvartalsintäkter + pris-historik
    try:
        rev = _quarterly_revenue_series(t)  # Series med index=periodslut
        if isinstance(rev, pd.Series) and len(rev) >= 4:

            rev = rev.sort_index()  # äldst->nyast
            # rulla TTM
            ttm = rev.rolling(4).sum().dropna()
            # ta de 4 senaste fönstren
            ttm = ttm.iloc[-4:]

            hist = None
            try:
                hist = t.history(period="2y", interval="1d", auto_adjust=False)
            except Exception:
                pass

            ps_windows = []
            q_labels = ["Q1", "Q2", "Q3", "Q4"]
            for i, (when, ttm_rev) in enumerate(ttm.items()):
                close = _closest_price(hist, when) if hist is not None else None
                if (not mcap) and shares and close:
                    mcap_at = float(shares) * float(close)
                else:
                    # om vi saknar bra pris – approximera med dagens mcap
                    mcap_at = mcap
                _ps = None
                if mcap_at and ttm_rev and ttm_rev != 0:
                    _ps = float(mcap_at) / float(ttm_rev)

                ps_windows.append({
                    "label": q_labels[i],
                    "date": pd.Timestamp(when).date().isoformat(),
                    "mcap": mcap_at,
                    "ps": _ps,
                })

            # Lägg dessutom ut P/S Q1..Q4 + datum + MCap
            for i, q in enumerate(ps_windows, 1):
                data[f"P/S Q{i}"] = q.get("ps")
                data[f"MCAP Q{i}"] = q.get("mcap")
                data[f"Period Q{i}"] = q.get("date")
    except Exception:
        pass

    return data, ps_windows
