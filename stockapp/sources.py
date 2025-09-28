# stockapp/sources.py
# -*- coding: utf-8 -*-
"""
Datakällor / Runners:
- run_update_price_only(ticker): hämtar snabb basdata (pris/valuta/namn/utdelning)
- run_update_full(ticker): hämtar pris/valuta/namn + implied shares + kvartalsintäkter
  via Yahoo (yfinance), räknar TTM-intäkter, P/S (TTM) nu och P/S Q1–Q4.

Bägge returnerar (vals: dict, debug: dict).

OBS: Denna modul använder endast yfinance & Yahoo-data (stabilt och gratis).
SEC/FMP kan läggas till senare om du vill, men detta ger en robust baseline.
"""

from __future__ import annotations

from typing import Dict, Tuple, List, Optional
from datetime import timedelta, date, datetime

import numpy as np
import pandas as pd
import yfinance as yf


# ------------------------------------------------------------
# Hjälpare
# ------------------------------------------------------------
def _safe_float(x, default=0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _yyyymmdd(d: date) -> str:
    try:
        return d.strftime("%Y-%m-%d")
    except Exception:
        return str(d)


def _yfi_info(tkr: yf.Ticker) -> dict:
    try:
        return tkr.info or {}
    except Exception:
        return {}


def _get_price_name_currency_dividend(ticker: str) -> Dict[str, float]:
    """
    Basfält från Yahoo/yfinance:
      - Aktuell kurs
      - Valuta
      - Bolagsnamn
      - Årlig utdelning (dividendRate om finns)
      - MarketCap (för implied shares)
    """
    out = {
        "Aktuell kurs": 0.0,
        "Valuta": "USD",
        "Bolagsnamn": "",
        "Årlig utdelning": 0.0,
        "_marketCap": 0.0,
    }
    t = yf.Ticker(ticker)

    info = _yfi_info(t)

    # Pris
    price = info.get("regularMarketPrice")
    if price is None:
        try:
            h = t.history(period="1d")
            if not h.empty and "Close" in h.columns:
                price = float(h["Close"].iloc[-1])
        except Exception:
            price = None
    if price is not None:
        out["Aktuell kurs"] = _safe_float(price, 0.0)

    # Valuta
    cur = info.get("currency")
    if cur:
        out["Valuta"] = str(cur).upper()

    # Namn
    name = info.get("shortName") or info.get("longName") or ""
    out["Bolagsnamn"] = str(name)

    # Utdelning (årstakt – Yahoo dividendRate)
    div_rate = info.get("dividendRate")
    if div_rate is not None:
        out["Årlig utdelning"] = _safe_float(div_rate, 0.0)

    # Market cap (behövs för implied shares)
    mc = info.get("marketCap")
    if mc is not None:
        out["_marketCap"] = _safe_float(mc, 0.0)

    return out


def _implied_shares(info_dict: dict) -> float:
    """
    Försök härleda antal utestående aktier = marketCap / price.
    Om det saknas pris eller MCAP → 0.0
    Returnerar antal aktier (styck, ej miljoner).
    """
    px = _safe_float(info_dict.get("Aktuell kurs"), 0.0)
    mcap = _safe_float(info_dict.get("_marketCap"), 0.0)
    if px > 0 and mcap > 0:
        try:
            return float(mcap / px)
        except Exception:
            return 0.0
    return 0.0


def _quarterly_revenues_yahoo(ticker: str) -> List[Tuple[date, float]]:
    """
    Läser kvartalsintäkter via yfinance:
    - Först via quarterly_financials (rad 'Total Revenue'/varianter)
    - Fallback via income_stmt (om quarterly_financials saknas)
    Returnerar lista [(period_end_date, value), ...] nyast→äldst
    """
    out: List[Tuple[date, float]] = []
    t = yf.Ticker(ticker)

    # 1) quarterly_financials
    try:
        qf = t.quarterly_financials
        if isinstance(qf, pd.DataFrame) and not qf.empty:
            idx = [str(x).strip() for x in qf.index]
            cand_rows = [
                "Total Revenue", "TotalRevenue", "Revenues", "Revenue", "Sales",
                "Total revenue", "Revenues from contracts with customers"
            ]
            for key in cand_rows:
                if key in idx:
                    row = qf.loc[key].dropna()
                    tmp = []
                    for c, v in row.items():
                        try:
                            d = c.date() if hasattr(c, "date") else pd.to_datetime(c).date()
                            tmp.append((d, _safe_float(v, 0.0)))
                        except Exception:
                            pass
                    tmp.sort(key=lambda x: x[0], reverse=True)
                    out = tmp
                    break
    except Exception:
        pass

    if out:
        return out

    # 2) fallback: income_stmt (quarterly)
    try:
        df_is = getattr(t, "income_stmt", None)
        if isinstance(df_is, pd.DataFrame) and not df_is.empty and "Total Revenue" in df_is.index:
            ser = df_is.loc["Total Revenue"].dropna()
            tmp = []
            for c, v in ser.items():
                try:
                    d = c.date() if hasattr(c, "date") else pd.to_datetime(c).date()
                    tmp.append((d, _safe_float(v, 0.0)))
                except Exception:
                    pass
            tmp.sort(key=lambda x: x[0], reverse=True)
            out = tmp
    except Exception:
        pass

    return out


def _history_prices_for_dates(ticker: str, dates: List[date]) -> Dict[date, float]:
    """
    Tar dagliga Close i ett spann som täcker datumen och returnerar närmast <= datumet.
    """
    if not dates:
        return {}
    dmin = min(dates) - timedelta(days=14)
    dmax = max(dates) + timedelta(days=2)
    try:
        t = yf.Ticker(ticker)
        hist = t.history(start=dmin, end=dmax, interval="1d")
        if hist is None or hist.empty:
            return {}
        hist = hist.sort_index()
        idx_dates = list(hist.index.date)
        closes = list(hist["Close"].values)
        out = {}
        for d in dates:
            px = None
            # gå bakifrån tills vi hittar närmast <= d
            for j in range(len(idx_dates) - 1, -1, -1):
                if idx_dates[j] <= d:
                    try:
                        px = float(closes[j])
                    except Exception:
                        px = None
                    break
            if px is not None:
                out[d] = px
        return out
    except Exception:
        return {}


def _ttm_windows(values: List[Tuple[date, float]], need: int = 5) -> List[Tuple[date, float]]:
    """
    Bygg TTM-summor från kvartalsvärden nyast→äldst.
    Returnerar upp till 'need' st:
      [(end_date0, ttm0), (end_date1, ttm1), ...]
    """
    out: List[Tuple[date, float]] = []
    if len(values) < 4:
        return out
    for i in range(0, min(need, len(values) - 3)):
        end_i = values[i][0]
        ttm_i = float(sum(v for (_, v) in values[i:i + 4]))
        out.append((end_i, ttm_i))
    return out


def _cagr_from_financials(ticker: str) -> float:
    """
    Grov CAGR% från årliga 'Total Revenue' via yfinance.financials.
    """
    try:
        t = yf.Ticker(ticker)
        df_fin = t.financials  # OBS: årlig
        if isinstance(df_fin, pd.DataFrame) and not df_fin.empty and "Total Revenue" in df_fin.index:
            ser = df_fin.loc["Total Revenue"].dropna().sort_index()
            if len(ser) >= 2:
                start = _safe_float(ser.iloc[0], 0.0)
                end = _safe_float(ser.iloc[-1], 0.0)
                n = max(1, len(ser) - 1)
                if start > 0:
                    cagr = (end / start) ** (1.0 / n) - 1.0
                    return round(cagr * 100.0, 2)
    except Exception:
        pass
    return 0.0


# ------------------------------------------------------------
# Runners
# ------------------------------------------------------------
def run_update_price_only(ticker: str) -> Tuple[Dict[str, object], Dict[str, object]]:
    """
    Snabb uppdatering: pris/valuta/namn/utdelning. Ingen P/S-beräkning.
    Returnerar (vals, debug).
    """
    base = _get_price_name_currency_dividend(ticker)
    vals: Dict[str, object] = {
        "Bolagsnamn": base.get("Bolagsnamn", ""),
        "Valuta": base.get("Valuta", "USD"),
        "Aktuell kurs": _safe_float(base.get("Aktuell kurs"), 0.0),
        "Årlig utdelning": _safe_float(base.get("Årlig utdelning"), 0.0),
    }
    debug = {"source": "yfinance", "ticker": ticker, "have_marketCap": bool(base.get("_marketCap", 0.0))}
    return vals, debug


def run_update_full(ticker: str) -> Tuple[Dict[str, object], Dict[str, object]]:
    """
    Full uppdatering med:
      - Bolagsnamn, Valuta, Aktuell kurs, Årlig utdelning
      - Utestående aktier (implied via marketCap/price) i miljoner
      - Kvartalsintäkter → TTM serier → P/S (TTM) nu + P/S Q1..Q4
      - CAGR 5 år (%) från årliga intäkter (grovt)
    Returnerar (vals, debug).
    """
    base = _get_price_name_currency_dividend(ticker)
    vals: Dict[str, object] = {
        "Bolagsnamn": base.get("Bolagsnamn", ""),
        "Valuta": base.get("Valuta", "USD"),
        "Aktuell kurs": _safe_float(base.get("Aktuell kurs"), 0.0),
        "Årlig utdelning": _safe_float(base.get("Årlig utdelning"), 0.0),
    }

    px = _safe_float(vals["Aktuell kurs"], 0.0)
    implied = _implied_shares(base)  # styck
    shares_mn = 0.0
    if implied > 0:
        shares_mn = float(implied) / 1e6
        vals["Utestående aktier"] = round(shares_mn, 6)

    # Kvartalsintäkter (nyast→äldst)
    q_rows = _quarterly_revenues_yahoo(ticker)  # [(date, revenue), ...]
    debug_q = {"len_quarters": len(q_rows), "head": [(str(d), float(v)) for (d, v) in q_rows[:4]]}

    # Bygg TTM (upp till 5 fönster så att vi kan få Q1..Q4 historik + ett extra om finns)
    ttm_list = _ttm_windows(q_rows, need=5)  # [(end_date, ttm_value)]
    debug_ttm = {"len_ttm": len(ttm_list), "head": [(str(d), float(v)) for (d, v) in ttm_list[:5]]}

    # P/S-beräkningar
    ps_vals = {}
    if shares_mn > 0 and px > 0 and ttm_list:
        # hämta historiska priser för respektive TTM-end-date (närmast <= d)
        dates = [d for (d, _) in ttm_list[:5]]
        px_map = _history_prices_for_dates(ticker, dates)
        # nuvarande MCAP (om möjligt via info, annars implied)
        mcap_now = _safe_float(base.get("_marketCap"), 0.0)
        if mcap_now <= 0 and implied > 0 and px > 0:
            mcap_now = implied * px

        # P/S TTM nu (fönster 0)
        ttm0 = float(ttm_list[0][1])
        if ttm0 > 0 and mcap_now > 0:
            vals["P/S"] = float(mcap_now / ttm0)

        # P/S Q1..Q4 – använd samma shares (implied) och pris vid respektive end-date
        # Q1 = nyaste TTM (0), Q2 = (1), Q3 = (2), Q4 = (3)
        for i in range(4):
            if i >= len(ttm_list):
                break
            d_end, ttm_val = ttm_list[i]
            if ttm_val and ttm_val > 0:
                px_hist = _safe_float(px_map.get(d_end), 0.0)
                if implied > 0 and px_hist > 0:
                    mcap_hist = implied * px_hist
                    vals[f"P/S Q{i+1}"] = float(mcap_hist / ttm_val)

    # CAGR (grovt)
    vals["CAGR 5 år (%)"] = _cagr_from_financials(ticker)

    debug = {
        "source": "yfinance",
        "ticker": ticker,
        "base_have_mcap": bool(base.get("_marketCap", 0.0)),
        "quarters": debug_q,
        "ttm": debug_ttm,
        "implied_shares": implied,
    }
    return vals, debug
