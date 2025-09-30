# -*- coding: utf-8 -*-
"""
stockapp/fetchers/yahoo.py

Yahoo-hämtare:
- Pris/valuta/namn/sektor/bransch/market cap/shares via yfinance
- Kvartalsintäkter (income statement) -> TTM-summor (upp till 4 fönster)
- Valutakonvertering (Frankfurter -> exchangerate.host) till prisvalutan
- P/S nu + P/S Q1..Q4 med historiska priser
- Extra nyckeltal (EV/EBITDA, Debt/Equity, bruttomarginal, nettomarginal) om tillgängligt

Returnerar: (vals, debug)
"""

from __future__ import annotations
from typing import Dict, Tuple, Any, List, Optional
import datetime as dt
import requests
import streamlit as st
import yfinance as yf

# ----------------------------- Hjälpare --------------------------------------


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v != v:  # NaN
            return default
        return v
    except Exception:
        return default


def _fx_rate(base: str, quote: str) -> float:
    """
    Dagens FX via Frankfurter -> exchangerate.host fallback.
    """
    base = (base or "").upper()
    quote = (quote or "").upper()
    if not base or not quote or base == quote:
        return 1.0
    # Frankfurter
    try:
        r = requests.get("https://api.frankfurter.app/latest", params={"from": base, "to": quote}, timeout=12)
        if r.status_code == 200:
            v = (r.json() or {}).get("rates", {}).get(quote)
            if v:
                return float(v)
    except Exception:
        pass
    # exchangerate.host
    try:
        r = requests.get("https://api.exchangerate.host/latest", params={"base": base, "symbols": quote}, timeout=12)
        if r.status_code == 200:
            v = (r.json() or {}).get("rates", {}).get(quote)
            if v:
                return float(v)
    except Exception:
        pass
    return 0.0


def _yahoo_prices_for_dates(ticker: str, dates: List[dt.date]) -> Dict[dt.date, float]:
    """
    Hämtar 'Close' för var och en av 'dates' (eller närmast föregående handelsdag).
    """
    if not dates:
        return {}
    dmin = min(dates) - dt.timedelta(days=14)
    dmax = max(dates) + dt.timedelta(days=2)
    try:
        t = yf.Ticker(ticker)
        hist = t.history(start=dmin, end=dmax, interval="1d")
        if hist is None or hist.empty:
            return {}
        hist = hist.sort_index()
        out = {}
        idx = list(hist.index.date)
        closes = list(hist["Close"].values)
        for d in dates:
            px = None
            for j in range(len(idx) - 1, -1, -1):
                if idx[j] <= d:
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


def _ttm_windows(values: List[Tuple[dt.date, float]], need: int = 4) -> List[Tuple[dt.date, float]]:
    """
    Tar [(end_date, kvartalsintäkt), ...] (nyast→äldst) och bygger upp till 'need' TTM-summor:
    [(end_date0, ttm0), (end_date1, ttm1), ...] där ttm0=sum(q0..q3), ttm1=sum(q1..q4), osv.
    """
    out: List[Tuple[dt.date, float]] = []
    if len(values) < 4:
        return out
    for i in range(0, min(need, len(values) - 3)):
        end_i = values[i][0]
        ttm_i = sum(v for (_, v) in values[i:i+4])
        out.append((end_i, float(ttm_i)))
    return out


# ----------------------------- Yahoo-parsing ---------------------------------


def _yahoo_basics(ticker: str) -> Dict[str, Any]:
    """
    Pris, valuta, namn, sektor, bransch, market cap, sharesOutstanding via yfinance.
    """
    out: Dict[str, Any] = {"Valuta": "USD"}
    try:
        t = yf.Ticker(ticker)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}

        # Pris
        px = info.get("regularMarketPrice")
        if px is None:
            try:
                h = t.history(period="1d")
                if not h.empty and "Close" in h:
                    px = float(h["Close"].iloc[-1])
            except Exception:
                px = None
        if px is not None:
            out["Aktuell kurs"] = float(px)

        # Valuta (prisets valuta)
        cur = info.get("currency")
        if cur:
            out["Valuta"] = str(cur).upper()

        # Namn / sektor / bransch
        nm = info.get("shortName") or info.get("longName") or ""
        if nm:
            out["Bolagsnamn"] = str(nm)
        sec = info.get("sector")
        if sec:
            out["Sektor"] = sec
        ind = info.get("industry")
        if ind:
            out["Bransch"] = ind

        # Market cap
        mc = info.get("marketCap")
        if mc is not None:
            try:
                out["Market Cap"] = float(mc)
            except Exception:
                pass

        # Shares outstanding (kan vara fallback)
        so = info.get("sharesOutstanding")
        if so is not None:
            try:
                out["_yf_shares_out"] = float(so)  # styck
            except Exception:
                pass

        # EV/EBITDA (direkt från Yahoo om tillgängligt)
        ev_to_ebitda = info.get("enterpriseToEbitda")
        if ev_to_ebitda is not None:
            out["EV/EBITDA"] = _safe_float(ev_to_ebitda, 0.0)

        # Marginaler (andelar -> %)
        gp = info.get("grossMargins")  # ex. 0.58
        if gp is not None:
            out["Bruttomarginal (%)"] = float(gp) * 100.0
        pm = info.get("profitMargins")
        if pm is not None:
            out["Nettomarginal (%)"] = float(pm) * 100.0

        # För Debt/Equity tittar vi i balance sheet om möjligt
        try:
            bs = t.quarterly_balance_sheet
            if bs is not None and not bs.empty:
                cols = list(bs.columns)
                if cols:
                    last = cols[0]
                    total_debt = bs.get("TotalDebt")
                    equity = bs.get("TotalStockholderEquity") or bs.get("StockholdersEquity")
                    td = _safe_float(total_debt.loc[last], 0.0) if total_debt is not None and last in total_debt.index else 0.0
                    eq = _safe_float(equity.loc[last], 0.0) if equity is not None and last in equity.index else 0.0
                    if td > 0 and eq > 0:
                        out["Debt/Equity"] = td / eq
        except Exception:
            pass

        # Finansiell valuta (för intäkter)
        fin_cur = info.get("financialCurrency")
        if fin_cur:
            out["_financial_currency"] = str(fin_cur).upper()

    except Exception:
        pass
    return out


def _yahoo_quarterly_revenues(ticker: str) -> Tuple[List[Tuple[dt.date, float]], Optional[str]]:
    """
    Hämtar kvartalsintäkter (Yahoo income statement). Returnerar (rows, unit).
    rows: [(end_date, revenue_value), ...] nyast→äldst
    unit: finansiell valuta (financialCurrency) om tillgänglig, annars None
    """
    try:
        t = yf.Ticker(ticker)
        # income statement – försök båda egenskaper
        qis = None
        try:
            qis = t.quarterly_income_stmt
        except Exception:
            qis = None

        if qis is None or qis.empty:
            # fallback till quarterly_financials (vissa versioner)
            try:
                qf = t.quarterly_financials
                qis = qf
            except Exception:
                qis = None

        if qis is None or qis.empty:
            return [], None

        # Hitta rätt rad för intäkter
        # Vanliga namn: 'TotalRevenue', 'Total Revenue', 'Revenue'
        cand_names = ["TotalRevenue", "Total Revenue", "Revenue"]
        rev_series = None
        for nm in cand_names:
            if nm in qis.index:
                rev_series = qis.loc[nm]
                break
        if rev_series is None:
            # prova case-insensitive
            for idx in qis.index:
                if str(idx).replace(" ", "").lower() in ("totalrevenue", "revenue"):
                    rev_series = qis.loc[idx]
                    break

        if rev_series is None:
            return [], None

        # rev_series är en Series där index=kolumner (datum), values=revenue
        rows: List[Tuple[dt.date, float]] = []
        for col, val in rev_series.items():
            # col kan vara Timestamp/DatetimeIndex
            try:
                d = col.date() if hasattr(col, "date") else dt.datetime.fromisoformat(str(col)).date()
            except Exception:
                # bästa gissning
                try:
                    d = dt.datetime.strptime(str(col)[:10], "%Y-%m-%d").date()
                except Exception:
                    d = None
            if d is None:
                continue
            v = _safe_float(val, 0.0)
            if v > 0:
                rows.append((d, v))

        # sortera nyast -> äldst
        rows.sort(key=lambda t2: t2[0], reverse=True)

        # Försök läsa finansiell valuta
        fin_cur = None
        try:
            info = t.info or {}
            if info.get("financialCurrency"):
                fin_cur = str(info["financialCurrency"]).upper()
        except Exception:
            fin_cur = None

        return rows, fin_cur
    except Exception:
        return [], None


# ----------------------------- Publikt API -----------------------------------


def fetch_yahoo_combo(ticker: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Hämtar Yahoo-data och bygger ut P/S nu + P/S Q1..Q4 från TTM-intäkter.
    Returnerar (vals, debug).
    """
    vals: Dict[str, Any] = {}
    dbg: Dict[str, Any] = {"ticker": ticker, "source": "Yahoo"}

    # 1) Basics
    yb = _yahoo_basics(ticker)
    for k in ("Bolagsnamn", "Valuta", "Aktuell kurs", "Market Cap", "Sektor", "Bransch",
              "EV/EBITDA", "Bruttomarginal (%)", "Nettomarginal (%)", "Debt/Equity"):
        v = yb.get(k)
        if v not in (None, "", 0, 0.0):
            vals[k] = v

    px_ccy = (yb.get("Valuta") or "USD").upper()
    px_now = _safe_float(yb.get("Aktuell kurs"), 0.0)
    mcap_now = _safe_float(yb.get("Market Cap"), 0.0)

    # 2) Utestående aktier
    shares_used = 0.0
    if mcap_now > 0 and px_now > 0:
        shares_used = mcap_now / max(px_now, 1e-9)
        dbg["_shares_source"] = "Yahoo implied (mcap/price)"
    else:
        so = _safe_float(yb.get("_yf_shares_out"), 0.0)
        if so > 0:
            shares_used = so
            dbg["_shares_source"] = "Yahoo sharesOutstanding"
        else:
            dbg["_shares_source"] = "unknown"

    if shares_used > 0:
        vals["Utestående aktier"] = shares_used / 1e6  # i miljoner

    # 3) Kvartalsintäkter
    q_rows, fin_cur = _yahoo_quarterly_revenues(ticker)
    dbg["q_rows_count"] = len(q_rows)
    dbg["financialCurrency"] = fin_cur

    if not q_rows:
        return vals, dbg

    # 4) Bygg TTM-fönster
    ttm_list = _ttm_windows(q_rows, need=4)
    if not ttm_list:
        return vals, dbg

    # 5) Konvertera TTM till prisvaluta (om fin_cur avviker)
    conv = 1.0
    if fin_cur and fin_cur.upper() != px_ccy:
        conv = _fx_rate(fin_cur.upper(), px_ccy)
        if conv <= 0:
            conv = 1.0
    ttm_px = [(d, v * conv) for (d, v) in ttm_list]

    # 6) P/S (nu)
    if mcap_now > 0 and ttm_px:
        ltm_now = _safe_float(ttm_px[0][1], 0.0)
        if ltm_now > 0:
            vals["P/S"] = mcap_now / ltm_now

    # 7) P/S Q1..Q4 historiskt (kräver shares & historiska priser)
    if shares_used > 0 and ttm_px:
        q_dates = [d for (d, _) in ttm_px[:4]]
        px_map = _yahoo_prices_for_dates(ticker, q_dates)
        for idx, (d_end, ttm_rev) in enumerate(ttm_px[:4], start=1):
            if ttm_rev and ttm_rev > 0:
                px_at = _safe_float(px_map.get(d_end), 0.0)
                if px_at > 0:
                    mcap_hist = shares_used * px_at
                    vals[f"P/S Q{idx}"] = mcap_hist / ttm_rev

    return vals, dbg
