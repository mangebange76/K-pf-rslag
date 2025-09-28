# stockapp/sources.py
# -*- coding: utf-8 -*-

from typing import Dict, Tuple, List, Optional
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta, datetime

# =========================
# Hjälpfunktioner (lokala)
# =========================

def _yfi_info_dict(t: yf.Ticker) -> dict:
    try:
        # yfinance >=0.2.x har fast_info; men info behövs ibland för namn/valuta
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}
        return info or {}
    except Exception:
        return {}

def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _get_last_price(t: yf.Ticker) -> Optional[float]:
    # prova fast_info
    try:
        fi = getattr(t, "fast_info", None)
        if fi:
            p = getattr(fi, "last_price", None)
            if p is None:
                p = getattr(fi, "last_price", None)
            if p is not None:
                return float(p)
    except Exception:
        pass
    # fallback: hist 1d
    try:
        h = t.history(period="1d")
        if not h.empty and "Close" in h:
            return float(h["Close"].iloc[-1])
    except Exception:
        pass
    return None

def _get_currency(t: yf.Ticker, info: dict) -> Optional[str]:
    # fast_info
    try:
        fi = getattr(t, "fast_info", None)
        if fi:
            c = getattr(fi, "currency", None)
            if c:
                return str(c).upper()
    except Exception:
        pass
    # info
    c = info.get("currency")
    if c:
        return str(c).upper()
    return None

def _get_company_name(info: dict) -> str:
    return str(info.get("shortName") or info.get("longName") or "")

def _implied_shares(price: float, mcap: float, info: dict) -> float:
    """
    Returnerar utestående aktier (styck) antingen direkt från info eller implied mcap/price.
    """
    so = info.get("sharesOutstanding")
    try:
        so = float(so) if so is not None else 0.0
    except Exception:
        so = 0.0
    if so and so > 0:
        return so
    if price and price > 0 and mcap and mcap > 0:
        return mcap / price
    return 0.0

def _market_cap(t: yf.Ticker, info: dict, price: Optional[float]) -> float:
    # prova fast_info
    try:
        fi = getattr(t, "fast_info", None)
        if fi:
            mc = getattr(fi, "market_cap", None)
            if mc:
                return float(mc)
    except Exception:
        pass
    # info
    mc = info.get("marketCap")
    mc = _safe_float(mc, 0.0)
    if mc <= 0 and price:
        so = _implied_shares(price, info.get("marketCap"), info)
        if so > 0:
            return float(price) * float(so)
    return mc

def _yahoo_quarterly_revenues(t: yf.Ticker) -> List[Tuple[date, float]]:
    """
    Försök plocka kvartalsintäkter (datum, värde) nyast→äldst.
    """
    # 1) quarterly_financials (DataFrame med index=rows och kolumner=perioder)
    try:
        qf = t.quarterly_financials
        if isinstance(qf, pd.DataFrame) and not qf.empty:
            idx = [str(x).strip() for x in qf.index]
            cand_rows = [
                "Total Revenue","TotalRevenue","Revenues","Revenue","Sales",
                "Total revenue","Revenues from contracts with customers"
            ]
            for key in cand_rows:
                if key in idx:
                    row = qf.loc[key].dropna()
                    out = []
                    for c, v in row.items():
                        try:
                            d = c.date() if hasattr(c, "date") else pd.to_datetime(c).date()
                            out.append((d, float(v)))
                        except Exception:
                            pass
                    out.sort(key=lambda x: x[0], reverse=True)
                    return out
    except Exception:
        pass

    # 2) income_stmt quarterly
    try:
        df_is = getattr(t, "income_stmt", None)
        if isinstance(df_is, pd.DataFrame) and not df_is.empty and "Total Revenue" in df_is.index:
            ser = df_is.loc["Total Revenue"].dropna()
            out = []
            for c, v in ser.items():
                try:
                    d = c.date() if hasattr(c, "date") else pd.to_datetime(c).date()
                    out.append((d, float(v)))
                except Exception:
                    pass
            out.sort(key=lambda x: x[0], reverse=True)
            return out
    except Exception:
        pass

    return []

def _ttm_windows(values: List[Tuple[date, float]], need: int = 4) -> List[Tuple[date, float]]:
    """
    Tar [(end_date, kvartalsintäkt), ...] (nyast→äldst) och bygger upp till 'need' TTM-summor:
    [(end_date0, ttm0), (end_date1, ttm1), ...] där ttm0 = sum(q0..q3), ttm1 = sum(q1..q4), osv.
    """
    out: List[Tuple[date, float]] = []
    if len(values) < 4:
        return out
    for i in range(0, min(need, len(values) - 3)):
        end_i = values[i][0]
        ttm_i = sum(v for (_, v) in values[i:i+4])
        out.append((end_i, float(ttm_i)))
    return out

def _prices_on_or_before(t: yf.Ticker, dates: List[date]) -> Dict[date, float]:
    """
    Hämtar dagliga priser i ett fönster som täcker alla 'dates' och returnerar
    'Close' på eller närmast FÖRE respektive datum.
    """
    if not dates:
        return {}
    dmin = min(dates) - timedelta(days=14)
    dmax = max(dates) + timedelta(days=2)
    try:
        hist = t.history(start=dmin, end=dmax, interval="1d")
        if hist is None or hist.empty:
            return {}
        hist = hist.sort_index()
        idx = list(hist.index.date)
        closes = list(hist["Close"].values)
        out = {}
        for d in dates:
            px = None
            for j in range(len(idx)-1, -1, -1):
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

# =========================
# Publika runners
# =========================

def run_update_price_only(ticker: str) -> Tuple[Dict, Dict]:
    """
    Snabb uppdatering av pris/namn/valuta/utdelning via Yahoo (yfinance).
    Returnerar (vals, debug).
    """
    t = yf.Ticker(ticker)
    info = _yfi_info_dict(t)

    price = _get_last_price(t)
    currency = _get_currency(t, info) or "USD"
    name = _get_company_name(info)

    vals: Dict[str, float | str] = {}
    if name:
        vals["Bolagsnamn"] = name
    if currency:
        vals["Valuta"] = currency
    if price is not None and price > 0:
        vals["Aktuell kurs"] = float(price)

    # Årlig utdelning om tillgänglig
    div_rate = info.get("dividendRate")
    try:
        if div_rate is not None:
            vals["Årlig utdelning"] = float(div_rate)
    except Exception:
        pass

    debug = {
        "source": "Yahoo (price-only)",
        "price": price,
        "currency": currency,
        "name": name,
    }
    return vals, debug


def run_update_full(ticker: str, df=None, user_rates=None) -> Tuple[Dict, Dict, str]:
    """
    Full uppdatering baserad på Yahoo (lagliga/gratis).
    Sätter: Bolagsnamn, Valuta, Aktuell kurs, Utestående aktier (implied),
            P/S (TTM nu), P/S Q1–Q4 (historiskt), Årlig utdelning.
    Returnerar (vals, debug, source).
    """
    source = "Yahoo-only"
    t = yf.Ticker(ticker)
    info = _yfi_info_dict(t)

    # Bas
    price = _get_last_price(t)
    currency = _get_currency(t, info) or "USD"
    name = _get_company_name(info)
    mcap = _market_cap(t, info, price)
    shares = _implied_shares(price or 0.0, mcap or 0.0, info)

    vals: Dict[str, float | str] = {}
    if name:
        vals["Bolagsnamn"] = name
    if currency:
        vals["Valuta"] = currency
    if price is not None and price > 0:
        vals["Aktuell kurs"] = float(price)
    if shares > 0:
        vals["Utestående aktier"] = float(shares) / 1e6  # lagra i miljoner, som din datamodell

    # Kvartalsintäkter → TTM-lista (upp till 4)
    q_rows = _yahoo_quarterly_revenues(t)  # [(date, revenue), nyast→äldst]
    ttm_list = _ttm_windows(q_rows, need=4)  # [(end_date, ttm_value)]
    debug = {
        "info_currency": currency,
        "price": price,
        "marketCap": mcap,
        "shares_out": shares,
        "q_revenues_count": len(q_rows),
        "ttm_count": len(ttm_list),
        "q_dates": [str(d) for (d, _) in q_rows[:8]],
    }

    # P/S nu (TTM0)
    if mcap and mcap > 0 and ttm_list:
        ttm_now = ttm_list[0][1]
        if ttm_now and ttm_now > 0:
            vals["P/S"] = float(mcap) / float(ttm_now)

    # P/S Q1–Q4 historik: använd samma shares och historiskt pris vid respektive TTM-datum
    if shares > 0 and ttm_list:
        dates = [d for (d, _) in ttm_list[:4]]
        px_map = _prices_on_or_before(t, dates)
        for idx, (d_end, ttm_val) in enumerate(ttm_list[:4], start=1):
            if ttm_val and ttm_val > 0:
                px = px_map.get(d_end)
                if px and px > 0:
                    mcap_hist = float(shares) * float(px)
                    vals[f"P/S Q{idx}"] = float(mcap_hist / float(ttm_val))

    # Årlig utdelning
    div_rate = info.get("dividendRate")
    try:
        if div_rate is not None:
            vals["Årlig utdelning"] = float(div_rate)
    except Exception:
        pass

    return vals, debug, source
