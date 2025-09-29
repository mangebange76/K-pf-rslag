# stockapp/sources.py
# -*- coding: utf-8 -*-
"""
Datainsamling från Yahoo Finance (och enkla derivat):
- price_only_values(ticker)  -> dict med {Bolagsnamn, Aktuell kurs, Valuta}
- full_auto_values(ticker)   -> dict med 'säkra' fält (ej manuella prognoser):
    Bolagsnamn, Valuta, Aktuell kurs, Årlig utdelning, CAGR 5 år (%),
    Utestående aktier (milj), P/S (TTM), P/S Q1..Q4, Sektor/Sector, Industry,
    _MC_USD, Risk
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import yfinance as yf

# ------------------------------------------------------------
# Hjälpare
# ------------------------------------------------------------

def _safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _implied_shares(market_cap: float, price: float) -> float:
    """Antal aktier (styck) via market cap / price."""
    try:
        mc = float(market_cap)
        px = float(price)
    except Exception:
        return 0.0
    if mc > 0 and px > 0:
        return mc / px
    return 0.0

def _yahoo_info(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}

def _yahoo_history_close_on_dates(ticker: str, dates: List[pd.Timestamp]) -> Dict[pd.Timestamp, float]:
    """
    Hämta 'Close' på eller närmast före respektive datum.
    """
    if not dates:
        return {}
    dmin = min(dates) - pd.Timedelta(days=14)
    dmax = max(dates) + pd.Timedelta(days=2)
    try:
        hist = yf.Ticker(ticker).history(start=dmin, end=dmax, interval="1d")
        if hist is None or hist.empty:
            return {}
        hist = hist.sort_index()
        idx = list(hist.index)
        closes = list(hist["Close"].values)
        out: Dict[pd.Timestamp, float] = {}
        for d in dates:
            px = None
            # gå bakåt tills vi hittar en dag <= d
            for j in range(len(idx) - 1, -1, -1):
                if idx[j].date() <= d.date():
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

def _quarterly_revenues(t: yf.Ticker) -> List[Tuple[pd.Timestamp, float]]:
    """
    Försöker läsa kvartalsintäkter ur Yahoo:
    1) quarterly_financials (rad 'Total Revenue' eller liknande)
    2) income_stmt (fallback)
    Returnerar [(period_end, value)] nyast->äldst.
    """
    # 1) quarterly_financials
    try:
        qf = t.quarterly_financials
        if isinstance(qf, pd.DataFrame) and not qf.empty:
            idx = [str(x).strip() for x in qf.index]
            cand_rows = [
                "Total Revenue", "TotalRevenue", "Revenues", "Revenue", "Sales",
                "Total revenue", "Revenues from contracts with customers",
            ]
            for key in cand_rows:
                if key in idx:
                    row = qf.loc[key].dropna()
                    out: List[Tuple[pd.Timestamp, float]] = []
                    for c, v in row.items():
                        try:
                            d = c if isinstance(c, pd.Timestamp) else pd.to_datetime(c)
                            out.append((pd.Timestamp(d.date()), float(v)))
                        except Exception:
                            pass
                    out.sort(key=lambda x: x[0], reverse=True)
                    return out
    except Exception:
        pass

    # 2) income_stmt fallback (vissa builds)
    try:
        df_is = getattr(t, "income_stmt", None)
        if isinstance(df_is, pd.DataFrame) and not df_is.empty and "Total Revenue" in df_is.index:
            ser = df_is.loc["Total Revenue"].dropna()
            out: List[Tuple[pd.Timestamp, float]] = []
            for c, v in ser.items():
                try:
                    d = c if isinstance(c, pd.Timestamp) else pd.to_datetime(c)
                    out.append((pd.Timestamp(d.date()), float(v)))
                except Exception:
                    pass
            out.sort(key=lambda x: x[0], reverse=True)
            return out
    except Exception:
        pass

    return []

def _ttm_windows(values: List[Tuple[pd.Timestamp, float]], need: int = 4) -> List[Tuple[pd.Timestamp, float]]:
    """
    Tar [(end_date, kvartalsintäkt)] nyast→äldst och bygger TTM-summor:
    [(end_date0, ttm0), (end_date1, ttm1), ...]
    """
    out: List[Tuple[pd.Timestamp, float]] = []
    if len(values) < 4:
        return out
    for i in range(0, min(need, len(values) - 3)):
        end_i = values[i][0]
        ttm_i = sum(v for (_, v) in values[i:i + 4])
        out.append((end_i, float(ttm_i)))
    return out

def _cagr_total_revenue_approx(t: yf.Ticker) -> float:
    """
    Enkel CAGR-approx över årsdata 'Total Revenue' om tillgängligt.
    """
    try:
        df_fin = getattr(t, "financials", None)
        if isinstance(df_fin, pd.DataFrame) and not df_fin.empty and "Total Revenue" in df_fin.index:
            ser = df_fin.loc["Total Revenue"].dropna()
        else:
            df_is = getattr(t, "income_stmt", None)
            if isinstance(df_is, pd.DataFrame) and not df_is.empty and "Total Revenue" in df_is.index:
                ser = df_is.loc["Total Revenue"].dropna()
            else:
                return 0.0
        if ser.empty or len(ser) < 2:
            return 0.0
        ser = ser.sort_index()
        start = float(ser.iloc[0]); end = float(ser.iloc[-1])
        years = max(1, len(ser) - 1)
        if start <= 0:
            return 0.0
        cagr = (end / start) ** (1.0 / years) - 1.0
        return round(cagr * 100.0, 2)
    except Exception:
        return 0.0

def _risk_label_from_mcap_usd(mcap_usd: float) -> str:
    if mcap_usd >= 200_000_000_000:  # >= $200B
        return "Megacap"
    if mcap_usd >= 10_000_000_000:
        return "Largecap"
    if mcap_usd >= 2_000_000_000:
        return "Midcap"
    if mcap_usd >= 300_000_000:
        return "Smallcap"
    return "Microcap"

# ------------------------------------------------------------
# Publika API-funktioner
# ------------------------------------------------------------

def price_only_values(ticker: str) -> Dict[str, object]:
    """
    Hämtar enbart pris/namn/valuta – no-regrets fallback som alltid kan köras snabbt.
    """
    out = {"_source": "Yahoo/price-only"}
    try:
        info = _yahoo_info(ticker)
        price = info.get("regularMarketPrice", None)
        if price is None:
            h = yf.Ticker(ticker).history(period="1d")
            if isinstance(h, pd.DataFrame) and not h.empty and "Close" in h:
                price = float(h["Close"].iloc[-1])
        if price is not None:
            out["Aktuell kurs"] = float(price)

        name = info.get("shortName") or info.get("longName") or ""
        if name: out["Bolagsnamn"] = str(name)

        ccy = info.get("currency")
        if ccy: out["Valuta"] = str(ccy).upper()

    except Exception:
        pass
    return out

def full_auto_values(ticker: str) -> Dict[str, object]:
    """
    Hämtar 'säkra' fält (ej dina manuella prognoser):
      - Bolagsnamn, Valuta, Aktuell kurs, Årlig utdelning
      - CAGR 5 år (%)
      - Utestående aktier (milj) via implied shares
      - P/S (TTM) + P/S Q1..Q4 via TTM fönster och historiska priser
      - Sektor/Sector (om finns), Industry (om finns)
      - _MC_USD, Risk
    """
    out: Dict[str, object] = {"_source": "Yahoo/full-auto"}
    t = yf.Ticker(ticker)
    info = _yahoo_info(ticker)

    # Namn/valuta/pris/utdelning/sector/industry/beta
    try:
        name = info.get("shortName") or info.get("longName") or ""
        if name: out["Bolagsnamn"] = str(name)
        ccy = info.get("currency")
        if ccy: out["Valuta"] = str(ccy).upper()
        price = info.get("regularMarketPrice", None)
        if price is None:
            h = t.history(period="1d")
            if isinstance(h, pd.DataFrame) and not h.empty and "Close" in h:
                price = float(h["Close"].iloc[-1])
        if price is not None:
            out["Aktuell kurs"] = float(price)

        div_rate = info.get("dividendRate", None)
        if div_rate is not None:
            out["Årlig utdelning"] = _safe_float(div_rate, 0.0)

        # metadata
        sector = info.get("sector") or info.get("Sector")
        industry = info.get("industry") or info.get("Industry")
        if sector: out["Sektor"] = str(sector)
        if industry: out["Industry"] = str(industry)

        # market cap / implied shares
        mcap = _safe_float(info.get("marketCap"), 0.0)
        px = _safe_float(out.get("Aktuell kurs", 0.0), 0.0)
        shares = _implied_shares(mcap, px)  # styck
        if shares > 0:
            out["Utestående aktier"] = round(shares / 1e6, 6)  # milj

        # USD market cap (om Yahoo valutan ej USD antar vi att mcap redan i USD – Yahoo brukar rapportera USD)
        out["_MC_USD"] = float(mcap) if mcap > 0 else 0.0
        out["Risk"] = _risk_label_from_mcap_usd(out["_MC_USD"])

        # CAGR 5 år approx
        out["CAGR 5 år (%)"] = _cagr_total_revenue_approx(t)

    except Exception:
        pass

    # P/S (TTM) + P/S Q1..Q4
    try:
        q_rows = _quarterly_revenues(t)  # [(date, rev)]
        ttm_list = _ttm_windows(q_rows, need=5)  # ta upp till 5 TTM för att säkra Q4-spann
        if ttm_list:
            # P/S (TTM nu)
            if out.get("_MC_USD", 0.0) > 0 and ttm_list[0][1] > 0:
                out["P/S"] = float(out["_MC_USD"]) / float(ttm_list[0][1])

            # P/S Q1..Q4 (historiska via samma shares * hist pris)
            if out.get("Utestående aktier", 0.0) > 0:
                shares_used = float(out["Utestående aktier"]) * 1e6
            else:
                shares_used = _implied_shares(out.get("_MC_USD", 0.0), out.get("Aktuell kurs", 0.0))
            if shares_used > 0:
                dates = [d for (d, _) in ttm_list]
                px_map = _yahoo_history_close_on_dates(ticker, dates)
                for i in range(1, 5):  # Q1..Q4
                    if i-1 < len(ttm_list):
                        d_end, ttm_rev = ttm_list[i-1]
                        if ttm_rev and ttm_rev > 0:
                            px_i = px_map.get(d_end)
                            if px_i and px_i > 0:
                                mcap_hist = shares_used * float(px_i)
                                out[f"P/S Q{i}"] = float(mcap_hist / ttm_rev)
    except Exception:
        pass

    return out
