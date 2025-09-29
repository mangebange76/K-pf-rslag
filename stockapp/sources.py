# stockapp/sources.py
# -*- coding: utf-8 -*-
"""
Datakällor & Runners för batch/enkelticker-uppdateringar.

Publika funktioner:
- run_update_price_only(ticker) -> (vals, source, err)
- run_update_full(ticker, df=None, user_rates=None) -> (vals, source, err)

Vad hämtas i run_update_full:
  ✓ Bolagsnamn
  ✓ Valuta
  ✓ Aktuell kurs
  ✓ Årlig utdelning (om tillgänglig i Yahoo)
  ✓ CAGR 5 år (%) (grovt, via Yahoo financials)
  ✓ Utestående aktier (implied = marketCap / price)
  ✓ P/S (TTM) NU
  ✓ P/S Q1–Q4 (via TTM-fönster, med kurs vid respektive periodslut)
    - Räknas i prisets valuta. Om finansiell valuta ≠ prisvaluta används user_rates för FX-konvertering.

OBS:
- Skriver INTE "Omsättning idag" / "Omsättning nästa år" (lämnas för manuell uppdatering).
- Returnerade fält skrivs endast om kolumnerna finns i df (batch.py hanterar detta).
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional, List
from datetime import timedelta as _td

import pandas as pd
import numpy as np
import requests
import yfinance as yf
import streamlit as st

from .utils import now_stamp

# ------------------------------------------------------------
# Yahoo helpers
# ------------------------------------------------------------

def _yfi_info(ticker: str) -> dict:
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        return info
    except Exception:
        return {}

def _yfi_price_and_currency(ticker: str) -> Tuple[Optional[float], Optional[str]]:
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        price = info.get("regularMarketPrice")
        if price is None:
            hist = t.history(period="1d")
            if isinstance(hist, pd.DataFrame) and not hist.empty and "Close" in hist:
                price = float(hist["Close"].iloc[-1])
        ccy = info.get("currency")
        return (float(price) if price is not None else None, str(ccy).upper() if ccy else None)
    except Exception:
        return (None, None)

def _yfi_dividend_rate(info: dict) -> float:
    try:
        v = info.get("dividendRate", None)
        return float(v) if v is not None else 0.0
    except Exception:
        return 0.0

def _yfi_cagr_approx(t: yf.Ticker) -> float:
    """
    Approximerad CAGR 5 år (%) från Yahoo financials (Total Revenue).
    """
    try:
        # Försök med income_stmt (årsbasis) via .financials eller .income_stmt
        df_fin = getattr(t, "financials", None)
        series = None
        if isinstance(df_fin, pd.DataFrame) and not df_fin.empty and "Total Revenue" in df_fin.index:
            series = df_fin.loc["Total Revenue"].dropna()
        else:
            df_is = getattr(t, "income_stmt", None)
            if isinstance(df_is, pd.DataFrame) and not df_is.empty and "Total Revenue" in df_is.index:
                series = df_is.loc["Total Revenue"].dropna()

        if series is None or len(series) < 2:
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

def _yfi_quarterly_revenues(t: yf.Ticker) -> List[Tuple[pd.Timestamp, float, str]]:
    """
    Läs kvartalsintäkter från Yahoo .quarterly_financials.
    Returnerar lista [(period_end_date, value_float, currency)], nyast -> äldst.
    Currency hämtas från info.financialCurrency om möjligt, annars prisvaluta.

    Vi letar efter radnamn som motsvarar Revenue.
    """
    out: List[Tuple[pd.Timestamp, float, str]] = []
    try:
        qf = t.quarterly_financials
        if isinstance(qf, pd.DataFrame) and not qf.empty:
            idx = [str(x).strip() for x in qf.index]
            cand_rows = [
                "Total Revenue", "TotalRevenue", "Revenues", "Revenue", "Sales",
                "Revenues from contracts with customers"
            ]
            key = None
            for k in cand_rows:
                if k in idx:
                    key = k
                    break
            if key:
                row = qf.loc[key].dropna()
                for c, v in row.items():
                    try:
                        d = c.to_pydatetime() if hasattr(c, "to_pydatetime") else pd.to_datetime(c).to_pydatetime()
                        out.append((pd.Timestamp(d), float(v), ""))  # currency fylls senare
                    except Exception:
                        pass
    except Exception:
        pass
    # sortera nyast -> äldst
    out.sort(key=lambda x: x[0], reverse=True)
    return out

def _yfi_prices_for_dates(ticker: str, dates: List[pd.Timestamp]) -> Dict[pd.Timestamp, float]:
    """
    Hämtar stängningskurs på eller närmast före respektive datum.
    """
    if not dates:
        return {}
    dmin = min(dates) - _td(days=14)
    dmax = max(dates) + _td(days=2)
    try:
        t = yf.Ticker(ticker)
        hist = t.history(start=dmin, end=dmax, interval="1d")
        if hist is None or hist.empty or "Close" not in hist:
            return {}
        hist = hist.sort_index()
        out: Dict[pd.Timestamp, float] = {}
        idx = list(hist.index)
        closes = list(hist["Close"].values)
        for d in dates:
            px = None
            # hitta senaste handelsdag <= d
            for j in range(len(idx) - 1, -1, -1):
                if idx[j].to_pydatetime().date() <= d.date():
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

def _ttm_windows(values: List[Tuple[pd.Timestamp, float]], need: int = 6) -> List[Tuple[pd.Timestamp, float]]:
    """
    Tar [(date, quarterly_revenue), ...] (nyast -> äldst) och bygger upp till 'need' TTM-summor:
    [(end_date0, ttm0), (end_date1, ttm1), ...]
    """
    out: List[Tuple[pd.Timestamp, float]] = []
    if len(values) < 4:
        return out
    for i in range(0, min(need, len(values) - 3)):
        end_i = values[i][0]
        ttm_i = float(values[i][1] + values[i+1][1] + values[i+2][1] + values[i+3][1])
        out.append((end_i, ttm_i))
    return out

# ------------------------------------------------------------
# SEC helpers (USA / FPI)
# ------------------------------------------------------------

_DEF_UA = st.secrets.get("SEC_USER_AGENT", "StockApp/1.0 (contact: you@example.com)")

def _sec_get(url: str, params=None):
    try:
        r = requests.get(url, params=params or {}, headers={"User-Agent": _DEF_UA}, timeout=30)
        if r.status_code == 200:
            return r.json(), 200
        return None, r.status_code
    except Exception:
        return None, 0

def _sec_ticker_map() -> Dict[str, str]:
    """
    {"AAPL": "0000320193", ...}
    """
    cache_key = "_sec_ticker_map_json"
    j = st.session_state.get(cache_key)
    if j is None:
        j, sc = _sec_get("https://www.sec.gov/files/company_tickers.json")
        if not isinstance(j, dict):
            return {}
        st.session_state[cache_key] = j
    out: Dict[str, str] = {}
    for _, v in j.items():
        try:
            out[str(v["ticker"]).upper()] = str(v["cik_str"]).zfill(10)
        except Exception:
            pass
    return out

def _sec_cik_for(ticker: str) -> Optional[str]:
    return _sec_ticker_map().get(str(ticker).upper())

def _parse_iso_date(d: str) -> Optional[pd.Timestamp]:
    try:
        return pd.to_datetime(d, errors="coerce")
    except Exception:
        return None

def _sec_quarterly_revenues_with_unit(facts: dict, prefer_units=("USD","EUR","CAD","GBP")) -> Tuple[List[Tuple[pd.Timestamp, float]], Optional[str]]:
    """
    Hämtar kvartalsintäkter från us-gaap/ifrs-full, endast 3-månadersperioder (10-Q/6-K).
    Returnerar (rows, unit) med rows=[(end_date, value), ...] nyast->äldst.
    """
    taxos = [
        ("us-gaap",  {"forms": ("10-Q", "10-Q/A")}),
        ("ifrs-full", {"forms": ("6-K", "6-K/A", "10-Q", "10-Q/A")}),
    ]
    rev_keys = [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "Revenues",
        "Revenue"
    ]
    for taxo, cfg in taxos:
        gaap = (facts.get("facts") or {}).get(taxo, {})
        for name in rev_keys:
            fact = gaap.get(name)
            if not fact:
                continue
            units = (fact.get("units") or {})
            for unit_code in prefer_units:
                arr = units.get(unit_code)
                if not isinstance(arr, list):
                    continue
                tmp: List[Tuple[pd.Timestamp, float]] = []
                for it in arr:
                    form = (it.get("form") or "").upper()
                    if not any(f in form for f in cfg["forms"]):
                        continue
                    end = _parse_iso_date(str(it.get("end", "")))
                    start = _parse_iso_date(str(it.get("start", "")))
                    val = it.get("val", None)
                    if not (isinstance(end, pd.Timestamp) and isinstance(start, pd.Timestamp) and val is not None):
                        continue
                    # ca 3-mån fönster
                    dur_days = (end - start).days if pd.notna(end) and pd.notna(start) else None
                    if dur_days is None or dur_days < 70 or dur_days > 100:
                        continue
                    try:
                        v = float(val)
                        tmp.append((end, v))
                    except Exception:
                        pass
                if not tmp:
                    continue
                # unique per end date, nyast->äldst
                ded: Dict[pd.Timestamp, float] = {}
                for end, v in tmp:
                    ded[end] = v
                rows = sorted(ded.items(), key=lambda t: t[0], reverse=True)
                return rows, unit_code
    return [], None

def _sec_companyfacts(cik10: str):
    return _sec_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json")

# ------------------------------------------------------------
# FX helper (för att omvandla finansiell valuta till prisvaluta)
# ------------------------------------------------------------

def _fx_rate(fin_ccy: str, px_ccy: str, user_rates: Optional[Dict[str, float]]) -> float:
    """
    Hämtar omräkningskurs via användarens sidopanels-kurser (user_rates).
    Om ej angivet eller saknas → 1.0.
    """
    try:
        if not fin_ccy or not px_ccy:
            return 1.0
        if fin_ccy.upper() == px_ccy.upper():
            return 1.0
        if user_rates and fin_ccy.upper() in user_rates and px_ccy.upper() in user_rates:
            # pris i px_ccy = fin_ccy * (px_ccy/fin_ccy) -> vi har rates till SEK, men här antar vi att user_rates innehåller direkta (om ej, lämna 1.0)
            # För enkelhet: anta user_rates är mot SEK; gör cross via SEK om båda finns.
            # SEK-cross:
            sek = float(user_rates.get("SEK", 1.0))
            if sek != 1.0:  # "SEK": 1.0 i vår app – så använd cross mot SEK:
                fin_to_sek = float(user_rates.get(fin_ccy.upper(), 1.0))
                px_to_sek = float(user_rates.get(px_ccy.upper(), 1.0))
                if fin_to_sek > 0 and px_to_sek > 0:
                    # 1 fin_ccy = fin_to_sek SEK; 1 px_ccy = px_to_sek SEK
                    # fin_ccy -> px_ccy = (fin_to_sek / px_to_sek)
                    return fin_to_sek / px_to_sek
            # om vi inte kan korsa, lämna 1.0
        return 1.0
    except Exception:
        return 1.0

# ------------------------------------------------------------
# Publika runners
# ------------------------------------------------------------

def run_update_price_only(ticker: str) -> Tuple[Dict[str, object], str, Optional[str]]:
    """
    Hämtar endast aktuell kurs/valuta/namn via Yahoo.
    Returnerar (vals, source, err)
    """
    try:
        info = _yfi_info(ticker)
        price, ccy = _yfi_price_and_currency(ticker)
        name = info.get("shortName") or info.get("longName") or ""

        out: Dict[str, object] = {}
        if name: out["Bolagsnamn"] = name
        if ccy:  out["Valuta"] = ccy
        if price and price > 0: out["Aktuell kurs"] = float(price)

        # dividend (om man vill uppdatera den även vid price-only)
        div = _yfi_dividend_rate(info)
        out["Årlig utdelning"] = float(div)

        return out, "Price only (Yahoo)", None
    except Exception as e:
        return {}, "Price only (Yahoo)", str(e)

def run_update_full(ticker: str, df: Optional[pd.DataFrame] = None, user_rates: Optional[Dict[str, float]] = None) -> Tuple[Dict[str, object], str, Optional[str]]:
    """
    Full uppdatering, utan att röra dina manuella fält för prognos-omsättning.
    Returnerar (vals, source, err)
    """
    source = "Full auto (Yahoo+SEC fallback)"
    out: Dict[str, object] = {}
    try:
        t = yf.Ticker(ticker)
        info = _yfi_info(ticker)

        # Bas: namn/valuta/price
        name = info.get("shortName") or info.get("longName") or ""
        if name: out["Bolagsnamn"] = name

        price, px_ccy = _yfi_price_and_currency(ticker)
        if px_ccy: out["Valuta"] = px_ccy
        if price and price > 0: out["Aktuell kurs"] = float(price)

        # dividend + CAGR 5y
        out["Årlig utdelning"] = _yfi_dividend_rate(info)
        out["CAGR 5 år (%)"] = _yfi_cagr_approx(t)

        # Market cap & implied shares
        mcap = info.get("marketCap")
        try:
            mcap = float(mcap) if mcap is not None else 0.0
        except Exception:
            mcap = 0.0

        shares = 0.0
        if price and price > 0 and mcap > 0:
            shares = mcap / float(price)
        else:
            so = info.get("sharesOutstanding")
            try:
                shares = float(so or 0.0)
            except Exception:
                shares = 0.0
        if shares > 0:
            out["Utestående aktier"] = shares / 1e6  # miljoner

        # Kvartalsintäkter
        q_rows = [(d, v) for (d, v, _) in _yfi_quarterly_revenues(t)]
        fin_ccy = str(info.get("financialCurrency") or (px_ccy or "USD")).upper()
        if not q_rows or len(q_rows) < 4:
            # SEC fallback (USA/FPI)
            cik = _sec_cik_for(ticker)
            if cik:
                facts, sc = _sec_companyfacts(cik)
                if sc == 200 and isinstance(facts, dict):
                    rows, unit = _sec_quarterly_revenues_with_unit(facts)
                    if rows:
                        q_rows = rows
                        fin_ccy = unit or fin_ccy

        # Om vi fortfarande saknar, avbryt utan fel (returnera det vi har)
        if not q_rows or len(q_rows) < 4 or shares <= 0:
            # returnera utan P/S (vi har ändå kurs/namn etc)
            return out, source, None

        # Bygg TTM-fönster (ta många, så vi kan hyvla bort speciella säsongsskiften)
        ttm_list = _ttm_windows(q_rows, need=6)
        if not ttm_list:
            return out, source, None

        # Konvertera TTM till prisvaluta om behövs
        conv = _fx_rate(fin_ccy, (px_ccy or fin_ccy), user_rates)
        ttm_px = [(d, v * conv) for (d, v) in ttm_list]

        # Market cap nu (om saknas)
        if (not mcap or mcap <= 0) and price and shares > 0:
            mcap = shares * float(price)

        # P/S (TTM) nu
        if mcap and mcap > 0 and ttm_px:
            ltm_now = ttm_px[0][1]
            if ltm_now > 0:
                out["P/S"] = float(mcap / ltm_now)

        # P/S Q1–Q4 historik: använd samma "shares" (implied now) och kurs på/strax före respektive periodslut
        dates = [d for (d, _) in ttm_px[:4]]
        px_map = _yfi_prices_for_dates(ticker, dates)

        for idx, (d_end, ttm_rev_px) in enumerate(ttm_px[:4], start=1):
            if ttm_rev_px and ttm_rev_px > 0:
                p = px_map.get(d_end)
                if p and p > 0 and shares > 0:
                    mcap_hist = shares * float(p)
                    out[f"P/S Q{idx}"] = float(mcap_hist / ttm_rev_px)

        return out, source, None

    except Exception as e:
        return out, source, str(e)
