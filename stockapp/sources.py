# stockapp/sources.py
# -*- coding: utf-8 -*-
"""
Datakällor & runner-funktioner.

Publika:
- run_update_price_only(ticker) -> (vals: dict, source: str, err: Optional[str])
- run_update_full(ticker, df=None, user_rates=None) -> (vals: dict, source: str, err: Optional[str])

Internt:
- Yahoo-basics (namn/valuta/kurs/utdelning/CAGR)
- SEC: robust shares (instant-summor) + kvartalsintäkter (10-Q/6-K)
- Yahoo: quarterly financials fallback
- TTM-summor och P/S (nu + historik Q1–Q4, med kvartals-justering)
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import math
import datetime as dt

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# Lokal beroende på rates för valutakonvertering när vi jämför SEC-unit vs prisvaluta
try:
    from .rates import hamta_valutakurs
except Exception:
    def hamta_valutakurs(v: str, _rates: Dict[str, float] = None) -> float:
        return 1.0


# ------------------------------- Utils ---------------------------------------

def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def _parse_iso(d: str) -> Optional[dt.date]:
    if not d:
        return None
    try:
        return dt.date.fromisoformat(d)
    except Exception:
        try:
            return dt.datetime.strptime(d, "%Y-%m-%d").date()
        except Exception:
            return None

def _yfi_info(t: yf.Ticker) -> dict:
    try:
        return t.info or {}
    except Exception:
        return {}

def _yfi_hist(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    try:
        t = yf.Ticker(ticker)
        df = t.history(start=start, end=end, interval="1d")
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def _prices_on_or_before(ticker: str, dates: List[dt.date]) -> Dict[dt.date, float]:
    """
    Hämtar dagliga priser i fönster som täcker alla datum och returnerar stängning
    på eller närmast FÖRE resp. datum.
    """
    if not dates:
        return {}
    dmin = min(dates) - dt.timedelta(days=14)
    dmax = max(dates) + dt.timedelta(days=2)
    hist = _yfi_hist(ticker, dmin, dmax)
    if hist.empty or "Close" not in hist:
        return {}
    hist = hist.sort_index()
    idx = [i.date() for i in hist.index]
    close = list(hist["Close"].values)

    out = {}
    for d in dates:
        px = None
        for j in range(len(idx) - 1, -1, -1):
            if idx[j] <= d:
                try:
                    px = float(close[j])
                except Exception:
                    px = None
                break
        if px is not None:
            out[d] = px
    return out

def _ttm_windows(values: List[Tuple[dt.date, float]], need: int = 6) -> List[Tuple[dt.date, float]]:
    """
    Tar [(end_date, kvartalsintäkt), ...] NYAST→ÄLDST och bygger upp till 'need' TTM-summor:
    [(end_date0, ttm0), (end_date1, ttm1), ...] där ttm0 = sum(q0..q3), ttm1 = sum(q1..q4), ...
    Minst 4 kvartal krävs.
    """
    if not values or len(values) < 4:
        return []
    out: List[Tuple[dt.date, float]] = []
    lim = min(need, len(values) - 3)
    for i in range(lim):
        end_i = values[i][0]
        ttm_i = float(sum(v for (_, v) in values[i:i+4]))
        out.append((end_i, ttm_i))
    return out

def _quarter_order_key(d: dt.date) -> Tuple[int, int]:
    """
    Sorteringsnyckel som säkerställer att “rapportering i dec/jan” kommer med i rätt ordning.
    Nyast först: (year, quarter_index) desc används externt.
    """
    if not isinstance(d, dt.date):
        return (0, 0)
    # Q-index 1..4
    m = d.month
    if m in (1,2,3): q = 1
    elif m in (4,5,6): q = 2
    elif m in (7,8,9): q = 3
    else: q = 4
    return (d.year, q)


# ------------------------------- SEC -----------------------------------------

_SEC_UA = "StockApp/1.0 (contact: you@example.com)"

def _sec_get(url: str, params=None):
    try:
        r = requests.get(url, params=params or {}, headers={"User-Agent": _SEC_UA}, timeout=30)
        if r.status_code == 200:
            return r.json(), 200
        return None, r.status_code
    except Exception:
        return None, 0

def _sec_ticker_map() -> Dict[str, str]:
    j, sc = _sec_get("https://www.sec.gov/files/company_tickers.json")
    if sc != 200 or not isinstance(j, dict):
        return {}
    out = {}
    for _, v in j.items():
        try:
            out[str(v["ticker"]).upper()] = str(v["cik_str"]).zfill(10)
        except Exception:
            pass
    return out

def _sec_cik_for(ticker: str) -> Optional[str]:
    return _sec_ticker_map().get(str(ticker).upper())

def _sec_companyfacts(cik10: str):
    return _sec_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json")

def _is_instant(it: dict) -> bool:
    end = it.get("end"); start = it.get("start")
    if not end:
        return False
    if not start:
        return True
    d1 = _parse_iso(str(start)); d2 = _parse_iso(str(end))
    if d1 and d2:
        try:
            return (d2 - d1).days <= 2
        except Exception:
            return False
    return False

def _collect_instant_shares(facts: dict) -> List[Tuple[dt.date, float]]:
    """
    Samlar alla instant ‘shares’ från dei/us-gaap/ifrs-full och summerar per datum.
    Returnerar [(end_date, total_shares), ...] NYAST→ÄLDST (unika datum).
    """
    facts_all = (facts.get("facts") or {})
    sources = [
        ("dei", ["EntityCommonStockSharesOutstanding", "EntityCommonSharesOutstanding"]),
        ("us-gaap", ["CommonStockSharesOutstanding", "ShareIssued"]),
        ("ifrs-full", ["NumberOfSharesIssued", "IssuedCapitalNumberOfShares", "OrdinarySharesNumber"]),
    ]
    unit_keys = ("shares","Shares","USD_shares","SHARES")

    rows = []
    for taxo, keys in sources:
        sect = facts_all.get(taxo, {})
        for key in keys:
            f = sect.get(key)
            if not f:
                continue
            units = f.get("units") or {}
            for uk in unit_keys:
                arr = units.get(uk)
                if not isinstance(arr, list):
                    continue
                for it in arr:
                    if not _is_instant(it):
                        continue
                    d = _parse_iso(str(it.get("end","")))
                    v = it.get("val", None)
                    if d and v is not None:
                        try:
                            rows.append((d, float(v)))
                        except Exception:
                            pass
    if not rows:
        return []
    # summera per datum
    agg: Dict[dt.date, float] = {}
    for d, v in rows:
        agg[d] = agg.get(d, 0.0) + float(v)
    out = sorted(agg.items(), key=lambda t: t[0], reverse=True)
    return out

def _sec_quarterly_revenues_dated_with_unit(facts: dict, max_quarters: int = 20) -> Tuple[List[Tuple[dt.date, float]], Optional[str]]:
    """
    Hämtar upp till 'max_quarters' kvartalsintäkter (3-mån) för US-GAAP (10-Q) och IFRS (6-K).
    Returnerar (rows, unit) med rows=[(end_date, value), ...] NYAST→ÄLDST.
    """
    taxos = [
        ("us-gaap",  {"forms": ("10-Q","10-Q/A")}),
        ("ifrs-full",{"forms": ("6-K","6-K/A","10-Q","10-Q/A")}),
    ]
    rev_keys = [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "Revenues",
        "Revenue",
        "RevenueFromContractsWithCustomers",
    ]
    prefer_units = ("USD","CAD","EUR","GBP")

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
                tmp = []
                for it in arr:
                    form = (it.get("form") or "").upper()
                    if not any(f in form for f in cfg["forms"]):
                        continue
                    end = _parse_iso(str(it.get("end",""))); start = _parse_iso(str(it.get("start","")))
                    val = it.get("val", None)
                    if not (end and start and val is not None):
                        continue
                    try:
                        dur = (end - start).days
                    except Exception:
                        dur = None
                    if dur is None or dur < 70 or dur > 100:
                        continue
                    try:
                        tmp.append((end, float(val)))
                    except Exception:
                        pass
                if not tmp:
                    continue
                # deduplicera per end-date, nyast först
                ded: Dict[dt.date, float] = {}
                for end, v in tmp:
                    ded[end] = v
                rows = sorted(ded.items(), key=lambda t: (_quarter_order_key(t[0])[0], _quarter_order_key(t[0])[1]), reverse=True)
                return rows[:max_quarters], unit_code
    return [], None


# ----------------------------- Yahoo-fallback --------------------------------

def _yahoo_quarterly_revenues(ticker: str) -> List[Tuple[dt.date, float]]:
    """
    Försöker läsa kvartalsintäkter från Yahoo (quarterly_financials eller income_stmt).
    Returnerar [(period_end_date, value), ...] NYAST→ÄLDST.
    """
    t = yf.Ticker(ticker)

    # 1) quarterly_financials
    try:
        qf = t.quarterly_financials
        if isinstance(qf, pd.DataFrame) and not qf.empty:
            idx = [str(x) for x in qf.index]
            for key in ["Total Revenue","TotalRevenue","Revenues","Revenue","Sales"]:
                if key in qf.index:
                    row = qf.loc[key].dropna()
                    out = []
                    for c, v in row.items():
                        try:
                            d = c.date() if hasattr(c,"date") else pd.to_datetime(c).date()
                            out.append((d, float(v)))
                        except Exception:
                            pass
                    out.sort(key=lambda t: (_quarter_order_key(t[0])[0], _quarter_order_key(t[0])[1]), reverse=True)
                    return out
    except Exception:
        pass

    # 2) income_stmt quarterly (vissa miljöer)
    try:
        df_is = getattr(t, "income_stmt", None)
        if isinstance(df_is, pd.DataFrame) and not df_is.empty and "Total Revenue" in df_is.index:
            ser = df_is.loc["Total Revenue"].dropna()
            out = []
            for c, v in ser.items():
                try:
                    d = c.date() if hasattr(c,"date") else pd.to_datetime(c).date()
                    out.append((d, float(v)))
                except Exception:
                    pass
            out.sort(key=lambda t: (_quarter_order_key(t[0])[0], _quarter_order_key(t[0])[1]), reverse=True)
            return out
    except Exception:
        pass

    return []


# ---------------------------- Yahoo-basics -----------------------------------

def _yahoo_basics(ticker: str) -> dict:
    out = {
        "Bolagsnamn": "",
        "Valuta": "USD",
        "Aktuell kurs": 0.0,
        "Årlig utdelning": 0.0,
        "P/E": None,
        "EV/EBITDA": None,
        "P/FCF": None,
    }
    try:
        t = yf.Ticker(ticker)
        info = _yfi_info(t)

        # Pris
        price = info.get("regularMarketPrice")
        if price is None:
            hist = t.history(period="1d")
            if isinstance(hist, pd.DataFrame) and not hist.empty and "Close" in hist:
                price = float(hist["Close"].iloc[-1])
        if price is not None:
            out["Aktuell kurs"] = float(price)

        # Valuta/namn
        if info.get("currency"):
            out["Valuta"] = str(info["currency"]).upper()
        out["Bolagsnamn"] = str(info.get("shortName") or info.get("longName") or "")

        # Utdelning
        if info.get("dividendRate") is not None:
            try:
                out["Årlig utdelning"] = float(info.get("dividendRate") or 0.0)
            except Exception:
                pass

        # Några multiplar om finns
        if info.get("trailingPE") is not None:
            try: out["P/E"] = float(info["trailingPE"])
            except Exception: pass
        if info.get("enterpriseToEbitda") is not None:
            try: out["EV/EBITDA"] = float(info["enterpriseToEbitda"])
            except Exception: pass
        if info.get("priceToFreeCashflows") is not None:
            try: out["P/FCF"] = float(info["priceToFreeCashflows"])
            except Exception: pass

    except Exception:
        pass
    return out


# --------------------------- P/S & aktier ------------------------------------

def _implied_shares_from_yahoo(ticker: str, price: float = None, info: dict = None) -> float:
    """
    Försök räkna shares = marketCap / price; fallback: sharesOutstanding.
    Returnerar antal aktier (styck).
    """
    t = yf.Ticker(ticker)
    ii = info or _yfi_info(t)
    if price is None:
        price = ii.get("regularMarketPrice")
    mcap = ii.get("marketCap")
    try:
        mcap = float(mcap) if mcap is not None else 0.0
        price = float(price) if price is not None else 0.0
    except Exception:
        mcap = 0.0
    if mcap > 0 and price > 0:
        return mcap / price
    so = ii.get("sharesOutstanding")
    try:
        so = float(so or 0.0)
    except Exception:
        so = 0.0
    return so

def _ps_from_mcap_and_ttm(mcap: float, ttm_rev: float) -> float:
    if ttm_rev and ttm_rev > 0:
        return float(mcap) / float(ttm_rev)
    return 0.0


# --------------------------- Full fetch byggsten ------------------------------

def _full_fetch_ttm_ps_and_quarters(ticker: str, price_ccy: str) -> Tuple[Dict[str, float], List[Tuple[dt.date,float]]]:
    """
    Returnerar (vals_ps, ttm_list_px)
    vals_ps innehåller ev. "P/S" samt "P/S Q1..Q4"
    ttm_list_px är [(end_date, ttm_in_price_ccy), ...] NYAST→ÄLDST
    """
    vals_ps: Dict[str, float] = {}
    ttm_list_px: List[Tuple[dt.date,float]] = []

    cik = _sec_cik_for(ticker)
    if cik:
        facts, sc = _sec_companyfacts(cik)
        if sc == 200 and isinstance(facts, dict):
            # SEC kvartalsintäkter
            rows, unit = _sec_quarterly_revenues_dated_with_unit(facts, max_quarters=20)
            if rows and unit:
                # konvertera till prisvaluta via rates-modulen (om möjligt)
                conv = 1.0
                if unit.upper() != price_ccy.upper():
                    try:
                        conv = hamta_valutakurs(unit.upper(), {price_ccy.upper():1.0})
                        # hamta_valutakurs i vår fallback tar alltid 1.0, men i appen skickar vi user_rates
                    except Exception:
                        conv = 1.0
                # Bygg TTM i SEC-unit → konvertera
                ttm_raw = _ttm_windows(rows, need=6)
                ttm_list_px = [(d, v * conv) for (d, v) in ttm_raw]
    if not ttm_list_px:
        # Yahoo fallback
        q_rows = _yahoo_quarterly_revenues(ticker)
        if q_rows and len(q_rows) >= 4:
            ttm_list_px = _ttm_windows(q_rows, need=6)

    # Om vi kan räkna P/S (nu) och historiska P/S Q1..Q4
    if ttm_list_px:
        # Hämta market cap nu
        t = yf.Ticker(ticker)
        info = _yfi_info(t)
        mcap_now = _safe_float(info.get("marketCap"), 0.0)
        if mcap_now <= 0:
            # fallback: implied shares * price
            px = _safe_float(info.get("regularMarketPrice"), 0.0)
            sh = _implied_shares_from_yahoo(ticker, px, info)
            mcap_now = sh * px

        # P/S (nu)
        ltm_now = _safe_float(ttm_list_px[0][1], 0.0)
        if mcap_now > 0 and ltm_now > 0:
            vals_ps["P/S"] = _ps_from_mcap_and_ttm(mcap_now, ltm_now)

        # Historik P/S Q1..Q4 (behöver shares ~ konstanta; approximera med implied shares nu)
        sh_now = 0.0
        try:
            px_now = _safe_float(info.get("regularMarketPrice"), 0.0)
            sh_now = _implied_shares_from_yahoo(ticker, px_now, info)
        except Exception:
            pass
        if sh_now > 0:
            q_dates = [d for (d, _) in ttm_list_px[:4]]
            px_map = _prices_on_or_before(ticker, q_dates)
            for idx, (d_end, ttm_rev) in enumerate(ttm_list_px[:4], start=1):
                if ttm_rev and ttm_rev > 0:
                    px = _safe_float(px_map.get(d_end), 0.0)
                    if px > 0:
                        mcap_hist = sh_now * px
                        vals_ps[f"P/S Q{idx}"] = _ps_from_mcap_and_ttm(mcap_hist, ttm_rev)

    return vals_ps, ttm_list_px


# ------------------------------ Runners --------------------------------------

def run_update_price_only(ticker: str) -> Tuple[Dict[str, object], str, Optional[str]]:
    """
    Hämtar endast: Bolagsnamn, Valuta, Aktuell kurs, Årlig utdelning.
    Returnerar (vals, source, err).
    """
    try:
        y = _yahoo_basics(ticker)
        vals = {}
        for k in ["Bolagsnamn","Valuta","Aktuell kurs","Årlig utdelning"]:
            v = y.get(k)
            if isinstance(v, (int,float)) and _safe_float(v) <= 0 and k != "Årlig utdelning":
                continue
            if v not in (None, "", 0, 0.0):
                vals[k] = v
        return vals, "Yahoo(price-only)", None
    except Exception as e:
        return {}, "Yahoo(price-only)", str(e)

def run_update_full(ticker: str, df: pd.DataFrame = None, user_rates: Dict[str,float] = None) -> Tuple[Dict[str, object], str, Optional[str]]:
    """
    Full uppdatering för EN ticker.
    Sätter INTE 'Omsättning idag/nästa år' (de är manuella enligt din spec).
    Returnerar (vals, source, err).
    """
    try:
        vals: Dict[str, object] = {}

        # 1) Yahoo-basics
        y = _yahoo_basics(ticker)
        for k in ["Bolagsnamn","Valuta","Aktuell kurs","Årlig utdelning","P/E","EV/EBITDA","P/FCF"]:
            v = y.get(k)
            if v not in (None, "", 0, 0.0):
                vals[k] = v

        price_ccy = str(y.get("Valuta","USD")).upper()

        # 2) Utestående aktier: implied först → SEC robust fallback
        sh = _implied_shares_from_yahoo(ticker, y.get("Aktuell kurs"), None)
        if sh and sh > 0:
            vals["Utestående aktier"] = float(sh) / 1e6
            share_source = "Yahoo implied"
        else:
            cik = _sec_cik_for(ticker)
            if cik:
                facts, sc = _sec_companyfacts(cik)
                if sc == 200 and isinstance(facts, dict):
                    rows = _collect_instant_shares(facts)
                    if rows:
                        latest_date = rows[0][0]
                        # summera alla rader på latest_date (rows redan aggregerad)
                        latest_total = rows[0][1]
                        if latest_total and latest_total > 0:
                            vals["Utestående aktier"] = float(latest_total) / 1e6
                            share_source = "SEC instant"
                        else:
                            share_source = "unknown"
                    else:
                        share_source = "unknown"
                else:
                    share_source = "unknown"
            else:
                share_source = "unknown"

        # 3) P/S nu + Q1..Q4
        ps_vals, ttm_list_px = _full_fetch_ttm_ps_and_quarters(ticker, price_ccy=price_ccy)
        for k, v in ps_vals.items():
            try:
                if v and float(v) > 0:
                    vals[k] = float(v)
            except Exception:
                pass

        # 4) Risk: skriv inte tomma/0-värden
        clean = {}
        for k, v in vals.items():
            ok = True
            if isinstance(v, (int, float)):
                if k in ("P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Utestående aktier","Aktuell kurs") and _safe_float(v) <= 0:
                    ok = False
            if isinstance(v, str) and not v.strip():
                ok = False
            if ok:
                clean[k] = v

        source = "Full (Yahoo + SEC + Yahoo-fallback)"
        return clean, source, None

    except Exception as e:
        return {}, "Full (Yahoo + SEC + Yahoo-fallback)", str(e)
