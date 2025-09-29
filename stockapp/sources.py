# stockapp/sources.py
# -*- coding: utf-8 -*-
"""
Datakällor för enskilda tickers.

Publika funktioner:
- fetch_price_only(ticker) -> dict
- fetch_full_ticker(ticker) -> (vals: dict, debug: dict)

Fält som sätts (om tillgängliga):
  "Bolagsnamn", "Valuta", "Aktuell kurs",
  "MCAP nu",
  "Utestående aktier",             # i miljoner
  "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime as _dt, timedelta as _td
import time
import math

import numpy as np
import pandas as pd
import requests
import yfinance as yf


# ------------------------------------------------------------
# --- Snabba hjälpare ----------------------------------------
# ------------------------------------------------------------
def _to_float(x, default: float = 0.0) -> float:
    if x is None:
        return float(default)
    if isinstance(x, (int, float, np.number)):
        try:
            return float(x)
        except Exception:
            return float(default)
    try:
        s = str(x).strip().replace("\u00A0", "").replace(" ", "")
        s = s.replace(",", ".")
        return float(s)
    except Exception:
        return float(default)

def _parse_date(s: str):
    try:
        return _dt.fromisoformat(s.replace("Z", "+00:00")).date()
    except Exception:
        try:
            return _dt.strptime(s, "%Y-%m-%d").date()
        except Exception:
            return None

def _is_close_to_quarter(d1, d2, tol_days=2) -> bool:
    try:
        return abs((d1 - d2).days) <= tol_days
    except Exception:
        return False


# ------------------------------------------------------------
# --- FX (Frankfurter -> exchangerate.host) ------------------
# ------------------------------------------------------------
def _fx_rate(base: str, quote: str, timeout: int = 12) -> float:
    """
    Enkelt växelkurshämtning utan cache mellan 'base' -> 'quote'.
    """
    base = (base or "").upper()
    quote = (quote or "").upper()
    if not base or not quote or base == quote:
        return 1.0
    try:
        r = requests.get(
            "https://api.frankfurter.app/latest",
            params={"from": base, "to": quote},
            timeout=timeout,
        )
        if r.status_code == 200:
            v = (r.json() or {}).get("rates", {}).get(quote)
            if v:
                return float(v)
    except Exception:
        pass
    try:
        r2 = requests.get(
            "https://api.exchangerate.host/latest",
            params={"base": base, "symbols": quote},
            timeout=timeout,
        )
        if r2.status_code == 200:
            v = (r2.json() or {}).get("rates", {}).get(quote)
            if v:
                return float(v)
    except Exception:
        pass
    return 1.0


# ------------------------------------------------------------
# --- SEC API ------------------------------------------------
# ------------------------------------------------------------
_SEC_TICKERS_CACHE: Dict[str, str] = {}
_SEC_TICKERS_TS: float = 0.0

def _sec_get(url: str, params=None, timeout=30) -> Tuple[Optional[dict], int]:
    try:
        r = requests.get(
            url,
            params=params or {},
            headers={"User-Agent": "StockApp/1.0 (contact: example@example.com)"},
            timeout=timeout,
        )
        if r.status_code == 200:
            return r.json(), 200
        return None, r.status_code
    except Exception:
        return None, 0

def _sec_ticker_map(refresh_sec: int = 86400) -> Dict[str, str]:
    global _SEC_TICKERS_CACHE, _SEC_TICKERS_TS
    now = time.time()
    if _SEC_TICKERS_CACHE and (now - _SEC_TICKERS_TS < refresh_sec):
        return _SEC_TICKERS_CACHE
    j, sc = _sec_get("https://www.sec.gov/files/company_tickers.json")
    out: Dict[str, str] = {}
    if isinstance(j, dict):
        for _, v in j.items():
            try:
                t = str(v.get("ticker", "")).upper()
                cik = str(v.get("cik_str", "")).zfill(10)
                if t and cik:
                    out[t] = cik
            except Exception:
                pass
    _SEC_TICKERS_CACHE = out
    _SEC_TICKERS_TS = now
    return out

def _sec_cik_for(ticker: str) -> Optional[str]:
    return _sec_ticker_map().get(str(ticker).upper())

def _sec_companyfacts(cik10: str) -> Tuple[Optional[dict], int]:
    return _sec_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json")

def _is_instant_entry(it: dict) -> bool:
    end = it.get("end")
    start = it.get("start")
    if not end:
        return False
    if not start:
        return True
    d1 = _parse_date(str(start)); d2 = _parse_date(str(end))
    if not d1 or not d2:
        return False
    try:
        return (d2 - d1).days <= 2
    except Exception:
        return False

def _collect_share_entries(facts: dict) -> List[dict]:
    """
    Hämtar alla 'instant' aktieposter från dei/us-gaap/ifrs-full (unit 'shares' m.fl.).
    Returnerar list[{"end": date, "val": float, ...}]
    """
    entries: List[dict] = []
    facts_all = (facts.get("facts") or {})
    sources = [
        ("dei", ["EntityCommonStockSharesOutstanding", "EntityCommonSharesOutstanding"]),
        ("us-gaap", ["CommonStockSharesOutstanding", "ShareIssued"]),
        ("ifrs-full", ["NumberOfSharesIssued", "IssuedCapitalNumberOfShares", "OrdinarySharesNumber", "NumberOfOrdinaryShares"]),
    ]
    unit_keys = ("shares", "USD_shares", "Shares", "SHARES")
    for taxo, keys in sources:
        sect = facts_all.get(taxo, {})
        for key in keys:
            fact = sect.get(key)
            if not fact:
                continue
            units = fact.get("units") or {}
            for uk in unit_keys:
                arr = units.get(uk)
                if not isinstance(arr, list):
                    continue
                for it in arr:
                    if not _is_instant_entry(it):
                        continue
                    end = _parse_date(str(it.get("end", "")))
                    val = it.get("val", None)
                    if end and val is not None:
                        try:
                            v = float(val)
                            entries.append({"end": end, "val": v})
                        except Exception:
                            pass
    return entries

def _sec_latest_shares_robust(facts: dict) -> float:
    rows = _collect_share_entries(facts)
    if not rows:
        return 0.0
    newest = max(r["end"] for r in rows)
    todays = [r for r in rows if r["end"] == newest]
    total = 0.0
    for r in todays:
        try:
            total += float(r["val"])
        except Exception:
            pass
    return total if total > 0 else 0.0

def _sec_quarterly_revenues_dated_with_unit(facts: dict, max_quarters: int = 20) -> Tuple[List[Tuple[Any, float]], Optional[str]]:
    """
    Hämtar kvartalsintäkter (3-mån) (nyast->äldst) och returnerar (rows, unit)
    där rows = [(end_date, value), ...].
    Filtrerar på formulär (10-Q, 6-K) och duration ~90 dagar.
    """
    taxos = [
        ("us-gaap",  {"forms": ("10-Q", "10-Q/A")}),
        ("ifrs-full", {"forms": ("6-K", "6-K/A", "10-Q", "10-Q/A")}),
    ]
    rev_keys = [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "Revenues",
        "Revenue",
        "RevenueFromContractsWithCustomers",
        "RevenueFromContractsWithCustomersExcludingSalesTaxes",
    ]
    prefer_units = ("USD","CAD","EUR","GBP","SEK","NOK","CHF","JPY")

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
                    end = _parse_date(str(it.get("end", "")))
                    start = _parse_date(str(it.get("start", "")))
                    val = it.get("val", None)
                    if not (end and start and val is not None):
                        continue
                    try:
                        dur = (end - start).days
                    except Exception:
                        dur = None
                    # ~kvartal (ca 90 dagar)
                    if dur is None or dur < 70 or dur > 110:
                        continue
                    try:
                        v = float(val)
                        tmp.append((end, v))
                    except Exception:
                        pass
                if not tmp:
                    continue
                # Deduplikera per end-date
                ded = {}
                for end, v in tmp:
                    ded[end] = v
                rows = sorted(ded.items(), key=lambda t: t[0], reverse=True)[:max_quarters]
                if rows:
                    return rows, unit_code
    return [], None


# ------------------------------------------------------------
# --- Yahoo (pris, mcap, historiska priser) -----------------
# ------------------------------------------------------------
def _yfi_info(t: yf.Ticker) -> dict:
    try:
        return t.info or {}
    except Exception:
        return {}

def _yfi_price_currency_name(t: yf.Ticker) -> Tuple[float, str, str]:
    price = np.nan; cur = "USD"; name = ""
    info = _yfi_info(t)
    p = info.get("regularMarketPrice")
    if p is None:
        try:
            hist = t.history(period="1d")
            if not hist.empty and "Close" in hist:
                p = float(hist["Close"].iloc[-1])
        except Exception:
            p = None
    if p is not None:
        price = float(p)
    c = info.get("currency")
    if c:
        cur = str(c).upper()
    nm = info.get("shortName") or info.get("longName") or ""
    if nm:
        name = str(nm)
    return price, cur, name

def _yfi_marketcap(t: yf.Ticker) -> float:
    info = _yfi_info(t)
    m = info.get("marketCap")
    try:
        return float(m) if m is not None else np.nan
    except Exception:
        return np.nan

def _yahoo_prices_for_dates(ticker: str, dates: List) -> Dict:
    """
    Returnera stängningskursen på eller närmast FÖRE respektive datum.
    """
    if not dates:
        return {}
    dmin = min(dates) - _td(days=14)
    dmax = max(dates) + _td(days=2)
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


# ------------------------------------------------------------
# --- Publikt API -------------------------------------------
# ------------------------------------------------------------
def fetch_price_only(ticker: str) -> Dict[str, Any]:
    """
    Snabbhämtning av kurs, valuta, bolagsnamn. Sätter inget annat.
    """
    out: Dict[str, Any] = {}
    try:
        t = yf.Ticker(ticker)
        price, cur, name = _yfi_price_currency_name(t)
        if not math.isnan(price):
            out["Aktuell kurs"] = float(price)
        if cur:
            out["Valuta"] = cur
        if name:
            out["Bolagsnamn"] = name
        # marketcap (om vi ändå får den gratis)
        mcap = _yfi_marketcap(t)
        if not math.isnan(mcap) and mcap > 0:
            out["MCAP nu"] = float(mcap)
    except Exception:
        pass
    return out


def fetch_full_ticker(ticker: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Full uppdatering för ett bolag. Returnerar (vals, debug).
    Sätter inte manuella prognosfält.
    """
    vals: Dict[str, Any] = {}
    dbg: Dict[str, Any] = {"ticker": ticker}

    t = yf.Ticker(ticker)
    price, px_ccy, name = _yfi_price_currency_name(t)
    mcap_now = _yfi_marketcap(t)

    if not math.isnan(price):
        vals["Aktuell kurs"] = float(price)
    if px_ccy:
        vals["Valuta"] = px_ccy
    if name:
        vals["Bolagsnamn"] = name
    if not math.isnan(mcap_now) and mcap_now > 0:
        vals["MCAP nu"] = float(mcap_now)

    # SEC / IFRS försök
    cik = _sec_cik_for(ticker)
    dbg["cik"] = cik
    used_shares = 0.0
    if cik:
        facts, sc = _sec_companyfacts(cik)
        dbg["sec_companyfacts_sc"] = sc
        if isinstance(facts, dict) and sc == 200:
            # Aktier (instant)
            sec_sh = _sec_latest_shares_robust(facts)
            # Implied via Yahoo
            implied = 0.0
            if not math.isnan(mcap_now) and mcap_now > 0 and not math.isnan(price) and price > 0:
                implied = mcap_now / price

            if implied > 0:
                used_shares = implied
                dbg["shares_source"] = "Yahoo implied (mcap/price)"
            elif sec_sh > 0:
                used_shares = sec_sh
                dbg["shares_source"] = "SEC instant robust"
            else:
                dbg["shares_source"] = "unknown"

            if used_shares > 0:
                vals["Utestående aktier"] = float(used_shares) / 1e6  # lagras i miljoner

            # Kvartalsintäkter (rows, unit)
            rows, unit = _sec_quarterly_revenues_dated_with_unit(facts, max_quarters=20)
            dbg["rev_unit"] = unit
            dbg["rev_rows_len"] = len(rows)

            # P/S (TTM) nu + historik
            if rows:
                # Skapa TTM-fönster (senaste 5 för säkerhets skull)
                rows_sorted = sorted(rows, key=lambda x: x[0], reverse=True)
                ttm = []
                for i in range(0, min(5, len(rows_sorted) - 3)):
                    end_i = rows_sorted[i][0]
                    ttm_i = sum(v for (_, v) in rows_sorted[i:i+4])
                    ttm.append((end_i, float(ttm_i)))

                # Konvertera till prisvalutan om behöver
                conv = 1.0
                if unit and px_ccy and unit.upper() != px_ccy.upper():
                    conv = _fx_rate(unit.upper(), px_ccy.upper())
                ttm_px = [(d, v * conv) for (d, v) in ttm]

                # P/S nu (mcap/ltm)
                if not math.isnan(mcap_now) and mcap_now > 0 and ttm_px:
                    ltm_now = ttm_px[0][1]
                    if ltm_now > 0:
                        vals["P/S"] = float(mcap_now / ltm_now)

                # P/S Q1..Q4 – använd samma 'used_shares' för enkelhet
                if used_shares > 0 and len(ttm_px) >= 4:
                    q_dates = [d for (d, _) in ttm_px[:4]]
                    px_map = _yahoo_prices_for_dates(ticker, q_dates)
                    for idx, (d_end, ttm_rev_px) in enumerate(ttm_px[:4], start=1):
                        if ttm_rev_px and ttm_rev_px > 0:
                            price_hist = px_map.get(d_end)
                            if price_hist and price_hist > 0:
                                mcap_hist = used_shares * float(price_hist)
                                vals[f"P/S Q{idx}"] = float(mcap_hist / ttm_rev_px)
        else:
            dbg["sec_companyfacts_err"] = f"HTTP {sc}"
    else:
        dbg["cik"] = None

    # Global fallback om P/S/aktier saknas
    if "Utestående aktier" not in vals or "P/S" not in vals:
        info = _yfi_info(t)
        px = price if not math.isnan(price) else np.nan
        mcap = mcap_now if not math.isnan(mcap_now) else np.nan

        shares = 0.0
        if mcap and not math.isnan(mcap) and px and not math.isnan(px) and px > 0:
            shares = mcap / px
            dbg["shares_source_fallback"] = "Yahoo implied (mcap/price)"
        else:
            so = info.get("sharesOutstanding")
            try:
                so = float(so or 0.0)
            except Exception:
                so = 0.0
            if so > 0:
                shares = so
                dbg["shares_source_fallback"] = "Yahoo sharesOutstanding"

        if shares > 0 and "Utestående aktier" not in vals:
            vals["Utestående aktier"] = float(shares) / 1e6

        # Försök med Yahoo quarterly financials för TTM
        try:
            qf = t.quarterly_financials
            if isinstance(qf, pd.DataFrame) and not qf.empty:
                idx = [str(x).strip() for x in qf.index]
                cand_rows = [
                    "Total Revenue","TotalRevenue","Revenues","Revenue","Sales",
                    "Total revenue","Revenues from contracts with customers"
                ]
                row = None
                for key in cand_rows:
                    if key in idx:
                        row = qf.loc[key].dropna()
                        break
                if row is not None and len(row) >= 4:
                    vals_list = []
                    for c, v in row.items():
                        try:
                            d = c.date() if hasattr(c, "date") else pd.to_datetime(c).date()
                            vals_list.append((d, float(v)))
                        except Exception:
                            pass
                    vals_list.sort(key=lambda x: x[0], reverse=True)
                    ttm = []
                    for i in range(0, min(5, len(vals_list)-3)):
                        end_i = vals_list[i][0]
                        ttm_i = sum(v for (_, v) in vals_list[i:i+4])
                        ttm.append((end_i, float(ttm_i)))

                    # valuta för räkenskaper
                    fin_cur = str(info.get("financialCurrency") or px_ccy).upper()
                    conv = 1.0
                    if fin_cur != px_ccy:
                        conv = _fx_rate(fin_cur, px_ccy)
                    ttm_px = [(d, v * conv) for (d, v) in ttm]

                    if mcap and not math.isnan(mcap) and mcap > 0 and ttm_px:
                        ltm_now = ttm_px[0][1]
                        if ltm_now > 0:
                            vals["P/S"] = float(mcap / ltm_now)

                    if shares > 0 and len(ttm_px) >= 4:
                        q_dates = [d for (d, _) in ttm_px[:4]]
                        px_map = _yahoo_prices_for_dates(ticker, q_dates)
                        for idx, (d_end, ttm_rev_px) in enumerate(ttm_px[:4], start=1):
                            if ttm_rev_px and ttm_rev_px > 0:
                                price_hist = px_map.get(d_end)
                                if price_hist and price_hist > 0:
                                    mcap_hist = shares * float(price_hist)
                                    vals[f"P/S Q{idx}"] = float(mcap_hist / ttm_rev_px)
        except Exception:
            pass

    # Sista touch: rensa orimliga P/S (negativa, noll)
    for k in ["P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"]:
        if k in vals:
            v = _to_float(vals.get(k), default=np.nan)
            if math.isnan(v) or v <= 0:
                vals.pop(k, None)

    return vals, dbg
