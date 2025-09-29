# stockapp/sources.py
# -*- coding: utf-8 -*-
"""
Datakällor och uppdaterare:
- fetch_price_only(ticker) -> dict
- fetch_full_ticker(ticker) -> (vals: dict, debug: dict)

Hämtar basdata via Yahoo och, när möjligt, robusta shares + kvartalsintäkter via SEC (US-GAAP / IFRS).
Räknar P/S (TTM) nu samt P/S Q1–Q4 historiskt med valutakonvertering.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta, date

import requests
import pandas as pd
import numpy as np
import yfinance as yf


# ----------------------------- Små utils ------------------------------------
def _float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def _parse_date(d: str) -> Optional[date]:
    if not d:
        return None
    try:
        return datetime.fromisoformat(d.replace("Z", "+00:00")).date()
    except Exception:
        try:
            return datetime.strptime(d, "%Y-%m-%d").date()
        except Exception:
            return None

def _dedupe_keep_latest(rows: List[Tuple[date, float]], max_items: int) -> List[Tuple[date, float]]:
    """Tar [(end, value)] (blandat) -> unika per 'end' (senaste vinner), sorterat nyast→äldst, max N."""
    if not rows:
        return []
    tmp: Dict[date, float] = {}
    for d, v in rows:
        if not d:
            continue
        tmp[d] = float(v)
    out = sorted(tmp.items(), key=lambda t: t[0], reverse=True)
    return out[:max_items]

def _last_n_unique_quarters(rows: List[Tuple[date, float]], n: int) -> List[Tuple[date, float]]:
    """Säkrar att vi inte tappar Dec/Jan: plocka senaste n UNIKA kvartals-slut."""
    return _dedupe_keep_latest(rows, n)

def _ttm_windows(values: List[Tuple[date, float]], need: int = 4) -> List[Tuple[date, float]]:
    """
    Tar [(end_date, kvartalsintäkt), ...] (nyast→äldst) och bygger upp till 'need' TTM-summor:
    [(end_date0, ttm0), (end_date1, ttm1), ...]
    """
    out: List[Tuple[date, float]] = []
    if len(values) < 4:
        return out
    for i in range(0, min(need, len(values) - 3)):
        end_i = values[i][0]
        ttm_i = sum(v for (_, v) in values[i:i+4])
        out.append((end_i, float(ttm_i)))
    return out

def _yahoo_prices_for_dates(ticker: str, dates: List[date]) -> Dict[date, float]:
    """Close på eller närmast FÖRE varje 'end_date'."""
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

def _fx_rate(base: str, quote: str) -> float:
    """Enkel FX via Frankfurter -> exchangerate.host; 1.0 om misslyckas eller samma valuta."""
    b = (base or "").upper().strip()
    q = (quote or "").upper().strip()
    if not b or not q or b == q:
        return 1.0
    try:
        r = requests.get("https://api.frankfurter.app/latest", params={"from": b, "to": q}, timeout=10)
        if r.status_code == 200:
            v = (r.json() or {}).get("rates", {}).get(q)
            if v:
                return float(v)
    except Exception:
        pass
    try:
        r = requests.get("https://api.exchangerate.host/latest", params={"base": b, "symbols": q}, timeout=10)
        if r.status_code == 200:
            v = (r.json() or {}).get("rates", {}).get(q)
            if v:
                return float(v)
    except Exception:
        pass
    return 1.0


# ------------------------------- SEC -----------------------------------------
_SEC_UA = "StockApp/1.0 (contact: your-email@example.com)"

def _sec_get(url: str, params=None):
    try:
        r = requests.get(url, params=params or {}, headers={"User-Agent": _SEC_UA}, timeout=25)
        if r.status_code == 200:
            return r.json(), 200
        return None, r.status_code
    except Exception:
        return None, 0

def _sec_ticker_map() -> Dict[str, str]:
    j, sc = _sec_get("https://www.sec.gov/files/company_tickers.json")
    if not isinstance(j, dict):
        return {}
    out: Dict[str, str] = {}
    for _, v in j.items():
        try:
            out[str(v["ticker"]).upper()] = str(v["cik_str"]).zfill(10)
        except Exception:
            pass
    return out

def _sec_companyfacts(cik10: str):
    return _sec_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json")

def _is_instant(it: dict) -> bool:
    end = it.get("end"); start = it.get("start")
    if not end:
        return False
    if not start:
        return True
    d1 = _parse_date(str(start)); d2 = _parse_date(str(end))
    if d1 and d2:
        try:
            return (d2 - d1).days <= 2
        except Exception:
            return False
    return False

def _collect_share_entries(facts: dict) -> List[dict]:
    """
    Hämtar alla 'instant' aktieposter från dei/us-gaap/ifrs-full (unit='shares' m.fl.).
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
                    if not _is_instant(it):
                        continue
                    end = _parse_date(str(it.get("end", "")))
                    val = it.get("val", None)
                    if end and val is not None:
                        try:
                            v = float(val)
                            frame = it.get("frame") or ""
                            form = (it.get("form") or "").upper()
                            entries.append({"end": end, "val": v, "frame": frame, "form": form, "taxo": taxo, "concept": key})
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

def _sec_quarterly_revenues_dated_with_unit(facts: dict, max_quarters: int = 20) -> Tuple[List[Tuple[date, float]], Optional[str]]:
    """
    Hämtar upp till 'max_quarters' kvartalsintäkter (3-mån) för US-GAAP (10-Q) och IFRS (6-K).
    Returnerar (rows, unit) där rows=[(end_date, value)] nyast→äldst.
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
                tmp: List[Tuple[date, float]] = []
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
                    # 3-mån fönster ~ 70–100 dagar
                    if dur is None or dur < 70 or dur > 100:
                        continue
                    try:
                        v = float(val)
                        tmp.append((end, v))
                    except Exception:
                        pass
                if not tmp:
                    continue
                rows = _dedupe_keep_latest(tmp, max_quarters)
                if rows:
                    return rows, unit_code
    return [], None


# ----------------------------- Yahoo helpers ---------------------------------
def _yfi_info(t: yf.Ticker) -> dict:
    try:
        return t.info or {}
    except Exception:
        return {}

def _yfi_quarterly_revenues(t: yf.Ticker) -> List[Tuple[date, float]]:
    """
    Försöker läsa kvartalsintäkter från Yahoo.
    Returnerar [(period_end_date, value)] sorterat nyast→äldst.
    """
    # 1) quarterly_financials
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
                    out: List[Tuple[date, float]] = []
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

    # 2) fallback: income_stmt quarterly via v1-api (kan vara tomt)
    try:
        df_is = getattr(t, "income_stmt", None)
        if isinstance(df_is, pd.DataFrame) and not df_is.empty and "Total Revenue" in df_is.index:
            ser = df_is.loc["Total Revenue"].dropna()
            out: List[Tuple[date, float]] = []
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


# ----------------------------- Public API ------------------------------------
def fetch_price_only(ticker: str) -> Dict:
    """
    Snabb uppdatering: kurs, valuta, bolagsnamn (+ev. shares/utdelning om enkelt).
    Returnerar dict med nycklar som matchar dina kolumnnamn.
    """
    out: Dict = {}
    try:
        t = yf.Ticker(ticker)
        info = _yfi_info(t)
        pris = info.get("regularMarketPrice")
        if pris is None:
            h = t.history(period="1d")
            if not h.empty and "Close" in h:
                pris = float(h["Close"].iloc[-1])
        if pris is not None:
            out["Aktuell kurs"] = float(pris)

        valuta = info.get("currency")
        if valuta:
            out["Valuta"] = str(valuta).upper()

        namn = info.get("shortName") or info.get("longName")
        if namn:
            out["Bolagsnamn"] = str(namn)

        # sharesOutstanding -> utestående aktier (miljoner)
        so = info.get("sharesOutstanding")
        if so:
            try:
                out["Utestående aktier"] = float(so) / 1e6
            except Exception:
                pass

        # utdelning (årlig rate)
        div_rate = info.get("dividendRate")
        if div_rate is not None:
            try:
                out["Årlig utdelning"] = float(div_rate)
            except Exception:
                pass

    except Exception:
        pass
    return out


def fetch_full_ticker(ticker: str) -> Tuple[Dict, Dict]:
    """
    Robust heluppdatering: SEC (om möjligt) + Yahoo fallback.
    Returnerar (vals, debug). 'vals' passar till apply_auto_updates_to_row(...).
    """
    debug: Dict = {"ticker": ticker}
    vals: Dict = {}

    # Bas från Yahoo (namn/valuta/pris), samt ev. utdelning
    base = fetch_price_only(ticker)
    vals.update(base)
    debug["base"] = base

    t = yf.Ticker(ticker)
    info = _yfi_info(t)
    price_ccy = (vals.get("Valuta") or info.get("currency") or "USD").upper()
    price_now = _float(vals.get("Aktuell kurs") or info.get("regularMarketPrice"), 0.0)

    # SEC-väg eller global fallback
    cik_map = _sec_ticker_map()
    cik10 = cik_map.get(str(ticker).upper())
    if not cik10:
        # Global fallback
        debug["path"] = "yahoo_global"
        return _yahoo_global_combo(ticker, vals, info), debug

    facts, sc = _sec_companyfacts(cik10)
    if sc != 200 or not isinstance(facts, dict):
        debug["path"] = "yahoo_global_no_facts"
        return _yahoo_global_combo(ticker, vals, info), debug

    debug["path"] = "sec_combo"
    # Shares: Yahoo implied -> SEC robust
    implied = _implied_shares_from_yahoo(info, price_now)
    sec_sh = _sec_latest_shares_robust(facts)
    shares_used = 0.0
    src_sh = ""
    if implied and implied > 0:
        shares_used = float(implied)
        src_sh = "Yahoo implied (mcap/price)"
    elif sec_sh and sec_sh > 0:
        shares_used = float(sec_sh)
        src_sh = "SEC instant (robust)"
    vals["_debug_shares_source"] = src_sh
    if shares_used > 0:
        vals["Utestående aktier"] = shares_used / 1e6

    # Market cap nu
    mcap_now = info.get("marketCap")
    mcap_now = _float(mcap_now, 0.0)
    if mcap_now <= 0 and price_now > 0 and shares_used > 0:
        mcap_now = price_now * shares_used

    # SEC kvartalsintäkter + unit
    q_rows, unit_code = _sec_quarterly_revenues_dated_with_unit(facts, max_quarters=20)
    debug["sec_rev_count"] = len(q_rows)
    if not q_rows or not unit_code:
        return vals, debug

    # Ta de SENASTE 5 unika kvartalen (så vi tappar inte Dec/Jan) -> gör TTM
    q5 = _last_n_unique_quarters(q_rows, 5)
    # Konvertera till prisvaluta
    conv = _fx_rate(unit_code, price_ccy) if unit_code.upper() != price_ccy else 1.0
    q5_px = [(d, v * conv) for (d, v) in q5]

    # TTM-fönster (min 4)
    ttm_list = _ttm_windows(q5_px, need=4)
    if not ttm_list:
        return vals, debug

    # P/S (TTM) nu
    if mcap_now > 0:
        ltm_now = ttm_list[0][1]
        if ltm_now > 0:
            vals["P/S"] = float(mcap_now / ltm_now)

    # P/S Q1–Q4 historik via shares_used * pris(end_date) / TTM
    if shares_used > 0:
        q_dates = [d for (d, _) in ttm_list]
        px_map = _yahoo_prices_for_dates(ticker, q_dates)
        for idx, (d_end, ttm_rev_px) in enumerate(ttm_list[:4], start=1):
            if ttm_rev_px and ttm_rev_px > 0:
                p = px_map.get(d_end)
                if p and p > 0:
                    mcap_hist = shares_used * float(p)
                    vals[f"P/S Q{idx}"] = float(mcap_hist / ttm_rev_px)

    return vals, debug


# ---------------------------- Helpers (global) -------------------------------
def _implied_shares_from_yahoo(info: dict, price: float) -> float:
    mcap = _float(info.get("marketCap"), 0.0)
    p = _float(price, 0.0)
    if mcap > 0 and p > 0:
        return mcap / p
    return 0.0

def _yahoo_global_combo(ticker: str, vals: Dict, info: dict) -> Dict:
    """
    Global fallback för icke-SEC: Yahoo kv-intäkter -> TTM -> P/S & P/S Q1–Q4.
    """
    t = yf.Ticker(ticker)
    px = _float(vals.get("Aktuell kurs") or info.get("regularMarketPrice"), 0.0)
    px_ccy = (vals.get("Valuta") or info.get("currency") or "USD").upper()

    # shares
    shares = 0.0
    mcap = _float(info.get("marketCap"), 0.0)
    if mcap > 0 and px > 0:
        shares = mcap / px
        vals["_debug_shares_source"] = "Yahoo implied (mcap/price)"
    else:
        so = _float(info.get("sharesOutstanding"), 0.0)
        if so > 0:
            shares = so
            vals["_debug_shares_source"] = "Yahoo sharesOutstanding"
    if shares > 0:
        vals["Utestående aktier"] = shares / 1e6

    # kvartalsintäkter -> TTM
    q_rows = _yfi_quarterly_revenues(t)
    if not q_rows:
        return vals
    q5 = _last_n_unique_quarters(q_rows, 5)

    fin_ccy = (info.get("financialCurrency") or px_ccy).upper()
    conv = _fx_rate(fin_ccy, px_ccy) if fin_ccy != px_ccy else 1.0
    q5_px = [(d, v * conv) for (d, v) in q5]

    ttm_list = _ttm_windows(q5_px, need=4)
    if not ttm_list:
        return vals

    # market cap nu
    if mcap <= 0 and shares > 0 and px > 0:
        mcap = shares * px

    # P/S nu
    if mcap > 0:
        ltm_now = ttm_list[0][1]
        if ltm_now > 0:
            vals["P/S"] = float(mcap / ltm_now)

    # P/S Q1–Q4
    if shares > 0:
        q_dates = [d for (d, _) in ttm_list]
        px_map = _yahoo_prices_for_dates(ticker, q_dates)
        for idx, (d_end, ttm_rev_px) in enumerate(ttm_list[:4], start=1):
            if ttm_rev_px and ttm_rev_px > 0:
                p = px_map.get(d_end)
                if p and p > 0:
                    mcap_hist = shares * float(p)
                    vals[f"P/S Q{idx}"] = float(mcap_hist / ttm_rev_px)

    return vals
