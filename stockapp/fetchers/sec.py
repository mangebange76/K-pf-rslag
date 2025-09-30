# -*- coding: utf-8 -*-
"""
stockapp/fetchers/sec.py

SEC + Yahoo-kombo:
- Hämta CIK, companyfacts (US-GAAP & IFRS), robusta "instant" aktier (multi-class).
- Kvartalsintäkter (10-Q, 6-K) -> bygg TTM-summor (upp till 4 fönster).
- Valutakonvertering (Frankfurter -> exchangerate.host fallback) till prisvaluta.
- Yahoo (yfinance) för pris, market cap, valuta, bolagsnamn, sektor/bransch,
  samt historiska priser vid respektive TTM-datum för P/S Q1–Q4.

Returnerar (vals, debug) där `vals` kan skrivas in i din DataFrame.
Sätter *inte* 'Omsättning idag' eller 'Omsättning nästa år' (manuellt fält).
"""

from __future__ import annotations
from typing import Dict, Tuple, Any, Optional, List
import requests
import streamlit as st
import datetime as dt

import yfinance as yf

# ----------------------------- SEC-konfig ------------------------------------

SEC_USER_AGENT = st.secrets.get(
    "SEC_USER_AGENT",
    "StockApp/1.0 (contact: example@example.com)"
)

# ----------------------------- Hjälpare --------------------------------------


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v != v:  # NaN
            return default
        return v
    except Exception:
        return default


def _parse_iso(d: str) -> Optional[dt.date]:
    if not d:
        return None
    try:
        # Hantera ev. "Z"
        if d.endswith("Z"):
            d = d[:-1] + "+00:00"
        return dt.datetime.fromisoformat(d).date()
    except Exception:
        # fallback vanligaste formatet
        try:
            return dt.datetime.strptime(d, "%Y-%m-%d").date()
        except Exception:
            return None


def _is_instant_entry(it: dict) -> bool:
    end = it.get("end")
    start = it.get("start")
    if not end:
        return False
    if not start:
        return True
    d1 = _parse_iso(str(start))
    d2 = _parse_iso(str(end))
    if not (d1 and d2):
        return False
    try:
        return (d2 - d1).days <= 2
    except Exception:
        return False


def _ttm_windows(values: List[Tuple[dt.date, float]], need: int = 4) -> List[Tuple[dt.date, float]]:
    """
    Tar [(end_date, kvartalsintäkt), ...] (nyast→äldst) och bygger upp till 'need' TTM-summor:
    [(end_date0, ttm0), (end_date1, ttm1), ...] där ttm0 = sum(q0..q3), ttm1 = sum(q1..q4), osv.
    """
    out: List[Tuple[dt.date, float]] = []
    if len(values) < 4:
        return out
    for i in range(0, min(need, len(values) - 3)):
        end_i = values[i][0]
        ttm_i = sum(v for (_, v) in values[i:i+4])
        out.append((end_i, float(ttm_i)))
    return out


# ----------------------------- SEC API ---------------------------------------


def _sec_get(url: str, params=None) -> Tuple[Optional[Any], int]:
    try:
        r = requests.get(
            url,
            params=params or {},
            headers={"User-Agent": SEC_USER_AGENT},
            timeout=30,
        )
        if r.status_code == 200:
            return r.json(), 200
        return None, r.status_code
    except Exception:
        return None, 0


def _sec_ticker_map() -> Dict[str, str]:
    """
    Hämtar mapping TICKER -> CIK (10 siffror som str).
    """
    j, sc = _sec_get("https://www.sec.gov/files/company_tickers.json")
    out: Dict[str, str] = {}
    if not isinstance(j, dict):
        return out
    # {"0":{"cik_str":320193,"ticker":"AAPL","title":"Apple Inc."}, ...}
    for _, v in j.items():
        try:
            t = str(v["ticker"]).upper()
            cik10 = str(v["cik_str"]).zfill(10)
            out[t] = cik10
        except Exception:
            pass
    return out


def _sec_cik_for(ticker: str) -> Optional[str]:
    return _sec_ticker_map().get(str(ticker).upper())


def _sec_companyfacts(cik10: str):
    return _sec_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json")


# ----------------------------- SEC parsing -----------------------------------


def _collect_share_entries(facts: dict) -> list:
    """
    Hämtar alla 'instant' aktieposter från dei/us-gaap/ifrs-full (unit='shares' m.fl.).
    Returnerar [{"end": date, "val": float, "frame": "...", "form": "..."}].
    """
    entries = []
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
                    end = _parse_iso(str(it.get("end", "")))
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
    """
    Summerar multi-class per senaste 'end' (instant). Om flera olika 'end' finns,
    välj det senaste datumet och summera alla frames för det datumet.
    Returnerar aktier (styck), 0 om ej hittat.
    """
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


def _sec_quarterly_revenues_dated_with_unit(facts: dict, max_quarters: int = 20):
    """
    Hämtar upp till 'max_quarters' kvartalsintäkter (3-mån) för US-GAAP (10-Q) och IFRS (6-K).
    Returnerar (rows, unit) med rows=[(end_date, value), ...] nyast→äldst.
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
    prefer_units = ("USD", "CAD", "EUR", "GBP")

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
                    end = _parse_iso(str(it.get("end", "")))
                    start = _parse_iso(str(it.get("start", "")))
                    val = it.get("val", None)
                    if not (end and start and val is not None):
                        continue
                    try:
                        dur = (end - start).days
                    except Exception:
                        dur = None
                    # 3-mån ~ 70..100 dagar
                    if dur is None or dur < 70 or dur > 100:
                        continue
                    try:
                        v = float(val)
                        tmp.append((end, v))
                    except Exception:
                        pass
                if not tmp:
                    continue
                # deduplicera per end-date (välj sista)
                ded = {}
                for end, v in tmp:
                    ded[end] = v
                rows = sorted(ded.items(), key=lambda t: t[0], reverse=True)[:max_quarters]
                if rows:
                    return rows, unit_code
    return [], None


# ----------------------------- Valutor ---------------------------------------


def _fx_rate_cached(base: str, quote: str) -> float:
    """
    Enkel FX (dagens) via Frankfurter -> exchangerate.host fallback.
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


# ----------------------------- Yahoo (pris m.m.) -----------------------------


def _yahoo_basics(ticker: str) -> Dict[str, Any]:
    """
    Hämtar pris, valuta, namn, market cap, sektor, bransch via yfinance.
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
        # Valuta
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
        # sharesOutstanding (kan vara bra fallback-debug)
        so = info.get("sharesOutstanding")
        if so is not None:
            try:
                out["_yf_shares_out"] = float(so)
            except Exception:
                pass
    except Exception:
        pass
    return out


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


# ----------------------------- Publikt API -----------------------------------


def fetch_sec_combo(ticker: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Full SEC+Yahoo combo:
    - SEC: robusta aktier (instant, multi-class), kvartalsintäkter (10-Q/6-K) -> TTM
    - Yahoo: pris/valuta/market cap/namn/sektor/bransch + historiska priser.
    - Beräknar P/S (TTM) nu och P/S Q1..Q4 historiskt.
    - Sätter *inte* 'Omsättning idag' eller 'Omsättning nästa år'.

    Returnerar (vals, debug).
    """
    vals: Dict[str, Any] = {}
    dbg: Dict[str, Any] = {"ticker": ticker}

    # Yahoo-basics (pris/valuta/namn/sektor/bransch/mcap)
    yb = _yahoo_basics(ticker)
    for k in ("Bolagsnamn", "Valuta", "Aktuell kurs", "Market Cap", "Sektor", "Bransch"):
        if yb.get(k) not in (None, "", 0, 0.0):
            vals[k] = yb[k]

    # SEC companyfacts
    cik = _sec_cik_for(ticker)
    dbg["cik"] = cik
    if not cik:
        dbg["error"] = "CIK saknas för ticker"
        dbg["source"] = "SEC+Yahoo (no CIK)"
        # ändå returnera Yahoo-basics som kan fyllas in
        return vals, dbg

    facts, sc = _sec_companyfacts(cik)
    dbg["companyfacts_sc"] = sc
    if sc != 200 or not isinstance(facts, dict):
        dbg["error"] = f"SEC companyfacts status {sc}"
        dbg["source"] = "SEC+Yahoo (facts err)"
        return vals, dbg

    # Shares (robust)
    sec_shares = _sec_latest_shares_robust(facts)  # styck
    dbg["sec_shares_instant_sum"] = sec_shares

    # Implied från Yahoo (marketCap/price)
    shares_used = 0.0
    yf_mcap = _safe_float(yb.get("Market Cap"), 0.0)
    yf_px = _safe_float(yb.get("Aktuell kurs"), 0.0)
    if yf_mcap > 0 and yf_px > 0:
        shares_used = yf_mcap / max(yf_px, 1e-9)
        dbg["_shares_source"] = "Yahoo implied (mcap/price)"
    elif sec_shares > 0:
        shares_used = sec_shares
        dbg["_shares_source"] = "SEC instant (robust)"
    else:
        dbg["_shares_source"] = "unknown"

    if shares_used > 0:
        vals["Utestående aktier"] = shares_used / 1e6  # i miljoner (matcha din DB)

    # Market cap nu
    mcap_now = yf_mcap
    if mcap_now <= 0 and shares_used > 0 and yf_px > 0:
        mcap_now = shares_used * yf_px
    if mcap_now > 0:
        vals["Market Cap"] = mcap_now

    # Kvartalsintäkter + unit
    q_rows, rev_unit = _sec_quarterly_revenues_dated_with_unit(facts, max_quarters=20)
    dbg["rev_unit"] = rev_unit
    dbg["q_rows_count"] = len(q_rows)
    if not q_rows or not rev_unit:
        dbg["warn"] = "Inga kvartalsintäkter funna (SEC)."
        dbg["source"] = "SEC+Yahoo (no rev)"
        return vals, dbg

    # Konvertera TTM till prisvaluta (om behövs)
    px_ccy = (vals.get("Valuta") or "USD").upper()
    conv = 1.0
    if rev_unit.upper() != px_ccy:
        conv = _safe_float(_fx_rate_cached(rev_unit.upper(), px_ccy), 1.0)
    ttm_list = _ttm_windows(q_rows, need=4)
    ttm_list_px = [(d, v * conv) for (d, v) in ttm_list]

    # P/S (TTM) nu
    if mcap_now > 0 and ttm_list_px:
        ltm_now = _safe_float(ttm_list_px[0][1], 0.0)
        if ltm_now > 0:
            vals["P/S"] = mcap_now / ltm_now

    # P/S Q1..Q4 historiskt (kräver aktier & historiska priser)
    if shares_used > 0 and ttm_list_px:
        q_dates = [d for (d, _) in ttm_list_px[:4]]
        px_map = _yahoo_prices_for_dates(ticker, q_dates)
        for idx, (d_end, ttm_rev) in enumerate(ttm_list_px[:4], start=1):
            if ttm_rev and ttm_rev > 0:
                px_at = _safe_float(px_map.get(d_end), 0.0)
                if px_at > 0:
                    mcap_hist = shares_used * px_at
                    vals[f"P/S Q{idx}"] = mcap_hist / ttm_rev

    dbg["source"] = "SEC+Yahoo"
    return vals, dbg
