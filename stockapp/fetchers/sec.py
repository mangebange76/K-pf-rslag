# -*- coding: utf-8 -*-
"""
stockapp/fetchers/sec.py

SEC-hämtare:
- Robust utestående aktier (instant; multi-class; dei/us-gaap/ifrs-full)
- Kvartalsintäkter (3m) för US-GAAP & IFRS (10-Q/6-K), inkl. Dec/Jan-fönster
- TTM-fönster (upp till 4)
- Valfri P/S (TTM) om market_cap & prisvaluta ges (med FX-konvertering)
- Valfri P/S Q1..Q4 om hist_price_lookup(date)->price ges och shares finns

Returnerar (vals, debug) och rör inte manuella prognosfält.

Kräver: st.secrets["SEC_USER_AGENT"] (valfritt, men starkt rekommenderat)
"""

from __future__ import annotations
from typing import Dict, Tuple, Any, Optional, Callable, List
import requests
import streamlit as st
from datetime import datetime, timedelta, date

# ----------------------------- Hjälpare --------------------------------------

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v != v:  # NaN
            return default
        return v
    except Exception:
        return default

def _ua() -> str:
    return st.secrets.get("SEC_USER_AGENT", "StockApp/1.0 (contact: your-email@example.com)")

def _sec_get(url: str, params: Optional[dict] = None):
    try:
        r = requests.get(url, params=params or {}, headers={"User-Agent": _ua()}, timeout=30)
        if r.status_code == 200:
            return r.json(), 200
        return None, r.status_code
    except Exception:
        return None, 0

@st.cache_data(show_spinner=False, ttl=86400)
def _sec_ticker_map() -> Dict[str, str]:
    j, sc = _sec_get("https://www.sec.gov/files/company_tickers.json")
    if not isinstance(j, dict):
        return {}
    out = {}
    # {"0":{"cik_str":320193,"ticker":"AAPL","title":"Apple Inc."}, ...}
    for _, v in j.items():
        try:
            out[str(v["ticker"]).upper()] = str(v["cik_str"]).zfill(10)
        except Exception:
            pass
    return out

def _cik_for(ticker: str) -> Optional[str]:
    return _sec_ticker_map().get(str(ticker).upper())

def _parse_iso(d: str) -> Optional[date]:
    try:
        # "2024-12-31" eller "2024-12-31Z"
        return datetime.fromisoformat(d.replace("Z", "+00:00")).date()
    except Exception:
        try:
            return datetime.strptime(d, "%Y-%m-%d").date()
        except Exception:
            return None

def _is_instant_entry(it: dict) -> bool:
    end = it.get("end"); start = it.get("start")
    if not end:
        return False
    if not start:
        return True  # instant
    d1 = _parse_iso(str(start)); d2 = _parse_iso(str(end))
    if d1 and d2:
        try:
            return (d2 - d1).days <= 2  # ≈ instant
        except Exception:
            return False
    return False

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

def _latest_shares_robust(facts: dict) -> float:
    """
    Summerar multi-class för senaste 'end' (instant).
    Returnerar antal aktier (styck), 0.0 om ej hittat.
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

@st.cache_data(show_spinner=False, ttl=21600)
def _fx_rate(base: str, quote: str) -> float:
    """Enkel FX via Frankfurter -> exchangerate.host fallback."""
    base = (base or "").upper(); quote = (quote or "").upper()
    if not base or not quote or base == quote:
        return 1.0
    try:
        r = requests.get("https://api.frankfurter.app/latest", params={"from": base, "to": quote}, timeout=12)
        if r.status_code == 200:
            v = (r.json() or {}).get("rates", {}).get(quote)
            if v:
                return float(v)
    except Exception:
        pass
    try:
        r = requests.get("https://api.exchangerate.host/latest", params={"base": base, "symbols": quote}, timeout=12)
        if r.status_code == 200:
            v = (r.json() or {}).get("rates", {}).get(quote)
            if v:
                return float(v)
    except Exception:
        pass
    return 1.0

def _quarterly_revenues_dated_with_unit(facts: dict, max_quarters: int = 20):
    """
    Hämtar upp till 'max_quarters' kvartalsintäkter (3-mån) från US-GAAP/IFRS.
    Returnerar (rows, unit) där rows=[(end_date, value), ...] nyast→äldst.
    Filtrerar på 10-Q/6-K (även -/A), och intervall 70–100 dagar för att få Dec/Jan rätt.
    """
    taxos = [
        ("us-gaap",  {"forms": ("10-Q", "10-Q/A")}),
        ("ifrs-full",{"forms": ("6-K", "6-K/A", "10-Q", "10-Q/A")}),
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
                    if dur is None or dur < 70 or dur > 100:
                        # kräver ~kvartalslängd för att få Dec/Jan-kvartal korrekt
                        continue
                    try:
                        v = float(val)
                        tmp.append((end, v))
                    except Exception:
                        pass
                if not tmp:
                    continue
                # deduplicera per end-datum (ta senaste värdet)
                ded = {}
                for end, v in tmp:
                    ded[end] = v
                rows = sorted(ded.items(), key=lambda t: t[0], reverse=True)[:max_quarters]
                if rows:
                    return rows, unit_code
    return [], None

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


# ----------------------------- Publikt API -----------------------------------

def fetch_sec_combo(
    ticker: str,
    market_cap: Optional[float] = None,
    price_ccy: Optional[str] = "USD",
    hist_price_lookup: Optional[Callable[[date], float]] = None,
    override_shares: Optional[float] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Hämta SEC-data för 'ticker'. Returnerar (vals, debug).

    - Sätter "Utestående aktier" (milj) via robust instant-summering (multi-class).
      Om override_shares anges (styck), används den istället.
    - Hämtar kvartalsintäkter (3m) + bygger TTM.
    - Om market_cap & price_ccy ges, försöker räkna P/S (TTM). Gör FX-konvertering
      om revenue-unit != price_ccy.
    - Om hist_price_lookup ges och vi har shares, räknas P/S Q1..Q4 historiskt.

    Parametrar:
      ticker:        SEC/Yahoo-ticker (AAPL, NVDA, SHOP, ...).
      market_cap:    Nuvarande market cap i prisvalutan (float).
      price_ccy:     Prisvaluta för market_cap (t.ex. "USD").
      hist_price_lookup: callable som tar 'date' -> close price i prisvalutan.
      override_shares: Om du redan har exakt shares (styck), skicka in här.

    Obs: Sektors/bransch-info finns inte i SEC-companyfacts; lämnas tomt här.
    """
    sym = str(ticker).strip().upper()
    vals: Dict[str, Any] = {}
    dbg: Dict[str, Any] = {"ticker": sym, "source": "SEC"}

    cik = _cik_for(sym)
    dbg["cik"] = cik
    if not cik:
        dbg["note"] = "no CIK mapping"
        return vals, dbg

    facts, sc = _sec_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json")
    dbg["companyfacts_sc"] = sc
    if sc != 200 or not isinstance(facts, dict):
        dbg["note"] = "companyfacts fetch failed"
        return vals, dbg

    # 1) Shares (robust)
    shares_used = 0.0
    if override_shares and override_shares > 0:
        shares_used = float(override_shares)
        dbg["_shares_source"] = "override"
    else:
        sec_sh = _latest_shares_robust(facts)
        if sec_sh and sec_sh > 0:
            shares_used = float(sec_sh)
            dbg["_shares_source"] = "SEC instant (robust)"

    if shares_used > 0:
        vals["Utestående aktier"] = shares_used / 1e6  # miljoner

    # 2) Kvartalsintäkter + unit
    q_rows, rev_unit = _quarterly_revenues_dated_with_unit(facts, max_quarters=20)
    dbg["rev_unit"] = rev_unit
    dbg["quarters_found"] = len(q_rows)

    if not q_rows or not rev_unit:
        # Vi kan ändå returnera shares om vi hittade det
        return vals, dbg

    # 3) Bygg TTM-lista
    ttm_list = _ttm_windows(q_rows, need=4)  # [(date, ttm_value_in_rev_unit), ...]
    dbg["ttm_count"] = len(ttm_list)

    # 4) P/S (TTM) nu, om mcap finns. Konvertera revenue till price_ccy
    if market_cap and market_cap > 0 and ttm_list:
        try:
            px_ccy = (price_ccy or "USD").upper()
            rev_ccy = rev_unit.upper()
            fx = 1.0 if rev_ccy == px_ccy else _fx_rate(rev_ccy, px_ccy)
            ltm_now = float(ttm_list[0][1]) * fx
            if ltm_now > 0:
                vals["P/S"] = float(market_cap) / ltm_now
                dbg["_ps_source"] = "SEC TTM via revenue + mcap"
        except Exception:
            pass

    # 5) P/S Q1..Q4 historiskt om vi kan få historiska priser & shares
    if hist_price_lookup and shares_used > 0 and ttm_list:
        # använd samma shares för historiken (approx)
        for idx, (d_end, ttm_rev) in enumerate(ttm_list[:4], start=1):
            try:
                px = float(hist_price_lookup(d_end) or 0.0)
            except Exception:
                px = 0.0
            if px > 0 and ttm_rev > 0:
                # market cap historiskt = shares * price
                mcap_hist = shares_used * px
                # ev. FX: ttm_rev är i rev_unit; price/mcap är i price_ccy
                px_ccy = (price_ccy or "USD").upper()
                rev_ccy = rev_unit.upper()
                fx = 1.0 if rev_ccy == px_ccy else _fx_rate(rev_ccy, px_ccy)
                ttm_px = float(ttm_rev) * fx
                if ttm_px > 0:
                    vals[f"P/S Q{idx}"] = mcap_hist / ttm_px

    # extra debug
    if ttm_list:
        dbg["ttm_dates"] = [d.strftime("%Y-%m-%d") for (d, _) in ttm_list[:4]]

    return vals, dbg
