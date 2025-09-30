# -*- coding: utf-8 -*-
"""
SEC EDGAR fetcher

Publik:
    fetch_sec(ticker: str) -> (data: dict, status_code: int, source: str)

Hämtar från EDGAR Company Facts:
  - Utestående aktier (robust summering av instant-poster för senaste datum)
  - Kvartalsintäkter (3-mån) + valutaenhet (US-GAAP 10-Q / IFRS 6-K)
  - Returnerar även ett nyckelfält _SEC_rev_quarterly (lista med {end, value, unit})
    så att orchestrator kan bygga TTM, P/S-historik osv med pris/marketcap från andra källor.

OBS:
  - SEC-exemplet beräknar INTE P/S då det kräver market cap/pris.
    Låt Yahoo/FMP stå för pris/mcap — sec.py fokuserar på shares & revenue.
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, List
import requests
import streamlit as st
from datetime import date, datetime, timedelta

TIMEOUT = 30

# User-Agent enligt SEC:s krav
SEC_USER_AGENT = st.secrets.get(
    "SEC_USER_AGENT",
    "StockApp/1.0 (contact: your-email@example.com)"
)


# ---------------------------- helpers ----------------------------
def _sec_get(url: str, params: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Any], int]:
    try:
        r = requests.get(url, params=params or {}, headers={"User-Agent": SEC_USER_AGENT}, timeout=TIMEOUT)
        if r.status_code == 200:
            return r.json(), 200
        return None, r.status_code
    except Exception:
        return None, 599


@st.cache_data(show_spinner=False, ttl=86400)
def _sec_ticker_map() -> Dict[str, str]:
    """
    SEC publicerar en stor JSON med ticker->CIK. Cache 24h.
    """
    j, sc = _sec_get("https://www.sec.gov/files/company_tickers.json")
    if not isinstance(j, dict):
        return {}
    out: Dict[str, str] = {}
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
        # "2025-01-31" eller "2025-01-31Z"
        return datetime.fromisoformat(d.replace("Z", "+00:00")).date()
    except Exception:
        try:
            return datetime.strptime(d, "%Y-%m-%d").date()
        except Exception:
            return None


def _is_instant_entry(it: Dict[str, Any]) -> bool:
    """
    Instant-poster har end men saknar start, alternativt (end-start) <= 2 dagar.
    """
    end = it.get("end")
    start = it.get("start")
    if not end:
        return False
    if not start:
        return True
    d1 = _parse_iso(str(start))
    d2 = _parse_iso(str(end))
    if d1 and d2:
        try:
            return (d2 - d1).days <= 2
        except Exception:
            return False
    return False


def _collect_share_entries(facts: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Samlar instant-poster för antal aktier i dei/us-gaap/ifrs-full.
    Returnerar [{"end": date, "val": float, "frame": str, "form": str, "taxo": str, "concept": str}, ...]
    """
    entries: List[Dict[str, Any]] = []
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


def _latest_shares_sum(facts: Dict[str, Any]) -> float:
    """
    Hämtar senaste end-datum och summerar alla instant-poster för just det datumet
    (hanterar multi-class).
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


def _quarterly_revenues_with_unit(facts: Dict[str, Any], max_quarters: int = 20) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Försöker läsa kvartalsintäkter (3-mån) med valutaenhet.
    Stöd för US-GAAP (10-Q/10-QA) och IFRS (6-K/6-KA). Returnerar (rows, unit_code)
    där rows = [{"end": "YYYY-MM-DD", "value": float, "unit": "USD"}, ...] nyast först.
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
                tmp: List[Tuple[date, float]] = []
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
                    # Godkänn ~kvartal (70-100 dagar)
                    if dur is None or dur < 70 or dur > 100:
                        continue
                    try:
                        v = float(val)
                        tmp.append((end, v))
                    except Exception:
                        pass
                if not tmp:
                    continue
                # deduplicera per end & ta senaste (SEC kan ha flera)
                ded: Dict[date, float] = {}
                for end_dt, v in tmp:
                    ded[end_dt] = v
                rows = sorted(ded.items(), key=lambda t: t[0], reverse=True)[:max_quarters]
                if rows:
                    out = [{"end": d.strftime("%Y-%m-%d"), "value": float(v), "unit": unit_code} for (d, v) in rows]
                    return out, unit_code
    return [], None


# ---------------------------- core fetch ----------------------------
def fetch_sec(ticker: str) -> Tuple[Dict[str, Any], int, str]:
    """
    Returnerar (data, status_code, "SEC").
    data-nycklar (kompatibla med övriga källor):
        - "Utestående aktier (milj.)"
        - "_SEC_rev_quarterly": [{"end","value","unit"}, ...] nyast först
        - "_SEC_rev_unit": "USD"/"EUR"/...
    Resten lämnas som None för orchestrationen att fylla (pris/mcap/marginaler etc).
    """
    cik = _cik_for(ticker)
    if not cik:
        return {}, 404, "SEC (CIK saknas)"

    facts, sc = _sec_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json")
    if sc != 200 or not isinstance(facts, dict):
        return {}, sc or 599, "SEC"

    out: Dict[str, Any] = {}

    # Basfält som SEC INTE fyller (behövs för schema-align)
    out["Namn"] = None
    out["Sektor"] = None
    out["Bransch"] = None
    out["Senaste kurs"] = None
    out["Market Cap"] = None
    out["Bruttomarginal (%)"] = None
    out["Operating margin (%)"] = None
    out["Net margin (%)"] = None
    out["Debt/Equity"] = None
    out["P/B"] = None
    out["Dividend yield (%)"] = None
    out["P/S"] = None
    out["P/S (Yahoo)"] = None
    out["P/S Q1"] = None
    out["P/S Q2"] = None
    out["P/S Q3"] = None
    out["P/S Q4"] = None

    # Utestående aktier (robust)
    shares = _latest_shares_sum(facts)
    if shares and shares > 0:
        out["Utestående aktier (milj.)"] = float(shares) / 1e6
    else:
        out["Utestående aktier (milj.)"] = None

    # Kvartalsintäkter + unit
    rows, unit_code = _quarterly_revenues_with_unit(facts, max_quarters=20)
    out["_SEC_rev_quarterly"] = rows
    out["_SEC_rev_unit"] = unit_code

    return out, 200, "SEC"
