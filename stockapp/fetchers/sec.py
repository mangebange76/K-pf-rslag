# stockapp/fetchers/sec.py
from __future__ import annotations

import datetime as dt
from typing import Dict, Any, List, Tuple, Optional

import requests
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None  # type: ignore

# ------- Konfiguration -------
UA = st.secrets.get("SEC_USER_AGENT") or (
    "Mozilla/5.0 (compatible; KpfRslag/1.0; +https://example.com)"
)

SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# ------- Hjälpare -------

def _is_quarter_item(it: dict) -> bool:
    """Filtrera kvartsvisa observationer (Q1..Q4 eller 10-Q/10-K för Q4)."""
    fp = (it.get("fp") or "").upper()
    form = (it.get("form") or "").upper()
    if fp in {"Q1", "Q2", "Q3", "Q4"}:
        return True
    if "10-Q" in form:
        return True
    if "10-K" in form and fp in {"FY", "Q4"}:
        return True
    return False

def _parse_date(s: str) -> Optional[dt.date]:
    try:
        return dt.date.fromisoformat(s)
    except Exception:
        return None

def _safe_float(x) -> float:
    try:
        return float(x) if x is not None else 0.0
    except Exception:
        return 0.0

def _nearest_price_on_or_after(ticker: str, date: dt.date) -> float:
    """Hämta stängningspris närmast på/efter 'date' (fallback närmast före)."""
    if yf is None or not ticker:
        return 0.0
    try:
        t = yf.Ticker(ticker)
        start = date - dt.timedelta(days=5)
        end = date + dt.timedelta(days=15)
        hist = t.history(start=start.isoformat(), end=end.isoformat(), auto_adjust=False)
        if hist.empty:
            return 0.0
        # först datum >= end-date
        sub = hist[hist.index.date >= date]
        if not sub.empty:
            return float(sub["Close"].iloc[0])
        # annars sista före
        return float(hist["Close"].iloc[-1])
    except Exception:
        return 0.0

def _pick_equity_tag(facts: dict) -> Tuple[str, str]:
    """Välj bästa equity-tag och enhet. Prova inkluderande NCI först."""
    candidates = [
        ("StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", "USD"),
        ("StockholdersEquity", "USD"),
        ("CommonStockholdersEquity", "USD"),
        ("TotalPartnersCapital", "USD"),  # partnerships fallback
    ]
    for tag, unit in candidates:
        node = facts.get("us-gaap", {}).get(tag, {})
        if node.get("units", {}).get(unit):
            return tag, unit
    return "", ""

def _pick_shares_tag(facts: dict) -> Tuple[str, str]:
    """Välj bästa shares-tag och enhet (period-end, ej W/A)."""
    candidates = [
        ("CommonStockSharesOutstanding", "shares"),
        ("EntityCommonStockSharesOutstanding", "shares"),
        ("CommonSharesOutstanding", "shares"),
    ]
    for tag, unit in candidates:
        node = facts.get("us-gaap", {}).get(tag, {})
        if node.get("units", {}).get(unit):
            return tag, unit
    return "", ""

def _extract_series(facts: dict, tag: str, unit: str) -> List[Tuple[dt.date, float]]:
    """Plocka ut (end_date, value) för given tag/unit, kvartsfiltrerad, senaste först."""
    items = facts.get("us-gaap", {}).get(tag, {}).get("units", {}).get(unit, []) or []
    out: List[Tuple[dt.date, float]] = []
    for it in items:
        end = _parse_date(it.get("end", ""))
        if not end:
            continue
        if not _is_quarter_item(it):
            continue
        val = _safe_float(it.get("val"))
        if val == 0.0:
            continue
        out.append((end, val))
    # sortera på datum, senaste först
    out.sort(key=lambda x: x[0], reverse=True)
    # deduplikation per datum (ibland flera dimensioner/frames samma end)
    seen = set()
    uniq: List[Tuple[dt.date, float]] = []
    for d, v in out:
        if d not in seen:
            uniq.append((d, v))
            seen.add(d)
    return uniq

# ------- SEC API -------

@st.cache_data(ttl=3600, show_spinner=False)
def _sec_ticker_map() -> Dict[str, Dict[str, Any]]:
    """Ticker → {cik, title}. Endast US-filers finns här."""
    r = requests.get(SEC_TICKER_MAP_URL, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    data = r.json()  # {"0": {"cik_str":..., "ticker":"A", "title":"Agilent"} , ...}
    out: Dict[str, Dict[str, Any]] = {}
    for _, v in data.items():
        t = str(v.get("ticker", "")).upper()
        if not t:
            continue
        out[t] = {"cik": str(v.get("cik_str", "")).zfill(10), "title": v.get("title", "")}
    return out

@st.cache_data(ttl=3600, show_spinner=False)
def _get_company_facts(cik: str) -> dict:
    url = SEC_COMPANYFACTS_URL.format(cik=cik)
    r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    return r.json()

# ------- Publika funktioner -------

@st.cache_data(ttl=3600, show_spinner=False)
def get_pb_quarters(ticker: str) -> Dict[str, Any]:
    """
    Beräkna P/B för de senaste upp till 4 kvartalen:
      1) SEC us-gaap:Equity-tag (USD) + CommonStockSharesOutstanding (shares)
      2) BVPS = Equity / Shares
      3) Pris via yfinance nära rapportdatum
      4) PB = Price / BVPS

    Returnerar:
      {
        "pb_quarters": [(YYYY-MM-DD, pb), ...],  # senaste först, max 4
        "details": [
            {"date": "...", "equity": ..., "shares": ..., "bvps": ..., "price": ..., "pb": ...},
            ...
        ],
        "source": "sec+yf",
        "cik": "0000320193",
        "company": "Apple Inc."
      }

    Obs:
      * Fungerar i praktiken för US-bolag som rapporterar till SEC.
      * För icke-US tickers returneras tom lista.
    """
    tkr = ticker.upper().strip()
    mapping = _sec_ticker_map()
    if tkr not in mapping:
        return {"pb_quarters": [], "details": [], "source": "sec+yf", "cik": "", "company": ""}

    cik = mapping[tkr]["cik"]
    company = mapping[tkr]["title"]

    facts = _get_company_facts(cik).get("facts", {})
    if not facts:
        return {"pb_quarters": [], "details": [], "source": "sec+yf", "cik": cik, "company": company}

    eq_tag, eq_unit = _pick_equity_tag(facts)
    sh_tag, sh_unit = _pick_shares_tag(facts)
    if not eq_tag or not sh_tag:
        return {"pb_quarters": [], "details": [], "source": "sec+yf", "cik": cik, "company": company}

    eq_series = _extract_series(facts, eq_tag, eq_unit)
    sh_series = _extract_series(facts, sh_tag, sh_unit)
    if not eq_series or not sh_series:
        return {"pb_quarters": [], "details": [], "source": "sec+yf", "cik": cik, "company": company}

    # matcha per kvartalsdatum: ta shares med minsta datumdiff
    details: List[Dict[str, Any]] = []
    for d_eq, eq in eq_series[:12]:  # kolla bakåt upp till ~3 år
        # närmast shares-punkt
        near_sh = min(sh_series, key=lambda x: abs((x[0] - d_eq).days))
        sh = _safe_float(near_sh[1]) if near_sh else 0.0
        if eq <= 0 or sh <= 0:
            continue
        bvps = eq / sh  # USD
        price = _nearest_price_on_or_after(tkr, d_eq)
        if price <= 0 or bvps <= 0:
            continue
        pb = price / bvps
        details.append({
            "date": d_eq.isoformat(),
            "equity": round(eq, 2),
            "shares": round(sh, 2),
            "bvps": round(bvps, 4),
            "price": round(price, 4),
            "pb": round(pb, 2),
        })

    # de 4 senaste
    details.sort(key=lambda x: x["date"], reverse=True)
    details = details[:4]
    pb_quarters = [(d["date"], d["pb"]) for d in details]

    return {
        "pb_quarters": pb_quarters,
        "details": details,
        "source": "sec+yf",
        "cik": cik,
        "company": company,
    }
