# stockapp/fetchers/sec.py
from __future__ import annotations

import math
import re
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd
import numpy as np
import streamlit as st

try:
    import requests
except Exception:
    requests = None  # type: ignore

# Vi använder yfinance endast för historiskt pris per datum
try:
    import yfinance as yf
except Exception:
    yf = None  # type: ignore


# ---------------- SEC helpers ----------------

SEC_UA = st.secrets.get("SEC_USER_AGENT") or "youremail@example.com"  # SEC rekommenderar identiferbar UA
SEC_TIMEOUT = 30
SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

ACCEPT_FORMS = {"10-Q", "10-K", "20-F", "40-F"}  # quarterly & annual filer typer
MAX_POINTS = 12  # hämta upp till 12 perioder, appen använder 4 (senaste) men details gynnar CAGR


def _http_get_json(url: str) -> Any:
    if requests is None:
        return None
    hdrs = {
        "User-Agent": SEC_UA,
        "Accept": "application/json",
    }
    r = requests.get(url, headers=hdrs, timeout=SEC_TIMEOUT)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=86400, show_spinner=False)  # 24h
def _ticker_to_cik_map() -> Dict[str, str]:
    """
    Hämtar hela mappingen ticker→CIK (SEC public JSON).
    Returnerar dict med UPPERCASE-ticker → 10-siffrig CIK-sträng (ledande nollor).
    """
    try:
        data = _http_get_json(SEC_TICKER_MAP_URL)
    except Exception:
        return {}

    # Formatet är { "0": { "cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc." }, ... }
    out: Dict[str, str] = {}
    if isinstance(data, dict):
        for _, v in data.items():
            try:
                t = str(v.get("ticker", "")).upper().strip()
                cik_str = str(v.get("cik_str", "")).strip()
                if t and cik_str:
                    out[t] = f"{int(cik_str):010d}"
            except Exception:
                continue
    return out


def _to_float(x) -> float:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return 0.0
        return float(x)
    except Exception:
        return 0.0


def _pick_facts_series(facts: Dict[str, Any], tag_candidates: List[str], unit_candidates: List[str]) -> List[Dict[str, Any]]:
    """
    Väljer första tagg som finns i facts['us-gaap'], sedan första unit som matchar.
    Returnerar list[ { 'end': 'YYYY-MM-DD', 'val': float, 'form': '10-Q', 'fy': 2024, 'fp': 'Q2' }, ... ]
    """
    if not facts:
        return []

    us = facts.get("facts", facts).get("us-gaap") if "facts" in facts else facts.get("us-gaap")
    if not isinstance(us, dict):
        return []

    for tag in tag_candidates:
        node = us.get(tag)
        if not node:
            continue
        units = node.get("units", {})
        # försök i prioriteringsordning
        for u in unit_candidates:
            arr = units.get(u)
            if not isinstance(arr, list):
                continue
            out = []
            for it in arr:
                # it: { "end": "2024-06-29", "val": 123, "form": "10-Q", ... }
                end = str(it.get("end") or it.get("fy"))  # end är viktigast
                form = str(it.get("form", ""))
                if form and ACCEPT_FORMS and form not in ACCEPT_FORMS:
                    continue
                val = _to_float(it.get("val"))
                if not end or val == 0.0:
                    continue
                out.append({"end": end, "val": val, "form": form, "fy": it.get("fy"), "fp": it.get("fp")})
            if out:
                # sortera kronologiskt och begränsa
                out = sorted(out, key=lambda x: x["end"])
                return out[-MAX_POINTS:]
    return []


def _align_by_date(equity: List[Dict[str, Any]], shares: List[Dict[str, Any]]) -> List[str]:
    """
    Returnerar gemensamma 'end'-datum i kronologisk ordning.
    """
    e_dates = {e["end"] for e in equity}
    s_dates = {s["end"] for s in shares}
    common = sorted(list(e_dates & s_dates))
    return common


# ---------------- Yahoo price helper ----------------

@st.cache_data(ttl=86400, show_spinner=False)  # 24h cache per datum
def _price_near_date(ticker: str, date_str: str) -> float:
    """
    Hämtar stängningspris nära 'date_str' (±3 dagar). Använder yfinance.
    """
    if yf is None or not ticker:
        return 0.0
    try:
        dt = pd.to_datetime(date_str)
        start = (dt - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
        end   = (dt + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
        t = yf.Ticker(ticker)
        h = t.history(start=start, end=end)
        if not h.empty and "Close" in h.columns:
            # välj närmast datumet
            h = h.copy()
            h["d"] = h.index
            h["dist"] = (h["d"] - dt).abs()
            row = h.sort_values("dist").iloc[0]
            return float(row["Close"])
    except Exception:
        pass
    return 0.0


# ---------------- Public API ----------------

@st.cache_data(ttl=3600, show_spinner=False)  # 1h cache
def get_pb_quarters(ticker: str) -> Dict[str, Any]:
    """
    Bygger P/B-kvartalsserie från SEC + Yahoo:
      1) Hämta equity (USD) och common shares (shares) per kvartal.
      2) Matcha på 'end'-datum.
      3) Beräkna BVPS = equity / shares.
      4) Hämta stängningspris nära 'end' via Yahoo.
      5) P/B = pris / BVPS.

    Returnerar dict med 'pb_quarters', 'details', 'source'.
    """
    out: Dict[str, Any] = {"pb_quarters": [], "details": [], "source": "SEC companyfacts + Yahoo price"}
    tkr = (ticker or "").strip().upper()
    if not tkr or requests is None:
        return out

    # 1) Ticker → CIK
    try:
        m = _ticker_to_cik_map()
        cik = m.get(tkr)
        if not cik:
            return out
    except Exception:
        return out

    # 2) Company facts
    try:
        facts = _http_get_json(SEC_FACTS_URL.format(cik=cik))
    except Exception:
        return out

    # 3) Equity-serie (USD) – flera möjliga taggar
    equity_tags = [
        "CommonStockholdersEquity",
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        "TotalEquityGrossMinorityInterest",
    ]
    equity_series = _pick_facts_series(facts, equity_tags, ["USD"])
    if not equity_series:
        return out

    # 4) Shares-serie (shares)
    shares_tags = [
        "CommonStockSharesOutstanding",
    ]
    shares_series = _pick_facts_series(facts, shares_tags, ["shares", "Shares"])
    if not shares_series:
        return out

    # 5) Gemensamma datum
    common_dates = _align_by_date(equity_series, shares_series)
    if not common_dates:
        return out

    # Gör uppslag för smidig hämtning
    eq_by_date = {e["end"]: _to_float(e["val"]) for e in equity_series}
    sh_by_date = {s["end"]: _to_float(s["val"]) for s in shares_series}

    details: List[Dict[str, Any]] = []
    pb_pairs: List[Tuple[str, float]] = []

    for d in common_dates[::-1]:  # börja från senaste
        eq = _to_float(eq_by_date.get(d))
        sh = _to_float(sh_by_date.get(d))
        if eq <= 0 or sh <= 0:
            continue
        bvps = eq / sh

        price = _price_near_date(tkr, d)
        pb = price / bvps if (price > 0 and bvps > 0) else 0.0

        details.append({
            "date": d,
            "equity": round(eq, 2),
            "shares": round(sh, 2),
            "bvps": round(bvps, 4),
            "price": round(price, 4),
            "pb": round(pb, 4),
        })

        if pb > 0:
            pb_pairs.append((d, round(pb, 2)))

        # Begränsa antal punkter vi behåller
        if len(details) >= MAX_POINTS:
            break

    # Sortera detaljer senast→äldst för konsekvent presentation
    details = sorted(details, key=lambda x: x["date"], reverse=True)
    pb_pairs = pb_pairs[:8]  # räcker mer än väl till 4Q

    out["details"] = details
    out["pb_quarters"] = pb_pairs
    return out
