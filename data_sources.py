# data_sources.py
from __future__ import annotations

import math
import time
import json
import typing as t
from datetime import datetime, timedelta

import pandas as pd
import requests
import streamlit as st

UA = {"User-Agent": "Mozilla/5.0 (compatible; p-s-app/1.0; +https://example.com)"}  # SEC kräver UA

# ---------------------------- Hjälpare ----------------------------

def _ts(dt: datetime | None = None) -> str:
    d = dt or datetime.utcnow()
    return d.strftime("%Y-%m-%d %H:%M")

def _safe_get(d: dict, *path, default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

@st.cache_data(show_spinner=False, ttl=3600)
def _get_json(url: str, headers: dict | None = None) -> dict:
    r = requests.get(url, headers=headers or UA, timeout=20)
    r.raise_for_status()
    return r.json()

# ---------------------------- Yahoo ----------------------------

@st.cache_data(show_spinner=False, ttl=900)
def _yahoo_quote_summary(ticker: str) -> dict:
    mods = "price,summaryDetail,defaultKeyStatistics,financialData,incomeStatementHistoryQuarterly"
    url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules={mods}"
    return _get_json(url)

def _parse_yahoo_basic(js: dict) -> dict:
    res: dict[str, t.Any] = {}
    q = _safe_get(js, "quoteSummary", "result", 0, default={})

    price = _safe_get(q, "price", "regularMarketPrice", "raw")
    currency = _safe_get(q, "price", "currency")
    longName = _safe_get(q, "price", "longName") or _safe_get(q, "price", "shortName")
    div_rate = _safe_get(q, "summaryDetail", "trailingAnnualDividendRate", "raw")  # per aktie, samma valuta som price
    ps_ttm = (
        _safe_get(q, "summaryDetail", "priceToSalesTrailing12Months", "raw")
        or _safe_get(q, "defaultKeyStatistics", "priceToSalesTrailing12Months", "raw")
    )
    shares_out = _safe_get(q, "defaultKeyStatistics", "sharesOutstanding", "raw") or _safe_get(q, "price", "sharesOutstanding", "raw")

    # Yahoo quarterly revenue
    q_is = _safe_get(q, "incomeStatementHistoryQuarterly", "incomeStatementHistory", default=[]) or []
    quarters: list[dict] = []
    for item in q_is:
        end_ts = _safe_get(item, "endDate", "raw")
        rev_raw = _safe_get(item, "totalRevenue", "raw")
        if end_ts and rev_raw:
            end = datetime.utcfromtimestamp(int(end_ts))
            quarters.append({"end": end, "revenue": float(rev_raw)})
    quarters.sort(key=lambda x: x["end"], reverse=True)

    res.update({
        "Bolagsnamn": longName or "",
        "Aktuell kurs": float(price or 0.0),
        "Valuta": currency or "",
        "Årlig utdelning": float(div_rate or 0.0),
        "yahoo_ps_ttm": float(ps_ttm or 0.0),
        "yahoo_shares": float(shares_out or 0.0),  # stycken
        "yahoo_quarterly_rev": quarters,           # i hela valutaenheter
    })
    return res

@st.cache_data(show_spinner=False, ttl=86400)
def _yahoo_last_fiscal_year_end_month(ticker: str) -> int | None:
    """För FY-mappning. Returnerar månad (1..12) då räkenskapsåret slutar, t.ex. 1 för Jan."""
    mods = "defaultKeyStatistics"
    url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules={mods}"
    js = _get_json(url)
    q = _safe_get(js, "quoteSummary", "result", 0, default={})
    last_end = _safe_get(q, "defaultKeyStatistics", "lastFiscalYearEnd", "raw")
    if not last_end:
        return None
    dt = datetime.utcfromtimestamp(int(last_end))
    return dt.month

def _fy_label_for_date(d: datetime, fy_end_month: int | None) -> tuple[str, str]:
    """Returnerar ('FY26', 'Q2') för ett givet kvartals slutdatum och ett FY-slutmånad."""
    if not fy_end_month:
        # Vanlig kalender → FY = år, kvartal = normal kvartal
        q = (d.month - 1) // 3 + 1
        return (f"FY{str(d.year)[-2:]}", f"Q{q}")
    # När slutar FY? t.ex. 1 (Jan) för NVDA
    start_month = (fy_end_month % 12) + 1  # månaden efter FY-slut
    # vilken FY tillhör datumet?
    fy_year = d.year + 1 if d.month > fy_end_month else d.year
    # hur många månader efter FY-start?
    offs = (d.month - start_month) % 12
    q = (offs // 3) + 1
    return (f"FY{str(fy_year)[-2:]}", f"Q{q}")

# ---------------------------- SEC ----------------------------

@st.cache_data(show_spinner=False, ttl=86400)
def _sec_tickers() -> dict[str, str]:
    """Mappar TICKER -> CIK (10 siffror)."""
    url = "https://www.sec.gov/files/company_tickers.json"
    js = _get_json(url, headers=UA)
    out: dict[str, str] = {}
    # Struktur: { "0": {"cik_str":320193,"ticker":"AAPL","title":"Apple Inc."}, ... }
    for _, v in js.items():
        tkr = str(v.get("ticker", "")).upper()
        cik = str(v.get("cik_str", "")).strip()
        if tkr and cik:
            out[tkr] = str(cik).zfill(10)
    return out

def _cik_for_ticker(ticker: str) -> str | None:
    try:
        return _sec_tickers().get(ticker.upper())
    except Exception:
        return None

@st.cache_data(show_spinner=False, ttl=3600)
def _sec_company_facts(cik: str) -> dict:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    return _get_json(url, headers=UA)

def _sec_shares_series(cik: str) -> list[tuple[datetime, float]]:
    """Returnerar lista [(datum, aktier_styck)] från olika möjliga taggar."""
    js = _sec_company_facts(cik)
    facts = js.get("facts", {})
    # Testa flera taggar – inte alla bolag använder samma
    candidates = [
        ("dei", "EntityCommonStockSharesOutstanding"),
        ("us-gaap", "CommonStockSharesOutstanding"),
        ("us-gaap", "EntityCommonStockSharesOutstanding"),  # ibland under us-gaap
    ]
    rows: list[tuple[datetime, float]] = []
    for ns, tag in candidates:
        u = facts.get(ns, {}).get(tag, {}).get("units", {})
        for unit_vals in u.values():
            for it in unit_vals:
                # preferera 'end' eller 'instant'
                dt_str = it.get("end") or it.get("instant")
                val = it.get("val")
                if dt_str and isinstance(val, (int, float)):
                    try:
                        dt = datetime.strptime(dt_str, "%Y-%m-%d")
                        rows.append((dt, float(val)))
                    except Exception:
                        pass
    # Unika på datum senast vinner
    rows.sort(key=lambda x: x[0])
    uniq = {}
    for d, v in rows:
        uniq[d] = v
    out = sorted(uniq.items(), key=lambda x: x[0])  # stigande datum
    return out

def _nearest_share_after(series: list[tuple[datetime, float]], ref_date: datetime, days_after: int = 1) -> float | None:
    """Tar första aktievärdet på/efter ref_date + days_after."""
    target = ref_date + timedelta(days=days_after)
    for d, v in series:
        if d >= target:
            return v
    return None

@st.cache_data(show_spinner=False, ttl=3600)
def _sec_recent_filings(cik: str, limit: int = 6) -> list[dict]:
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    js = _get_json(url, headers=UA)
    forms = js.get("filings", {}).get("recent", {})
    out = []
    for form, date, acc in zip(forms.get("form", []), forms.get("filingDate", []), forms.get("accessionNumber", [])):
        href = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}"
        viewer = f"https://www.sec.gov/ixviewer/doc?action=display&source=content&accno={acc}"
        out.append({"form": form, "date": date, "url": href, "viewer": viewer, "cik": cik})
        if len(out) >= limit:
            break
    return out

# ---------------------------- Publika funktioner ----------------------------

def hamta_live_valutakurser() -> dict:
    # enkel och robust – Yahoo "allt i USD" + manuella defaults om fel
    try:
        url = "https://query1.finance.yahoo.com/v7/finance/quote?symbols=SEK=X,EURSEK=X,NOKSEK=X,CADSEK=X,USDSEK=X"
        js = _get_json(url)
        res = { "SEK": 1.0 }
        for it in js.get("quoteResponse", {}).get("result", []):
            sym = it.get("symbol")
            raw = it.get("regularMarketPrice")
            if sym == "USDSEK=X": res["USD"] = float(raw)
            if sym == "NOKSEK=X": res["NOK"] = float(raw)
            if sym == "CADSEK=X": res["CAD"] = float(raw)
            if sym == "EURSEK=X": res["EUR"] = float(raw)
        # defensiva defaults
        for k, v in {"USD": 10.0, "NOK": 1.0, "CAD": 7.5, "EUR": 11.0}.items():
            res.setdefault(k, v)
        return res
    except Exception:
        return {"USD": 10.0, "NOK": 1.0, "CAD": 7.5, "EUR": 11.0, "SEK": 1.0}

def hamta_sec_filing_lankar(ticker: str) -> list[dict]:
    cik = _cik_for_ticker(ticker)
    if not cik:
        return []
    try:
        return _sec_recent_filings(cik)
    except Exception:
        return []

def _compute_ps_quarters(
    price: float,
    shares_default: float,   # stycken (Yahoo)
    rev_quarters: list[dict],  # [{"end": dt, "revenue": float}]
    fy_end_month: int | None,
    cik: str | None,
) -> tuple[dict, dict]:
    """
    Returnerar:
      values: {"P/S Q1":..., "P/S Q2":..., ..., "P/S Qk datum": "...", "Källa P/S Qk": "..."}
      debug:  {"qrev_pts": n, "sec_cik": cik or "", "sec_shares_pts": m, "rev_src": "...", "shares_src": "..."}
    """
    values: dict[str, t.Any] = {}
    dbg: dict[str, t.Any] = {"qrev_pts": len(rev_quarters), "sec_cik": cik or ""}

    # shares series från SEC (kan saknas)
    sec_series: list[tuple[datetime, float]] = []
    if cik:
        try:
            sec_series = _sec_shares_series(cik)
        except Exception:
            sec_series = []
    dbg["sec_shares_pts"] = len(sec_series)

    # 4 senaste kvartal
    rev_quarters = [x for x in rev_quarters if x.get("revenue", 0) > 0]
    rev_quarters.sort(key=lambda x: x["end"], reverse=True)
    rev_quarters = rev_quarters[:4]

    for idx, qd in enumerate(rev_quarters, start=1):
        end = qd["end"]
        rev = float(qd["revenue"])  # i originalvaluta
        # välj aktier (stycken) vid datumet (1 dag efter)
        shares = _nearest_share_after(sec_series, end, days_after=1)
        src = "Computed/Yahoo-revenue+SEC-shares+1d-after"
        if not shares or shares <= 0:
            shares = float(shares_default or 0.0)
            src = "Computed/Yahoo-revenue+Yahoo-shares"

        # PS = Price * Shares / Revenue
        ps_val = 0.0
        if price and shares and rev:
            ps_val = float(price) * float(shares) / float(rev)
        values[f"P/S Q{idx}"] = round(ps_val, 2)
        values[f"P/S Q{idx} datum"] = end.strftime("%Y-%m-%d")
        fy, qname = _fy_label_for_date(end, fy_end_month)
        values[f"Källa P/S Q{idx}"] = f"{src} ({fy} {qname})"

    # om färre än 4 kvartal hittades – fyll ut datum/källa tomt
    for k in range(len(rev_quarters) + 1, 5):
        values[f"P/S Q{k}"] = 0.0
        values[f"P/S Q{k} datum"] = "–"
        values[f"Källa P/S Q{k}"] = "n/a"

    # vilken källa användes mest? (för loggning)
    if any("SEC-shares" in values.get(f"Källa P/S Q{i}", "") for i in range(1, 5)):
        dbg["shares_src"] = "sec→yahoo"
    else:
        dbg["shares_src"] = "yahoo"
    dbg["rev_src"] = "yahoo"

    return values, dbg

def _compute_ps_ttm_fallback(price: float, shares: float, rev_quarters: list[dict]) -> float:
    """Om Yahoo ps_ttm saknas: beräkna P/S TTM = (price * shares)/sum(4 kvartal)."""
    last4 = [float(x["revenue"]) for x in rev_quarters[:4] if x.get("revenue")]
    if len(last4) < 1:
        return 0.0
    ttm_rev = float(sum(last4))
    if not price or not shares or not ttm_rev:
        return 0.0
    return round(price * shares / ttm_rev, 2)

# ---------------------------- Huvud: hamta_yahoo_fält ----------------------------

def hamta_yahoo_fält(ticker: str) -> dict:
    """
    Hämtar och beräknar:
      - Bolagsnamn, Aktuell kurs, Valuta, Årlig utdelning
      - Utestående aktier (miljoner) (Yahoo/info; SEC används bara i P/S-kvartal)
      - P/S (TTM) (Yahoo → fallback beräkning)
      - P/S Q1..Q4 + datum + källa med korrekt FY/Q-märkning
      - Tidsstämplar + källa-fält
    """
    out: dict[str, t.Any] = {}
    fetch_meta: dict[str, t.Any] = {}

    try:
        js = _yahoo_quote_summary(ticker)
        y = _parse_yahoo_basic(js)
    except Exception:
        y = {}

    price = float(y.get("Aktuell kurs", 0.0))
    currency = y.get("Valuta", "")
    shares_yahoo = float(y.get("yahoo_shares", 0.0))  # stycken
    shares_mn = shares_yahoo / 1e6 if shares_yahoo else 0.0

    out.update({
        "Bolagsnamn": y.get("Bolagsnamn", ""),
        "Aktuell kurs": price,
        "Valuta": currency,
        "Årlig utdelning": float(y.get("Årlig utdelning", 0.0)),
        "Utestående aktier": round(shares_mn, 2) if shares_mn else 0.0,
        "Källa Utestående aktier": "Yahoo/info",
        "TS Utestående aktier": _ts(),
    })

    # P/S TTM
    ps_ttm = float(y.get("yahoo_ps_ttm", 0.0))
    ps_source = "yahoo_ps_ttm"
    if ps_ttm <= 0.0:
        ps_ttm = _compute_ps_ttm_fallback(price, shares_yahoo, y.get("yahoo_quarterly_rev", []))
        if ps_ttm > 0:
            ps_source = "computed(yahoo_quarters+yahoo_shares)"
    out["P/S"] = ps_ttm
    out["Källa P/S"] = ps_source
    out["TS P/S"] = _ts()

    # Kvartals-P/S
    fy_end_month = _yahoo_last_fiscal_year_end_month(ticker)
    cik = _cik_for_ticker(ticker)
    qvals, dbg = _compute_ps_quarters(
        price=price,
        shares_default=shares_yahoo,
        rev_quarters=y.get("yahoo_quarterly_rev", []),
        fy_end_month=fy_end_month,
        cik=cik,
    )
    out.update(qvals)

    # Lägg i fetch-loggen för UI
    fetch_meta.update({
        "ps_source": ps_source,
        "qrev_pts": dbg.get("qrev_pts", 0),
        "price_hits": 1 if price else 0,
        "sec_cik": cik or "",
        "sec_shares_pts": dbg.get("sec_shares_pts", 0),
        "rev_src": dbg.get("rev_src", "yahoo"),
        "shares_src": dbg.get("shares_src", "yahoo"),
    })
    try:
        st.session_state.setdefault("fetch_logs", []).append({"ticker": ticker.upper(), "ts": _ts(), "ps": fetch_meta})
    except Exception:
        pass

    return out
