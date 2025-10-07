# data_sources.py
# ---------------------------------------------
# Nät-hämtningar från Yahoo & SEC + live-valutor
# Robust mot 403/429/999 (kraschar inte appen).
# ---------------------------------------------

from __future__ import annotations
import time
import math
import json
import random
import typing as t
from datetime import datetime, timedelta, timezone

import requests
import streamlit as st

# =======================
# Gemensamma inställningar
# =======================

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept": "*/*",
}
_TIMEOUT = 12

def _log_fetch(evt: dict) -> None:
    st.session_state.setdefault("fetch_logs", []).append(evt)

# SEC kräver kontakt i UA. Försök läsa från secrets, annars fallback.
def _sec_headers() -> dict:
    ua = None
    try:
        ua = st.secrets.get("sec", {}).get("user_agent")
    except Exception:
        ua = None
    if not ua:
        ua = "kpf-app/1.0 (contact: example@example.com)"  # byt gärna i st.secrets
    return {"User-Agent": ua, "Accept": "application/json"}

# =======================
# Hjälpare för HTTP/JSON
# =======================

@st.cache_data(show_spinner=False)
def _get_json(url: str, headers: dict | None = None, ttl: int = 0) -> dict:
    """
    Hämtar JSON men returnerar {} vid fel (kraschar aldrig).
    Param 'ttl' finns kvar för kompatibilitet med ev. gamla anrop.
    """
    try:
        r = requests.get(url, headers=headers or _DEFAULT_HEADERS, timeout=_TIMEOUT)
        r.raise_for_status()
        try:
            return r.json()
        except Exception as e:
            _log_fetch({"type": "parse_error", "url": url, "err": str(e)})
            return {}
    except Exception as e:
        _log_fetch({"type": "http_error", "url": url, "err": str(e)})
        return {}

# En liten paus så vi inte spammar externa API:er
def _gentle_sleep(a: float = 0.15, b: float = 0.35) -> None:
    time.sleep(random.uniform(a, b))

# =======================
# YAHOO – Live valutakurser
# =======================

@st.cache_data(ttl=600, show_spinner=False)
def hamta_live_valutakurser() -> dict:
    """
    Hämtar USD/NOK/CAD/EUR → SEK via Yahoo Quote API (cache 10 min).
    Returnerar {} vid fel (så att UI inte skriver över sparade kurser).
    """
    pairs = {
        "USD": "USDSEK=X",
        "NOK": "NOKSEK=X",
        "CAD": "CADSEK=X",
        "EUR": "EURSEK=X",
    }
    url = "https://query1.finance.yahoo.com/v7/finance/quote?symbols=" + ",".join(pairs.values())
    js = _get_json(url)
    if not js:
        return {}
    try:
        quotes = js.get("quoteResponse", {}).get("result", [])
        px = {q.get("symbol"): q.get("regularMarketPrice") for q in quotes}
        out = {}
        for k, sym in pairs.items():
            v = px.get(sym)
            if isinstance(v, (int, float)) and v > 0:
                out[k] = float(v)
        if out:
            out["SEK"] = 1.0
        return out
    except Exception as e:
        _log_fetch({"type": "parse_error", "url": url, "err": str(e)})
        return {}

# =======================
# YAHOO – Basinfo & P/S TTM
# =======================

@st.cache_data(ttl=600, show_spinner=False)
def _yahoo_quote_summary(ticker: str) -> dict:
    mods = "price,summaryDetail,defaultKeyStatistics,assetProfile,calendarEvents"
    url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules={mods}"
    return _get_json(url)

def _yahoo_price_to_sales_ttm(qsum: dict) -> float | None:
    # Finns ibland i summaryDetail eller defaultKeyStatistics
    try:
        res = qsum["quoteSummary"]["result"][0]
    except Exception:
        return None
    # 1) summaryDetail
    try:
        val = res.get("summaryDetail", {}).get("priceToSalesTrailing12Months", {}).get("raw")
        if isinstance(val, (int, float)) and val > 0:
            return float(val)
    except Exception:
        pass
    # 2) defaultKeyStatistics
    try:
        val = res.get("defaultKeyStatistics", {}).get("priceToSalesTrailing12Months", {}).get("raw")
        if isinstance(val, (int, float)) and val > 0:
            return float(val)
    except Exception:
        pass
    return None

def _yahoo_shares_outstanding(qsum: dict) -> float | None:
    try:
        res = qsum["quoteSummary"]["result"][0]
    except Exception:
        return None
    # Prova flera ställen
    keys = [
        ("defaultKeyStatistics", "sharesOutstanding"),
        ("price", "sharesOutstanding"),
    ]
    for grp, field in keys:
        try:
            raw = res.get(grp, {}).get(field, {}).get("raw")
            if isinstance(raw, (int, float)) and raw > 0:
                return float(raw) / 1e6  # → miljoner
        except Exception:
            pass
    return None

def _yahoo_basic_fields(ticker: str) -> dict:
    qsum = _yahoo_quote_summary(ticker)
    out = {
        "Bolagsnamn": "",
        "Valuta": "",
        "Aktuell kurs": 0.0,
        "Årlig utdelning": 0.0,
        "CAGR 5 år (%)": 0.0,
        "Utestående aktier": 0.0,
        "P/S": 0.0,
        "Källa P/S": "Yahoo/ps_ttm",
        "TS P/S": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Källa Utestående aktier": "Yahoo/info",
        "TS Utestående aktier": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    try:
        res = qsum["quoteSummary"]["result"][0]
    except Exception:
        return out

    # Namn/valuta/kurs
    price = res.get("price", {})
    out["Bolagsnamn"] = (
        price.get("shortName")
        or price.get("longName")
        or price.get("quoteSourceName")
        or ""
    )
    out["Valuta"] = price.get("currency") or ""
    p = price.get("regularMarketPrice", {}).get("raw")
    if isinstance(p, (int, float)):
        out["Aktuell kurs"] = float(p)

    # Utdelning (trailing)
    div = res.get("summaryDetail", {}).get("dividendRate", {}).get("raw")
    if isinstance(div, (int, float)):
        out["Årlig utdelning"] = float(div)

    # P/S (TTM)
    ps = _yahoo_price_to_sales_ttm(qsum)
    if isinstance(ps, (int, float)) and ps > 0:
        out["P/S"] = float(ps)

    # Utestående aktier (miljoner)
    sh = _yahoo_shares_outstanding(qsum)
    if isinstance(sh, (int, float)) and sh > 0:
        out["Utestående aktier"] = float(sh)

    return out

# =======================
# SEC – CIK & länkar & fakta
# =======================

@st.cache_data(ttl=24*3600, show_spinner=False)
def _sec_ticker_map() -> dict:
    # https://www.sec.gov/files/company_tickers.json  (ca 4-6MB)
    url = "https://www.sec.gov/files/company_tickers.json"
    js = _get_json(url, headers=_sec_headers())
    # Struktur: {"0":{"cik_str":320193,"ticker":"AAPL","title":"Apple Inc."}, ...}
    out = {}
    try:
        for _, obj in js.items():
            tkr = str(obj.get("ticker", "")).upper()
            cik = str(obj.get("cik_str", "")).zfill(10)
            if tkr and cik:
                out[tkr] = cik
    except Exception:
        pass
    return out

@st.cache_data(ttl=24*3600, show_spinner=False)
def _cik_from_ticker(ticker: str) -> str | None:
    m = _sec_ticker_map()
    return m.get(ticker.upper())

@st.cache_data(ttl=3600, show_spinner=False)
def _sec_company_submissions(cik: str) -> dict:
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    return _get_json(url, headers=_sec_headers())

@st.cache_data(ttl=3600, show_spinner=False)
def _sec_company_facts(cik: str) -> dict:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    return _get_json(url, headers=_sec_headers())

def hamta_sec_filing_lankar(ticker: str) -> list[dict]:
    """
    Returnerar en lista med {form, date, url, viewer, cik}.
    """
    cik = _cik_from_ticker(ticker)
    if not cik:
        return []
    sub = _sec_company_submissions(cik)
    out = []
    try:
        recent = sub.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accs  = recent.get("accessionNumber", [])
        prim  = recent.get("primaryDocument", [])
        for form, d, acc, doc in zip(forms, dates, accs, prim):
            # Bygg arkiv-URL
            acc_nodash = acc.replace("-", "")
            base = f"/Archives/edgar/data/{int(cik)}/{acc_nodash}/{doc}"
            url = f"https://www.sec.gov{base}"
            viewer = f"https://www.sec.gov/ixviewer/doc?action=display&source=content&source_url={base}"
            out.append({
                "form": form, "date": d, "url": url, "viewer": viewer, "cik": cik
            })
    except Exception:
        return []
    return out[:12]

# =======================
# YAHOO – Kvartalsomsättning (för P/S Q1..Q4)
# =======================

@st.cache_data(ttl=3600, show_spinner=False)
def _yahoo_quarterly_revenues(ticker: str) -> list[tuple[str, float]]:
    """
    Hämtar senaste kvartalens 'totalRevenue' via quoteSummary.
    Returnerar lista [(YYYY-MM-DD, revenue_usd), ...] i fallande datumordning.
    """
    url = (
        f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/"
        f"{ticker}?modules=incomeStatementHistoryQuarterly"
    )
    js = _get_json(url)
    try:
        items = js["quoteSummary"]["result"][0]["incomeStatementHistoryQuarterly"]["incomeStatementHistory"]
    except Exception:
        return []
    out = []
    for it in items:
        end = it.get("endDate", {}).get("fmt")  # "2025-07-31"
        rev = it.get("totalRevenue", {}).get("raw")
        if isinstance(rev, (int, float)) and rev > 0 and isinstance(end, str):
            out.append((end, float(rev)))
    # Vanligen nyast först – men vi säkerställer:
    out.sort(key=lambda x: x[0], reverse=True)
    return out

# =======================
# YAHOO – Dagliga priser för datumspan
# =======================

@st.cache_data(ttl=3600, show_spinner=False)
def _yahoo_daily_close(ticker: str, start_date: str, end_date: str) -> dict[str, float]:
    """
    Returnerar {YYYY-MM-DD: close} för intervallet [start_date, end_date].
    """
    # Unix-tider (sekunder)
    sd = int(datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc).timestamp())
    ed = int(datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc).timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={sd}&period2={ed}&interval=1d"
    )
    js = _get_json(url)
    try:
        tms = js["chart"]["result"][0]["timestamp"]
        closes = js["chart"]["result"][0]["indicators"]["quote"][0]["close"]
    except Exception:
        return {}
    out = {}
    for ts, cl in zip(tms, closes):
        if cl is None:
            continue
        dt = datetime.utcfromtimestamp(ts).date().isoformat()
        out[dt] = float(cl)
    return out

# =======================
# SEC – Aktier & intäkter (kvartal)
# =======================

def _sec_quarterly_revenues_from_facts(facts: dict) -> list[tuple[str, float]]:
    """
    Plockar kvartalsintäkter från SEC CompanyFacts. Försöker flera taggar.
    Returnerar [(YYYY-MM-DD, revenue_usd)], nyast först.
    """
    tags = [
        ("Revenues", "USD"),
        ("Revenues", "USDm"),
        ("SalesRevenueNet", "USD"),
        ("SalesRevenueNet", "USDm"),
        ("RevenueFromContractWithCustomerExcludingAssessedTax", "USD"),
    ]
    out = []
    try:
        facts_d = facts.get("facts", {}).get("us-gaap", {})
        for tag, unit in tags:
            unit_obj = facts_d.get(tag, {}).get("units", {}).get(unit)
            if not unit_obj:
                continue
            for v in unit_obj:
                # Kvartalsperioder har ofta frame som "CY2025Q3I" etc, men vi kan också titta på 'fp': 'Q1'..'Q4'
                end = v.get("end")
                val = v.get("val")
                fp = v.get("fp")
                # Ta bara kvartal
                if not end or fp not in ("Q1","Q2","Q3","Q4"):
                    continue
                if isinstance(val, (int, float)) and val > 0:
                    out.append((end, float(val)))
            if out:
                break
    except Exception:
        return []
    # nyast först
    out.sort(key=lambda x: x[0], reverse=True)
    return out

def _sec_shares_series_from_facts(facts: dict) -> list[tuple[str, float]]:
    """
    Returnerar [(YYYY-MM-DD, shares)] för rapporterade antal utestående aktier (nära kvartalsdatum).
    """
    candidates = [
        ("EntityCommonStockSharesOutstanding", "shares"),
        ("CommonStockSharesOutstanding", "shares"),
    ]
    out = []
    try:
        facts_d = facts.get("facts", {}).get("dei", {})
        for tag, unit in candidates:
            arr = facts_d.get(tag, {}).get("units", {}).get(unit)
            if not arr:
                continue
            for v in arr:
                # 'end' eller 'instant' (tar det som finns)
                d = v.get("end") or v.get("instant")
                val = v.get("val")
                if d and isinstance(val, (int, float)) and val > 0:
                    out.append((d, float(val)))
    except Exception:
        return []
    out.sort(key=lambda x: x[0])
    return out

def _closest_before(series: list[tuple[str, float]], date_iso: str) -> float | None:
    """
    Väljer senaste värde i 'series' som är <= date_iso.
    """
    best = None
    for d, v in series:
        if d <= date_iso:
            best = v
    return best

# =======================
# Huvud: hämta fält för en ticker
# =======================

def hamta_yahoo_fält(ticker: str) -> dict:
    """
    Returnerar ett dict med allt appen förväntar sig:
      - Basfält (Yahoo)
      - P/S (TTM) (Yahoo)
      - P/S Q1..Q4 (försöker Yahoo->SEC mix, annars 0) + datum & käll-etiketter
      - Utestående aktier (Yahoo; fallback SEC* om du vill utöka)
    """
    out = _yahoo_basic_fields(ticker)

    # Initiera kvartals-fält (nollade)
    for q in (1, 2, 3, 4):
        out[f"P/S Q{q}"] = 0.0
        out[f"P/S Q{q} datum"] = ""
        out[f"Källa P/S Q{q}"] = "— (n/a)"

    # 1) Hämta kvartalsintäkter
    qrevs = _yahoo_quarterly_revenues(ticker)
    ps_source = "Computed/Yahoo-revenue"

    # 2) Försök komplettera med SEC-shares för bättre precision
    cik = _cik_from_ticker(ticker)
    sec_shares_ser = []
    if cik:
        facts = _sec_company_facts(cik)
        # Om Yahoo inte gav intäkter → prova SEC-intäkter
        if not qrevs:
            qrevs = _sec_quarterly_revenues_from_facts(facts)
            ps_source = "Computed/SEC-revenue"
        sec_shares_ser = _sec_shares_series_from_facts(facts)

    # 3) Hämta dagspriser kring perioderna vi vill beräkna
    #    Vi tar +1 handelsdag efter kvartalsslut (eller närmaste dag efter).
    if qrevs:
        # Vi behöver en prisserie som täcker från (min_end - 5d) till (max_end + 7d)
        ends = [datetime.fromisoformat(d) for d, _ in qrevs[:6]]
        min_d = min(ends)
        max_d = max(ends)
        start = (min_d - timedelta(days=5)).date().isoformat()
        stop  = (max_d + timedelta(days=10)).date().isoformat()
        px_map = _yahoo_daily_close(ticker, start, stop)
    else:
        px_map = {}

    # 4) Beräkna P/S för de senaste 4 kvartalen (Q1 nyast, Q4 4:e nyaste)
    for idx, (end_iso, rev) in enumerate(qrevs[:4], start=1):
        qnum = idx  # 1..4 i nyhetsordning
        # hitta pris första dag efter periodslut som finns i px_map
        d0 = datetime.fromisoformat(end_iso).date()
        price = None
        for plus in range(1, 8):
            d_try = (d0 + timedelta(days=plus)).isoformat()
            if d_try in px_map:
                price = px_map[d_try]
                break
        if price is None:
            # fallback: samma dag eller närmaste innan
            for minus in range(0, 7):
                d_try = (d0 - timedelta(days=minus)).isoformat()
                if d_try in px_map:
                    price = px_map[d_try]
                    break
        # hitta aktier (SEC-serie) senaste <= end_iso
        shares = _closest_before(sec_shares_ser, end_iso) if sec_shares_ser else None

        if (isinstance(price, (int, float)) and price > 0) and (isinstance(shares, (int, float)) and shares > 0):
            ps_val = (price * shares) / rev
            out[f"P/S Q{qnum}"] = round(float(ps_val), 2)
            out[f"P/S Q{qnum} datum"] = end_iso
            out[f"Källa P/S Q{qnum}"] = f"{ps_source}+SEC-shares+1d-after"
        elif isinstance(price, (int, float)) and price > 0 and out.get("Utestående aktier", 0) > 0:
            # fallback: Yahoo-shares (miljoner → *1e6)
            shares_y = float(out["Utestående aktier"]) * 1e6
            ps_val = (price * shares_y) / rev
            out[f"P/S Q{qnum}"] = round(float(ps_val), 2)
            out[f"P/S Q{qnum} datum"] = end_iso
            out[f"Källa P/S Q{qnum}"] = f"{ps_source}+Yahoo-shares+1d-after"
        else:
            # misslyckades – lämna 0
            out[f"P/S Q{qnum}"] = 0.0
            out[f"P/S Q{qnum} datum"] = end_iso  # vi visar ändå datum vi försökte beräkna på
            out[f"Källa P/S Q{qnum}"] = f"{ps_source}+n/a"

    # Slutlig källa-kommentar för P/S (TTM)
    if out.get("P/S", 0.0) > 0:
        out["Källa P/S"] = "Yahoo/ps_ttm"
    else:
        out["Källa P/S"] = "— (n/a)"

    # Nyttjas i UI-debug
    meta = {
        "ps_source": "yahoo_ps_ttm" if out.get("P/S", 0.0) > 0 else "n/a",
        "qrev_pts": len(qrevs),
        "sec_cik": cik or "",
        "sec_shares_pts": len(sec_shares_ser),
    }
    _log_fetch({"type": "ticker_fetch", "ticker": ticker.upper(), "ps": meta})
    _gentle_sleep()
    return out
