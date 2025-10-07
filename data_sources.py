# data_sources.py
from __future__ import annotations
import time
import random
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Dict, Optional

import requests
import streamlit as st

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

def _now_ts() -> float:
    return time.time()

def _circuit_open() -> bool:
    if st.session_state.get("offline_mode", False):
        return True
    until = st.session_state.get("net_down_until", 0.0)
    return _now_ts() < float(until)

def _trip_circuit(sec: int = 900) -> None:
    st.session_state["net_down_until"] = _now_ts() + sec
    st.session_state["net_fail_count"] = 0

def _note_fail() -> None:
    n = int(st.session_state.get("net_fail_count", 0)) + 1
    st.session_state["net_fail_count"] = n
    if n >= 3:
        _trip_circuit(600)  # 10 min

def _reset_fail() -> None:
    st.session_state["net_fail_count"] = 0

def _sec_headers() -> dict:
    try:
        ua = st.secrets.get("sec", {}).get("user_agent")
    except Exception:
        ua = None
    if not ua:
        ua = "kpf-app/1.0 (contact: your-email@example.com)"
    return {"User-Agent": ua, "Accept": "application/json"}

@st.cache_data(show_spinner=False)
def _get_json(url: str, headers: dict | None = None, ttl: int = 0) -> dict:
    if _circuit_open():
        _log_fetch({"type": "net_paused", "url": url})
        return {}
    try:
        r = requests.get(url, headers=headers or _DEFAULT_HEADERS, timeout=_TIMEOUT)
        r.raise_for_status()
        js = r.json()
        _reset_fail()
        return js
    except Exception as e:
        _note_fail()
        _log_fetch({"type": "http_error", "url": url, "err": str(e)})
        return {}

def _gentle_sleep(a: float = 0.12, b: float = 0.28) -> None:
    time.sleep(random.uniform(a, b))

# ---------- Valutakurser ----------

def _fx_from_yahoo() -> dict:
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
        _log_fetch({"type": "parse_error", "provider": "yahoo_fx", "err": str(e)})
        return {}

def _fx_from_exchangerate_host() -> dict:
    url = "https://api.exchangerate.host/latest?base=SEK&symbols=USD,NOK,CAD,EUR"
    js = _get_json(url)
    if not js:
        return {}
    try:
        rates = js.get("rates", {})
        out = {}
        for k in ["USD", "NOK", "CAD", "EUR"]:
            v = rates.get(k)
            if isinstance(v, (int, float)) and v > 0:
                out[k] = float(1.0 / v)  # invertera SEK-basen
        if out:
            out["SEK"] = 1.0
        return out
    except Exception as e:
        _log_fetch({"type": "parse_error", "provider": "exchangerate_host", "err": str(e)})
        return {}

@st.cache_data(ttl=600, show_spinner=False)
def hamta_live_valutakurser() -> dict:
    fx = _fx_from_yahoo()
    if fx:
        return fx
    fx = _fx_from_exchangerate_host()
    if fx:
        return fx
    return {}

# ---------- Yahoo basics ----------

@st.cache_data(ttl=600, show_spinner=False)
def _yahoo_quote_summary(ticker: str) -> dict:
    mods = "price,summaryDetail,defaultKeyStatistics,assetProfile,calendarEvents"
    url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules={mods}"
    return _get_json(url)

def _yahoo_price_to_sales_ttm(qsum: dict) -> Optional[float]:
    try:
        res = qsum["quoteSummary"]["result"][0]
    except Exception:
        return None
    try:
        val = res.get("summaryDetail", {}).get("priceToSalesTrailing12Months", {}).get("raw")
        if isinstance(val, (int, float)) and val > 0:
            return float(val)
    except Exception:
        pass
    try:
        val = res.get("defaultKeyStatistics", {}).get("priceToSalesTrailing12Months", {}).get("raw")
        if isinstance(val, (int, float)) and val > 0:
            return float(val)
    except Exception:
        pass
    return None

def _yahoo_shares_outstanding(qsum: dict) -> Optional[float]:
    try:
        res = qsum["quoteSummary"]["result"][0]
    except Exception:
        return None
    candidates = [
        ("defaultKeyStatistics", "sharesOutstanding"),
        ("price", "sharesOutstanding"),
    ]
    for grp, field in candidates:
        try:
            raw = res.get(grp, {}).get(field, {}).get("raw")
            if isinstance(raw, (int, float)) and raw > 0:
                return float(raw) / 1e6  # till MILJONER
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

    price = res.get("price", {})
    out["Bolagsnamn"] = price.get("shortName") or price.get("longName") or ""
    out["Valuta"] = price.get("currency") or ""
    p = price.get("regularMarketPrice", {}).get("raw")
    if isinstance(p, (int, float)): out["Aktuell kurs"] = float(p)

    div = res.get("summaryDetail", {}).get("dividendRate", {}).get("raw")
    if isinstance(div, (int, float)): out["Årlig utdelning"] = float(div)

    ps = _yahoo_price_to_sales_ttm(qsum)
    if isinstance(ps, (int, float)) and ps > 0: out["P/S"] = float(ps)

    sh_mn = _yahoo_shares_outstanding(qsum)
    if isinstance(sh_mn, (int, float)) and sh_mn > 0: out["Utestående aktier"] = float(sh_mn)

    return out

# ---------- Yahoo quarterly revenues (for TTM) ----------

@st.cache_data(ttl=3600, show_spinner=False)
def _yahoo_quarterly_revenues(ticker: str) -> List[Tuple[str, float]]:
    url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules=incomeStatementHistoryQuarterly"
    js = _get_json(url)
    try:
        items = js["quoteSummary"]["result"][0]["incomeStatementHistoryQuarterly"]["incomeStatementHistory"]
    except Exception:
        return []
    out = []
    for it in items:
        end = it.get("endDate", {}).get("fmt")
        rev = it.get("totalRevenue", {}).get("raw")
        if isinstance(rev, (int, float)) and rev > 0 and isinstance(end, str):
            out.append((end, float(rev)))
    out.sort(key=lambda x: x[0], reverse=True)
    return out

# ---------- Yahoo daily closes ----------

@st.cache_data(ttl=3600, show_spinner=False)
def _yahoo_daily_close(ticker: str, start_date: str, end_date: str) -> Dict[str, float]:
    sd = int(datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc).timestamp())
    ed = int(datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc).timestamp())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1={sd}&period2={ed}&interval=1d"
    js = _get_json(url)
    try:
        tms = js["chart"]["result"][0]["timestamp"]
        closes = js["chart"]["result"][0]["indicators"]["quote"][0]["close"]
    except Exception:
        return {}
    out = {}
    for ts, cl in zip(tms, closes):
        if cl is None: continue
        dt = datetime.utcfromtimestamp(ts).date().isoformat()
        out[dt] = float(cl)
    return out

# ---------- SEC helpers ----------

@st.cache_data(ttl=24*3600, show_spinner=False)
def _sec_ticker_map() -> dict:
    url = "https://www.sec.gov/files/company_tickers.json"
    js = _get_json(url, headers=_sec_headers())
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
def _cik_from_ticker(ticker: str) -> Optional[str]:
    return _sec_ticker_map().get(ticker.upper())

@st.cache_data(ttl=3600, show_spinner=False)
def _sec_company_facts(cik: str) -> dict:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    return _get_json(url, headers=_sec_headers())

@st.cache_data(ttl=3600, show_spinner=False)
def _sec_company_submissions(cik: str) -> dict:
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    return _get_json(url, headers=_sec_headers())

def hamta_sec_filing_lankar(ticker: str) -> List[dict]:
    cik = _cik_from_ticker(ticker)
    if not cik: return []
    sub = _sec_company_submissions(cik)
    out = []
    try:
        recent = sub.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accs  = recent.get("accessionNumber", [])
        prim  = recent.get("primaryDocument", [])
        for form, d, acc, doc in zip(forms, dates, accs, prim):
            acc_nodash = acc.replace("-", "")
            base = f"/Archives/edgar/data/{int(cik)}/{acc_nodash}/{doc}"
            url = f"https://www.sec.gov{base}"
            viewer = f"https://www.sec.gov/ixviewer/doc?action=display&source=content&source_url={base}"
            out.append({"form": form, "date": d, "url": url, "viewer": viewer, "cik": cik})
    except Exception:
        return []
    return out[:12]

def _sec_quarterly_revenues_from_facts(facts: dict) -> List[Tuple[str, float]]:
    tags = [
        ("Revenues", "USD"),
        ("SalesRevenueNet", "USD"),
        ("RevenueFromContractWithCustomerExcludingAssessedTax", "USD"),
    ]
    out = []
    try:
        usgaap = facts.get("facts", {}).get("us-gaap", {})
        for tag, unit in tags:
            arr = usgaap.get(tag, {}).get("units", {}).get(unit)
            if not arr: continue
            for v in arr:
                fp = v.get("fp")
                end = v.get("end")
                val = v.get("val")
                if fp in ("Q1","Q2","Q3","Q4") and isinstance(val, (int, float)) and end:
                    out.append((end, float(val)))
            if out: break
    except Exception:
        return []
    out.sort(key=lambda x: x[0], reverse=True)
    return out

def _sec_shares_series_from_facts(facts: dict) -> List[Tuple[str, float]]:
    candidates = [
        ("EntityCommonStockSharesOutstanding", "shares"),
        ("CommonStockSharesOutstanding", "shares"),
    ]
    out = []
    try:
        dei = facts.get("facts", {}).get("dei", {})
        for tag, unit in candidates:
            arr = dei.get(tag, {}).get("units", {}).get(unit)
            if not arr: continue
            for v in arr:
                d = v.get("end") or v.get("instant")
                val = v.get("val")
                if d and isinstance(val, (int, float)) and val > 0:
                    out.append((d, float(val)))
    except Exception:
        return []
    out.sort(key=lambda x: x[0])
    return out

def _closest_before(series: List[Tuple[str, float]], date_iso: str) -> Optional[float]:
    best = None
    for d, v in series:
        if d <= date_iso:
            best = v
    return best

def _ttm_series_from_quarterlies(qrevs: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    out = []
    for i in range(0, max(0, len(qrevs) - 3)):
        ttm = qrevs[i][1] + qrevs[i+1][1] + qrevs[i+2][1] + qrevs[i+3][1]
        out.append((qrevs[i][0], float(ttm)))
    return out

def hamta_yahoo_fält(ticker: str) -> dict:
    out = _yahoo_basic_fields(ticker)
    for q in (1,2,3,4):
        out[f"P/S Q{q}"] = 0.0
        out[f"P/S Q{q} datum"] = ""
        out[f"Källa P/S Q{q}"] = "— (n/a)"

    qrevs = _yahoo_quarterly_revenues(ticker)
    ps_source = "TTM/Yahoo-revenue"
    cik = _cik_from_ticker(ticker)
    sec_shares_ser: List[Tuple[str, float]] = []
    if cik:
        facts = _sec_company_facts(cik)
        sec_shares_ser = _sec_shares_series_from_facts(facts)
        if not qrevs:
            qrevs = _sec_quarterly_revenues_from_facts(facts)
            ps_source = "TTM/SEC-revenue"

    if not qrevs or len(qrevs) < 4:
        _log_fetch({"type": "ps_quarters_skipped", "ticker": ticker.upper(), "reason": "too_few_quarters"})
        _gentle_sleep()
        return out

    ttm_list = _ttm_series_from_quarterlies(qrevs)
    ttm_slice = ttm_list[:4]
    ends = [datetime.fromisoformat(d) for d, _ in ttm_slice]
    start = (min(ends) - timedelta(days=7)).date().isoformat()
    stop  = (max(ends) + timedelta(days=14)).date().isoformat()

    px_map = _yahoo_daily_close(ticker, start, stop)

    for idx, (end_iso, ttm_rev) in enumerate(ttm_slice, start=1):
        d0 = datetime.fromisoformat(end_iso).date()
        price = None
        for plus in range(1, 8):
            d_try = (d0 + timedelta(days=plus)).isoformat()
            if d_try in px_map:
                price = px_map[d_try]
                break
        if price is None:
            for minus in range(0, 8):
                d_try = (d0 - timedelta(days=minus)).isoformat()
                if d_try in px_map:
                    price = px_map[d_try]
                    break

        shares = _closest_before(sec_shares_ser, end_iso) if sec_shares_ser else None
        if shares is None:
            y_mn = float(out.get("Utestående aktier", 0.0) or 0.0)
            shares = y_mn * 1e6 if y_mn > 0 else None

        if isinstance(price, (int, float)) and price > 0 and isinstance(shares, (int, float)) and shares > 0:
            market_cap = price * shares
            ps_val = market_cap / float(ttm_rev)
            out[f"P/S Q{idx}"] = round(float(ps_val), 2)
            out[f"P/S Q{idx} datum"] = end_iso
            source_detail = "SEC-shares" if sec_shares_ser else "Yahoo-shares"
            out[f"Källa P/S Q{idx}"] = f"{ps_source}+{source_detail}+D+1"
        else:
            out[f"P/S Q{idx}"] = 0.0
            out[f"P/S Q{idx} datum"] = end_iso
            out[f"Källa P/S Q{idx}"] = f"{ps_source}+incomplete"

    meta = {
        "ps_source": "yahoo_ps_ttm" if out.get("P/S", 0.0) > 0 else "n/a",
        "qrev_pts": len(qrevs),
        "sec_cik": cik or "",
        "sec_shares_pts": len(sec_shares_ser),
    }
    _log_fetch({"type": "ticker_fetch", "ticker": ticker.upper(), "ps": meta})
    _gentle_sleep()
    return out
