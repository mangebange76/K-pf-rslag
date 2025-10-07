# data_sources.py
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import requests, time, json
from datetime import datetime, timedelta
import yfinance as yf

# ---------- Små helpers ----------
def _now_iso():
    try:
        import pytz
        tz = pytz.timezone("Europe/Stockholm")
        return datetime.now(tz).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d %H:%M")

def _b(x) -> bool:
    if isinstance(x, bool): return x
    return str(x).lower() == "true"

def _sec_cfg():
    cutoff = int(st.session_state.get("SEC_CUTOFF_YEARS",
                    int(st.secrets.get("SEC_CUTOFF_YEARS", 6))))
    backfill = _b(st.session_state.get("SEC_ALLOW_BACKFILL_BEYOND_CUTOFF",
                    st.secrets.get("SEC_ALLOW_BACKFILL_BEYOND_CUTOFF","false")))
    return cutoff, backfill

def _sec_headers():
    ua = st.secrets.get("SEC_USER_AGENT") or "MyApp/1.0 admin@example.com"
    return {"User-Agent": ua, "Accept-Encoding": "gzip, deflate", "Host": "data.sec.gov"}

@st.cache_data(show_spinner=False, ttl=24*3600)
def _sec_company_tickers_index() -> dict:
    """Hämta SECs masterlista med (ticker -> CIK) en gång per dag."""
    url = "https://www.sec.gov/files/company_tickers.json"
    try:
        r = requests.get(url, headers=_sec_headers(), timeout=15)
        r.raise_for_status()
        data = r.json()
        out = {}
        for _, row in data.items():
            out[str(row["ticker"]).upper()] = str(row["cik_str"]).zfill(10)
        return out
    except Exception:
        return {}

def _cik_for_ticker(ticker: str) -> str | None:
    # 1) override i secrets
    try:
        overrides = json.loads(st.secrets.get("CIK_OVERRIDES", "{}"))
        if ticker.upper() in overrides:
            return str(overrides[ticker.upper()]).zfill(10)
    except Exception:
        pass
    # 2) SEC masterindex
    idx = _sec_company_tickers_index()
    if ticker.upper() in idx:
        return idx[ticker.upper()]
    return None

# ---------- YAHOO helpers ----------
def _yahoo_price_currency_shares(t: yf.Ticker) -> tuple[float, str, float]:
    """(price, currency, sharesOutstanding)"""
    price, curr, shares = 0.0, "USD", 0.0
    info = {}
    try:
        info = t.info or {}
    except Exception:
        info = {}
    price = info.get("regularMarketPrice")
    if price is None:
        try:
            h = t.history(period="5d")
            if not h.empty and "Close" in h:
                price = float(h["Close"].dropna().iloc[-1])
        except Exception:
            price = None
    curr = (info.get("currency") or "USD")
    shares = float(info.get("sharesOutstanding") or 0.0)
    return float(price or 0.0), str(curr), float(shares or 0.0)

def _yahoo_ps_ttm_or_marketcap(t: yf.Ticker, price: float, shares: float) -> tuple[float, str]:
    """Försök läsa P/S (TTM) från Yahoo. Fallback: marketCap / TTM-rev."""
    # 1) P/S från info
    src = "Yahoo/ps_ttm"
    ps = 0.0
    try:
        info = t.info or {}
        ps = float(info.get("priceToSalesTrailing12Months") or 0.0)
        if ps > 0:
            return ps, src
    except Exception:
        pass

    # 2) marketCap / TTM
    try:
        info = t.info or {}
        mcap = float(info.get("marketCap") or 0.0)
        # försök få TTM-revenue: summera senaste 4 kvartal från yahoo
        qrev = _yahoo_quarterly_revenue_all(t)  # dict date->rev
        ttm = 0.0
        if qrev:
            for _, v in list(sorted(qrev.items(), key=lambda x: x[0], reverse=True))[:4]:
                ttm += float(v or 0.0)
        if ttm <= 0:
            # fallback: annual totalRevenue från financials
            try:
                df_fin = getattr(t, "financials", None)
                if isinstance(df_fin, pd.DataFrame) and not df_fin.empty and "Total Revenue" in df_fin.index:
                    ser = df_fin.loc["Total Revenue"].dropna()
                    if not ser.empty:
                        ttm = float(ser.iloc[-1])
            except Exception:
                pass
        if mcap > 0 and ttm > 0:
            ps = mcap / ttm
            return float(ps), "Computed/Yahoo-marketcap/TTM"
    except Exception:
        pass
    # 3) fallback via price & sales per share om möjligt
    if price > 0 and shares > 0:
        qrev = _yahoo_quarterly_revenue_all(t)
        if qrev:
            ttm = sum(float(v or 0.0) for _, v in list(sorted(qrev.items(), key=lambda x: x[0], reverse=True))[:4])
            if ttm > 0:
                ps = (price * shares) / ttm
                return float(ps), "Computed/Yahoo-price*shares/TTM"
    return 0.0, "n/a"

def _yahoo_quarterly_revenue_all(t: yf.Ticker) -> dict[str, float]:
    """
    Samlar kvartalsintäkter från flera Yahoo-datakällor och deduplar på datum (ISO).
    Returnerar t.ex. {'2025-07-31': 30850000000.0, ...}
    """
    rev = {}
    cands = []
    for attr, row_name in [
        ("quarterly_income_stmt", "Total Revenue"),
        ("quarterly_financials",  "Total Revenue"),
    ]:
        try:
            df = getattr(t, attr, None)
            if isinstance(df, pd.DataFrame) and not df.empty and row_name in df.index:
                cands.append(df.loc[row_name].dropna())
        except Exception:
            pass
    # ibland finns "Total Revenue" i income_stmt med kolumner som kvartal
    try:
        df_is = getattr(t, "income_stmt", None)
        if isinstance(df_is, pd.DataFrame) and not df_is.empty and "Total Revenue" in df_is.index:
            cands.append(df_is.loc["Total Revenue"].dropna().tail(8))
    except Exception:
        pass

    for s in cands:
        for dt, val in s.items():
            try:
                d = pd.to_datetime(str(dt)).date().isoformat()
                rev[d] = float(val)
            except Exception:
                continue
    return rev

# ---------- SEC helpers ----------
@st.cache_data(show_spinner=False, ttl=3600)
def _sec_companyfacts(cik: str) -> dict:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{str(cik).zfill(10)}.json"
    r = requests.get(url, headers=_sec_headers(), timeout=20)
    r.raise_for_status()
    return r.json()

@st.cache_data(show_spinner=False, ttl=3600)
def _sec_submissions(cik: str) -> dict:
    url = f"https://data.sec.gov/submissions/CIK{str(cik).zfill(10)}.json"
    r = requests.get(url, headers=_sec_headers(), timeout=20)
    r.raise_for_status()
    return r.json()

def _sec_extract_series(facts: dict, tags: list[str], unit_keys: list[str]) -> list[dict]:
    """Returnerar lista av punkter [{'date': 'YYYY-MM-DD', 'val': float}] för första träffande tag + unit."""
    if not facts or "facts" not in facts:
        return []
    usgaap = facts.get("facts", {}).get("us-gaap", {})
    for tag in tags:
        if tag in usgaap:
            units = usgaap[tag].get("units", {})
            for uk in unit_keys:
                if uk in units:
                    out = []
                    for p in units[uk]:
                        d = p.get("end") or p.get("frame") or p.get("fy")
                        try:
                            d_iso = pd.to_datetime(str(d)).date().isoformat()
                            out.append({"date": d_iso, "val": float(p.get("val") or 0.0)})
                        except Exception:
                            continue
                    return out
    return []

def _nearest_value(points: list[dict], ref_date: datetime, days=60) -> float:
    """Returnera värde vars datum är närmast ref_date inom 'days' dagar."""
    if not points:
        return 0.0
    best = None
    for p in points:
        try:
            d = datetime.fromisoformat(p["date"])
            delta = abs((d - ref_date).days)
            if delta <= days:
                if best is None or delta < best[0]:
                    best = (delta, float(p["val"]))
        except Exception:
            continue
    return best[1] if best else 0.0

def _compute_quarter_ps(t: yf.Ticker, dates_vals: list[tuple[str, float]], shares_series: list[dict]) -> tuple[list[tuple[str,float,str]], int]:
    """
    För varje (datum, revenue) beräkna P/S = price_(d+1) * shares_(nära d) / revenue_d.
    Returnerar [(date_iso, ps_val, src_label), ...], samt hur många pris-träffar vi lyckades få.
    """
    hits = 0
    rows = []
    for d_iso, revenue in dates_vals:
        # pris: använd d + 1 handelsdag
        try:
            d0 = pd.to_datetime(d_iso) + pd.Timedelta(days=1)
            d1 = d0 + pd.Timedelta(days=6)
            h = t.history(start=d0.strftime("%Y-%m-%d"), end=d1.strftime("%Y-%m-%d"))
            px = float(h["Close"].dropna().iloc[0]) if not h.empty else 0.0
        except Exception:
            px = 0.0
        if px > 0: hits += 1

        # shares: närmast datumet (±60d), från SEC shares-serien
        sh = _nearest_value(shares_series, pd.to_datetime(d_iso), days=60)
        if sh <= 0:
            # fallback: Yahoo sharesOutstanding
            try:
                _, _, yshares = _yahoo_price_currency_shares(t)
                sh = yshares
            except Exception:
                sh = 0.0

        ps = 0.0
        if px > 0 and sh > 0 and revenue > 0:
            ps = (px * float(sh)) / float(revenue)
        rows.append((d_iso, float(ps), "Computed/Yahoo-revenue+SEC-shares+1d-after"))
    return rows, hits

# ---------- Offentliga funktioner ----------
def hamta_live_valutakurser() -> dict:
    """Hämta USD/NOK/CAD/EUR -> SEK via Yahoo."""
    out = {"USD": 9.75, "NOK": 0.95, "CAD": 7.05, "EUR": 11.18, "SEK": 1.0}
    pairs = {"USD": "USDSEK=X", "NOK": "NOKSEK=X", "CAD": "CADSEK=X", "EUR": "EURSEK=X"}
    for k, y in pairs.items():
        try:
            t = yf.Ticker(y)
            h = t.history(period="5d")
            if not h.empty:
                out[k] = round(float(h["Close"].dropna().iloc[-1]), 4)
        except Exception:
            pass
    return out

def hamta_sec_filing_lankar(ticker: str) -> list[dict]:
    """Senaste 8-K/10-Q/10-K med iXBRL-länk, arkiv-länk och CIK."""
    cik = _cik_for_ticker(ticker)
    if not cik:
        return []
    try:
        sub = _sec_submissions(cik)
        forms = []
        for f, acc, date, primary in zip(
            sub.get("filings", {}).get("recent", {}).get("form", []),
            sub.get("filings", {}).get("recent", {}).get("accessionNumber", []),
            sub.get("filings", {}).get("recent", {}).get("filingDate", []),
            sub.get("filings", {}).get("recent", {}).get("primaryDocument", []),
        ):
            if f in ("10-Q","10-K","8-K"):
                acc_clean = str(acc).replace("-", "")
                base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}"
                viewer = f"https://www.sec.gov/ixviewer/doc?action=display&source={base}/{primary}"
                forms.append({
                    "form": f, "date": date, "viewer": viewer, "url": f"{base}/{primary}", "cik": cik
                })
        return forms[:10]
    except Exception:
        return []

def beräkna_cagr_från_finansiella(tkr: yf.Ticker) -> float:
    try:
        df_is = getattr(tkr, "income_stmt", None)
        series = None
        if isinstance(df_is, pd.DataFrame) and not df_is.empty and "Total Revenue" in df_is.index:
            series = df_is.loc["Total Revenue"].dropna()
        else:
            df_fin = getattr(tkr, "financials", None)
            if isinstance(df_fin, pd.DataFrame) and not df_fin.empty and "Total Revenue" in df_fin.index:
                series = df_fin.loc["Total Revenue"].dropna()
        if series is None or series.empty or len(series) < 2:
            return 0.0
        series = series.sort_index()
        start = float(series.iloc[0]); end = float(series.iloc[-1]); years = max(1, len(series)-1)
        if start <= 0: return 0.0
        cagr = (end / start) ** (1.0 / years) - 1.0
        return round(cagr * 100.0, 2)
    except Exception:
        return 0.0

def hamta_yahoo_fält(ticker: str) -> dict:
    """
    Hämtar och beräknar:
      - Bolagsnamn, Aktuell kurs, Valuta, Årlig utdelning, CAGR 5 år (%)
      - Utestående aktier (miljoner) (Yahoo/SEC fallback)
      - P/S (TTM)
      - P/S Q1...Q4 (+ datum + källrader)
    Lägger en loggrad i st.session_state["fetch_logs"].
    """
    out = {
        "Bolagsnamn": "", "Aktuell kurs": 0.0, "Valuta": "USD", "Årlig utdelning": 0.0,
        "CAGR 5 år (%)": 0.0, "Utestående aktier": 0.0,
        "Källa Aktuell kurs": "", "Källa Utestående aktier": "", "Källa P/S": "",
        "Källa P/S Q1": "", "Källa P/S Q2": "", "Källa P/S Q3": "", "Källa P/S Q4": "",
        "P/S Q1 datum": "", "P/S Q2 datum": "", "P/S Q3 datum": "", "P/S Q4 datum": "",
    }
    logs = {"ticker": ticker.upper(), "ts": _now_iso(), "ps": {}}

    t = yf.Ticker(ticker)

    # --- Grundfält från Yahoo ---
    try:
        info = t.info or {}
    except Exception:
        info = {}

    # pris/valuta/shares
    price, curr, y_shares = _yahoo_price_currency_shares(t)
    out["Aktuell kurs"] = float(price or 0.0)
    out["Valuta"] = (curr or "USD").upper()
    out["Källa Aktuell kurs"] = "Yahoo/info"

    name = info.get("shortName") or info.get("longName") or ""
    out["Bolagsnamn"] = str(name)
    out["Årlig utdelning"] = float(info.get("dividendRate") or 0.0)
    out["CAGR 5 år (%)"] = beräkna_cagr_från_finansiella(t)

    # --- SEC fakta (shares & revenue) ---
    cik = _cik_for_ticker(ticker)
    cutoff_years, allow_backfill = _sec_cfg()

    sec_shares = []
    sec_rev = []
    if cik:
        try:
            facts = _sec_companyfacts(cik)
            sec_shares = _sec_extract_series(
                facts,
                tags=["CommonStockSharesOutstanding", "EntityCommonStockSharesOutstanding"],
                unit_keys=["shares"]
            )
            sec_rev = _sec_extract_series(
                facts,
                tags=["RevenueFromContractWithCustomerExcludingAssessedTax","SalesRevenueNet","Revenues"],
                unit_keys=["USD"]
            )
        except Exception:
            sec_shares, sec_rev = [], []

    # Utestående aktier: prioritera SEC senaste, annars Yahoo
    try:
        if sec_shares:
            last = sorted(sec_shares, key=lambda x: x["date"])[-1]
            out["Utestående aktier"] = round(float(last["val"]) / 1e6, 2)  # miljoner
            out["Källa Utestående aktier"] = "SEC/companyfacts"
        elif y_shares > 0:
            out["Utestående aktier"] = round(float(y_shares) / 1e6, 2)
            out["Källa Utestående aktier"] = "Yahoo/info"
    except Exception:
        pass

    # --- P/S (TTM) ---
    ps_ttm, ps_src = _yahoo_ps_ttm_or_marketcap(t, price, y_shares)
    out["P/S"] = float(round(ps_ttm, 2) if ps_ttm else 0.0)
    out["Källa P/S"] = ps_src

    # --- Bygg kvartals-P/S (Q1..Q4) ---
    # Kandidat-revenue: Yahoo + SEC, sortera senaste först
    yrev = _yahoo_quarterly_revenue_all(t)          # {'YYYY-MM-DD': val}
    all_rev = {}
    all_rev.update(sec_rev and {p["date"]: p["val"] for p in sec_rev} or {})
    all_rev.update(yrev)  # Yahoo får sista ordet för senaste
    # filter på cutoff: endast de senaste cutoff_years åren
    if all_rev:
        cutoff_date = datetime.now() - timedelta(days=365*cutoff_years)
        rev_items = [(d, v) for d, v in all_rev.items() if datetime.fromisoformat(d) >= cutoff_date]
        rev_items_sorted = sorted(rev_items, key=lambda x: x[0], reverse=True)
    else:
        rev_items_sorted = []

    # om färre än 4 efter cutoff → backfill med äldre kvartal (om tillåtet)
    rev_after_cutoff = rev_items_sorted.copy()
    if len(rev_after_cutoff) < 4 and allow_backfill and all_rev:
        older = [(d, v) for d, v in all_rev.items() if d not in dict(rev_after_cutoff)]
        older = sorted(older, key=lambda x: x[0], reverse=True)
        need = 4 - len(rev_after_cutoff)
        rev_after_cutoff += older[:max(0, need)]

    q_dates_vals = rev_after_cutoff[:4]  # ta senaste 4
    ps_rows, price_hits = _compute_quarter_ps(t, q_dates_vals, sec_shares)

    # Map till Q1..Q4
    # Q1 = senaste kvartal, Q4 = fjärde senaste
    for idx, (d_iso, ps_val, label) in enumerate(ps_rows):
        qcol = f"P/S Q{idx+1}"
        out[qcol] = float(round(ps_val, 2) if ps_val else 0.0)
        out[f"{qcol} datum"] = d_iso
        # märk om datapunkt låg före cutoff (backfillad)
        backfill_note = ""
        try:
            cut = datetime.now() - timedelta(days=365*cutoff_years)
            if datetime.fromisoformat(d_iso) < cut:
                backfill_note = " [pre-cutoff]"
        except Exception:
            pass
        out[f"Källa P/S Q{idx+1}"] = label + backfill_note

    # Fyll 0 på ev. tomma Q-kolumner
    for q in (1,2,3,4):
        out.setdefault(f"P/S Q{q}", 0.0)
        out.setdefault(f"P/S Q{q} datum", "")
        out.setdefault(f"Källa P/S Q{q}", "")

    # --- Logg ---
    logs["ps"] = {
        "ps_source": ps_src,
        "q_cols": len(q_dates_vals),
        "price_hits": price_hits,
        "sec_cik": cik,
        "sec_ixbrl_pts": 0,                   # (vi använder companyfacts här)
        "sec_shares_pts": len(sec_shares),
        "sec_rev_pts": len(sec_rev),
        "cutoff_years": int(cutoff_years),
        "backfill_used": bool(allow_backfill)
    }
    logs["summary"] = f"Yahoo price/currency/shares, SEC facts shares+revenue, Q1..Q4={len(q_dates_vals)}"
    st.session_state.setdefault("fetch_logs", []).append(logs)

    return out
