# data_sources.py
from __future__ import annotations

import re
import math
import time
import json
import requests
import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf

# -----------------------------
# Allmänna helpers
# -----------------------------

def _secret_bool(name: str, default: bool = False) -> bool:
    try:
        v = st.secrets.get(name, default)
        if isinstance(v, bool): return v
        if isinstance(v, (int, float)): return bool(v)
        if isinstance(v, str): return v.strip().lower() in {"1","true","yes","y"}
    except Exception:
        pass
    return default

def _naive_index(df: pd.DataFrame) -> pd.DataFrame:
    try:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = df.index.tz_localize(None)
        return df
    except Exception:
        return df

def _to_naive_ts(col) -> pd.Timestamp | None:
    try:
        ts = pd.to_datetime(col).tz_localize(None)
        return ts
    except Exception:
        return None

def _clean_iso(ts: pd.Timestamp | str) -> str:
    try:
        if isinstance(ts, str):
            ts = pd.to_datetime(ts).tz_localize(None)
        return ts.strftime("%Y-%m-%d")
    except Exception:
        return ""

# -----------------------------
# Valutor (till SEK)
# -----------------------------

def hamta_valutakurser_live() -> dict:
    """
    Hämtar USD, EUR, NOK, CAD → SEK via exchangerate.host.
    Returnerar {"USD": x, "EUR": y, ...}. Vid fel returneras {}.
    """
    try:
        # Hämta SEK-baserat och invertera för USD→SEK etc
        url = "https://api.exchangerate.host/latest?base=SEK&symbols=USD,EUR,NOK,CAD"
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        j = r.json()
        rates = j.get("rates", {})
        out = {}
        for code in ["USD", "EUR", "NOK", "CAD"]:
            v = rates.get(code)
            if v and v > 0:
                out[code] = 1.0 / float(v)  # SEK-baserad → invertera
        out["SEK"] = 1.0
        return out
    except Exception:
        return {}

# -----------------------------
# Yahoo Finance
# -----------------------------

def _try_get_quarterly_revenue_df(tkr: yf.Ticker) -> pd.DataFrame | None:
    """
    Hämtar kvartals-IS via nya fälten i yfinance (quarterly_financials / qtrly income_stmt).
    Returnerar DataFrame där index innehåller "Total Revenue" om tillgängligt.
    """
    for attr in ["quarterly_financials", "quarterly_income_stmt", "quarterly_income_statement", "income_stmt"]:
        try:
            df = getattr(tkr, attr, None)
            if isinstance(df, pd.DataFrame) and not df.empty and ("Total Revenue" in df.index):
                return df
        except Exception:
            continue
    # sista chans via .get_financials(freq="q")
    try:
        df = tkr.get_financials(freq="q")
        if isinstance(df, pd.DataFrame) and not df.empty and "Total Revenue" in df.index:
            return df
    except Exception:
        pass
    return None

def _nearest_price_on_or_after(tkr: yf.Ticker, end_ts: pd.Timestamp, lookahead_days: int = 5):
    """
    Hämtar närmaste stängningskurs på eller efter end_ts (± lookahead).
    """
    try:
        start = (end_ts - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        stop  = (end_ts + pd.Timedelta(days=lookahead_days)).strftime("%Y-%m-%d")
        h = tkr.history(start=start, end=stop, auto_adjust=False)
        h = _naive_index(h)
        if not h.empty and "Close" in h.columns:
            px = float(h["Close"].iloc[-1])
            return px, "shares+1d-after"
    except Exception:
        pass
    # fallback: regularMarketPrice
    try:
        info = tkr.info or {}
        px = info.get("regularMarketPrice")
        if px:
            return float(px), "regular"
    except Exception:
        pass
    return None, "n/a"

def berakna_cagr_från_finansiella(tkr: yf.Ticker) -> float:
    """
    CAGR ~ ‘Total Revenue’ årligen (om möjligt).
    """
    try:
        df_is = getattr(tkr, "income_stmt", None)
        if isinstance(df_is, pd.DataFrame) and not df_is.empty and "Total Revenue" in df_is.index:
            series = df_is.loc["Total Revenue"].dropna()
        else:
            df_fin = getattr(tkr, "financials", None)
            if isinstance(df_fin, pd.DataFrame) and not df_fin.empty and "Total Revenue" in df_fin.index:
                series = df_fin.loc["Total Revenue"].dropna()
            else:
                return 0.0
        if series.empty or len(series) < 2:
            return 0.0
        series = series.sort_index()
        start = float(series.iloc[0])
        end   = float(series.iloc[-1])
        years = max(1, len(series)-1)
        if start <= 0:
            return 0.0
        cagr = (end / start) ** (1.0/years) - 1.0
        return round(cagr * 100.0, 2)
    except Exception:
        return 0.0

def hamta_yahoo_falt(ticker: str) -> dict:
    """
    Returnerar:
      Bolagsnamn, Aktuell kurs, Valuta, Årlig utdelning, CAGR 5 år (%),
      Utestående aktier (miljoner) + källa.
    """
    out = {
        "Bolagsnamn": "",
        "Aktuell kurs": 0.0,
        "Valuta": "USD",
        "Årlig utdelning": 0.0,
        "CAGR 5 år (%)": 0.0,
        "Utestående aktier": 0.0,
        "Källa Utestående": "",
        "Källa Kurs": "",
    }
    try:
        t = yf.Ticker(ticker)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}

        pris = info.get("regularMarketPrice")
        if pris is None:
            h = t.history(period="1d")
            if not h.empty and "Close" in h:
                pris = float(h["Close"].iloc[-1])
                out["Källa Kurs"] = "Yahoo/history"
        if pris is not None:
            out["Aktuell kurs"] = float(pris)
            if not out["Källa Kurs"]:
                out["Källa Kurs"] = "Yahoo/info"

        valuta = info.get("currency")
        if valuta:
            out["Valuta"] = str(valuta).upper()

        namn = info.get("shortName") or info.get("longName") or ""
        if namn:
            out["Bolagsnamn"] = str(namn)

        div_rate = info.get("dividendRate")
        if div_rate is not None:
            out["Årlig utdelning"] = float(div_rate)

        shares = info.get("sharesOutstanding")
        if shares:
            out["Utestående aktier"] = round(float(shares) / 1_000_000.0, 2)  # i miljoner
            out["Källa Utestående"] = "Yahoo/info"

        out["CAGR 5 år (%)"] = berakna_cagr_från_finansiella(t)

    except Exception:
        pass
    return out

# -----------------------------
# SEC helpers
# -----------------------------

def _sec_headers():
    ua = st.secrets.get("SEC_USER_AGENT", "StockApp/1.0 contact@example.com")
    return {"User-Agent": ua, "Accept-Encoding": "gzip, deflate", "Host": "data.sec.gov"}

@st.cache_data(show_spinner=False, ttl=3600)
def _ticker_to_cik_any(ticker: str) -> str | None:
    """
    Provar SEC:s full submissions-index (via /submissions/CIKxxxxx) genom att
    söka baklänges på ticker i /companyfacts om direkt mappning saknas.
    """
    try:
        # direktsök: SEC har inte officiell ticker→CIK, men yfinance kan ha CIK
        t = yf.Ticker(ticker)
        info = t.info or {}
        cik = info.get("cik")
        if cik:
            return str(cik).zfill(10)
    except Exception:
        pass
    # fallback via SEC search API (inofficiell/alternativ tjänst finns ej stabil)
    # lämna None om vi inte hittar
    return None

@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_companyfacts(cik: str) -> dict | None:
    try:
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{str(int(cik)).zfill(10)}.json"
        r = requests.get(url, headers=_sec_headers(), timeout=30)
        if r.ok:
            return r.json()
    except Exception:
        return None
    return None

def _extract_shares_history(facts_json: dict) -> dict:
    """
    Returnerar {date_iso: shares_float} från WeightedAverage... eller CommonStockSharesOutstanding.
    """
    out = {}
    try:
        facts = facts_json.get("facts", {}).get("us-gaap", {})
        candidates = [
            "WeightedAverageNumberOfDilutedSharesOutstanding",
            "WeightedAverageNumberOfSharesOutstandingBasic",
            "CommonStockSharesOutstanding",
        ]
        best = None
        for c in candidates:
            if c in facts:
                best = facts[c]; break
        if not best: 
            return out
        units = best.get("units", {})
        first_key = next(iter(units.keys())) if units else None
        arr = units.get(first_key, [])
        for item in arr:
            d = item.get("end") or item.get("fy") or item.get("filed")
            v = item.get("val")
            if d and v:
                try:
                    iso = _clean_iso(d)
                    out[iso] = float(v)
                except Exception:
                    continue
        return out
    except Exception:
        return out

def _nearest_shares_for_date(date_iso: str, shares_map: dict) -> float | None:
    if not shares_map: 
        return None
    try:
        target = pd.to_datetime(date_iso)
        best_key = None
        best_dt = None
        for k in shares_map.keys():
            dt = pd.to_datetime(k)
            if dt <= target and (best_dt is None or dt > best_dt):
                best_dt = dt; best_key = k
        if best_key:
            return float(shares_map[best_key])
    except Exception:
        return None
    return None

def _extract_sec_quarter_revenues(facts_json: dict) -> list[tuple[pd.Timestamp, float]]:
    """
    Tar kvartalsintäkter ur companyfacts (duration ~ kvartal).
    """
    out = []
    try:
        facts = facts_json.get("facts", {}).get("us-gaap", {})
        tags = [
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues",
            "SalesRevenueNet",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
            "SalesRevenueGoodsNet",
        ]
        series = None
        for t in tags:
            if t in facts:
                series = facts[t]; break
        if not series:
            return []
        units = series.get("units", {})
        for uvals in units.values():
            for item in uvals:
                s = item.get("start"); e = item.get("end")
                v = item.get("val")
                if not (s and e and v is not None):
                    continue
                try:
                    ts_s = pd.to_datetime(s).tz_localize(None)
                    ts_e = pd.to_datetime(e).tz_localize(None)
                    days = (ts_e - ts_s).days
                    if 80 <= days <= 100:
                        out.append((ts_e, float(v)))
                except Exception:
                    continue
        # dedupe per end date (behåll största beloppet)
        ded = {}
        for ts, v in out:
            if (ts not in ded) or (abs(v) > abs(ded[ts])):
                ded[ts] = v
        return sorted(ded.items(), key=lambda x: x[0], reverse=True)
    except Exception:
        return []

# -----------------------------
# SEC iXBRL (direkt ur filings)
# -----------------------------

def hamta_sec_filing_lankar(ticker: str, forms=("10-Q","10-K","8-K"), limit: int = 6) -> list[dict]:
    """Senaste filing-länkar för ticker → [{form,date,url,viewer,cik,accession}]"""
    cik = _ticker_to_cik_any(ticker)
    if not cik:
        return []
    try:
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        r = requests.get(url, headers=_sec_headers(), timeout=20)
        r.raise_for_status()
        j = r.json()
        recent = (j.get("filings", {}) or {}).get("recent", {})
        forms_list = list(recent.get("form", []))
        dates      = list(recent.get("filingDate", []))
        acc        = list(recent.get("accessionNumber", []))
        prim       = list(recent.get("primaryDocument", []))
        out = []
        for i, f in enumerate(forms_list):
            if f not in forms:
                continue
            an = str(acc[i]); primdoc = str(prim[i])
            archive = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{an.replace('-','')}/{primdoc}"
            viewer  = f"https://www.sec.gov/ixviewer/doc?action=display&source={archive}"
            out.append({"form": f, "date": str(dates[i]), "url": archive, "viewer": viewer, "cik": cik, "accession": an})
            if len(out) >= limit:
                break
        return out
    except Exception:
        return []

def _parse_ixbrl_contexts(html: str) -> dict:
    ctx = {}
    for m in re.finditer(r'<xbrli:context[^>]*id="([^"]+)"[^>]*>(.*?)</xbrli:context>', html, flags=re.I|re.S):
        cid, block = m.group(1), m.group(2)
        inst = re.search(r'<xbrli:instant>([^<]+)</xbrli:instant>', block, flags=re.I)
        if inst:
            try:
                end = pd.to_datetime(inst.group(1)).tz_localize(None)
                ctx[cid] = (None, end, False)
            except Exception:
                pass
            continue
        start = re.search(r'<xbrli:startDate>([^<]+)</xbrli:startDate>', block, flags=re.I)
        end   = re.search(r'<xbrli:endDate>([^<]+)</xbrli:endDate>', block, flags=re.I)
        if start and end:
            try:
                s = pd.to_datetime(start.group(1)).tz_localize(None)
                e = pd.to_datetime(end.group(1)).tz_localize(None)
                ctx[cid] = (s, e, True)
            except Exception:
                pass
    return ctx

def _parse_ixbrl_revenues(html: str) -> list[tuple[pd.Timestamp, float]]:
    """
    Returnerar [(end_ts, revenue)] för kvartal genom att läsa ix:nonFraction.
    """
    tags = [
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
        "us-gaap:Revenues",
        "us-gaap:SalesRevenueNet",
        "us-gaap:RevenueFromContractWithCustomerIncludingAssessedTax",
        "us-gaap:SalesRevenueGoodsNet",
    ]
    name_re = r'(' + "|".join(re.escape(t) for t in tags) + r')'
    contexts = _parse_ixbrl_contexts(html)
    out = []
    for m in re.finditer(
        rf'<ix:nonFraction[^>]*name="(?P<name>{name_re})"[^>]*contextRef="(?P<ctx>[^"]+)"[^>]*([^>]*?scale="(?P<scale>-?\d+)")?[^>]*?(?:value="(?P<val_attr>[^"]+)")?[^>]*>(?P<val_text>.*?)</ix:nonFraction>',
        html, flags=re.I|re.S
    ):
        ctx_id = m.group("ctx")
        if ctx_id not in contexts:
            continue
        start, end, is_dur = contexts[ctx_id]
        if not is_dur or end is None:
            continue
        # kvartalslängd ~ 80–100 dagar
        if start is None or not (80 <= (end - start).days <= 100):
            continue
        scale = int(m.group("scale") or 0)
        raw = (m.group("val_attr") or m.group("val_text") or "").strip()
        raw = re.sub(r'<[^>]+>', '', raw)
        neg = raw.startswith("(") and raw.endswith(")")
        raw = raw.strip("()").replace("\xa0", "").replace(",", "")
        try:
            val = float(raw)
            if neg: val = -val
            if scale: val *= (10 ** scale)
        except Exception:
            continue
        out.append((end, float(val)))

    if not out:
        return []
    # dedupe per end date – behåll högsta beloppet
    ded = {}
    for ts, v in out:
        if (ts not in ded) or (abs(v) > abs(ded[ts])):
            ded[ts] = v
    return sorted(ded.items(), key=lambda x: x[0], reverse=True)

def hamta_sec_ixbrl_quarter_revenues(ticker: str, max_docs: int = 6) -> list[tuple[pd.Timestamp, float]]:
    links = hamta_sec_filing_lankar(ticker, forms=("10-Q","10-K","8-K"), limit=max_docs)
    pairs = []
    for L in links:
        try:
            r = requests.get(L["url"], headers=_sec_headers(), timeout=25)
            if not r.ok:
                continue
            pairs.extend(_parse_ixbrl_revenues(r.text))
        except Exception:
            continue
    if not pairs:
        return []
    # dedupe nyast->äldst
    ded = {}
    for ts, v in pairs:
        if (ts not in ded) or (abs(v) > abs(ded[ts])):
            ded[ts] = v
    return sorted(ded.items(), key=lambda x: x[0], reverse=True)

# -----------------------------
# P/S-motorn (Yahoo → iXBRL → companyfacts → fallback)
# -----------------------------

def hamta_ps_kvartal(ticker: str) -> dict:
    """
    Returnerar P/S TTM samt P/S för senaste 4 kvartalen, med datum + källa per fält.
    Följande prioritet:
      1) Yahoo kvartalsintäkter
      2) SEC iXBRL (direkt ur filing)
      3) SEC companyfacts
      4) Fallback: MarketCap / TTM
    Respekterar SEC_CUTOFF_YEARS och SEC_ALLOW_BACKFILL_BEYOND_CUTOFF.
    """
    out = {
        "P/S": 0.0,
        "P/S Q1": 0.0, "P/S Q2": 0.0, "P/S Q3": 0.0, "P/S Q4": 0.0,
        "P/S Q1 datum":"", "P/S Q2 datum":"", "P/S Q3 datum":"", "P/S Q4 datum":"",
        "Källa P/S":"", "Källa P/S Q1":"", "Källa P/S Q2":"", "Källa P/S Q3":"", "Källa P/S Q4":"",
        "_DEBUG_PS": {}
    }
    try:
        try:
            SEC_CUTOFF_YEARS = int(st.secrets.get("SEC_CUTOFF_YEARS", 6))
        except Exception:
            SEC_CUTOFF_YEARS = 6
        ALLOW_BACKFILL = _secret_bool("SEC_ALLOW_BACKFILL_BEYOND_CUTOFF", False)

        cutoff_ts = pd.Timestamp.today().tz_localize(None) - pd.DateOffset(years=SEC_CUTOFF_YEARS)
        dbg = {"ticker": ticker, "ps_source": "-", "q_cols": 0, "ttm_points": 0, "price_hits": 0,
               "
