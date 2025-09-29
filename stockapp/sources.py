# stockapp/sources.py
# -*- coding: utf-8 -*-
"""
Datakällor och single-ticker uppdaterare:
- Yahoo, SEC, FMP (light), (valfritt Finnhub för estimat)
- P/S (TTM) nu + historiska P/S Q1..Q4 (TTM-fönster)
- Robust Q4 via 10-K minus Q1..Q3 om 3-mån kvartal saknas
- run_update_price_only(...) & run_update_full(...)
"""

from __future__ import annotations

from typing import Dict, Tuple, List, Optional
from datetime import datetime as _dt, timedelta as _td, date

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import streamlit as st

from .compute import (
    apply_auto_updates_to_row,
    uppdatera_berakningar,
)

# ------------------------------------------------------------
# Konstanter / Secrets
# ------------------------------------------------------------
FMP_BASE = st.secrets.get("FMP_BASE", "https://financialmodelingprep.com")
FMP_KEY  = st.secrets.get("FMP_API_KEY", "")
FMP_CALL_DELAY = float(st.secrets.get("FMP_CALL_DELAY", 1.0))      # försiktig default
FMP_BLOCK_MINUTES = float(st.secrets.get("FMP_BLOCK_MINUTES", 20)) # paus efter 429

SEC_USER_AGENT = st.secrets.get(
    "SEC_USER_AGENT",
    "StockApp/1.0 (contact: your-email@example.com)"
)

# ------------------------------------------------------------
# Små hjälpare
# ------------------------------------------------------------
def _safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def _parse_iso(d: str) -> Optional[date]:
    try:
        return _dt.fromisoformat(d.replace("Z", "+00:00")).date()
    except Exception:
        try:
            return _dt.strptime(d, "%Y-%m-%d").date()
        except Exception:
            return None

# ------------------------------------------------------------
# Yahoo-hjälpare
# ------------------------------------------------------------
def _yfi_get(tkr: yf.Ticker, *keys):
    """Safe get från yfinance.info med fallback."""
    try:
        info = tkr.info or {}
        for k in keys:
            if k in info and info[k] is not None:
                return info[k]
    except Exception:
        pass
    return None

def _yfi_info_dict(t: yf.Ticker) -> dict:
    try:
        return t.info or {}
    except Exception:
        return {}

def _yahoo_prices_for_dates(ticker: str, dates: List[date]) -> Dict[date, float]:
    """
    Hämtar dagliga priser i fönster som täcker 'dates' och returnerar Close på/närmast FÖRE respektive datum.
    """
    if not dates:
        return {}
    dmin = min(dates) - _td(days=14)
    dmax = max(dates) + _td(days=3)
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
            for j in range(len(idx)-1, -1, -1):
                if idx[j] <= d:
                    try: px = float(closes[j])
                    except: px = None
                    break
            if px is not None:
                out[d] = px
        return out
    except Exception:
        return {}

def berakna_cagr_fran_finansiella(tkr: yf.Ticker) -> float:
    """
    Enkel CAGR ~ baserat på Total Revenue (årsvärden). Approx om data saknas.
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
        start = float(series.iloc[0]); end = float(series.iloc[-1])
        years = max(1, len(series)-1)
        if start <= 0:
            return 0.0
        cagr = (end / start) ** (1.0/years) - 1.0
        return round(cagr * 100.0, 2)
    except Exception:
        return 0.0

def hamta_yahoo_falt(ticker: str) -> dict:
    """
    Basfält från Yahoo: Bolagsnamn, Kurs, Valuta, Årlig utdelning, CAGR 5 år (%).
    """
    out = {
        "Bolagsnamn": "",
        "Aktuell kurs": 0.0,
        "Valuta": "USD",
        "Årlig utdelning": 0.0,
        "CAGR 5 år (%)": 0.0,
    }
    try:
        t = yf.Ticker(ticker)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}

        pris = info.get("regularMarketPrice", None)
        if pris is None:
            h = t.history(period="1d")
            if not h.empty and "Close" in h:
                pris = float(h["Close"].iloc[-1])
        if pris is not None:
            out["Aktuell kurs"] = float(pris)

        valuta = info.get("currency", None)
        if valuta:
            out["Valuta"] = str(valuta).upper()

        namn = info.get("shortName") or info.get("longName") or ""
        if namn:
            out["Bolagsnamn"] = str(namn)

        div_rate = info.get("dividendRate", None)
        if div_rate is not None:
            try:
                out["Årlig utdelning"] = float(div_rate)
            except Exception:
                pass

        out["CAGR 5 år (%)"] = berakna_cagr_fran_finansiella(t)
    except Exception:
        pass
    return out

# ------------------------------------------------------------
# FMP (light)
# ------------------------------------------------------------
def _fmp_get(path: str, params=None) -> Tuple[Optional[dict|list], int]:
    url = f"{FMP_BASE}/{path}"
    params = (params or {}).copy()
    if FMP_KEY:
        params["apikey"] = FMP_KEY
    try:
        if FMP_CALL_DELAY > 0:
            import time as _time
            _time.sleep(FMP_CALL_DELAY)
        r = requests.get(url, params=params, timeout=20)
        sc = r.status_code
        try:
            j = r.json()
        except Exception:
            j = None
        # circuit breaker vid 429?
        if sc == 429:
            # ingen stateful block här (hålls i appen om önskat)
            pass
        return j, sc
    except Exception:
        return None, 0

def hamta_fmp_falt_light(yahoo_ticker: str) -> dict:
    """
    Lätt variant: quote (pris/mcap/shares) + ratios-ttm (P/S).
    """
    out = {"_debug": {}, "_symbol": str(yahoo_ticker).strip().upper()}
    sym = out["_symbol"]

    # pris, marketCap, sharesOutstanding
    q, sc_q = _fmp_get(f"api/v3/quote/{sym}")
    out["_debug"]["quote_sc"] = sc_q
    if isinstance(q, list) and q:
        q0 = q[0]
        if "price" in q0:
            try: out["Aktuell kurs"] = float(q0["price"])
            except: pass
        if q0.get("marketCap") is not None:
            try: out["_marketCap"] = float(q0["marketCap"])
            except: pass
        if q0.get("sharesOutstanding") is not None:
            try: out["Utestående aktier"] = float(q0["sharesOutstanding"]) / 1e6
            except: pass

    # P/S TTM
    rttm, sc_rttm = _fmp_get(f"api/v3/ratios-ttm/{sym}")
    out["_debug"]["ratios_ttm_sc"] = sc_rttm
    if isinstance(rttm, list) and rttm:
        v = rttm[0].get("priceToSalesTTM") or rttm[0].get("priceToSalesRatioTTM")
        try:
            if v and float(v) > 0:
                out["P/S"] = float(v)
                out["_debug"]["ps_source"] = "ratios-ttm"
        except Exception:
            pass

    return out

# ------------------------------------------------------------
# SEC
# ------------------------------------------------------------
def _sec_get(url: str, params=None):
    try:
        r = requests.get(url, params=params or {}, headers={"User-Agent": SEC_USER_AGENT}, timeout=30)
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
    out: Dict[str, str] = {}
    # {"0":{"cik_str":320193,"ticker":"AAPL","title":"Apple Inc."}, ...}
    for _, v in j.items():
        try:
            out[str(v["ticker"]).upper()] = str(v["cik_str"]).zfill(10)
        except Exception:
            pass
    return out

def _sec_cik_for(ticker: str) -> Optional[str]:
    return _sec_ticker_map().get(str(ticker).upper())

def _sec_companyfacts(cik10: str):
    return _sec_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json")

def _is_instant_entry(it: dict) -> bool:
    end = it.get("end"); start = it.get("start")
    if not end:
        return False
    if not start:
        return True  # instant
    d1 = _parse_iso(str(start)); d2 = _parse_iso(str(end))
    if d1 and d2:
        try:
            return (d2 - d1).days <= 2
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
                            form = (it.get("form") or "").upper()
                            entries.append({"end": end, "val": v, "form": form})
                        except Exception:
                            pass
    return entries

def _sec_latest_shares_robust(facts: dict) -> float:
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

def _fx_rate_cached(base: str, quote: str) -> float:
    """
    Mycket enkel FX (dagens) via Frankfurter → exchangerate.host fallback.
    """
    base = (base or "").upper(); quote = (quote or "").upper()
    if not base or not quote or base == quote:
        return 1.0
    try:
        r = requests.get("https://api.frankfurter.app/latest", params={"from": base, "to": quote}, timeout=12)
        if r.status_code == 200:
            return float((r.json() or {}).get("rates", {}).get(quote, 0.0) or 0.0)
    except Exception:
        pass
    try:
        r = requests.get("https://api.exchangerate.host/latest", params={"base": base, "symbols": quote}, timeout=12)
        if r.status_code == 200:
            return float((r.json() or {}).get("rates", {}).get(quote, 0.0) or 0.0)
    except Exception:
        pass
    return 0.0

def _sec_quarterly_revenues_dated_with_unit(facts: dict, max_quarters: int = 8):
    """
    Hämtar 3-mån kvartalsintäkter (us-gaap) för 10-Q/10-K.
    Om ett Q saknas (t.ex. dec/jan) försöker vi deriviera Q4 = (År 10-K) - (Q1+Q2+Q3).
    Returnerar (rows, unit) där rows=[(end_date, value), ...] nyast→äldst.
    """
    prefer_units = ("USD", "CAD", "EUR", "GBP")

    gaap = (facts.get("facts") or {}).get("us-gaap", {})
    # Kandidat-koncept för revenue
    rev_keys = [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "Revenues",
        "Revenue",
        "RevenueFromContractsWithCustomers",
        "RevenueFromContractsWithCustomersExcludingSalesTaxes",
    ]

    # 1) Läs 3-mån kvartal direkt
    def read_quarters_3m() -> Tuple[List[Tuple[date, float, str]], Optional[str]]:
        for key in rev_keys:
            fact = gaap.get(key)
            if not fact:
                continue
            units = fact.get("units") or {}
            for unit_code in prefer_units:
                arr = units.get(unit_code)
                if not isinstance(arr, list):
                    continue
                tmp = []
                for it in arr:
                    form = (it.get("form") or "").upper()
                    # Vi accepterar både 10-Q och 10-K (vissa Q4 kan vara i 10-K som 3m)
                    if form not in ("10-Q", "10-Q/A", "10-K", "10-K/A"):
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
                    # 3-mån ≈ 70..100 dagar
                    if dur is None or dur < 70 or dur > 100:
                        continue
                    try:
                        v = float(val)
                        tmp.append((end, v, unit_code))
                    except Exception:
                        pass
                if not tmp:
                    continue
                # Deduplicera per end-date
                best: Dict[date, float] = {}
                for end, v, _u in tmp:
                    best[end] = v
                rows = sorted(best.items(), key=lambda t: t[0], reverse=True)
                # returnera nyast → äldst med unit
                return [(d, float(v), unit_code) for (d, v) in rows[:max_quarters]], unit_code
        return [], None

    q3m, unit = read_quarters_3m()

    # 2) Om vi saknar ett Q (typ Q4), försök deriviera från 10-K (årsdata)
    #    Q4 = Annual(Year) - (Q1+Q2+Q3)
    if unit and len(q3m) >= 3:
        # Bygg karta år→Q1..Q3 som finns
        by_year: Dict[int, List[Tuple[date, float]]] = {}
        for d, v, _u in q3m:
            by_year.setdefault(d.year, []).append((d, v))
        # Läs årsdata (10-K) för samma koncept
        annual_vals: Dict[int, float] = {}
        for key in rev_keys:
            fact = gaap.get(key)
            if not fact:
                continue
            units = fact.get("units") or {}
            arr = units.get(unit)
            if not isinstance(arr, list):
                continue
            for it in arr:
                form = (it.get("form") or "").upper()
                end = _parse_iso(str(it.get("end", "")))
                start = _parse_iso(str(it.get("start", "")))
                val = it.get("val", None)
                if not (end and start and val is not None):
                    continue
                try:
                    dur = (end - start).days
                except Exception:
                    dur = None
                # 1 år ≈ 350..380 dagar
                if dur is None or dur < 320 or dur > 400:
                    continue
                if form not in ("10-K", "10-K/A"):
                    continue
                try:
                    v = float(val)
                    annual_vals[end.year] = v
                except Exception:
                    pass
            # hittade vi någon annual? stanna vid första koncept som gav något
            if annual_vals:
                break

        # Om året har annual och Q1..Q3 men Q4 saknas, lägg till Q4
        # Identifiera Q1..Q3 genom att plocka 3 kvartal tidigast det året
        if annual_vals:
            # gör en set över befintliga end-datum i q3m
            existing_dates = set(d for (d, _v, _u) in q3m)
            add_rows: List[Tuple[date, float, str]] = []
            for yr, qlist in by_year.items():
                if yr not in annual_vals:
                    continue
                # sortera äldst->nyast för det året
                q_sorted = sorted(qlist, key=lambda t: t[0])
                # ta de tidigaste 3 (troligen Q1..Q3)
                if len(q_sorted) >= 3:
                    first_three = q_sorted[:3]
                    s = sum(v for (_d, v) in first_three)
                    q4_val = annual_vals[yr] - s
                    # försök gissa Q4 end-date: ta nästa kvartals slut 3 månader efter tredje
                    # eller använd års slutdatum  (yr-12-xx). Vi approximerar som sista kända + ~90 dagar.
                    guess_end = first_three[-1][0] + _td(days=90)
                    # lägg inte in dublett
                    if guess_end not in existing_dates and q4_val > 0:
                        add_rows.append((guess_end, float(q4_val), unit))
            if add_rows:
                # slå ihop och sortera
                all_rows = q3m + add_rows
                all_rows.sort(key=lambda t: t[0], reverse=True)
                q3m = all_rows[:max_quarters]

    # 3) Returnera som [(end_date, value), ...] och unit
    rows = [(d, v) for (d, v, _u) in q3m]
    return rows, unit

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

# ------------------------------------------------------------
# SEC + Yahoo combo / Yahoo fallback
# ------------------------------------------------------------
def hamta_sec_yahoo_combo(ticker: str) -> dict:
    """
    US/FPIs: Shares + kvartalsintäkter från SEC (3m 10-Q, Q4 deriv från 10-K),
    pris/valuta/namn från Yahoo. P/S (TTM) nu + P/S Q1–Q4 historik.
    Om CIK saknas → hamta_yahoo_global_combo.
    """
    out: Dict[str, float|str] = {}
    cik = _sec_cik_for(ticker)
    if not cik:
        return hamta_yahoo_global_combo(ticker)

    facts, sc = _sec_companyfacts(cik)
    if sc != 200 or not isinstance(facts, dict):
        return hamta_yahoo_global_combo(ticker)

    # Yahoo-basics
    y = hamta_yahoo_falt(ticker)
    for k in ("Bolagsnamn", "Valuta", "Aktuell kurs", "Årlig utdelning", "CAGR 5 år (%)"):
        if y.get(k) not in (None, "", 0, 0.0):
            out[k] = y[k]
    px_ccy = (out.get("Valuta") or "USD").upper()

    # Shares: Yahoo implied → fallback SEC robust
    implied = _implied_shares_from_yahoo(ticker, price=out.get("Aktuell kurs"), mcap=None)
    sec_shares = _sec_latest_shares_robust(facts)
    shares_used = 0.0
    if implied and implied > 0:
        shares_used = float(implied)
        out["_debug_shares_source"] = "Yahoo implied (mcap/price)"
    elif sec_shares and sec_shares > 0:
        shares_used = float(sec_shares)
        out["_debug_shares_source"] = "SEC instant (robust)"
    else:
        out["_debug_shares_source"] = "unknown"

    if shares_used > 0:
        out["Utestående aktier"] = shares_used / 1e6  # miljoner

    # Market cap (nu)
    mcap_now = _yfi_get(yf.Ticker(ticker), "market_cap", "marketCap")
    try:
        mcap_now = float(mcap_now or 0.0)
    except Exception:
        mcap_now = 0.0
    if mcap_now <= 0 and out.get("Aktuell kurs", 0) > 0 and shares_used > 0:
        mcap_now = float(out["Aktuell kurs"]) * shares_used

    # SEC kvartalsintäkter + unit → TTM & konvertering
    q_rows, rev_unit = _sec_quarterly_revenues_dated_with_unit(facts, max_quarters=8)
    if not q_rows or not rev_unit:
        return out

    conv = 1.0
    if rev_unit.upper() != px_ccy:
        c = _fx_rate_cached(rev_unit.upper(), px_ccy)
        if c > 0:
            conv = c
    ttm_list = _ttm_windows(q_rows, need=4)
    ttm_list_px = [(d, v * conv) for (d, v) in ttm_list]

    # P/S (TTM) nu
    if mcap_now > 0 and ttm_list_px:
        ltm_now = ttm_list_px[0][1]
        if ltm_now > 0:
            out["P/S"] = mcap_now / ltm_now

    # P/S Q1–Q4 historik
    if shares_used > 0 and ttm_list_px:
        q_dates = [d for (d, _) in ttm_list_px]
        px_map = _yahoo_prices_for_dates(ticker, q_dates)
        for idx, (d_end, ttm_rev_px) in enumerate(ttm_list_px[:4], start=1):
            if ttm_rev_px and ttm_rev_px > 0:
                px = px_map.get(d_end, None)
                if px and px > 0:
                    mcap_hist = shares_used * float(px)
                    out[f"P/S Q{idx}"] = float(mcap_hist / ttm_rev_px)

    # Spara även MCAP_hist om du vill visa i investeringsförslag
    if shares_used > 0 and ttm_list_px:
        q_dates = [d for (d, _) in ttm_list_px[:4]]
        px_map = _yahoo_prices_for_dates(ticker, q_dates)
        for idx, d_end in enumerate(q_dates, start=1):
            pxx = _safe_float(px_map.get(d_end, 0.0))
            if pxx > 0:
                out[f"MCAP Q{idx}"] = shares_used * pxx
    if mcap_now > 0:
        out["MCAP nu"] = mcap_now

    return out

def _implied_shares_from_yahoo(ticker: str, price: float = None, mcap: float = None) -> float:
    t = yf.Ticker(ticker)
    if mcap is None:
        mcap = _yfi_get(t, "market_cap", "marketCap")
    if price is None:
        price = _yfi_get(t, "last_price", "regularMarketPrice")
    try:
        mcap = float(mcap or 0.0); price = float(price or 0.0)
    except Exception:
        return 0.0
    if mcap > 0 and price > 0:
        return mcap / price
    return 0.0

def _yfi_quarterly_revenues(t: yf.Ticker) -> List[Tuple[date, float]]:
    """
    Försöker läsa kvartalsintäkter från Yahoo.
    Returnerar [(period_end_date, value), ...] sorterat nyast→äldst.
    """
    # 1) quarterly_financials
    try:
        qf = t.quarterly_financials
        if isinstance(qf, pd.DataFrame) and not qf.empty:
            idx = [str(x).strip() for x in qf.index]
            cand_rows = [
                "Total Revenue","TotalRevenue","Revenues","Revenue","Sales",
                "Total revenue","Revenues from contracts with customers"
            ]
            for key in cand_rows:
                if key in idx:
                    row = qf.loc[key].dropna()
                    out = []
                    for c, v in row.items():
                        try:
                            d = c.date() if hasattr(c, "date") else pd.to_datetime(c).date()
                            out.append((d, float(v)))
                        except Exception:
                            pass
                    out.sort(key=lambda x: x[0], reverse=True)
                    return out
    except Exception:
        pass

    # 2) fallback: income_stmt quarterly via v1-api (kan vara tomt)
    try:
        df_is = getattr(t, "income_stmt", None)
        if isinstance(df_is, pd.DataFrame) and not df_is.empty and "Total Revenue" in df_is.index:
            ser = df_is.loc["Total Revenue"].dropna()
            out = []
            for c, v in ser.items():
                try:
                    d = c.date() if hasattr(c, "date") else pd.to_datetime(c).date()
                    out.append((d, float(v)))
                except Exception:
                    pass
            out.sort(key=lambda x: x[0], reverse=True)
            return out
    except Exception:
        pass

    return []

def hamta_yahoo_global_combo(ticker: str) -> dict:
    """
    Global fallback för tickers utan SEC (.TO/.V/.CN + EU/Norden m.fl.).
    Räknar implied shares, P/S (TTM) nu, samt P/S Q1–Q4 historik.
    """
    out: Dict[str, float|str] = {}
    t = yf.Ticker(ticker)

    # Bas: namn/valuta/price/utdelning/CAGR
    y = hamta_yahoo_falt(ticker)
    for k in ("Bolagsnamn","Valuta","Aktuell kurs","Årlig utdelning","CAGR 5 år (%)"):
        if y.get(k) not in (None, "", 0, 0.0):
            out[k] = y[k]
    px = float(out.get("Aktuell kurs") or 0.0)
    px_ccy = (out.get("Valuta") or "USD").upper()

    info = _yfi_info_dict(t)
    mcap = info.get("marketCap")
    try:
        mcap = float(mcap) if mcap is not None else 0.0
    except Exception:
        mcap = 0.0

    # Implied shares → fallback sharesOutstanding
    shares = 0.0
    if mcap > 0 and px > 0:
        shares = mcap / px
        out["_debug_shares_source"] = "Yahoo implied (mcap/price)"
    else:
        so = info.get("sharesOutstanding")
        try:
            so = float(so or 0.0)
        except Exception:
            so = 0.0
        if so > 0:
            shares = so
            out["_debug_shares_source"] = "Yahoo sharesOutstanding"

    if shares > 0:
        out["Utestående aktier"] = shares / 1e6  # miljoner

    # Kvartalsintäkter → TTM
    q_rows = _yfi_quarterly_revenues(t)
    if not q_rows or len(q_rows) < 4:
        if mcap > 0:
            out["MCAP nu"] = mcap
        return out
    ttm_list = _ttm_windows(q_rows, need=4)

    # Valutakonvertering om financialCurrency != prisvaluta
    fin_ccy = str(info.get("financialCurrency") or px_ccy).upper()
    conv = 1.0
    if fin_ccy != px_ccy:
        c = _fx_rate_cached(fin_ccy, px_ccy)
        if c > 0:
            conv = c
    ttm_list_px = [(d, v * conv) for (d, v) in ttm_list]

    # Market cap (nu)
    if mcap <= 0 and shares > 0 and px > 0:
        mcap = shares * px

    # P/S (TTM) nu
    if mcap > 0 and ttm_list_px:
        ltm_now = ttm_list_px[0][1]
        if ltm_now > 0:
            out["P/S"] = mcap / ltm_now

    # P/S Q1–Q4 (historisk)
    if shares > 0 and ttm_list_px:
        q_dates = [d for (d, _) in ttm_list_px]
        px_map = _yahoo_prices_for_dates(ticker, q_dates)
        for idx, (d_end, ttm_rev_px) in enumerate(ttm_list_px[:4], start=1):
            if ttm_rev_px and ttm_rev_px > 0:
                p = px_map.get(d_end)
                if p and p > 0:
                    out[f"P/S Q{idx}"] = (shares * p) / ttm_rev_px

    # MCAP historik (Q1..Q4) + nu
    if shares > 0 and ttm_list_px:
        q_dates = [d for (d, _) in ttm_list_px[:4]]
        px_map = _yahoo_prices_for_dates(ticker, q_dates)
        for idx, d_end in enumerate(q_dates, start=1):
            pxx = _safe_float(px_map.get(d_end, 0.0))
            if pxx > 0:
                out[f"MCAP Q{idx}"] = shares * pxx
    if mcap > 0:
        out["MCAP nu"] = mcap

    return out

# ------------------------------------------------------------
# Finnhub (valfritt – används ej som default)
# ------------------------------------------------------------
FINNHUB_KEY = st.secrets.get("FINNHUB_API_KEY", "")

def hamta_finnhub_revenue_estimates(ticker: str) -> dict:
    """
    Kräver FINNHUB_API_KEY i secrets. Hämtar annual revenue estimates:
    current FY + next FY (om finns). Returnerar miljoner.
    """
    if not FINNHUB_KEY:
        return {}
    try:
        r = requests.get(
            "https://finnhub.io/api/v1/stock/revenue-estimate",
            params={"symbol": ticker.upper(), "freq": "annual", "token": FINNHUB_KEY},
            timeout=20,
        )
        if r.status_code != 200:
            return {}
        j = r.json() or {}
        data = j.get("data") or []
        if not data:
            return {}
        data.sort(key=lambda d: d.get("period", ""), reverse=False)
        out = {}
        last_two = data[-2:] if len(data) >= 2 else data[-1:]
        if len(last_two) >= 1:
            v = last_two[0].get("revenueAvg") or last_two[0].get("revenueMean") or last_two[0].get("revenue")
            try:
                if v and float(v) > 0:
                    out["Omsättning idag"] = float(v) / 1e6
            except Exception:
                pass
        if len(last_two) == 2:
            v = last_two[1].get("revenueAvg") or last_two[1].get("revenueMean") or last_two[1].get("revenue")
            try:
                if v and float(v) > 0:
                    out["Omsättning nästa år"] = float(v) / 1e6
            except Exception:
                pass
        return out
    except Exception:
        return {}

# ------------------------------------------------------------
# Single-ticker uppdaterare (UI anropar dessa)
# ------------------------------------------------------------
def run_update_price_only(df: pd.DataFrame, ticker: str) -> Tuple[pd.DataFrame, Dict[str, List[str]], dict]:
    """
    Uppdaterar endast kurs/valuta/namn för en ticker. Stämplar auto-uppdaterad.
    """
    tkr = str(ticker).strip().upper()
    debug = {"ticker": tkr}
    changes_map: Dict[str, List[str]] = {}

    # hämta från Yahoo
    y = hamta_yahoo_falt(tkr)
    vals = {}
    for k in ("Bolagsnamn", "Valuta", "Aktuell kurs", "Årlig utdelning", "CAGR 5 år (%)"):
        v = y.get(k)
        if v not in (None, "", 0, 0.0):
            vals[k] = v

    # skriv in till df (om ticker finns)
    if "Ticker" not in df.columns:
        return df, changes_map, {"err": "Saknar kolumn 'Ticker'."}

    rows = df.index[df["Ticker"].astype(str).str.upper() == tkr]
    if len(rows) == 0:
        return df, changes_map, {"err": f"{tkr} hittades inte i tabellen."}

    ridx = rows[0]
    # Vi sätter källa och TS via compute.apply_auto_updates_to_row
    _ = apply_auto_updates_to_row(
        df, ridx, vals, source="Auto (Kurs via Yahoo)", changes_map=changes_map, always_stamp=True
    )
    # inga globala omräkningar behövs för bara kurs
    return df, changes_map, debug


def run_update_full(
    df: pd.DataFrame,
    ticker: str,
    user_rates: dict | None = None,
    use_estimates: bool = False
) -> Tuple[pd.DataFrame, Dict[str, List[str]], dict]:
    """
    Full uppdatering för en ticker:
      1) SEC+Yahoo combo (inkl. Q4-deriv och historiska P/S via TTM + pris)
      2) Yahoo global fallback om SEC saknas
      3) (valfritt) Finnhub estimat för 'Omsättning idag/ nästa år' – AV per default
      4) FMP light för P/S om saknas
    Skriver in fält med apply_auto_updates_to_row(...), stämplar TS även om samma värde,
    och kör sedan uppdatera_berakningar(df, user_rates).
    """
    tkr = str(ticker).strip().upper()
    debug = {"ticker": tkr}
    changes_map: Dict[str, List[str]] = {}

    if "Ticker" not in df.columns:
        return df, changes_map, {"err": "Saknar kolumn 'Ticker'."}

    rows = df.index[df["Ticker"].astype(str).str.upper() == tkr]
    if len(rows) == 0:
        return df, changes_map, {"err": f"{tkr} hittades inte i tabellen."}
    ridx = rows[0]

    vals: Dict[str, float|str] = {}

    # 1) SEC/Yahoo combo
    try:
        base = hamta_sec_yahoo_combo(tkr)
        debug["sec_yahoo"] = {k: base.get(k) for k in [
            "Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
            "Aktuell kurs","Bolagsnamn","Valuta",
            "MCAP nu","MCAP Q1","MCAP Q2","MCAP Q3","MCAP Q4",
            "Årlig utdelning","CAGR 5 år (%)"
        ]}
        for k in ["Bolagsnamn","Valuta","Aktuell kurs","Årlig utdelning","CAGR 5 år (%)",
                  "Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
                  "MCAP nu","MCAP Q1","MCAP Q2","MCAP Q3","MCAP Q4"]:
            v = base.get(k, None)
            if v not in (None, "", 0, 0.0):
                vals[k] = v
    except Exception as e:
        debug["sec_yahoo_err"] = str(e)

    # 2) Yahoo global fallback om centrala fält saknas
    try:
        need_any = not any(k in vals for k in ["Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4"])
        if need_any:
            yglob = hamta_yahoo_global_combo(tkr)
            debug["yahoo_global"] = yglob
            for k in ["Bolagsnamn","Valuta","Aktuell kurs","Årlig utdelning","CAGR 5 år (%)",
                      "Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
                      "MCAP nu","MCAP Q1","MCAP Q2","MCAP Q3","MCAP Q4"]:
                v = yglob.get(k, None)
                if v not in (None, "", 0, 0.0):
                    vals[k] = v
    except Exception as e:
        debug["yahoo_global_err"] = str(e)

    # 3) (valfritt) Finnhub – AV per default pga krav på manuella prognoser
    if use_estimates:
        try:
            fh = hamta_finnhub_revenue_estimates(tkr)
            debug["finnhub"] = fh
            for k in ["Omsättning idag","Omsättning nästa år"]:
                v = fh.get(k)
                if v not in (None, "", 0, 0.0):
                    vals[k] = v
        except Exception as e:
            debug["finnhub_err"] = str(e)

    # 4) FMP light P/S om saknas
    try:
        if ("P/S" not in vals):
            fmpl = hamta_fmp_falt_light(tkr)
            debug["fmp_light"] = {"P/S": fmpl.get("P/S"), "Utestående aktier": fmpl.get("Utestående aktier")}
            v = fmpl.get("P/S")
            if v not in (None, "", 0, 0.0):
                vals["P/S"] = v
            if ("Utestående aktier" not in vals) and (fmpl.get("Utestående aktier") not in (None, "", 0, 0.0)):
                vals["Utestående aktier"] = fmpl["Utestående aktier"]
    except Exception as e:
        debug["fmp_light_err"] = str(e)

    # Skriv till df-raden (TS stämplas alltid)
    _ = apply_auto_updates_to_row(
        df, ridx, vals, source="Auto (SEC/Yahoo→Yahoo→FMP)", changes_map=changes_map, always_stamp=True
    )

    # Räkna om beräkningar (hela df för enkelhet/konsekvens)
    df = uppdatera_berakningar(df, user_rates or {})
    return df, changes_map, debug
