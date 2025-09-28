# stockapp/sources.py
# -*- coding: utf-8 -*-
"""
Datakällor och härledningar:
- Yahoo Finance (yfinance) för pris, mcap, sektor/industri, balans/kassaflöde
- SEC Company Facts för säkra kvartalsintäkter (inkl. fix för dec/jan "Q4 i 10-K")
- FMP fallback för P/S TTM (om API-nyckel finns i st.secrets)
- FX-kurser via Frankfurter -> exchangerate.host (enbart för intern konvertering)
- Härleder: P/S TTM nu + historiska P/S Q1..Q4, Market Cap Q1..Q4, CAGR 5 år (revenue)

OBS: Denna modul uppdaterar inte Google Sheet. Den returnerar en dict med fält
som sedan kan skrivas av den anropande koden. Vi skriver INTE
'Omsättning idag' eller 'Omsättning nästa år' (manuella fält enligt krav).
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
from datetime import datetime as _dt, timedelta as _td, date

import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf

from .config import now_stamp

# =============================================================================
# Små hjälpare
# =============================================================================

def _yfi_get(tkr: yf.Ticker, *keys):
    """Säker hämtning ur yfinance.info med fallback på hist."""
    try:
        info = tkr.info or {}
        for k in keys:
            if k in info and info[k] is not None:
                return info[k]
    except Exception:
        pass
    return None

@st.cache_data(show_spinner=False, ttl=21600)
def _fx_rate_cached(base: str, quote: str) -> float:
    """FX (dagens) via Frankfurter -> exchangerate.host (6h-cache)."""
    base = (base or "").upper(); quote = (quote or "").upper()
    if not base or not quote or base == quote:
        return 1.0
    try:
        r = requests.get("https://api.frankfurter.app/latest", params={"from": base, "to": quote}, timeout=12)
        if r.status_code == 200:
            v = (r.json() or {}).get("rates", {}).get(quote)
            if v: return float(v)
    except Exception:
        pass
    try:
        r = requests.get("https://api.exchangerate.host/latest", params={"base": base, "symbols": quote}, timeout=12)
        if r.status_code == 200:
            v = (r.json() or {}).get("rates", {}).get(quote)
            if v: return float(v)
    except Exception:
        pass
    return 1.0

def _parse_iso(d: str) -> Optional[date]:
    try:
        return _dt.fromisoformat(d.replace("Z", "+00:00")).date()
    except Exception:
        try:
            return _dt.strptime(d, "%Y-%m-%d").date()
        except Exception:
            return None

# =============================================================================
# SEC helpers (CIK-map + companyfacts)
# =============================================================================

SEC_USER_AGENT = st.secrets.get("SEC_USER_AGENT", "StockApp/1.0 (contact: you@example.com)")

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

# ---- shares (instant) robust (summa över multi-class vid senaste datum) -----

def _is_instant_entry(it: dict) -> bool:
    end = it.get("end"); start = it.get("start")
    if not end:
        return False
    if not start:
        return True
    d1 = _parse_iso(str(start)); d2 = _parse_iso(str(end))
    if d1 and d2:
        try:
            return (d2 - d1).days <= 2
        except Exception:
            return False
    return False

def _collect_share_entries(facts: dict) -> list:
    entries = []
    facts_all = (facts.get("facts") or {})
    sources = [
        ("dei", ["EntityCommonStockSharesOutstanding", "EntityCommonSharesOutstanding"]),
        ("us-gaap", ["CommonStockSharesOutstanding", "ShareIssued"]),
        ("ifrs-full", [
            "NumberOfSharesIssued", "IssuedCapitalNumberOfShares",
            "OrdinarySharesNumber", "NumberOfOrdinaryShares"
        ]),
    ]
    unit_keys = ("shares", "USD_shares", "Shares", "SHARES")
    for taxo, keys in sources:
        sect = facts_all.get(taxo, {})
        for key in keys:
            fact = sect.get(key)
            if not fact: continue
            units = fact.get("units") or {}
            for uk in unit_keys:
                arr = units.get(uk)
                if not isinstance(arr, list): continue
                for it in arr:
                    if not _is_instant_entry(it): continue
                    end = _parse_iso(str(it.get("end", "")))
                    val = it.get("val", None)
                    if end and val is not None:
                        try:
                            v = float(val)
                            entries.append({
                                "end": end, "val": v,
                                "frame": it.get("frame") or "",
                                "form": (it.get("form") or "").upper(),
                                "taxo": taxo, "concept": key
                            })
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

# ---- quarterly revenue (robust, inkl. Q4 i 10-K) ----------------------------

def _dur_days(it) -> Optional[int]:
    try:
        d1 = _parse_iso(str(it.get("start", "")))
        d2 = _parse_iso(str(it.get("end", "")))
        if not (d1 and d2): return None
        return (d2 - d1).days
    except Exception:
        return None

def _pick_quarterly_revenue_rows(facts: dict) -> Tuple[List[Tuple[date, float]], Optional[str]]:
    """
    Returnerar [(end_date, revenue_value), ...] (nyast -> äldst) och unit (valuta).
    Tillåter 10-Q **och** 10-K så länge duration är ~kvartal (70..100 dagar).
    Detta fångar Q4 som ofta ligger i 10-K (dec/jan-problemet).
    """
    taxos = [
        ("us-gaap",  {"forms": ("10-Q", "10-Q/A", "10-K", "10-K/A")}),
        ("ifrs-full", {"forms": ("6-K", "6-K/A", "10-Q", "10-Q/A")}),  # vissa FPIs
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

    def _is_quarter_form(form: str, allowed: Tuple[str, ...]) -> bool:
        f = (form or "").upper()
        return any(f == a for a in allowed)

    for taxo, cfg in taxos:
        gaap = (facts.get("facts") or {}).get(taxo, {})
        for name in rev_keys:
            fact = gaap.get(name)
            if not fact: continue
            units = (fact.get("units") or {})
            for unit_code in prefer_units:
                arr = units.get(unit_code)
                if not isinstance(arr, list): continue
                tmp = []
                for it in arr:
                    form = (it.get("form") or "").upper()
                    dur = _dur_days(it)
                    if not _is_quarter_form(form, cfg["forms"]): 
                        continue
                    if dur is None or dur < 70 or dur > 100:
                        # kräver "kvartalslängd"
                        continue
                    end = _parse_iso(str(it.get("end", "")))
                    start = _parse_iso(str(it.get("start", "")))
                    val = it.get("val", None)
                    if not (end and start and val is not None):
                        continue
                    try:
                        v = float(val)
                        tmp.append((end, v))
                    except Exception:
                        pass
                if not tmp: 
                    continue
                # deduplicera på end-date (behåll senaste värde per datum)
                ded: Dict[date, float] = {}
                for end, v in tmp:
                    ded[end] = v
                rows = sorted(ded.items(), key=lambda t: t[0], reverse=True)
                return rows, unit_code

    return [], None

def _ttm_windows(values: List[Tuple[date, float]], need: int = 6) -> List[Tuple[date, float]]:
    """
    Tar [(end_date, kvartalsintäkt), ...] (nyast→äldst) och bygger upp till 'need' TTM-summor:
    [(end_date0, ttm0), (end_date1, ttm1), ...] där ttm0 = sum(q0..q3), ttm1 = sum(q1..q4), ...
    """
    out: List[Tuple[date, float]] = []
    if len(values) < 4:
        return out
    for i in range(0, min(need, len(values) - 3)):
        end_i = values[i][0]
        ttm_i = sum(v for (_, v) in values[i:i+4])
        out.append((end_i, float(ttm_i)))
    return out

def _yahoo_prices_for_dates(ticker: str, dates: List[date]) -> Dict[date, float]:
    """Dagliga priser – returnera 'Close' på eller närmast FÖRE respektive datum."""
    if not dates:
        return {}
    dmin = min(dates) - _td(days=14)
    dmax = max(dates) + _td(days=2)
    try:
        t = yf.Ticker(ticker)
        hist = t.history(start=dmin, end=dmax, interval="1d")
        if hist is None or hist.empty:
            return {}
        hist = hist.sort_index()
        out: Dict[date, float] = {}
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

# =============================================================================
# Yahoo-finansiellt: basics + kvalitetsmått
# =============================================================================

def _compute_cagr_from_yf(t: yf.Ticker) -> float:
    """CAGR baserat på 'Total Revenue' (årsbasis, enkel approx, %)."""
    try:
        df_fin = getattr(t, "financials", None)
        if isinstance(df_fin, pd.DataFrame) and not df_fin.empty and "Total Revenue" in df_fin.index:
            series = df_fin.loc["Total Revenue"].dropna().sort_index()
        else:
            df_is = getattr(t, "income_stmt", None)
            if isinstance(df_is, pd.DataFrame) and not df_is.empty and "Total Revenue" in df_is.index:
                series = df_is.loc["Total Revenue"].dropna().sort_index()
            else:
                return 0.0
        if len(series) < 2:
            return 0.0
        start = float(series.iloc[0]); end = float(series.iloc[-1])
        years = max(1, len(series)-1)
        if start <= 0:
            return 0.0
        cagr = (end / start) ** (1.0/years) - 1.0
        return round(cagr * 100.0, 2)
    except Exception:
        return 0.0

def _margins_and_balance_from_yf(t: yf.Ticker) -> Dict[str, float]:
    """
    Hämtar bruttomarginal, nettomarginal (enkelt, senaste år),
    Debt/Equity (senaste kvartal), Cash, Total Debt, FCF (TTM).
    """
    out: Dict[str, float] = {
        "Bruttomarginal (%)": 0.0,
        "Nettomarginal (%)": 0.0,
        "Debt/Equity": 0.0,
        "Kassa": 0.0,
        "Total skuld": 0.0,
        "FCF (TTM)": 0.0,
    }
    # Marginaler (annual om möjligt)
    try:
        is_annual = getattr(t, "financials", None)
        if isinstance(is_annual, pd.DataFrame) and not is_annual.empty:
            revenue = None
            for cand in ["Total Revenue", "TotalRevenue", "Revenues", "Revenue", "Sales"]:
                if cand in is_annual.index:
                    revenue = is_annual.loc[cand].dropna()
                    break
            gp = None
            for cand in ["Gross Profit", "GrossProfit"]:
                if cand in is_annual.index:
                    gp = is_annual.loc[cand].dropna()
                    break
            ni = None
            for cand in ["Net Income", "NetIncome", "Net Income Common Stockholders"]:
                if cand in is_annual.index:
                    ni = is_annual.loc[cand].dropna()
                    break
            if revenue is not None and len(revenue) > 0:
                rev = float(revenue.iloc[-1]) if len(revenue) else 0.0
                if gp is not None and len(gp) > 0 and rev > 0:
                    out["Bruttomarginal (%)"] = round(float(gp.iloc[-1]) / rev * 100.0, 2)
                if ni is not None and len(ni) > 0 and rev > 0:
                    out["Nettomarginal (%)"] = round(float(ni.iloc[-1]) / rev * 100.0, 2)
    except Exception:
        pass

    # Debt/Equity + Cash + TotalDebt (senaste kvartals-balans)
    try:
        bs_q = getattr(t, "quarterly_balance_sheet", None)
        if isinstance(bs_q, pd.DataFrame) and not bs_q.empty:
            # equity
            eq = 0.0
            for cand in ["Total Stockholder Equity", "Total Equity Gross Minority Interest", "Stockholders Equity"]:
                if cand in bs_q.index and not bs_q.loc[cand].dropna().empty:
                    eq = float(bs_q.loc[cand].dropna().iloc[0])
                    break
            debt = 0.0
            for cand in ["Total Debt", "Long Term Debt", "Short Long Term Debt Total", "Short Long-Term Debt Total"]:
                if cand in bs_q.index and not bs_q.loc[cand].dropna().empty:
                    try:
                        debt = float(bs_q.loc[cand].dropna().iloc[0])
                        break
                    except Exception:
                        pass
            cash = 0.0
            for cand in ["Cash And Cash Equivalents", "CashAndCashEquivalents", "Cash Only"]:
                if cand in bs_q.index and not bs_q.loc[cand].dropna().empty:
                    try:
                        cash = float(bs_q.loc[cand].dropna().iloc[0])
                        break
                    except Exception:
                        pass
            out["Kassa"] = cash
            out["Total skuld"] = debt
            if eq != 0:
                out["Debt/Equity"] = round(debt / eq, 2)
    except Exception:
        pass

    # FCF (TTM) = sum(last 4 quarters: Operating CF - CapEx)
    try:
        cf_q = getattr(t, "quarterly_cashflow", None)
        if isinstance(cf_q, pd.DataFrame) and not cf_q.empty:
            ocf = None
            capex = None
            for cand in ["Operating Cash Flow", "Total Cash From Operating Activities"]:
                if cand in cf_q.index:
                    ocf = cf_q.loc[cand].dropna().astype(float)
                    break
            for cand in ["Capital Expenditures"]:
                if cand in cf_q.index:
                    capex = cf_q.loc[cand].dropna().astype(float)
                    break
            if ocf is not None and len(ocf) > 0:
                if capex is None or len(capex) == 0:
                    fcf_series = ocf
                else:
                    # capex oftast negativt; fcf = ocf + capex
                    fcf_series = ocf + capex
                out["FCF (TTM)"] = float(fcf_series.sort_index().tail(4).sum())
    except Exception:
        pass

    return out

def _yahoo_basics(ticker: str) -> Dict[str, any]:
    """Hämtar basfält från Yahoo."""
    out = {
        "Bolagsnamn": "",
        "Valuta": "USD",
        "Aktuell kurs": 0.0,
        "Sektor": "",
        "Industri": "",
        "Årlig utdelning": 0.0,
        "Direktavkastning (%)": 0.0,
        "CAGR 5 år (%)": 0.0,
        "Market Cap (nu)": 0.0,
        "Utestående aktier": 0.0,  # milj
    }
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        pris = info.get("regularMarketPrice")
        if pris is None:
            h = t.history(period="1d")
            if not h.empty and "Close" in h:
                pris = float(h["Close"].iloc[-1])
        if pris is not None:
            out["Aktuell kurs"] = float(pris)

        valuta = info.get("currency")
        if valuta:
            out["Valuta"] = str(valuta).upper()

        namn = info.get("shortName") or info.get("longName") or ""
        if namn:
            out["Bolagsnamn"] = str(namn)

        sect = info.get("sector") or ""
        ind = info.get("industry") or ""
        out["Sektor"] = str(sect)
        out["Industri"] = str(ind)

        mcap = info.get("marketCap") or 0.0
        try:
            out["Market Cap (nu)"] = float(mcap or 0.0)
        except Exception:
            pass

        so = info.get("sharesOutstanding") or 0.0
        try:
            if so and so > 0:
                out["Utestående aktier"] = float(so) / 1e6
        except Exception:
            pass

        div_rate = info.get("dividendRate")
        if div_rate is not None:
            try:
                out["Årlig utdelning"] = float(div_rate)
            except Exception:
                pass

        dy = info.get("dividendYield")
        if dy is not None and dy > 0:
            try:
                out["Direktavkastning (%)"] = float(dy) * 100.0
            except Exception:
                pass

        out["CAGR 5 år (%)"] = _compute_cagr_from_yf(t)
        out.update(_margins_and_balance_from_yf(t))

    except Exception:
        pass
    return out

# =============================================================================
# P/S & kvartal – SEC + Yahoo pris + valuta
# =============================================================================

def _implied_shares(price: float, mcap: float) -> float:
    try:
        price = float(price or 0.0); mcap = float(mcap or 0.0)
        if price > 0 and mcap > 0:
            return mcap / price
    except Exception:
        pass
    return 0.0

def _ps_from_sec_yahoo(ticker: str, price_ccy: str, mcap_now: float, px_now: float) -> Dict[str, float]:
    """
    Bygger P/S och P/S Q1..Q4 + Market Cap Q1..Q4 via SEC kvartalsintäkter (TTM) och Yahoo-priser.
    Återger även (eventuellt) uppdaterat 'Utestående aktier' via SEC instant shares om Yahoo-implied saknas.
    """
    out: Dict[str, float] = {}
    cik = _sec_cik_for(ticker)
    if not cik:
        return out
    facts, sc = _sec_companyfacts(cik)
    if sc != 200 or not isinstance(facts, dict):
        return out

    # Shares (SEC instant som backup om implied saknas)
    sec_shares = _sec_latest_shares_robust(facts)  # styck
    implied = _implied_shares(px_now, mcap_now)
    shares_used = implied if implied > 0 else (sec_shares if sec_shares > 0 else 0.0)
    if shares_used > 0:
        out["Utestående aktier"] = shares_used / 1e6  # milj

    # Kvartalsintäkter + unit → TTM & ev. FX-konvertering
    q_rows, rev_unit = _pick_quarterly_revenue_rows(facts)
    if not q_rows or not rev_unit:
        return out

    conv = 1.0
    if rev_unit.upper() != price_ccy.upper():
        conv = _fx_rate_cached(rev_unit.upper(), price_ccy.upper()) or 1.0

    ttm_list = _ttm_windows(q_rows, need=6)
    if not ttm_list:
        return out

    ttm_px = [(d, v * conv) for (d, v) in ttm_list]

    # P/S (TTM) nu
    if mcap_now > 0 and ttm_px:
        ltm_now = ttm_px[0][1]
        if ltm_now > 0:
            out["P/S"] = float(mcap_now / ltm_now)

    # Historiska P/S Q1..Q4 + Market Cap Q1..Q4
    if shares_used > 0 and ttm_px:
        q_dates = [d for (d, _) in ttm_px[:4]]
        px_map = _yahoo_prices_for_dates(ticker, q_dates)
        for idx, (d_end, ttm_rev_px) in enumerate(ttm_px[:4], start=1):
            if ttm_rev_px and ttm_rev_px > 0:
                px_hist = px_map.get(d_end)
                if px_hist and px_hist > 0:
                    mcap_hist = shares_used * float(px_hist)
                    out[f"P/S Q{idx}"] = float(mcap_hist / ttm_rev_px)
                    out[f"Market Cap Q{idx}"] = float(mcap_hist)

    return out

# =============================================================================
# FMP-fallback (valfritt)
# =============================================================================

FMP_BASE = st.secrets.get("FMP_BASE", "https://financialmodelingprep.com")
FMP_KEY  = st.secrets.get("FMP_API_KEY", "")

def _fmp_get(path: str, params=None):
    if not FMP_KEY:
        return None, 0
    params = (params or {}).copy()
    params["apikey"] = FMP_KEY
    url = f"{FMP_BASE}/{path}"
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code == 200:
            try:
                return r.json(), 200
            except Exception:
                return None, 200
        return None, r.status_code
    except Exception:
        return None, 0

def _fmp_ps_ttm(ticker: str) -> Optional[float]:
    """Hämta P/S TTM som sista utväg."""
    js, sc = _fmp_get(f"api/v3/ratios-ttm/{ticker}")
    if isinstance(js, list) and js:
        v = js[0].get("priceToSalesTTM") or js[0].get("priceToSalesRatioTTM")
        try:
            if v and float(v) > 0:
                return float(v)
        except Exception:
            return None
    return None

# =============================================================================
# Publika huvudfunktioner
# =============================================================================

def fetch_all_fields_for_ticker(ticker: str) -> Dict[str, any]:
    """
    Kör komplett hämtning för ett bolag (utom manuella fält 'Omsättning idag' / 'Omsättning nästa år').
    Returnerar en dict med fält som kan skrivas in i DF/Sheets.
    """
    tkr = str(ticker).strip().upper()
    out: Dict[str, any] = {}

    # 1) Yahoo basics (pris/valuta/namn/sector/industry/mcap/shares + kvalitetsmått)
    yb = _yahoo_basics(tkr)
    out.update({k: v for k, v in yb.items() if v not in (None, "")})

    price_ccy = (out.get("Valuta") or "USD").upper()
    mcap_now = float(out.get("Market Cap (nu)", 0.0) or 0.0)
    px_now   = float(out.get("Aktuell kurs", 0.0) or 0.0)

    # 2) SEC TTM & kvartal → P/S och MCAP Q1..Q4 (fixar dec/jan via 10-K med kvartalsdur)
    ps_pack = _ps_from_sec_yahoo(tkr, price_ccy, mcap_now, px_now)
    out.update(ps_pack)

    # 3) FMP fallback för P/S om saknas
    if "P/S" not in out or not out["P/S"] or out["P/S"] <= 0:
        v = _fmp_ps_ttm(tkr)
        if v and v > 0:
            out["P/S"] = float(v)

    # 4) P/S-snitt (av positiva Q1..Q4)
    ps_quarters = [out.get(f"P/S Q{i}", 0.0) for i in range(1,5)]
    ps_clean = [float(x) for x in ps_quarters if float(x) > 0]
    out["P/S-snitt"] = round(np.mean(ps_clean), 2) if ps_clean else 0.0

    # 5) Marknadsvärden Q1..Q4 – se till att kolumner finns (0 om ej beräknat)
    for i in range(1,5):
        out.setdefault(f"Market Cap Q{i}", 0.0)

    # 6) OBS – manuella prognosfält uppdateras INTE här (medvetet):
    #    "Omsättning idag", "Omsättning nästa år"

    # 7) Riktkurser – beror på manuella/beräknade omsättningar & P/S-snitt; lämnas till calc-modul.
    #    (vi sätter inte här för att undvika att skriva över).

    # 8) Stäm av utestående aktier: om Yahoo saknade och SEC gav – det ligger redan i out.

    return out
