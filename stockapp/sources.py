# stockapp/sources.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
import math
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from functools import lru_cache
from typing import Dict, Tuple, List, Optional

try:
    import streamlit as st
except Exception:
    st = None

try:
    import yfinance as yf
except Exception:
    yf = None

# ------------------------------------------------------------
# Konfig & helpers
# ------------------------------------------------------------

def _now_ts():
    try:
        import pytz
        tz = pytz.timezone("Europe/Stockholm")
        return datetime.now(tz)
    except Exception:
        return datetime.now()

def _env_secret(name: str, default: str = "") -> str:
    # Hämta från st.secrets om möjligt, annars env-var
    val = ""
    if st is not None:
        try:
            val = st.secrets.get(name, "")
        except Exception:
            val = ""
    if not val:
        val = os.environ.get(name, "")
    return val or default

SEC_USER_AGENT = _env_secret("SEC_USER_AGENT", "StockApp/1.0 (contact: you@example.com)")
FMP_BASE       = _env_secret("FMP_BASE", "https://financialmodelingprep.com")
FMP_KEY        = _env_secret("FMP_API_KEY", "")

# ------------------------------------------------------------
# Valuta
# ------------------------------------------------------------

@lru_cache(maxsize=256)
def fx_rate(base: str, quote: str) -> float:
    base = (base or "").upper()
    quote = (quote or "").upper()
    if not base or not quote or base == quote:
        return 1.0
    try:
        r = requests.get("https://api.frankfurter.app/latest",
                         params={"from": base, "to": quote}, timeout=12)
        if r.status_code == 200:
            j = r.json() or {}
            v = (j.get("rates") or {}).get(quote)
            if v:
                return float(v)
    except Exception:
        pass
    try:
        r = requests.get("https://api.exchangerate.host/latest",
                         params={"base": base, "symbols": quote}, timeout=12)
        if r.status_code == 200:
            j = r.json() or {}
            v = (j.get("rates") or {}).get(quote)
            if v:
                return float(v)
    except Exception:
        pass
    return 1.0

# ------------------------------------------------------------
# Yahoo helpers
# ------------------------------------------------------------

def _yf_ticker(ticker: str):
    if yf is None:
        return None
    try:
        return yf.Ticker(ticker)
    except Exception:
        return None

def _yf_info(tkr) -> dict:
    try:
        return tkr.info or {}
    except Exception:
        return {}

def _yf_price_currency_name_sector(ticker: str) -> dict:
    out = {"Aktuell kurs": 0.0, "Valuta": "USD", "Bolagsnamn": "", "Sektor": ""}
    t = _yf_ticker(ticker)
    if t is None:
        return out
    info = _yf_info(t)
    # Pris
    price = info.get("regularMarketPrice")
    if price is None:
        try:
            h = t.history(period="1d")
            if not h.empty and "Close" in h:
                price = float(h["Close"].iloc[-1])
        except Exception:
            price = None
    if price is not None:
        out["Aktuell kurs"] = float(price)
    # Valuta & namn
    if info.get("currency"):
        out["Valuta"] = str(info.get("currency")).upper()
    nm = info.get("shortName") or info.get("longName")
    if nm:
        out["Bolagsnamn"] = str(nm)
    # Sektor
    if info.get("sector"):
        out["Sektor"] = str(info.get("sector"))
    return out

def _yf_marketcap_and_shares(ticker: str, fallback_price: float = 0.0) -> Tuple[float, float]:
    """
    Returnerar (marketCap, shares) från Yahoo.
    shares via implied (mcap/price) eller 'sharesOutstanding'.
    """
    t = _yf_ticker(ticker)
    if t is None:
        return 0.0, 0.0
    info = _yf_info(t)
    mcap = info.get("marketCap") or 0.0
    try:
        mcap = float(mcap or 0.0)
    except Exception:
        mcap = 0.0
    px = info.get("regularMarketPrice")
    if px is None or px == 0:
        px = fallback_price or 0.0
    sh = 0.0
    if mcap > 0 and px and px > 0:
        sh = mcap / px
    else:
        so = info.get("sharesOutstanding") or 0.0
        try:
            sh = float(so or 0.0)
        except Exception:
            sh = 0.0
    return (mcap, sh)

def _yf_quarterly_revenues(ticker: str) -> List[Tuple[date, float]]:
    """
    Försöker läsa kvartalsintäkter från Yahoo.
    Returnerar [(period_end_date, value), ...] nyast först.
    """
    t = _yf_ticker(ticker)
    if t is None:
        return []
    # 1) quarterly_financials
    try:
        qf = t.quarterly_financials
        if isinstance(qf, pd.DataFrame) and not qf.empty:
            # Letar efter total revenue-liknande rader
            wanted = {"total revenue","totalrevenue","revenues","revenue","sales",
                      "total revenue (ttm)","revenues from contracts with customers"}
            idx_norm = {str(i).strip().lower(): i for i in qf.index}
            key = None
            for k in idx_norm:
                if k in wanted:
                    key = idx_norm[k]; break
            if key is None:
                # fallback: brute contains
                for i in qf.index:
                    s = str(i).lower()
                    if "revenue" in s:
                        key = i; break
            if key is not None:
                ser = qf.loc[key].dropna()
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
    # 2) income_stmt alt (kan vara tomt)
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

def _yf_margins_de_cash_fcf(ticker: str) -> dict:
    """
    Försöker hämta bruttomarginal, nettomarginal, D/E, kassa och FCF TTM via Yahoo.
    Robust mot tom data.
    """
    out = {
        "Bruttomarginal (%)": 0.0,
        "Nettomarginal (%)": 0.0,
        "Debt/Equity": 0.0,
        "Kassa (valuta)": 0.0,
        "FCF TTM (valuta)": 0.0,
    }
    t = _yf_ticker(ticker)
    if t is None:
        return out
    info = _yf_info(t)

    # Margins från info (andel)
    try:
        gm = info.get("grossMargins", None)
        if gm is not None:
            out["Bruttomarginal (%)"] = float(gm) * 100.0
    except Exception:
        pass
    try:
        pm = info.get("profitMargins", None)
        if pm is not None:
            out["Nettomarginal (%)"] = float(pm) * 100.0
    except Exception:
        pass

    # D/E via balance sheet (quarterly först)
    def _pick_balance_total(dic: dict, keys: List[str]) -> float:
        # case-insensitive
        for k in dic.keys():
            for target in keys:
                if str(k).strip().lower() == target.lower():
                    try:
                        v = dic.get(k)
                        return float(v or 0.0)
                    except Exception:
                        pass
        return 0.0

    # Ta senaste kolumn
    try:
        qbs = t.quarterly_balance_sheet
        if isinstance(qbs, pd.DataFrame) and not qbs.empty:
            col = qbs.columns[0]
            dic = {str(i): qbs.loc[i, col] for i in qbs.index}
            total_debt = _pick_balance_total(dic, ["Total Debt","TotalDebt"])
            equity = _pick_balance_total(dic, ["Total Stockholder Equity","Total Equity Gross Minority Interest","TotalEquityGrossMinorityInterest","StockholdersEquity"])
            if equity and equity != 0:
                out["Debt/Equity"] = float(total_debt) / float(equity)
    except Exception:
        pass
    if out["Debt/Equity"] == 0.0:
        try:
            bs = t.balance_sheet
            if isinstance(bs, pd.DataFrame) and not bs.empty:
                col = bs.columns[0]
                dic = {str(i): bs.loc[i, col] for i in bs.index}
                total_debt = _pick_balance_total(dic, ["Total Debt","TotalDebt"])
                equity = _pick_balance_total(dic, ["Total Stockholder Equity","Total Equity Gross Minority Interest","TotalEquityGrossMinorityInterest","StockholdersEquity"])
                if equity and equity != 0:
                    out["Debt/Equity"] = float(total_debt) / float(equity)
        except Exception:
            pass

    # Kassa & FCF
    # Kassa via info om finns, annars via balance_sheet "Cash And Cash Equivalents"
    try:
        cash = info.get("totalCash", None)
        if cash is not None:
            out["Kassa (valuta)"] = float(cash)
    except Exception:
        pass
    if out["Kassa (valuta)"] == 0.0:
        try:
            qbs = t.quarterly_balance_sheet
            if isinstance(qbs, pd.DataFrame) and not qbs.empty:
                col = qbs.columns[0]
                dic = {str(i): qbs.loc[i, col] for i in qbs.index}
                # Försök hitta "Cash And Cash Equivalents"
                for key in dic.keys():
                    if "cash" in key.lower() and "equivalents" in key.lower():
                        out["Kassa (valuta)"] = float(dic[key] or 0.0)
                        break
        except Exception:
            pass

    # FCF TTM via info eller cashflow
    try:
        fcf = info.get("freeCashflow", None)
        if fcf is not None:
            out["FCF TTM (valuta)"] = float(fcf)
    except Exception:
        pass
    if out["FCF TTM (valuta)"] == 0.0:
        # Cashflow DataFrame kan ha "Free Cash Flow"
        try:
            qcf = t.quarterly_cashflow
            if isinstance(qcf, pd.DataFrame) and not qcf.empty:
                # Summera 4 senaste "Free Cash Flow" eller "FreeCashFlow"
                row = None
                for idx in qcf.index:
                    if str(idx).lower().replace(" ","") in ("freecashflow","free cash flow".replace(" ","")):
                        row = qcf.loc[idx].dropna()
                        break
                if row is not None and not row.empty:
                    # Ta absolut senaste (detta är inte exakt TTM men approximation om bara 1 kolumn)
                    # Bättre: summera upp till 4 om >1 kolumn
                    vals = [float(v) for v in row.values[:4] if pd.notna(v)]
                    out["FCF TTM (valuta)"] = float(sum(vals)) if vals else float(row.iloc[0])
        except Exception:
            pass

    return out

def _yf_sector(ticker: str) -> str:
    t = _yf_ticker(ticker)
    if t is None:
        return ""
    info = _yf_info(t)
    return str(info.get("sector") or "") if info else ""

# ------------------------------------------------------------
# SEC helpers
# ------------------------------------------------------------

def _sec_get(url: str, params=None):
    try:
        r = requests.get(url, params=(params or {}), headers={"User-Agent": SEC_USER_AGENT}, timeout=30)
        if r.status_code == 200:
            return r.json(), 200
        return None, r.status_code
    except Exception:
        return None, 0

@lru_cache(maxsize=1)
def _sec_ticker_map() -> Dict[str, str]:
    j, sc = _sec_get("https://www.sec.gov/files/company_tickers.json")
    out = {}
    if isinstance(j, dict):
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

def _parse_iso(d: str):
    try:
        return datetime.fromisoformat(d.replace("Z", "+00:00")).date()
    except Exception:
        try:
            return datetime.strptime(d, "%Y-%m-%d").date()
        except Exception:
            return None

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

def _sec_quarterly_revenues_dated_with_unit(facts: dict, max_quarters: int = 24):
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
    prefer_units = ("USD","CAD","EUR","GBP")

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
                tmp = []
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
                    if dur is None or dur < 70 or dur > 100:
                        continue
                    try:
                        v = float(val)
                        tmp.append((end, v))
                    except Exception:
                        pass
                if not tmp:
                    continue
                # dedupe by end-date
                ded = {}
                for end, v in tmp:
                    ded[end] = v
                rows = sorted(ded.items(), key=lambda t: t[0], reverse=True)[:max_quarters]
                if rows:
                    return rows, unit_code
    return [], None

# ------------------------------------------------------------
# P/S-bygge m.fl.
# ------------------------------------------------------------

def _yahoo_prices_for_dates(ticker: str, dates: List[date]) -> Dict[date, float]:
    if yf is None or not dates:
        return {}
    t = _yf_ticker(ticker)
    if t is None:
        return {}
    dmin = min(dates) - timedelta(days=14)
    dmax = max(dates) + timedelta(days=2)
    try:
        hist = t.history(start=dmin, end=dmax, interval="1d")
        if hist is None or hist.empty:
            return {}
        hist = hist.sort_index()
        idx = list(hist.index.date)
        closes = list(hist["Close"].values)
        out = {}
        for d in dates:
            px = None
            for j in range(len(idx)-1, -1, -1):
                if idx[j] <= d:
                    try:
                        px = float(closes[j])
                    except Exception:
                        px = None
                    break
            if px is not None:
                out[d] = px
        return out
    except Exception:
        return {}

def _merge_sec_yahoo_quarters(ticker: str) -> Tuple[List[Tuple[date,float]], Optional[str]]:
    """
    Kombinerar kvartalsintäkter från SEC och Yahoo för att undvika Dec/Jan-gap.
    Returnerar (rows, unit) där rows är nyast→äldst [(end_date, value_in_unit)].
    Om SEC saknas → unit=None (Yahoo saknar garanterad unit); vi använder prisvaluta separat.
    """
    cik = _sec_cik_for(ticker)
    rows_sec, unit = [], None
    if cik:
        facts, sc = _sec_companyfacts(cik)
        if sc == 200 and isinstance(facts, dict):
            rows_sec, unit = _sec_quarterly_revenues_dated_with_unit(facts, max_quarters=24)

    rows_yf = _yf_quarterly_revenues(ticker)

    # Merge på end_date (nyast→äldst)
    ded: Dict[date, float] = {}
    for d, v in rows_sec:
        ded[d] = float(v)
    for d, v in rows_yf:
        # skriv inte över SEC om redan finns
        if d not in ded:
            ded[d] = float(v)
    rows = sorted(ded.items(), key=lambda t: t[0], reverse=True)
    return rows, unit

def _ttm_windows(values: List[Tuple[date,float]], need: int = 4) -> List[Tuple[date, float]]:
    """
    values = [(end_date, kvartalsintäkt)] nyast→äldst
    Returnerar upp till 'need' rullande TTM-summor (nyast först).
    """
    out = []
    if len(values) < 4:
        return out
    # Bygg från NYAST (index 0) => ttm0 = q0+q1+q2+q3, ttm1 = q1+q2+q3+q4, ...
    for i in range(0, min(need, len(values)-3)):
        end_i = values[i][0]
        ttm_i = sum(v for (_, v) in values[i:i+4])
        out.append((end_i, float(ttm_i)))
    return out

def _ps_quarters_from_rows(ticker: str, rows_merged: List[Tuple[date,float]], rev_unit: Optional[str], price_ccy: str, shares: float) -> Dict[str, float]:
    """
    Beräknar P/S (nu) och P/S Q1..Q4 baserat på TTM över de sammanslagna kvartalen.
    Konverterar revenue till prisvaluta om unit != price_ccy och vi vet unit.
    """
    # Konvertera intäkter till prisvaluta om vi kan
    conv = 1.0
    if rev_unit and price_ccy and rev_unit.upper() != price_ccy.upper():
        conv = fx_rate(rev_unit.upper(), price_ccy.upper()) or 1.0
    rows_px = [(d, v * conv) for (d, v) in rows_merged]

    ttm_list = _ttm_windows(rows_px, need=6)  # hämta gärna 6, så vi kan fylla upp till 4 P/S-historik
    out = {}

    # Dagens marketcap & pris (behövs separat)
    prices = _yahoo_prices_for_dates(ticker, [d for (d, _) in rows_px[:6]])
    # shares ges in (robust) — om 0 skippar vi historik
    if shares <= 0:
        return out

    # P/S Q1..Q4: använd upp till 4 TTM-fönster
    for idx, (d_end, ttm_rev_px) in enumerate(ttm_list[:4], start=1):
        if ttm_rev_px and ttm_rev_px > 0:
            px = prices.get(d_end, None)
            if px and px > 0:
                mcap_hist = shares * float(px)
                out[f"P/S Q{idx}"] = float(mcap_hist / ttm_rev_px)

    # P/S nu (senaste TTM)
    if ttm_list:
        ttm_now = ttm_list[0][1]
        # MarketCap nu via Yahoo (igen), eller shares*price_now
        mcap_now, _ = _yf_marketcap_and_shares(ticker)
        if (not mcap_now or mcap_now <= 0) and prices:
            # ta nyaste priset i prices (matchar rows_px[0] end)
            px0 = prices.get(ttm_list[0][0], None)
            if px0 and px0 > 0:
                mcap_now = float(px0) * shares
        if mcap_now and mcap_now > 0 and ttm_now > 0:
            out["P/S"] = float(mcap_now / ttm_now)

    return out

# ------------------------------------------------------------
# FMP fallback för P/S (TTM)
# ------------------------------------------------------------

def _fmp_ps_ttm(ticker: str) -> Optional[float]:
    if not FMP_KEY:
        return None
    try:
        url = f"{FMP_BASE}/api/v3/ratios-ttm/{ticker.upper()}"
        r = requests.get(url, params={"apikey": FMP_KEY}, timeout=20)
        if r.status_code == 200:
            j = r.json() or []
            if isinstance(j, list) and j:
                v = j[0].get("priceToSalesTTM") or j[0].get("priceToSalesRatioTTM")
                if v is not None and float(v) > 0:
                    return float(v)
    except Exception:
        return None
    return None

# ------------------------------------------------------------
# Publika API: runners
# ------------------------------------------------------------

def fetch_price_only(ticker: str) -> Tuple[Dict, Dict]:
    """
    Snabb uppdatering: pris, valuta, namn, sector, market cap.
    Returnerar (vals, debug)
    """
    dbg = {"runner": "price_only", "ticker": ticker}
    vals = _yf_price_currency_name_sector(ticker)
    # Market cap & implied shares (sparar endast mcap här)
    mcap, sh = _yf_marketcap_and_shares(ticker, fallback_price=vals.get("Aktuell kurs", 0.0))
    if mcap and mcap > 0:
        vals["Market Cap (nu)"] = float(mcap)
    # Sektor redan i vals
    dbg["vals"] = {k: vals.get(k) for k in ["Aktuell kurs","Valuta","Bolagsnamn","Sektor","Market Cap (nu)"]}
    return vals, dbg

def fetch_all_fields_for_ticker(ticker: str) -> Tuple[Dict, Dict]:
    """
    Full auto: försöker fylla “allt” som inte är ren prognos (Omsättning-idag/nästa år).
    Returnerar (vals, debug) där vals bara innehåller nycklar som bör skrivas.
    """
    dbg = {"runner": "full_auto", "ticker": ticker}
    vals: Dict[str, float | str] = {}

    # 1) Bas via Yahoo
    base = _yf_price_currency_name_sector(ticker)
    vals.update({k: v for k, v in base.items() if v not in (None, "", 0, 0.0)})
    price_ccy = (base.get("Valuta") or "USD").upper()
    price_now = float(base.get("Aktuell kurs", 0.0) or 0.0)

    # 2) Market cap & shares
    mcap_now, shares_imp = _yf_marketcap_and_shares(ticker, fallback_price=price_now)
    dbg["yahoo_mcap"] = mcap_now
    dbg["yahoo_implied_shares"] = shares_imp

    # SEC shares robust fallback om implied saknas
    shares_used = float(shares_imp or 0.0)
    cik = _sec_cik_for(ticker)
    if cik and (shares_used <= 0):
        facts, sc = _sec_companyfacts(cik)
        if sc == 200 and isinstance(facts, dict):
            sh_sec = _sec_latest_shares_robust(facts)
            if sh_sec and sh_sec > 0:
                shares_used = float(sh_sec)

    if shares_used > 0:
        vals["Utestående aktier"] = float(shares_used) / 1e6  # i miljoner

    if mcap_now and mcap_now > 0:
        vals["Market Cap (nu)"] = float(mcap_now)

    # 3) Marginaler / D/E / Kassa / FCF
    fin = _yf_margins_de_cash_fcf(ticker)
    for k, v in fin.items():
        if v not in (None, "", 0, 0.0):
            vals[k] = float(v)

    # 4) Kvartalsintäkter (SEC + Yahoo merge) → P/S TTM + P/S Q1..Q4
    rows_merged, rev_unit = _merge_sec_yahoo_quarters(ticker)
    dbg["quarters_count"] = len(rows_merged)
    if rows_merged:
        ps_vals = _ps_quarters_from_rows(ticker, rows_merged, rev_unit=rev_unit, price_ccy=price_ccy, shares=shares_used)
        # Om P/S (TTM) saknas men FMP kan leverera → fyll
        if ("P/S" not in ps_vals) or (not ps_vals.get("P/S") or ps_vals.get("P/S") <= 0):
            ps_fmp = _fmp_ps_ttm(ticker)
            if ps_fmp and ps_fmp > 0:
                ps_vals["P/S"] = float(ps_fmp)
        for k, v in ps_vals.items():
            if v and float(v) > 0:
                vals[k] = float(v)

    # 5) Runway (mån) om FCF<0 och kassa>0
    cash = float(vals.get("Kassa (valuta)", 0.0) or 0.0)
    fcf  = float(vals.get("FCF TTM (valuta)", 0.0) or 0.0)
    runway_m = 0.0
    if cash > 0 and fcf < 0:
        try:
            runway_m = 12.0 * cash / abs(fcf)
        except Exception:
            runway_m = 0.0
    if runway_m > 0:
        vals["Runway (mån)"] = float(round(runway_m, 1))

    # Debug
    dbg["vals_keys"] = list(vals.keys())
    dbg["rev_unit"] = rev_unit
    dbg["price_ccy"] = price_ccy
    return vals, dbg
