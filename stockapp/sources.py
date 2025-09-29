# -*- coding: utf-8 -*-
"""
stockapp/sources.py
Hämtar fundamentala data och bygger alla nyckeltal vi kommit överens om.

Källordning:
- Yahoo Finance (yfinance) för bas, snabbhet och sektor/industri
- FMP (FinancialModelingPrep) som fallback/komplettering (kräver API-nyckel i st.secrets["FMP_API_KEY"])
- SEC-stilad logik via Yahoo quarterly_financials för P/S-historik och TTM-fönster (robust mot dec/jan)

Output:
- Platt dict { fältnamn: värde } med bl.a.:
    Bolagsnamn, Valuta, Aktuell kurs, Market Cap, Enterprise Value,
    EV/EBITDA (ttm), Gross margin (%), Operating margin (%), Net margin (%),
    ROE (%), P/B, Debt/Equity, Total cash, Total debt, Net debt,
    OCF, CapEx, FCF, FCF Yield (%), Dividend yield (%), Dividend payout (FCF) (%),
    Net debt / EBITDA, Sector, Industry,
    Utestående aktier (miljoner), P/S, P/S Q1-4, MktCap Q1-4, P/S-snitt
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import math
import time
import datetime as dt

import numpy as np
import pandas as pd
import yfinance as yf
import requests
import streamlit as st

# ---- Hjälpare ---------------------------------------------------------------

def _now_stamp() -> str:
    try:
        import pytz
        tz = pytz.timezone("Europe/Stockholm")
        return dt.datetime.now(tz).strftime("%Y-%m-%d")
    except Exception:
        return dt.datetime.now().strftime("%Y-%m-%d")

def _safe_float(x, default=0.0) -> float:
    try:
        if x is None or (isinstance(x, str) and not x.strip()):
            return float(default)
        return float(x)
    except Exception:
        return float(default)

def _pos(x: float) -> bool:
    try:
        return float(x) > 0
    except Exception:
        return False

def _nonneg(x: float) -> bool:
    try:
        return float(x) >= 0
    except Exception:
        return False

def _fmt_pct(x: float) -> float:
    # Säker runda till 2 decimaler
    try:
        return round(float(x), 2)
    except Exception:
        return 0.0

def _ttm_windows(values: List[Tuple[dt.date, float]], need: int = 5) -> List[Tuple[dt.date, float]]:
    """
    Tar [(period_end_date, quarterly_value), ...] nyast→äldst och bygger upp till 'need' TTM-summor.
    Returnerar [(end_date0, ttm0), (end_date1, ttm1), ...]
    """
    out = []
    if len(values) < 4:
        return out
    # värden förväntas vara sorterade nyast→äldst
    for i in range(0, min(need, len(values) - 3)):
        end_i = values[i][0]
        ttm_i = sum(v for (_, v) in values[i:i+4])
        out.append((end_i, float(ttm_i)))
    return out

def _yahoo_prices_for_dates(ticker: str, dates: List[dt.date]) -> Dict[dt.date, float]:
    """
    Hämtar dagliga priser i ett fönster kring givna datum och returnerar Close på eller närmast FÖRE respektive datum.
    """
    if not dates:
        return {}
    dmin = min(dates) - dt.timedelta(days=14)
    dmax = max(dates) + dt.timedelta(days=2)
    try:
        t = yf.Ticker(ticker)
        hist = t.history(start=dmin, end=dmax, interval="1d")
        if hist is None or hist.empty:
            return {}
        hist = hist.sort_index()
        out = {}
        idx = list(hist.index.date)
        closes = list(hist["Close"].values)
        # Map: för varje datum, ta närmast föregående Close
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

# ---- Yahoo ------------------------------------------------------------------

def _yahoo_info(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    try:
        info = t.info or {}
    except Exception:
        info = {}
    return info

def _yahoo_quarterly_revenues(ticker: str) -> List[Tuple[dt.date, float]]:
    """
    Försöker läsa kvartalsintäkter från Yahoo och returnera [(period_end_date, value)] nyast→äldst.
    Robust mot december/januari – vi använder datumen som finns utan att filtrera bort "konstiga" perioder.
    """
    t = yf.Ticker(ticker)

    # 1) quarterly_financials (vanligaste)
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

    # 2) income_stmt fallback (kvartalsraden kan heta "Total Revenue")
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

def fetch_yahoo_basics(ticker: str) -> Dict[str, float]:
    """
    Hämtar basfält & många nyckeltal direkt från Yahoo info.
    Fyller bara in nycklar om de finns.
    """
    info = _yahoo_info(ticker)
    out: Dict[str, float] = {}

    # Namn, valuta, kurs
    name = info.get("shortName") or info.get("longName")
    if name: out["Bolagsnamn"] = str(name)
    ccy = info.get("currency")
    if ccy: out["Valuta"] = str(ccy).upper()

    price = info.get("regularMarketPrice")
    if price is None:
        try:
            h = yf.Ticker(ticker).history(period="1d")
            if not h.empty and "Close" in h:
                price = float(h["Close"].iloc[-1])
        except Exception:
            price = None
    if price is not None:
        out["Aktuell kurs"] = float(price)

    # Sektor / industri
    sec = info.get("sector")
    ind = info.get("industry")
    if sec: out["Sector"] = str(sec)
    if ind: out["Industry"] = str(ind)

    # Market cap & shares
    mc = info.get("marketCap")
    if mc is not None:
        out["Market Cap"] = float(mc)
    so = info.get("sharesOutstanding")
    if so is not None and _pos(so):
        out["Utestående aktier"] = float(so) / 1e6  # i miljoner, konsekvent med din databas

    # Enterprise Value
    ev = info.get("enterpriseValue")
    if ev is not None:
        out["Enterprise Value"] = float(ev)

    # Marginaler (proportioner -> %)
    gm = info.get("grossMargins")
    if gm is not None:
        out["Gross margin (%)"] = _fmt_pct(float(gm) * 100.0)
    om = info.get("operatingMargins")
    if om is not None:
        out["Operating margin (%)"] = _fmt_pct(float(om) * 100.0)
    nm = info.get("profitMargins")
    if nm is not None:
        out["Net margin (%)"] = _fmt_pct(float(nm) * 100.0)

    # ROE (%)
    roe = info.get("returnOnEquity")
    if roe is not None:
        out["ROE (%)"] = _fmt_pct(float(roe) * 100.0)

    # P/B
    pb = info.get("priceToBook")
    if pb is not None:
        out["P/B"] = float(pb)

    # Debt/Equity
    de = info.get("debtToEquity")
    if de is not None:
        out["Debt/Equity"] = float(de)

    # Cash, Debt, Net Debt
    tc = info.get("totalCash")
    if tc is not None:
        out["Total cash"] = float(tc)
    td = info.get("totalDebt")
    if td is not None:
        out["Total debt"] = float(td)
    nd = info.get("netDebt")
    if nd is not None:
        out["Net debt"] = float(nd)

    # Cashflow & CapEx & FCF
    ocf = info.get("operatingCashflow")
    if ocf is not None:
        out["OCF"] = float(ocf)
    capex = info.get("capitalExpenditures")
    if capex is not None:
        out["CapEx"] = float(capex)
    fcf = info.get("freeCashflow")
    if fcf is not None:
        out["FCF"] = float(fcf)

    # Dividend yield (%)
    dy = info.get("dividendYield")
    if dy is not None:
        out["Dividend yield (%)"] = _fmt_pct(float(dy) * 100.0)

    # EBITDA (för EV/EBITDA)
    ebitda = info.get("ebitda")
    if ebitda is not None:
        out["_EBITDA_TTM"] = float(ebitda)

    return out

# ---- FMP Fallback/Komplement ------------------------------------------------

def _fmp_get(path: str, params=None) -> Tuple[Optional[dict], int]:
    key = st.secrets.get("FMP_API_KEY", "")
    base = st.secrets.get("FMP_BASE", "https://financialmodelingprep.com")
    p = (params or {}).copy()
    if key:
        p["apikey"] = key
    url = f"{base}/{path}"
    try:
        r = requests.get(url, params=p, timeout=25)
        sc = r.status_code
        if 200 <= sc < 300:
            try:
                j = r.json()
            except Exception:
                j = None
            return j, sc
        return None, sc
    except Exception:
        return None, 0

def _fmp_first(lst: Optional[list]) -> dict:
    if isinstance(lst, list) and lst:
        return lst[0] or {}
    return {}

def fetch_fmp_metrics(ticker: str) -> Dict[str, float]:
    """
    Plockar nyckeltal från FMP som komplement.
    """
    out: Dict[str, float] = {}
    sym = str(ticker).upper().strip()

    # profile -> mcap, shares, currency, companyName
    prof, scp = _fmp_get(f"api/v3/profile/{sym}")
    p0 = _fmp_first(prof)
    if p0:
        if p0.get("price") is not None and "Aktuell kurs" not in out:
            try: out["Aktuell kurs"] = float(p0["price"])
            except: pass
        if p0.get("currency") and "Valuta" not in out:
            out["Valuta"] = str(p0["currency"]).upper()
        if p0.get("companyName") and "Bolagsnamn" not in out:
            out["Bolagsnamn"] = p0["companyName"]
        if p0.get("mktCap") is not None and "Market Cap" not in out:
            try: out["Market Cap"] = float(p0["mktCap"])
            except: pass
        if p0.get("sharesOutstanding") is not None and "Utestående aktier" not in out:
            try: out["Utestående aktier"] = float(p0["sharesOutstanding"]) / 1e6
            except: pass
        if p0.get("sector"): out["Sector"] = p0["sector"]
        if p0.get("industry"): out["Industry"] = p0["industry"]

    # key-metrics-ttm -> EV/EBITDA ttm, P/B ttm
    kttm, sck = _fmp_get(f"api/v3/key-metrics-ttm/{sym}")
    k0 = _fmp_first(kttm)
    if k0:
        v = k0.get("evToEbitdaTTM") or k0.get("enterpriseValueOverEBITDATTM")
        if v is not None:
            out["EV/EBITDA (ttm)"] = float(v)
        pbttm = k0.get("pbRatioTTM") or k0.get("priceToBookTTM")
        if pbttm is not None and "P/B" not in out:
            out["P/B"] = float(pbttm)

    # ratios-ttm -> margins, ROE, Debt/Equity, P/S
    rttm, scr = _fmp_get(f"api/v3/ratios-ttm/{sym}")
    r0 = _fmp_first(rttm)
    if r0:
        if r0.get("grossProfitMarginTTM") is not None and "Gross margin (%)" not in out:
            out["Gross margin (%)"] = _fmt_pct(float(r0["grossProfitMarginTTM"]) * 100.0)
        if r0.get("operatingProfitMarginTTM") is not None and "Operating margin (%)" not in out:
            out["Operating margin (%)"] = _fmt_pct(float(r0["operatingProfitMarginTTM"]) * 100.0)
        if r0.get("netProfitMarginTTM") is not None and "Net margin (%)" not in out:
            out["Net margin (%)"] = _fmt_pct(float(r0["netProfitMarginTTM"]) * 100.0)
        if r0.get("returnOnEquityTTM") is not None and "ROE (%)" not in out:
            out["ROE (%)"] = _fmt_pct(float(r0["returnOnEquityTTM"]) * 100.0)
        if r0.get("debtEquityRatioTTM") is not None and "Debt/Equity" not in out:
            out["Debt/Equity"] = float(r0["debtEquityRatioTTM"])
        # P/S TTM (kan användas som "P/S" om saknas)
        ps_ttm = r0.get("priceToSalesRatioTTM") or r0.get("priceToSalesTTM")
        if ps_ttm is not None and "P/S" not in out:
            out["P/S"] = float(ps_ttm)

    # income-statement-ttm -> EBITDA
    isttm, sci = _fmp_get(f"api/v3/income-statement-ttm/{sym}")
    i0 = _fmp_first(isttm)
    if i0 and i0.get("ebitdaTTM") is not None and "_EBITDA_TTM" not in out:
        out["_EBITDA_TTM"] = float(i0["ebitdaTTM"])

    # enterprise-values -> EV
    evs, sce = _fmp_get(f"api/v3/enterprise-values/{sym}", {"period": "quarter", "limit": 1})
    e0 = _fmp_first(evs)
    if e0:
        if e0.get("enterpriseValue") is not None and "Enterprise Value" not in out:
            out["Enterprise Value"] = float(e0["enterpriseValue"])

    # cash-flow-statement (annual, senaste)
    cfa, scf = _fmp_get(f"api/v3/cash-flow-statement/{sym}", {"period": "annual", "limit": 1})
    cf0 = _fmp_first(cfa)
    if cf0:
        if cf0.get("operatingCashFlow") is not None and "OCF" not in out:
            out["OCF"] = float(cf0["operatingCashFlow"])
        if cf0.get("capitalExpenditure") is not None and "CapEx" not in out:
            out["CapEx"] = float(cf0["capitalExpenditure"])
        if cf0.get("freeCashFlow") is not None and "FCF" not in out:
            out["FCF"] = float(cf0["freeCashFlow"])
        if cf0.get("dividendsPaid") is not None:
            # notera: dividendsPaid är oftast negativt i FMP
            out["_DividendsPaid"] = float(cf0["dividendsPaid"])

    # balance-sheet-statement (annual, senaste)
    bsa, scb = _fmp_get(f"api/v3/balance-sheet-statement/{sym}", {"period": "annual", "limit": 1})
    bs0 = _fmp_first(bsa)
    if bs0:
        if bs0.get("cashAndShortTermInvestments") is not None and "Total cash" not in out:
            out["Total cash"] = float(bs0["cashAndShortTermInvestments"])
        if bs0.get("totalDebt") is not None and "Total debt" not in out:
            out["Total debt"] = float(bs0["totalDebt"])
        if bs0.get("netDebt") is not None and "Net debt" not in out:
            out["Net debt"] = float(bs0["netDebt"])

    return out

# ---- P/S historik & MktCap historik ----------------------------------------

def _implied_shares(price: float, mcap: float, yahoo_shares: float) -> float:
    """
    Beräkna implicita antal aktier från mcap/price – annars använd Yahoo sharesOutstanding.
    Returnerar antal aktier (styck).
    """
    price = _safe_float(price, 0.0)
    mcap = _safe_float(mcap, 0.0)
    shares = 0.0
    if mcap > 0 and price > 0:
        shares = mcap / price
    elif _pos(yahoo_shares):
        shares = float(yahoo_shares)
    return shares

def compute_ps_series_and_mcap(
    ticker: str,
    price_now: float,
    mcap_now: float,
    yahoo_shares: float,
    price_ccy: str
) -> Dict[str, float]:
    """
    Räknar P/S TTM nu + P/S Q1–Q4 via Yahoo quarterly revenues + historiska priser.
    Returnerar även MktCap Q1–Q4 (beräknat med konstanta shares = implied/yahoo_shares).
    """
    out: Dict[str, float] = {}
    q_rows = _yahoo_quarterly_revenues(ticker)
    if len(q_rows) < 4:
        return out

    # Bygg TTM-fönster (5 st så vi säkert får Q1–Q4 nyast → äldre)
    ttm_list = _ttm_windows(q_rows, need=5)  # [(date, ttm)], nyast→äldst

    # Antal aktier (styck)
    shares_used = _implied_shares(price_now, mcap_now, yahoo_shares)

    # Market cap nu
    if (not _pos(mcap_now)) and _pos(price_now) and _pos(shares_used):
        mcap_now = price_now * shares_used

    # P/S nu
    if _pos(mcap_now) and ttm_list:
        ltm_now = ttm_list[0][1]
        if _pos(ltm_now):
            out["P/S"] = float(mcap_now / ltm_now)

    # P/S Q1–Q4 + MktCap Q1–Q4
    if _pos(shares_used) and ttm_list:
        q_dates = [d for (d, _) in ttm_list[:4]]
        px_map = _yahoo_prices_for_dates(ticker, q_dates)
        for idx, (d_end, ttm_rev) in enumerate(ttm_list[:4], start=1):
            px = px_map.get(d_end)
            if _pos(px) and _pos(ttm_rev):
                mcap_hist = shares_used * float(px)
                out[f"MktCap Q{idx}"] = float(mcap_hist)
                out[f"P/S Q{idx}"] = float(mcap_hist / ttm_rev)

    # P/S-snitt
    ps_vals = [out.get(f"P/S Q{i}", 0.0) for i in range(1, 5)]
    ps_clean = [float(x) for x in ps_vals if float(x) > 0]
    out["P/S-snitt"] = round(float(np.mean(ps_clean)), 2) if ps_clean else float(out.get("P/S", 0.0))

    return out

# ---- Deriverade nyckeltal ---------------------------------------------------

def compute_derived_metrics(vals: Dict[str, float]) -> Dict[str, float]:
    """
    Beräknar:
    - EV/EBITDA (ttm) om EV & EBITDA finns
    - Net debt / EBITDA
    - FCF Yield (%) = FCF / MarketCap * 100
    - Dividend payout (FCF) (%) = (utdelningar / FCF) * 100 (klamrad)
    """
    out = {}

    ev = _safe_float(vals.get("Enterprise Value"))
    ebitda = _safe_float(vals.get("_EBITDA_TTM"))
    if _pos(ev) and _pos(ebitda):
        out["EV/EBITDA (ttm)"] = ev / ebitda

    net_debt = _safe_float(vals.get("Net debt"))
    if _pos(ebitda):
        out["Net debt / EBITDA"] = net_debt / ebitda

    mcap = _safe_float(vals.get("Market Cap"))
    fcf = _safe_float(vals.get("FCF"))
    if _pos(mcap) and fcf != 0:
        out["FCF Yield (%)"] = _fmt_pct((fcf / mcap) * 100.0)

    div_paid = _safe_float(vals.get("_DividendsPaid"))
    # FMP: dividendsPaid är oftast negativt. Vi vill ha beloppet i positivt tecken.
    div_amt = abs(div_paid)
    if fcf != 0:
        payout = max(0.0, (div_amt / abs(fcf)) * 100.0)
        out["Dividend payout (FCF) (%)"] = _fmt_pct(payout)

    # Om Yahoo hade Dividend yield (%), behåll den. Annars försök räkna via DividendsPaid / MarketCap (proxy)
    if "Dividend yield (%)" not in vals and _pos(mcap) and div_amt > 0:
        out["Dividend yield (%)"] = _fmt_pct((div_amt / mcap) * 100.0)

    return out

# ---- Publika API ------------------------------------------------------------

def run_price_only(ticker: str) -> Tuple[Dict[str, float], Dict]:
    """
    Lätt uppdatering: Aktuell kurs (+ ev. Market Cap via Yahoo) för en ticker.
    """
    dbg = {"src": "price-only"}
    vals: Dict[str, float] = {}

    y = fetch_yahoo_basics(ticker)
    for k in ["Bolagsnamn","Valuta","Aktuell kurs","Market Cap","Utestående aktier","Sector","Industry"]:
        if k in y and y[k] not in (None, "", 0, 0.0):
            vals[k] = y[k]

    dbg["yahoo_ok"] = True if vals else False
    return vals, dbg

def run_full_update(ticker: str) -> Tuple[Dict[str, float], Dict]:
    """
    Full uppdatering för en ticker:
      1) Yahoo bas + många nyckeltal
      2) FMP kompletterar
      3) P/S Q1–Q4 + MktCap Q1–Q4 via Yahoo quarterly + prisdata
      4) Härledda nyckeltal (EV/EBITDA, NetDebt/EBITDA, FCF Yield, Dividend payout FCF)
    Returnerar (vals, debug)
    """
    dbg = {"ticker": ticker, "ts": _now_stamp()}
    vals: Dict[str, float] = {}

    # 1) Yahoo
    try:
        y = fetch_yahoo_basics(ticker)
        vals.update({k: v for k, v in y.items() if v not in (None, "", 0, 0.0) or k in ["Sector","Industry"]})
        dbg["yahoo"] = {k: vals.get(k) for k in ["Bolagsnamn","Valuta","Aktuell kurs","Market Cap","Utestående aktier","Sector","Industry"]}
    except Exception as e:
        dbg["yahoo_err"] = str(e)

    # 2) FMP komplettering
    try:
        f = fetch_fmp_metrics(ticker)
        for k, v in f.items():
            # skriv bara in om saknas eller 0
            if k not in vals or (isinstance(v, (int,float)) and float(vals.get(k,0.0)) == 0.0):
                vals[k] = v
        dbg["fmp_keys"] = list(f.keys())[:10]  # preview
    except Exception as e:
        dbg["fmp_err"] = str(e)

    # 3) P/S-serier & MktCap historik
    try:
        px = _safe_float(vals.get("Aktuell kurs"))
        mcap = _safe_float(vals.get("Market Cap"))
        shares_y = _safe_float(vals.get("Utestående aktier")) * 1e6 if _pos(vals.get("Utestående aktier", 0.0)) else 0.0
        ccy = str(vals.get("Valuta") or "USD").upper()
        series = compute_ps_series_and_mcap(ticker, px, mcap, shares_y, ccy)
        vals.update(series)
    except Exception as e:
        dbg["ps_series_err"] = str(e)

    # 4) Härledda nyckeltal
    try:
        derived = compute_derived_metrics(vals)
        vals.update(derived)
    except Exception as e:
        dbg["derived_err"] = str(e)

    return vals, dbg
