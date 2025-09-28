# -*- coding: utf-8 -*-
"""
stockapp/sources.py

Runner-funktioner för att hämta data per ticker.

- run_update_price_only(ticker): snabb uppdatering av enbart pris/valuta/namn (Yahoo)
- run_update_full(ticker): robust heluppdatering (Yahoo + SEC-fallback) med P/S-historik, Mcap-historik,
  D/E, marginaler, FCF, Kassa, EV/EBITDA, Sektor/Industri, CAGR 5y, m.m.
"""

from __future__ import annotations
from typing import Dict, Tuple, Any, Optional, List
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import time
import os

# ---------------------------
# Konfiguration
# ---------------------------
SEC_USER_AGENT = os.environ.get(
    "SEC_USER_AGENT",
    "StockApp/1.0 (contact: your-email@example.com)"
)

# ---------------------------
# Små hjälpare
# ---------------------------
def _num(x, default=0.0) -> float:
    try:
        if x is None or (isinstance(x, str) and not x.strip()):
            return float(default)
        return float(x)
    except Exception:
        return float(default)

def _safe_div(a, b, default=0.0) -> float:
    a = _num(a, 0.0); b = _num(b, 0.0)
    if b == 0:
        return float(default)
    return float(a) / float(b)

def _fmt_date_idx(col) -> Optional[pd.Timestamp]:
    try:
        if hasattr(col, "to_pydatetime"):
            return pd.to_datetime(col)
        return pd.to_datetime(str(col))
    except Exception:
        return None

def _ttm_windows(values: List[Tuple[pd.Timestamp, float]], need: int = 4) -> List[Tuple[pd.Timestamp, float]]:
    """
    Tar [(end_date, kvartalsintäkt), ...] (nyast→äldst) och bygger upp till 'need' TTM-summor:
    [(end_date0, ttm0), (end_date1, ttm1), ...].
    """
    out = []
    if len(values) < 4:
        return out
    for i in range(0, min(need, len(values) - 3)):
        end_i = values[i][0]
        ttm_i = sum(v for (_, v) in values[i:i+4])
        out.append((end_i, float(ttm_i)))
    return out

def _yahoo_prices_for_dates(ticker: str, dates: List[pd.Timestamp]) -> Dict[pd.Timestamp, float]:
    """
    Hämtar Closing price för datum (eller närmast före) i ett fönster.
    """
    if not dates:
        return {}
    dmin = min(dates) - pd.Timedelta(days=14)
    dmax = max(dates) + pd.Timedelta(days=2)
    try:
        t = yf.Ticker(ticker)
        hist = t.history(start=dmin, end=dmax, interval="1d")
        if hist is None or hist.empty:
            return {}
        hist = hist.sort_index()
        out = {}
        idx = list(hist.index)
        closes = list(hist["Close"].values)
        for d in dates:
            px = None
            # hitta närmast <= d
            for j in range(len(idx)-1, -1, -1):
                if idx[j].to_pydatetime().date() <= d.date():
                    try: px = float(closes[j])
                    except: px = None
                    break
            if px is not None:
                out[d] = px
        return out
    except Exception:
        return {}

# ---------------------------
# Yahoo-block
# ---------------------------

def _yahoo_basics(ticker: str) -> Dict[str, Any]:
    """
    Hämtar pris, valuta, namn, dividendRate, sector, industry samt marketCap, enterpriseValue, ebitda.
    """
    out = {
        "Bolagsnamn": "",
        "Aktuell kurs": 0.0,
        "Valuta": "USD",
        "Årlig utdelning": 0.0,
        "Utdelningsyield": 0.0,
        "Sektor": "",
        "Industri": "",
        "_marketCap": 0.0,
        "_enterpriseValue": 0.0,
        "_ebitda": 0.0,
    }
    t = yf.Ticker(ticker)
    info = {}
    try:
        info = t.info or {}
    except Exception:
        info = {}

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

    # Valuta
    cur = info.get("currency")
    if cur:
        out["Valuta"] = str(cur).upper()

    # Namn
    nm = info.get("shortName") or info.get("longName") or ""
    out["Bolagsnamn"] = str(nm)

    # Utdelning
    div_rate = info.get("dividendRate")
    try:
        if div_rate is not None:
            out["Årlig utdelning"] = float(div_rate)
    except Exception:
        pass

    # Yield
    try:
        if out["Årlig utdelning"] > 0 and _num(out["Aktuell kurs"])>0:
            out["Utdelningsyield"] = (out["Årlig utdelning"] / out["Aktuell kurs"]) * 100.0
    except Exception:
        pass

    # Sector/Industry
    out["Sektor"] = str(info.get("sector") or "")
    out["Industri"] = str(info.get("industry") or "")

    # EV/EBITDA underlag
    try:
        if info.get("marketCap") is not None:
            out["_marketCap"] = float(info.get("marketCap"))
    except Exception:
        pass
    try:
        if info.get("enterpriseValue") is not None:
            out["_enterpriseValue"] = float(info.get("enterpriseValue"))
    except Exception:
        pass
    try:
        if info.get("ebitda") is not None:
            out["_ebitda"] = float(info.get("ebitda"))
    except Exception:
        pass

    return out

def _yahoo_quarterly_revenues(ticker: str) -> List[Tuple[pd.Timestamp, float]]:
    """
    Kvartalsintäkter (nyast → äldst) från Yahoo.
    """
    t = yf.Ticker(ticker)
    # Först försök quarterly_financials
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
                        d = _fmt_date_idx(c)
                        if d is not None:
                            try:
                                out.append((d, float(v)))
                            except Exception:
                                pass
                    out.sort(key=lambda x: x[0], reverse=True)
                    return out
    except Exception:
        pass

    # fallback income_stmt kvartalsvis
    try:
        df_is = getattr(t, "income_stmt", None)
        if isinstance(df_is, pd.DataFrame) and not df_is.empty and "Total Revenue" in df_is.index:
            ser = df_is.loc["Total Revenue"].dropna()
            out = []
            for c, v in ser.items():
                d = _fmt_date_idx(c)
                if d is not None:
                    try:
                        out.append((d, float(v)))
                    except Exception:
                        pass
            out.sort(key=lambda x: x[0], reverse=True)
            return out
    except Exception:
        pass

    return []

def _yahoo_annual_revenues(ticker: str) -> List[Tuple[pd.Timestamp, float]]:
    """
    Årsintäkter (äldst→nyast) för CAGR.
    """
    t = yf.Ticker(ticker)
    try:
        af = t.financials  # annual
        if isinstance(af, pd.DataFrame) and not af.empty and "Total Revenue" in af.index:
            ser = af.loc["Total Revenue"].dropna()
            out = []
            for c, v in ser.items():
                d = _fmt_date_idx(c)
                if d is not None:
                    out.append((d, float(v)))
            out.sort(key=lambda x: x[0])  # äldst→nyast
            return out
    except Exception:
        pass
    return []

def _calc_cagr_5y_from_annual(rev_series: List[Tuple[pd.Timestamp, float]]) -> float:
    """
    Enkel CAGR >≈5 år (eller mellan äldsta och senaste om <5 datapunkter).
    """
    if not rev_series or len(rev_series) < 2:
        return 0.0
    # använd första och sista
    start = float(rev_series[0][1]); end = float(rev_series[-1][1])
    n_years = max(1, len(rev_series)-1)
    if start <= 0:
        return 0.0
    cagr = (end/start) ** (1.0/n_years) - 1.0
    return round(cagr * 100.0, 2)

# ---------------------------
# SEC-block (aktier & kvartalsintäkter)
# ---------------------------

def _sec_get(url: str, params=None):
    try:
        r = requests.get(url, params=params or {}, headers={"User-Agent": SEC_USER_AGENT}, timeout=30)
        if r.status_code == 200:
            return r.json(), 200
        return None, r.status_code
    except Exception:
        return None, 0

def _sec_ticker_map():
    j, sc = _sec_get("https://www.sec.gov/files/company_tickers.json")
    if not isinstance(j, dict):
        return {}
    out = {}
    for _, v in j.items():
        try:
            out[str(v["ticker"]).upper()] = str(v["cik_str"]).zfill(10)
        except Exception:
            pass
    return out

def _sec_cik_for(ticker: str) -> Optional[str]:
    return _sec_ticker_map().get(str(ticker).upper())

def _parse_iso(d: str) -> Optional[pd.Timestamp]:
    try:
        return pd.to_datetime(d)
    except Exception:
        return None

def _is_instant_entry(it: dict) -> bool:
    end = it.get("end"); start = it.get("start")
    if not end:
        return False
    if not start:
        return True
    d1 = _parse_iso(str(start)); d2 = _parse_iso(str(end))
    if d1 is not None and d2 is not None:
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
                    end = _parse_iso(str(it.get("end","")))
                    val = it.get("val", None)
                    if end is not None and val is not None:
                        try:
                            v = float(val)
                            entries.append({"end": end, "val": v})
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

def _sec_companyfacts(cik10: str):
    return _sec_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json")

def _sec_quarterly_revenues_dated_with_unit(facts: dict, max_quarters: int = 20):
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
                    if not (end is not None and start is not None and val is not None):
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
                ded = {}
                for end, v in tmp:
                    ded[end] = v
                rows = sorted(ded.items(), key=lambda t: t[0], reverse=True)[:max_quarters]
                if rows:
                    return rows, unit_code
    return [], None

# ---------------------------
# Publika runners
# ---------------------------

def run_update_price_only(ticker: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Snabb uppdatering: pris/valuta/namn/utdelning/yield/sector/industry.
    Returnerar (values, debug)
    """
    vals = {}
    dbg = {"runner": "price-only", "ticker": ticker}
    try:
        y = _yahoo_basics(ticker)
        for k in ["Bolagsnamn","Valuta","Aktuell kurs","Årlig utdelning","Utdelningsyield","Sektor","Industri"]:
            if y.get(k) not in (None, "", 0, 0.0):
                vals[k] = y[k]
        vals["Senast uppdaterad källa"] = "Auto (Yahoo, price-only)"
    except Exception as e:
        dbg["error"] = str(e)
    return vals, dbg

def run_update_full(ticker: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Full uppdatering:
    - Yahoo basics
    - Shares: implied via mcap/price → SEC instant fallback
    - SEC kvartalsintäkter → TTM → P/S (nu) + P/S Q1–Q4 (historik)
    - Mcap Q1–Q4 historik (via samma shares & pris på datum)
    - CAGR 5 år (annual revenues Yahoo)
    - D/E, marginaler, EV/EBITDA, Kassa, FCF
    Returnerar (values, debug)
    """
    vals: Dict[str, Any] = {}
    dbg: Dict[str, Any] = {"runner": "full", "ticker": ticker}

    # --- Yahoo basics
    y = _yahoo_basics(ticker)
    dbg["yahoo_basics"] = {k: y.get(k) for k in ["Aktuell kurs","Valuta","Bolagsnamn","_marketCap"]}
    for k in ["Bolagsnamn","Valuta","Aktuell kurs","Årlig utdelning","Utdelningsyield","Sektor","Industri"]:
        if y.get(k) not in (None, "", 0, 0.0):
            vals[k] = y[k]

    price_now = _num(y.get("Aktuell kurs",0.0))
    mcap_now = _num(y.get("_marketCap",0.0))
    px_ccy = str(y.get("Valuta") or "USD").upper()

    # --- SEC shares + revenues
    cik = _sec_cik_for(ticker)
    facts = None; sc = None
    if cik:
        facts, sc = _sec_companyfacts(cik)
        dbg["sec_sc"] = sc

    # shares implied
    implied_shares = 0.0
    if mcap_now > 0 and price_now > 0:
        implied_shares = mcap_now / price_now
        dbg["shares_source"] = "implied(mcap/price)"

    # SEC robust instant shares fallback
    if (not implied_shares or implied_shares <= 0) and isinstance(facts, dict):
        sec_sh = _sec_latest_shares_robust(facts)
        if sec_sh > 0:
            implied_shares = sec_sh
            dbg["shares_source"] = "SEC instant robust"

    if implied_shares > 0:
        vals["Utestående aktier"] = implied_shares / 1_000_000.0

    # Quarterly revenues via SEC (med unit), annars Yahoo
    q_rows = []
    unit = None
    if isinstance(facts, dict):
        q_rows, unit = _sec_quarterly_revenues_dated_with_unit(facts, max_quarters=20)
        dbg["sec_revenue_unit"] = unit

    if not q_rows:
        # Yahoo fallback
        yq = _yahoo_quarterly_revenues(ticker)
        if yq:
            q_rows = yq
            unit = px_ccy

    # TTM list
    ttm_list = []
    if q_rows:
        ttm_list = _ttm_windows(q_rows, need=4)

    # P/S nu & historik
    if implied_shares > 0 and ttm_list:
        # priser på TTM-datum
        dates = [d for (d, _) in ttm_list[:4]]
        px_map = _yahoo_prices_for_dates(ticker, dates)

        # Market cap historik & P/S Qx
        for idx, (d_end, ttm_rev) in enumerate(ttm_list[:4], start=1):
            px = _num(px_map.get(d_end, 0.0))
            if px > 0 and ttm_rev > 0:
                mcap_hist = implied_shares * px
                vals[f"Mcap Q{idx}"] = float(mcap_hist)
                vals[f"P/S Q{idx}"] = float(mcap_hist / ttm_rev)

        # P/S nu (enklast via mcap_now / ltm_now)
        ltm_now = _num(ttm_list[0][1], 0.0)
        if mcap_now > 0 and ltm_now > 0:
            vals["P/S"] = float(mcap_now / ltm_now)

    # CAGR 5 år från annual revenues
    ann = _yahoo_annual_revenues(ticker)
    vals["CAGR 5 år (%)"] = _calc_cagr_5y_from_annual(ann) if ann else 0.0

    # --- EV/EBITDA
    ev = _num(y.get("_enterpriseValue",0.0))
    ebitda = _num(y.get("_ebitda",0.0))
    if ev > 0 and ebitda > 0:
        vals["EV/EBITDA"] = float(ev / ebitda)

    # --- Balans & kassaflöde
    # D/E, Kassa, FCF (TTM)
    t = yf.Ticker(ticker)

    # Balansräkning (quarterly eller annual)
    total_debt = None
    total_equity = None
    total_cash = None
    try:
        qb = t.quarterly_balance_sheet
        if isinstance(qb, pd.DataFrame) and not qb.empty:
            # Senaste kolumn
            col = qb.columns[0]
            if "Total Debt" in qb.index:
                total_debt = _num(qb.loc["Total Debt", col])
            elif "TotalDebt" in qb.index:
                total_debt = _num(qb.loc["TotalDebt", col])
            if "Total Stockholder Equity" in qb.index:
                total_equity = _num(qb.loc["Total Stockholder Equity", col])
            elif "TotalStockholderEquity" in qb.index:
                total_equity = _num(qb.loc["TotalStockholderEquity", col])
            if "Cash And Cash Equivalents" in qb.index:
                total_cash = _num(qb.loc["Cash And Cash Equivalents", col])
            elif "CashAndCashEquivalents" in qb.index:
                total_cash = _num(qb.loc["CashAndCashEquivalents", col])
    except Exception:
        pass
    if total_debt is None or total_equity is None or total_cash is None:
        try:
            ab = t.balance_sheet
            if isinstance(ab, pd.DataFrame) and not ab.empty:
                col = ab.columns[0]
                if total_debt is None:
                    if "Total Debt" in ab.index:
                        total_debt = _num(ab.loc["Total Debt", col])
                    elif "TotalDebt" in ab.index:
                        total_debt = _num(ab.loc["TotalDebt", col])
                if total_equity is None:
                    if "Total Stockholder Equity" in ab.index:
                        total_equity = _num(ab.loc["Total Stockholder Equity", col])
                    elif "TotalStockholderEquity" in ab.index:
                        total_equity = _num(ab.loc["TotalStockholderEquity", col])
                if total_cash is None:
                    if "Cash And Cash Equivalents" in ab.index:
                        total_cash = _num(ab.loc["Cash And Cash Equivalents", col])
                    elif "CashAndCashEquivalents" in ab.index:
                        total_cash = _num(ab.loc["CashAndCashEquivalents", col])
        except Exception:
            pass

    if total_debt is not None and total_equity and total_equity != 0:
        vals["Debt/Equity"] = float(total_debt / total_equity)
    if total_cash is not None:
        vals["Kassa"] = float(total_cash)

    # FCF (TTM): försök via quarterly_cashflow “Free Cash Flow”, annars CFO - CapEx
    fcf_ttm = None
    try:
        qcf = t.quarterly_cashflow
        if isinstance(qcf, pd.DataFrame) and not qcf.empty:
            # summera sista 4 kvartal
            if "Free Cash Flow" in qcf.index:
                ser = qcf.loc["Free Cash Flow"].dropna()
                if not ser.empty:
                    fcf_ttm = float(ser.iloc[:4].sum())
            if fcf_ttm is None:
                # CFO - CapEx
                cfo = qcf.loc["Total Cash From Operating Activities"].dropna() if "Total Cash From Operating Activities" in qcf.index else None
                capex = qcf.loc["Capital Expenditures"].dropna() if "Capital Expenditures" in qcf.index else None
                if cfo is not None and capex is not None:
                    fcf_ttm = float(cfo.iloc[:4].sum() - capex.iloc[:4].sum())
    except Exception:
        pass
    if fcf_ttm is None:
        try:
            acf = t.cashflow
            if isinstance(acf, pd.DataFrame) and not acf.empty:
                if "Free Cash Flow" in acf.index:
                    ser = acf.loc["Free Cash Flow"].dropna()
                    if not ser.empty:
                        fcf_ttm = float(ser.iloc[0])  # senaste år
                if fcf_ttm is None:
                    cfo = acf.loc["Total Cash From Operating Activities"].dropna() if "Total Cash From Operating Activities" in acf.index else None
                    capex = acf.loc["Capital Expenditures"].dropna() if "Capital Expenditures" in acf.index else None
                    if cfo is not None and capex is not None and not cfo.empty and not capex.empty:
                        fcf_ttm = float(cfo.iloc[0] - capex.iloc[0])
        except Exception:
            pass
    if fcf_ttm is not None:
        vals["FCF"] = float(fcf_ttm)

    # Marginaler (brutto/profit) via info om tillgängligt (proportioner)
    # yfinance.info ofta i decimaltal (0.65 ~ 65%)
    # Försök tolka båda fall (0.65 eller 65)
    def _as_pct(v):
        x = _num(v, 0.0)
        if x <= 0:
            return 0.0
        return float(x*100.0) if x < 1.0 else float(x)

    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        info = {}
    gm = info.get("grossMargins", None)
    pm = info.get("profitMargins", None)
    if gm is not None:
        vals["Bruttomarginal"] = round(_as_pct(gm), 2)
    if pm is not None:
        vals["Nettomarginal"] = round(_as_pct(pm), 2)

    # Slutlig källa-info
    vals["Senast uppdaterad källa"] = "Auto (Yahoo+SEC)"

    return vals, dbg
