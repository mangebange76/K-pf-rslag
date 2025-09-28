# stockapp/sources.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, Tuple, List, Any
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import date

# -------------------------------------------------
# Små utils (rena, utan Streamlit-beroenden)
# -------------------------------------------------

def _safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

def _human_err(e: Exception) -> str:
    return f"{type(e).__name__}: {e}"

def _fx_rate(base: str, quote: str) -> float:
    """Frankfurter → exchangerate.host → 1.0 fallback."""
    base = (base or "").upper()
    quote = (quote or "").upper()
    if not base or not quote or base == quote:
        return 1.0
    # Frankfurter (ECB)
    try:
        r = requests.get("https://api.frankfurter.app/latest",
                         params={"from": base, "to": quote}, timeout=12)
        if r.status_code == 200:
            v = (r.json() or {}).get("rates", {}).get(quote)
            if v:
                return float(v)
    except Exception:
        pass
    # exchangerate.host
    try:
        r = requests.get("https://api.exchangerate.host/latest",
                         params={"base": base, "symbols": quote}, timeout=12)
        if r.status_code == 200:
            v = (r.json() or {}).get("rates", {}).get(quote)
            if v:
                return float(v)
    except Exception:
        pass
    return 1.0

def _quarter_key(d: date) -> Tuple[int, int]:
    """Nyckel för 'fiscal quarter' – hanterar dec/jan-gap som Q4."""
    # Enkel kalender-Q (funkar väl med Yahoo:s datumstämplar)
    m = d.month
    if m in (1,):
        q = 4  # behandla jan som Q4-fönstrets 'late post'
    elif m <= 3:
        q = 1
    elif m <= 6:
        q = 2
    elif m <= 9:
        q = 3
    else:
        q = 4
    # år: om jan->Q4, bind till föregående år
    y = d.year if m != 1 else d.year - 1
    return (y, q)

def _dedupe_quarters(rows: List[Tuple[date, float]]) -> List[Tuple[date, float]]:
    """
    Tar [(end_date, value)] och behåller *senaste* post per (year, quarter).
    Fixar vanliga "dec/jan"-duplikat.
    """
    seen = {}
    for d, v in rows:
        key = _quarter_key(d)
        # behåll senast datum i samma kvartal
        if key not in seen or d > seen[key][0]:
            seen[key] = (d, float(v))
    out = list(seen.values())
    out.sort(key=lambda t: t[0], reverse=True)
    return out

def _ttm_windows(values: List[Tuple[date, float]], need: int = 4) -> List[Tuple[date, float]]:
    """
    Från kvartalsvärden (nyast→äldst) bygger TTM-summor:
    [(end_date0, ttm0), (end_date1, ttm1), ...]
    """
    out = []
    if len(values) < 4:
        return out
    for i in range(0, min(need, len(values) - 3)):
        end_i = values[i][0]
        ttm_i = sum(v for (_, v) in values[i:i+4])
        out.append((end_i, float(ttm_i)))
    return out

def _yahoo_prices_for_dates(ticker: str, dates: List[date]) -> Dict[date, float]:
    """Dagliga stängningar – pris på/närmast före respektive datum."""
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
        out: Dict[date, float] = {}
        idx = list(hist.index.date)
        closes = list(hist["Close"].values)
        for d in dates:
            px = None
            for j in range(len(idx)-1, -1, -1):
                if idx[j] <= d:
                    px = float(closes[j]); break
            if px is not None:
                out[d] = px
        return out
    except Exception:
        return {}

def _quarterly_revenues_yf(t: yf.Ticker) -> Tuple[List[Tuple[date, float]], str]:
    """
    Hämtar kvartalsintäkter (nyast→äldst) och returnerar [(date, value)], unit.
    Försöker först quarterly_financials, annars income_stmt.
    """
    # 1) quarterly_financials
    try:
        qf = t.quarterly_financials
        if isinstance(qf, pd.DataFrame) and not qf.empty:
            # Leta 'Total Revenue' (olika label-varianter förekommer)
            idx = [str(x).strip() for x in qf.index]
            candidates = [
                "Total Revenue","TotalRevenue","Revenues","Revenue",
                "Sales","SalesRevenueNet","Revenues from contracts with customers"
            ]
            for key in candidates:
                if key in idx:
                    row = qf.loc[key].dropna()
                    tmp = []
                    for c, v in row.items():
                        try:
                            d = c.date() if hasattr(c, "date") else pd.to_datetime(c).date()
                            tmp.append((d, float(v)))
                        except Exception:
                            pass
                    tmp = _dedupe_quarters(sorted(tmp, key=lambda x: x[0], reverse=True))
                    if tmp:
                        # valuta (financialCurrency om möjligt)
                        info = t.info or {}
                        unit = str(info.get("financialCurrency") or info.get("currency") or "USD").upper()
                        return tmp, unit
    except Exception:
        pass

    # 2) fallback: income_stmt (kvartal – kan vara tomt beroende på yfinance-version)
    try:
        df_is = getattr(t, "income_stmt", None)
        if isinstance(df_is, pd.DataFrame) and not df_is.empty and "Total Revenue" in df_is.index:
            ser = df_is.loc["Total Revenue"].dropna()
            tmp = []
            for c, v in ser.items():
                try:
                    d = c.date() if hasattr(c, "date") else pd.to_datetime(c).date()
                    tmp.append((d, float(v)))
                except Exception:
                    pass
            tmp = _dedupe_quarters(sorted(tmp, key=lambda x: x[0], reverse=True))
            if tmp:
                info = t.info or {}
                unit = str(info.get("financialCurrency") or info.get("currency") or "USD").upper()
                return tmp, unit
    except Exception:
        pass

    return [], None

def _implied_shares(info: Dict[str, Any], px: float) -> float:
    """Försök räkna implied shares (styck) från marketCap/price, annars sharesOutstanding."""
    mcap = _safe_float(info.get("marketCap"), 0.0)
    if mcap > 0 and px > 0:
        return mcap / px
    so = _safe_float(info.get("sharesOutstanding"), 0.0)
    return so if so > 0 else 0.0

def _balance_metrics(t: yf.Ticker) -> Dict[str, float]:
    """Debt/Equity, Cash, etc. från balansräkning."""
    out: Dict[str, float] = {}
    try:
        bsq = t.quarterly_balance_sheet
        if isinstance(bsq, pd.DataFrame) and not bsq.empty:
            col = bsq.columns[0]
            # Total debt (approx: short+long debt)
            td = 0.0
            for k in ["Short Long Term Debt","ShortLongTermDebt","Short Term Debt","ShortTermDebt","Long Term Debt","LongTermDebt","Total Debt","TotalDebt"]:
                if k in bsq.index and pd.notna(bsq.loc[k, col]):
                    td += _safe_float(bsq.loc[k, col], 0.0)
            # Equity
            eq = 0.0
            for k in ["Total Stockholder Equity","TotalStockholderEquity","Stockholders Equity","Total Equity","TotalEquity"]:
                if k in bsq.index and pd.notna(bsq.loc[k, col]):
                    eq = _safe_float(bsq.loc[k, col], 0.0); break
            out["Debt/Equity"] = td/eq if eq > 0 else 0.0

            # Cash & equivalents
            cash = 0.0
            for k in ["Cash And Cash Equivalents","CashAndCashEquivalents","Cash And Cash Equivalents Including Restricted Cash","CashAndShortTermInvestments"]:
                if k in bsq.index and pd.notna(bsq.loc[k, col]):
                    cash = _safe_float(bsq.loc[k, col], 0.0); break
            out["Kassa (valuta)"] = cash
    except Exception:
        pass
    return out

def _margin_metrics(t: yf.Ticker) -> Dict[str, float]:
    """Brutto- & netto-marginal (senaste årsdata om möjligt)."""
    out: Dict[str, float] = {}
    try:
        is_annual = t.financials  # annual
        if isinstance(is_annual, pd.DataFrame) and not is_annual.empty:
            col = is_annual.columns[0]
            rev = None
            gp = None
            ni = None
            if "Total Revenue" in is_annual.index and pd.notna(is_annual.loc["Total Revenue", col]):
                rev = _safe_float(is_annual.loc["Total Revenue", col], 0.0)
            for k in ["Gross Profit","GrossProfit"]:
                if k in is_annual.index and pd.notna(is_annual.loc[k, col]):
                    gp = _safe_float(is_annual.loc[k, col], 0.0); break
            for k in ["Net Income","NetIncome"]:
                if k in is_annual.index and pd.notna(is_annual.loc[k, col]):
                    ni = _safe_float(is_annual.loc[k, col], 0.0); break
            if rev and rev > 0:
                if gp is not None:
                    out["Bruttomarginal (%)"] = round(gp/rev*100.0, 2)
                if ni is not None:
                    out["Nettomarginal (%)"] = round(ni/rev*100.0, 2)
    except Exception:
        pass
    return out

def _fcf_metrics(t: yf.Ticker) -> Dict[str, float]:
    """FCF TTM + runway (kvartal) från quarterly_cashflow."""
    out: Dict[str, float] = {}
    try:
        qcf = t.quarterly_cashflow
        if isinstance(qcf, pd.DataFrame) and not qcf.empty:
            # FCF per kvartal ≈ CFO - CapEx
            rows = []
            for c in qcf.columns[:4]:
                cfo = 0.0; capex = 0.0
                for k in ["Total Cash From Operating Activities","TotalCashFromOperatingActivities","Operating Cash Flow","OperatingCashFlow"]:
                    if k in qcf.index and pd.notna(qcf.loc[k, c]):
                        cfo = _safe_float(qcf.loc[k, c]); break
                for k in ["Capital Expenditures","CapitalExpenditures"]:
                    if k in qcf.index and pd.notna(qcf.loc[k, c]):
                        capex = _safe_float(qcf.loc[k, c]); break
                rows.append(cfo - capex)
            rows = [x for x in rows if x is not None]
            if rows:
                out["FCF TTM (valuta)"] = float(sum(rows[:4]))
                # runway (kvartal) om negativt fcf i snitt
                neg_avg = np.mean([x for x in rows[:4] if x < 0]) if any(x < 0 for x in rows[:4]) else 0.0
                if neg_avg < 0:
                    # behövs "Kassa (valuta)"
                    bal = _balance_metrics(t)
                    cash = bal.get("Kassa (valuta)", 0.0)
                    out["Runway (kvartal)"] = float(cash / abs(neg_avg)) if abs(neg_avg) > 0 else 0.0
                else:
                    out["Runway (kvartal)"] = 0.0
    except Exception:
        pass
    return out

# -------------------------------------------------
# Runners
# -------------------------------------------------

def fetch_price_only(ticker: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Minimal uppdatering: pris, market cap, implied shares, namn, valuta, sektor.
    Skriver INTE manuella prognosfält.
    """
    out: Dict[str, Any] = {}
    dbg: Dict[str, Any] = {"runner": "price_only"}

    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
    except Exception as e:
        dbg["error"] = _human_err(e)
        return {}, dbg

    # Pris & valuta
    px = _safe_float(info.get("regularMarketPrice") or info.get("currentPrice") or 0.0)
    if px <= 0:
        try:
            h = t.history(period="1d")
            if not h.empty and "Close" in h:
                px = float(h["Close"].iloc[-1])
        except Exception:
            pass
    if px > 0:
        out["Aktuell kurs"] = px

    ccy = (info.get("currency") or "USD")
    out["Valuta"] = str(ccy).upper()

    # Namn & sektor
    name = info.get("shortName") or info.get("longName")
    if name:
        out["Bolagsnamn"] = str(name)
    sector = info.get("sector") or info.get("industry") or ""
    if sector:
        out["Sektor"] = str(sector)

    # Market cap & implied shares
    mcap = _safe_float(info.get("marketCap"), 0.0)
    if mcap <= 0 and px > 0:
        so = _safe_float(info.get("sharesOutstanding"), 0.0)
        if so > 0:
            mcap = so * px
    if mcap > 0:
        out["Market Cap (nu)"] = mcap
        # Utestående aktier (miljoner)
        out["Utestående aktier"] = (mcap/px)/1e6 if px > 0 else _safe_float(info.get("sharesOutstanding"),0.0)/1e6

    return out, dbg

def fetch_all_fields_for_ticker(ticker: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Full auto – hämtar allt utom manuella prognosfält:
      - Bolagsnamn, Valuta, Sektor
      - Aktuell kurs, Market Cap (nu), Utestående aktier (miljoner)
      - P/S (nu) + P/S Q1–Q4 (TTM-metodik med dec/jan-fix)
      - Debt/Equity, Brutto/Netto-marginal, Kassa, FCF TTM, Runway
    """
    out: Dict[str, Any] = {}
    dbg: Dict[str, Any] = {"runner": "full_auto"}

    # Basinfo
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
    except Exception as e:
        dbg["error"] = _human_err(e)
        return {}, dbg

    # Namn/valuta/sektor
    px = _safe_float(info.get("regularMarketPrice") or info.get("currentPrice") or 0.0)
    if px <= 0:
        try:
            h = t.history(period="1d")
            if not h.empty and "Close" in h:
                px = float(h["Close"].iloc[-1])
        except Exception:
            pass
    if px > 0:
        out["Aktuell kurs"] = px

    price_ccy = str(info.get("currency") or "USD").upper()
    out["Valuta"] = price_ccy

    name = info.get("shortName") or info.get("longName")
    if name:
        out["Bolagsnamn"] = str(name)
    sector = info.get("sector") or info.get("industry") or ""
    if sector:
        out["Sektor"] = str(sector)

    # Market cap & shares
    mcap_now = _safe_float(info.get("marketCap"), 0.0)
    if mcap_now <= 0 and px > 0:
        so = _safe_float(info.get("sharesOutstanding"), 0.0)
        if so > 0:
            mcap_now = so * px
    if mcap_now > 0:
        out["Market Cap (nu)"] = mcap_now

    shares = 0.0
    if px > 0 and mcap_now > 0:
        shares = mcap_now / px
    else:
        shares = _safe_float(info.get("sharesOutstanding"), 0.0)
    if shares > 0:
        out["Utestående aktier"] = shares / 1e6  # miljoner

    # Kvartalsintäkter (Yahoo) → TTM-fönster
    q_rows, fin_ccy = _quarterly_revenues_yf(t)
    dbg["quarters"] = [(str(d), v) for (d, v) in q_rows[:6]]
    # valuta-kvot
    rate = 1.0
    if fin_ccy and price_ccy and fin_ccy.upper() != price_ccy.upper():
        rate = _fx_rate(fin_ccy, price_ccy)
    # TTM-lista (nyast→)
    ttm_list = _ttm_windows(q_rows, need=6)
    ttm_px = [(d, v * rate) for (d, v) in ttm_list]  # samma valuta som 'Valuta'

    # P/S (nu)
    if mcap_now > 0 and ttm_px:
        ltm_now = _safe_float(ttm_px[0][1], 0.0)
        if ltm_now > 0:
            out["P/S"] = mcap_now / ltm_now

    # P/S Q1–Q4 (historik)
    if shares > 0 and ttm_px:
        q_dates = [d for (d, _) in ttm_px]
        px_map = _yahoo_prices_for_dates(ticker, q_dates)
        # räkna P/S för upp till 4 TTM-slut
        ps_hist = []
        for idx, (d_end, ttm_rev_px) in enumerate(ttm_px[:4], start=1):
            if ttm_rev_px and ttm_rev_px > 0:
                p_hist = _safe_float(px_map.get(d_end), 0.0)
                if p_hist > 0:
                    mcap_hist = shares * p_hist
                    ps_val = mcap_hist / ttm_rev_px
                    out[f"P/S Q{idx}"] = ps_val
                    ps_hist.append((str(d_end), ps_val))
        dbg["ps_hist"] = ps_hist

    # Extra nyckeltal
    out.update(_balance_metrics(t))
    out.update(_margin_metrics(t))
    out.update(_fcf_metrics(t))

    return out, dbg
