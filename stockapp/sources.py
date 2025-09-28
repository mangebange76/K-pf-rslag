# stockapp/sources.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Dict, Tuple, List, Optional
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd

try:
    import requests
except Exception:
    requests = None

try:
    import yfinance as yf
except Exception:
    yf = None


# -------------------------------------------------------------------
# Små verktyg
# -------------------------------------------------------------------

def _sf(v, d=0.0) -> float:
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return d
        return float(v)
    except Exception:
        return d

def _si(v, d=0) -> int:
    try:
        return int(v)
    except Exception:
        return d

def _to_date(x) -> Optional[date]:
    try:
        if isinstance(x, (pd.Timestamp,)):
            return x.date()
        return pd.to_datetime(x).date()
    except Exception:
        return None

def _fx_rate(base: str, quote: str) -> float:
    """
    Gratis FX (Frankfurter → exchangerate.host). Returnerar 1.0 om något strular.
    """
    base = (base or "").upper()
    quote = (quote or "").upper()
    if not base or not quote or base == quote:
        return 1.0
    if requests is None:
        return 1.0
    # Frankfurter
    try:
        r = requests.get("https://api.frankfurter.app/latest",
                         params={"from": base, "to": quote}, timeout=10)
        if r.status_code == 200:
            j = r.json() or {}
            v = (j.get("rates") or {}).get(quote)
            if v:
                return float(v)
    except Exception:
        pass
    # exchangerate.host
    try:
        r = requests.get("https://api.exchangerate.host/latest",
                         params={"base": base, "symbols": quote}, timeout=10)
        if r.status_code == 200:
            j = r.json() or {}
            v = (j.get("rates") or {}).get(quote)
            if v:
                return float(v)
    except Exception:
        pass
    return 1.0


# -------------------------------------------------------------------
# Yahoo helpers
# -------------------------------------------------------------------

def _y_ticker(ticker: str):
    if yf is None:
        raise RuntimeError("yfinance saknas i miljön.")
    return yf.Ticker(str(ticker).strip())

def _y_info(tkr) -> dict:
    try:
        return tkr.info or {}
    except Exception:
        return {}

def _y_hist_close_on_or_before(ticker: str, dates: List[date]) -> Dict[date, float]:
    """
    Hämtar dagliga 'Close' i ett fönster runt efterfrågade datum och
    ger närmaste pris PÅ eller NÄRMAST FÖRE respektive datum.
    """
    if not dates:
        return {}
    dmin = min(dates) - timedelta(days=14)
    dmax = max(dates) + timedelta(days=3)
    try:
        t = _y_ticker(ticker)
        hist = t.history(start=dmin, end=dmax, interval="1d")
        if hist is None or hist.empty:
            return {}
        hist = hist.sort_index()
        idx_dates = [ix.date() for ix in hist.index]
        closes = list(hist["Close"].values)
        out = {}
        for d in dates:
            px = None
            for j in range(len(idx_dates) - 1, -1, -1):
                if idx_dates[j] <= d:
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

def _y_quarterly_financials(tkr) -> pd.DataFrame:
    try:
        df = tkr.quarterly_financials
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    except Exception:
        pass
    return pd.DataFrame()

def _y_quarterly_income_stmt(tkr) -> pd.DataFrame:
    try:
        df = tkr.income_stmt
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    except Exception:
        pass
    return pd.DataFrame()

def _y_quarterly_balance_sheet(tkr) -> pd.DataFrame:
    try:
        df = tkr.balance_sheet
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    except Exception:
        pass
    return pd.DataFrame()

def _y_quarterly_cashflow(tkr) -> pd.DataFrame:
    try:
        df = tkr.quarterly_cashflow
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    except Exception:
        pass
    return pd.DataFrame()


# -------------------------------------------------------------------
# TTM & nyckeltals-beräkningar (Yahoo-baserade, valutajusterade vid behov)
# -------------------------------------------------------------------

def _revenue_quarter_series(tkr) -> Tuple[List[Tuple[date, float]], str]:
    """
    Returnerar lista [(period_end_date, revenue_value), ...] (nyast->äldst) och finansvaluta.
    Försöker först quarterly_financials ("Total Revenue"), annars income_stmt ("Total Revenue").
    """
    fin_ccy = None
    out: List[Tuple[date, float]] = []

    # 1) quarterly_financials
    qf = _y_quarterly_financials(tkr)
    if not qf.empty:
        fin_ccy = _y_info(tkr).get("financialCurrency") or fin_ccy
        idx_names = [str(x) for x in qf.index]
        cand = None
        for key in ["Total Revenue", "TotalRevenue", "Revenues", "Revenue", "Sales",
                    "Total revenue", "Revenues from contracts with customers"]:
            if key in idx_names:
                cand = key
                break
        if cand:
            ser = qf.loc[cand].dropna()
            for c, v in ser.items():
                d = _to_date(c)
                if d and _sf(v, 0.0) > 0:
                    out.append((d, float(v)))
    # 2) fallback: income_stmt
    if not out:
        is_df = _y_quarterly_income_stmt(tkr)
        if not is_df.empty and "Total Revenue" in is_df.index:
            fin_ccy = _y_info(tkr).get("financialCurrency") or fin_ccy
            ser = is_df.loc["Total Revenue"].dropna()
            for c, v in ser.items():
                d = _to_date(c)
                if d and _sf(v, 0.0) > 0:
                    out.append((d, float(v)))
    out.sort(key=lambda x: x[0], reverse=True)
    return out, (fin_ccy or _y_info(tkr).get("financialCurrency") or "")

def _ttm_from_quarters(rev_rows: List[Tuple[date, float]], take: int = 4) -> List[Tuple[date, float]]:
    """
    Rullande TTM: returnerar upp till 'take' fönster [(end_date, ttm_sum), ...] nyast->äldst.
    Tar höjd för årsskifte: om fler än 4 finns, tar de 4 senaste *unika* kvartalen.
    """
    if len(rev_rows) < 4:
        return []
    # deduplicera på datum (om dubbletter)
    ded: Dict[date, float] = {}
    for d, v in rev_rows:
        ded[d] = float(v)
    rows = sorted(ded.items(), key=lambda x: x[0], reverse=True)
    # bygg TTM-fönster
    out = []
    for i in range(0, min(take, len(rows) - 3)):
        end_d = rows[i][0]
        ttm = rows[i][1] + rows[i+1][1] + rows[i+2][1] + rows[i+3][1]
        out.append((end_d, float(ttm)))
    return out

def _gross_and_net_margin_ttm(tkr) -> Tuple[float, float]:
    """
    Approximerar bruttomarginal & nettomarginal på TTM-basis.
    """
    qf = _y_quarterly_financials(tkr)
    if qf.empty:
        return 0.0, 0.0

    # Hitta rader
    def _pick_row(name_list: List[str]) -> Optional[pd.Series]:
        for nm in name_list:
            if nm in qf.index:
                return qf.loc[nm]
        return None

    rev = _pick_row(["Total Revenue","TotalRevenue","Revenues","Revenue","Sales","Total revenue"])
    gross = _pick_row(["Gross Profit","GrossProfit"])
    net = _pick_row(["Net Income","NetIncome","Net Income Common Stockholders","NetIncomeCommonStockholders"])

    def _sum_last4(s: Optional[pd.Series]) -> float:
        if s is None:
            return 0.0
        vals = [ _sf(v, 0.0) for v in s.dropna().values[:4] ]
        return float(sum(vals)) if len(vals) >= 1 else 0.0

    rev_ttm = _sum_last4(rev)
    gp_ttm = _sum_last4(gross)
    ni_ttm = _sum_last4(net)

    gm = (gp_ttm / rev_ttm * 100.0) if rev_ttm > 0 else 0.0
    nm = (ni_ttm / rev_ttm * 100.0) if rev_ttm > 0 else 0.0
    return float(round(gm, 2)), float(round(nm, 2))

def _debt_equity_latest(tkr) -> float:
    """
    Debt/Equity från senaste balans: (Total Debt / Total Equity).
    """
    bs = _y_quarterly_balance_sheet(tkr)
    if bs.empty:
        return 0.0

    def _pick(name_list: List[str]) -> Optional[pd.Series]:
        for nm in name_list:
            if nm in bs.index:
                return bs.loc[nm]
        return None

    tot_debt = _pick(["Total Debt","TotalDebt"])
    tot_equity = _pick(["Total Stockholder Equity","TotalStockholderEquity","StockholdersEquity"])

    try:
        d = float(tot_debt.dropna().iloc[0]) if tot_debt is not None else 0.0
        e = float(tot_equity.dropna().iloc[0]) if tot_equity is not None else 0.0
        if e > 0:
            return float(round(d / e, 3))
    except Exception:
        pass
    return 0.0

def _cash_latest(tkr) -> float:
    """
    Cash (eller Cash + ST Investments) från senaste balans.
    """
    bs = _y_quarterly_balance_sheet(tkr)
    if bs.empty:
        return 0.0

    # Prefer: Cash, Cash Equivalents & Short Term Investments
    for k in ["Cash Cash Equivalents And Short Term Investments",
              "CashAndCashEquivalentsAndShortTermInvestments",
              "Cash And Cash Equivalents",
              "CashAndCashEquivalents"]:
        if k in bs.index:
            try:
                return float(bs.loc[k].dropna().iloc[0])
            except Exception:
                pass
    return 0.0

def _fcf_ttm(tkr) -> float:
    """
    FCF TTM ≈ CFO TTM - CapEx TTM (kvartalscashflow).
    """
    cf = _y_quarterly_cashflow(tkr)
    if cf.empty:
        return 0.0

    def _sum4(name_opts: List[str]) -> float:
        for nm in name_opts:
            if nm in cf.index:
                try:
                    vals = [ _sf(v, 0.0) for v in cf.loc[nm].dropna().values[:4] ]
                    return float(sum(vals)) if len(vals) >= 1 else 0.0
                except Exception:
                    pass
        return 0.0

    cfo = _sum4(["Total Cash From Operating Activities", "Operating Cash Flow", "CashFlowFromOperations"])
    capex = _sum4(["Capital Expenditures","Investing Cashflow Capital Expenditures"])
    return float(cfo - abs(capex))  # capex brukar vara negativ i rapporten

def _runway_months(cash: float, fcf_ttm: float) -> float:
    """
    Om FCF TTM < 0 → brännhastighet ≈ |FCF|/12 per månad → månader kvar = cash / burn_m.
    Om FCF >= 0 → returnera '24' som cap (minst 24 mån stabilt).
    """
    if fcf_ttm < 0:
        burn_m = abs(fcf_ttm) / 12.0
        if burn_m > 0:
            return float(round(cash / burn_m, 1))
        return 0.0
    return 24.0

def _ps_quarter_history(ticker: str, tkr, shares: float, price_ccy: str) -> Dict[str, float]:
    """
    Beräknar P/S Q1..Q4 via TTM-fönster från kvartalsintäkter och historiska priser.
    Konverterar TTM-revenue till prisvaluta vid behov.
    """
    rev_rows, fin_ccy = _revenue_quarter_series(tkr)
    ttm_rows = _ttm_from_quarters(rev_rows, take=5)  # ta fler och välj 4 senaste
    if not ttm_rows:
        return {}

    # Valuta-konvertering
    conv = 1.0
    if fin_ccy and price_ccy and fin_ccy.upper() != price_ccy.upper():
        conv = _fx_rate(fin_ccy.upper(), price_ccy.upper())

    # hämta priser vid kvartals-slut
    q_dates = [d for (d, _) in ttm_rows]
    px_map = _y_hist_close_on_or_before(ticker, q_dates)

    out: Dict[str, float] = {}
    qnum = 1
    for (d_end, ttm_val) in ttm_rows[:4]:
        ttm_px = float(ttm_val) * float(conv)
        px = _sf(px_map.get(d_end, None), 0.0)
        if ttm_px > 0 and px > 0 and shares > 0:
            mcap_hist = shares * px
            ps = float(mcap_hist / ttm_px)
            out[f"P/S Q{qnum}"] = float(round(ps, 4))
        qnum += 1
    return out


# -------------------------------------------------------------------
# Publika fetch-funktioner
# -------------------------------------------------------------------

def fetch_price_only(ticker: str) -> Tuple[Dict, Dict]:
    """
    Snabb uppdatering: Aktuell kurs, Valuta, Market Cap, Namn, Sektor.
    Returnerar (vals, debug)
    """
    t = _y_ticker(ticker)
    info = _y_info(t)
    vals: Dict = {}

    # Pris
    px = info.get("regularMarketPrice")
    if px is None:
        try:
            h = t.history(period="1d")
            if not h.empty and "Close" in h:
                px = float(h["Close"].iloc[-1])
        except Exception:
            px = None
    if px is not None:
        vals["Aktuell kurs"] = float(px)

    # Valuta/Namn/Sektor
    if info.get("currency"):
        vals["Valuta"] = str(info["currency"]).upper()
    if info.get("shortName") or info.get("longName"):
        vals["Bolagsnamn"] = str(info.get("shortName") or info.get("longName"))
    if info.get("sector"):
        vals["Sektor"] = str(info.get("sector"))

    # Market cap
    mcap = info.get("marketCap")
    try:
        if mcap is not None:
            vals["Market Cap (nu)"] = float(mcap)
    except Exception:
        pass

    # Årlig utdelning (per aktie) – om finns
    div_rate = info.get("dividendRate")
    if div_rate is not None:
        try:
            vals["Årlig utdelning"] = float(div_rate)
        except Exception:
            pass

    dbg = {"source": "yfinance.info"}
    return vals, dbg


def fetch_all_fields_for_ticker(ticker: str) -> Tuple[Dict, Dict]:
    """
    Försök hämta samtliga nyckeltal för investeringsvyerna (lagligt & gratis).
    Returnerar (vals, debug). Hämtar *inte* omsättningsprognoser – dessa ska matas manuellt.
    """
    t = _y_ticker(ticker)
    info = _y_info(t)
    vals: Dict = {}
    debug: Dict = {"steps": []}

    # --- Bas
    px = info.get("regularMarketPrice")
    if px is None:
        try:
            h = t.history(period="1d")
            if not h.empty and "Close" in h:
                px = float(h["Close"].iloc[-1])
        except Exception:
            px = None
    if px is not None:
        vals["Aktuell kurs"] = float(px); debug["steps"].append("price-ok")

    if info.get("currency"):
        vals["Valuta"] = str(info["currency"]).upper()
    price_ccy = vals.get("Valuta") or (info.get("currency") or "USD")

    if info.get("shortName") or info.get("longName"):
        vals["Bolagsnamn"] = str(info.get("shortName") or info.get("longName"))
    if info.get("sector"):
        vals["Sektor"] = str(info.get("sector"))

    mcap = _sf(info.get("marketCap"), 0.0)
    if mcap > 0:
        vals["Market Cap (nu)"] = mcap

    # --- Shares (implied → fallback)
    shares = 0.0
    if mcap > 0 and _sf(px, 0.0) > 0:
        shares = mcap / float(px)
        debug["steps"].append("shares-implied")
    else:
        so = _sf(info.get("sharesOutstanding"), 0.0)
        if so > 0:
            shares = so
            debug["steps"].append("shares-direct")
    if shares > 0:
        vals["Utestående aktier"] = float(shares / 1e6)  # i miljoner

    # --- Dividend (per aktie)
    div_rate = info.get("dividendRate")
    if div_rate is not None:
        try:
            vals["Årlig utdelning"] = float(div_rate)
        except Exception:
            pass

    # --- Marginaler (TTM)
    gm, nm = _gross_and_net_margin_ttm(t)
    if gm or nm:
        vals["Bruttomarginal (%)"] = gm
        vals["Nettomarginal (%)"] = nm

    # --- D/E
    de = _debt_equity_latest(t)
    if de:
        vals["Debt/Equity"] = de

    # --- Cash & FCF & Runway
    cash = _sf(_cash_latest(t), 0.0)
    fcf  = _sf(_fcf_ttm(t), 0.0)
    if cash > 0:
        vals["Kassa (valuta)"] = cash
    vals["FCF TTM (valuta)"] = fcf
    vals["Runway (mån)"] = _runway_months(cash, fcf)

    # --- P/S (TTM)
    ps_now = _sf(info.get("priceToSalesTrailing12Months"), 0.0)
    if ps_now <= 0 and mcap > 0:
        # beräkna via revenue TTM (i fin.valuta → konvertera)
        rev_rows, fin_ccy = _revenue_quarter_series(t)
        ttm = _ttm_from_quarters(rev_rows, take=1)
        if ttm:
            conv = 1.0
            if fin_ccy and price_ccy and fin_ccy.upper() != price_ccy.upper():
                conv = _fx_rate(fin_ccy.upper(), price_ccy.upper())
            rev_ttm_px = float(ttm[0][1]) * conv
            if rev_ttm_px > 0:
                ps_now = float(mcap / rev_ttm_px)
                debug["steps"].append("ps-calced")
    if ps_now > 0:
        vals["P/S"] = float(round(ps_now, 4))

    # --- P/S Q1..Q4 (historik)
    if shares > 0 and _sf(px, 0.0) > 0:
        # price_ccy redan satt
        ps_hist = _ps_quarter_history(ticker, t, shares, price_ccy)
        for k, v in ps_hist.items():
            vals[k] = v

    debug["source"] = "yfinance+free-fx"
    return vals, debug
