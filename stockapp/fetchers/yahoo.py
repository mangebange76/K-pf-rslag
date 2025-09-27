# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta

def _yfi_info_dict(t: yf.Ticker) -> dict:
    try:
        return t.info or {}
    except Exception:
        return {}

def implied_shares_from_yahoo(ticker: str, price: float = None, mcap: float = None) -> float:
    t = yf.Ticker(ticker)
    info = _yfi_info_dict(t)
    if mcap is None:
        mcap = info.get("marketCap")
    if price is None:
        price = info.get("regularMarketPrice")
    try:
        mcap = float(mcap or 0.0); price = float(price or 0.0)
    except Exception:
        return 0.0
    if mcap > 0 and price > 0:
        return mcap / price
    so = info.get("sharesOutstanding")
    try:
        return float(so or 0.0)
    except Exception:
        return 0.0

def fetch_yahoo_basics(ticker: str) -> dict:
    out = {}
    t = yf.Ticker(ticker)
    info = _yfi_info_dict(t)

    # pris
    px = info.get("regularMarketPrice")
    if px is None:
        try:
            hist = t.history(period="1d")
            if not hist.empty and "Close" in hist:
                px = float(hist["Close"].iloc[-1])
        except Exception:
            px = None
    if px is not None:
        out["Aktuell kurs"] = float(px)

    # namn, valuta
    out["Bolagsnamn"] = info.get("shortName") or info.get("longName") or ""
    out["Valuta"] = str(info.get("currency") or "USD").upper()

    # utdelning
    div_rate = info.get("dividendRate")
    if div_rate is not None:
        try: out["Årlig utdelning"] = float(div_rate)
        except: pass

    # market cap
    if info.get("marketCap") is not None:
        try: out["Market Cap (valuta)"] = float(info["marketCap"])
        except: pass

    # sector/industry
    if info.get("sector"): out["Sector"] = info["sector"]
    if info.get("industry"): out["Industry"] = info["industry"]

    # EV/EBITDA
    ev = info.get("enterpriseValue"); ebitda = info.get("ebitda")
    try:
        if ev is not None: out["EV"] = float(ev)
        if ebitda is not None: out["EBITDA"] = float(ebitda)
        if ev and ebitda and float(ebitda)!=0:
            out["EV/EBITDA"] = float(ev)/float(ebitda)
    except: pass

    # Margins & D/E approx
    # grossProfit / totalRevenue , netIncome / totalRevenue
    try:
        fin = t.financials  # annual
        if isinstance(fin, pd.DataFrame) and not fin.empty:
            if "Total Revenue" in fin.index and "Gross Profit" in fin.index:
                tr = float(fin.loc["Total Revenue"].dropna().iloc[-1] or 0.0)
                gp = float(fin.loc["Gross Profit"].dropna().iloc[-1] or 0.0)
                if tr>0: out["Gross Margin (%)"] = round(gp/tr*100.0, 2)
            if "Total Revenue" in fin.index and "Net Income" in fin.index:
                tr = float(fin.loc["Total Revenue"].dropna().iloc[-1] or 0.0)
                ni = float(fin.loc["Net Income"].dropna().iloc[-1] or 0.0)
                if tr>0: out["Net Margin (%)"] = round(ni/tr*100.0, 2)
    except Exception:
        pass

    # Cash & FCF approx
    try:
        cf = t.cashflow
        if isinstance(cf, pd.DataFrame) and not cf.empty:
            # Cash & equivalents: from balance_sheet (more robust)
            bs = t.balance_sheet
            if isinstance(bs, pd.DataFrame) and not bs.empty:
                for key in ("Cash And Cash Equivalents", "CashAndCashEquivalents", "CashAndCashEquivalentsAtCarryingValue"):
                    if key in bs.index:
                        out["Cash & Equivalents"] = float(bs.loc[key].dropna().iloc[-1] or 0.0); break
            # Free cash flow: Operating CF - CapEx (approx)
            op = None; capex = None
            for k in ("Total Cash From Operating Activities","Operating Cash Flow"):
                if k in cf.index:
                    op = float(cf.loc[k].dropna().iloc[-1] or 0.0); break
            for k in ("Capital Expenditures","CapitalExpenditures"):
                if k in cf.index:
                    capex = float(cf.loc[k].dropna().iloc[-1] or 0.0); break
            if op is not None and capex is not None:
                out["Free Cash Flow"] = float(op - abs(capex))
    except Exception:
        pass

    # D/E approx (totalDebt / totalStockholderEquity)
    try:
        bs = t.balance_sheet
        if isinstance(bs, pd.DataFrame) and not bs.empty:
            td = None; eq = None
            for k in ("Total Debt","TotalDebt"):
                if k in bs.index: td = float(bs.loc[k].dropna().iloc[-1] or 0.0); break
            for k in ("Total Stockholder Equity","TotalStockholderEquity"):
                if k in bs.index: eq = float(bs.loc[k].dropna().iloc[-1] or 0.0); break
            if td is not None and eq not in (None, 0.0):
                out["Debt/Equity"] = float(td/eq) if eq!=0 else 0.0
    except Exception:
        pass

    # implied shares
    out["_implied_shares"] = implied_shares_from_yahoo(ticker, price=px, mcap=info.get("marketCap"))

    return out

def yahoo_quarterly_revenues(ticker: str) -> list[tuple]:
    """
    [(period_end_date, value)], nyast→äldst
    """
    t = yf.Ticker(ticker)
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
    # 2) backup via income_stmt quarterly
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

def yahoo_prices_for_dates(ticker: str, dates: list) -> dict:
    """
    Dagliga priser i fönster för alla 'dates' och returnerar Close på eller närmast FÖRE respektive datum.
    """
    if not dates:
        return {}
    dmin = min(dates) - timedelta(days=14)
    dmax = max(dates) + timedelta(days=2)
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
