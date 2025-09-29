# stockapp/sources.py
# -*- coding: utf-8 -*-
"""
Datakällor & uppdaterings-runners.

Publika funktioner:
- run_update_price_only(df, ticker, user_rates) -> (df2, changed_fields, msg)
- run_update_full(df, ticker, user_rates) -> (df2, changed_fields, msg)

Intern logik använder yfinance som huvudkälla. Vi beräknar:
- Aktuell kurs, valuta, namn, beta, market cap, sektor/industri
- Utestående aktier (implied: mcap/price), P/S (nu), P/S Q1–Q4 baserat på TTM-intäkter och historiska priser
- CAGR 5 år (revenue), bruttomarginal/nettomarginal, EBITDA-margin (proxy), FCF-margin (proxy)
- Debt/Equity, Cash & Equivalents
- Risklabel via market cap i USD (hämtar FX USD-kors via Frankfurter fallback)

OBS: Fält som inte finns i df ignoreras vid skrivning.
"""

from __future__ import annotations
from typing import Dict, Tuple, List, Optional
from datetime import timedelta, date

import numpy as np
import pandas as pd
import yfinance as yf
import requests
import streamlit as st

from .config import TS_FIELDS
from .utils import now_stamp, now_dt

# ------------------------------------------------------------
# Hjälpare
# ------------------------------------------------------------

def _f(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _safe_date(x) -> Optional[date]:
    try:
        if hasattr(x, "date"):
            return x.date()
        return pd.to_datetime(x).date()
    except Exception:
        return None

def _with_nan_to_zero(v):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return 0.0
        return float(v)
    except Exception:
        return 0.0

def _risk_from_mcap_usd(mc_usd: float) -> str:
    mc = _f(mc_usd)
    if mc >= 200_000_000_000:  # >= 200B
        return "Megacap"
    if mc >= 10_000_000_000:
        return "Largecap"
    if mc >= 2_000_000_000:
        return "Midcap"
    if mc >= 300_000_000:
        return "Smallcap"
    return "Microcap"

def _fx_rate(base: str, quote: str) -> float:
    """Enkel FX via Frankfurter -> exchangerate.host fallback."""
    try:
        base = (base or "").upper()
        quote = (quote or "").upper()
        if not base or not quote or base == quote:
            return 1.0
        r = requests.get("https://api.frankfurter.app/latest",
                         params={"from": base, "to": quote}, timeout=10)
        if r.status_code == 200:
            v = (r.json() or {}).get("rates", {}).get(quote)
            if v:
                return float(v)
    except Exception:
        pass
    try:
        r = requests.get("https://api.exchangerate.host/latest",
                         params={"base": base, "symbols": quote}, timeout=10)
        if r.status_code == 200:
            v = (r.json() or {}).get("rates", {}).get(quote)
            if v:
                return float(v)
    except Exception:
        pass
    return 1.0

def _yfi_info(tkr: yf.Ticker) -> dict:
    try:
        return tkr.info or {}
    except Exception:
        return {}

def _hist_price_at_or_before(tkr: yf.Ticker, d: date) -> Optional[float]:
    """Daglig close på eller närmast före d (± 10 dagar fönster)."""
    try:
        start = pd.Timestamp(d) - pd.Timedelta(days=14)
        end = pd.Timestamp(d) + pd.Timedelta(days=2)
        h = tkr.history(start=start, end=end, interval="1d")
        if h is None or h.empty:
            return None
        h = h.sort_index()
        idx = list(h.index.date)
        closes = list(h["Close"].values)
        for j in range(len(idx) - 1, -1, -1):
            if idx[j] <= d:
                return float(closes[j])
        return None
    except Exception:
        return None

def _ensure_ts_cols(df: pd.DataFrame, row_idx: int, changed_fields: List[str], stamp_even_if_same: bool = True):
    """Stämplar TS_ för fält i TS_FIELDS när dessa finns i changed_fields (eller alltid om stamp_even_if_same=True)."""
    for f, ts_col in TS_FIELDS.items():
        if ts_col not in df.columns:
            continue
        if stamp_even_if_same:
            # stämpla alltid för de fält som uppdateras/beräknas i denna modul om de råkar vara med i changed_fields
            if f in changed_fields or f in ("Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "Omsättning idag", "Omsättning nästa år"):
                df.at[row_idx, ts_col] = now_stamp()
        else:
            if f in changed_fields:
                df.at[row_idx, ts_col] = now_stamp()

def _note_auto(df: pd.DataFrame, row_idx: int, src: str):
    if "Senast auto-uppdaterad" in df.columns:
        df.at[row_idx, "Senast auto-uppdaterad"] = now_stamp()
    if "Senast uppdaterad källa" in df.columns:
        df.at[row_idx, "Senast uppdaterad källa"] = src

def _apply_to_row(df: pd.DataFrame, row_idx: int, new_vals: Dict[str, object],
                  source: str, stamp_even_if_same: bool = True) -> List[str]:
    """Skriver in new_vals där kolumnen finns. Returnerar lista på fält som ändrats (eller skrivits in)."""
    changed: List[str] = []
    for k, v in new_vals.items():
        if k not in df.columns:
            continue
        old = df.at[row_idx, k]
        must_write = True
        # Tillåt skrivning även om samma (så att TS kan stämplas “senast hanterad”)
        if not stamp_even_if_same:
            must_write = str(old) != str(v)
        if must_write:
            df.at[row_idx, k] = v
            changed.append(k)
    if changed:
        _note_auto(df, row_idx, source)
        _ensure_ts_cols(df, row_idx, changed, stamp_even_if_same=stamp_even_if_same)
    return changed

# ------------------------------------------------------------
# Hämtningar & beräkningar via yfinance
# ------------------------------------------------------------

def _pull_basics(ticker: str) -> Dict[str, object]:
    """
    Basdata: namn, valuta, pris, utd, beta, mcap, shares (implied), sektor/industri.
    """
    out = {
        "Bolagsnamn": "",
        "Valuta": "USD",
        "Aktuell kurs": 0.0,
        "Årlig utdelning": 0.0,
        "Beta": 1.0,
        "Sektor": "",
        "Industri": "",
        "_MC": 0.0,         # market cap i prisvaluta
        "_Shares": 0.0,     # styck
        "Utestående aktier": 0.0,  # milj
    }
    t = yf.Ticker(ticker)
    info = _yfi_info(t)

    # namn
    nm = info.get("shortName") or info.get("longName") or ""
    out["Bolagsnamn"] = str(nm)

    # valuta & pris
    if info.get("currency"):
        out["Valuta"] = str(info["currency"]).upper()
    px = info.get("regularMarketPrice")
    if px is None:
        h = t.history(period="1d")
        if h is not None and not h.empty and "Close" in h:
            px = float(h["Close"].iloc[-1])
    out["Aktuell kurs"] = _with_nan_to_zero(px)

    # utdelning (årlig rate om tillgänglig)
    div_rate = info.get("dividendRate")
    out["Årlig utdelning"] = _with_nan_to_zero(div_rate)

    # beta
    out["Beta"] = _with_nan_to_zero(info.get("beta", 1.0))

    # mcap & shares
    mc = info.get("marketCap")
    mc = _with_nan_to_zero(mc)
    out["_MC"] = mc
    shares = info.get("sharesOutstanding")
    if shares is None or float(shares) <= 0:
        # implied
        px_now = out["Aktuell kurs"]
        shares = (mc / px_now) if (mc > 0 and px_now > 0) else 0.0
    out["_Shares"] = _with_nan_to_zero(shares)
    out["Utestående aktier"] = round(out["_Shares"] / 1e6, 6)

    # sektor/industri
    out["Sektor"] = str(info.get("sector") or "")
    out["Industri"] = str(info.get("industry") or "")

    return out

def _pull_quarterly_revenues(ticker: str) -> List[Tuple[date, float]]:
    """
    Kvartalsintäkter via yfinance.quarterly_financials.
    Returnerar [(period_end_date, revenue_value), ...] (nyast -> äldst).
    Försöker vara robust med datumparsing.
    """
    t = yf.Ticker(ticker)
    out: List[Tuple[date, float]] = []

    try:
        qf = t.quarterly_financials
        if isinstance(qf, pd.DataFrame) and not qf.empty:
            # leta på flera möjliga rader
            candidates = [
                "Total Revenue", "TotalRevenue", "Revenues", "Revenue",
                "Sales", "Revenues from contracts with customers"
            ]
            idx = [str(x).strip() for x in qf.index]
            key = None
            for c in candidates:
                if c in idx:
                    key = c
                    break
            if key:
                row = qf.loc[key].dropna()
                for col, val in row.items():
                    d = _safe_date(col)
                    if d and float(val) and float(val) > 0:
                        out.append((d, float(val)))
    except Exception:
        pass

    # sortera nyast->äldst
    out.sort(key=lambda x: x[0], reverse=True)
    return out

def _ttm_windows(values: List[Tuple[date, float]], need: int = 4) -> List[Tuple[date, float]]:
    """
    Bygg TTM-summor: [(end_date0, ttm0), ...] där ttm0 = sum(q0..q3)
    """
    res: List[Tuple[date, float]] = []
    if len(values) < 4:
        return res
    # vi tar de 5 senaste och bygger 4 fönster för att täcka “missing Jan/Dec”-case bättre
    limit = min(len(values), 5)
    vals = values[:limit]
    for i in range(0, min(need, len(vals) - 3)):
        end_i = vals[i][0]
        ttm_i = sum(v for (_, v) in vals[i : i + 4])
        res.append((end_i, float(ttm_i)))
    return res

def _ps_series_from_quarters(ticker: str, px_ccy: str, shares: float) -> Dict[str, float]:
    """
    Beräknar P/S (nu) och P/S Q1..Q4 från kvartalsintäkter + historiskt pris på kvartalets slutdag.
    shares ska vara antal aktier (styck).
    """
    t = yf.Ticker(ticker)
    rows = _pull_quarterly_revenues(ticker)
    if len(rows) < 4 or shares <= 0:
        return {}

    # TTM-lista
    ttm = _ttm_windows(rows, need=4)
    if not ttm:
        return {}

    # Hämta historiska priser på de valda slutdatumen
    ps_out: Dict[str, float] = {}
    px_map: Dict[date, float] = {}
    for (d_end, _) in ttm:
        px = _hist_price_at_or_before(t, d_end)
        if px and px > 0:
            px_map[d_end] = px

    # Nuvarande P/S via (marketCap / senaste TTM-intäkt)
    # market cap nu:
    info = _yfi_info(t)
    mc_now = _with_nan_to_zero(info.get("marketCap"))
    if mc_now <= 0:
        px_now = _with_nan_to_zero(info.get("regularMarketPrice"))
        if px_now > 0:
            mc_now = shares * px_now

    if mc_now > 0 and ttm:
        ltm_now = ttm[0][1]
        if ltm_now > 0:
            ps_out["P/S"] = float(mc_now / ltm_now)

    # P/S Q1..Q4 via historiskt pris * shares / ttm_rev
    for idx, (d_end, ttm_rev) in enumerate(ttm[:4], start=1):
        px = px_map.get(d_end)
        if px and px > 0 and ttm_rev > 0:
            mcap_hist = shares * px
            ps_out[f"P/S Q{idx}"] = float(mcap_hist / ttm_rev)

    return ps_out

def _cagr_5y_from_financials(ticker: str) -> float:
    """
    CAGR på revenue från årliga financials (så långt data räcker, minst 2 punkter).
    """
    t = yf.Ticker(ticker)
    try:
        is_annual = t.financials
        if isinstance(is_annual, pd.DataFrame) and not is_annual.empty:
            candidates = ["Total Revenue", "TotalRevenue", "Revenues", "Revenue", "Sales"]
            idx = [str(x).strip() for x in is_annual.index]
            key = None
            for c in candidates:
                if c in idx:
                    key = c
                    break
            if not key:
                return 0.0
            ser = is_annual.loc[key].dropna()
            if ser is None or ser.empty or len(ser) < 2:
                return 0.0
            ser = ser.sort_index()
            start = float(ser.iloc[0])
            end = float(ser.iloc[-1])
            years = max(1, len(ser) - 1)
            if start <= 0:
                return 0.0
            cagr = (end / start) ** (1.0 / years) - 1.0
            return round(cagr * 100.0, 2)
    except Exception:
        pass
    return 0.0

def _margin_metrics(ticker: str) -> Dict[str, float]:
    """
    Brutto/Netto/EBITDA margin proxies från annual financials om tillgängligt.
    """
    out = {
        "Gross Margin (%)": 0.0,
        "Net Margin (%)": 0.0,
        "EBITDA Margin (%)": 0.0,
        "FCF Margin (%)": 0.0,
    }
    t = yf.Ticker(ticker)
    try:
        is_annual = t.financials
        cf_annual = t.cashflow
        bs_annual = t.balance_sheet

        # Bruttomarginal = grossProfit / totalRevenue
        if isinstance(is_annual, pd.DataFrame) and not is_annual.empty:
            def pick(idx_name):
                idx = [str(x).strip() for x in is_annual.index]
                return idx_name if idx_name in idx else None

            rev_key = None
            for cand in ["Total Revenue", "TotalRevenue", "Revenues", "Revenue", "Sales"]:
                if pick(cand):
                    rev_key = cand; break
            gp_key = None
            for cand in ["Gross Profit", "GrossProfit"]:
                if pick(cand):
                    gp_key = cand; break
            ni_key = None
            for cand in ["Net Income", "NetIncome"]:
                if pick(cand):
                    ni_key = cand; break
            ebitda_key = None
            for cand in ["EBITDA"]:
                if pick(cand):
                    ebitda_key = cand; break

            # Ta senaste kolumn
            if rev_key:
                rev = _f(is_annual.loc[rev_key].dropna().iloc[-1], 0.0)
                if gp_key:
                    gp = _f(is_annual.loc[gp_key].dropna().iloc[-1], 0.0)
                    if rev > 0:
                        out["Gross Margin (%)"] = round(gp / rev * 100.0, 2)
                if ni_key:
                    ni = _f(is_annual.loc[ni_key].dropna().iloc[-1], 0.0)
                    if rev > 0:
                        out["Net Margin (%)"] = round(ni / rev * 100.0, 2)
                if ebitda_key:
                    ebitda = _f(is_annual.loc[ebitda_key].dropna().iloc[-1], 0.0)
                    if rev > 0:
                        out["EBITDA Margin (%)"] = round(ebitda / rev * 100.0, 2)

            # FCF margin proxy: (OperatingCashFlow - Capex) / Revenue
            if isinstance(cf_annual, pd.DataFrame) and not cf_annual.empty and rev_key:
                ocf = None
                capex = None
                for cand in ["Total Cash From Operating Activities", "Operating Cash Flow", "OperatingCashFlow"]:
                    if cand in [str(x).strip() for x in cf_annual.index]:
                        try:
                            ocf = _f(cf_annual.loc[cand].dropna().iloc[-1], 0.0)
                            break
                        except Exception:
                            pass
                for cand in ["Capital Expenditures", "CapitalExpenditures"]:
                    if cand in [str(x).strip() for x in cf_annual.index]:
                        try:
                            capex = _f(cf_annual.loc[cand].dropna().iloc[-1], 0.0)
                            break
                        except Exception:
                            pass
                if ocf is not None and capex is not None and rev_key:
                    rev = _f(is_annual.loc[rev_key].dropna().iloc[-1], 0.0)
                    if rev > 0:
                        fcf = ocf - capex
                        out["FCF Margin (%)"] = round((fcf / rev) * 100.0, 2)

    except Exception:
        pass
    return out

def _leverage_and_cash(ticker: str) -> Dict[str, float]:
    """
    Debt/Equity samt Cash & Equivalents om möjligt (annual balance sheet).
    """
    out = {
        "Debt/Equity": 0.0,
        "Cash & Equivalents": 0.0,
        "Total Debt": 0.0
    }
    t = yf.Ticker(ticker)
    try:
        bs = t.balance_sheet
        if isinstance(bs, pd.DataFrame) and not bs.empty:
            idx = [str(x).strip() for x in bs.index]
            def v(name):
                return _f(bs.loc[name].dropna().iloc[-1], 0.0) if name in idx else 0.0
            total_debt = v("Total Debt") or v("Short Long Term Debt Total") or 0.0
            total_equity = v("Total Stockholder Equity") or v("StockholdersEquity") or 0.0
            cash = v("Cash") + v("Cash And Cash Equivalents") + v("CashAndCashEquivalents")
            out["Total Debt"] = round(total_debt, 2)
            out["Cash & Equivalents"] = round(cash, 2)
            out["Debt/Equity"] = round(_safe_div(total_debt, total_equity), 3) if total_equity != 0 else 0.0
    except Exception:
        pass
    return out

# ------------------------------------------------------------
# Offentliga runners
# ------------------------------------------------------------

def run_update_price_only(df: pd.DataFrame, ticker: str, user_rates: Dict[str, float]) -> Tuple[pd.DataFrame, List[str], str]:
    """
    Uppdaterar bara pris, valuta, namn, beta, market cap, sektor/industri.
    """
    if "Ticker" not in df.columns:
        raise ValueError("DataFrame saknar kolumnen 'Ticker'.")
    mask = df["Ticker"].astype(str).str.upper() == str(ticker).upper()
    if not mask.any():
        raise ValueError(f"{ticker} hittades inte i tabellen.")

    idx = df.index[mask][0]
    base = _pull_basics(ticker)

    new_vals = {
        "Bolagsnamn": base["Bolagsnamn"],
        "Valuta": base["Valuta"],
        "Aktuell kurs": base["Aktuell kurs"],
        "Årlig utdelning": base["Årlig utdelning"],
        "Beta": base["Beta"],
        "Sektor": base["Sektor"],
        "Industri": base["Industri"],
        "Utestående aktier": base["Utestående aktier"],
        "_MC": base["_MC"],
    }
    # Härled USD-mcap för risklabel
    try:
        if base["Valuta"].upper() != "USD" and base["_MC"] > 0:
            fx = _fx_rate(base["Valuta"].upper(), "USD")
            mc_usd = base["_MC"] * fx
        else:
            mc_usd = base["_MC"]
        new_vals["_MC_USD"] = float(mc_usd)
        new_vals["Risk"] = _risk_from_mcap_usd(mc_usd)
    except Exception:
        pass

    changed = _apply_to_row(df, idx, new_vals, source="Auto (Price-only via Yahoo)", stamp_even_if_same=True)
    return df, changed, "Kurs m.m. uppdaterad" if changed else "Inga ändringar"

def run_update_full(df: pd.DataFrame, ticker: str, user_rates: Dict[str, float]) -> Tuple[pd.DataFrame, List[str], str]:
    """
    Full uppdatering (yfinance-baserad):
    - Pris/valuta/namn/beta/mcap/shares/sektor/industri
    - P/S (nu) + P/S Q1..Q4
    - CAGR 5 år, marginaler (gross/net/EBITDA), FCF-margin
    - Debt/Equity, Cash, Total Debt
    - Risk (från _MC_USD)
    """
    if "Ticker" not in df.columns:
        raise ValueError("DataFrame saknar kolumnen 'Ticker'.")
    mask = df["Ticker"].astype(str).str.upper() == str(ticker).upper()
    if not mask.any():
        raise ValueError(f"{ticker} hittades inte i tabellen.")

    idx = df.index[mask][0]

    # Basdata
    base = _pull_basics(ticker)

    # P/S nu + Q1..Q4 via TTM på kvartalsintäkter och historiska priser
    ps = {}
    try:
        ps = _ps_series_from_quarters(ticker, base["Valuta"], base["_Shares"])
    except Exception:
        ps = {}

    # CAGR
    cagr = _cagr_5y_from_financials(ticker)

    # Marginaler
    margins = _margin_metrics(ticker)

    # Leverage & Cash
    lev = _leverage_and_cash(ticker)

    new_vals = {
        "Bolagsnamn": base["Bolagsnamn"],
        "Valuta": base["Valuta"],
        "Aktuell kurs": base["Aktuell kurs"],
        "Årlig utdelning": base["Årlig utdelning"],
        "Beta": base["Beta"],
        "Sektor": base["Sektor"],
        "Industri": base["Industri"],
        "Utestående aktier": base["Utestående aktier"],
        "_MC": base["_MC"],
        "CAGR 5 år (%)": cagr,
        **margins,
        **lev
    }
    # P/S fält
    for k, v in ps.items():
        new_vals[k] = float(v)

    # USD-mcap + Risk
    try:
        if base["Valuta"].upper() != "USD" and base["_MC"] > 0:
            fx = _fx_rate(base["Valuta"].upper(), "USD")
            mc_usd = base["_MC"] * fx
        else:
            mc_usd = base["_MC"]
        new_vals["_MC_USD"] = float(mc_usd)
        new_vals["Risk"] = _risk_from_mcap_usd(mc_usd)
    except Exception:
        pass

    changed = _apply_to_row(df, idx, new_vals, source="Auto (Full via Yahoo)", stamp_even_if_same=True)

    msg_parts = []
    if any(k.startswith("P/S") for k in ps.keys()):
        msg_parts.append("P/S-serie uppdaterad")
    if cagr:
        msg_parts.append(f"CAGR {cagr}%")
    if margins.get("Gross Margin (%)") or margins.get("Net Margin (%)"):
        msg_parts.append("Marginaler")
    if lev.get("Debt/Equity"):
        msg_parts.append("D/E")

    msg = ", ".join(msg_parts) if msg_parts else "Inga större fält ändrades"
    return df, changed, msg
