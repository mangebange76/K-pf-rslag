# stockapp/sources.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Dict, List, Tuple, Optional
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import requests

# ----------------------------
# Cache & små utils
# ----------------------------

def _safe_float(x, d=0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return d
        return float(x)
    except Exception:
        return d

def _now_stamp() -> str:
    try:
        import pytz
        tz = pytz.timezone("Europe/Stockholm")
        return pd.Timestamp.now(tz).strftime("%Y-%m-%d")
    except Exception:
        return pd.Timestamp.now().strftime("%Y-%m-%d")

def _ensure_col(df: pd.DataFrame, col: str, numeric_default: float = 0.0):
    if col not in df.columns:
        if any(k in col.lower() for k in ["kurs","omsättning","p/s","utdelning","cagr","antal","riktkurs","aktier","snitt","market cap","marginal","debt","cash","fcf","runway"]):
            df[col] = numeric_default
        else:
            df[col] = ""

def _ts_col_for(field: str) -> Optional[str]:
    # TS-kolumner enligt din app
    mapping = {
        "Utestående aktier": "TS_Utestående aktier",
        "P/S": "TS_P/S",
        "P/S Q1": "TS_P/S Q1",
        "P/S Q2": "TS_P/S Q2",
        "P/S Q3": "TS_P/S Q3",
        "P/S Q4": "TS_P/S Q4",
        "Omsättning idag": "TS_Omsättning idag",
        "Omsättning nästa år": "TS_Omsättning nästa år",
    }
    return mapping.get(field)

def _stamp_ts(df: pd.DataFrame, ridx: int, field: str):
    ts = _ts_col_for(field)
    if not ts:
        return
    _ensure_col(df, ts, 0.0)  # textfält men default behövs
    df.at[ridx, ts] = _now_stamp()

# ----------------------------
# FX (Frankfurter -> exchangerate.host)
# ----------------------------

@st.cache_data(show_spinner=False, ttl=21600)
def _fx_rate(base: str, quote: str) -> float:
    base = (base or "").upper(); quote = (quote or "").upper()
    if not base or not quote or base == quote:
        return 1.0
    try:
        r = requests.get("https://api.frankfurter.app/latest", params={"from": base, "to": quote}, timeout=10)
        if r.status_code == 200:
            v = (r.json() or {}).get("rates", {}).get(quote)
            if v: return float(v)
    except Exception:
        pass
    try:
        r = requests.get("https://api.exchangerate.host/latest", params={"base": base, "symbols": quote}, timeout=10)
        if r.status_code == 200:
            v = (r.json() or {}).get("rates", {}).get(quote)
            if v: return float(v)
    except Exception:
        pass
    return 1.0

# ----------------------------
# Yahoo helpers
# ----------------------------

def _y_ticker(ticker: str) -> yf.Ticker:
    return yf.Ticker(str(ticker).strip())

def _y_info(t: yf.Ticker) -> dict:
    try:
        return t.info or {}
    except Exception:
        return {}

def _y_hist_close_on_or_before(ticker: str, dates: List[date]) -> Dict[date, float]:
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
        idx = list(hist.index.date)
        closes = list(hist["Close"].values)
        out = {}
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

def _q_fin_row(df: pd.DataFrame, *keys: str) -> Optional[pd.Series]:
    """Hämta en rad ur quarterly_financials via kandidatnycklar."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    idx = [str(x).strip() for x in df.index]
    for k in keys:
        if k in idx:
            return df.loc[k].dropna()
    return None

def _quarterly_revenue_series(t: yf.Ticker) -> List[Tuple[date, float]]:
    try:
        qf = t.quarterly_financials
        row = _q_fin_row(qf,
                         "Total Revenue","TotalRevenue","Revenues","Revenue","Sales",
                         "Revenues from contracts with customers","Revenues From Contracts With Customers")
        if row is not None and not row.empty:
            out = []
            for c, v in row.items():
                try:
                    d = c.date() if hasattr(c, "date") else pd.to_datetime(c).date()
                    out.append((d, float(v)))
                except Exception:
                    pass
            out.sort(key=lambda x: x[0], reverse=True)
            # deduplicera på år/kvartal (hantera dec/jan överlapp)
            ded = {}
            for d, v in out:
                q = (d.year, ((d.month-1)//3)+1)
                if q not in ded:
                    ded[q] = (d, v)
            return [ded[k] for k in sorted(ded.keys(), reverse=True)]
    except Exception:
        pass
    return []

def _ttm_windows(values: List[Tuple[date,float]], need: int = 6) -> List[Tuple[date, float]]:
    """Bygg TTM-summor nyast→äldst."""
    out = []
    if len(values) < 4:
        return out
    for i in range(0, min(need, len(values)-3)):
        end_i = values[i][0]
        ttm_i = sum(v for (_, v) in values[i:i+4])
        out.append((end_i, float(ttm_i)))
    return out

# ----------------------------
# Nyckeltal från Yahoo
# ----------------------------

def fetch_yahoo_basics(ticker: str) -> Dict:
    t = _y_ticker(ticker)
    info = _y_info(t)
    out = {
        "Bolagsnamn": info.get("shortName") or info.get("longName") or "",
        "Valuta": (info.get("currency") or "USD").upper(),
        "Aktuell kurs": _safe_float(info.get("regularMarketPrice"), 0.0),
        "Årlig utdelning": _safe_float(info.get("dividendRate"), 0.0),
        "Sektor": info.get("sector") or info.get("industry") or "",
        "Market Cap (nu)": _safe_float(info.get("marketCap"), 0.0),
    }
    if out["Aktuell kurs"] <= 0:
        try:
            h = t.history(period="1d")
            if not h.empty and "Close" in h:
                out["Aktuell kurs"] = float(h["Close"].iloc[-1])
        except Exception:
            pass
    # Shares (implied)
    price = out["Aktuell kurs"]
    mcap = out["Market Cap (nu)"]
    if mcap > 0 and price > 0:
        out["Utestående aktier"] = mcap / price / 1e6  # i miljoner
    else:
        so = _safe_float(info.get("sharesOutstanding"), 0.0)
        if so > 0:
            out["Utestående aktier"] = so / 1e6
    return out

def fetch_yahoo_fundamentals(ticker: str) -> Dict:
    """Hämta D/E, marginaler, FCF TTM, Kassa."""
    t = _y_ticker(ticker)
    info = _y_info(t)
    fin_ccy = (info.get("financialCurrency") or info.get("currency") or "USD").upper()

    # Marginaler & D/E från financials / balance sheet
    gm_pct, nm_pct, de = 0.0, 0.0, 0.0
    try:
        fin = t.financials  # annual
        if isinstance(fin, pd.DataFrame) and not fin.empty:
            gp = _safe_float(fin.loc.get("Gross Profit", pd.Series()).dropna().iloc[0] if "Gross Profit" in fin.index else None, 0.0)
            rev = _safe_float(fin.loc.get("Total Revenue", pd.Series()).dropna().iloc[0] if "Total Revenue" in fin.index else None, 0.0)
            ni = _safe_float(fin.loc.get("Net Income", pd.Series()).dropna().iloc[0] if "Net Income" in fin.index else None, 0.0)
            if rev > 0:
                gm_pct = round(gp/rev * 100.0, 2) if gp>0 else gm_pct
                nm_pct = round(ni/rev * 100.0, 2)
    except Exception:
        pass

    try:
        bs = t.balance_sheet
        if isinstance(bs, pd.DataFrame) and not bs.empty:
            td = _safe_float(bs.loc.get("Total Debt", pd.Series()).dropna().iloc[0] if "Total Debt" in bs.index else None, 0.0)
            eq = _safe_float(bs.loc.get("Total Stockholder Equity", pd.Series()).dropna().iloc[0] if "Total Stockholder Equity" in bs.index else None, 0.0)
            cash = _safe_float(bs.loc.get("Cash And Cash Equivalents", pd.Series()).dropna().iloc[0] if "Cash And Cash Equivalents" in bs.index else None, 0.0)
            kassa = cash
            de = (td/eq) if (eq and eq != 0) else de
        else:
            kassa = 0.0
    except Exception:
        kassa = 0.0

    # FCF TTM ~ Operating CF - CapEx (TTM om möjligt, annars senaste)
    fcf_ttm = 0.0
    try:
        cf = t.cashflow
        if isinstance(cf, pd.DataFrame) and not cf.empty:
            ocf = _safe_float(cf.loc.get("Total Cash From Operating Activities", pd.Series()).dropna().iloc[0]
                              if "Total Cash From Operating Activities" in cf.index else None, 0.0)
            capex = _safe_float(cf.loc.get("Capital Expenditures", pd.Series()).dropna().iloc[0]
                                if "Capital Expenditures" in cf.index else None, 0.0)
            fcf_ttm = ocf - capex
    except Exception:
        pass

    # Runway ≈ kassa / |negativ FCF| * 12 (om FCF < 0)
    runway_m = 0.0
    if fcf_ttm < 0:
        yearly_burn = abs(fcf_ttm)
        runway_m = round((kassa / yearly_burn) * 12.0, 1) if yearly_burn > 0 else 0.0
    elif fcf_ttm >= 0 and kassa > 0:
        runway_m = 60.0  # "lång runway" när FCF positivt

    return {
        "Bruttomarginal (%)": gm_pct,
        "Nettomarginal (%)": nm_pct,
        "Debt/Equity": round(float(de), 2) if de else 0.0,
        "Kassa (valuta)": float(kassa),
        "FCF TTM (valuta)": float(fcf_ttm),
        "Runway (mån)": float(runway_m),
        "Financial Currency": fin_ccy
    }

def fetch_ps_series(ticker: str) -> Dict:
    """
    Beräkna P/S (TTM) nu + P/S Q1..Q4 via Yahoo quarterly_financials + historiska priser.
    Hanterar dec/jan-överlapp via deduplicering per kvartal.
    """
    t = _y_ticker(ticker)
    info = _y_info(t)
    px_ccy = (info.get("currency") or "USD").upper()
    fin_ccy = (info.get("financialCurrency") or px_ccy).upper()

    # Revenue TTM-lista
    q_rows = _quarterly_revenue_series(t)
    if not q_rows or len(q_rows) < 4:
        return {}
    ttm_list = _ttm_windows(q_rows, need=5)  # få fram minst 4 fönster + 1 extra för robusthet

    # FX till prisvaluta
    conv = _fx_rate(fin_ccy, px_ccy) if fin_ccy != px_ccy else 1.0
    ttm_px = [(d, v*conv) for (d, v) in ttm_list]

    # Shares (implied)
    price_now = _safe_float(info.get("regularMarketPrice"), 0.0)
    mcap_now  = _safe_float(info.get("marketCap"), 0.0)
    shares = 0.0
    if mcap_now > 0 and price_now > 0:
        shares = mcap_now / price_now
    else:
        so = _safe_float(info.get("sharesOutstanding"), 0.0)
        if so > 0: shares = so

    # P/S nu
    out = {}
    if mcap_now > 0 and ttm_px:
        latest_ttm = ttm_px[0][1]
        if latest_ttm > 0:
            out["P/S"] = float(mcap_now / latest_ttm)

    # Historiska P/S vid kvartals-slut
    dates = [d for (d, _) in ttm_px[:4]]
    px_map = _y_hist_close_on_or_before(ticker, dates)
    if shares > 0:
        for i, (d_end, ttm) in enumerate(ttm_px[:4], start=1):
            px = _safe_float(px_map.get(d_end), 0.0)
            if px > 0 and ttm > 0:
                mcap_hist = shares * px
                out[f"P/S Q{i}"] = float(mcap_hist / ttm)

    # P/S-snitt
    ps_vals = [out.get(f"P/S Q{i}", 0.0) for i in range(1,5)]
    ps_valid = [v for v in ps_vals if _safe_float(v,0.0) > 0]
    if ps_valid:
        out["P/S-snitt"] = round(float(np.mean(ps_valid)), 2)

    return out

# ----------------------------
# Hög-nivå: hämta "allt" för ticker
# ----------------------------

def fetch_all_keys_for_ticker(ticker: str) -> Dict:
    base = fetch_yahoo_basics(ticker)
    ps   = fetch_ps_series(ticker)
    fun  = fetch_yahoo_fundamentals(ticker)
    out = {}
    out.update(base); out.update(ps); out.update(fun)
    return out

# ----------------------------
# DF-helpers (skriva värden + stämpla)
# ----------------------------

def apply_changes(df: pd.DataFrame, ticker: str, new_vals: Dict, *, source: str = "Auto (Yahoo)") -> Tuple[pd.DataFrame, List[str]]:
    """
    Skriv new_vals till raden för ticker (skapar kolumner vid behov).
    Stämplar TS_* för spårade fält och sätter 'Senast auto-uppdaterad' + källa.
    Stämplar även om värdet är oförändrat (enligt din önskan).
    """
    if "Ticker" not in df.columns:
        df["Ticker"] = ""
    mask = df["Ticker"].astype(str).str.upper() == str(ticker).upper()
    if not mask.any():
        # skapa ny rad
        row = {c: "" for c in df.columns}
        row["Ticker"] = ticker.upper()
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        mask = df["Ticker"].astype(str).str.upper() == str(ticker).upper()

    ridx = df.index[mask][0]

    changed_fields: List[str] = []

    for k, v in new_vals.items():
        _ensure_col(df, k)
        old = df.at[ridx, k]
        # skriv alltid
        df.at[ridx, k] = v
        changed_fields.append(k)
        # TS-stämpel för spårade fält
        _stamp_ts(df, ridx, k)

    # auto-meta
    _ensure_col(df, "Senast auto-uppdaterad")
    _ensure_col(df, "Senast uppdaterad källa")
    df.at[ridx, "Senast auto-uppdaterad"] = _now_stamp()
    df.at[ridx, "Senast uppdaterad källa"] = source
    return df, changed_fields

# ----------------------------
# Offentliga "runners"
# ----------------------------

def update_price_only(df: pd.DataFrame, ticker: str) -> Tuple[pd.DataFrame, Dict]:
    base = fetch_yahoo_basics(ticker)
    vals = {k: base[k] for k in ["Aktuell kurs"] if k in base and _safe_float(base[k], -1) >= 0}
    df2, changed = apply_changes(df, ticker, vals, source="Kurs (Yahoo)")
    return df2, {"changed": changed, "source": "Kurs (Yahoo)"}

def update_full_for_ticker(df: pd.DataFrame, ticker: str) -> Tuple[pd.DataFrame, Dict]:
    allv = fetch_all_keys_for_ticker(ticker)
    df2, changed = apply_changes(df, ticker, allv, source="Full auto (Yahoo)")
    return df2, {"changed": changed, "source": "Full auto (Yahoo)", "payload_keys": list(allv.keys())}
