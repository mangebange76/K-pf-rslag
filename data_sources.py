# data_sources.py
from __future__ import annotations

import re
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ------------------------------------------------------------
# Tidsstämplar
# ------------------------------------------------------------
def _now_ts() -> str:
    try:
        import pytz
        tz = pytz.timezone("Europe/Stockholm")
        return datetime.now(tz).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d %H:%M")

def _to_date(x) -> datetime | None:
    try:
        return pd.to_datetime(x).to_pydatetime()
    except Exception:
        return None

# ------------------------------------------------------------
# SEC – läsning från secrets (offline-kompatibelt)
# ------------------------------------------------------------
def _sec_company_ticker_to_cik(ticker: str) -> str | None:
    tkr = (ticker or "").upper().strip()
    # Tillåt overrides i secrets
    for key in ("SEC_CIK_OVERRIDES", "SEC_CIK"):
        mp = st.secrets.get(key, {})
        if isinstance(mp, dict):
            v = mp.get(tkr) or mp.get(tkr.lower())
            if v:
                return str(v).zfill(10)
    return None

def _sec_points_from_secrets(key: str, cik: str) -> list[dict]:
    data = st.secrets.get(key, {})
    if not isinstance(data, dict):
        return []
    arr = data.get(str(cik)) or data.get(str(cik).zfill(10)) or []
    if not isinstance(arr, list):
        return []
    out = []
    for it in arr:
        try:
            d = _to_date(it.get("date"))
            v = float(it.get("value") or it.get("shares") or it.get("revenue"))
            out.append({"date": d, "value": v})
        except Exception:
            pass
    out.sort(key=lambda x: x["date"] or datetime(1970, 1, 1))
    return out

def _sec_recent_shares_points(cik: str) -> list[dict]:
    # list of {"date": datetime, "value": shares_count}
    return _sec_points_from_secrets("SEC_SHARES_POINTS", cik)

def _sec_recent_revenue_points(cik: str) -> list[dict]:
    # list of {"date": datetime, "value": revenue_amount}
    return _sec_points_from_secrets("SEC_REVENUE_POINTS", cik)

def _sec_filing_links_from_secrets(ticker: str) -> list[dict]:
    mp = st.secrets.get("SEC_FILING_LINKS", {})
    if not isinstance(mp, dict):
        return []
    arr = mp.get(ticker.upper(), [])
    if not isinstance(arr, list):
        return []
    # förväntat: [{"form":"10-Q","date":"2025-08-27","viewer":"...","url":"...","cik":"0001045810"}]
    return arr

# ------------------------------------------------------------
# Yahoo – robust pris/kurser/financials
# ------------------------------------------------------------
def _yahoo_fast_price(ticker: str) -> float | None:
    try:
        t = yf.Ticker(ticker)
        finfo = getattr(t, "fast_info", {}) or {}
        p = finfo.get("last_price") or finfo.get("lastPrice")
        if p:
            return float(p)
        info = getattr(t, "info", {}) or {}
        p = info.get("regularMarketPrice")
        if p:
            return float(p)
        h = t.history(period="1d")
        if not h.empty:
            return float(h["Close"].iloc[-1])
    except Exception:
        pass
    return None

def _yahoo_price_on_or_after(ticker: str, d: datetime, window_days: int = 30) -> tuple[float | None, str]:
    """
    Stängningskurs första handelsdag >= d (upp till window_days). Returnerar (pris, prisdatum_str).
    Mindre känslig för helger/helgdagar och tidszoner.
    """
    try:
        start = (d - timedelta(days=3)).strftime("%Y-%m-%d")
        end   = (d + timedelta(days=window_days)).strftime("%Y-%m-%d")
        t = yf.Ticker(ticker)
        h = t.history(start=start, end=end, interval="1d", auto_adjust=False)
        if h.empty:
            return None, ""
        df = h.reset_index()
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        target = pd.to_datetime(d).tz_localize(None)
        row = df[df["Date"] >= target].head(1)
        if row.empty:
            row = df.sort_values("Date").tail(1)
        price = float(row["Close"].iloc[0])
        pdate = row["Date"].iloc[0].strftime("%Y-%m-%d")
        return price, pdate
    except Exception:
        return None, ""

def _yahoo_quarterly_revenue(ticker: str) -> pd.DataFrame:
    """
    Returnerar DataFrame med kolumner: ['date','revenue'] (nyast först).
    Källa: yfinance quarterly_income_stmt/quarterly_financials (Total Revenue).
    """
    t = yf.Ticker(ticker)
    series = None
    # Nyare yfinance
    try:
        qis = getattr(t, "quarterly_income_stmt", None)
        if isinstance(qis, pd.DataFrame) and not qis.empty:
            idx = [i.lower().replace(" ", "") for i in qis.index]
            # hitta "Total Revenue"
            if "totalrevenue" in idx:
                key = qis.index[idx.index("totalrevenue")]
                series = qis.loc[key]
    except Exception:
        pass
    # Äldre fallback
    if series is None:
        try:
            qfin = getattr(t, "quarterly_financials", None)
            if isinstance(qfin, pd.DataFrame) and not qfin.empty:
                idx = [i.lower().replace(" ", "") for i in qfin.index]
                if "totalrevenue" in idx:
                    key = qfin.index[idx.index("totalrevenue")]
                    series = qfin.loc[key]
        except Exception:
            pass
    if series is None or series.empty:
        return pd.DataFrame(columns=["date","revenue"])

    # series: kolumner = datum
    ser = series.dropna()
    df = ser.to_frame(name="revenue").T if isinstance(ser.name, str) else ser.to_frame(name="revenue")
    # När yfinance ger (index=lineitems, columns=datums) → vi vill transponera
    if "revenue" not in df.columns or len(df.columns) > 1:
        df = ser.to_frame().reset_index()
        df.columns = ["date","revenue"]
    else:
        df = df.T.reset_index()
        df.columns = ["date","revenue"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date", ascending=False).reset_index(drop=True)
    return df[["date","revenue"]]

# ------------------------------------------------------------
# Kvartal/FY-etiketter & aktier-val
# ------------------------------------------------------------
def _parse_fiscal_year_end_month(info: dict, qrev: pd.DataFrame) -> int:
    """
    Försök läsa Yahoo 'fiscalYearEnd' (t.ex. '0131' -> 1). Om saknas, gissa av mönster.
    """
    try:
        fye = info.get("fiscalYearEnd", None) if isinstance(info, dict) else None
        if fye:
            s = str(fye).strip()
            if len(s) >= 2 and s[:2].isdigit():
                m = int(s[:2])
                if 1 <= m <= 12:
                    return m
    except Exception:
        pass

    if isinstance(qrev, pd.DataFrame) and not qrev.empty:
        months = qrev["date"].dt.month.tolist()
        patterns = {
            1:  [1,4,7,10],
            12: [12,3,6,9],
            3:  [3,6,9,12],
            7:  [7,10,1,4],
        }
        best_m = 1
        best_hits = -1
        for m_end, pats in patterns.items():
            hits = sum(1 for m in months if any(((m - p) % 12 == 0) for p in pats))
            if hits > best_hits:
                best_hits, best_m = hits, m_end
        return best_m

    return 1

def _fy_quarter_label(d: datetime, fy_end_month: int) -> tuple[str, int]:
    """Ex: ('FY25 Q2', 2) givet kvartalsdatum d och FYE-månad."""
    if not isinstance(d, datetime):
        d = _to_date(d)
    if d is None:
        return "—", 0
    fy_year = d.year if d.month <= fy_end_month else d.year + 1
    start_m = 1 if fy_end_month == 12 else fy_end_month + 1
    offset = ((d.year * 12 + d.month) - (d.year * 12 + start_m)) % 12
    q = offset // 3 + 1
    return f"FY{str(fy_year)[-2:]} Q{q}", int(q)

def _find_point_on_or_after(points: list[dict], d: datetime) -> float | None:
    """Returnera första punktens value med datum >= d, annars None."""
    if not points:
        return None
    for it in points:
        if it["date"] and it["date"] >= d:
            return float(it["value"])
    return None

def _shares_on_or_after(cik: str, d: datetime, implied: float | None, sec_pts: list[dict]) -> float | None:
    """
    Välj aktieantal vid datum d: SEC-punkt (>= d+1d) före Yahoo-implied.
    Returnerar *antal aktier* (ej miljoner).
    """
    if d is None:
        return implied
    # SEC: välj dag efter kvartalsslut (filing-period + 1 dag)
    sec_val = _find_point_on_or_after(sec_pts, d + timedelta(days=1))
    if sec_val and sec_val > 0:
        return float(sec_val)
    return implied

# ------------------------------------------------------------
# P/S-beräkning: TTM nu + fyra senaste TTM per kvartal
# ------------------------------------------------------------
def _compute_ps_ttm_series(ticker: str) -> dict:
    t = yf.Ticker(ticker)

    # 1) Revenue-kvartal (nyast först)
    qrev = _yahoo_quarterly_revenue(ticker)
    if not qrev.empty:
        qrev = qrev.sort_values("date", ascending=False).reset_index(drop=True)

    # 2) Pris nu + info (för FYE)
    price_now = _yahoo_fast_price(ticker)
    try:
        info = t.info or {}
    except Exception:
        info = {}

    # 3) Yahoo implied shares
    implied_shares = None
    source_sh = ""
    try:
        finfo = t.fast_info or {}
        implied_shares = finfo.get("shares")
        if implied_shares is not None:
            implied_shares = float(implied_shares)
            source_sh = "Yahoo/fast_info"
    except Exception:
        implied_shares = None
    if implied_shares is None:
        try:
            s = (t.info or {}).get("sharesOutstanding", None)
            if s:
                implied_shares = float(s)
                source_sh = "Yahoo/info"
        except Exception:
            pass

    # 4) SEC seed
    cik = _sec_company_ticker_to_cik(ticker) or ""
    sec_sh_pts = _sec_recent_shares_points(cik) if cik else []

    # 5) P/S (TTM) nu
    ps_now = 0.0
    ts_ps = _now_ts()
    if len(qrev) >= 4 and price_now and
