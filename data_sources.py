# data_sources.py
from __future__ import annotations
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

# -------------------- Tid / tidszon --------------------
try:
    import pytz
    TZ_STHLM = pytz.timezone("Europe/Stockholm")
    def ts_now():
        return datetime.now(TZ_STHLM).strftime("%Y-%m-%d %H:%M")
except Exception:
    def ts_now():
        return datetime.now().strftime("%Y-%m-%d %H:%M")


def _to_date(x) -> datetime | None:
    if isinstance(x, datetime):
        return x
    try:
        return pd.to_datetime(str(x)).to_pydatetime()
    except Exception:
        return None


# -------------------- SEC hjälp (enkla, secrets-baserade) --------------------
def _sec_company_ticker_to_cik(ticker: str) -> str | None:
    """
    Hämtar CIK via st.secrets["SEC_CIK_MAP"][TICKER] om du lagt den där.
    Ex:
      [SEC_CIK_MAP]
      NVDA = "0001045810"
    """
    return str(st.secrets.get("SEC_CIK_MAP", {}).get(ticker.upper(), "")) or None


def _sec_recent_shares_points(cik: str) -> list[dict]:
    """
    Manuell lista i secrets: st.secrets["SEC_SHARES_POINTS"][CIK] = [{date, shares}, ...]
    Används för att approximera utestående aktier nära kvartalet.
    """
    out = []
    for p in st.secrets.get("SEC_SHARES_POINTS", {}).get(cik or "", []):
        try:
            d = _to_date(p.get("date"))
            sh = float(p.get("shares"))
            if d and sh > 0:
                out.append({"date": d, "shares": sh})
        except Exception:
            pass
    return sorted(out, key=lambda x: x["date"], reverse=True)[:12]


def _sec_quarterly_revenue_points(cik: str) -> list[dict]:
    """
    Valfri manuell fallback: st.secrets["SEC_REVENUE_POINTS"][CIK] = [{date, revenue}, ...]
    """
    out = []
    for p in st.secrets.get("SEC_REVENUE_POINTS", {}).get(cik or "", []):
        try:
            d = _to_date(p.get("date"))
            r = float(p.get("revenue"))
            if d and r > 0:
                out.append({"date": d, "revenue": r})
        except Exception:
            pass
    return sorted(out, key=lambda x: x["date"], reverse=True)[:12]


# -------------------- Yahoo: pris & data --------------------
def _yahoo_fast_price(ticker: str) -> float | None:
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        p = info.get("regularMarketPrice", None)
        if p is None:
            h = t.history(period="1d")
            if not h.empty:
                p = float(h["Close"].iloc[-1])
        return float(p) if p is not None else None
    except Exception:
        return None


def _yahoo_price_on_or_after(ticker: str, d: datetime, window_days: int = 30) -> tuple[float | None, str]:
    """
    Stängningskurs första handelsdag >= d (upp till window_days).
    Returnerar (pris, prisdatum_str).
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
    DF med kolumner: date (datetime), revenue (float, USD), nyast först.
    Fyller ut från flera Yahoo-källor + valfritt secrets-SEC fallback.
    """
    t = yf.Ticker(ticker)
    buckets: dict = {}

    # 1) Primärt: quarterly_income_stmt / quarterly_financials (Total Revenue)
    for attr in ["quarterly_income_stmt", "quarterly_financials"]:
        try:
            df = getattr(t, attr, None)
            if isinstance(df, pd.DataFrame) and not df.empty and "Total Revenue" in df.index:
                ser = df.loc["Total Revenue"].dropna()
                for col, val in ser.items():
                    d = _to_date(col)
                    if d and pd.notna(val) and float(val) > 0:
                        buckets.setdefault(pd.to_datetime(d).date(), float(val))
                break
        except Exception:
            pass

    # 2) Fallback/komplement: quarterly_earnings['Revenue']
    try:
        qe = t.quarterly_earnings
        if isinstance(qe, pd.DataFrame) and not qe.empty:
            colname = None
            for c in qe.columns:
                if str(c).lower().strip() == "revenue":
                    colname = c
                    break
            if colname:
                for idx, row in qe.iterrows():
                    d = _to_date(idx)
                    rev = row.get(colname, None)
                    if d and rev is not None and float(rev) > 0:
                        buckets.setdefault(pd.to_datetime(d).date(), float(rev))
    except Exception:
        pass

    # 3) Valfritt: manuella SEC-punkter i secrets
    cik = _sec_company_ticker_to_cik(ticker)
    if cik:
        for p in _sec_quarterly_revenue_points(cik):
            buckets.setdefault(pd.to_datetime(p["date"]).date(), float(p["revenue"]))

    if not buckets:
        return pd.DataFrame(columns=["date", "revenue"])

    df = (
        pd.DataFrame([{"date": pd.to_datetime(d), "revenue": float(v)} for d, v in buckets.items()])
        .sort_values("date", ascending=False)
        .reset_index(drop=True)
    )
    df = df[df["revenue"] > 0]
    return df.head(12)


# -------------------- FY-etiketter --------------------
def _parse_fiscal_year_end_month(info: dict, qrev: pd.DataFrame) -> int:
    """
    Försök läsa Yahoo 'fiscalYearEnd' ('MMDD' som str). Om saknas, gissa.
    Returnerar månad (1..12) då räkenskapsåret slutar.
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
            1:  [1, 4, 7, 10],   # FY slutar i jan
            12: [12, 3, 6, 9],   # FY slutar i dec
            3:  [3, 6, 9, 12],   # FY slutar i mar
            7:  [7, 10, 1, 4],   # FY slutar i jul
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
    """Returnera ('FY25 Q2', 2) för ett kvartals slutdatum d och FY-slutmånad."""
    if not isinstance(d, datetime):
        d = _to_date(d)
    if d is None:
        return "—", 0

    fy_year = d.year if d.month <= fy_end_month else d.year + 1
    start_m = 1 if fy_end_month == 12 else fy_end_month + 1
    offset = ((d.year * 12 + d.month) - (d.year * 12 + start_m)) % 12
    q = offset // 3 + 1
    return f"FY{str(fy_year)[-2:]} Q{q}", int(q)


# -------------------- P/S-beräkning (TTM nu + TTM per kvartal) --------------------
def _shares_on_or_after(cik: str, d: datetime, yahoo_fallback: float | None, pts: list[dict]) -> float | None:
    """Närmaste aktiepunkt kring datum d; annars Yahoo implied shares."""
    if pts:
        pts_sorted = sorted(pts, key=lambda x: abs((x["date"] - d).days))
        if pts_sorted:
            return float(pts_sorted[0]["shares"])
    return yahoo_fallback


def _compute_ps_ttm_series(ticker: str) -> tuple[dict, dict]:
    """
    Returnerar bundle:
      {"ps_vals":{"P/S":...}, "meta":{...}, "ps_quarters":{1:{value,date,source,fy_label},...}}
    samt {"qrev": DataFrame}
    """
    t = yf.Ticker(ticker)

    qrev = _yahoo_quarterly_revenue(ticker)
    if not qrev.empty:
        qrev = qrev.sort_values("date", ascending=False).reset_index(drop=True)

    price_now = _yahoo_fast_price(ticker)

    info = {}
    try:
        info = t.info or {}
    except Exception:
        info = {}

    implied_shares = None
    try:
        finfo = t.fast_info or {}
        implied_shares = finfo.get("shares")
        if implied_shares is not None:
            implied_shares = float(implied_shares)
    except Exception:
        implied_shares = None

    cik = _sec_company_ticker_to_cik(ticker) or ""
    sec_pts = _sec_recent_shares_points(cik) if cik else []

    # P/S nu (TTM)
    p_s_now = None
    try:
        if len(qrev) >= 4 and price_now and (implied_shares or sec_pts):
            ttm_now = float(qrev.iloc[0:4]["revenue"].sum())
            shares_now = _shares_on_or_after(cik, _to_date(qrev.iloc[0]["date"]), implied_shares, sec_pts)
            if shares_now and shares_now > 0 and ttm_now > 0:
                p_s_now = float(price_now * shares_now / ttm_now)
    except Exception:
        p_s_now = None

    fy_end_month = _parse_fiscal_year_end_month(info, qrev)
    ps_quarters = {}
    for qi in range(4):
        if len(qrev) >= qi + 4:
            qdate = _to_date(qrev.iloc[qi]["date"])
            fy_lbl, qnum = _fy_quarter_label(qdate, fy_end_month)

            ttm = float(qrev.iloc[qi:qi+4]["revenue"].sum())
            price_q, price_dt = _yahoo_price_on_or_after(ticker, qdate + timedelta(days=1))
            if price_q is None:
                price_q, price_dt = price_now, "now"

            shares_q = _shares_on_or_after(cik, qdate, implied_shares, sec_pts)

            ps_val = None
            if price_q and shares_q and shares_q > 0 and ttm > 0:
                ps_val = float(price_q * shares_q / ttm)
                if ps_val < 0 or ps_val > 1000:
                    ps_val = None

            ps_quarters[qi+1] = {
                "value": round(ps_val, 2) if ps_val is not None else 0.0,
                "date": qdate.strftime("%Y-%m-%d") if qdate else "–",
                "source": f"Computed/{fy_lbl}/price@{price_dt}" if ps_val is not None else "n/a",
                "fy_label": fy_lbl,
            }
        else:
            ps_quarters[qi+1] = {"value": 0.0, "date": "–", "source": "n/a", "fy_label": "—"}

    ps_vals = {"P/S": round(p_s_now, 2) if p_s_now is not None else 0.0}
    meta = {
        "ps_source": "yahoo_ps_ttm",
        "q_cols": len(qrev),
        "sec_cik": cik,
        "sec_shares_pts": len(sec_pts),
    }
    return {"ps_vals": ps_vals
