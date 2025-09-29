# stockapp/sources.py
# -*- coding: utf-8 -*-
"""
Källmodul för uppdateringar:
- run_update_price_only(df, ticker, user_rates) -> (df2, changed_fields, msg)
- run_update_full(df, ticker, user_rates)      -> (df2, changed_fields, msg)

Full använder Yahoo (yfinance) för:
  - Basdata: pris/valuta/namn/utdelning/CAGR(5y)
  - Utestående aktier (implied via marketCap/price, fallback sharesOutstanding)
  - Kvartalsintäkter -> TTM-fönster -> P/S (TTM) och P/S Q1–Q4 (historik)
OBS: Fälten "Omsättning idag" och "Omsättning nästa år" uppdateras inte här
      (manuella enligt din specifikation).
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta, date

import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf

from .config import TS_FIELDS

# ------------------------------------------------------------
# Små hjälpare
# ------------------------------------------------------------

def _now_stamp() -> str:
    # Stockholm om pytz finns, annars systemtid
    try:
        import pytz
        tz = pytz.timezone("Europe/Stockholm")
        return datetime.now(tz).strftime("%Y-%m-%d")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d")

def _ensure_row_exists(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Säkerställ att raden finns för tickern (case-insensitive jämförelse)."""
    mask = df["Ticker"].astype(str).str.upper() == str(ticker).upper()
    if not mask.any():
        # skapa tom rad med åtminstone Ticker
        empty_row = {c: (0.0 if c not in ["Ticker","Bolagsnamn","Valuta","Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa"] and not str(c).startswith("TS_") else "") for c in df.columns}
        empty_row["Ticker"] = str(ticker).upper()
        df = pd.concat([df, pd.DataFrame([empty_row])], ignore_index=True)
    return df

def _set_ts_for_field(df: pd.DataFrame, ridx: int, field: str, when: Optional[str] = None) -> None:
    ts_col = TS_FIELDS.get(field)
    if not ts_col:
        return
    try:
        df.at[ridx, ts_col] = when if when else _now_stamp()
    except Exception:
        pass

def _note_auto_update(df: pd.DataFrame, ridx: int, source: str) -> None:
    try:
        df.at[ridx, "Senast auto-uppdaterad"] = _now_stamp()
        df.at[ridx, "Senast uppdaterad källa"] = source
    except Exception:
        pass

def _apply_val(df: pd.DataFrame, ridx: int, field: str, value, changed: List[str], track_ts: bool = True) -> None:
    """Skriv nytt värde (om meningsfullt) och TS om ändrat."""
    if field not in df.columns:
        return
    old = df.at[ridx, field]
    # meningsfullt för numeriskt: > 0 för vissa fält, >=0 för pris etc.
    write_ok = False
    if isinstance(value, (int, float, np.floating)):
        if field in ["P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Utestående aktier"]:
            write_ok = (float(value) > 0)
        else:
            write_ok = (float(value) >= 0)
    elif isinstance(value, str):
        write_ok = (value.strip() != "")
    else:
        write_ok = value is not None

    if not write_ok:
        return

    if (pd.isna(old) and not pd.isna(value)) or (str(old) != str(value)):
        df.at[ridx, field] = value
        changed.append(field)
        if track_ts and field in TS_FIELDS:
            _set_ts_for_field(df, ridx, field)

# ------------------------------------------------------------
# Yahoo-hjälpare
# ------------------------------------------------------------

def _yfi_info(tkr: yf.Ticker) -> dict:
    try:
        return tkr.info or {}
    except Exception:
        return {}

def _yahoo_basics(ticker: str) -> dict:
    """
    Bolagsnamn, Valuta, Aktuell kurs, Årlig utdelning, CAGR(5y approx)
    """
    out = {"Bolagsnamn": "", "Valuta": "USD", "Aktuell kurs": 0.0, "Årlig utdelning": 0.0, "CAGR 5 år (%)": 0.0}
    try:
        t = yf.Ticker(ticker)
        info = _yfi_info(t)

        # Pris
        px = info.get("regularMarketPrice")
        if px is None:
            h = t.history(period="1d")
            if not h.empty and "Close" in h:
                px = float(h["Close"].iloc[-1])
        if px is not None:
            out["Aktuell kurs"] = float(px)

        # Valuta
        ccy = info.get("currency")
        if ccy:
            out["Valuta"] = str(ccy).upper()

        # Namn
        nm = info.get("shortName") or info.get("longName") or ""
        if nm:
            out["Bolagsnamn"] = str(nm)

        # Utdelning (årstakt)
        div_rate = info.get("dividendRate")
        if div_rate is not None:
            try:
                out["Årlig utdelning"] = float(div_rate)
            except Exception:
                pass

        # CAGR (5y approx) via income_stmt/financials Total Revenue
        out["CAGR 5 år (%)"] = _cagr_5y_from_financials(t)
    except Exception:
        pass
    return out

def _cagr_5y_from_financials(tkr: yf.Ticker) -> float:
    try:
        df_is = getattr(tkr, "income_stmt", None)
        series = None
        if isinstance(df_is, pd.DataFrame) and not df_is.empty and "Total Revenue" in df_is.index:
            series = df_is.loc["Total Revenue"].dropna()
        else:
            df_fin = getattr(tkr, "financials", None)
            if isinstance(df_fin, pd.DataFrame) and not df_fin.empty and "Total Revenue" in df_fin.index:
                series = df_fin.loc["Total Revenue"].dropna()
        if series is None or series.empty or len(series) < 2:
            return 0.0
        series = series.sort_index()
        start = float(series.iloc[0]); end = float(series.iloc[-1])
        years = max(1, len(series)-1)
        if start <= 0:
            return 0.0
        cagr = (end / start) ** (1.0/years) - 1.0
        return round(cagr * 100.0, 2)
    except Exception:
        return 0.0

def _yfi_quarterly_revenues(t: yf.Ticker) -> List[Tuple[date, float]]:
    """
    Försök hämta kvartalsintäkter nyast→äldst som [(period_end_date, value), ...]
    1) quarterly_financials (rad med 'Total Revenue' eller liknande)
    2) fallback income_stmt quarterly
    """
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
                    out: List[Tuple[date, float]] = []
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

    # 2) fallback income_stmt
    try:
        df_is = getattr(t, "income_stmt", None)
        if isinstance(df_is, pd.DataFrame) and not df_is.empty and "Total Revenue" in df_is.index:
            ser = df_is.loc["Total Revenue"].dropna()
            out: List[Tuple[date, float]] = []
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

def _ttm_windows(values: List[Tuple[date, float]], need: int = 4) -> List[Tuple[date, float]]:
    """
    Tar [(end_date, kvartalsintäkt), ...] nyast→äldst, returnerar upp till 'need' TTM-summor:
    [(end_date0, ttm0), (end_date1, ttm1), ...] där ttm0 = sum(q0..q3), ttm1 = sum(q1..q4), osv.
    """
    out: List[Tuple[date, float]] = []
    if len(values) < 4:
        return out
    for i in range(0, min(need, len(values) - 3)):
        end_i = values[i][0]
        ttm_i = sum(v for (_, v) in values[i:i+4])
        out.append((end_i, float(ttm_i)))
    return out

def _yahoo_prices_for_dates(tkr: str, dates: List[date]) -> Dict[date, float]:
    """
    Dagliga priser i fönster som täcker alla dates. Returnerar Close på eller närmast FÖRE respektive datum.
    """
    if not dates:
        return {}
    dmin = min(dates) - timedelta(days=14)
    dmax = max(dates) + timedelta(days=2)
    try:
        y = yf.Ticker(tkr)
        hist = y.history(start=dmin, end=dmax, interval="1d")
        if hist is None or hist.empty:
            return {}
        hist = hist.sort_index()
        idx_dates = list(hist.index.date)
        closes = list(hist["Close"].values)
        out: Dict[date, float] = {}
        for d in dates:
            px = None
            for j in range(len(idx_dates)-1, -1, -1):
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

def _implied_shares(info: dict, price: float) -> float:
    """
    Försök estimera antal utestående aktier via marketCap/price, fallback sharesOutstanding.
    Returnerar antal aktier (styck).
    """
    mcap = info.get("marketCap")
    try:
        mcap = float(mcap) if mcap is not None else 0.0
    except Exception:
        mcap = 0.0

    if mcap > 0 and price and price > 0:
        return mcap / float(price)

    so = info.get("sharesOutstanding")
    try:
        so = float(so or 0.0)
    except Exception:
        so = 0.0
    return so if so > 0 else 0.0


# ------------------------------------------------------------
# Publika runner-funktioner
# ------------------------------------------------------------

def run_update_price_only(
    df: pd.DataFrame,
    ticker: str,
    user_rates: Dict[str, float],
) -> Tuple[pd.DataFrame, List[str], str]:
    """
    Uppdaterar pris/valuta/namn/utdelning/CAGR. Sätter auto-stämplar.
    Returnerar (df2, changed_fields, msg).
    """
    tkr = str(ticker).upper().strip()
    df2 = _ensure_row_exists(df.copy(), tkr)
    ridx = df2.index[df2["Ticker"].astype(str).str.upper() == tkr][0]

    basics = _yahoo_basics(tkr)
    changed: List[str] = []

    # Skriv fält
    for k in ["Bolagsnamn", "Valuta", "Aktuell kurs", "Årlig utdelning", "CAGR 5 år (%)"]:
        _apply_val(df2, ridx, k, basics.get(k), changed, track_ts=False)

    _note_auto_update(df2, ridx, source="Auto (Yahoo pris)")
    msg = "Pris uppdaterat" if "Aktuell kurs" in changed else "Ingen prisförändring"
    return df2, changed, msg


def run_update_full(
    df: pd.DataFrame,
    ticker: str,
    user_rates: Dict[str, float],
) -> Tuple[pd.DataFrame, List[str], str]:
    """
    Full uppdatering via Yahoo:
      - Pris/valuta/namn/utdelning/CAGR(5y)
      - Utestående aktier (implied/fallback)
      - Kvartalsintäkter -> TTM -> P/S (TTM) + P/S Q1–Q4
    OBS: rör inte "Omsättning idag" eller "Omsättning nästa år".
    """
    tkr = str(ticker).upper().strip()
    df2 = _ensure_row_exists(df.copy(), tkr)
    ridx = df2.index[df2["Ticker"].astype(str).str.upper() == tkr][0]

    changed: List[str] = []

    # 1) Basics
    basics = _yahoo_basics(tkr)
    for k in ["Bolagsnamn", "Valuta", "Aktuell kurs", "Årlig utdelning", "CAGR 5 år (%)"]:
        _apply_val(df2, ridx, k, basics.get(k), changed, track_ts=False)

    price = float(df2.at[ridx, "Aktuell kurs"] or 0.0)
    info = _yfi_info(yf.Ticker(tkr))
    shares = _implied_shares(info, price)
    if shares > 0:
        # lagra i miljoner som i din app
        _apply_val(df2, ridx, "Utestående aktier", shares / 1e6, changed, track_ts=True)

    # 2) Kvartalsintäkter -> TTM
    try:
        y = yf.Ticker(tkr)
        q_rows = _yfi_quarterly_revenues(y)  # [(date, value), ...] nyast→äldst
    except Exception:
        q_rows = []

    ttm_list = _ttm_windows(q_rows, need=4)  # [(end_date, ttm_value), ...]
    # 3) P/S nu (TTM) och historik P/S Q1..Q4
    if ttm_list:
        # marketcap nu
        mcap = info.get("marketCap")
        try:
            mcap = float(mcap) if mcap is not None else 0.0
        except Exception:
            mcap = 0.0
        # fallback mcap via shares*price
        if (not mcap or mcap <= 0) and shares > 0 and price > 0:
            mcap = shares * price

        # P/S nu
        ltm0 = float(ttm_list[0][1])
        if mcap > 0 and ltm0 > 0:
            ps_now = mcap / ltm0
            _apply_val(df2, ridx, "P/S", ps_now, changed, track_ts=True)

        # P/S Q1..Q4 historik via historiska priser
        if shares > 0:
            q_dates = [d for (d, _) in ttm_list[:4]]
            px_map = _yahoo_prices_for_dates(tkr, q_dates)
            for idx, (d_end, ttm_rev) in enumerate(ttm_list[:4], start=1):
                px = px_map.get(d_end)
                if px and px > 0 and ttm_rev and ttm_rev > 0:
                    mcap_hist = shares * float(px)
                    ps_hist = mcap_hist / float(ttm_rev)
                    _apply_val(df2, ridx, f"P/S Q{idx}", ps_hist, changed, track_ts=True)

    # Stämpla auto
    _note_auto_update(df2, ridx, source="Auto (Yahoo full)")

    msg = "Uppdaterad"
    if not changed:
        msg = "Inga ändringar upptäcktes"
    return df2, changed, msg
