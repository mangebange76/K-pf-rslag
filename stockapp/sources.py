# stockapp/sources.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import List, Tuple, Dict, Optional

import pandas as pd
import numpy as np
import yfinance as yf

from .config import TS_FIELDS
from .utils import now_stamp

# ---------------------------------------------------------------------
# Hjälpmetoder för uppdatering & stämpling
# ---------------------------------------------------------------------

AUTO_SOURCE_PRICE = "Auto (yfinance: price/basic)"
AUTO_SOURCE_FULL  = "Auto (yfinance: price+financials)"

def _stamp_ts(df: pd.DataFrame, row_idx: int, field: str, date_str: Optional[str] = None):
    """Sätt TS-kolumn för ett spårat fält om det finns i TS_FIELDS."""
    ts_col = TS_FIELDS.get(field)
    if not ts_col:
        return
    try:
        df.at[row_idx, ts_col] = date_str or now_stamp()
    except Exception:
        pass

def _note_auto_update(df: pd.DataFrame, row_idx: int, source: str):
    """Uppdatera 'Senast auto-uppdaterad' & 'Senast uppdaterad källa'."""
    try:
        df.at[row_idx, "Senast auto-uppdaterad"] = now_stamp()
        df.at[row_idx, "Senast uppdaterad källa"] = source
    except Exception:
        pass

def _safe_set(df: pd.DataFrame, row_idx: int, field: str, value, changed: List[str], force_stamp: bool = True):
    """
    Sätt fältet, registrera 'changed' om värdet faktiskt ändras.
    TS stämplas alltid (force_stamp=True), även om värdet inte ändras – enligt din preferens.
    """
    if field not in df.columns:
        return
    old = df.at[row_idx, field]
    same = False
    try:
        same = (pd.isna(old) and pd.isna(value)) or (str(old) == str(value))
    except Exception:
        same = False

    df.at[row_idx, field] = value
    if not same:
        changed.append(field)

    if force_stamp and field in TS_FIELDS:
        _stamp_ts(df, row_idx, field)

def _coerce_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def _mean_pos(values: List[float]) -> float:
    clean = [float(v) for v in values if _coerce_float(v, 0.0) > 0]
    return round(float(np.mean(clean)), 2) if clean else 0.0

# ---------------------------------------------------------------------
# yfinance-hjälpare
# ---------------------------------------------------------------------

def _yfi_info(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}

def _yfi_history_close_on_or_before(ticker: str, date: pd.Timestamp) -> Optional[float]:
    """Hämta dagsstängning på eller närmast FÖRE 'date'."""
    try:
        start = (pd.to_datetime(date) - pd.Timedelta(days=30)).date()
        end   = (pd.to_datetime(date) + pd.Timedelta(days=2)).date()
        hist = yf.Ticker(ticker).history(start=start, end=end, interval="1d")
        if hist is None or hist.empty:
            return None
        hist = hist.sort_index()
        idx_dates = list(hist.index.date)
        closes = list(hist["Close"].values)
        for j in range(len(idx_dates)-1, -1, -1):
            if idx_dates[j] <= pd.to_datetime(date).date():
                try:
                    return float(closes[j])
                except Exception:
                    return None
        return None
    except Exception:
        return None

def _yfi_quarterly_revenue_rows(ticker: str) -> List[Tuple[pd.Timestamp, float]]:
    """
    Läser quarterly_financials och letar efter en rad som motsvarar 'Total Revenue' eller motsv.
    Returnerar [(period_end, revenue_value), ...] sorterat nyast→äldst.
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
                            d = (c.to_pydatetime() if hasattr(c, "to_pydatetime") else pd.to_datetime(c)).normalize()
                            out.append((d, float(v)))
                        except Exception:
                            pass
                    out.sort(key=lambda x: x[0], reverse=True)
                    return out
    except Exception:
        pass

    # 2) fallback: income_stmt quarterly via v1-api (kan ibland finnas)
    try:
        df_is = getattr(t, "income_stmt", None)
        if isinstance(df_is, pd.DataFrame) and not df_is.empty and "Total Revenue" in df_is.index:
            ser = df_is.loc["Total Revenue"].dropna()
            out = []
            for c, v in ser.items():
                try:
                    d = (c.to_pydatetime() if hasattr(c, "to_pydatetime") else pd.to_datetime(c)).normalize()
                    out.append((d, float(v)))
                except Exception:
                    pass
            out.sort(key=lambda x: x[0], reverse=True)
            return out
    except Exception:
        pass

    return []

def _ttm_windows(values: List[Tuple[pd.Timestamp, float]], need: int = 4) -> List[Tuple[pd.Timestamp, float]]:
    """
    Ta [(end_date, q_rev), ...] (nyast→äldst) och bygg TTM-summor:
    [(end_date0, ttm0), (end_date1, ttm1), ...] där ttm0 = q0+q1+q2+q3, ttm1 = q1+q2+q3+q4, osv.
    """
    out: List[Tuple[pd.Timestamp, float]] = []
    if len(values) < 4:
        return out
    for i in range(0, min(need, len(values) - 3)):
        end_i = values[i][0]
        ttm_i = sum(v for (_, v) in values[i:i+4])
        out.append((end_i, float(ttm_i)))
    return out

# ---------------------------------------------------------------------
# Härledda beräkningar (per rad)
# ---------------------------------------------------------------------

def _recompute_row_derivatives(df: pd.DataFrame, ridx: int):
    """
    Beräkna P/S-snitt från Q1..Q4 samt riktkurser idag/1/2/3 för just denna rad.
    Antar att:
      - 'Omsättning idag' / 'Omsättning nästa år' är i miljoner (bolagets valuta)
      - 'Utestående aktier' är i miljoner
    """
    ps_q = [df.at[ridx, c] if c in df.columns else 0.0 for c in ["P/S Q1","P/S Q2","P/S Q3","P/S Q4"]]
    ps_avg = _mean_pos([_coerce_float(x, 0.0) for x in ps_q])
    df.at[ridx, "P/S-snitt"] = ps_avg

    shares_m = _coerce_float(df.at[ridx, "Utestående aktier"], 0.0)
    if shares_m <= 0 or ps_avg <= 0:
        # kan inte räkna riktkurser utan dessa
        for c in ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"]:
            if c in df.columns:
                df.at[ridx, c] = 0.0
        return

    # Omsättning (M)
    rev_today_m   = _coerce_float(df.at[ridx, "Omsättning idag"], 0.0)
    rev_next_m    = _coerce_float(df.at[ridx, "Omsättning nästa år"], 0.0)
    rev_2y_m      = _coerce_float(df.at[ridx, "Omsättning om 2 år"], 0.0)
    rev_3y_m      = _coerce_float(df.at[ridx, "Omsättning om 3 år"], 0.0)

    # Fyll 2y/3y om tomt med enkel CAGR clamp från 'CAGR 5 år (%)'
    if rev_next_m > 0 and (rev_2y_m <= 0 or rev_3y_m <= 0):
        cagr = _coerce_float(df.at[ridx, "CAGR 5 år (%)"], 0.0)
        if cagr > 100.0:
            cagr = 50.0
        if cagr < 0.0:
            cagr = 2.0
        g = cagr / 100.0
        if rev_2y_m <= 0:
            rev_2y_m = round(rev_next_m * (1.0 + g), 2)
            df.at[ridx, "Omsättning om 2 år"] = rev_2y_m
        if rev_3y_m <= 0:
            rev_3y_m = round(rev_next_m * ((1.0 + g) ** 2), 2)
            df.at[ridx, "Omsättning om 3 år"] = rev_3y_m

    # Riktkurser (enklare PS-modell)
    def _px(rev_m):  # alla i miljoner → pris i samma valuta
        if rev_m > 0:
            return round((rev_m * ps_avg) / shares_m, 2)
        return 0.0

    if "Riktkurs idag" in df.columns:    df.at[ridx, "Riktkurs idag"]    = _px(rev_today_m)
    if "Riktkurs om 1 år" in df.columns: df.at[ridx, "Riktkurs om 1 år"] = _px(rev_next_m)
    if "Riktkurs om 2 år" in df.columns: df.at[ridx, "Riktkurs om 2 år"] = _px(rev_2y_m)
    if "Riktkurs om 3 år" in df.columns: df.at[ridx, "Riktkurs om 3 år"] = _px(rev_3y_m)

# ---------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------

def run_update_price_only(df: pd.DataFrame, user_rates: Dict[str, float], ticker: str, **kwargs) -> Tuple[pd.DataFrame, List[str], str]:
    """
    Hämtar pris/namn/valuta/utdelning/marketcap/aktier från yfinance och uppdaterar en rad.
    Returnerar (df, changed_fields, message).
    """
    tkr = (ticker or "").strip().upper()
    if not tkr:
        return df, [], "Ingen ticker angiven."

    if "Ticker" not in df.columns or tkr not in set(df["Ticker"].astype(str).str.upper()):
        return df, [], f"{tkr} hittades inte i tabellen."

    ridx = df.index[df["Ticker"].astype(str).str.upper() == tkr][0]
    changed: List[str] = []

    info = _yfi_info(tkr)
    # price
    price = info.get("regularMarketPrice")
    if price is None:
        try:
            h = yf.Ticker(tkr).history(period="1d")
            if not h.empty and "Close" in h:
                price = float(h["Close"].iloc[-1])
        except Exception:
            price = None
    if price is not None and price > 0:
        _safe_set(df, ridx, "Aktuell kurs", float(price), changed)

    # name
    name = info.get("shortName") or info.get("longName")
    if name:
        _safe_set(df, ridx, "Bolagsnamn", str(name), changed, force_stamp=False)  # ingen TS för namn

    # currency
    ccy = info.get("currency")
    if ccy:
        _safe_set(df, ridx, "Valuta", str(ccy).upper(), changed, force_stamp=False)

    # dividend
    div_rate = info.get("dividendRate")
    if div_rate is not None:
        try:
            _safe_set(df, ridx, "Årlig utdelning", float(div_rate), changed, force_stamp=False)
        except Exception:
            pass

    # shares (styck) → miljoner
    shares = info.get("sharesOutstanding")
    if shares is not None and float(shares) > 0:
        _safe_set(df, ridx, "Utestående aktier", float(shares)/1e6, changed)  # TS

    _note_auto_update(df, ridx, AUTO_SOURCE_PRICE)
    # Räkna om P/S-snitt & riktkurser för raden (ifall något beroende fält fanns)
    _recompute_row_derivatives(df, ridx)

    msg = f"{tkr}: uppdaterade {', '.join(changed) if changed else 'inga fält (oförändrat)'}."
    return df, changed, msg


def run_update_full(df: pd.DataFrame, user_rates: Dict[str, float], ticker: str, force_stamp: bool = True, **kwargs) -> Tuple[pd.DataFrame, List[str], str]:
    """
    Full uppdatering via yfinance: pris/namn/valuta/utdelning/aktier + P/S TTM & P/S Q1–Q4.
    Returnerar (df, changed_fields, message).
    """
    tkr = (ticker or "").strip().upper()
    if not tkr:
        return df, [], "Ingen ticker angiven."

    if "Ticker" not in df.columns or tkr not in set(df["Ticker"].astype(str).str.upper()):
        return df, [], f"{tkr} hittades inte i tabellen."

    ridx = df.index[df["Ticker"].astype(str).str.upper() == tkr][0]
    changed: List[str] = []

    # ---- Bas (pris/namn/valuta/utdelning/aktier) ----
    info = _yfi_info(tkr)

    # price
    price = info.get("regularMarketPrice")
    if price is None:
        try:
            h = yf.Ticker(tkr).history(period="1d")
            if not h.empty and "Close" in h:
                price = float(h["Close"].iloc[-1])
        except Exception:
            price = None
    if price is not None and price > 0:
        _safe_set(df, ridx, "Aktuell kurs", float(price), changed, force_stamp=force_stamp)

    # name
    name = info.get("shortName") or info.get("longName")
    if name:
        _safe_set(df, ridx, "Bolagsnamn", str(name), changed, force_stamp=False)

    # currency
    ccy = info.get("currency")
    if ccy:
        _safe_set(df, ridx, "Valuta", str(ccy).upper(), changed, force_stamp=False)

    # dividend
    div_rate = info.get("dividendRate")
    if div_rate is not None:
        try:
            _safe_set(df, ridx, "Årlig utdelning", float(div_rate), changed, force_stamp=False)
        except Exception:
            pass

    # shares (styck) → miljoner
    shares = info.get("sharesOutstanding")
    if shares is not None and float(shares) > 0:
        _safe_set(df, ridx, "Utestående aktier", float(shares)/1e6, changed, force_stamp=force_stamp)

    # market cap kan vara intressant att spara i ev. kolumn om den finns
    if "Market cap (nu)" in df.columns:
        mc = info.get("marketCap")
        if mc is not None and float(mc) > 0:
            _safe_set(df, ridx, "Market cap (nu)", float(mc), changed, force_stamp=False)

    # ---- P/S (TTM) + P/S Q1–Q4 via quarterly_financials ----
    q_rows = _yfi_quarterly_revenue_rows(tkr)  # [(date, revenue), ...] nyast → äldst
    ttm_list = _ttm_windows(q_rows, need=4)    # [(end_date, TTM revenue), ...]
    # P/S (TTM) nu
    if ttm_list:
        ttm_end0, ttm0 = ttm_list[0]
        # market cap / TTM revenue
        mc_now = info.get("marketCap")
        try:
            mc_now = float(mc_now) if mc_now is not None else 0.0
        except Exception:
            mc_now = 0.0
        if mc_now > 0 and ttm0 and ttm0 > 0:
            ps_now = mc_now / float(ttm0)
            _safe_set(df, ridx, "P/S", float(ps_now), changed, force_stamp=force_stamp)

        # P/S Q1..Q4 historiskt: använd samma aktieantal (senast) + pris på/strax före respektive TTM-slut
        # Alt 1: implied shares (om marketCap & price finns)
        implied_shares = None
        try:
            px_now = float(price or 0.0)
            if (mc_now or 0.0) > 0 and px_now > 0:
                implied_shares = mc_now / px_now
        except Exception:
            implied_shares = None

        if implied_shares is None or implied_shares <= 0:
            implied_shares = float(shares or 0.0)  # kan vara 0 → skippar historik då

        if implied_shares and implied_shares > 0:
            for idx, (d_end, ttm_rev) in enumerate(ttm_list[:4], start=1):
                if ttm_rev and ttm_rev > 0:
                    px_hist = _yfi_history_close_on_or_before(tkr, d_end)
                    if px_hist and px_hist > 0:
                        mcap_hist = implied_shares * float(px_hist)
                        ps_hist = mcap_hist / float(ttm_rev)
                        _safe_set(df, ridx, f"P/S Q{idx}", float(ps_hist), changed, force_stamp=force_stamp)

    # Notera auto-källa och räkna härledda
    _note_auto_update(df, ridx, AUTO_SOURCE_FULL)
    _recompute_row_derivatives(df, ridx)

    msg = f"{tkr}: uppdaterade {', '.join(changed) if changed else 'inga fält (oförändrat)'}."
    return df, changed, msg
