# stockapp/sources.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Tuple, List, Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from .config import TS_FIELDS

# ------------------------------------------------------------
# Små hjälpare (lokala för att göra modulen självständig)
# ------------------------------------------------------------

def _now_stamp() -> str:
    """YYYY-MM-DD i Stockholmstid om pytz finns, annars systemtid."""
    try:
        import pytz
        tz = pytz.timezone("Europe/Stockholm")
        return datetime.now(tz).strftime("%Y-%m-%d")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d")

def _get_row_index(df: pd.DataFrame, ticker: str) -> Optional[int]:
    m = df.index[df["Ticker"].astype(str).str.upper() == str(ticker).upper()]
    return int(m[0]) if len(m) else None

def _record_change(ch: Dict[str, Tuple[object, object]], field: str, old, new) -> None:
    # Spara även om samma (så att loggen visar att vi "skrev" värdet)
    ch[field] = (old, new)

def _set_ts(df: pd.DataFrame, ridx: int, field: str, date_str: Optional[str] = None) -> None:
    ts_col = TS_FIELDS.get(field)
    if not ts_col:
        return
    try:
        df.at[ridx, ts_col] = date_str if date_str else _now_stamp()
    except Exception:
        pass

# ------------------------------------------------------------
# Yahoo-hjälpare
# ------------------------------------------------------------

def _yfi_info_dict(t: yf.Ticker) -> dict:
    try:
        return t.info or {}
    except Exception:
        return {}

def _yfi_quarterly_revenues(t: yf.Ticker) -> List[Tuple[pd.Timestamp, float]]:
    """
    Försöker läsa kvartalsintäkter från Yahoo.
    Returnerar [(period_end_date, value), ...] sorterad nyast→äldst (pd.Timestamp, float)
    """
    # 1) quarterly_financials
    try:
        qf = t.quarterly_financials
        if isinstance(qf, pd.DataFrame) and not qf.empty:
            idx = [str(x).strip() for x in qf.index]
            cand_rows = [
                "Total Revenue", "TotalRevenue", "Revenues", "Revenue", "Sales",
                "Total revenue", "Revenues from contracts with customers"
            ]
            for key in cand_rows:
                if key in idx:
                    row = qf.loc[key].dropna()
                    out: List[Tuple[pd.Timestamp, float]] = []
                    for c, v in row.items():
                        try:
                            d = pd.to_datetime(c)
                            out.append((d, float(v)))
                        except Exception:
                            pass
                    out.sort(key=lambda x: x[0], reverse=True)
                    return out
    except Exception:
        pass

    # 2) fallback: income_stmt quarterly via v1-api (ibland tomt)
    try:
        df_is = getattr(t, "income_stmt", None)
        if isinstance(df_is, pd.DataFrame) and not df_is.empty and "Total Revenue" in df_is.index:
            ser = df_is.loc["Total Revenue"].dropna()
            out = []
            for c, v in ser.items():
                try:
                    d = pd.to_datetime(c)
                    out.append((d, float(v)))
                except Exception:
                    pass
            out.sort(key=lambda x: x[0], reverse=True)
            return out
    except Exception:
        pass

    return []

def _ttm_windows(values: List[Tuple[pd.Timestamp, float]], need: int = 6) -> List[Tuple[pd.Timestamp, float]]:
    """
    Tar [(end_date, kvartalsintäkt), ...] (nyast→äldst) och bygger upp till 'need' TTM-summor:
    [(end_date0, ttm0), (end_date1, ttm1), ...] där ttm0 = sum(q0..q3), ttm1 = sum(q1..q4), osv.
    """
    out: List[Tuple[pd.Timestamp, float]] = []
    if len(values) < 4:
        return out
    limit = min(need, len(values) - 3)
    for i in range(0, limit):
        end_i = values[i][0]
        ttm_i = sum(v for (_, v) in values[i:i+4])
        out.append((end_i, float(ttm_i)))
    return out

def _yahoo_prices_for_dates(ticker: str, dates: List[pd.Timestamp]) -> Dict[pd.Timestamp, float]:
    """
    Hämtar Close-pris på eller närmast före respektive datum (daglig upplösning).
    """
    if not dates:
        return {}
    dmin = min(dates) - pd.Timedelta(days=14)
    dmax = max(dates) + pd.Timedelta(days=2)
    try:
        t = yf.Ticker(ticker)
        hist = t.history(start=dmin.to_pydatetime(), end=dmax.to_pydatetime(), interval="1d")
        if hist is None or hist.empty:
            return {}
        hist = hist.sort_index()
        out: Dict[pd.Timestamp, float] = {}
        idx = list(hist.index)
        closes = list(hist["Close"].values)
        for d in dates:
            px = None
            # gå bakåt tills vi hittar <= d
            for j in range(len(idx)-1, -1, -1):
                if idx[j] <= d:
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

# ------------------------------------------------------------
# Runners
# ------------------------------------------------------------

def run_update_price_only(
    df: pd.DataFrame,
    ticker: str,
    user_rates: Dict[str, float],
) -> Tuple[pd.DataFrame, Dict[str, Tuple[object, object]], str]:
    """
    Uppdaterar endast pris/valuta/namn från Yahoo.
    Returnerar (df2, changed_map, status_msg)
    """
    ridx = _get_row_index(df, ticker)
    if ridx is None:
        raise ValueError(f"{ticker} hittades inte i tabellen.")

    t = yf.Ticker(ticker)
    info = _yfi_info_dict(t)

    changed: Dict[str, Tuple[object, object]] = {}

    # Pris
    pris = info.get("regularMarketPrice")
    if pris is None:
        try:
            h = t.history(period="1d")
            if not h.empty and "Close" in h:
                pris = float(h["Close"].iloc[-1])
        except Exception:
            pris = None
    if pris is not None:
        old = df.at[ridx, "Aktuell kurs"]
        df.at[ridx, "Aktuell kurs"] = float(pris)
        _record_change(changed, "Aktuell kurs", old, float(pris))

    # Valuta
    valuta = info.get("currency")
    if valuta:
        old = df.at[ridx, "Valuta"]
        df.at[ridx, "Valuta"] = str(valuta).upper()
        _record_change(changed, "Valuta", old, df.at[ridx, "Valuta"])

    # Namn
    namn = info.get("shortName") or info.get("longName")
    if namn:
        old = df.at[ridx, "Bolagsnamn"]
        df.at[ridx, "Bolagsnamn"] = str(namn)
        _record_change(changed, "Bolagsnamn", old, df.at[ridx, "Bolagsnamn"])

    # Metadata
    df.at[ridx, "Senast auto-uppdaterad"] = _now_stamp()
    df.at[ridx, "Senast uppdaterad källa"] = "Yahoo (pris)"

    return df, changed, "OK"

def run_update_full(
    df: pd.DataFrame,
    ticker: str,
    user_rates: Dict[str, float],
) -> Tuple[pd.DataFrame, Dict[str, Tuple[object, object]], str]:
    """
    Full uppdatering med Yahoo:
      - Bolagsnamn, Valuta, Aktuell kurs
      - Utestående aktier (implied: mcap/price, fallback sharesOutstanding)
      - P/S (TTM) + P/S Q1–Q4 via kvartalsintäkter och historiska priser
    OBS: uppdaterar INTE 'Omsättning idag' eller 'Omsättning nästa år' (manuellt enligt kravspec).
    Stämplar TS_* för fälten i TS_FIELDS som skrivs (även om samma värde).
    """
    ridx = _get_row_index(df, ticker)
    if ridx is None:
        raise ValueError(f"{ticker} hittades inte i tabellen.")

    t = yf.Ticker(ticker)
    info = _yfi_info_dict(t)

    changed: Dict[str, Tuple[object, object]] = {}

    # ---- Bas: pris/valuta/namn ---------------------------------
    pris = info.get("regularMarketPrice")
    if pris is None:
        try:
            h = t.history(period="1d")
            if not h.empty and "Close" in h:
                pris = float(h["Close"].iloc[-1])
        except Exception:
            pris = None
    if pris is not None:
        old = df.at[ridx, "Aktuell kurs"]
        df.at[ridx, "Aktuell kurs"] = float(pris)
        _record_change(changed, "Aktuell kurs", old, float(pris))

    valuta = info.get("currency")
    if valuta:
        old = df.at[ridx, "Valuta"]
        df.at[ridx, "Valuta"] = str(valuta).upper()
        _record_change(changed, "Valuta", old, df.at[ridx, "Valuta"])

    namn = info.get("shortName") or info.get("longName")
    if namn:
        old = df.at[ridx, "Bolagsnamn"]
        df.at[ridx, "Bolagsnamn"] = str(namn)
        _record_change(changed, "Bolagsnamn", old, df.at[ridx, "Bolagsnamn"])

    # ---- Market cap & implied shares ----------------------------
    px = float(df.at[ridx, "Aktuell kurs"]) if df.at[ridx, "Aktuell kurs"] else float(pris or 0.0)
    mcap = info.get("marketCap")
    try:
        mcap = float(mcap) if mcap is not None else 0.0
    except Exception:
        mcap = 0.0

    shares = 0.0
    if mcap > 0 and px > 0:
        shares = mcap / px
        shares_source = "implied(mcap/price)"
    else:
        so = info.get("sharesOutstanding")
        try:
            shares = float(so or 0.0)
            shares_source = "sharesOutstanding"
        except Exception:
            shares = 0.0
            shares_source = "unknown"

    if shares > 0:
        old = df.at[ridx, "Utestående aktier"]
        new_val = shares / 1e6  # milj.
        df.at[ridx, "Utestående aktier"] = float(new_val)
        _record_change(changed, "Utestående aktier", old, float(new_val))
        _set_ts(df, ridx, "Utestående aktier")

    # ---- Kvartalsintäkter & P/S --------------------------------
    # Läs kvartalsintäkter
    q_rows = _yfi_quarterly_revenues(t)  # [(date, revenue), ...] i financialCurrency
    if q_rows and len(q_rows) >= 4:
        # Yahoo financialCurrency
        fin_ccy = str(info.get("financialCurrency") or df.at[ridx, "Valuta"] or "USD").upper()
        px_ccy = str(df.at[ridx, "Valuta"] or "USD").upper()

        # Konvertering: vi behöver user_rates; om saknas => 1.0
        def _fx_rate(base: str, quote: str) -> float:
            if not base or not quote or base.upper() == quote.upper():
                return 1.0
            # Hämta från sessionens user_rates som redan finns i appen
            rates = user_rates or {}
            b = base.upper(); q = quote.upper()
            if b == "SEK":
                # SEK -> q
                r = float(rates.get(q, 1.0))
                return 1.0 / r if r > 0 else 1.0
            if q == "SEK":
                return float(rates.get(b, 1.0))
            # annars via SEK som pivot: base->SEK->quote
            base_to_sek = float(rates.get(b, 1.0))
            quote_to_sek = float(rates.get(q, 1.0))
            if base_to_sek <= 0 or quote_to_sek <= 0:
                return 1.0
            # (base/SEK) / (quote/SEK) = base->quote
            return base_to_sek / quote_to_sek

        conv = _fx_rate(fin_ccy, px_ccy)

        # TTM-fönster (ta upp till 6, så vi kan välja de 4 senaste)
        ttm_list = _ttm_windows(q_rows, need=6)  # [(date, ttm_rev_finccy), ...]
        ttm_list_px = [(d, v * conv) for (d, v) in ttm_list]

        # Market cap nu om saknas men shares/px finns
        if (mcap <= 0) and shares > 0 and px > 0:
            mcap = shares * px

        # P/S (TTM) nu
        if mcap > 0 and ttm_list_px:
            ltm_now = ttm_list_px[0][1]
            if ltm_now > 0:
                old = df.at[ridx, "P/S"]
                new_ps = float(mcap / ltm_now)
                df.at[ridx, "P/S"] = new_ps
                _record_change(changed, "P/S", old, new_ps)
                _set_ts(df, ridx, "P/S")

        # P/S Q1–Q4 historik (använder implied shares * historisk Close / historisk TTM)
        if shares > 0 and ttm_list_px:
            q_dates = [d for (d, _) in ttm_list_px]
            px_map = _yahoo_prices_for_dates(ticker, q_dates)
            # ta de 4 första TTM-punkterna om de finns
            for idx, (d_end, ttm_rev_px) in enumerate(ttm_list_px[:4], start=1):
                if ttm_rev_px and ttm_rev_px > 0:
                    hist_px = px_map.get(d_end)
                    if hist_px and hist_px > 0:
                        psq = float((shares * hist_px) / ttm_rev_px)
                        col = f"P/S Q{idx}"
                        old = df.at[ridx, col] if col in df.columns else 0.0
                        df.at[ridx, col] = psq
                        _record_change(changed, col, old, psq)
                        _set_ts(df, ridx, col)

    # ---- Metadata ------------------------------------------------
    df.at[ridx, "Senast auto-uppdaterad"] = _now_stamp()
    df.at[ridx, "Senast uppdaterad källa"] = "Yahoo (full)"

    return df, changed, "OK"
