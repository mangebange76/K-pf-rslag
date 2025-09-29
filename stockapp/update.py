# stockapp/update.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, Tuple, List

import time
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from .config import TS_FIELDS
from .utils import now_stamp, ensure_schema, safe_float
from .rates import hamta_valutakurs


# -----------------------------------------------------------
# Interna hjälpare
# -----------------------------------------------------------
def _stamp_tracked(df: pd.DataFrame, ridx: int, field: str, when: str | None = None) -> None:
    ts_col = TS_FIELDS.get(field)
    if not ts_col:
        return
    try:
        df.at[ridx, ts_col] = when if when else now_stamp()
    except Exception:
        pass

def _note_auto(df: pd.DataFrame, ridx: int, source: str) -> None:
    try:
        df.at[ridx, "Senast auto-uppdaterad"] = now_stamp()
        df.at[ridx, "Senast uppdaterad källa"] = source
    except Exception:
        pass

def _ps_quarters_from_yahoo(tkr: yf.Ticker, shares_float: float, price_ccy: str) -> Dict[str, float]:
    """
    Försök hämta kvartalsintäkter via Yahoo och bygg P/S Q1–Q4 och P/S (TTM).
    """
    out: Dict[str, float] = {}
    # 1) Hämta kvartalsintäkter
    qf = None
    try:
        qf = tkr.quarterly_financials
    except Exception:
        qf = None

    series = None
    if isinstance(qf, pd.DataFrame) and not qf.empty:
        for row_name in ["Total Revenue","Revenues","Revenue","Sales","Total revenue"]:
            if row_name in qf.index:
                series = qf.loc[row_name].dropna()
                break

    if series is None or series.empty:
        # fallback via income_stmt
        try:
            is_df = getattr(tkr, "income_stmt", None)
            if isinstance(is_df, pd.DataFrame) and not is_df.empty and "Total Revenue" in is_df.index:
                series = is_df.loc["Total Revenue"].dropna()
        except Exception:
            pass

    if series is None or series.empty:
        return out  # saknar underlag

    # Sortera nyast -> äldst
    q_rows = []
    for c, v in series.items():
        try:
            d = c.date() if hasattr(c, "date") else pd.to_datetime(c).date()
            q_rows.append((d, float(v)))
        except Exception:
            pass
    q_rows.sort(key=lambda t: t[0], reverse=True)

    # Skapa TTM-fönster (upp till 4)
    def _ttm_windows(values: List[Tuple], need: int = 4):
        res = []
        if len(values) < 4:
            return res
        for i in range(0, min(need, len(values)-3)):
            end_i = values[i][0]
            ttm_i = sum(v for (_, v) in values[i:i+4])
            res.append((end_i, float(ttm_i)))
        return res

    ttm = _ttm_windows(q_rows, need=4)
    if not ttm:
        return out

    # Pris vid respektive TTM-slut (eller närmast före)
    try:
        hist = tkr.history(start=min(d for (d, _) in ttm) - pd.Timedelta(days=14),
                           end=max(d for (d, _) in ttm) + pd.Timedelta(days=2),
                           interval="1d")
    except Exception:
        hist = pd.DataFrame()

    px_map = {}
    if not hist.empty and "Close" in hist:
        hist = hist.sort_index()
        idx_dates = list(hist.index.date)
        closes = list(hist["Close"].values)
        for (d_end, _) in ttm[:4]:
            px = None
            for j in range(len(idx_dates)-1, -1, -1):
                if idx_dates[j] <= d_end:
                    px = float(closes[j]); break
            if px is not None:
                px_map[d_end] = px

    # Bygg P/S Q1..Q4
    for i, (d_end, ttm_rev) in enumerate(ttm[:4], start=1):
        p = px_map.get(d_end)
        if p and p > 0 and ttm_rev > 0 and shares_float > 0:
            mcap_hist = shares_float * p
            out[f"P/S Q{i}"] = float(mcap_hist / ttm_rev)

    # Sätt P/S (TTM) nu om vi kan
    try:
        info = tkr.info or {}
    except Exception:
        info = {}
    mcap_now = safe_float(info.get("marketCap"))
    if mcap_now <= 0:
        # Implied via pris * shares
        last_price = safe_float(info.get("regularMarketPrice"))
        if last_price > 0 and shares_float > 0:
            mcap_now = last_price * shares_float

    if mcap_now > 0 and len(ttm) >= 1 and ttm[0][1] > 0:
        out["P/S"] = float(mcap_now / float(ttm[0][1]))

    # P/S-snitt (positiva Q)
    ps_vals = [out.get("P/S Q1", 0), out.get("P/S Q2", 0), out.get("P/S Q3", 0), out.get("P/S Q4", 0)]
    ps_clean = [float(x) for x in ps_vals if safe_float(x) > 0]
    out["P/S-snitt"] = round(float(np.mean(ps_clean)), 2) if ps_clean else 0.0

    return out


# -----------------------------------------------------------
# Publika uppdateringsfunktioner
# -----------------------------------------------------------
def update_price_for_ticker(df: pd.DataFrame, ticker: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Uppdaterar bara Aktuell kurs, Valuta, Bolagsnamn.
    Stämplar 'Senast auto-uppdaterad' + 'Senast uppdaterad källa'.
    """
    ticker = str(ticker).upper().strip()
    if "Ticker" not in df.columns or ticker not in df["Ticker"].astype(str).str.upper().values:
        raise ValueError(f"{ticker} hittades inte i tabellen.")

    ridx = df.index[df["Ticker"].astype(str).str.upper() == ticker][0]
    t = yf.Ticker(ticker)
    info = {}
    try:
        info = t.info or {}
    except Exception:
        info = {}

    # Pris
    price = info.get("regularMarketPrice", None)
    if price is None:
        try:
            h = t.history(period="1d")
            if not h.empty and "Close" in h:
                price = float(h["Close"].iloc[-1])
        except Exception:
            price = None
    if price is not None:
        df.at[ridx, "Aktuell kurs"] = float(price)

    # Valuta
    ccy = info.get("currency")
    if ccy:
        df.at[ridx, "Valuta"] = str(ccy).upper()

    # Namn
    name = info.get("shortName") or info.get("longName")
    if name:
        df.at[ridx, "Bolagsnamn"] = str(name)

    _note_auto(df, ridx, source="Kurs (Yahoo)")
    # NOTERA: medvetet ingen “ändringsjämförelse” – vi stämplar alltid om datumet
    return df, {"ticker": ticker, "fields": ["Aktuell kurs","Valuta","Bolagsnamn"]}


def update_price_for_all(df: pd.DataFrame, sleep_secs: float = 0.8) -> Tuple[pd.DataFrame, Dict]:
    """
    Kör kursuppdatering på alla tickers. Visar progress i UI.
    """
    if "Ticker" not in df.columns:
        return df, {"error": "Saknar Ticker-kolumn."}

    tickers = [str(t).upper().strip() for t in df["Ticker"].astype(str).tolist() if str(t).strip()]
    total = len(tickers)
    prog = st.sidebar.progress(0)
    status = st.sidebar.empty()

    changed = []
    for i, tkr in enumerate(tickers, start=1):
        status.write(f"Uppdaterar kurs {i}/{total}: {tkr}")
        try:
            df, meta = update_price_for_ticker(df, tkr)
            changed.append(meta)
        except Exception as e:
            changed.append({"ticker": tkr, "error": str(e)})
        prog.progress(i / max(1, total))
        if sleep_secs > 0:
            time.sleep(sleep_secs)

    return df, {"changed": changed, "total": total}


def update_full_for_ticker(df: pd.DataFrame, ticker: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Full uppdatering (Yahoo-baserad, global fallback):
      - Bolagsnamn, Valuta, Aktuell kurs
      - Market Cap (nu)
      - Utestående aktier (implied via marketCap/price) i miljoner
      - P/S (TTM)
      - P/S Q1–Q4 + P/S-snitt
      - Stämplar TS_ för P/S & P/S Q1..Q4 och 'Senast auto-uppdaterad'/'Senast uppdaterad källa'
    """
    ticker = str(ticker).upper().strip()
    if "Ticker" not in df.columns or ticker not in df["Ticker"].astype(str).str.upper().values:
        raise ValueError(f"{ticker} hittades inte i tabellen.")

    ridx = df.index[df["Ticker"].astype(str).str.upper() == ticker][0]
    t = yf.Ticker(ticker)

    # Bas: pris/valuta/namn
    info = {}
    try:
        info = t.info or {}
    except Exception:
        info = {}

    # Pris
    price = info.get("regularMarketPrice", None)
    if price is None:
        try:
            h = t.history(period="1d")
            if not h.empty and "Close" in h:
                price = float(h["Close"].iloc[-1])
        except Exception:
            price = None
    if price is not None:
        df.at[ridx, "Aktuell kurs"] = float(price)

    # Valuta
    ccy = info.get("currency")
    if ccy:
        df.at[ridx, "Valuta"] = str(ccy).upper()

    # Namn
    name = info.get("shortName") or info.get("longName")
    if name:
        df.at[ridx, "Bolagsnamn"] = str(name)

    # Market cap nu (eller implied)
    mcap = safe_float(info.get("marketCap"))
    if mcap <= 0:
        last_price = safe_float(info.get("regularMarketPrice"))
        shares_out = safe_float(info.get("sharesOutstanding"))
        if last_price > 0 and shares_out > 0:
            mcap = last_price * shares_out

    if mcap > 0:
        df.at[ridx, "Market Cap"] = float(mcap)

    # Implied shares i miljoner
    implied_shares = 0.0
    if mcap > 0 and safe_float(df.at[ridx, "Aktuell kurs"]) > 0:
        implied_shares = mcap / safe_float(df.at[ridx, "Aktuell kurs"])
    elif safe_float(info.get("sharesOutstanding")) > 0:
        implied_shares = float(info.get("sharesOutstanding"))
    if implied_shares > 0:
        df.at[ridx, "Utestående aktier"] = float(implied_shares) / 1e6
        _stamp_tracked(df, ridx, "Utestående aktier")

    # P/S-derivat från Yahoo kvartalsdata
    ps_block = _ps_quarters_from_yahoo(t, shares_float=implied_shares, price_ccy=str(df.at[ridx, "Valuta"] or "USD"))
    changed_fields = []
    for k, v in ps_block.items():
        if k in ["P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "P/S-snitt"]:
            df.at[ridx, k] = float(v)
            changed_fields.append(k)
            if k in ["P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4"]:
                _stamp_tracked(df, ridx, k)

    # Tidsstämplar
    _note_auto(df, ridx, source="Full (Yahoo global)")

    # **Viktigt**: vi stämplar ALLTID om datum även om värdet blev samma
    return df, {"ticker": ticker, "changed": changed_fields or ["(stämplad)"]}
