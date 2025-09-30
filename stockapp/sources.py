# -*- coding: utf-8 -*-
"""
Runners som appen anropar vid:
- "Uppdatera kurs" (snabb)
- "Full uppdatering" (alla nyckeltal)
Skriver in värden i df + stämplar TS-kolumner om de finns (skapar annars).
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, List
import datetime as dt

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None

from .utils import now_stamp, stamp_fields_ts, ensure_schema
from .fetchers.yahoo import fetch_ticker_yahoo


# -------- hjälp
def _ensure_ts_cols(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    for k in keys:
        ts_col = f"TS {k}"
        if ts_col not in df.columns:
            df[ts_col] = ""
    return df


def run_update_price(df: pd.DataFrame, ticker: str, user_rates: Dict[str, float]) -> Tuple[pd.DataFrame, Dict[str, Any], str]:
    """
    Uppdaterar endast pris/marketcap snabbt via yfinance.
    Returnerar (df, changes, lograd)
    """
    changes: Dict[str, Any] = {}
    log = ""

    if yf is None:
        return df, changes, f"{ticker}: yfinance saknas."

    try:
        t = yf.Ticker(ticker)
        fi = {}
        try:
            fi = t.fast_info or {}
        except Exception:
            pass

        price = fi.get("last_price") or fi.get("lastPrice") or fi.get("current_price") or fi.get("last")
        mcap = fi.get("market_cap") or fi.get("marketCap")

        ridx = df.index[df["Ticker"] == ticker]
        if len(ridx) == 0:
            return df, changes, f"{ticker}: hittades inte i tabellen."

        ridx = ridx[0]
        if price:
            df.at[ridx, "Kurs"] = float(price)
            changes["Kurs"] = float(price)
        if mcap:
            df.at[ridx, "Market Cap"] = float(mcap)
            changes["Market Cap"] = float(mcap)

        df = _ensure_ts_cols(df, list(changes.keys()))
        df = stamp_fields_ts(df, ridx, list(changes.keys()), ts_value=now_stamp())

        log = f"{ticker}: uppdaterade {', '.join(changes.keys()) or 'inget'}."
        return df, changes, log

    except Exception as e:
        return df, changes, f"{ticker}: Fel: {e}"


def run_update_full(df: pd.DataFrame, ticker: str, user_rates: Dict[str, float]) -> Tuple[pd.DataFrame, Dict[str, Any], str]:
    """
    Full uppdatering via Yahoo-hämtaren: skriver namn, sektor, P/S, kvartalsfönster,
    marginaler, skuld, FCF/utdelning m.m.
    """
    ridxs = df.index[df["Ticker"] == ticker]
    if len(ridxs) == 0:
        return df, {}, f"{ticker}: hittades inte i tabellen."
    ridx = ridxs[0]

    data, _psw = fetch_ticker_yahoo(ticker)

    # Skriv in alla kända fält om de finns
    write_keys = [
        "Namn", "Sektor", "Valuta",
        "Market Cap", "P/S",
        "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "MCAP Q1", "MCAP Q2", "MCAP Q3", "MCAP Q4",
        "Period Q1", "Period Q2", "Period Q3", "Period Q4",
        "Debt/Equity", "Net debt / EBITDA",
        "P/B",
        "Gross margin (%)", "Operating margin (%)", "Net margin (%)",
        "FCF (TTM)", "FCF Yield (%)", "Dividend yield (%)",
        "Utestående aktier (milj.)",
    ]

    changes: Dict[str, Any] = {}
    for k in write_keys:
        if k in data and data[k] is not None:
            df.at[ridx, k] = data[k]
            changes[k] = data[k]

    # Stämpla tidsstämplar
    if changes:
        df = _ensure_ts_cols(df, list(changes.keys()))
        df = stamp_fields_ts(df, ridx, list(changes.keys()), ts_value=now_stamp())

    log = f"{ticker}: uppdaterade {', '.join(changes.keys()) or 'inget'}."
    return df, changes, log
