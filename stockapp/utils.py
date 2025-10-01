# -*- coding: utf-8 -*-
"""
stockapp.utils
--------------
Gemensamma hjälp-funktioner:
- normalize_columns(df)
- ensure_schema(df, cols)
- dedupe_tickers(df)
- with_backoff(fn, *args, **kwargs)
- now_stamp(), stamp_fields_ts(df, cols)
- to_float(x), to_int(x)
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Sequence, Tuple
import time
import datetime as dt

import numpy as np
import pandas as pd


# ---- Enkel backoff för gspread/HTTP-anrop -----------------------------------
def with_backoff(fn: Callable, *args, **kwargs):
    delays = [0.0, 0.5, 1.0, 2.0, 3.0]
    last_exc = None
    for d in delays:
        try:
            if d:
                time.sleep(d)
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            continue
    raise last_exc


# ---- Datumstämplar -----------------------------------------------------------
def now_stamp() -> str:
    return dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def stamp_fields_ts(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    ts = now_stamp()
    for c in cols:
        if c in df.columns:
            df.loc[:, c] = ts
    return df


# ---- Numeriska konverterare --------------------------------------------------
def to_float(x) -> float:
    if x is None:
        return 0.0
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip().replace(" ", "").replace("\u202f", "").replace(",", ".")
    if s in ("", "nan", "None", "-"):
        return 0.0
    try:
        return float(s)
    except Exception:
        return 0.0


def to_int(x) -> int:
    return int(round(to_float(x)))


# ---- Kolumn-normalisering & schema ------------------------------------------
_NORMALIZE_MAP: Dict[str, str] = {
    # namn
    "bolagsnamn": "Namn",
    "company": "Namn",
    "name": "Namn",
    # ticker lämnas som "Ticker"
    # valuta
    "currency": "Valuta",
    # antal aktier
    "antal": "Antal aktier",
    "antal aktier du äger": "Antal aktier",
    "antal aktier (st)": "Antal aktier",
    "qty": "Antal aktier",
    "quantity": "Antal aktier",
    # GAV
    "gav": "GAV (SEK)",
    "gav (sek)": "GAV (SEK)",
    "gav sek": "GAV (SEK)",
    # sektor
    "sector": "Sektor",
    "sektor": "Sektor",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Byter kolumnnamn enligt _NORMALIZE_MAP och trimmar whitespace."""
    if df is None or df.empty:
        return df

    new_cols = []
    for c in df.columns:
        base = (c or "").strip()
        key = base.lower()
        new_cols.append(_NORMALIZE_MAP.get(key, base))
    df = df.rename(columns=dict(zip(df.columns, new_cols)))
    # trimma strängkolumner
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]):
            df[c] = df[c].astype(str).str.strip()
    # Ticker alltid VERSAL
    if "Ticker" in df.columns:
        df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    return df


def ensure_schema(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """
    Säkerställ att alla förväntade kolumner finns. Saknade läggs till med default.
    Default för numeriska nyckeltal = 0.0, strängfält = "".
    """
    if df is None or df.empty:
        df = pd.DataFrame(columns=list(cols))

    # Normalisera först (för att få t.ex. 'Bolagsnamn' -> 'Namn')
    df = normalize_columns(df)

    # Lägg till saknade kolumnnamn
    for c in cols:
        if c not in df.columns:
            df[c] = ""  # sätts om till 0.0 för numeriska längre ner

    # rimliga defaultar för numeriska nyckeltal
    numeric_like = {
        "Antal aktier",
        "GAV (SEK)",
        "Market Cap",
        "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Utestående aktier (milj.)",
        "Debt/Equity", "P/B", "ROE (%)",
        "Gross margin (%)", "Operating margin (%)", "Net margin (%)",
        "FCF Yield (%)", "Dividend yield (%)", "Dividend payout (FCF) (%)",
        "Net debt / EBITDA",
        "Pris", "Kurs", "Vikt (%)", "Andel (%)",
    }
    for c in df.columns:
        if c in numeric_like:
            df[c] = df[c].apply(to_float)

    return df


def dedupe_tickers(df: pd.DataFrame) -> pd.DataFrame:
    if "Ticker" not in df.columns:
        return df
    return df.drop_duplicates(subset=["Ticker"], keep="first").reset_index(drop=True)
