# stockapp/utils.py
# -*- coding: utf-8 -*-
"""
Fristående hjälpfunktioner (inga beroenden till andra stockapp-moduler):
- with_backoff(fn, *args, **kwargs)
- safe_float(x, default=np.nan)
- parse_date(x) -> pd.Timestamp | pd.NaT
- now_stamp() -> str
- stamp_fields_ts(df, fields, ts_suffix=" TS")
- ensure_schema(df, cols)
- dedupe_tickers(df) -> (df_no_dups, removed_list)
- add_oldest_ts_col(df, dest_col="__oldest_ts__", ts_suffix=" TS")
- format_large_number(value, currency="SEK")
- risk_label_from_mcap(mcap_usd_or_local) -> str
"""

from __future__ import annotations

import math
import time
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Backoff/retry
# ------------------------------------------------------------
def with_backoff(fn, *args, **kwargs):
    """Exekvera fn med enkel exponential backoff."""
    delay = 0.5
    last_err: Exception | None = None
    for _ in range(6):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            time.sleep(delay)
            delay = min(delay * 2, 8.0)
    if last_err:
        raise last_err
    raise RuntimeError("Okänt fel i with_backoff")


# ------------------------------------------------------------
# Robust parsing/konvertering
# ------------------------------------------------------------
def safe_float(x, default=np.nan) -> float:
    """Konvertera till float, annars default."""
    try:
        if x is None:
            return default
        if isinstance(x, (int, float, np.floating)):
            return float(x)
        s = str(x).strip().replace(" ", "").replace("\u00A0", "")
        s = s.replace(",", ".")
        # ta bort tusentals-separatorer om det ser ut som 1.234.567,89
        if s.count(".") > 1 and "," not in s:
            parts = s.split(".")
            s = "".join(parts[:-1]) + "." + parts[-1]
        return float(s)
    except Exception:
        return default


def parse_date(x) -> pd.Timestamp:
    """Parsa datum/timestamp till pandas Timestamp, annars NaT."""
    if x is None or (isinstance(x, float) and math.isfinite(x) and math.isnan(x)):
        return pd.NaT
    try:
        return pd.to_datetime(x, utc=False, errors="coerce")
    except Exception:
        return pd.NaT


def now_stamp() -> str:
    """ISO-liknande tidsstämpel (lokal tid)."""
    return pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")


# ------------------------------------------------------------
# DataFrame utilities
# ------------------------------------------------------------
def stamp_fields_ts(df: pd.DataFrame, fields: Iterable[str], ts_suffix: str = " TS") -> pd.DataFrame:
    """
    Stämpla fält med tidsstämpel-kolumner (alltid stämpla, även om värdet är oförändrat).
    Skapar kolumnen '<fält><ts_suffix>' om den saknas.
    """
    ts = now_stamp()
    out = df.copy()
    for f in fields:
        ts_col = f"{f}{ts_suffix}"
        if ts_col not in out.columns:
            out[ts_col] = np.nan
        out.loc[:, ts_col] = ts
    return out


def ensure_schema(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Säkerställ att alla kolumner i 'cols' finns i df.
    Saknade kolumner läggs till med NaN / tomma strängar beroende på typ.
    """
    out = df.copy()
    # normalisera kolumnnamn
    out.columns = [str(c).strip() for c in out.columns]
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    # flytta om kolumnordning så BEGINNING följer 'cols' först, resten efter
    ordered = [c for c in cols if c in out.columns] + [c for c in out.columns if c not in cols]
    out = out[ordered]
    return out


def dedupe_tickers(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Ta bort dubbletter baserat på kolumn 'Ticker' (behåll första förekomsten).
    Returnerar (df_utan_dubletter, lista_med_borttagna_tickrar)
    """
    if "Ticker" not in df.columns:
        return df.copy(), []
    out = df.copy()
    out["__TKR__"] = out["Ticker"].astype(str).str.upper().str.strip()
    dup_mask = out["__TKR__"].duplicated(keep="first")
    removed = out.loc[dup_mask, "Ticker"].astype(str).tolist()
    out = out.loc[~dup_mask].drop(columns="__TKR__")
    return out.reset_index(drop=True), removed


def add_oldest_ts_col(df: pd.DataFrame, dest_col: str = "__oldest_ts__", ts_suffix: str = " TS") -> pd.DataFrame:
    """
    Beräkna äldsta (minsta) tidsstämpeln bland alla kolumner som slutar på ts_suffix.
    Lägger resultatet i 'dest_col' (pd.Timestamp). NaT ignoreras i min-beräkningen.
    """
    out = df.copy()
    ts_cols = [c for c in out.columns if str(c).endswith(ts_suffix)]
    if not ts_cols:
        out[dest_col] = pd.NaT
        return out

    def _row_oldest(s: pd.Series):
        vals = [parse_date(s[c]) for c in ts_cols]
        vals = [v for v in vals if not pd.isna(v)]
        if not vals:
            return pd.NaT
        return min(vals)

    out[dest_col] = out.apply(_row_oldest, axis=1)
    return out


# ------------------------------------------------------------
# Presentation
# ------------------------------------------------------------
def format_large_number(value, currency: str = "SEK") -> str:
    """
    Formatera stora tal med svenska enheter:
    - tn  (triljoner)
    - mdr (miljarder)
    - m   (miljoner)
    - k   (tusen)
    Ex: 4_250_000_000_000 -> '4,25 tn USD'
    """
    v = safe_float(value, default=np.nan)
    if pd.isna(v):
        return "–"
    neg = v < 0
    v = abs(v)

    unit = ""
    if v >= 1_000_000_000_000:
        v = v / 1_000_000_000_000
        unit = " tn"
    elif v >= 1_000_000_000:
        v = v / 1_000_000_000
        unit = " mdr"
    elif v >= 1_000_000:
        v = v / 1_000_000
        unit = " m"
    elif v >= 1_000:
        v = v / 1_000
        unit = " k"

    s = f"{'-' if neg else ''}{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", " ")
    # Lägg på valuta sist
    return f"{s}{unit} {currency}"


def risk_label_from_mcap(mcap) -> str:
    """
    Grov storleksklass baserat på market cap (antar samma valuta för jämförelse).
    Trösklar ungefär:
      Micro < 0,3 mdr
      Small < 2 mdr
      Mid   < 10 mdr
      Large < 200 mdr
      Mega  ≥ 200 mdr
    """
    x = safe_float(mcap, default=np.nan)
    if pd.isna(x):
        return "Unknown"
    if x < 300_000_000:
        return "Micro"
    if x < 2_000_000_000:
        return "Small"
    if x < 10_000_000_000:
        return "Mid"
    if x < 200_000_000_000:
        return "Large"
    return "Mega"
