# -*- coding: utf-8 -*-
"""
Utility-funktioner som används i hela appen.

Exponerar:
- with_backoff(func, *args, **kwargs)
- now_stamp()
- ensure_schema(df, cols)
- dedupe_tickers(df) -> (df2, removed_list)
- add_oldest_ts_col(df)
- stamp_fields_ts(row: dict, fields: list[str])  # sätter "TS <field>" = now
- format_large_number(x, currency=None)
- coerce_float(x)
"""

from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import math
import time
import datetime as dt

import pandas as pd


# ---------------------------------------------------------------------
# Robust retries för Google Sheets-anrop
# ---------------------------------------------------------------------
def with_backoff(func: Callable, *args, **kwargs):
    """
    Kör func(*args, **kwargs) med enkel exponentiell backoff.
    Svalar typiska 429/5xx fel och försöker igen några gånger.
    """
    delays = [0.4, 0.8, 1.6, 3.2, 6.4]
    last_exc: Optional[Exception] = None
    for d in delays:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exc = e
            time.sleep(d)
    # Sista, låt ev. fel bubbla upp (så vi får en tydlig traceback)
    return func(*args, **kwargs)


# ---------------------------------------------------------------------
# Tid & tidsstämplar
# ---------------------------------------------------------------------
def now_stamp() -> str:
    """UTC ISO8601 utan mikrosekunder + 'Z'."""
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


# ---------------------------------------------------------------------
# DataFrame-hjälp
# ---------------------------------------------------------------------
def ensure_schema(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """
    Säkerställ att alla kolumner i 'cols' finns i df (läggs till som None om de saknas).
    Ordningen bibehålls (df:s befintliga kolumner + saknade i slutet).
    """
    if df is None or not isinstance(df, pd.DataFrame):
        df = pd.DataFrame()
    # lägg till saknade
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df


def dedupe_tickers(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Tar bort dubbletter på 'Ticker' (case-insensitive). Behåller första förekomsten.
    Returnerar (ny_df, lista_med_borttagna_tickers).
    """
    if "Ticker" not in df.columns:
        return df.reset_index(drop=True), []
    work = df.copy()
    work["_T"] = work["Ticker"].astype(str).str.upper().str.strip()
    keep = ~work["_T"].duplicated(keep="first")
    removed = work.loc[~keep, "Ticker"].astype(str).tolist()
    work = work.loc[keep].drop(columns=["_T"]).reset_index(drop=True)
    return work, removed


def add_oldest_ts_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lägger till kolumnen 'OldestTS' = minsta (äldsta) av alla kolumner som börjar med 'TS '.
    """
    work = df.copy()
    ts_cols = [c for c in work.columns if c.startswith("TS ")]
    if not ts_cols:
        work["OldestTS"] = None
        return work

    def _min_ts(row):
        vals = []
        for c in ts_cols:
            v = row.get(c)
            if pd.notna(v) and v not in (None, "", "NaT"):
                vals.append(pd.to_datetime(v, utc=True, errors="coerce"))
        vals = [x for x in vals if pd.notna(x)]
        if not vals:
            return None
        return min(vals)

    work["OldestTS"] = work.apply(_min_ts, axis=1)
    return work


# ---------------------------------------------------------------------
# TS-stämpling av fält
# ---------------------------------------------------------------------
def stamp_fields_ts(row: Dict[str, Any], fields: List[str]) -> Dict[str, Any]:
    """
    Sätter tidsstämpel på 'TS <field>' (om fält finns/uppdateras).
    Den här varianten stämplar bara nu direkt; comparison mot tidigare värde
    görs i anroparen vid behov.
    """
    now = now_stamp()
    out = dict(row)
    for f in fields:
        ts_name = f if f.startswith("TS ") else f"TS {f}"
        out[ts_name] = now
    return out


# ---------------------------------------------------------------------
# Numerik
# ---------------------------------------------------------------------
def coerce_float(x: Any) -> Optional[float]:
    """Försöker konvertera x till float. Returnerar None vid miss."""
    try:
        if x is None:
            return None
        if isinstance(x, str):
            x = x.replace(" ", "").replace(",", ".")
        fx = float(x)
        if math.isnan(fx):
            return None
        return fx
    except Exception:
        return None


def format_large_number(x: Any, currency: Optional[str] = None) -> str:
    """
    Fint format av stora tal:
      >= 1e12  -> "{:.2f} tn"
      >= 1e9   -> "{:.2f} mdr"
      >= 1e6   -> "{:.2f} M"
      annars   -> vanlig tusentals-separering
    Lägger till valutasuffix om angivet.
    """
    val = coerce_float(x)
    if val is None:
        return "—"

    suffix = f" {currency}" if currency else ""
    abs_v = abs(val)

    if abs_v >= 1e12:
        return f"{val/1e12:.2f} tn{suffix}"
    if abs_v >= 1e9:
        return f"{val/1e9:.2f} mdr{suffix}"
    if abs_v >= 1e6:
        return f"{val/1e6:.2f} M{suffix}"

    # Vanligt heltal/decimal med tusentalsavskiljare (svensk stil: blanksteg)
    if float(val).is_integer():
        return f"{int(val):,}".replace(",", " ") + suffix
    return f"{val:,.2f}".replace(",", " ") + suffix
