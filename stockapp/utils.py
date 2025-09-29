# stockapp/utils.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import time
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import streamlit as st

try:
    import pytz
    TZ_STHLM = pytz.timezone("Europe/Stockholm")
    def now_dt() -> datetime:
        return datetime.now(TZ_STHLM)
except Exception:
    def now_dt() -> datetime:
        return datetime.now()

def now_stamp() -> str:
    return now_dt().strftime("%Y-%m-%d")

# Importera ENDAST det vi behöver från config för att undvika cirkulära beroenden
from .config import TS_FIELDS, FINAL_COLS


# ---------------------------------------------------------------------
# Backoff för nät-/Sheets-anrop
# ---------------------------------------------------------------------
def with_backoff(func: Callable, *args, **kwargs):
    """
    Kör func med mild backoff för att tåla 429/transienta fel.
    Ex: with_backoff(ws.update, data)
    """
    delays = [0.0, 0.5, 1.0, 2.0]
    last_err = None
    for d in delays:
        if d > 0:
            time.sleep(d)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_err = e
    # sista felet
    raise last_err


# ---------------------------------------------------------------------
# DataFrame-schema & typer
# ---------------------------------------------------------------------
_NUMERIC_COLS = [
    # Pris/aktier
    "Aktuell kurs", "Utestående aktier",
    # P/S
    "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "P/S-snitt",
    # Omsättning
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
    # Riktkurser
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
    # Portfölj
    "Antal aktier", "Årlig utdelning", "GAV (SEK)",
    # Övriga nyckeltal
    "CAGR 5 år (%)",
    "Market Cap (valuta)", "Market Cap (SEK)",
    "Bruttomarginal (%)", "Nettomarginal (%)",
    "FCF (M)", "Debt/Equity", "Kassa (M)", "Runway (kvartal)",
    "EV/EBITDA", "Dividend Yield (%)", "Payout Ratio CF (%)",
]

_STR_COLS = [
    "Ticker", "Bolagsnamn", "Valuta", "Risklabel", "Sektor", "Industri",
    "Senast manuellt uppdaterad", "Senast auto-uppdaterad", "Senast uppdaterad källa"
]

def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Säkerställ att alla FINAL_COLS finns och att typerna är rimliga.
    - Skapar saknade kolumner.
    - Undviker dubblettkolumner.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        df = pd.DataFrame({c: [] for c in FINAL_COLS})

    # Skapa saknade kolumner
    for col in FINAL_COLS:
        if col not in df.columns:
            if col in _NUMERIC_COLS:
                df[col] = 0.0
            elif col in _STR_COLS or col.startswith("TS_"):
                df[col] = ""
            else:
                # default str
                df[col] = ""

    # Ta bort dubblett-kolumner
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # Konvertera typer
    for c in _NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in _STR_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str)

    for c in df.columns:
        if str(c).startswith("TS_"):
            df[c] = df[c].astype(str)

    return df


# ---------------------------------------------------------------------
# TS-hjälpare (tidsstämplar per fält)
# ---------------------------------------------------------------------
def stamp_fields_ts(df: pd.DataFrame, row_idx: int, fields: Union[str, Sequence[str]], when: Optional[str] = None):
    """
    Sätter TS_-kolumner för ett eller flera fält.
    """
    if isinstance(fields, str):
        fields = [fields]
    date_str = when if when else now_stamp()
    for f in fields:
        ts_col = TS_FIELDS.get(f)
        if ts_col and ts_col in df.columns:
            try:
                df.at[row_idx, ts_col] = date_str
            except Exception:
                pass

def note_auto_update(df: pd.DataFrame, row_idx: int, source: str):
    try:
        df.at[row_idx, "Senast auto-uppdaterad"] = now_stamp()
        df.at[row_idx, "Senast uppdaterad källa"] = source
    except Exception:
        pass

def note_manual_update(df: pd.DataFrame, row_idx: int):
    try:
        df.at[row_idx, "Senast manuellt uppdaterad"] = now_stamp()
    except Exception:
        pass


# ---------------------------------------------------------------------
# Äldsta TS (för kontroll/batch-ordning)
# ---------------------------------------------------------------------
def _safe_parse_date(s: Any) -> Optional[pd.Timestamp]:
    try:
        d = pd.to_datetime(str(s).strip(), errors="coerce")
        if pd.notna(d):
            return d
    except Exception:
        pass
    return None

def oldest_any_ts(row: pd.Series) -> Optional[pd.Timestamp]:
    dates = []
    for c in TS_FIELDS.values():
        if c in row and str(row[c]).strip():
            d = _safe_parse_date(row[c])
            if d is not None:
                dates.append(d)
    if not dates:
        return None
    return min(dates)

def add_oldest_ts_col(df: pd.DataFrame) -> pd.DataFrame:
    df["_oldest_any_ts"] = df.apply(oldest_any_ts, axis=1)
    # Hjälpkolumn för sortering (None blir “långt i framtiden”)
    df["_oldest_any_ts_fill"] = df["_oldest_any_ts"].fillna(pd.Timestamp("2099-12-31"))
    return df


# ---------------------------------------------------------------------
# Sätt nya värden + spåra ändringar
# ---------------------------------------------------------------------
def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return default
        return float(x)
    except Exception:
        return default

def apply_changes_to_row(
    df: pd.DataFrame,
    row_idx: int,
    new_vals: Dict[str, Any],
    changes_log: Optional[Dict[str, List[str]]] = None,
    stamp_ts_if_same: bool = False,
    fields_to_stamp: Optional[Sequence[str]] = None
) -> bool:
    """
    Skriver in new_vals i raden. Returnerar True om nåt ändrades (eller stämplades).
    - stamp_ts_if_same: stämpla TS även om värdet är oförändrat (enligt din spec).
    - fields_to_stamp: stämpla TS för dessa fält även om de inte ändrats.
    """
    changed = False
    touched_fields: List[str] = []

    for f, v in (new_vals or {}).items():
        if f not in df.columns:
            continue
        old = df.at[row_idx, f]
        equal = False
        try:
            # jämför numeriskt om båda är numeriska
            if isinstance(old, (int, float, np.floating)) and isinstance(v, (int, float, np.floating)):
                equal = float(old) == float(v)
            else:
                equal = str(old) == str(v)
        except Exception:
            equal = False

        if not equal:
            df.at[row_idx, f] = v
            changed = True
            touched_fields.append(f)
        else:
            # oförändrat – men vi kanske ska stämpla TS ändå
            if stamp_ts_if_same:
                touched_fields.append(f)

    # Stämpla TS för berörda fält som finns i TS_FIELDS
    if touched_fields:
        for f in touched_fields:
            if f in TS_FIELDS:
                stamp_fields_ts(df, row_idx, f)

    # Extra TS-fält att stämpla
    if fields_to_stamp:
        for f in fields_to_stamp:
            if f in TS_FIELDS:
                stamp_fields_ts(df, row_idx, f)

    if changes_log is not None and touched_fields:
        ticker = str(df.at[row_idx, "Ticker"]) if "Ticker" in df.columns else f"row{row_idx}"
        changes_log.setdefault(ticker, []).extend(touched_fields)

    return changed or (bool(touched_fields) and stamp_ts_if_same)


# ---------------------------------------------------------------------
# Formatterare
# ---------------------------------------------------------------------
def compact_number(x: Union[int, float]) -> str:
    """
    1 234 -> 1.23K, 12 345 678 -> 12.35M, 1.2e12 -> 1.20T
    """
    try:
        x = float(x)
    except Exception:
        return str(x)

    neg = x < 0
    x = abs(x)
    units = [("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)]
    for suf, val in units:
        if x >= val:
            out = f"{x/val:.2f}{suf}"
            return f"-{out}" if neg else out
    out = f"{x:.2f}"
    return f"-{out}" if neg else out

def human_market_cap(val: Union[int, float], currency: str = "") -> str:
    """
    Market cap med suffix + (valuta).
    """
    base = compact_number(val)
    return f"{base} {currency}".strip()

def percent_str(x: Union[int, float]) -> str:
    try:
        return f"{float(x):.2f}%"
    except Exception:
        return str(x)


# ---------------------------------------------------------------------
# Dedupe
# ---------------------------------------------------------------------
def dedupe_tickers(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Tar bort exakta dubbletter (samma 'Ticker') och returnerar (df, duplicates_removed)
    (behåller första förekomsten).
    """
    if "Ticker" not in df.columns:
        return df, []
    before = set()
    dups = []
    keep_idx = []
    for idx, t in enumerate(df["Ticker"].astype(str)):
        if t in before:
            dups.append(t)
        else:
            before.add(t)
            keep_idx.append(idx)
    out = df.iloc[keep_idx].reset_index(drop=True)
    return out, dups
