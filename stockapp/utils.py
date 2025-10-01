# -*- coding: utf-8 -*-
"""
Allmänna hjälp-funktioner som används av flera moduler.
Helt frikopplad från Streamlit (ingen st-import här).

Innehåll (urval):
- safe_float, parse_date, now_stamp
- format_large_number, risk_label_from_mcap
- ensure_schema, dedupe_tickers
- stamp_fields_ts, add_oldest_ts_col
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import math
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Bas-helpers
# ---------------------------------------------------------------------
def safe_float(x, default=np.nan) -> float:
    """Robust konvertering till float."""
    try:
        if x is None:
            return default
        if isinstance(x, (int, float, np.floating)):
            return float(x)
        s = str(x).strip().replace(" ", "").replace(",", ".")
        if s == "" or s.lower() in ("nan", "none", "null", "-"):
            return default
        return float(s)
    except Exception:
        return default


def parse_date(x) -> pd.Timestamp | pd.NaT:
    """Robust datumtolkning -> pandas Timestamp eller NaT."""
    try:
        if x is None:
            return pd.NaT
        if isinstance(x, pd.Timestamp):
            return x
        s = str(x).strip()
        if s == "" or s.lower() in ("nan", "none", "null", "-"):
            return pd.NaT
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.NaT


def now_stamp(date_only: bool = True) -> str:
    """
    Tidsstämpel som text (standard: YYYY-MM-DD).
    Används för * TS-kolumner.
    """
    ts = pd.Timestamp.now()
    return ts.strftime("%Y-%m-%d") if date_only else ts.strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------
def format_large_number(value, currency: str | None = None) -> str:
    """
    Formatera stora tal med suffix: K, M, B, T.
    Ex: format_large_number(5140605000000, "USD") -> '5.14T USD'
    """
    v = safe_float(value, default=np.nan)
    if math.isnan(v):
        return "–"

    abs_v = abs(v)
    if abs_v >= 1_000_000_000_000:
        s = f"{v/1_000_000_000_000:.2f}T"
    elif abs_v >= 1_000_000_000:
        s = f"{v/1_000_000_000:.2f}B"
    elif abs_v >= 1_000_000:
        s = f"{v/1_000_000:.2f}M"
    elif abs_v >= 1_000:
        s = f"{v/1_000:.2f}K"
    else:
        s = f"{v:.2f}"

    return f"{s} {currency}" if currency else s


def risk_label_from_mcap(mcap_value) -> str:
    """
    Grov storleksklass baserat på Market Cap (valutaneutralt).
    Trösklar (typiska USD-nivåer):
      Mega:  >= 200B
      Large: >= 10B
      Mid:   >= 2B
      Small: >= 300M
      Micro: < 300M
    """
    v = safe_float(mcap_value, default=np.nan)
    if math.isnan(v) or v <= 0:
        return "Unknown"
    if v >= 200_000_000_000:
        return "Mega"
    if v >= 10_000_000_000:
        return "Large"
    if v >= 2_000_000_000:
        return "Mid"
    if v >= 300_000_000:
        return "Small"
    return "Micro"


# ---------------------------------------------------------------------
# DataFrame-hjälp
# ---------------------------------------------------------------------
def ensure_schema(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """
    Säkerställ att samtliga kolumner i `cols` finns i df.
    Saknade kolumner läggs till med NaN. Övriga kolumner lämnas orörda.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        df = pd.DataFrame()
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    # behåll kolumnordning så långt det går (flytta kända i framkant)
    ordered = [c for c in cols if c in df.columns]
    rest = [c for c in df.columns if c not in ordered]
    return df[ordered + rest]


def dedupe_tickers(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Ta bort dubbletter baserat på kolumnen 'Ticker' (behåll första).
    Returnerar (ny_df, lista_med_borttagna_tickers).
    """
    if "Ticker" not in df.columns:
        return df, []
    tick_col = df["Ticker"].astype(str).str.upper()
    duplicated_mask = tick_col.duplicated(keep="first")
    removed = df.loc[duplicated_mask, "Ticker"].astype(str).str.upper().tolist()
    clean = df.loc[~duplicated_mask].copy()
    # normalisera 'Ticker' (uppercase utan whitespace)
    clean["Ticker"] = clean["Ticker"].astype(str).str.upper().str.strip()
    return clean, removed


def stamp_fields_ts(df: pd.DataFrame, fields: Iterable[str], ts_suffix: str = " TS") -> pd.DataFrame:
    """
    Sätter tidsstämpel (YYYY-MM-DD) i kolumner som heter '<field><ts_suffix>'.
    Skapar TS-kolumnen om den saknas.
    """
    if df is None or df.empty:
        return df
    stamp = now_stamp()
    fields = list(fields)
    for f in fields:
        col = f"{f}{ts_suffix}"
        if col not in df.columns:
            df[col] = pd.NaT
        df.loc[:, col] = stamp
    return df


def add_oldest_ts_col(df: pd.DataFrame, dest_col: str = "Senaste TS (min av två)") -> pd.DataFrame:
    """
    Beräknar äldsta (minsta) tidsstämpel per rad utifrån:
      - Alla kolumnnamn som slutar med ' TS'
      - Samt (om de finns) 'Senast manuellt uppdaterad' och 'Senast auto-uppdaterad'
    Lägger resultatet i `dest_col` (pd.Timestamp). NaT om allt saknas.
    """
    if df is None or df.empty:
        df = pd.DataFrame()

    ts_cols = [c for c in df.columns if isinstance(c, str) and c.endswith(" TS")]
    # lägg till explicita datumfält om de finns
    for c in ("Senast manuellt uppdaterad", "Senast auto-uppdaterad"):
        if c in df.columns and c not in ts_cols:
            ts_cols.append(c)

    def _row_min_ts(row) -> pd.Timestamp | pd.NaT:
        vals = []
        for c in ts_cols:
            vals.append(parse_date(row.get(c)))
        if not vals:
            return pd.NaT
        s = pd.Series(vals, dtype="datetime64[ns]")
        # ignorera NaT
        s = s.dropna()
        if s.empty:
            return pd.NaT
        return s.min()

    df = df.copy()
    df[dest_col] = df.apply(_row_min_ts, axis=1)
    return df
