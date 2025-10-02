# -*- coding: utf-8 -*-
"""
stockapp.utils
--------------
Robusta hjälp-funktioner som återanvänds av appen.

OBS: Hanterar dubbletta kolumnnamn (t.ex. TS-fält) säkert.
"""

from __future__ import annotations

import math
import random
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Konverteringar & datum
# ---------------------------------------------------------------------

def safe_float(x, default=np.nan) -> float:
    """Konvertera till float, eller default (NaN som standard)."""
    try:
        if x is None:
            return float(default)
        if isinstance(x, str):
            s = x.strip().replace(" ", "").replace("\u00a0", "")
            if s == "" or s.lower() in {"nan", "none"}:
                return float(default)
            # Byt ev. kommatecken
            s = s.replace(",", ".")
            return float(s)
        return float(x)
    except Exception:
        try:
            return float(default)
        except Exception:
            return np.nan  # sista fallback


def to_float(x, default=np.nan) -> float:
    """Alias till safe_float (bakåtkompatibilitet)."""
    return safe_float(x, default=default)


def parse_date(x) -> Optional[pd.Timestamp]:
    """Försök tolka datum; returnerar pandas Timestamp eller None."""
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return None
        ts = pd.to_datetime(x, errors="coerce")
        if pd.isna(ts):
            return None
        return ts
    except Exception:
        return None


def now_stamp() -> str:
    """Sträng med nuvarande tid i ISO-format (sek)."""
    return pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------
# DataFrame schema & dubbletter
# ---------------------------------------------------------------------

def ensure_schema(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """
    Säkerställ att alla kolumner i 'cols' finns i df (lägger till saknade som NaN)
    och droppa dubbletter av kolumnnamn (behåll första).
    Ordna sedan kolumner i samma ordning som 'cols' + resterande efteråt.
    """
    if df is None:
        df = pd.DataFrame()

    # droppa dubbletter av kolumn-namn (behåll första förekomsten)
    if not df.columns.is_unique:
        df = df.loc[:, ~df.columns.duplicated()]

    # lägg till saknade
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan

    # ordna – först enligt cols, sedan ev. andra
    ordered = [c for c in cols if c in df.columns]
    tail = [c for c in df.columns if c not in ordered]
    df = df[ordered + tail]
    return df


def dedupe_tickers(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Ta bort dubbletter baserat på 'Ticker' (behåll första).
    Returnerar (df_ut, lista_med_dubbletttickers).
    """
    if df is None or df.empty or "Ticker" not in df.columns:
        return df, []

    # normalisera för jämförelse
    key = df["Ticker"].astype(str).str.upper().str.strip()
    dup_mask = key.duplicated(keep="first")
    dups = df.loc[dup_mask, "Ticker"].astype(str).tolist()
    out = df.loc[~dup_mask].copy()
    return out, dups


# ---------------------------------------------------------------------
# Formatering & etiketter
# ---------------------------------------------------------------------

def format_large_number(v: Union[float, int, None], currency: str = "SEK") -> str:
    """
    Formatera stort tal med K/M/G och valutasymbol (enkel).
    """
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    try:
        x = float(v)
    except Exception:
        return "–"

    absx = abs(x)
    suf = ""
    div = 1.0
    if absx >= 1_000_000_000:
        suf, div = "B", 1_000_000_000.0
    elif absx >= 1_000_000:
        suf, div = "M", 1_000_000.0
    elif absx >= 1_000:
        suf, div = "K", 1_000.0

    sym = {"SEK": "kr", "USD": "$", "EUR": "€", "NOK": "kr", "CAD": "$"}.get(str(currency).upper(), "")
    val = x / div
    if suf:
        return f"{val:,.2f}{suf} {sym}".replace(",", " ")
    return f"{val:,.0f} {sym}".replace(",", " ")


def risk_label_from_mcap(mcap: Union[float, int, None]) -> str:
    """
    Grov indelning efter Market Cap (USD).
    """
    v = safe_float(mcap, np.nan)
    if math.isnan(v):
        return "Unknown"
    if v >= 200e9:
        return "Mega"
    if v >= 10e9:
        return "Large"
    if v >= 2e9:
        return "Mid"
    if v >= 300e6:
        return "Small"
    return "Micro"


# ---------------------------------------------------------------------
# Backoff-wrapper för API/Sheets-anrop
# ---------------------------------------------------------------------

def with_backoff(fn: Callable, *args, retries: int = 3, base_delay: float = 0.6, jitter: float = 0.2, **kwargs):
    """
    Kör fn(*args, **kwargs) med exponentiell backoff.
    Höjer sista felet om alla försök misslyckas.
    """
    last: Optional[Exception] = None
    for i in range(int(retries)):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # pragma: no cover - robust i produktion
            last = e
            sleep = base_delay * (2 ** i) + random.random() * jitter
            time.sleep(sleep)
    if last:
        raise last
    raise RuntimeError("with_backoff: okänt fel")


# ---------------------------------------------------------------------
# TS-hantering
# ---------------------------------------------------------------------

def _is_ts_column_name(name: str) -> bool:
    """
    Upptäck TS-kolumner robust:
    - slut med " TS"  (ex: "Kurs TS")
    - börja med "TS " (ex: "TS Kurs")
    - sluta/börja med "_TS" / "TS_"
    - exakt "TS" i början/slutet (case-insensitive)
    """
    if not isinstance(name, str):
        return False
    n = name.strip()
    u = n.upper()
    return (
        u.endswith(" TS")
        or u.startswith("TS ")
        or u.endswith("_TS")
        or u.startswith("TS_")
        or u == "TS"
    )


def _first_series_for_column(df: pd.DataFrame, colname: str) -> pd.Series:
    """
    Returnera Series för FÖRSTA förekomsten av 'colname' om df har dubbletter.
    Om kolumn inte finns returneras en Na-serie.
    """
    idxs = [i for i, c in enumerate(df.columns) if c == colname]
    if not idxs:
        return pd.Series([np.nan] * len(df), index=df.index)
    return df.iloc[:, idxs[0]]


def stamp_fields_ts(df: pd.DataFrame, fields: Sequence[str], ts_suffix: str = " TS") -> pd.DataFrame:
    """
    Stämpla tidsstämpel i (fält + ts_suffix) för alla rader där fältet finns.
    Skapar kolumnen om den saknas. Kolliderar inte om det redan finns dubbletter
    – vi uppdaterar första förekomsten.
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    now = now_stamp()

    # säkerställ unika kolumnnamn i arbetskopian
    if not out.columns.is_unique:
        out = out.loc[:, ~out.columns.duplicated()]

    for f in fields:
        ts_col = f"{f}{ts_suffix}"
        # se till att ts-kolumnen finns (första förekomsten används)
        if ts_col not in out.columns:
            out[ts_col] = np.nan
        # uppdatera där fältet inte är NaN
        mask = out.get(f, pd.Series([np.nan] * len(out), index=out.index))
        mask = pd.notna(mask)
        out.loc[mask, ts_col] = now

    return out


def add_oldest_ts_col(df: pd.DataFrame, ts_cols: Optional[Sequence[str]] = None, dest_col: str = "__oldest_ts__") -> pd.DataFrame:
    """
    Skapa en kolumn med det ÄLDSTA (min) datumet över givna TS-kolumner.
    - Hanterar dubbletter av kolumnnamn (tar första förekomsten).
    - Om ts_cols=None: autodetektera TS-kolumner med _is_ts_column_name().
    - Returnerar en kopia av df med 'dest_col' som pandas.Timestamp (kan vara NaT).
    """
    if df is None or df.empty:
        out = df.copy() if df is not None else pd.DataFrame()
        out[dest_col] = pd.NaT
        return out

    work = df.copy()
    # Droppa ev. dubbletter av kolumnnamn för att undvika pandas ValueError
    # N.B. Vi behåller FÖRSTA förekomsten.
    if not work.columns.is_unique:
        work = work.loc[:, ~work.columns.duplicated()]

    # Autodetektera TS-kolumner om inget explicit skickas in
    if ts_cols is None:
        ts_cols = [c for c in work.columns if _is_ts_column_name(str(c))]

    if not ts_cols:
        # inget att göra – skapa tom kolumn
        work[dest_col] = pd.NaT
        return work

    # Bygg en tids-DataFrame med första förekomsten per TS-kolumn
    parts: List[pd.Series] = []
    for c in ts_cols:
        s = _first_series_for_column(work, c)
        s = pd.to_datetime(s, errors="coerce")
        parts.append(s)

    if not parts:
        work[dest_col] = pd.NaT
        return work

    tsdf = pd.concat(parts, axis=1)
    # Äldsta (min) över rader, ignorera NaT
    oldest = tsdf.min(axis=1, skipna=True)
    work[dest_col] = oldest
    return work


# ---------------------------------------------------------------------
# Små hjälpare
# ---------------------------------------------------------------------

def with_nan_as_none(x):
    """Returnera None om x är NaN, annars x."""
    try:
        if x is None:
            return None
        if isinstance(x, float) and math.isnan(x):
            return None
        if pd.isna(x):
            return None
        return x
    except Exception:
        return None
