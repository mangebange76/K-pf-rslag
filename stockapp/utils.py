# -*- coding: utf-8 -*-
"""
stockapp.utils
--------------
Gemensamma verktyg:
- with_backoff(fn, *args, **kwargs)  : robusta API-anrop
- now_stamp()                         : ISO8601-tidstämpel (UTC)
- coerce_float(x)                     : konvertera till float säkert
- format_large_number(n, currency)    : snygg formattering av stora belopp
- ensure_schema(df, cols=None)        : garanterar att alla kolumner finns
- stamp_fields_ts(row, ts_fields)     : lägger tidsstämplar på TS-fält
- dedupe_tickers(df)                  : tar bort dubbletter av tickers
- parse_ts(s)                         : tolkar tidsstämplar
- add_oldest_ts_col(df, ts_fields=None, out_col="_oldest_ts") : minsta TS per rad
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, List, Optional

import numpy as np
import pandas as pd

from .config import FINAL_COLS, TS_FIELDS


# -----------------------------------------------------------
# Robust backoff för yttre API-anrop (gspread, requests, etc.)
# -----------------------------------------------------------
def with_backoff(
    fn: Callable, *args, retries: int = 4, base_sleep: float = 0.6, **kwargs
):
    """
    Kör fn(*args, **kwargs) med enkel exponentiell backoff.
    Bra mot tillfälliga 429/5xx-fel från Sheets eller HTTP.
    """
    last_err = None
    for i in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            last_err = e
            # 429/Quota/Timeout tenderar att lösa sig med kort paus
            time.sleep(base_sleep * (2 ** i))
    # Sista försöket – låt exception bubbla upp
    if last_err:
        raise last_err


# -----------------------
# Tidsstämplar & parsing
# -----------------------
def now_stamp() -> str:
    """UTC ISO8601, sekunder precision."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_ts(s: Any) -> Optional[datetime]:
    """
    Försök tolka en tidsstämpel i några vanliga format.
    Returnerar datetime (naive UTC) eller None.
    """
    if not s or (isinstance(s, float) and math.isnan(s)):
        return None
    val = str(s).strip()
    if not val:
        return None
    # Vanliga varianter
    fmts = [
        "%Y-%m-%dT%H:%M:%SZ",      # 2025-03-04T12:34:56Z
        "%Y-%m-%d %H:%M:%S",       # 2025-03-04 12:34:56
        "%Y-%m-%d",                # 2025-03-04
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d",
        "%d/%m/%Y",
        "%d-%m-%Y",
    ]
    for f in fmts:
        try:
            dt = datetime.strptime(val, f)
            return dt
        except Exception:
            pass
    # Fallback: pandas parser
    try:
        dt = pd.to_datetime(val, utc=True, errors="coerce")
        if pd.isna(dt):
            return None
        # return naive (UTC)
        return dt.tz_convert("UTC").tz_localize(None) if getattr(dt, "tzinfo", None) else dt.to_pydatetime()
    except Exception:
        return None


# -------------------------
# Datatyper & formattering
# -------------------------
def coerce_float(x: Any) -> Optional[float]:
    """
    Robust konvertering till float:
    - Hanterar '' och None -> None
    - Ersätter kommatecken med punkt
    - Trim och typecasting
    """
    if x is None:
        return None
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        # NaN -> None
        return None if (isinstance(x, float) and math.isnan(x)) else float(x)
    try:
        s = str(x).strip().replace(",", ".")
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def _suffix_for_number(n: float) -> str:
    # Svenska: tusen/miljoner/miljarder/biljoner
    # Vi använder korta suffix: K / M / Md / Bn (biljoner).
    # (För att undvika språkförvirring runt trillion/billion.)
    absn = abs(n)
    if absn >= 1e12:
        return "Bn"   # biljoner
    if absn >= 1e9:
        return "Md"   # miljarder
    if absn >= 1e6:
        return "M"    # miljoner
    if absn >= 1e3:
        return "K"    # tusen
    return ""


def _scale_number(n: float) -> float:
    absn = abs(n)
    if absn >= 1e12:
        return n / 1e12
    if absn >= 1e9:
        return n / 1e9
    if absn >= 1e6:
        return n / 1e6
    if absn >= 1e3:
        return n / 1e3
    return n


def format_large_number(n: Any, currency: Optional[str] = None, decimals: int = 2) -> str:
    """
    Formaterar stora belopp snyggt: 1.23 Md, 45.6 M, 123.4 K etc.
    Om currency anges (t.ex. 'SEK', 'USD') prefixas/suffixas inte med symbol,
    men visas som "123.4 M SEK".
    """
    val = coerce_float(n)
    if val is None:
        return "-"
    suff = _suffix_for_number(val)
    scaled = _scale_number(val)
    s = f"{scaled:.{decimals}f}".rstrip("0").rstrip(".")
    if currency:
        return f"{s} {suff} {currency}".strip()
    return f"{s} {suff}".strip()


# -------------------------------
# Schema & kolumnhanteringshjälp
# -------------------------------
def ensure_schema(df: pd.DataFrame, cols: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Säkerställ att alla kolumner i 'cols' finns i df.
    Om cols är None används FINAL_COLS från config.
    Saknade kolumner läggs till med tomma värden.
    Returnerar en NY DataFrame (modifiera inte originalreferensen i app-vyer).
    """
    if df is None or not isinstance(df, pd.DataFrame):
        df = pd.DataFrame()

    need_cols = list(cols) if cols is not None else list(FINAL_COLS)

    work = df.copy()
    for c in need_cols:
        if c not in work.columns:
            work[c] = np.nan

    # ordna kolumnordning: de vi vet först, sedan ev. övriga längst bak
    other = [c for c in work.columns if c not in need_cols]
    work = work[need_cols + other]
    return work


def stamp_fields_ts(row: pd.Series, ts_fields: Optional[Iterable[str]] = None) -> pd.Series:
    """
    Sätt tidsstämpel "nu" på de TS-fält som saknar värde i den här raden.
    Om ts_fields=None används TS_FIELDS från config.
    Returnerar en NY rad (Series).
    """
    use_fields = list(ts_fields) if ts_fields is not None else list(TS_FIELDS)
    out = row.copy()
    stamp = now_stamp()
    for f in use_fields:
        # Om f saknas i raden, hoppa – ensure_schema bör ha sett till att kolumnen finns.
        if f not in out.index:
            continue
        v = out.get(f)
        if v is None or (isinstance(v, float) and math.isnan(v)) or str(v).strip() == "":
            out[f] = stamp
    return out


def dedupe_tickers(df: pd.DataFrame, ticker_col: str = "Ticker") -> pd.DataFrame:
    """
    Tar bort dubbletter av tickers – behåll första förekomsten.
    Returnerar ny DataFrame.
    """
    if df is None or df.empty or ticker_col not in df.columns:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    work = df.copy()
    work = work.drop_duplicates(subset=[ticker_col], keep="first").reset_index(drop=True)
    return work


def add_oldest_ts_col(
    df: pd.DataFrame,
    ts_fields: Optional[Iterable[str]] = None,
    out_col: str = "_oldest_ts",
) -> pd.DataFrame:
    """
    Beräknar minsta (äldsta) tidsstämpel i de angivna TS-fälten per rad och lägger i out_col.
    Om ts_fields=None används TS_FIELDS.
    Returnerar ny DataFrame.
    """
    if df is None or df.empty:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()

    use_fields = list(ts_fields) if ts_fields is not None else list(TS_FIELDS)
    work = df.copy()

    def _min_dt(row: pd.Series):
        dts = []
        for f in use_fields:
            if f in row.index:
                dt = parse_ts(row.get(f))
                if dt is not None:
                    dts.append(dt)
        if not dts:
            return None
        return min(dts)

    work[out_col] = work.apply(_min_dt, axis=1)
    return work
