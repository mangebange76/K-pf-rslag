# stockapp/utils.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import re
import time
from datetime import datetime, date
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "now_stamp",
    "with_backoff",
    "safe_float",
    "parse_date",
    "format_large_number",
    "ensure_schema",
    "stamp_fields_ts",
    "dedupe_tickers",
    "add_oldest_ts_col",
    "risk_label_from_mcap",
]


# --------------------------------------------------------------------
# Tids- & backoff-hjälpare
# --------------------------------------------------------------------
def now_stamp() -> str:
    """ISO8601 med sekundprecision, lokal tid."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def with_backoff(fn: Callable, *args, retries: int = 5, base_delay: float = 0.6, **kwargs):
    """
    Kör fn(*args, **kwargs) med exponentiell backoff.
    Returnerar fn:s returvärde eller re-raisar sista felet.
    """
    last_exc: Optional[BaseException] = None
    for i in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            last_exc = e
            sleep_s = base_delay * (2**i)
            time.sleep(sleep_s)
    if last_exc:
        raise last_exc


# --------------------------------------------------------------------
# Parsning & format
# --------------------------------------------------------------------
def safe_float(x: Any, default: float = float("nan")) -> float:
    """
    Robust str->float: hanterar kommatecken, whitespace, "N/A", None.
    """
    if x is None:
        return default
    if isinstance(x, (int, float, np.number)):
        try:
            return float(x)
        except Exception:
            return default
    s = str(x).strip()
    if s == "" or s.upper() in {"NA", "N/A", "NONE", "NULL", "-", "—"}:
        return default
    s = s.replace(" ", "").replace("\xa0", "").replace(",", ".")
    # plocka ut första talet i strängen
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if not m:
        return default
    try:
        return float(m.group(0))
    except Exception:
        return default


def parse_date(x: Any) -> Optional[datetime]:
    """Försöker tolkas som datum/tid -> datetime eller None."""
    if x is None or x == "":
        return None
    try:
        return pd.to_datetime(x, errors="coerce")
    except Exception:
        try:
            return datetime.fromisoformat(str(x))
        except Exception:
            return None


def format_large_number(n: Any, currency: Optional[str] = None, decimals: int = 2) -> str:
    """
    Snygg formattering: 4.34 biljoner USD, 56.8 miljarder SEK, 125.3 miljoner NOK, etc.
    Antag n i 'hela valutanheter' (t.ex. 4_340_000_000_000 = 4.34 biljoner).
    """
    v = safe_float(n, default=float("nan"))
    if math.isnan(v):
        return "–"
    sign = "-" if v < 0 else ""
    v = abs(v)

    unit = ""
    scaled = v
    if v >= 1_000_000_000_000:
        scaled = v / 1_000_000_000_000.0
        unit = "biljoner"
    elif v >= 1_000_000_000:
        scaled = v / 1_000_000_000.0
        unit = "miljarder"
    elif v >= 1_000_000:
        scaled = v / 1_000_000.0
        unit = "miljoner"

    cur = f" {currency}" if currency else ""
    if unit:
        return f"{sign}{scaled:.{decimals}f} {unit}{cur}"
    # mindre än miljoner – visa med tusavgränsare
    return f"{sign}{scaled:,.{decimals}f}{cur}".replace(",", " ").replace(".", ",")


# --------------------------------------------------------------------
# DataFrame-stöd
# --------------------------------------------------------------------
def ensure_schema(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """
    Säkerställ att alla kolumner i `cols` finns, i rätt ordning.
    Behåll övriga kolumner efter `cols`.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return pd.DataFrame(columns=list(cols))

    work = df.copy()
    for c in cols:
        if c not in work.columns:
            work[c] = np.nan

    # behåll kolumner som inte finns i cols längst bak
    other = [c for c in work.columns if c not in cols]
    return work[list(cols) + other]


def stamp_fields_ts(
    df: pd.DataFrame,
    fields: Iterable[str],
    ts_suffix: str = " TS",
    stamp: Optional[str] = None,
) -> pd.DataFrame:
    """
    Sätt/uppdatera tidsstämplar för givna fält.
    Skapar '[fält] TS' om saknas.
    """
    if df is None or not len(df):
        return df
    stamp = stamp or now_stamp()
    out = df.copy()
    for field in fields:
        ts_col = f"{field}{ts_suffix}"
        if ts_col not in out.columns:
            out[ts_col] = np.nan
        out.loc[:, ts_col] = stamp
    return out


def dedupe_tickers(df: pd.DataFrame, key: str = "Ticker") -> Tuple[pd.DataFrame, List[str]]:
    """
    Ta bort dubbletter på Ticker (case-insensitive). Returnerar (df_utan_dup, lista_med_dup_tickers).
    Bevarar första förekomsten.
    """
    if df is None or key not in df.columns:
        return df, []

    work = df.copy()
    norm = work[key].astype(str).str.upper().str.strip()
    dup_mask = norm.duplicated(keep="first")
    dups = sorted(norm[dup_mask].unique().tolist())
    if dups:
        work = work[~dup_mask].reset_index(drop=True)
    return work, dups


def add_oldest_ts_col(
    df: pd.DataFrame,
    ts_fields: Optional[Sequence[str]] = None,
    dest_col: str = "Senaste TS (min av två)",
) -> pd.DataFrame:
    """
    Skapa en hjälpkolumn med minsta (äldsta) TS över alla TS-kolumner i df.
    - Om ts_fields anges: använd just dessa kolumner (som redan är TS-kolumner),
      annars alla kolumner som innehåller 'TS' i namnet.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df

    work = df.copy()
    if ts_fields:
        ts_cols = [c for c in ts_fields if c in work.columns]
    else:
        ts_cols = [c for c in work.columns if "TS" in str(c)]

    if not ts_cols:
        work[dest_col] = pd.NaT
        return work

    tmp = work[ts_cols].apply(pd.to_datetime, errors="coerce")
    work[dest_col] = tmp.min(axis=1)
    return work


# --------------------------------------------------------------------
# Övrigt
# --------------------------------------------------------------------
def risk_label_from_mcap(mcap_value: Any) -> str:
    """
    Klassificera bolag efter börsvärde (USD).
    Trösklar: Micro <0.3B, Small <2B, Mid <10B, Large <200B, annars Mega.
    """
    v = safe_float(mcap_value, default=float("nan"))
    if math.isnan(v) or v <= 0:
        return "Unknown"
    if v < 0.3e9:
        return "Micro"
    if v < 2e9:
        return "Small"
    if v < 10e9:
        return "Mid"
    if v < 200e9:
        return "Large"
    return "Mega"
