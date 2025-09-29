# stockapp/utils.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Iterable

import numpy as np
import pandas as pd

# Streamlit är bara valfritt här; många hjälpfunktioner klarar sig utan.
try:
    import streamlit as st
except Exception:
    st = None

# --- Lokal Stockholm-tid om pytz finns (annars systemtid) ---
try:
    import pytz
    TZ_STHLM = pytz.timezone("Europe/Stockholm")
    def now_stamp() -> str:
        return datetime.now(TZ_STHLM).strftime("%Y-%m-%d")
    def now_dt() -> datetime:
        return datetime.now(TZ_STHLM)
except Exception:
    def now_stamp() -> str:
        return datetime.now().strftime("%Y-%m-%d")
    def now_dt() -> datetime:
        return datetime.now()

# ------------------------------------------------------------------
# Allmänna helpers
# ------------------------------------------------------------------
def with_backoff(func, *args, **kwargs):
    """Liten backoff-hjälpare för kvot-/429-fel etc."""
    delays = [0, 0.5, 1.0, 2.0]
    last_err = None
    for d in delays:
        if d:
            time.sleep(d)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_err = e
    raise last_err

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        try:
            return float(str(x).replace(" ", "").replace(",", "."))
        except Exception:
            return default

def format_large_number(v: float, no_currency: bool = False) -> str:
    """Snygg formattering av stora tal (k/M/B/T)."""
    try:
        x = float(v)
    except Exception:
        x = 0.0
    sign = "-" if x < 0 else ""
    x = abs(x)
    unit = ""
    if x >= 1e12:
        val, unit = x / 1e12, "T"
    elif x >= 1e9:
        val, unit = x / 1e9, "B"
    elif x >= 1e6:
        val, unit = x / 1e6, "M"
    elif x >= 1e3:
        val, unit = x / 1e3, "k"
    else:
        val, unit = x, ""
    s = f"{sign}{val:.2f}{unit}"
    if no_currency:
        return s
    return s

# ------------------------------------------------------------------
# TS-kontroller
# ------------------------------------------------------------------
def oldest_any_ts(row: pd.Series, ts_fields: Iterable[str]) -> Optional[pd.Timestamp]:
    """Returnerar äldsta (minsta) TS bland angivna TS_-kolumner."""
    dates = []
    for c in ts_fields:
        if c in row and str(row[c]).strip():
            try:
                d = pd.to_datetime(str(row[c]).strip(), errors="coerce")
                if pd.notna(d):
                    dates.append(d)
            except Exception:
                pass
    return min(dates) if dates else None

def add_oldest_ts_col(df: pd.DataFrame, ts_fields: Iterable[str]) -> pd.DataFrame:
    """Beräknar kolumner _oldest_any_ts / _oldest_any_ts_fill för sortering/visning."""
    df = df.copy()
    df["_oldest_any_ts"] = df.apply(lambda r: oldest_any_ts(r, ts_fields), axis=1)
    df["_oldest_any_ts"] = pd.to_datetime(df["_oldest_any_ts"], errors="coerce")
    df["_oldest_any_ts_fill"] = df["_oldest_any_ts"].fillna(pd.Timestamp("2099-12-31"))
    return df

# ------------------------------------------------------------------
# Auto-uppdateringsskrivare (NY – stämplar alltid TS när data anländer)
# ------------------------------------------------------------------
def apply_auto_updates_to_row(
    df: pd.DataFrame,
    row_idx: int,
    new_vals: Dict[str, Any],
    source: str,
    changes_map: Dict[str, list],
    stamp_even_if_same: bool = True,
    allow_zero_for: tuple = ("Aktuell kurs", "Årlig utdelning"),  # fält där 0.0 är tillåtet att skriva
) -> bool:
    """
    Skriver fält från new_vals in i df-raden 'row_idx'.
    - Stämplar TS_ för alla spårade fält som vi *får in från källan*, även om värdet är oförändrat.
    - Sätter 'Senast auto-uppdaterad' + 'Senast uppdaterad källa' *om minst ett meningsfullt fält kom in*.
    - 'Meningsfullt' = strippat icke-tomt för strängar, >0 för tal (eller 0.0 om fältet finns i allow_zero_for).
    Returnerar True om någon cell skrevs om; False annars.
    """
    from .config import TS_FIELDS  # mapping "Fält" -> "TS_Fält"

    def _stamp_ts_for_field(df_local, ridx, field_name):
        ts_col = TS_FIELDS.get(field_name)
        if ts_col and ts_col in df_local.columns:
            df_local.at[ridx, ts_col] = now_stamp()

    def _note_auto_update(df_local, ridx, src):
        if "Senast auto-uppdaterad" in df_local.columns:
            df_local.at[ridx, "Senast auto-uppdaterad"] = now_stamp()
        if "Senast uppdaterad källa" in df_local.columns:
            df_local.at[ridx, "Senast uppdaterad källa"] = src

    wrote_any = False
    stamped_any = False
    changed_fields = []

    for f, v in (new_vals or {}).items():
        if f not in df.columns:
            continue

        # "Meningsfull" indata?
        meaningful = False
        if isinstance(v, (int, float, np.floating)):
            if f in allow_zero_for:
                meaningful = (v is not None)
            else:
                try:
                    meaningful = (float(v) > 0.0)
                except Exception:
                    meaningful = False
        else:
            meaningful = (str(v).strip() != "")

        if not meaningful:
            continue

        # Stämpla TS om fältet är spårat
        if f in TS_FIELDS:
            _stamp_ts_for_field(df, row_idx, f)
            stamped_any = True

        # Skriv bara om faktisk skillnad – men markera ändå i loggen om vi vill
        old = df.at[row_idx, f]
        should_write = False
        if isinstance(v, (int, float, np.floating)):
            if f in allow_zero_for:
                should_write = (str(old) != str(v))
            else:
                should_write = (float(v) > 0.0 and str(old) != str(v))
        else:
            should_write = (str(old) != str(v))

        if should_write:
            df.at[row_idx, f] = v
            wrote_any = True
            changed_fields.append(f)
        else:
            if stamp_even_if_same and f in TS_FIELDS:
                changed_fields.append(f + " (samma)")

    # Auto-meta om vi faktiskt haft *något* att stämpla/skriva
    if stamped_any or wrote_any:
        _note_auto_update(df, row_idx, source)
        tkr = df.at[row_idx, "Ticker"] if "Ticker" in df.columns else f"row{row_idx}"
        if changed_fields:
            changes_map.setdefault(tkr, []).extend(changed_fields)

    return wrote_any
