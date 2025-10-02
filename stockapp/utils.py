# -*- coding: utf-8 -*-
"""
stockapp.utils
--------------
Hjälpfunktioner som används av hela appen. Denna modul ska vara
låg-nivå och inte importera andra stockapp-moduler (för att undvika
cirkulära importer).
"""

from __future__ import annotations
from typing import Callable, Iterable, List, Tuple, Dict, Any, Optional
import time
import math
import re

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Tidsstämplar & datum
# ------------------------------------------------------------
def now_stamp() -> str:
    """Returnerar en kort tidsstämpel 'YYYY-MM-DD HH:MM' i lokal tid."""
    return pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")


def parse_date(x) -> Optional[pd.Timestamp]:
    """
    Försöker tolka x som datum/tid och returnera pd.Timestamp eller None.
    Hanterar NaN/None/tom sträng robust.
    """
    if x is None:
        return None
    if isinstance(x, pd.Timestamp):
        return x
    try:
        s = str(x).strip()
        if not s or s.lower() in ("nan", "nat", "none"):
            return None
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return None


# ------------------------------------------------------------
# Numerik
# ------------------------------------------------------------
_NUM_RE = re.compile(r"[-+]?\d+(\.\d+)?([eE][-+]?\d+)?")

def to_float(x, default: float = 0.0) -> float:
    """
    Robust konvertering till float:
    - accepterar strängar med kommatecken
    - plockar första talet ur en sträng om den innehåller text
    - None/NaN -> default
    """
    try:
        if x is None:
            return float(default)
        if isinstance(x, (int, float, np.number)):
            f = float(x)
            if math.isnan(f) or math.isinf(f):
                return float(default)
            return f
        s = str(x).strip()
        if not s:
            return float(default)
        s = s.replace(",", ".")
        # försök direkt
        try:
            return float(s)
        except Exception:
            m = _NUM_RE.search(s)
            if m:
                return float(m.group(0))
            return float(default)
    except Exception:
        return float(default)


def safe_float(x, default: float = np.nan) -> float:
    """Som to_float men default = NaN (praktisk i beräkningar)."""
    try:
        v = to_float(x, default)
        return v
    except Exception:
        return float(default)


# ------------------------------------------------------------
# Backoff-wrapper
# ------------------------------------------------------------
def with_backoff(fn: Callable, *args, retries: int = 3, base_sleep: float = 0.7, **kwargs):
    """
    Kör fn(*args, **kwargs) med enkel exponential backoff.
    Vid sista miss kastas felet vidare.
    """
    last_err = None
    for i in range(max(1, retries)):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            if i == retries - 1:
                raise
            time.sleep(base_sleep * (2 ** i))
    if last_err:
        raise last_err


# ------------------------------------------------------------
# DataFrame-schema & dubbletter
# ------------------------------------------------------------
def ensure_schema(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Säkerställ att df har minst kolumnerna i 'cols'. Saknade läggs till med NaN
    och kolumnordningen justeras så att 'cols' kommer först (övriga kolumner behålls efter).
    """
    if df is None or not isinstance(df, pd.DataFrame):
        df = pd.DataFrame()
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    # behåll extra kolumner också
    ordered = [c for c in cols] + [c for c in out.columns if c not in cols]
    return out[ordered]


def dedupe_tickers(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Tar bort dubbletter på kolumnen 'Ticker' (case-insensitive), behåller första förekomsten.
    Returnerar (df_utan_dubletter, lista_med_dubbletttickers).
    """
    if df is None or "Ticker" not in df.columns:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame(), []

    work = df.copy()
    work["__tkr_up__"] = work["Ticker"].astype(str).str.upper().str.strip()
    dups_mask = work["__tkr_up__"].duplicated(keep="first")
    dups = work.loc[dups_mask, "__tkr_up__"].tolist()
    out = work.loc[~dups_mask].drop(columns=["__tkr_up__"])
    return out, dups


# ------------------------------------------------------------
# TS-verktyg (timestamp-kolumner)
# ------------------------------------------------------------
def stamp_fields_ts(df: pd.DataFrame, fields: Iterable[str], ts_suffix: str = " TS") -> pd.DataFrame:
    """
    Sätter tidsstämpel för varje fält i 'fields' (om fältet finns i df) i en kolumn
    med namn f"{fält}{ts_suffix}" (default: 'Fält TS').
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    out = df.copy()
    stamp = now_stamp()
    for f in fields:
        if f in out.columns:
            ts_col = f"{f}{ts_suffix}"
            out[ts_col] = stamp
    return out


def _collect_ts_cols(columns: Iterable[str]) -> List[str]:
    """
    Samlar alla kolumnnamn som sannolikt är TS-kolumner (både 'Fält TS' och 'TS Fält').
    """
    ts_cols = []
    for c in columns:
        s = str(c)
        if s.endswith(" TS") or s.startswith("TS "):
            ts_cols.append(s)
    return ts_cols


def add_oldest_ts_col(df: pd.DataFrame, dest_col: str = "__oldest_ts__") -> pd.DataFrame:
    """
    Letar upp alla TS-kolumner i df och lägger till en kolumn 'dest_col' med den
    ÄLDSTA (minsta) tidsstämpeln per rad. TS-kolumner upptäcks som:
      - kolumner som slutar med ' TS' (t.ex. 'Kurs TS', 'P/S TS')
      - eller kolumner som börjar med 'TS ' (t.ex. 'TS Kurs')
    Saknas TS-kolumner sätts NaT.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        out = pd.DataFrame(columns=list(df.columns) + [dest_col]) if isinstance(df, pd.DataFrame) else pd.DataFrame(columns=[dest_col])
        out[dest_col] = pd.NaT
        return out

    out = df.copy()
    ts_cols = _collect_ts_cols(out.columns)
    if not ts_cols:
        out[dest_col] = pd.NaT
        return out

    # Konvertera alla TS-kolumner till tidsstämplar
    ts_frame = pd.DataFrame(index=out.index)
    for c in ts_cols:
        ts_frame[c] = pd.to_datetime(out[c], errors="coerce")

    # min per rad
    out[dest_col] = ts_frame.min(axis=1, skipna=True)
    return out


# ------------------------------------------------------------
# Presentation
# ------------------------------------------------------------
def format_large_number(value: Any, currency_code: str = "") -> str:
    """
    Formaterar stora tal med suffix:
      - T (triljoner), B (miljarder), M (miljoner), k (tusen)
    Ex: 4 250 000 000 000 -> '4.25 T USD'
    """
    v = to_float(value, default=np.nan)
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "–"

    abs_v = abs(v)
    if abs_v >= 1e12:
        num = v / 1e12
        suf = "T"
    elif abs_v >= 1e9:
        num = v / 1e9
        suf = "B"
    elif abs_v >= 1e6:
        num = v / 1e6
        suf = "M"
    elif abs_v >= 1e3:
        num = v / 1e3
        suf = "k"
    else:
        num = v
        suf = ""

    if currency_code:
        return f"{num:.2f} {suf} {currency_code}".strip()
    return f"{num:.2f} {suf}".strip()


def risk_label_from_mcap(mcap_value: Any) -> str:
    """
    Grov klassning av bolagsstorlek baserat på Market Cap (i bolagets valuta):
      Mega:  >= 200 B
      Large: >= 10 B
      Mid:   >= 2 B
      Small: >= 0.3 B
      Micro: <  0.3 B
    """
    mcap = to_float(mcap_value, default=np.nan)
    if mcap is None or (isinstance(mcap, float) and math.isnan(mcap)):
        return "Unknown"

    b = mcap / 1e9  # miljarder
    if b >= 200:
        return "Mega"
    if b >= 10:
        return "Large"
    if b >= 2:
        return "Mid"
    if b >= 0.3:
        return "Small"
    return "Micro"
