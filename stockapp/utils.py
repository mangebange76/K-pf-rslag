# stockapp/utils.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Streamlit är valfritt – vi faller tillbaka om det inte finns
try:
    import streamlit as st
except Exception:
    st = None

# Lokal tidszon om tillgänglig
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

# För schema & TS-listor
try:
    from .config import FINAL_COLS, TS_FIELDS
except Exception:
    # Fallback om config inte hunnit laddas
    FINAL_COLS = [
        "Ticker","Bolagsnamn","Valuta","Aktuell kurs","Antal aktier","Årlig utdelning",
        "Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt",
        "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
        "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
        "CAGR 5 år (%)","Bruttomarginal (%)","Netto-marginal (%)","Debt/Equity",
        "Utdelningsyield (%)","Utdelning/FCF coverage","Payout ratio (%)",
        "Market Cap","Sektor",
        "Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa",
    ]
    TS_FIELDS: Dict[str, str] = {
        "Aktuell kurs": "TS_Aktuell kurs",
        "Utestående aktier": "TS_Utestående aktier",
        "P/S": "TS_P/S", "P/S Q1": "TS_P/S Q1", "P/S Q2": "TS_P/S Q2", "P/S Q3": "TS_P/S Q3", "P/S Q4": "TS_P/S Q4",
        "Omsättning idag": "TS_Omsättning idag", "Omsättning nästa år": "TS_Omsättning nästa år",
        "Omsättning om 2 år": "TS_Omsättning om 2 år", "Omsättning om 3 år": "TS_Omsättning om 3 år",
        "Riktkurs idag": "TS_Riktkurs idag", "Riktkurs om 1 år": "TS_Riktkurs om 1 år",
        "Riktkurs om 2 år": "TS_Riktkurs om 2 år", "Riktkurs om 3 år": "TS_Riktkurs om 3 år",
        "CAGR 5 år (%)": "TS_CAGR 5 år (%)",
        "Bruttomarginal (%)": "TS_Bruttomarginal (%)", "Netto-marginal (%)": "TS_Netto-marginal (%)",
        "Debt/Equity": "TS_Debt/Equity",
        "Utdelningsyield (%)": "TS_Utdelningsyield (%)",
        "Utdelning/FCF coverage": "TS_Utdelning/FCF coverage",
        "Payout ratio (%)": "TS_Payout ratio (%)",
        "Market Cap": "TS_Market Cap",
    }

# ------------------------------------------------------------------
# Allmänna helpers
# ------------------------------------------------------------------
def with_backoff(func, *args, **kwargs):
    """Kör en funktion med enkel backoff (bra mot 429/kvotfel)."""
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
    """Formatterar stora tal (k/M/B/T)."""
    try:
        x = float(v)
    except Exception:
        x = 0.0
    sign = "-" if x < 0 else ""
    x = abs(x)
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
    return s if no_currency else s

def normalize_ticker(t: Any) -> str:
    return str(t or "").upper().strip()

# ------------------------------------------------------------------
# Schema-helpers
# ------------------------------------------------------------------
def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Säkerställ att alla kolumner finns:
    - FINAL_COLS + alla TS_-kolumner i TS_FIELDS
    - Normaliserar 'Ticker'
    Returnerar samma df-objekt (modifierat).
    """
    if df is None or not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(columns=FINAL_COLS)

    # Se till att 'Ticker' finns tidigt
    if "Ticker" not in df.columns:
        df["Ticker"] = ""

    # Lägg till saknade final-kolumner
    for c in FINAL_COLS:
        if c not in df.columns:
            df[c] = ("" if c in ("Ticker","Bolagsnamn","Valuta","Sektor","Senast manuellt uppdaterad",
                                 "Senast auto-uppdaterad","Senast uppdaterad källa")
                     else 0.0)

    # Lägg till alla TS_-kolumner
    for ts_col in TS_FIELDS.values():
        if ts_col not in df.columns:
            df[ts_col] = ""

    # Normalisera tickers
    df["Ticker"] = df["Ticker"].apply(normalize_ticker)

    return df

def dedupe_tickers(df: pd.DataFrame, keep: str = "first") -> Tuple[pd.DataFrame, List[str]]:
    """
    Tar bort dubbletter på 'Ticker'. Returnerar (df_ut, borttagna_tickers).
    - `keep="first"` behåller första förekomsten, `keep="last"` sista.
    """
    if "Ticker" not in df.columns:
        return df, []

    df = df.copy()
    df["Ticker"] = df["Ticker"].apply(normalize_ticker)

    before = len(df)
    # hitta dubbletter
    dups_mask = df.duplicated(subset=["Ticker"], keep=keep)
    removed_tickers = df.loc[dups_mask, "Ticker"].tolist()
    # droppa dubblettrader
    df = df.drop_duplicates(subset=["Ticker"], keep=keep).reset_index(drop=True)
    after = len(df)

    # (valfritt) logga i Streamlit
    if st is not None and removed_tickers:
        st.warning(f"Dubbletter borttagna för tickers: {', '.join(removed_tickers)}")

    return df, removed_tickers

# ------------------------------------------------------------------
# TS-kontroller
# ------------------------------------------------------------------
def oldest_any_ts(row: pd.Series, ts_fields: Iterable[str]) -> Optional[pd.Timestamp]:
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

def add_oldest_ts_col(df: pd.DataFrame, ts_fields: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Beräknar _oldest_any_ts & _oldest_any_ts_fill.
    Om ts_fields=None används alla TS_-kolumner från TS_FIELDS som finns i df.
    """
    df = df.copy()
    if ts_fields is None:
        ts_fields = [c for c in TS_FIELDS.values() if c in df.columns]
    df["_oldest_any_ts"] = df.apply(lambda r: oldest_any_ts(r, ts_fields), axis=1)
    df["_oldest_any_ts"] = pd.to_datetime(df["_oldest_any_ts"], errors="coerce")
    df["_oldest_any_ts_fill"] = df["_oldest_any_ts"].fillna(pd.Timestamp("2099-12-31"))
    return df

# ------------------------------------------------------------------
# Auto-uppdateringsskrivare
# ------------------------------------------------------------------
def apply_auto_updates_to_row(
    df: pd.DataFrame,
    row_idx: int,
    new_vals: Dict[str, Any],
    source: str,
    changes_map: Dict[str, List[str]],
    stamp_even_if_same: bool = True,
    allow_zero_for: tuple = ("Aktuell kurs","Årlig utdelning"),
) -> bool:
    """
    Skriver in fält från new_vals i df.loc[row_idx] och stämplar TS_ för de fält som kom in.
    Sätter även 'Senast auto-uppdaterad' & 'Senast uppdaterad källa' om något meningsfullt kom in.
    Returnerar True om någon cell faktiskt ändrades (värdeskillnad).
    """
    def _stamp(field_name: str):
        ts_col = TS_FIELDS.get(field_name)
        if ts_col and ts_col in df.columns:
            df.at[row_idx, ts_col] = now_stamp()

    def _note_meta():
        if "Senast auto-uppdaterad" in df.columns:
            df.at[row_idx, "Senast auto-uppdaterad"] = now_stamp()
        if "Senast uppdaterad källa" in df.columns:
            df.at[row_idx, "Senast uppdaterad källa"] = source

    wrote_any = False
    stamped_any = False
    changed: List[str] = []

    for f, v in (new_vals or {}).items():
        if f not in df.columns:
            continue

        # meningsfullt värde?
        meaningful = False
        if isinstance(v, (int, float, np.floating)):
            if f in allow_zero_for:
                meaningful = (v is not None)
            else:
                try:
                    meaningful = float(v) > 0.0
                except Exception:
                    meaningful = False
        else:
            meaningful = (str(v).strip() != "")

        if not meaningful:
            continue

        if f in TS_FIELDS:
            _stamp(f)
            stamped_any = True

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
            changed.append(f)
        else:
            if stamp_even_if_same and f in TS_FIELDS:
                changed.append(f + " (samma)")

    if stamped_any or wrote_any:
        _note_meta()
        try:
            tkr = df.at[row_idx, "Ticker"]
        except Exception:
            tkr = f"row{row_idx}"
        if changed:
            changes_map.setdefault(tkr, []).extend(changed)

    return wrote_any
