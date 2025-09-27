# -*- coding: utf-8 -*-
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import streamlit as st

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

from .config import FINAL_COLS, TS_FIELDS, STANDARD_VALUTAKURSER

def _safe_float(x, default=0.0):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return float(default)
        return float(str(x).replace(" ", "").replace(",", "."))
    except Exception:
        return float(default)

def ensure_columns(df: pd.DataFrame, cols=FINAL_COLS) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            if any(x in c.lower() for x in ["kurs","omsättning","p/s","utdelning","cagr","antal","riktkurs","aktier","market cap","mcap","debt","ev","ebitda","gross","net","cash","free","burn","runway","gav"]):
                df[c] = 0.0
            elif c.startswith("TS_") or c in ("Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa"):
                df[c] = ""
            else:
                df[c] = ""
    return df.loc[:, ~df.columns.duplicated()].copy()

def to_float_cols(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

def format_money_short(x: float, ccy: str = "") -> str:
    try:
        v = float(x)
    except Exception:
        return "-"
    units = [("T", 1e12), ("B", 1e9), ("M", 1e6), ("k", 1e3)]
    for suf, lim in units:
        if abs(v) >= lim:
            return f"{v/lim:.2f}{suf} {ccy}".strip()
    return f"{v:.0f} {ccy}".strip()

def format_pct(x: float) -> str:
    try:
        return f"{float(x):.2f}%"
    except Exception:
        return "-"

def oldest_any_ts(row: pd.Series):
    dates = []
    for c in TS_FIELDS.values():
        if c in row and str(row[c]).strip():
            try:
                d = pd.to_datetime(str(row[c]).strip(), errors="coerce")
                if pd.notna(d):
                    dates.append(d)
            except Exception:
                pass
    return min(dates) if dates else None

def add_oldest_ts_col(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_oldest_any_ts"] = df.apply(oldest_any_ts, axis=1)
    df["_oldest_any_ts"] = pd.to_datetime(df["_oldest_any_ts"], errors="coerce")
    df["_oldest_any_ts_fill"] = df["_oldest_any_ts"].fillna(pd.Timestamp("2099-12-31"))
    return df

def safe_date_compare(oldest_ts, cutoff_days=365) -> bool:
    cutoff_date = (now_dt() - timedelta(days=cutoff_days)).date()
    if isinstance(oldest_ts, pd.Timestamp) and not pd.isna(oldest_ts):
        try:
            return oldest_ts.date() < cutoff_date
        except Exception:
            return False
    return False

def hamta_valutakurs(valuta: str, user_rates: dict) -> float:
    if not valuta:
        return 1.0
    return float(user_rates.get(valuta.upper(), STANDARD_VALUTAKURSER.get(valuta.upper(), 1.0)))
