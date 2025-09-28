# stockapp/utils.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import datetime
from typing import List, Tuple, Optional, Dict

import pandas as pd
import numpy as np


# ------------------------------------------------------------
# Datum / tidsstämplar
# ------------------------------------------------------------
def safe_parse_date(x: str) -> Optional[pd.Timestamp]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return pd.to_datetime(datetime.strptime(s, fmt))
        except Exception:
            pass
    try:
        # sista utväg
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return None


def add_oldest_ts_col(df: pd.DataFrame) -> pd.DataFrame:
    """Beräkna äldsta tidsstämpel bland alla TS_-kolumner."""
    df = df.copy()
    ts_cols = [c for c in df.columns if str(c).startswith("TS_")]
    def _oldest(row):
        ds = []
        for c in ts_cols:
            v = str(row.get(c, "")).strip()
            if not v:
                continue
            d = safe_parse_date(v)
            if d is not None and not pd.isna(d):
                ds.append(d)
        if not ds:
            return pd.NaT
        return min(ds)
    df["_oldest_any_ts"] = df.apply(_oldest, axis=1)
    df["_oldest_any_ts_fill"] = df["_oldest_any_ts"].fillna(pd.Timestamp("2099-12-31"))
    return df


# ------------------------------------------------------------
# Tickerhjälp + dubbletter
# ------------------------------------------------------------
def ensure_ticker_col(df: pd.DataFrame) -> pd.DataFrame:
    """Trim/upper Ticker och ta bort helt tomma rader."""
    df = df.copy()
    if "Ticker" not in df.columns:
        df["Ticker"] = ""
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    # sortera stabilt på Bolagsnamn, Ticker om de finns
    sort_cols = [c for c in ["Bolagsnamn", "Ticker"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(by=sort_cols).reset_index(drop=True)
    return df


def find_duplicate_tickers(df: pd.DataFrame) -> pd.DataFrame:
    """Returnera ev. dubbletter (Ticker) för användarvarning."""
    if "Ticker" not in df.columns:
        return pd.DataFrame(columns=["Ticker","count"])
    s = df["Ticker"].astype(str).str.strip().str.upper()
    grp = s.value_counts()
    dups = grp[grp > 1]
    if dups.empty:
        return pd.DataFrame(columns=["Ticker","count"])
    out = dups.rename_axis("Ticker").reset_index(name="count")
    return out


# ------------------------------------------------------------
# Formatering
# ------------------------------------------------------------
def human_money(x: float, unit: str = "", decimals: int = 2) -> str:
    """
    Snygg formattering av stora tal (market cap etc).
    Ex: 4_250_000_000_000 -> '4.25 T' (trillion), 123_000_000_000 -> '123.00 B', 45_600_000 -> '45.6 M'
    """
    try:
        v = float(x)
    except Exception:
        return f"0 {unit}".strip()

    sign = "-" if v < 0 else ""
    v = abs(v)

    if v >= 1e12:
        val, suf = v / 1e12, "T"
    elif v >= 1e9:
        val, suf = v / 1e9, "B"
    elif v >= 1e6:
        val, suf = v / 1e6, "M"
    elif v >= 1e3:
        val, suf = v / 1e3, "K"
    else:
        val, suf = v, ""

    if decimals is None:
        s_val = f"{val}"
    else:
        s_val = f"{val:.{decimals}f}".rstrip("0").rstrip(".")

    return f"{sign}{s_val} {suf}{(' ' + unit) if unit else ''}".strip()


# ------------------------------------------------------------
# P/S-snittsberäkning + riktkurser (robust)
# ------------------------------------------------------------
def compute_ps_average(row: pd.Series) -> float:
    vals = []
    for k in ["P/S Q1","P/S Q2","P/S Q3","P/S Q4"]:
        try:
            v = float(row.get(k, 0.0))
            if v > 0:
                vals.append(v)
        except Exception:
            pass
    if not vals:
        try:
            v = float(row.get("P/S", 0.0))
            return round(v, 2) if v > 0 else 0.0
        except Exception:
            return 0.0
    return round(float(np.mean(vals)), 2)


def recompute_derived(df: pd.DataFrame) -> pd.DataFrame:
    """
    - P/S-snitt = genomsnitt av Q1..Q4 (>0)
    - Market Cap (nu) = Aktuell kurs * Utestående aktier(M) * 1e6
    - Riktkurs* = (Omsättning* * P/S-snitt) / Utestående aktier(M)   (allt i samma valuta som 'Aktuell kurs')
    - Runway (mån) heuristik = max(0, Kassa / max(1, |FCF TTM|/12))
    """
    if df.empty:
        return df
    df = df.copy()

    # P/S-snitt
    df["P/S-snitt"] = df.apply(compute_ps_average, axis=1)

    # Market cap (nu)
    def _mcap(row):
        try:
            px = float(row.get("Aktuell kurs", 0.0))
            shares_m = float(row.get("Utestående aktier", 0.0))  # i miljoner
            return px * shares_m * 1e6 if px > 0 and shares_m > 0 else 0.0
        except Exception:
            return 0.0
    df["Market Cap (nu)"] = df.apply(_mcap, axis=1)

    # Riktkurser
    def _target(rev, ps, sh_m):
        try:
            rev = float(rev); ps = float(ps); sh_m = float(sh_m)
            if rev > 0 and ps > 0 and sh_m > 0:
                return round((rev * ps) / sh_m, 2)
            return 0.0
        except Exception:
            return 0.0

    for col_src, col_tgt in [
        ("Omsättning idag","Riktkurs idag"),
        ("Omsättning nästa år","Riktkurs om 1 år"),
        ("Omsättning om 2 år","Riktkurs om 2 år"),
        ("Omsättning om 3 år","Riktkurs om 3 år"),
    ]:
        df[col_tgt] = [
            _target(row.get(col_src, 0.0), row.get("P/S-snitt", 0.0), row.get("Utestående aktier", 0.0))
            for _, row in df.iterrows()
        ]

    # Runway heuristik (mån)
    def _runway(row):
        try:
            cash = float(row.get("Kassa (valuta)", 0.0))
            fcf = float(row.get("FCF TTM (valuta)", 0.0))
            if cash <= 0:
                return 0.0
            burn = abs(fcf) / 12.0 if fcf < 0 else 0.0
            if burn <= 0:
                # Ingen negativ FCF => "oändlig" runway … vi cappar till 999
                return 999.0
            return round(cash / burn, 1)
        except Exception:
            return 0.0

    if "Runway (mån)" in df.columns:
        df["Runway (mån)"] = [ _runway(row) for _, row in df.iterrows() ]

    return df


# ------------------------------------------------------------
# Små utiliteter till vyer/batch
# ------------------------------------------------------------
def top_missing_by_ts(df: pd.DataFrame, fields: List[str], limit: int = 20) -> pd.DataFrame:
    """
    Returnera en lista över bolag där angivna fält INTE har TS_ eller där TS är äldst.
    Sorterar på äldsta TS för dessa fält.
    """
    if df.empty:
        return pd.DataFrame(columns=["Ticker","Bolagsnamn","Fält","TS"])

    rows = []
    for _, r in df.iterrows():
        for f in fields:
            ts_col = f"TS_{f}"
            ts_val = str(r.get(ts_col, "")).strip()
            d = safe_parse_date(ts_val)
            rows.append({
                "Ticker": r.get("Ticker",""),
                "Bolagsnamn": r.get("Bolagsnamn",""),
                "Fält": f,
                "TS": d if d is not None else pd.NaT
            })
    out = pd.DataFrame(rows)
    out = out.sort_values(by=["TS","Bolagsnamn"]).head(limit)
    return out


def ps_consistency_flag(row: pd.Series) -> str:
    """
    En enkel sanity-check-flaggning för extrema P/S.
    """
    try:
        ps = float(row.get("P/S", 0.0))
        psn = float(row.get("P/S-snitt", 0.0))
    except Exception:
        return "?"
    vals = [v for v in [ps, psn] if v and v > 0]
    if not vals:
        return "?"
    mx = max(vals)
    if mx > 1e4:
        return "⚠️ extrem"
    if mx > 200:
        return "⚠️ hög"
    return "ok"
