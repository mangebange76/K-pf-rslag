# stockapp/utils.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Lokal Stockholmstid om möjligt
try:
    import pytz
    TZ_STHLM = pytz.timezone("Europe/Stockholm")
    def now_dt(): return datetime.now(TZ_STHLM)
except Exception:
    def now_dt(): return datetime.now()


# ------------------------------------------------------------
# Formatering
# ------------------------------------------------------------
def human_money(v, unit: str = "") -> str:
    """Formaterar stora tal till T/B/M/K."""
    try:
        x = float(v)
    except Exception:
        return "-"
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1_000_000_000_000:
        return f"{sign}{x/1_000_000_000_000:.2f}T {unit}".strip()
    if x >= 1_000_000_000:
        return f"{sign}{x/1_000_000_000:.2f}B {unit}".strip()
    if x >= 1_000_000:
        return f"{sign}{x/1_000_000:.2f}M {unit}".strip()
    if x >= 1_000:
        return f"{sign}{x/1_000:.2f}K {unit}".strip()
    return f"{sign}{x:.0f} {unit}".strip()


# ------------------------------------------------------------
# TS & hjälpare
# ------------------------------------------------------------
def add_oldest_ts_col(df: pd.DataFrame) -> pd.DataFrame:
    """Beräkna äldsta TS_ kolumn per rad."""
    ts_cols = [c for c in df.columns if str(c).startswith("TS_")]
    def _oldest(row):
        dates = []
        for c in ts_cols:
            s = str(row.get(c,"")).strip()
            if not s:
                continue
            try:
                d = pd.to_datetime(s, errors="coerce")
                if pd.notna(d):
                    dates.append(d)
            except Exception:
                pass
        return min(dates) if dates else pd.NaT

    out = df.copy()
    out["_oldest_any_ts"] = out.apply(_oldest, axis=1)
    out["_oldest_any_ts_fill"] = out["_oldest_any_ts"].fillna(pd.Timestamp("2099-12-31"))
    return out


def top_missing_by_ts(df: pd.DataFrame, fields: List[str], limit: int = 30) -> pd.DataFrame:
    """Lista bolag med äldst TS bland utvalda fält."""
    cols = []
    for f in fields:
        ts = f"TS_{f}"
        if ts not in df.columns:
            df[ts] = ""
        cols.append(ts)

    def _oldest_row(row):
        dts = []
        for f in fields:
            ts = f"TS_{f}"
            s = str(row.get(ts,"")).strip()
            d = pd.to_datetime(s, errors="coerce") if s else pd.NaT
            dts.append(d)
        dates = [d for d in dts if pd.notna(d)]
        return min(dates) if dates else pd.NaT

    tmp = df.copy()
    tmp["_old_sel"] = tmp.apply(_oldest_row, axis=1)
    tmp["_old_sel_fill"] = tmp["_old_sel"].fillna(pd.Timestamp("1990-01-01"))
    tmp = tmp.sort_values(by="_old_sel_fill", ascending=True)

    show = ["Ticker","Bolagsnamn"] + [f"TS_{f}" for f in fields] + ["_old_sel"]
    show = [c for c in show if c in tmp.columns]
    return tmp[show].head(limit)


# ------------------------------------------------------------
# Sanity
# ------------------------------------------------------------
def ps_consistency_flag(row: pd.Series) -> str:
    try:
        ps = float(row.get("P/S", 0.0) or 0.0)
        psn = float(row.get("P/S-snitt", 0.0) or 0.0)
    except Exception:
        return "bad"
    if ps <= 0 and psn <= 0:
        return "missing"
    if ps > 2000 or psn > 2000:
        return "extreme"
    if psn > 0 and ps / psn > 20:
        return ">> snitt"
    return "ok"


# ------------------------------------------------------------
# Härledda fält (samma anda som din tidigare uppdatera_berakningar)
# ------------------------------------------------------------
def recompute_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # P/S-snitt (positiva Q1..Q4)
    ps_cols = [c for c in ["P/S Q1","P/S Q2","P/S Q3","P/S Q4"] if c in out.columns]
    if ps_cols:
        def _psn(row):
            vals = []
            for c in ps_cols:
                try:
                    v = float(row.get(c, 0.0))
                    if v > 0:
                        vals.append(v)
                except Exception:
                    pass
            return round(float(np.mean(vals)), 2) if vals else 0.0
        out["P/S-snitt"] = out.apply(_psn, axis=1)
    else:
        if "P/S-snitt" not in out.columns:
            out["P/S-snitt"] = 0.0

    # Market Cap (nu) = pris * shares (miljoner * 1e6)
    if "Market Cap (nu)" not in out.columns:
        out["Market Cap (nu)"] = 0.0
    for i, r in out.iterrows():
        try:
            px = float(r.get("Aktuell kurs", 0.0) or 0.0)
            sh_m = float(r.get("Utestående aktier", 0.0) or 0.0) * 1_000_000.0
            out.at[i, "Market Cap (nu)"] = px * sh_m if px > 0 and sh_m > 0 else 0.0
        except Exception:
            out.at[i, "Market Cap (nu)"] = 0.0

    # Prognoser för omsättning (om saknas): använd "Omsättning nästa år" och en enkel CAGR clamp (2–50%)
    if "CAGR 5 år (%)" not in out.columns:
        out["CAGR 5 år (%)"] = 0.0

    for i, r in out.iterrows():
        cagr = 0.0
        try:
            cagr_raw = float(r.get("CAGR 5 år (%)", 0.0) or 0.0)
            cagr = 50.0 if cagr_raw > 100.0 else (2.0 if cagr_raw < 0.0 else cagr_raw)
        except Exception:
            cagr = 0.0
        g = cagr / 100.0

        try:
            nxt = float(r.get("Omsättning nästa år", 0.0) or 0.0)
        except Exception:
            nxt = 0.0

        # fyll Omsättning om 2 & 3 år om de saknas, utifrån growth g
        if "Omsättning om 2 år" in out.columns:
            if float(r.get("Omsättning om 2 år", 0.0) or 0.0) <= 0 and nxt > 0:
                out.at[i, "Omsättning om 2 år"] = round(nxt * (1.0 + g), 2)
        else:
            out["Omsättning om 2 år"] = 0.0
            if nxt > 0:
                out.at[i, "Omsättning om 2 år"] = round(nxt * (1.0 + g), 2)

        if "Omsättning om 3 år" in out.columns:
            if float(r.get("Omsättning om 3 år", 0.0) or 0.0) <= 0 and nxt > 0:
                out.at[i, "Omsättning om 3 år"] = round(nxt * ((1.0 + g) ** 2), 2)
        else:
            out["Omsättning om 3 år"] = 0.0
            if nxt > 0:
                out.at[i, "Omsättning om 3 år"] = round(nxt * ((1.0 + g) ** 2), 2)

    # Riktkurser = (Omsättning * P/S-snitt) / Utestående aktier
    for i, r in out.iterrows():
        try:
            sh_m = float(r.get("Utestående aktier", 0.0) or 0.0)
            psn  = float(r.get("P/S-snitt", 0.0) or 0.0)
        except Exception:
            sh_m = 0.0; psn = 0.0

        def _rk(v):
            try:
                rev_m = float(v or 0.0) # miljoner
                if sh_m > 0 and psn > 0 and rev_m > 0:
                    return round((rev_m * 1_000_000.0 * psn) / (sh_m * 1_000_000.0), 2)
            except Exception:
                pass
            return 0.0

        out.at[i, "Riktkurs idag"]    = _rk(r.get("Omsättning idag", 0.0))
        out.at[i, "Riktkurs om 1 år"] = _rk(r.get("Omsättning nästa år", 0.0))
        out.at[i, "Riktkurs om 2 år"] = _rk(out.at[i, "Omsättning om 2 år"])
        out.at[i, "Riktkurs om 3 år"] = _rk(out.at[i, "Omsättning om 3 år"])

    return out
