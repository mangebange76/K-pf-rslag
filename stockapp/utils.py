# stockapp/utils.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Optional, Dict, List
import pandas as pd
import numpy as np

# Importera ditt kolumnschema
try:
    from .config import FINAL_COLS, TS_FIELDS
except Exception:
    # Fallback om config inte hunnit laddas i miljön ännu
    FINAL_COLS = [
        "Ticker", "Bolagsnamn", "Utestående aktier",
        "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
        "Antal aktier", "Valuta", "Årlig utdelning", "Aktuell kurs",
        "CAGR 5 år (%)", "P/S-snitt",
        "Senast manuellt uppdaterad", "Senast auto-uppdaterad", "Senast uppdaterad källa",
        "TS_Utestående aktier","TS_P/S","TS_P/S Q1","TS_P/S Q2","TS_P/S Q3","TS_P/S Q4",
        "TS_Omsättning idag","TS_Omsättning nästa år"
    ]
    TS_FIELDS = {
        "Utestående aktier":"TS_Utestående aktier",
        "P/S":"TS_P/S","P/S Q1":"TS_P/S Q1","P/S Q2":"TS_P/S Q2","P/S Q3":"TS_P/S Q3","P/S Q4":"TS_P/S Q4",
        "Omsättning idag":"TS_Omsättning idag","Omsättning nästa år":"TS_Omsättning nästa år",
    }

# ----------------------------
# Tid/nyttodelar (enkla)
# ----------------------------
def now_stamp() -> str:
    try:
        import pytz
        from datetime import datetime
        tz = pytz.timezone("Europe/Stockholm")
        return datetime.now(tz).strftime("%Y-%m-%d")
    except Exception:
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d")

# ----------------------------
# Schema & typer
# ----------------------------
_NUM_COLS = [
    "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
    "Antal aktier", "Årlig utdelning", "Aktuell kurs", "CAGR 5 år (%)", "P/S-snitt"
]

_STR_COLS = ["Ticker","Bolagsnamn","Valuta","Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa"]

def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Säkerställ att alla förväntade kolumner finns."""
    df = df.copy()
    for c in FINAL_COLS:
        if c not in df.columns:
            if c.startswith("TS_") or c in _STR_COLS:
                df[c] = ""
            elif c in _NUM_COLS:
                df[c] = 0.0
            else:
                df[c] = ""
    # Ta bort ev. duplikat-kolumner
    df = df.loc[:, ~df.columns.duplicated()].copy()
    return df

def konvertera_typer(df: pd.DataFrame) -> pd.DataFrame:
    """Konvertera kolumn-typer defensivt."""
    df = df.copy()
    for c in _NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in _STR_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str)
    for c in df.columns:
        if str(c).startswith("TS_"):
            df[c] = df[c].astype(str)
    return df

# ----------------------------
# Äldsta TS-hjälp
# ----------------------------
def oldest_any_ts(row: pd.Series) -> Optional[pd.Timestamp]:
    dates = []
    for c in TS_FIELDS.values():
        if c in row and str(row[c]).strip():
            d = pd.to_datetime(str(row[c]).strip(), errors="coerce")
            if pd.notna(d):
                dates.append(d)
    return min(dates) if dates else None

def add_oldest_ts_col(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_oldest_any_ts"] = df.apply(oldest_any_ts, axis=1)
    df["_oldest_any_ts_fill"] = df["_oldest_any_ts"].fillna(pd.Timestamp("2099-12-31"))
    return df

# ----------------------------
# Kärnberäkningar (robusta)
# ----------------------------
def _num(v) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0

def uppdatera_berakningar(df: pd.DataFrame, user_rates: Dict[str, float] | None = None) -> pd.DataFrame:
    """
    Räknar:
      - P/S-snitt = snitt av positiva P/S Q1–Q4
      - Omsättning om 2 & 3 år från 'Omsättning nästa år' med CAGR clamp ( >100%→50%, <0%→2% )
      - Riktkurser idag/1/2/3 = (Omsättning * P/S-snitt) / Utestående aktier
    Robust mot saknade kolumner och konstiga typer.
    """
    df = ensure_schema(df)
    df = konvertera_typer(df)

    # Säkerställ kolumner som vi fyller
    for c in ["P/S-snitt","Omsättning om 2 år","Omsättning om 3 år","Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"]:
        if c not in df.columns:
            df[c] = 0.0

    # Gå rad för rad
    for i in range(len(df)):
        # P/S-snitt
        ps_vals = [
            _num(df.at[i, "P/S Q1"]) if "P/S Q1" in df.columns else 0.0,
            _num(df.at[i, "P/S Q2"]) if "P/S Q2" in df.columns else 0.0,
            _num(df.at[i, "P/S Q3"]) if "P/S Q3" in df.columns else 0.0,
            _num(df.at[i, "P/S Q4"]) if "P/S Q4" in df.columns else 0.0,
        ]
        ps_clean = [x for x in ps_vals if x and x > 0]
        ps_snitt = round(float(np.mean(ps_clean)), 2) if ps_clean else 0.0
        df.at[i, "P/S-snitt"] = ps_snitt

        # CAGR clamp
        cagr = _num(df.at[i, "CAGR 5 år (%)"]) if "CAGR 5 år (%)" in df.columns else 0.0
        if cagr > 100.0:
            just_cagr = 50.0
        elif cagr < 0.0:
            just_cagr = 2.0
        else:
            just_cagr = cagr
        g = just_cagr / 100.0

        # Omsättningar
        oms_next = _num(df.at[i, "Omsättning nästa år"]) if "Omsättning nästa år" in df.columns else 0.0
        if oms_next > 0:
            df.at[i, "Omsättning om 2 år"] = round(oms_next * (1.0 + g), 2)
            df.at[i, "Omsättning om 3 år"] = round(oms_next * ((1.0 + g) ** 2), 2)
        else:
            # Lämna ev. tidigare värden orörda (de är redan konverterade till numeriska)
            pass

        # Riktkurser
        aktier_ut_m = _num(df.at[i, "Utestående aktier"]) if "Utestående aktier" in df.columns else 0.0  # i miljoner
        # konvertera till styck om >0
        aktier_ut = aktier_ut_m * 1e6 if aktier_ut_m > 0 else 0.0

        if aktier_ut > 0 and ps_snitt > 0:
            oms_idag  = _num(df.at[i, "Omsättning idag"]) if "Omsättning idag" in df.columns else 0.0
            oms_1     = _num(df.at[i, "Omsättning nästa år"]) if "Omsättning nästa år" in df.columns else 0.0
            oms_2     = _num(df.at[i, "Omsättning om 2 år"]) if "Omsättning om 2 år" in df.columns else 0.0
            oms_3     = _num(df.at[i, "Omsättning om 3 år"]) if "Omsättning om 3 år" in df.columns else 0.0

            df.at[i, "Riktkurs idag"]    = round((oms_idag * ps_snitt) / aktier_ut, 4) if oms_idag > 0 else 0.0
            df.at[i, "Riktkurs om 1 år"] = round((oms_1   * ps_snitt) / aktier_ut, 4) if oms_1   > 0 else 0.0
            df.at[i, "Riktkurs om 2 år"] = round((oms_2   * ps_snitt) / aktier_ut, 4) if oms_2   > 0 else 0.0
            df.at[i, "Riktkurs om 3 år"] = round((oms_3   * ps_snitt) / aktier_ut, 4) if oms_3   > 0 else 0.0
        else:
            df.at[i, "Riktkurs idag"] = 0.0
            df.at[i, "Riktkurs om 1 år"] = 0.0
            df.at[i, "Riktkurs om 2 år"] = 0.0
            df.at[i, "Riktkurs om 3 år"] = 0.0

    return df
