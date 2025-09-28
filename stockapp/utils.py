# stockapp/utils.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Optional, Callable, Any, Dict, List
import time
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st

# Viktigt: importera ENDAST config här (ingen import från rates för att undvika cirkulär import)
from .config import FINAL_COLS, TS_FIELDS

# ------------------------------------------------------------
# Tids-helpers (Stockholm-tid om pytz finns)
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Backoff-hjälpare
# ------------------------------------------------------------
def with_backoff(func: Callable[..., Any], *args, **kwargs) -> Any:
    """
    Kör func(*args, **kwargs) med mild backoff för att mildra kvot/transienta fel.
    Kastar sista felet vidare om alla försök misslyckas.
    """
    delays = [0.0, 0.5, 1.0, 2.0]
    last_err = None
    for d in delays:
        if d:
            time.sleep(d)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_err = e
    raise last_err

# ------------------------------------------------------------
# DataFrame schema/typer
# ------------------------------------------------------------
def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Säkerställer att alla FINAL_COLS finns. Skapar saknade kolumner med rimliga defaults.
    Tar även bort dubblett-kolumner.
    """
    if df is None or df.empty:
        df = pd.DataFrame({c: [] for c in FINAL_COLS})

    for kol in FINAL_COLS:
        if kol not in df.columns:
            if any(x in kol.lower() for x in ["kurs","omsättning","p/s","utdelning","cagr","antal","riktkurs","aktier","snitt","cap","ebitda","pe","ev"]):
                df[kol] = 0.0
            elif kol.startswith("TS_"):
                df[kol] = ""  # tidsstämplar som YYYY-MM-DD
            elif kol in ("Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa"):
                df[kol] = ""
            else:
                df[kol] = ""

    # Ta bort eventuella dubletter
    df = df.loc[:, ~df.columns.duplicated()].copy()
    return df

def konvertera_typer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kasta numeriska kolumner till float och strängkolumner till str.
    Ignorerar kolumner som inte finns.
    """
    if df is None or df.empty:
        return df

    num_cols = [
        "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
        "Antal aktier", "Årlig utdelning", "Aktuell kurs",
        "CAGR 5 år (%)", "P/S-snitt",
        # Vanliga nya fält i din app (ignoreras om de inte finns)
        "Market Cap", "EV/EBITDA", "PE", "Net Debt", "Gross Margin", "Net Margin",
        "MCAP_Q1", "MCAP_Q2", "MCAP_Q3", "MCAP_Q4",
        "GAV (SEK)"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    str_cols = [
        "Ticker","Bolagsnamn","Valuta",
        "Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa",
        "Sektor","Industri","Risklabel","Värderingsstatus"
    ]
    for c in str_cols:
        if c in df.columns:
            df[c] = df[c].astype(str)

    for c in df.columns:
        if str(c).startswith("TS_"):
            df[c] = df[c].astype(str)

    return df

# ------------------------------------------------------------
# Beräkningar (P/S-snitt, riktkurser, framtida omsättning)
# ------------------------------------------------------------
def _clamp_cagr(cagr: float) -> float:
    """
    Clamp-logik enligt tidigare app: >100% → 50%, <0% → 2%.
    """
    if cagr > 100.0:
        return 50.0
    if cagr < 0.0:
        return 2.0
    return cagr

def uppdatera_berakningar(df: pd.DataFrame, user_rates: Dict[str, float]) -> pd.DataFrame:
    """
    Beräknar:
      - P/S-snitt som snitt av positiva Q1–Q4
      - Omsättning om 2 & 3 år från 'Omsättning nästa år' med CAGR clamp
      - Riktkurser (idag/1/2/3) = (Omsättning * P/S-snitt) / Utestående aktier
    OBS: Omsättningsfält är i miljoner i bolagets valuta; riktkurs i samma valuta.
    """
    if df is None or df.empty:
        return df

    # Säkerställ typer före loop
    df = konvertera_typer(df)

    for i, rad in df.iterrows():
        # P/S-snitt
        ps_vals = [
            float(rad.get("P/S Q1", 0.0)),
            float(rad.get("P/S Q2", 0.0)),
            float(rad.get("P/S Q3", 0.0)),
            float(rad.get("P/S Q4", 0.0)),
        ]
        ps_clean = [x for x in ps_vals if x > 0.0]
        ps_snitt = round(float(np.mean(ps_clean)), 2) if ps_clean else 0.0
        df.at[i, "P/S-snitt"] = ps_snitt

        # CAGR clamp
        cagr5 = float(rad.get("CAGR 5 år (%)", 0.0))
        adj_cagr = _clamp_cagr(cagr5) / 100.0

        # Omsättning om 2 & 3 år
        oms_next = float(rad.get("Omsättning nästa år", 0.0))
        if oms_next > 0:
            df.at[i, "Omsättning om 2 år"] = round(oms_next * (1.0 + adj_cagr), 2)
            df.at[i, "Omsättning om 3 år"] = round(oms_next * ((1.0 + adj_cagr) ** 2), 2)
        else:
            # behåll ev. befintliga
            df.at[i, "Omsättning om 2 år"] = float(rad.get("Omsättning om 2 år", 0.0))
            df.at[i, "Omsättning om 3 år"] = float(rad.get("Omsättning om 3 år", 0.0))

        # Riktkurser (förutsätter Utestående aktier i miljoner och omsättning i miljoner)
        aktier_ut_milj = float(rad.get("Utestående aktier", 0.0))  # milj.
        if aktier_ut_milj > 0.0 and ps_snitt > 0.0:
            def _rk(oms_milj: float) -> float:
                # värde = oms_milj * ps / (aktier i milj)  => pris per aktie
                try:
                    return round((float(oms_milj) * ps_snitt) / aktier_ut_milj, 2)
                except Exception:
                    return 0.0

            df.at[i, "Riktkurs idag"]    = _rk(float(rad.get("Omsättning idag", 0.0)))
            df.at[i, "Riktkurs om 1 år"] = _rk(float(rad.get("Omsättning nästa år", 0.0)))
            df.at[i, "Riktkurs om 2 år"] = _rk(float(df.at[i, "Omsättning om 2 år"]))
            df.at[i, "Riktkurs om 3 år"] = _rk(float(df.at[i, "Omsättning om 3 år"]))
        else:
            df.at[i, "Riktkurs idag"] = 0.0
            df.at[i, "Riktkurs om 1 år"] = 0.0
            df.at[i, "Riktkurs om 2 år"] = 0.0
            df.at[i, "Riktkurs om 3 år"] = 0.0

    return df

# ------------------------------------------------------------
# TS-hjälpare (används av vyer/formulär ibland)
# ------------------------------------------------------------
def stamp_ts_for_field(df: pd.DataFrame, row_idx: int, field: str, when: Optional[str] = None) -> None:
    """
    Sätt TS_-kolumnen för ett spårat fält om den finns.
    """
    ts_col = TS_FIELDS.get(field)
    if not ts_col:
        return
    date_str = when if when else now_stamp()
    try:
        df.at[row_idx, ts_col] = date_str
    except Exception:
        pass

def note_auto_update(df: pd.DataFrame, row_idx: int, source: str) -> None:
    """
    Sätt senaste auto-uppdaterad + källa.
    """
    try:
        df.at[row_idx, "Senast auto-uppdaterad"] = now_stamp()
        df.at[row_idx, "Senast uppdaterad källa"] = source
    except Exception:
        pass

def note_manual_update(df: pd.DataFrame, row_idx: int) -> None:
    """
    Sätt senaste manuellt uppdaterad.
    """
    try:
        df.at[row_idx, "Senast manuellt uppdaterad"] = now_stamp()
    except Exception:
        pass

# ------------------------------------------------------------
# List-helpers för "äldsta TS" (används i kontroll/batch)
# ------------------------------------------------------------
def oldest_any_ts(row: pd.Series) -> Optional[pd.Timestamp]:
    """
    Returnera äldsta (minsta) tidsstämpeln bland alla TS_-kolumner i raden.
    """
    dates: List[pd.Timestamp] = []
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
    """
    Lägg till två hjälpkolumner: _oldest_any_ts och _oldest_any_ts_fill (för sortering).
    """
    df["_oldest_any_ts"] = df.apply(oldest_any_ts, axis=1)
    df["_oldest_any_ts"] = pd.to_datetime(df["_oldest_any_ts"], errors="coerce")
    df["_oldest_any_ts_fill"] = df["_oldest_any_ts"].fillna(pd.Timestamp("2099-12-31"))
    return df

# ------------------------------------------------------------
# Små helpers
# ------------------------------------------------------------
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    try:
        return df.to_csv(index=False).encode("utf-8")
    except Exception:
        return b""
