# stockapp/utils.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from datetime import datetime
import time
import random

import pandas as pd
import numpy as np

# Läs konfig som definierar schema och TS-fält
from .config import FINAL_COLS, TS_FIELDS

# ------------------------------------------------------------
# Små hjälpare
# ------------------------------------------------------------
def now_stamp() -> str:
    """Tidsstämpel i lokal tid, kompakt och läsbar."""
    return datetime.now().strftime("%Y-%m-%d %H:%M")

def safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return default
        return float(str(x).replace(" ", "").replace(",", "."))
    except Exception:
        return default

def _as_upper_ticker(x: Any) -> str:
    return ("" if x is None else str(x)).strip().upper()

def coalesce(*vals):
    for v in vals:
        if v is not None:
            return v
    return None

def parse_dt_maybe(s: Any) -> Optional[datetime]:
    if s is None:
        return None
    if isinstance(s, datetime):
        return s
    txt = str(s).strip()
    if not txt:
        return None
    # testa några vanliga format
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(txt, fmt)
        except Exception:
            pass
    return None

# ------------------------------------------------------------
# Backoff runt API/Sheets-anrop
# ------------------------------------------------------------
def with_backoff(fn: Callable, *args, retries: int = 5, min_sleep: float = 0.6, max_sleep: float = 2.4, **kwargs):
    """
    Kör fn(*args, **kwargs) med enkel backoff. Höjer sista exception om den ej lyckas.
    Användning:
        with_backoff(ws.update, [["A","B"]])
        rows = with_backoff(ws.get_all_records)
    """
    last_err = None
    for i in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            # slumpa lite så parallella körningar inte synkar sönder kvoterna
            slp = min_sleep + random.random() * (max_sleep - min_sleep)
            time.sleep(slp)
    # ge upp
    raise last_err

# ------------------------------------------------------------
# DataFrame-schema & dubbletter
# ------------------------------------------------------------
def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Säkerställ att alla kolumner i FINAL_COLS finns. Lägg till saknade med rimliga default.
    Städa Ticker: uppercase, trim, och deduplicera.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        df = pd.DataFrame({c: [] for c in FINAL_COLS})

    # Skapa saknade kolumner
    for c in FINAL_COLS:
        if c not in df.columns:
            # gissa typ: Ticker/namn-strängar -> "", annars 0.0
            if c == "Ticker" or c.upper().startswith("TS_") or "Namn" in c or "Kommentar" in c:
                df[c] = ""
            else:
                df[c] = 0.0

    # Säkerställ Ticker-kolumn
    if "Ticker" not in df.columns:
        df["Ticker"] = ""

    # Normalisera Ticker
    df["Ticker"] = df["Ticker"].map(_as_upper_ticker)

    # Deduplicera tickers
    df = dedupe_tickers(df)

    # Sätt kolumnordning om möjligt
    try:
        df = df[[c for c in FINAL_COLS if c in df.columns] + [c for c in df.columns if c not in FINAL_COLS]]
    except Exception:
        pass

    df.reset_index(drop=True, inplace=True)
    return df

def dedupe_tickers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Slå ihop dubbletter på Ticker (behåll första förekomsten).
    """
    if "Ticker" not in df.columns:
        return df
    before = len(df)
    df = df.drop_duplicates(subset=["Ticker"], keep="first")
    df.reset_index(drop=True, inplace=True)
    after = len(df)
    # Inga prints/logg här (Streamlit skriver i UI), men funktionen är robust
    return df

# ------------------------------------------------------------
# TS (tidsstämplar)
# ------------------------------------------------------------
def _ts_col_for(field_name: str) -> str:
    """
    Hitta rätt TS-kolumn för ett visst datfält (via TS_FIELDS).
    Om ej mappat -> använd 'TS_<field>'.
    """
    if field_name in TS_FIELDS:
        return TS_FIELDS[field_name]
    # fallback
    return f"TS_{field_name}"

def stamp_fields_ts(df: pd.DataFrame, row_idx: int, fields: Sequence[str]) -> pd.DataFrame:
    """
    Sätt TS-fält för givna fields på rad row_idx. Skapar TS-kolumner om de saknas.
    TS stämplas alltid (även om värdet ej ändrats), för att visa när appen hämtade/satte fältet senast.
    """
    ts_now = now_stamp()
    for f in fields:
        ts_col = _ts_col_for(f)
        if ts_col not in df.columns:
            df[ts_col] = ""
        df.at[row_idx, ts_col] = ts_now
    return df

def add_oldest_ts_col(df: pd.DataFrame, out_col: str = "_oldest_ts") -> pd.DataFrame:
    """
    Beräkna äldsta (äldsta datum) bland alla TS-kolumner i TS_FIELDS för varje rad.
    Lagra i out_col som datetime-objekt (för sortering).
    """
    # Lista alla TS-kolumner
    ts_cols = set(TS_FIELDS.values())
    ts_cols = [c for c in ts_cols if c in df.columns]
    if not ts_cols:
        df[out_col] = pd.NaT
        return df

    def _row_min_dt(row) -> Optional[datetime]:
        mins: List[datetime] = []
        for c in ts_cols:
            dt = parse_dt_maybe(row.get(c))
            if dt is not None:
                mins.append(dt)
        if not mins:
            return None
        return min(mins)

    df[out_col] = df.apply(_row_min_dt, axis=1)
    return df

# ------------------------------------------------------------
# Grundläggande beräkningar
# ------------------------------------------------------------
def _valutakurs_for(row: pd.Series, user_rates: Dict[str, float]) -> float:
    """
    Hämta valutakurs→SEK för raden (förväntar kolumn 'Valuta').
    Saknas 'Valuta' -> 1.0
    """
    if user_rates is None:
        return 1.0
    cur = str(row.get("Valuta", "SEK")).strip().upper()
    return float(user_rates.get(cur, 1.0))

def _compute_value_sek(row: pd.Series, user_rates: Dict[str, float]) -> Optional[float]:
    kurs = safe_float(row.get("Kurs"))
    antal = safe_float(row.get("Antal"))
    if kurs is None or antal is None:
        return None
    rate = _valutakurs_for(row, user_rates)
    return float(kurs) * float(antal) * float(rate)

def _compute_ps_nu(row: pd.Series) -> Optional[float]:
    """
    P/S (nu) = Market Cap / Omsättning (TTM eller 'i år (förv.)' som fallback).
    Market Cap måste vara i samma valuta som omsättningen (vanligen bolagets rapportvaluta).
    Vi utgår från att dessa fält redan är i rapportvaluta, inte SEK.
    """
    mcap = safe_float(row.get("Market Cap"))
    if mcap is None or mcap <= 0:
        return None

    # TTM eller denna årets prognos som fallback
    sales_ttm = safe_float(row.get("Omsättning TTM"))
    sales_fy = safe_float(row.get("Omsättning i år (förv.)"))

    sales = sales_ttm if (sales_ttm is not None and sales_ttm > 0) else sales_fy
    if sales is None or sales <= 0:
        return None
    return float(mcap) / float(sales)

def _compute_upside_pct(row: pd.Series) -> Optional[float]:
    """
    Upside (%) = (Riktkurs / Kurs - 1) * 100
    Båda i bolagets valuta.
    """
    rk = safe_float(row.get("Riktkurs (valuta)"))
    kp = safe_float(row.get("Kurs"))
    if rk is None or kp is None or kp <= 0:
        return None
    return (float(rk) / float(kp) - 1.0) * 100.0

def uppdatera_berakningar(df: pd.DataFrame, user_rates: Dict[str, float]) -> pd.DataFrame:
    """
    Beräkna härledda fält som används i olika vyer. Skadar aldrig – saknas kolumner ignoreras de.
    - Värde (SEK) om 'Kurs','Antal','Valuta' finns
    - P/S (nu) om 'Market Cap' + ('Omsättning TTM' eller 'Omsättning i år (förv.)') finns
    - Upside (%) om 'Riktkurs (valuta)' och 'Kurs' finns
    - Fyll vissa "visnings"-fält med sensibla default
    """
    if df is None or df.empty:
        return ensure_schema(pd.DataFrame({c: [] for c in FINAL_COLS}))

    df = ensure_schema(df.copy())

    # Värde (SEK)
    if all(c in df.columns for c in ("Kurs", "Antal", "Valuta")):
        df["Värde (SEK)"] = df.apply(lambda r: _compute_value_sek(r, user_rates) or 0.0, axis=1)

    # P/S (nu)
    if "Market Cap" in df.columns and (("Omsättning TTM" in df.columns) or ("Omsättning i år (förv.)" in df.columns)):
        df["P/S (nu)"] = df.apply(lambda r: _compute_ps_nu(r) if _compute_ps_nu(r) is not None else np.nan, axis=1)

    # Upside (%)
    if "Riktkurs (valuta)" in df.columns and "Kurs" in df.columns:
        df["Upside (%)"] = df.apply(lambda r: _compute_upside_pct(r) if _compute_upside_pct(r) is not None else np.nan, axis=1)

    # Säkerställ rimliga typer för några kolumner
    for c in ("Antal", "Kurs", "Värde (SEK)", "Market Cap", "Omsättning TTM", "Omsättning i år (förv.)", "P/S (nu)", "Upside (%)"):
        if c in df.columns:
            try:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            except Exception:
                pass

    return df
