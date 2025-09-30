# stockapp/manuals.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, List
import pandas as pd


def _to_datetime_silent(series: pd.Series) -> pd.Series:
    """Konvertera valfri serie till datetime (tyst felhantering)."""
    return pd.to_datetime(series, errors="coerce")


def _min_ts_of_columns(row: pd.Series, cols: List[str]) -> Optional[pd.Timestamp]:
    """Returnera minsta TS i raden för givna kolumner (ignorerar NaT)."""
    vals = []
    for c in cols:
        if c in row and pd.notna(row[c]):
            vals.append(row[c])
    if not vals:
        return pd.NaT
    return min(vals)


def build_requires_manual_df(df: pd.DataFrame, older_than_days: Optional[int] = None) -> pd.DataFrame:
    """
    Bygger listan 'Manuell prognoslista':
      - Primär sort: 'Senaste TS (min av två)' = min(ts Omsättning idag, ts Omsättning nästa år)
      - Fallback-sort: '_Äldsta TS (alla fält)' = min av alla TS_-kolumner i raden
      - older_than_days: om satt, filtrerar till endast rader äldre än cutoff

    Returnerar en DataFrame färdig för visning/sortering.
    Förutsätter att TS-kolumner heter i stil med 'TS: Omsättning idag' (kolon-notationen).
    """

    work = df.copy()

    # Identifiera TS-kolumner
    ts_cols_all = [c for c in work.columns if str(c).startswith("TS:")]
    ts_today = "TS: Omsättning idag" if "TS: Omsättning idag" in work.columns else None
    ts_nexty = "TS: Omsättning nästa år" if "TS: Omsättning nästa år" in work.columns else None

    # Konvertera alla TS-kolumner till datetime
    for c in ts_cols_all:
        work[c] = _to_datetime_silent(work[c])

    # Primär metrik: min av de två manuella TS om någon finns
    if ts_today or ts_nexty:
        mins = []
        for _, row in work.iterrows():
            vals = []
            if ts_today and pd.notna(row.get(ts_today)):
                vals.append(row.get(ts_today))
            if ts_nexty and pd.notna(row.get(ts_nexty)):
                vals.append(row.get(ts_nexty))
            mins.append(min(vals) if vals else pd.NaT)
        work["Senaste TS (min av två)"] = mins
    else:
        work["Senaste TS (min av två)"] = pd.NaT

    # Fallback: min av alla TS_-fält (om de finns)
    if ts_cols_all:
        work["_Äldsta TS (alla fält)"] = work[ts_cols_all].min(axis=1, skipna=True)
    else:
        work["_Äldsta TS (alla fält)"] = pd.NaT

    # Filtrera på ålder (om efterfrågat)
    if older_than_days is not None:
        cutoff = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=int(older_than_days))
        mask = (
            (work["Senaste TS (min av två)"].notna() & (work["Senaste TS (min av två)"] < cutoff))
            |
            (work["Senaste TS (min av två)"].isna() & work["_Äldsta TS (alla fält)"].notna() & (work["_Äldsta TS (alla fält)"] < cutoff))
        )
        work = work.loc[mask].copy()

    # Slutlig sorteringsnyckel: använd primär om någon rad har den, annars fallback
    use_primary = work["Senaste TS (min av två)"].notna().any()
    sort_key = "Senaste TS (min av två)" if use_primary else "_Äldsta TS (alla fält)"

    # Sortera — rader utan TS hamnar först (mest akuta)
    work = work.sort_values(by=sort_key, ascending=True, na_position="first")

    # Plocka ut relevanta kolumner
    show_cols = ["Ticker", "Bolagsnamn"]
    for c in ("Sector", "Industry"):
        if c in work.columns:
            show_cols.append(c)
    for c in ("Omsättning idag", "Omsättning nästa år"):
        if c in work.columns:
            show_cols.append(c)
    for c in ("TS: Omsättning idag", "TS: Omsättning nästa år"):
        if c in work.columns:
            show_cols.append(c)
    show_cols += ["Senaste TS (min av två)", "_Äldsta TS (alla fält)"]
    show_cols = [c for c in show_cols if c in work.columns]

    return work[show_cols].reset_index(drop=True)
