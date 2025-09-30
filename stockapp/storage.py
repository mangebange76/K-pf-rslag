# stockapp/storage.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import time
from typing import List
import pandas as pd
import streamlit as st

from .config import SHEET_NAME, FINAL_COLS
from .sheets import get_ws
from .utils import ensure_schema, with_backoff


HEADER_ROW = FINAL_COLS  # vi skriver alltid ut vår definierade schema-header


def _values_to_df(values: List[List[str]]) -> pd.DataFrame:
    """
    Tar list[list] från Google Sheets och returnerar DataFrame.
    Robust mot tomma ark och varierande rader.
    """
    if not values:
        # tomt ark → returnera tom df med rätt schema
        return ensure_schema(pd.DataFrame(columns=FINAL_COLS))

    header = [str(h).strip() for h in values[0]]
    rows = values[1:] if len(values) > 1 else []

    # Alignera antal kolumner (padding/trim)
    width = len(header)
    norm_rows = []
    for r in rows:
        if len(r) < width:
            r = r + [""] * (width - len(r))
        elif len(r) > width:
            r = r[:width]
        norm_rows.append(r)

    df = pd.DataFrame(norm_rows, columns=header)
    df = ensure_schema(df)
    return df


def hamta_data() -> pd.DataFrame:
    """
    Läser huvudarket (SHEET_NAME) och levererar DataFrame med garanterat schema (FINAL_COLS).
    Fixar bl.a. klassikern “too many values to unpack” genom korrekt hantering av get_all_values().
    """
    try:
        ws = get_ws(SHEET_NAME)
        values = with_backoff(ws.get_all_values)  # -> List[List[str]]
        df = _values_to_df(values)
        return df
    except Exception as e:
        st.error(f"⚠️ Kunde inte läsa data från Google Sheet: {e}")
        # Falla tillbaka till tom df med schema så appen kan fortsätta köras
        return ensure_schema(pd.DataFrame(columns=FINAL_COLS))


def spara_data(df: pd.DataFrame, do_snapshot: bool = False) -> None:
    """
    Skriver hela DataFrame till Google Sheet (SHEET_NAME) med HEADER_ROW överst.
    Om do_snapshot=True tas en enkel backup till ett worksheet 'Snapshots' med tidsstämpel i A1.
    """
    ws = get_ws(SHEET_NAME)

    # Säkerställ kolumnordning och schema innan skrivning
    out = ensure_schema(df).copy()
    out = out[FINAL_COLS]

    # Konvertera till str för Sheets
    body = [HEADER_ROW]
    for _, row in out.iterrows():
        body.append([str(row.get(c, "")) if pd.notna(row.get(c, "")) else "" for c in FINAL_COLS])

    with_backoff(ws.clear)
    with_backoff(ws.update, body)

    if do_snapshot:
        try:
            snap = get_ws("Snapshots", create_if_missing=True, rows=2, cols=2)
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            with_backoff(snap.update, [[f"Snapshot {ts}"], ["Radantal", str(len(out))]])
        except Exception:
            # snapshot får aldrig fälla hela sparningen
            pass
