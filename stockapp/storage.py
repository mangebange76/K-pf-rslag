# -*- coding: utf-8 -*-
"""
Högre nivå ovanpå sheets.py:
- hamta_data()  -> DataFrame (med fallback till rätt flik)
- spara_data(df)
"""

from __future__ import annotations
from typing import List

import pandas as pd
import streamlit as st

from .config import SHEET_NAME, FINAL_COLS, MAX_ROWS_WRITE
from .sheets import get_ws, ws_read_df, ws_write_df, ensure_headers
from .utils import ensure_schema


def hamta_data() -> pd.DataFrame:
    """
    Läser DataFrame från Google Sheet.
    - Försöker först SHEET_NAME; annars fallback (flik med rubrik som innehåller 'Ticker').
    - Ingen filtrering – hela tabellen returneras.
    """
    # get_ws gör redan fallback och skriver caption om det sker
    ws = get_ws(SHEET_NAME)
    df = ws_read_df(ws)

    # Se till att vi åtminstone har vårt minimischema (ordning fixas i appen igen)
    df = ensure_schema(df, FINAL_COLS)
    return df


def spara_data(df: pd.DataFrame) -> None:
    """
    Skriver DataFrame till SHEET_NAME (eller samma fallback-flik som vid läsning).
    - Trimmar ner till MAX_ROWS_WRITE rader för säkerhet.
    """
    if df is None:
        return
    df2 = ensure_schema(df, FINAL_COLS)
    if len(df2) > MAX_ROWS_WRITE:
        st.warning(f"Antalet rader ({len(df2)}) överstiger MAX_ROWS_WRITE ({MAX_ROWS_WRITE}). Klipper vid gränsen.")
        df2 = df2.iloc[:MAX_ROWS_WRITE].copy()

    ws = get_ws(SHEET_NAME)
    ensure_headers(ws, list(df2.columns))
    ws_write_df(ws, df2)
