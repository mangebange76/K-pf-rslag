# stockapp/storage.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import datetime
from typing import Tuple

import pandas as pd
import streamlit as st

from .config import SHEET_NAME, FINAL_COLS, SNAPSHOT_PREFIX
from .utils import ensure_schema, dedupe_tickers
from .sheets import get_ws, ws_read_df, ws_write_df, ensure_headers


# ------------------------------------------------------------
# Publika funktioner
# ------------------------------------------------------------
def hamta_data() -> pd.DataFrame:
    """
    Läs huvudbladet (SHEET_NAME) som DataFrame.
    - Returnerar alltid en DataFrame med minst FINAL_COLS som kolumner
    - Dubbletter (Ticker) tas inte bort här – det görs endast i minnet av appen
    """
    ws = get_ws(SHEET_NAME)
    try:
        df = ws_read_df(ws)
    except Exception as e:
        # Vid läsfel: exponera varning och ge en tom DF
        st.warning(f"⚠️ Kunde inte läsa Google Sheet: {e}")
        df = pd.DataFrame(columns=FINAL_COLS)

    # Säkerställ schema (lägg till saknade kolumner)
    df = ensure_schema(df, FINAL_COLS)
    return df


def spara_data(df: pd.DataFrame, do_snapshot: bool = False) -> Tuple[pd.DataFrame, int]:
    """
    Skriv DataFrame till huvudbladet (SHEET_NAME).
    - Säkerställer schema (FINAL_COLS)
    - Tar bort dubbletter (Ticker) innan skrivning (för att hålla bladet “rent”)
    - Optionellt skapa snapshot-blad

    Returnerar (df_som_skrev, antal_dubbletter_borttagna)
    """
    if df is None:
        raise ValueError("df är None")

    # 1) Säkerställ schema och ordning
    work = ensure_schema(df.copy(), FINAL_COLS)
    work = work[FINAL_COLS]  # skriv i en konsekvent kolumnordning

    # 2) Dubblettstädning (Ticker)
    cleaned, dups = dedupe_tickers(work)
    dup_count = len(dups)

    # 3) Skriv till blad
    ws = get_ws(SHEET_NAME)
    ensure_headers(ws, FINAL_COLS)
    ws_write_df(ws, cleaned)

    # 4) Snapshot om begärt
    if do_snapshot:
        try:
            _spara_snapshot(cleaned)
        except Exception as e:
            st.warning(f"⚠️ Kunde inte skapa snapshot: {e}")

    return cleaned, dup_count


# ------------------------------------------------------------
# Hjälpare
# ------------------------------------------------------------
def _spara_snapshot(df: pd.DataFrame) -> None:
    """
    Skapa ett nytt blad med en tidsstämplad kopia av df.
    """
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    title = f"{SNAPSHOT_PREFIX}{ts}"
    ws = get_ws(title, rows=max(1000, len(df) + 10), cols=max(60, len(df.columns) + 5))
    ensure_headers(ws, list(df.columns))
    ws_write_df(ws, df)
