# -*- coding: utf-8 -*-
"""
Läs/skriv portföljdatan via sheets.py.
- hamta_data(): läser aldrig och skriver aldrig rubriker; endast läsning.
- spara_data(df): ersätter allt innehåll på målfliken.
"""

from __future__ import annotations
import pandas as pd
import streamlit as st

from .config import FINAL_COLS
from .sheets import get_ws, ws_read_df, ws_write_df
from .utils import ensure_schema, dedupe_tickers


def hamta_data() -> pd.DataFrame:
    ws = get_ws()
    df = ws_read_df(ws)

    # Normalisera kolumnnamn: trimma whitespace
    df.columns = [str(c).strip() for c in df.columns]

    # Säkerställ schema (lägger till saknade kolumner, ändrar inte befintliga värden)
    df = ensure_schema(df, FINAL_COLS)

    # Ta bort tomma rader (helt tom ticker)
    df = df[df["Ticker"].astype(str).str.strip() != ""].copy()

    # Dubblettskydd i minnet
    df, _ = dedupe_tickers(df)

    return df.reset_index(drop=True)


def spara_data(df: pd.DataFrame) -> None:
    if df is None:
        return
    df = ensure_schema(df, FINAL_COLS)
    ws = get_ws()
    ws_write_df(ws, df)
    st.toast(f"Sparade {len(df)} rader till fliken '{ws.title}'.")
