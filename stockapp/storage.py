# -*- coding: utf-8 -*-
"""
stockapp.storage
-----------------
Stabil in/ut till Google Sheets.
- hamta_data() -> pd.DataFrame (ALDRIG None)
- spara_data(df) : säker skrivning (raderar inte om df är tomt)
"""

from __future__ import annotations

from typing import List
import streamlit as st
import pandas as pd
import numpy as np

from .config import FINAL_COLS, SHEET_NAME
from .sheets import get_ws, safe_get_all_values
from .utils import ensure_schema, with_backoff, dedupe_tickers


def _normalize_header(h: List[str]) -> List[str]:
    """Trimma / normalisera rubriker marginellt."""
    out = []
    for x in h:
        s = (x or "").strip()
        out.append(s)
    return out


def _to_dataframe(values: List[List[str]]) -> pd.DataFrame:
    """
    Bygger en DataFrame från get_all_values().
    Antagande: första raden = rubriker. Resterande = data.
    Tål extra tomma rader/kolumner.
    """
    if not values:
        return pd.DataFrame(columns=FINAL_COLS)

    # Ta bort helt tomma rader i slutet
    cleaned = []
    for row in values:
        # Kolla om raden är helt tom
        if not any(cell.strip() for cell in row):
            continue
        cleaned.append(row)

    if not cleaned:
        return pd.DataFrame(columns=FINAL_COLS)

    header = _normalize_header(cleaned[0])
    data_rows = cleaned[1:] if len(cleaned) > 1 else []

    # Klipp rader till header-längd (gspread kan ge ojämna rader)
    width = len(header)
    fixed_rows = []
    for r in data_rows:
        if len(r) < width:
            r = r + [""] * (width - len(r))
        elif len(r) > width:
            r = r[:width]
        fixed_rows.append(r)

    try:
        df = pd.DataFrame(fixed_rows, columns=header)
    except Exception:
        # Om header är tom eller trasig: returnera tom df med FINAL_COLS
        return pd.DataFrame(columns=FINAL_COLS)

    # Ta bort helt tomma rader (t.ex. där Ticker saknas + alla andra fält tomma)
    def _row_all_empty(sr: pd.Series) -> bool:
        return all((str(v).strip() == "" for v in sr.values))

    if not df.empty:
        mask = ~df.apply(_row_all_empty, axis=1)
        df = df.loc[mask].copy()

    # Säkerställ schema
    df = ensure_schema(df, FINAL_COLS)

    # Trimma Ticker/Name/Sektor på whitespace
    for col in ["Ticker", "Namn", "Sektor", "Sector"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Ta bort rader utan Ticker
    if "Ticker" in df.columns:
        df = df[df["Ticker"] != ""].copy()

    # Deduplicera Ticker (behåll första)
    if "Ticker" in df.columns and not df.empty:
        before = len(df)
        df = dedupe_tickers(df)
        after = len(df)
        if after < before:
            st.info(f"Deduplicerade tickers: {before - after} dubblett(er) togs bort.")

    return df


def hamta_data() -> pd.DataFrame:
    """
    Läser in hela databasen från Google Sheet.
    Returnerar alltid en DataFrame med FINAL_COLS (kan vara tom men inte None).
    """
    try:
        ws = get_ws(SHEET_NAME)
        vals = safe_get_all_values(ws)
        df = _to_dataframe(vals)
        # För tydlighet i UI:
        st.session_state["_last_read_rows"] = len(df)
        st.session_state["_last_read_sheet"] = ws.title
        return df
    except Exception as e:
        st.error(f"⚠️ Kunde inte läsa databasen: {e}")
        # Sista skydd: returera tom df i korrekt format
        return pd.DataFrame(columns=FINAL_COLS)


def spara_data(df: pd.DataFrame) -> None:
    """
    Skriv tillbaka till Google Sheet.
    - Raderar INTE databasen om df är tom -> avbryter med varning.
    - Skriver rubriker (FINAL_COLS) + rader.
    """
    if df is None:
        st.error("spara_data: Fick None – avbryter skrivning.")
        return

    if df.empty:
        st.warning("spara_data: DataFrame är tom. Avbryter skrivning (databasen rensas inte).")
        return

    # Re-ordna kolumner enligt FINAL_COLS och säkerställ schema
    df2 = ensure_schema(df.copy(), FINAL_COLS)
    df2 = df2[FINAL_COLS].copy()

    # Konvertera till str + ersätt NaN med tom sträng
    body = [FINAL_COLS]
    for _, row in df2.iterrows():
        body.append([("" if (pd.isna(v) or v is None) else str(v)) for v in row.values])

    ws = get_ws(SHEET_NAME)
    # Skriv i ett svep
    with_backoff(ws.clear)
    with_backoff(ws.update, body)
    st.success(f"✅ Sparade {len(df2)} rader till bladet '{ws.title}'.")
