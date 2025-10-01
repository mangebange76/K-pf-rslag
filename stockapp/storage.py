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

from .config import FINAL_COLS, SHEET_NAME
from .sheets import get_ws, safe_get_all_values
from .utils import ensure_schema, with_backoff, dedupe_tickers


def _normalize_header(row: List[str]) -> List[str]:
    return [(c or "").strip() for c in row]


def _to_dataframe(values: List[List[str]]) -> pd.DataFrame:
    if not values:
        return pd.DataFrame(columns=FINAL_COLS)

    # ta bort helt tomma rader
    rows = [r for r in values if any((c or "").strip() for c in r)]
    if not rows:
        return pd.DataFrame(columns=FINAL_COLS)

    header = _normalize_header(rows[0])
    body = rows[1:]

    width = len(header)
    fixed = []
    for r in body:
        rr = list(r)
        if len(rr) < width:
            rr += [""] * (width - len(rr))
        elif len(rr) > width:
            rr = rr[:width]
        fixed.append(rr)

    try:
        df = pd.DataFrame(fixed, columns=header)
    except Exception:
        df = pd.DataFrame(columns=FINAL_COLS)

    # Säkerställ schema + normalisering (lägger till t.ex. "Antal aktier")
    df = ensure_schema(df, FINAL_COLS)

    # Släng helt tomma (efter normalisering)
    if not df.empty:
        mask = df.apply(lambda s: any(str(v).strip() for v in s.values), axis=1)
        df = df.loc[mask].copy()

    # Ta bort rader utan ticker
    if "Ticker" in df.columns:
        df = df[df["Ticker"].astype(str).str.strip() != ""].copy()

    # Deduplikat
    df = dedupe_tickers(df)

    return df.reset_index(drop=True)


def hamta_data() -> pd.DataFrame:
    try:
        ws = get_ws(SHEET_NAME)
        vals = safe_get_all_values(ws)
        df = _to_dataframe(vals)
        st.session_state["_last_read_rows"] = len(df)
        st.session_state["_last_read_sheet"] = ws.title
        return df
    except Exception as e:
        st.error(f"⚠️ Kunde inte läsa databasen: {e}")
        return pd.DataFrame(columns=FINAL_COLS)


def spara_data(df: pd.DataFrame) -> None:
    if df is None:
        st.error("spara_data: Fick None – avbryter.")
        return
    if df.empty:
        st.warning("spara_data: DF är tom – databasen rensas inte.")
        return

    df2 = ensure_schema(df.copy(), FINAL_COLS)
    df2 = df2[FINAL_COLS].copy()

    body = [FINAL_COLS]
    for _, row in df2.iterrows():
        body.append([("" if pd.isna(v) else str(v)) for v in row.values])

    ws = get_ws(SHEET_NAME)
    with_backoff(ws.clear)
    with_backoff(ws.update, body)
    st.success(f"✅ Sparade {len(df2)} rader till '{ws.title}'.")
