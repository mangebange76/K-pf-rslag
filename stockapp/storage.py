# -*- coding: utf-8 -*-
"""
Läs/skriv portfölj-/bolagsdata till Google Sheet via sheets.py.
- hamta_data()  -> pd.DataFrame   (med kolumn-synonymer normaliserade)
- spara_data(df) -> None
"""

from __future__ import annotations
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

from .config import SHEET_NAME, FINAL_COLS, MAX_ROWS_WRITE
from .sheets import get_ws, ws_read_df, ws_write_df


# -----------------------------
# Kolumn-synonymer → interna namn
# -----------------------------
# Mappa dina rubriker i bladet till appens standardnamn.
COL_RENAME: Dict[str, str] = {
    # bas
    "Namn": "Bolagsnamn",
    "Bolagsnamn": "Bolagsnamn",
    "Ticker": "Ticker",
    "Valuta": "Valuta",
    "Antal aktier": "Antal du äger",         # bladet → appens interna
    "Antal du äger": "Antal du äger",

    # kurs/pris
    "Aktuell kurs": "Kurs",
    "Pris": "Kurs",
    "Kurs": "Kurs",

    # market cap
    "Market Cap (valuta)": "Market Cap",
    "Market Cap": "Market Cap",
    "Market Cap (SEK)": "Market Cap (SEK)",

    # shares
    "Utestående aktier": "Utestående aktier (milj.)",
    "Utestående aktier (milj.)": "Utestående aktier (milj.)",

    # P/S & kvartal
    "P/S": "P/S",
    "P/S Q1": "P/S Q1",
    "P/S Q2": "P/S Q2",
    "P/S Q3": "P/S Q3",
    "P/S Q4": "P/S Q4",
    "P/S-snitt": "P/S-snitt (Q1..Q4)",
    "PS-snitt": "P/S-snitt (Q1..Q4)",
    "P/S snitt": "P/S-snitt (Q1..Q4)",

    # prognoser (M = i miljoner i bolagets valuta)
    "Omsättning idag": "Omsättning i år (M)",
    "Omsättning i år": "Omsättning i år (M)",
    "Omsättning i år (est.)": "Omsättning i år (M)",
    "Omsättning i år (M)": "Omsättning i år (M)",
    "Omsättning nästa år": "Omsättning nästa år (M)",
    "Omsättning nästa år (est.)": "Omsättning nästa år (M)",
    "Omsättning nästa år (M)": "Omsättning nästa år (M)",

    # margins / lönsamhet
    "Bruttomarginal (%)": "Gross margin (%)",
    "Nettomarginal (%)": "Net margin (%)",
    "Operating margin (%)": "Operating margin (%)",

    # övriga nyckeltal
    "Debt/Equity": "Debt/Equity",
    "EV/EBITDA": "EV/EBITDA (ttm)",
    "EV/EBITDA (ttm)": "EV/EBITDA (ttm)",
    "ROE (%)": "ROE (%)",
    "FCF (M)": "FCF (M)",
    "FCF Yield (%)": "FCF Yield (%)",
    "Dividend Yield (%)": "Dividend yield (%)",
    "Dividend yield (%)": "Dividend yield (%)",
    "Payout Ratio CF (%)": "Dividend payout (FCF) (%)",
    "Dividend payout (FCF) (%)": "Dividend payout (FCF) (%)",
    "Kassa (M)": "Kassa (M)",

    # meta
    "Risklabel": "Risklabel",
    "Sektor": "Sektor",
    "Industri": "Industri",
    "Senast manuellt uppdaterad": "TS Omsättning i år",
    "Senast auto-uppdaterad": "TS Full",
}


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Byter rubriker enligt COL_RENAME och behåller även okända kolumner.
    Lägger till saknade standardkolumner med tomma värden.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=FINAL_COLS)

    # strip:a rubriker
    df = df.rename(columns={c: c.strip() for c in df.columns})

    # mappa synonymer
    renamed = {}
    for c in df.columns:
        renamed[c] = COL_RENAME.get(c, c)
    df = df.rename(columns=renamed)

    # lägg till kolumner som saknas för appen
    for c in FINAL_COLS:
        if c not in df.columns:
            df[c] = pd.NA

    # ordna kolumnordning (men släng inte okända – lägg dem sist)
    known = [c for c in FINAL_COLS if c in df.columns]
    unknown = [c for c in df.columns if c not in FINAL_COLS]
    df = df[known + unknown]

    return df


def hamta_data() -> pd.DataFrame:
    """
    Läser arket. Faller tillbaka till första fliken om SHEET_NAME inte finns.
    Returnerar DataFrame med standardiserade kolumnnamn.
    """
    try:
        ws = get_ws(SHEET_NAME)
        raw = ws_read_df(ws)
        if raw is None:
            raise RuntimeError("Tomt svar från Google Sheet.")
        df = _standardize_columns(raw)
        # filtrera bort rader utan ticker
        if "Ticker" in df.columns:
            df = df[df["Ticker"].astype(str).str.strip() != ""]
        return df.reset_index(drop=True)
    except Exception as e:
        st.error(f"🚫 Kunde inte läsa data från Google Sheet: {e}")
        return pd.DataFrame(columns=FINAL_COLS)


def spara_data(df: pd.DataFrame) -> None:
    """
    Skriver tillbaka till fliken SHEET_NAME. Klipper antal rader vid MAX_ROWS_WRITE.
    Vi skriver bara de kolumner som finns i df i nuvarande ordning.
    """
    if df is None:
        st.warning("Inget att spara.")
        return

    if len(df) > MAX_ROWS_WRITE:
        raise RuntimeError(f"För många rader ({len(df)}) > MAX_ROWS_WRITE={MAX_ROWS_WRITE}.")

    ws = get_ws(SHEET_NAME)
    # skriv exakt det som finns (bevara eventuella extra kolumner användaren lagt till)
    out = df.copy()
    ws_write_df(ws, out)
    st.toast("✅ Sparat till Google Sheet.")
