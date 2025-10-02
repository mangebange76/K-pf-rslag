# -*- coding: utf-8 -*-
"""
Läs/skriv portfölj-/bolagsdata till Google Sheet via sheets.py.

- hamta_data()  -> pd.DataFrame   (med kolumn-synonymer normaliserade)
- spara_data(df) -> None

Extra robusthet:
- Normaliserar rubriker (tar bort NBSP osv) innan mappning.
- Upptäcker tickerkolumn även om den heter 'Symbol' eller har konstiga mellanslag.
- Städar cellvärden (inkl. NBSP) innan filtrering av tomma tickers.
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import re

import pandas as pd
import streamlit as st

from .config import SHEET_NAME, FINAL_COLS, MAX_ROWS_WRITE
from .sheets import get_ws, ws_read_df, ws_write_df


# ---------------------------------------------------------------------
# Normalisering & hjälpare
# ---------------------------------------------------------------------
_WS_CHARS = ("\u00A0", "\u2007", "\u202F")  # NBSP-varianter

def _norm_header(name: str) -> str:
    """Normalisera rubriknamn: ersätt NBSP, trimma, komprimera whitespace, case-bevara."""
    s = str(name)
    for ch in _WS_CHARS:
        s = s.replace(ch, " ")
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _clean_str_cell(x) -> str:
    """String-städning i cellvärden (för t.ex. tickers)."""
    s = "" if x is None else str(x)
    for ch in _WS_CHARS:
        s = s.replace(ch, " ")
    return s.strip()

def _ci(s: str) -> str:
    """Case-insensitive nyckel (för jämförelser)."""
    return _norm_header(s).lower()


# -----------------------------
# Kolumn-synonymer → interna namn (på normaliserad nyckel)
# -----------------------------
COL_RENAME_RAW: Dict[str, str] = {
    # bas
    "Namn": "Bolagsnamn",
    "Bolagsnamn": "Bolagsnamn",
    "Ticker": "Ticker",
    "Symbol": "Ticker",
    "Valuta": "Valuta",
    "Antal aktier": "Antal du äger",
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
# normaliserad variant (nyckel = _ci(k))
COL_RENAME_NORM: Dict[str, str] = {_ci(k): v for k, v in COL_RENAME_RAW.items()}


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Byter rubriker enligt COL_RENAME_NORM och behåller även okända kolumner.
    Lägger till saknade standardkolumner med tomma värden.
    Upptäcker tickerkolumn även om den inte exakt heter 'Ticker'.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=FINAL_COLS)

    # 1) normalisera rubriker
    norm_cols = [_norm_header(c) for c in df.columns]
    df.columns = norm_cols

    # 2) mappa synonymer (normaliserat)
    renamed: Dict[str, str] = {}
    for c in df.columns:
        renamed[c] = COL_RENAME_NORM.get(_ci(c), c)
    df = df.rename(columns=renamed)

    # 3) om 'Ticker' saknas: försök hitta kandidat (t.ex. 'Symbol')
    if "Ticker" not in df.columns:
        # plocka första kolumn vars normaliserade namn är 'symbol'/'ticker'
        for c in list(df.columns):
            n = _ci(c)
            if n in ("ticker", "symbol"):
                df = df.rename(columns={c: "Ticker"})
                break

    # 4) städa strängvärden i Ticker (viktigt p.g.a. NBSP)
    if "Ticker" in df.columns:
        df["Ticker"] = df["Ticker"].apply(_clean_str_cell)

    # 5) lägg till kolumner som saknas för appen
    for c in FINAL_COLS:
        if c not in df.columns:
            df[c] = pd.NA

    # 6) ordna kolumnordning (okända sist)
    known = [c for c in FINAL_COLS if c in df.columns]
    unknown = [c for c in df.columns if c not in FINAL_COLS]
    df = df[known + unknown]

    return df


def hamta_data() -> pd.DataFrame:
    """
    Läser arket. Faller tillbaka till första fliken om SHEET_NAME inte finns.
    Returnerar DataFrame med standardiserade kolumnnamn.
    Filtrerar ENBART uppenbart tomma rader; tar hänsyn till NBSP i Ticker.
    """
    try:
        ws = get_ws(SHEET_NAME)
        raw = ws_read_df(ws)
        if raw is None:
            raise RuntimeError("Tomt svar från Google Sheet.")
        df = _standardize_columns(raw)

        # filtrera bort rader där Ticker saknas helt (efter städning)
        if "Ticker" in df.columns:
            t = df["Ticker"].apply(_clean_str_cell)
            df = df[ t != "" ]
        else:
            st.warning("⚠️ Ingen 'Ticker'-kolumn hittades – visar raderna orörda.")
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
    out = df.copy()
    ws_write_df(ws, out)
    st.toast("✅ Sparat till Google Sheet.")
