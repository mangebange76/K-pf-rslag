# -*- coding: utf-8 -*-
"""
Läs/skriv portfölj-/bolagsdata till Google Sheet via sheets.py.

- hamta_data()  -> pd.DataFrame   (med kolumn-synonymer normaliserade)
- spara_data(df) -> None

Robusthet:
- Normaliserar rubriker (tar bort NBSP osv) innan mappning.
- Upptäcker tickerkolumn även om den heter 'Symbol' eller har konstiga mellanslag.
- Städar inte bort rader här – vi lämnar filtrering till vyerna.
"""

from __future__ import annotations
from typing import Dict
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
    s = str(name)
    for ch in _WS_CHARS:
        s = s.replace(ch, " ")
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _ci(s: str) -> str:
    return _norm_header(s).lower()


# Kolumn-synonymer (key = normaliserat namn) → internt namn
_COL_RENAME_RAW: Dict[str, str] = {
    # bas
    "ticker": "Ticker",
    "symbol": "Ticker",
    "bolagsnamn": "Bolagsnamn",
    "namn": "Bolagsnamn",
    "valuta": "Valuta",
    "sektor": "Sektor",
    "risklabel": "Risklabel",
    "antal aktier": "Antal du äger",
    "antal du äger": "Antal du äger",

    # pris/kurs
    "aktuell kurs": "Kurs",
    "pris": "Kurs",
    "kurs": "Kurs",

    # market cap
    "market cap (valuta)": "Market Cap",
    "market cap": "Market Cap",
    "market cap (sek)": "Market Cap (SEK)",

    # shares
    "utestående aktier": "Utestående aktier (milj.)",
    "utestående aktier (milj.)": "Utestående aktier (milj.)",

    # P/S
    "p/s": "P/S",
    "p/s q1": "P/S Q1",
    "p/s q2": "P/S Q2",
    "p/s q3": "P/S Q3",
    "p/s q4": "P/S Q4",
    "p/s-snitt": "P/S-snitt (Q1..Q4)",
    "ps-snitt": "P/S-snitt (Q1..Q4)",
    "p/s snitt": "P/S-snitt (Q1..Q4)",

    # prognoser
    "omsättning idag": "Omsättning i år (M)",
    "omsättning i år": "Omsättning i år (M)",
    "omsättning i år (est.)": "Omsättning i år (M)",
    "omsättning i år (m)": "Omsättning i år (M)",
    "omsättning nästa år": "Omsättning nästa år (M)",
    "omsättning nästa år (est.)": "Omsättning nästa år (M)",
    "omsättning nästa år (m)": "Omsättning nästa år (M)",

    # margins & nyckeltal
    "bruttomarginal (%)": "Gross margin (%)",
    "nettormarginal (%)": "Net margin (%)",
    "nettomarginal (%)": "Net margin (%)",
    "operating margin (%)": "Operating margin (%)",
    "debt/equity": "Debt/Equity",
    "ev/ebitda": "EV/EBITDA (ttm)",
    "ev/ebitda (ttm)": "EV/EBITDA (ttm)",
    "roe (%)": "ROE (%)",
    "fcf (m)": "FCF (M)",
    "fcf yield (%)": "FCF Yield (%)",
    "dividend yield (%)": "Dividend yield (%)",
    "payout ratio cf (%)": "Dividend payout (FCF) (%)",
    "dividend payout (fcf) (%)": "Dividend payout (FCF) (%)",
    "kassa (m)": "Kassa (M)",

    # meta
    "industri": "Industri",
    "senast manuellt uppdaterad": "TS Omsättning i år",
    "senast auto-uppdaterad": "TS Full",
}
_COL_RENAME_NORM = {k: v for k, v in _COL_RENAME_RAW.items()}


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=FINAL_COLS)

    # 1) normalisera rubriker
    df.columns = [_norm_header(c) for c in df.columns]

    # 2) mappa synonymer case-insensitivt
    ren = {}
    for c in df.columns:
        key = _ci(c)
        ren[c] = _COL_RENAME_NORM.get(key, c)
    df = df.rename(columns=ren)

    # 3) om 'Ticker' saknas men 'Symbol' fanns, se till att den blev Ticker (täckning finns ovan)
    if "Ticker" not in df.columns:
        for c in list(df.columns):
            if _ci(c) in ("ticker", "symbol"):
                df = df.rename(columns={c: "Ticker"})
                break

    # 4) lägg till saknade FINAL_COLS
    for c in FINAL_COLS:
        if c not in df.columns:
            df[c] = pd.NA

    # 5) ordna kolumner: kända först
    known = [c for c in FINAL_COLS if c in df.columns]
    unknown = [c for c in df.columns if c not in FINAL_COLS]
    df = df[known + unknown]
    return df


# ---------------------------------------------------------------------
# Publika API
# ---------------------------------------------------------------------
def hamta_data() -> pd.DataFrame:
    """
    Läser Worksheet (SHEET_NAME eller första fliken) och returnerar DataFrame
    med **standardiserade** kolumnnamn. Vi filtrerar inte bort några rader här.
    """
    try:
        ws = get_ws(SHEET_NAME)
        raw = ws_read_df(ws)
        df = _standardize_columns(raw)
        return df.reset_index(drop=True)
    except Exception as e:
        st.error(f"🚫 Kunde inte läsa data från Google Sheet: {e}")
        return pd.DataFrame(columns=FINAL_COLS)


def spara_data(df: pd.DataFrame) -> None:
    """
    Skriver tillbaka DataFrame i dess nuvarande kolumnordning. Klipper antal rader
    vid MAX_ROWS_WRITE.
    """
    if df is None:
        st.warning("Inget att spara.")
        return

    if len(df) > MAX_ROWS_WRITE:
        raise RuntimeError(
            f"För många rader ({len(df)}) > MAX_ROWS_WRITE={MAX_ROWS_WRITE}."
        )

    ws = get_ws(SHEET_NAME)
    ws_write_df(ws, df.fillna(""))
    st.toast("✅ Sparat till Google Sheet.")
