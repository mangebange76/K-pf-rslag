# -*- coding: utf-8 -*-
"""
stockapp.config
----------------
Konstanter & namnsättning för hela appen.
Håll denna modul fri från beroenden till andra stockapp-moduler.
"""

from __future__ import annotations
from typing import Dict, List
import streamlit as st

# ---------------------------------------------------------------------
# App-metadata
# ---------------------------------------------------------------------
APP_TITLE: str = st.secrets.get("APP_TITLE", "K-pf-rslag")
DISPLAY_CURRENCY: str = st.secrets.get("DISPLAY_CURRENCY", "SEK")

# ---------------------------------------------------------------------
# Google Sheets
# ---------------------------------------------------------------------
# URL till kalkylbladet (läggs normalt i st.secrets).
SHEET_URL: str = st.secrets.get("SHEET_URL", "").strip()
# Huvudfliken där portfölj/bolagslista finns
SHEET_NAME: str = st.secrets.get("SHEET_NAME", "Data")
# Fliken för valutakurser
RATES_SHEET_NAME: str = st.secrets.get("RATES_SHEET_NAME", "Valutakurser")
# Prefix för snapshots/backup-blad (används av storage)
SNAPSHOT_PREFIX: str = st.secrets.get("SNAPSHOT_PREFIX", "SNAP_")

# ---------------------------------------------------------------------
# Valutor
# ---------------------------------------------------------------------
# Standardvärden om API/blads-läsning faller.
STANDARD_VALUTAKURSER: Dict[str, float] = {
    "USD": float(st.secrets.get("DEFAULT_USDSEK", 10.0)),
    "EUR": float(st.secrets.get("DEFAULT_EURSEK", 11.0)),
    "CAD": float(st.secrets.get("DEFAULT_CADSEK", 7.5)),
    "NOK": float(st.secrets.get("DEFAULT_NOKSEK", 1.0)),
    "SEK": 1.0,
}

# ---------------------------------------------------------------------
# Kolumnschema (namn som resten av appen förväntar sig)
# ---------------------------------------------------------------------
# Basfält som beskriver ett bolag
BASE_COLS: List[str] = [
    "Ticker",
    "Bolagsnamn",
    "Valuta",
    "Sektor",
    "Risklabel",                 # härleder vi från Market Cap när vi kan
    "Antal aktier",
    "GAV (SEK)",
]

# Nyckeltal / fakta som hämtas
FACT_COLS: List[str] = [
    "Kurs",
    "Market Cap",                         # i bolagets valuta
    "Utestående aktier",                  # miljoner (om vi vet), annars antal
    "Utestående aktier (milj.)",          # explicit miljoner om vi kan räkna fram
    "Debt/Equity",
    "Net debt / EBITDA",
    "P/B",
    "Gross margin (%)",
    "Operating margin (%)",
    "Net margin (%)",
    "ROE (%)",
    "FCF Yield (%)",
    "Dividend yield (%)",
    "Dividend payout (FCF) (%)",
    "Kassa (M)",
    "EV/EBITDA (ttm)",
    # P/S-spår
    "P/S",
    "P/S Q1",
    "P/S Q2",
    "P/S Q3",
    "P/S Q4",
    # Prognoser (manuella, i miljoner i bolagets valuta)
    "Omsättning i år (förv.)",
    "Omsättning nästa år (förv.)",
]

# Tidsstämplar – OBS: appen sätter TS-kolumner som "<Fält> TS"
TS_COLS: List[str] = [
    "Kurs TS",
    "Full TS",
    "Omsättning i år (förv.) TS",
    "Omsättning nästa år (förv.) TS",
]

# Beräknade fält (visning/analys)
CALC_COLS: List[str] = [
    "P/S-snitt (Q1..Q4)",
    "P/S (TTM, modell)",
    "Riktkurs (valuta)",
    "Uppsida (%)",
    "TotalScore",
    "Coverage",
    "Recommendation",
    "Värde (SEK)",
    "Andel (%)",
]

# Slutlig ordning (för säker sparning/visning)
FINAL_COLS: List[str] = BASE_COLS + FACT_COLS + TS_COLS + CALC_COLS

# För “Manuell prognoslista” (vilka fält som anses manuella)
MANUAL_PROGNOS_FIELDS: List[str] = [
    "Omsättning i år (förv.)",
    "Omsättning nästa år (förv.)",
]

# För validering av tider – dessa används när vi listar “manuell prognoslista”
TS_FIELDS: List[str] = [f"{f} TS" for f in MANUAL_PROGNOS_FIELDS]

# ---------------------------------------------------------------------
# Limits och defaults
# ---------------------------------------------------------------------
# Maxrad-skydd vid skrivning (guard)
MAX_ROWS_WRITE: int = int(st.secrets.get("MAX_ROWS_WRITE", 4000))

# Hur många förslag vi visar i investeringsvyn (per sida)
PROPOSALS_PAGE_SIZE: int = int(st.secrets.get("PROPOSALS_PAGE_SIZE", 10))

# Batch – standardstorlek
BATCH_DEFAULT_SIZE: int = int(st.secrets.get("BATCH_DEFAULT_SIZE", 20))

# ---------------------------------------------------------------------
# Käll-toggles
# ---------------------------------------------------------------------
USE_YAHOO: bool = bool(st.secrets.get("USE_YAHOO", True))
USE_FMP: bool = bool(st.secrets.get("USE_FMP", True))
USE_SEC: bool = bool(st.secrets.get("USE_SEC", True))

# ---------------------------------------------------------------------
# Scoring – vilka nyckeltal som vägs (vikt per sektor sätts i scoring-modulen)
# ---------------------------------------------------------------------
SCORABLE_KEYS: List[str] = [
    "P/S (TTM, modell)",
    "Net margin (%)",
    "Gross margin (%)",
    "Operating margin (%)",
    "ROE (%)",
    "FCF Yield (%)",
    "Debt/Equity",
    "Net debt / EBITDA",
    "P/B",
    "Dividend yield (%)",
    "Dividend payout (FCF) (%)",
    "Uppsida (%)",
]
