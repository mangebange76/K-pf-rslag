# -*- coding: utf-8 -*-
"""
stockapp.config
----------------
Konstanter & kolumnschema som hela appen använder.
Håll denna modul fri från beroenden till andra stockapp-moduler.
"""

from __future__ import annotations
from typing import Dict, List
import streamlit as st

# ---------------------------------------------------------------------
# Google Sheets
# ---------------------------------------------------------------------
# URL till kalkylbladet (läggs normalt i st.secrets)
SHEET_URL: str = st.secrets.get("SHEET_URL", "").strip()
# Huvudfliken där portföljen/bolagslistan finns
SHEET_NAME: str = st.secrets.get("SHEET_NAME", "Data")
# Fliken för valutakurser
RATES_SHEET_NAME: str = st.secrets.get("RATES_SHEET_NAME", "Valutakurser")

# App-titel
APP_TITLE: str = st.secrets.get("APP_TITLE", "K-pf-rslag")

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

DISPLAY_CURRENCY: str = "SEK"

# ---------------------------------------------------------------------
# Kolumnschema (harmoniserat med app.py & vyer)
# ---------------------------------------------------------------------
BASE_COLS: List[str] = [
    "Ticker",
    "Bolagsnamn",
    "Valuta",
    "Sektor",
    "Industri",
    "Risklabel",
    "Antal aktier",
    "GAV (SEK)",
]

FACT_COLS: List[str] = [
    # pris & värde
    "Kurs",                    # primärt fält i appen
    "Aktuell kurs",            # accepterad fallback som vi kan läsa/visa
    "Market Cap",
    "Market Cap (SEK)",
    "Utestående aktier (milj.)",

    # P/S-historik
    "P/S",
    "P/S Q1",
    "P/S Q2",
    "P/S Q3",
    "P/S Q4",

    # Marginaler och lönsamhet
    "Bruttomarginal (%)",
    "Rörelsemarginal (%)",
    "Nettomarginal (%)",
    "ROE (%)",

    # Kapitalstruktur & värdering
    "Debt/Equity",
    "Net debt / EBITDA",
    "EV/EBITDA",
    "P/B",

    # Kassaflöde & utdelning
    "FCF (M)",
    "FCF Yield (%)",
    "Dividend Yield (%)",
    "Payout Ratio CF (%)",

    # Likviditet
    "Kassa (M)",
    "Runway (kvartal)",

    # Prognoser (alltid manuella i din process)
    "Omsättning i år (est.)",
    "Omsättning nästa år (est.)",
]

# Tidsstämplar – OBS: appen använder suffix " TS"
TS_COLS: List[str] = [
    "Kurs TS",
    "Full TS",
    "Omsättning i år (est.) TS",
    "Omsättning nästa år (est.) TS",
    "P/S TS",
    "P/S Q1 TS",
    "P/S Q2 TS",
    "P/S Q3 TS",
    "P/S Q4 TS",
]

# Beräknade fält (visning/analys)
CALC_COLS: List[str] = [
    "P/S-snitt (Q1..Q4)",
    "Uppsida (%)",
    "Riktkurs (valuta)",
    "TotalScore",
    "Coverage",
    "Recommendation",
    "Värde (SEK)",
    "Andel (%)",
]

# Slutlig ordning
FINAL_COLS: List[str] = BASE_COLS + FACT_COLS + TS_COLS + CALC_COLS

# Dessa används när vi listar “Manuell prognoslista”
MANUAL_PROGNOS_FIELDS: List[str] = ["Omsättning i år (est.)", "Omsättning nästa år (est.)"]

# Maxrad-skydd vid skrivning (guard)
MAX_ROWS_WRITE: int = int(st.secrets.get("MAX_ROWS_WRITE", 4000))

# Hur många förslag per sida i investeringsvyn (kan ändras i UI)
PROPOSALS_PAGE_SIZE: int = int(st.secrets.get("PROPOSALS_PAGE_SIZE", 5))

# Batch – standardstorlek
BATCH_DEFAULT_SIZE: int = int(st.secrets.get("BATCH_DEFAULT_SIZE", 10))

# Togglar för källor (kan styras via secrets)
USE_YAHOO: bool = bool(st.secrets.get("USE_YAHOO", True))
USE_FMP: bool = bool(st.secrets.get("USE_FMP", True))
USE_SEC: bool = bool(st.secrets.get("USE_SEC", True))

# Nyckeltal som ofta vägs i scoring (info – själva vikterna ligger i scoring-modulen)
SCORABLE_KEYS: List[str] = [
    "Uppsida (%)",
    "EV/EBITDA",
    "Net debt / EBITDA",
    "Bruttomarginal (%)",
    "Rörelsemarginal (%)",
    "Nettomarginal (%)",
    "ROE (%)",
    "P/B",
    "FCF Yield (%)",
    "Debt/Equity",
    "Dividend Yield (%)",
    "Payout Ratio CF (%)",
    "P/S",
]
