# stockapp/config.py
# -*- coding: utf-8 -*-
"""
Konfig & kolumn-definitioner för hela appen.
Den här modulen har inga beroenden (ingen Streamlit-import).
"""

# ---------------------------------------------------------------------------
# Google Sheet / ark-namn
# ---------------------------------------------------------------------------
DATA_SHEET_NAME: str = "Data"          # huvudarket med alla bolag
RATES_SHEET_NAME: str = "Valutakurser"  # separat ark för sparade växelkurser
SNAPSHOT_PREFIX: str = "snapshot_"      # snapshots döps t.ex. snapshot_2025-09-28

# ---------------------------------------------------------------------------
# Standardvalutakurser (fallback om inga sparade/auto-hämtade finns)
# Alla kurser uttrycks som 1 BASVALUTA = X SEK
# ---------------------------------------------------------------------------
STANDARD_VALUTAKURSER = {
    "SEK": 1.0,
    "USD": 10.0,
    "EUR": 11.0,
    "NOK": 1.0,
    "CAD": 7.5,
}

# ---------------------------------------------------------------------------
# Visnings-/formateringsregler
# ---------------------------------------------------------------------------
MARKETCAP_LABELS = [
    (1_000_000_000_000_000, "kvadriljoner"),
    (1_000_000_000_000, "biljoner"),   # 10^12
    (1_000_000_000, "miljarder"),      # 10^9
    (1_000_000, "miljoner"),           # 10^6
]

# Risklabel baserat på market cap (USD-ekvivalent)
RISK_BUCKETS = [
    (2_000_000_000_000, "Megacap"),
    (200_000_000_000, "Largecap"),
    (10_000_000_000, "Midcap"),
    (2_000_000_000, "Smallcap"),
    (0, "Microcap"),
]

# Default storlek vid batch-körningar
BATCH_DEFAULT_SIZE = 10

# ---------------------------------------------------------------------------
# Kolumner i huvudarket (Data)
# OBS: håll listan stabil – andra moduler förlitar sig på dessa namn.
# ---------------------------------------------------------------------------
FINAL_COLS = [
    # Basinfo
    "Bolagsnamn", "Ticker", "Valuta", "Land", "Lista", "Sektor", "Industri",

    # Pris & aktier
    "Senast", "Senast (TS)",
    "Utest. aktier", "Utest. aktier (TS)",

    # Market cap nu + TTM
    "Market Cap (nu)", "Market Cap (TS)",
    "Omsättning TTM", "Omsättning TTM (TS)",

    # P/S
    "P/S (Yahoo)", "P/S (TTM)",
    "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "P/S-snitt (Q1..Q4)",

    # Mcap-historik (4 TTM-fönster)
    "MCap Q1", "MCap Q2", "MCap Q3", "MCap Q4",
    "MCap-datum Q1", "MCap-datum Q2", "MCap-datum Q3", "MCap-datum Q4",

    # Lönsamhet & balans
    "Bruttomarginal (%)", "Nettomarginal (%)", "Debt/Equity",
    "Kassa (valuta)", "Kassa (TS)",
    "FCF (TTM)", "FCF (TS)",
    "CapEx (TTM)", "Opex (TTM)", "Opex (TS)",
    "Eget kapital", "Eget kapital (TS)",
    "EPS TTM", "PE (TTM)",

    # Prognoser (alltid manuella – i bolagets valuta)
    "Prognos omsättning i år (valuta)", "Prognos omsättning i år (TS)",
    "Prognos omsättning nästa år (valuta)", "Prognos omsättning nästa år (TS)",

    # Portföljrelaterat
    "Antal du äger", "GAV (SEK)", "Andel portfölj (%)",

    # Metafält
    "Senast uppdaterad (auto)", "Senast uppdaterad (manuell)",
    "Notis",
]

# Alla tidsstämpels-kolumner (identifieras på suffixet "(TS)")
TS_FIELDS = [c for c in FINAL_COLS if c.endswith("(TS)")]

# Kolumner som räknas som "kvartalsrader" i P/S/MCap-tabellen i UI
PS_HISTORY_FIELDS = ["P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"]
MCAP_HISTORY_FIELDS = ["MCap Q1", "MCap Q2", "MCap Q3", "MCap Q4"]
MCAP_DATE_FIELDS = ["MCap-datum Q1", "MCap-datum Q2", "MCap-datum Q3", "MCap-datum Q4"]

# Kolumner vi tillåter att uppdatera i "Kurs endast"-flödet
COURSE_ONLY_UPDATABLE = ["Senast", "Senast (TS)", "Market Cap (nu)", "Market Cap (TS)"]

# Kolumner som ALDRIG skrivs över automatiskt (manuellt fält)
MANUAL_ONLY_FIELDS = [
    "Prognos omsättning i år (valuta)",
    "Prognos omsättning i år (TS)",
    "Prognos omsättning nästa år (valuta)",
    "Prognos omsättning nästa år (TS)",
    "Antal du äger",
    "GAV (SEK)",
    "Notis",
]

# Kolumner som bör finnas även om hämtning misslyckas (skydd mot KeyError)
REQUIRED_MIN_COLS = ["Bolagsnamn", "Ticker", "Valuta", "Senast", "Utest. aktier"]

# Defaultvärden om cell saknas
DEFAULTS = {
    "Bolagsnamn": "",
    "Valuta": "USD",
    "Senast": 0.0,
    "Utest. aktier": 0.0,
    "Market Cap (nu)": 0.0,
    "P/S (TTM)": 0.0,
    "P/S (Yahoo)": 0.0,
    "GAV (SEK)": 0.0,
    "Antal du äger": 0.0,
}
