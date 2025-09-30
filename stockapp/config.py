# stockapp/config.py
# -*- coding: utf-8 -*-
"""
Central konfiguration & kolumnschema för appen.
OBS: Denna modul är fristående (importerar inget annat) för att undvika cirkulära imports.
"""

# --- Google Sheets ---
# Dessa hämtas normalt via st.secrets i andra moduler.
# Lämna namnen nedan i fred (Blad1 / Valutakurser) om du inte bytt fliknamn i Google Sheet.
SHEET_NAME = "Blad1"
RATES_SHEET_NAME = "Valutakurser"

# --- Standardkurser (fallback) ---
STANDARD_VALUTAKURSER = {
    "USD": 9.75,
    "NOK": 0.95,
    "CAD": 7.05,
    "EUR": 11.18,
    "SEK": 1.0,
}

# --- Fält med separata tidsstämpelkolumner (TS_) när de ändras/uppdateras ---
TS_FIELDS = {
    "Utestående aktier": "TS_Utestående aktier",
    "P/S": "TS_P/S",
    "P/S Q1": "TS_P/S Q1",
    "P/S Q2": "TS_P/S Q2",
    "P/S Q3": "TS_P/S Q3",
    "P/S Q4": "TS_P/S Q4",
    "Omsättning idag": "TS_Omsättning idag",
    "Omsättning nästa år": "TS_Omsättning nästa år",
}

# --- Bas- & beräkningsfält som används på många ställen ---
BASE_COLS = [
    "Ticker", "Bolagsnamn", "Valuta",
    "Aktuell kurs", "Årlig utdelning",
    "Antal aktier", "GAV SEK",
    "CAGR 5 år (%)",
]

PS_COLS = ["P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "P/S-snitt"]

REVENUE_COLS = [
    "Omsättning idag",            # innevarande/aktuellt år (manuell)
    "Omsättning nästa år",        # nästa år (manuell)
    "Omsättning om 2 år",
    "Omsättning om 3 år",
]

TARGET_COLS = [
    "Riktkurs idag",
    "Riktkurs om 1 år",
    "Riktkurs om 2 år",
    "Riktkurs om 3 år",
]

STAMP_COLS = [
    "Senast manuellt uppdaterad",
    "Senast auto-uppdaterad",
    "Senast uppdaterad källa",
]

TS_COLS = list(TS_FIELDS.values())

# --- Market cap & nyckeltal för bred analys/scoring ---
VALUATION_COLS = [
    "Market Cap (nu)",
    "Market Cap Q1", "Market Cap Q2", "Market Cap Q3", "Market Cap Q4",
    "EV", "EBITDA (TTM)", "EV/EBITDA",
]

PROFITABILITY_COLS = [
    "Bruttomarginal (%)",
    "Nettomarginal (%)",
]

BALANCE_CASHFLOW_COLS = [
    "Skulder (Debt)",
    "Eget kapital (Equity)",
    "Debt/Equity",
    "Kassa & kortfristiga placeringar",
    "CFO (TTM)",          # Cash Flow from Operations
    "CapEx (TTM)",
    "FCF (TTM)",
    "FCF-täckning (kvartal)",   # hur många kvartal kassa+FCF räcker
]

META_COLS = [
    "Sektor", "Industri",
    "_RiskLabel",              # Micro/Small/Mid/Large
    "_Score",                  # total poäng
    "_Score_Detalj",           # text/JSON-lik summering
    "_Score_Nyckeltal_Täckning" # hur många nyckeltal som fanns
]

# --- Slutlig ordning i arket (kan utökas utan att bryta något) ---
FINAL_COLS = (
    BASE_COLS
    + PS_COLS
    + REVENUE_COLS
    + TARGET_COLS
    + VALUATION_COLS
    + PROFITABILITY_COLS
    + BALANCE_CASHFLOW_COLS
    + META_COLS
    + STAMP_COLS
    + TS_COLS
)

# Fält som bör antas numeriska (för schema-säkring)
NUMERIC_DEFAULT_ZERO = set(
    [
        "Aktuell kurs", "Årlig utdelning", "Antal aktier", "GAV SEK", "CAGR 5 år (%)",
        "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "P/S-snitt",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
        "Market Cap (nu)", "Market Cap Q1", "Market Cap Q2", "Market Cap Q3", "Market Cap Q4",
        "EV", "EBITDA (TTM)", "EV/EBITDA",
        "Bruttomarginal (%)", "Nettomarginal (%)",
        "Skulder (Debt)", "Eget kapital (Equity)", "Debt/Equity",
        "Kassa & kortfristiga placeringar",
        "CFO (TTM)", "CapEx (TTM)", "FCF (TTM)",
        "FCF-täckning (kvartal)",
        "_Score",
        "_Score_Nyckeltal_Täckning",
    ]
)

# Fält som antas strängar
STRING_DEFAULT_EMPTY = set(
    [
        "Ticker", "Bolagsnamn", "Valuta",
        "Senast manuellt uppdaterad", "Senast auto-uppdaterad", "Senast uppdaterad källa",
        "Sektor", "Industri", "_RiskLabel", "_Score_Detalj",
    ] + TS_COLS
)

# Hjälp-konstanter för batch-sidor etc.
BATCH_DEFAULT_SIZE = 10
BATCH_SORT_MODES = ("Äldst uppdaterade först", "A–Ö (bolagsnamn)")

# Fält som du (användaren) alltid uppdaterar manuellt (ska inte autoskrivas över)
MANUAL_ONLY_FIELDS = ["Omsättning idag", "Omsättning nästa år"]

# För att visa tydlig label i UI när ett fält TS-satts
TS_BADGE_AUTO = "Auto"
TS_BADGE_MANUAL = "Manuellt"

# Minimal sanity-check (körs vid import)
assert "Ticker" in FINAL_COLS, "FINAL_COLS måste innehålla 'Ticker'."
assert "Bolagsnamn" in FINAL_COLS, "FINAL_COLS måste innehålla 'Bolagsnamn'."
assert "Aktuell kurs" in FINAL_COLS, "FINAL_COLS måste innehålla 'Aktuell kurs'."
assert "Omsättning idag" in FINAL_COLS and "Omsättning nästa år" in FINAL_COLS, "FINAL_COLS måste innehålla manuella omsättningsfält."
