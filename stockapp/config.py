# stockapp/config.py
# -*- coding: utf-8 -*-

"""
Central konfiguration & kolumnschema för appen.

Viktigt:
- SHEET_NAME: huvudbladet i Google Sheet (databasen)
- RATES_SHEET_NAME: bladet där valutakurser sparas
- STANDARD_VALUTAKURSER: start-/fallbackkurser
- TS_FIELDS: vilka fält som har TS_-kolumner
- FINAL_COLS: fullständig kolumnlista som appen förväntar sig
  (vi inkluderar även extra nyckeltal så vyer/score kan köras utan KeyError)
"""

# --- Google Sheets-flikar ----------------------------------------------------
SHEET_NAME = "Blad1"
RATES_SHEET_NAME = "Valutakurser"

# --- Valutakurser (fallback/startvärden) -------------------------------------
STANDARD_VALUTAKURSER = {
    "USD": 9.75,
    "NOK": 0.95,
    "CAD": 7.05,
    "EUR": 11.18,
    "SEK": 1.0,
}

# --- Tidsstämpel-spårning ----------------------------------------------------
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

# --- Fullständig kolumnlista -------------------------------------------------
# Notera:
# - "Utestående aktier" lagras i MILJONER (styck/1e6)
# - Omsättning/FCF/Kassa i MILJONER av bolagets valuta
# - Market cap kan sparas både i bolagsvaluta och SEK om du vill
FINAL_COLS = [
    # Grundidentitet
    "Ticker", "Bolagsnamn", "Valuta",

    # Pris & aktiedata
    "Aktuell kurs", "Utestående aktier",

    # P/S & historik (TTM/Q)
    "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "P/S-snitt",

    # Omsättning (M, bolagsvaluta)
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",

    # Riktkurser (bolagsvaluta)
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",

    # Portfölj
    "Antal aktier", "Årlig utdelning", "GAV (SEK)",

    # Tillväxt & övrigt
    "CAGR 5 år (%)",

    # --- Extra nyckeltal för scoring/analys ---
    # Marknadsvärde
    "Market Cap (valuta)", "Market Cap (SEK)",

    # Lönsamhet/marginaler (%)
    "Bruttomarginal (%)", "Nettomarginal (%)",

    # Kassaflöde / skuld / kassa (alla i M, bolagsvaluta)
    "FCF (M)", "Debt/Equity", "Kassa (M)", "Runway (kvartal)",

    # Multiplar
    "EV/EBITDA",

    # Dividend
    "Dividend Yield (%)", "Payout Ratio CF (%)",

    # Klassning
    "Risklabel", "Sektor", "Industri",

    # Käll- & datumfält
    "Senast manuellt uppdaterad", "Senast auto-uppdaterad", "Senast uppdaterad källa",

    # TS-kolumner (en per spårat fält)
    TS_FIELDS["Utestående aktier"],
    TS_FIELDS["P/S"], TS_FIELDS["P/S Q1"], TS_FIELDS["P/S Q2"], TS_FIELDS["P/S Q3"], TS_FIELDS["P/S Q4"],
    TS_FIELDS["Omsättning idag"], TS_FIELDS["Omsättning nästa år"],
]

# (Valfritt) För appens rubrik/branding
APP_TITLE = "📊 Aktieanalys och investeringsförslag"

# (Valfritt) Risklabel-trösklar (MCAP i bolagsvaluta, ungefärliga nivåer – justera i scoring om du vill)
RISK_BUCKET_LIMITS = {
    "Microcap": 300_000_000,    # < 300M
    "Smallcap": 2_000_000_000,  # < 2B
    "Midcap": 10_000_000_000,   # < 10B
    "Largecap": float("inf"),   # >= 10B
}

# (Valfritt) Standardinställningar för batch
DEFAULT_BATCH_SIZE = 10
DEFAULT_BATCH_SORT = "Äldst TS"  # eller "A–Ö"
