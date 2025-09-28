# -*- coding: utf-8 -*-
from __future__ import annotations

# === Google Sheets ===
SHEET_NAME = "Blad1"
RATES_SHEET_NAME = "Valutakurser"

# === Valutor: startvärden (fallback) ===
STANDARD_VALUTAKURSER = {
    "USD": 9.75,
    "NOK": 0.95,
    "CAD": 7.05,
    "EUR": 11.18,
    "SEK": 1.0,
}

# === Spårade fält → tidsstämpelkolumn (TS_) ===
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

# Fält som (vid manuell ändring) sätter "Senast manuellt uppdaterad" + respektive TS_
MANUELL_FALT_FOR_DATUM = [
    "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
    "Omsättning idag","Omsättning nästa år",
]

# === Risklabel-bucketar (kan användas av scoring/labels) ===
# gränser i USD (kan omräknas innan label sätts)
RISK_BUCKETS = [
    (0,          "Nano/Micro"),   # < 300M
    (300e6,      "Small"),        # 300M–2B
    (2e9,        "Mid"),          # 2B–10B
    (10e9,       "Large"),        # 10B–200B
    (200e9,      "Mega"),         # > 200B
]

# === Slutlig kolumnlista i databasen ===
FINAL_COLS = [
    # Bas
    "Ticker", "Bolagsnamn", "Valuta",
    "Aktuell kurs", "Årlig utdelning", "CAGR 5 år (%)",
    "Utestående aktier",

    # P/S och historik
    "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "P/S-snitt",

    # Omsättning (miljoner, bolagets valuta) – *manuellt ansvar: idag/nästa år*
    "Omsättning idag", "Omsättning nästa år",
    "Omsättning om 2 år", "Omsättning om 3 år",

    # Riktkurser (i bolagets valuta)
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",

    # Portfölj
    "Antal aktier", "GAV (SEK)",

    # Profil & etiketter
    "Sektor", "Industri", "Risklabel", "Värderingslabel",

    # Market cap historik (i prisvalutan)
    "Mcap Q1", "Mcap Q2", "Mcap Q3", "Mcap Q4",

    # Tidsstämplar & källa
    "Senast manuellt uppdaterad", "Senast auto-uppdaterad", "Senast uppdaterad källa",

    # TS-kolumner (en per spårat fält)
    TS_FIELDS["Utestående aktier"],
    TS_FIELDS["P/S"], TS_FIELDS["P/S Q1"], TS_FIELDS["P/S Q2"], TS_FIELDS["P/S Q3"], TS_FIELDS["P/S Q4"],
    TS_FIELDS["Omsättning idag"], TS_FIELDS["Omsättning nästa år"],
]
