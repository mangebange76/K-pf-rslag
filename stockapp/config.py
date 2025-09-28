# -*- coding: utf-8 -*-
from __future__ import annotations

# Huvudbladets namn i Google Sheet
SHEET_NAME = "Blad1"

# Vilka fält som får en TS_-kolumn som stämplas vid ändring
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

# Fält där en manuell förändring ska sätta "Senast manuellt uppdaterad" + respektive TS_
MANUELL_FALT_FOR_DATUM = [
    "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
    "Omsättning idag","Omsättning nästa år"
]

# Slutlig kolumnlista som vi vill hålla i Google Sheet
# (Ordningen är mest för UI; saknade kolumner fylls ut i storage/utils.)
FINAL_COLS = [
    # Bas
    "Ticker", "Bolagsnamn", "Valuta",
    "Aktuell kurs", "Årlig utdelning", "CAGR 5 år (%)",
    "Utestående aktier",

    # P/S och historik
    "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "P/S-snitt",

    # Omsättning (miljoner, bolagets valuta) – *manuell policy* för idag/nästa år
    "Omsättning idag", "Omsättning nästa år",
    "Omsättning om 2 år", "Omsättning om 3 år",

    # Riktkurser (beräknade i bolagets valuta)
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",

    # Portfölj
    "Antal aktier", "GAV (SEK)",

    # Extra profil
    "Sektor", "Industri", "Risklabel", "Värderingslabel",

    # Market cap historik (i basvaluta, enligt Yahoo)
    "Mcap Q1", "Mcap Q2", "Mcap Q3", "Mcap Q4",

    # Tidsstämplar & källor
    "Senast manuellt uppdaterad", "Senast auto-uppdaterad", "Senast uppdaterad källa",

    # TS-kolumner (genereras via TS_FIELDS)
    TS_FIELDS["Utestående aktier"],
    TS_FIELDS["P/S"], TS_FIELDS["P/S Q1"], TS_FIELDS["P/S Q2"], TS_FIELDS["P/S Q3"], TS_FIELDS["P/S Q4"],
    TS_FIELDS["Omsättning idag"], TS_FIELDS["Omsättning nästa år"],
]
