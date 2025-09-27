# -*- coding: utf-8 -*-
from typing import Dict, List

# TS-spårning
TS_FIELDS: Dict[str, str] = {
    "Utestående aktier": "TS_Utestående aktier",
    "P/S": "TS_P/S",
    "P/S Q1": "TS_P/S Q1",
    "P/S Q2": "TS_P/S Q2",
    "P/S Q3": "TS_P/S Q3",
    "P/S Q4": "TS_P/S Q4",
    "Omsättning idag": "TS_Omsättning idag",
    "Omsättning nästa år": "TS_Omsättning nästa år",
}

# Standardkolumner
FINAL_COLS: List[str] = [
    "Ticker","Bolagsnamn","Valuta","Aktuell kurs",
    "Utestående aktier","Antal aktier","GAV SEK",
    "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt",
    "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
    "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
    "Årlig utdelning","CAGR 5 år (%)",
    "Market Cap (valuta)","Market Cap (SEK)","MCAP Q1","MCAP Q2","MCAP Q3","MCAP Q4",
    "Sector","Industry","Risklabel",
    "Debt/Equity","Gross Margin (%)","Net Margin (%)",
    "EV","EBITDA","EV/EBITDA",
    "Cash & Equivalents","Free Cash Flow","FCF Margin (%)","Monthly Burn","Runway (quarters)",
    "Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa",
    TS_FIELDS["Utestående aktier"],
    TS_FIELDS["P/S"], TS_FIELDS["P/S Q1"], TS_FIELDS["P/S Q2"], TS_FIELDS["P/S Q3"], TS_FIELDS["P/S Q4"],
    TS_FIELDS["Omsättning idag"], TS_FIELDS["Omsättning nästa år"],
]

# Valutor (fallback)
STANDARD_VALUTAKURSER = {"USD": 9.75, "NOK": 0.95, "CAD": 7.05, "EUR": 11.18, "SEK": 1.0}

# Risklabel band (i mcap, prisvaluta – visning formatters via utils)
RISK_BANDS = [
    ("Nano", 0, 50e6),
    ("Micro", 50e6, 300e6),
    ("Small", 300e6, 2e9),
    ("Mid", 2e9, 10e9),
    ("Large", 10e9, 200e9),
    ("Mega", 200e9, 1e99),
]

# Scoring-vikter (default – sektorspecifika justeringar i scoring.py)
SCORING_WEIGHTS_GROWTH = {
    "valuation": 0.35,
    "growth":    0.35,
    "quality":   0.20,
    "safety":    0.10,
}
SCORING_WEIGHTS_DIVIDEND = {
    "yield":     0.45,
    "safety":    0.30,
    "quality":   0.15,
    "valuation": 0.10,
}
