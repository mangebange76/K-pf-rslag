# -*- coding: utf-8 -*-

# --- TS-fält som ska tidsstämplas när de ändras/auto-uppdateras -------------
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

# --- Full kolumnlista som vi vill ha i databasen ----------------------------
FINAL_COLS = [
    # Grund
    "Ticker", "Bolagsnamn", "Utestående aktier",
    "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
    "Antal aktier", "Valuta", "Årlig utdelning", "Aktuell kurs",
    "CAGR 5 år (%)", "P/S-snitt",

    # Extra nyckeltal
    "Sector","Industry",
    "EV","EBITDA","EV/EBITDA",
    "Market Cap (valuta)","Market Cap (SEK)",
    "Debt/Equity","Gross Margin (%)","Net Margin (%)",
    "Cash & Equivalents","Free Cash Flow","FCF Margin (%)",
    "Monthly Burn","Runway (quarters)",
    "MCAP Q1","MCAP Q2","MCAP Q3","MCAP Q4",
    "GAV SEK",  # användarens GAV i SEK

    # Tidsstämplar & källor
    "Senast manuellt uppdaterad", "Senast auto-uppdaterad", "Senast uppdaterad källa",

    # TS-kolumner
    TS_FIELDS["Utestående aktier"],
    TS_FIELDS["P/S"], TS_FIELDS["P/S Q1"], TS_FIELDS["P/S Q2"], TS_FIELDS["P/S Q3"], TS_FIELDS["P/S Q4"],
    TS_FIELDS["Omsättning idag"], TS_FIELDS["Omsättning nästa år"],
]

# Standard valutakurser (fallback/startvärden)
STANDARD_VALUTAKURSER = {"USD": 10.00, "NOK": 1.0, "CAD": 7.5, "EUR": 11.0, "SEK": 1.0}

# Scoring-vikter
SCORING_WEIGHTS_GROWTH = {
    "valuation": 0.35,   # P/S-gap + riktkursgap
    "growth":    0.30,   # CAGR
    "quality":   0.20,   # marginaler
    "safety":    0.15,   # runway, skuldsättning
}

SCORING_WEIGHTS_DIVIDEND = {
    "yield":     0.45,
    "safety":    0.30,   # payout mot FCF, D/E
    "quality":   0.15,   # FCF-marginal
    "valuation": 0.10,   # ps-gap
}

# Cap-klasser (SEK) – används för risklabel/filter
CAP_BOUNDS_SEK = [
    ("Nano",   0,            1e9),
    ("Micro",  1e9,          3e9),
    ("Small",  3e9,          30e9),
    ("Mid",    30e9,         200e9),
    ("Large",  200e9,        1000e9),
    ("Mega",   1000e9,       9e12),
]
