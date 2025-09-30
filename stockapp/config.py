# -*- coding: utf-8 -*-
"""
stockapp/config.py
Centrala konstanter och kolumnschema för appen.

OBS:
- SHEET_URL tas från st.secrets["SHEET_URL"] i körning. Här anger vi bara bladnamn.
- Uppdatera FINAL_COLS om du lägger till nya fält i databasen.
"""

# ------------------------- Google Sheets -------------------------
SHEET_NAME = "Blad1"
RATES_SHEET_NAME = "Valutakurser"

# ------------------------- SEC / API defaults --------------------
# Kan överskridas i st.secrets["SEC_USER_AGENT"] om du vill.
SEC_USER_AGENT_DEFAULT = "StockApp/1.0 (contact: your-email@example.com)"

# ------------------------- Valutor -------------------------------
STANDARD_VALUTAKURSER = {
    "USD": 9.75,
    "NOK": 0.95,
    "CAD": 7.05,
    "EUR": 11.18,
    "SEK": 1.0,
}

# ------------------------- TS-fält (per spårat fält) ------------
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

# ------------------------- Databasschema -------------------------
# Dessa kolumner förväntas finnas i Google Sheet:en (Blad1).
FINAL_COLS = [
    # Grund
    "Ticker", "Bolagsnamn", "Valuta",
    "Aktuell kurs", "Market Cap", "Utestående aktier",

    # P/S (nu + historik och snitt)
    "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "P/S-snitt",

    # Omsättning (manuell prognos)
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",

    # Riktkurser (beräknade)
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",

    # Portfölj
    "Antal aktier", "GAV (SEK)", "Årlig utdelning",

    # Meta
    "Sektor", "Bransch", "CAGR 5 år (%)",

    # Tidsstämplar & källa
    "Senast manuellt uppdaterad", "Senast auto-uppdaterad", "Senast uppdaterad källa",

    # TS-kolumner för spårade fält
    TS_FIELDS["Utestående aktier"],
    TS_FIELDS["P/S"], TS_FIELDS["P/S Q1"], TS_FIELDS["P/S Q2"], TS_FIELDS["P/S Q3"], TS_FIELDS["P/S Q4"],
    TS_FIELDS["Omsättning idag"], TS_FIELDS["Omsättning nästa år"],
]

# Valfria historik/diagnostik-kolumner (skadas inte om de saknas i arket).
OPTIONAL_COLS = [
    # Historiskt mcap (om du vill spara kvartalsmcap separat)
    "Mcap Q1", "Mcap Q2", "Mcap Q3", "Mcap Q4",
    # Score/klassificering (om du vill spara senaste beräkning)
    "Score (senast)", "Värdering (klass)",
]

# ------------------------- Batch / UI -----------------------------
BATCH_SIZE_DEFAULT = 10
BATCH_ORDER_MODES = ["Äldst först (TS)", "A–Ö (bolagsnamn)", "A–Ö (ticker)"]

# ------------------------- Risklabel thresholds -------------------
# Enkla gränser i USD. Justeras vid behov / med FX om du vill.
RISKLABEL_THRESHOLDS = {
    "MICRO_MAX": 3e8,      # < $300M
    "SMALL_MAX": 2e9,      # < $2B
    "MID_MAX":   1e10,     # < $10B
    # Large: >= MID_MAX
}

# ------------------------- Scoring defaults -----------------------
# Basvikter kan justeras per sektor i scoring-modulen.
SCORING_DEFAULT_WEIGHTS = {
    "ps_now": 0.25,
    "ps_avg4": 0.15,
    "growth_proxy": 0.20,         # t.ex. CAGR eller prognostillväxt
    "margins_quality": 0.15,      # bruttomarginal/nettomarginal när de finns
    "balance_quality": 0.15,      # debt/equity, cash runway etc när de finns
    "dividend_quality": 0.10,     # yield/payout/coverage när de finns
}

# ------------------------- Utdelningsfilter -----------------------
DIVIDEND_MIN_YIELD = 0.02   # 2% baseline
DIVIDEND_MAX_PAYOUT_CFO = 0.8  # payout mot operativt kassaflöde
DIVIDEND_MAX_PAYOUT_FCF = 0.9  # payout mot fritt kassaflöde

# ------------------------- Övrigt ----------------------------------
# Kolumner som tolkas som numeriska i utils.ensure_schema()
NUMERIC_COLS = [
    "Aktuell kurs", "Market Cap", "Utestående aktier",
    "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "P/S-snitt",
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
    "Antal aktier", "GAV (SEK)", "Årlig utdelning",
    "CAGR 5 år (%)",
    # Valfria:
    "Mcap Q1", "Mcap Q2", "Mcap Q3", "Mcap Q4",
    "Score (senast)",
]

# Kolumner som är strängar
STRING_COLS = [
    "Ticker", "Bolagsnamn", "Valuta", "Sektor", "Bransch",
    "Senast manuellt uppdaterad", "Senast auto-uppdaterad", "Senast uppdaterad källa",
    "Värdering (klass)",
] + list(TS_FIELDS.values())
