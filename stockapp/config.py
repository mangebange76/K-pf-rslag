# -*- coding: utf-8 -*-
"""
Konfiguration & kolumnschema för appen.

VIKTIGT:
- Denna modul ska INTE importera något från andra moduler i projektet.
- Enbart konstanter och enkla hjälpfunktioner utan beroenden.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Google Sheet / bladnamn
# ---------------------------------------------------------------------------
# SHEET_URL läses från st.secrets i körning, men vi sätter inget här i config.
# Huvudbladets namn:
SHEET_NAME: str = "Blad1"

# Blad för valutakurser:
RATES_SHEET_NAME: str = "Valutakurser"

# ---------------------------------------------------------------------------
# Standardkurser (fallback) till SEK
# ---------------------------------------------------------------------------
STANDARD_VALUTAKURSER = {
    "USD": 9.75,
    "NOK": 0.95,
    "CAD": 7.05,
    "EUR": 11.18,
    "SEK": 1.0,
}

# ---------------------------------------------------------------------------
# Spårade fält → tidsstämpel-kolumner (TS_)
# Dessa TS-kolumner används för att logga när respektive fält ändrades.
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Standardkolumner i databasen (huvudbladet)
# OBS: Det är OK om verklig Sheet har fler kolumner – dessa är "baseline".
# ---------------------------------------------------------------------------
FINAL_COLS = [
    # Grund
    "Ticker", "Bolagsnamn", "Utestående aktier",
    "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
    "Antal aktier", "Valuta", "Årlig utdelning", "Aktuell kurs",
    "CAGR 5 år (%)", "P/S-snitt",

    # Extra nyckeltal / metadata (kan fyllas av fetchers vid behov)
    "Market Cap", "EV/EBITDA", "Debt/Equity", "Gross Margin (%)", "Net Margin (%)",
    "Operating Cash Flow", "CapEx", "Free Cash Flow", "FCF Margin (%)",
    "Cash & Equivalents", "Long-term Debt", "Short-term Debt",
    "Sector", "Industry",
    "Mcap Q1", "Mcap Q2", "Mcap Q3", "Mcap Q4",

    # Portfölj-relaterat
    "GAV (SEK)",  # ditt genomsnittliga anskaffningsvärde i SEK

    # Tidsstämplar & källor
    "Senast manuellt uppdaterad", "Senast auto-uppdaterad", "Senast uppdaterad källa",

    # TS-kolumner (en per spårat fält)
    TS_FIELDS["Utestående aktier"],
    TS_FIELDS["P/S"], TS_FIELDS["P/S Q1"], TS_FIELDS["P/S Q2"], TS_FIELDS["P/S Q3"], TS_FIELDS["P/S Q4"],
    TS_FIELDS["Omsättning idag"], TS_FIELDS["Omsättning nästa år"],
]

# ---------------------------------------------------------------------------
# Batch-inställningar
# ---------------------------------------------------------------------------
BATCH_DEFAULT_SIZE: int = 10  # hur många tickers per körning
BATCH_ORDER_MODES = ("Äldst först", "A–Ö (bolagsnamn)")

# ---------------------------------------------------------------------------
# Risketiketter (market cap baserat)
# ---------------------------------------------------------------------------
def risk_label_from_mcap(mcap: float) -> str:
    """
    Returnerar en snabb risklabel baserat på market cap (i samma valuta som priset).
    Trösklarna är ungefärliga och används bara för UI-etiketter.
    """
    try:
        v = float(mcap or 0.0)
    except Exception:
        v = 0.0
    if v <= 0:
        return "Unknown"
    # nivåer i USD-liknande storleksordning (kan extrapoleras även i andra valutor)
    if v < 300_000_000:   # < $0.3B
        return "Microcap"
    if v < 2_000_000_000: # < $2B
        return "Smallcap"
    if v < 10_000_000_000: # < $10B
        return "Midcap"
    if v < 200_000_000_000: # < $200B
        return "Largecap"
    return "Megacap"

# ---------------------------------------------------------------------------
# Sektor-vikter (default) för poängsättning – kan överskridas i körning
# ---------------------------------------------------------------------------
SECTOR_WEIGHTS_DEFAULT = {
    # Exempelvikter – används som riktvärden om du inte överstyr i appen
    "Technology":   {"growth": 0.40, "profit": 0.25, "balance": 0.20, "valuation": 0.15},
    "Communication Services": {"growth": 0.35, "profit": 0.25, "balance": 0.20, "valuation": 0.20},
    "Consumer Discretionary": {"growth": 0.35, "profit": 0.25, "balance": 0.20, "valuation": 0.20},
    "Industrials":  {"growth": 0.30, "profit": 0.30, "balance": 0.20, "valuation": 0.20},
    "Health Care":  {"growth": 0.30, "profit": 0.25, "balance": 0.25, "valuation": 0.20},
    "Financials":   {"growth": 0.20, "profit": 0.35, "balance": 0.25, "valuation": 0.20},
    "Energy":       {"growth": 0.20, "profit": 0.30, "balance": 0.30, "valuation": 0.20},
    "Utilities":    {"growth": 0.15, "profit": 0.35, "balance": 0.30, "valuation": 0.20},
    "Real Estate":  {"growth": 0.20, "profit": 0.30, "balance": 0.30, "valuation": 0.20},
    "Materials":    {"growth": 0.25, "profit": 0.30, "balance": 0.25, "valuation": 0.20},
    "Consumer Staples": {"growth": 0.20, "profit": 0.35, "balance": 0.25, "valuation": 0.20},
    # fallback
    "_default":     {"growth": 0.30, "profit": 0.30, "balance": 0.20, "valuation": 0.20},
}

# ---------------------------------------------------------------------------
# Hjälp: lista med nyckeltal för investeringsförslag (visningsordning)
# ---------------------------------------------------------------------------
INVEST_KEY_METRICS_ORDER = [
    "Market Cap", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "P/S-snitt",
    "EV/EBITDA",
    "Debt/Equity",
    "Gross Margin (%)", "Net Margin (%)",
    "Operating Cash Flow", "CapEx", "Free Cash Flow", "FCF Margin (%)",
    "Cash & Equivalents", "Long-term Debt", "Short-term Debt",
    "Mcap Q1", "Mcap Q2", "Mcap Q3", "Mcap Q4",
    "CAGR 5 år (%)",
]

# ---------------------------------------------------------------------------
# Små hjälpare (utan beroenden)
# ---------------------------------------------------------------------------
def normalize_colname(name: str) -> str:
    """Normaliserar kolumnnamn till ett förutsägbart format (enkelt)."""
    return str(name).strip()
