# -*- coding: utf-8 -*-
"""
Central konfig för appen.

Här ligger endast KONSTANTER och små hjälpfunktioner som inte
pratar nätverk eller Streamlit-UI.

Andra moduler importerar härifrån:
- Sheets/ark-namn
- Standardvalutor
- Stapel av "finala" kolumner i databladet
- Flaggor för vilka källor som används (Yahoo/FMP/SEC)
"""

from __future__ import annotations
from typing import Dict
import os

# --------------------------------------------------------------------
# Google Sheet
# --------------------------------------------------------------------
# URL till Google Sheet hämtas primärt från Streamlit secrets i körning,
# men vi har en fallback-variabel här så verktyg/test kan importera modulen.
SHEET_URL_FALLBACK: str = os.environ.get("SHEET_URL", "")
SHEET_NAME: str = os.environ.get("SHEET_NAME", "Data")
RATES_SHEET_NAME: str = os.environ.get("RATES_SHEET_NAME", "Valutakurser")

# --------------------------------------------------------------------
# Valutor – default/fallbackvärden (SEK per 1 enhet basvaluta)
# Dessa används om blad/extern hämtning inte går.
# --------------------------------------------------------------------
STANDARD_VALUTAKURSER: Dict[str, float] = {
    "USD": 10.50,
    "EUR": 11.30,
    "CAD": 7.70,
    "NOK": 1.00,
    "SEK": 1.00,  # alltid 1:1 som bas
}

# --------------------------------------------------------------------
# Databladets kolumnordning (”schema”)
# OBS! Håll namn identiska mot det du vill se i Google Sheet.
# Lägg hellre till längst bak än att byta namn på befintliga.
# --------------------------------------------------------------------
FINAL_COLS = [
    # Identitet
    "Ticker",
    "Namn",
    "Sektor",
    "Bransch",

    # Basdata/Marknad
    "Senaste kurs",
    "Market Cap",
    "Utestående aktier (milj.)",

    # Värdering & P/S-historik
    "P/S (Yahoo)",
    "P/S",
    "P/S Q1",
    "P/S Q2",
    "P/S Q3",
    "P/S Q4",

    # Skuld & marginaler
    "Debt/Equity",
    "Net debt / EBITDA",
    "Bruttomarginal (%)",
    "Operating margin (%)",
    "Net margin (%)",
    "ROE (%)",
    "P/B",

    # Kassaflöde/Utdelning
    "FCF (TTM)",
    "FCF Yield (%)",
    "Dividend yield (%)",
    "Dividend payout (FCF) (%)",
    "Kassa",

    # Prognoser (manuella)
    "Omsättning (i år, prognos)",
    "Omsättning (nästa år, prognos)",

    # Portföljfält
    "GAV (SEK)",
    "Antal aktier du äger",

    # Tidsstämplar (ISO8601 strängar)
    "Senast kurs-uppdaterad",
    "Senast uppdaterad",

    # Interna hjälp/etiketter (valfritt)
    "Notering",
]

# --------------------------------------------------------------------
# Källflaggor: slå på/av fetchers (kan även överskridas av st.secrets)
# --------------------------------------------------------------------
USE_FMP: bool = os.environ.get("USE_FMP", "true").lower() not in ("0", "false", "no")
USE_SEC: bool = os.environ.get("USE_SEC", "true").lower() not in ("0", "false", "no")

# --------------------------------------------------------------------
# Risklabel-trösklar (Market Cap i USD)
# används av scoring/visningar för snabb etikett
# --------------------------------------------------------------------
RISKLABEL_BINS_USD = [
    (0,            "Nano"),     # < 50M
    (50e6,         "Micro"),    # 50M – 300M
    (300e6,        "Small"),    # 300M – 2B
    (2e9,          "Mid"),      # 2B – 10B
    (10e9,         "Large"),    # 10B – 200B
    (200e9,        "Mega"),     # > 200B
]

def risk_label_from_mcap(mcap_usd: float | None) -> str:
    """Returnerar etikett baserat på Market Cap (USD)."""
    if mcap_usd is None or mcap_usd <= 0:
        return "Okänd"
    label = "Nano"
    for thr, name in RISKLABEL_BINS_USD:
        if mcap_usd >= thr:
            label = name
        else:
            break
    return label

# --------------------------------------------------------------------
# Sektorvikter (exempel – används av scoring/investeringsförslag)
# värdena är relativa och används som multiplikatorer i poängberäkning
# --------------------------------------------------------------------
SECTOR_WEIGHTS: Dict[str, Dict[str, float]] = {
    # Exempel: Tech prioriterar marginaler/ROE lite högre än P/B
    "Technology": {
        "ps": 1.0, "margin": 1.2, "roe": 1.2, "de": 1.0, "pb": 0.9, "fcf": 1.2,
    },
    "Communication Services": {
        "ps": 1.0, "margin": 1.1, "roe": 1.1, "de": 1.0, "pb": 1.0, "fcf": 1.0,
    },
    "Consumer Discretionary": {
        "ps": 1.0, "margin": 1.0, "roe": 1.0, "de": 1.0, "pb": 1.0, "fcf": 1.0,
    },
    "Consumer Staples": {
        "ps": 0.9, "margin": 1.0, "roe": 1.1, "de": 1.2, "pb": 1.0, "fcf": 1.1,
    },
    "Financials": {
        "ps": 0.8, "margin": 0.9, "roe": 1.3, "de": 1.2, "pb": 1.2, "fcf": 0.9,
    },
    "Health Care": {
        "ps": 1.0, "margin": 1.1, "roe": 1.0, "de": 1.0, "pb": 1.0, "fcf": 1.0,
    },
    "Industrials": {
        "ps": 1.0, "margin": 1.0, "roe": 1.0, "de": 1.0, "pb": 1.0, "fcf": 1.0,
    },
    "Energy": {
        "ps": 0.9, "margin": 1.0, "roe": 1.0, "de": 1.1, "pb": 1.0, "fcf": 1.2,
    },
    "Utilities": {
        "ps": 0.8, "margin": 1.0, "roe": 1.0, "de": 1.3, "pb": 1.0, "fcf": 1.0,
    },
    "Real Estate": {
        "ps": 0.7, "margin": 0.9, "roe": 1.0, "de": 1.3, "pb": 1.1, "fcf": 0.9,
    },
    "Materials": {
        "ps": 0.9, "margin": 1.0, "roe": 1.0, "de": 1.0, "pb": 1.0, "fcf": 1.0,
    },
}

# --------------------------------------------------------------------
# Presentation – hur stora tal formatteras (Market Cap, FCF etc.)
# --------------------------------------------------------------------
MCAP_UNITS = [
    (1_000_000_000_000, " tn"),
    (1_000_000_000,     " md"),
    (1_000_000,         " mn"),
]

def format_money_short(value: float | None, unit_suffix: str = " USD") -> str:
    """Gör 4.35e12 -> '4.35 tn USD' osv."""
    if value is None:
        return "—"
    v = float(value)
    for thr, lab in MCAP_UNITS:
        if v >= thr:
            return f"{v/thr:.2f}{lab}{unit_suffix}"
    return f"{v:.0f}{unit_suffix}"
