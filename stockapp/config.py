# stockapp/config.py
# -*- coding: utf-8 -*-
"""
Bas-konfiguration och schema-hjälp för hela appen.

Innehåll:
- Tidszon & tidsstämplar (Stockholm)
- STANDARD_VALUTAKURSER
- TS_FIELDS (vilka fält som spåras med TS_)
- FINAL_COLS (fulla kolumnschemat som appen använder)
- Hjälpfunktioner: säkerställ_kolumner, konvertera_typer,
  _stamp_ts_for_field, _note_auto_update, _note_manual_update,
  add_oldest_ts_col
- MANUELL_FALT_FOR_DATUM (vilka fält som triggar manuell TS)
"""

from typing import Optional, Dict, Any
from datetime import datetime

import pandas as pd
import numpy as np

# --- Lokal Stockholm-tid om pytz finns (annars systemtid) ---
try:
    import pytz
    TZ_STHLM = pytz.timezone("Europe/Stockholm")
    def now_stamp() -> str:
        return datetime.now(TZ_STHLM).strftime("%Y-%m-%d")
    def now_dt() -> datetime:
        return datetime.now(TZ_STHLM)
except Exception:  # pytz saknas
    def now_stamp() -> str:
        return datetime.now().strftime("%Y-%m-%d")
    def now_dt() -> datetime:
        return datetime.now()

# --- Standard valutakurser till SEK (fallback/startvärden) ---
STANDARD_VALUTAKURSER: Dict[str, float] = {
    "USD": 9.75,
    "NOK": 0.95,
    "CAD": 7.05,
    "EUR": 11.18,
    "SEK": 1.0,
}

# --- Spårade fält → respektive TS-kolumn ---
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

# --- Fullt kolumnschema som appen förväntar sig ---
# Obs: lägg hellre till än ta bort – så vi inte tappar existerande data.
FINAL_COLS = [
    # Grund
    "Ticker", "Bolagsnamn", "Valuta", "Aktuell kurs",
    "Sektor", "Industri",

    # Kärn-nyckeltal
    "Utestående aktier",               # milj (styck / 1e6)
    "Market Cap (nu)",
    "Market Cap Q1", "Market Cap Q2", "Market Cap Q3", "Market Cap Q4",

    "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "P/S-snitt",

    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",

    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",

    # Utdelning & portfölj
    "Årlig utdelning",
    "Antal aktier",
    "GAV (SEK)",                      # användarens snittpris i SEK

    # Kvalitativa / finansiella nyckeltal
    "CAGR 5 år (%)",
    "Debt/Equity",
    "Bruttomarginal (%)",
    "Nettomarginal (%)",
    "FCF (TTM)",
    "Kassa",
    "Total skuld",
    "Direktavkastning (%)",           # beräknas i calc men kolumnen kan ligga här

    # Poäng & etiketter
    "GrowthScore",
    "DividendScore",
    "Värdering",

    # Tidsstämplar & källa
    "Senast manuellt uppdaterad",
    "Senast auto-uppdaterad",
    "Senast uppdaterad källa",

    # TS-kolumner (en per spårat fält)
    TS_FIELDS["Utestående aktier"],
    TS_FIELDS["P/S"], TS_FIELDS["P/S Q1"], TS_FIELDS["P/S Q2"], TS_FIELDS["P/S Q3"], TS_FIELDS["P/S Q4"],
    TS_FIELDS["Omsättning idag"], TS_FIELDS["Omsättning nästa år"],
]

# Fält som triggar "Senast manuellt uppdaterad" i formuläret
MANUELL_FALT_FOR_DATUM = [
    "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "Omsättning idag", "Omsättning nästa år"
]

# --- Schemahjälp --------------------------------------------------------------

def säkerställ_kolumner(df: pd.DataFrame) -> pd.DataFrame:
    """
    Skapa saknade kolumner och sätt rimliga defaultvärden.
    Nollställer aldrig existerande värden – endast tillägg av saknat.
    """
    if df is None or df.empty:
        # Skapa tom DF med alla kolumner
        df = pd.DataFrame({c: [] for c in FINAL_COLS})

    for kol in FINAL_COLS:
        if kol not in df.columns:
            # Heuristik för defaults
            low = kol.lower()
            if any(x in low for x in [
                "kurs", "omsättning", "p/s", "utdelning", "cagr",
                "antal", "riktkurs", "aktier", "snitt", "market cap",
                "debt", "marginal", "fcf", "kassa", "skuld", "yield", "score",
            ]):
                df[kol] = 0.0
            elif kol.startswith("TS_"):
                df[kol] = ""   # tidsstämplar som strängar YYYY-MM-DD
            elif kol in ("Senast manuellt uppdaterad", "Senast auto-uppdaterad", "Senast uppdaterad källa"):
                df[kol] = ""
            else:
                # textfält/övrigt
                df[kol] = ""

    # Ta bort eventuella dubbletter
    df = df.loc[:, ~df.columns.duplicated()].copy()
    return df

def konvertera_typer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Konvertera numeriska kolumner → float, datumsträngar → str osv.
    Varken droppar rader eller sätter NaN → vi fyller 0.0 på numeriska.
    """
    if df is None or df.empty:
        return df

    num_cols = [
        "Aktuell kurs",
        "Utestående aktier",
        "Market Cap (nu)", "Market Cap Q1", "Market Cap Q2", "Market Cap Q3", "Market Cap Q4",
        "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "P/S-snitt",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
        "Årlig utdelning",
        "Antal aktier",
        "GAV (SEK)",
        "CAGR 5 år (%)",
        "Debt/Equity",
        "Bruttomarginal (%)",
        "Nettomarginal (%)",
        "FCF (TTM)",
        "Kassa",
        "Total skuld",
        "Direktavkastning (%)",
        "GrowthScore",
        "DividendScore",
    ]

    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    str_cols = [
        "Ticker", "Bolagsnamn", "Valuta", "Sektor", "Industri",
        "Senast manuellt uppdaterad", "Senast auto-uppdaterad", "Senast uppdaterad källa",
        "Värdering"
    ]
    for c in str_cols:
        if c in df.columns:
            df[c] = df[c].astype(str)

    for c in df.columns:
        if str(c).startswith("TS_"):
            df[c] = df[c].astype(str)

    return df

# --- Tidsstämpelshjälpare ----------------------------------------------------

def _stamp_ts_for_field(df: pd.DataFrame, row_idx: int, field: str, when: Optional[str] = None):
    """
    Sätt TS-kolumn för ett spårat fält om det finns.
    """
    ts_col = TS_FIELDS.get(field)
    if not ts_col or ts_col not in df.columns:
        return
    date_str = when if when else now_stamp()
    try:
        df.at[row_idx, ts_col] = date_str
    except Exception:
        pass

def _note_auto_update(df: pd.DataFrame, row_idx: int, source: str):
    """
    Sätt auto-uppdaterad-tidsstämpel och källa.
    """
    try:
        if "Senast auto-uppdaterad" in df.columns:
            df.at[row_idx, "Senast auto-uppdaterad"] = now_stamp()
        if "Senast uppdaterad källa" in df.columns:
            df.at[row_idx, "Senast uppdaterad källa"] = source
    except Exception:
        pass

def _note_manual_update(df: pd.DataFrame, row_idx: int):
    """
    Sätt manuell uppdateringstid (anropas i formulär-flödet).
    """
    try:
        if "Senast manuellt uppdaterad" in df.columns:
            df.at[row_idx, "Senast manuellt uppdaterad"] = now_stamp()
    except Exception:
        pass

# --- Hjälp för “äldsta TS” ---------------------------------------------------

def oldest_any_ts(row: pd.Series) -> Optional[pd.Timestamp]:
    """
    Returnera äldsta (minsta) tidsstämpeln bland alla TS_-kolumner för en rad.
    None om inga tidsstämplar.
    """
    dates = []
    for c in row.index:
        if str(c).startswith("TS_"):
            val = str(row.get(c, "")).strip()
            if not val:
                continue
            d = pd.to_datetime(val, errors="coerce")
            if pd.notna(d):
                dates.append(d)
    return min(dates) if dates else None

def add_oldest_ts_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    Beräknar kolumnen `_oldest_any_ts` + `_oldest_any_ts_fill` för sortering/filtrering.
    """
    if df is None or df.empty:
        return df
    df = df.copy()
    df["_oldest_any_ts"] = df.apply(oldest_any_ts, axis=1)
    df["_oldest_any_ts"] = pd.to_datetime(df["_oldest_any_ts"], errors="coerce")
    df["_oldest_any_ts_fill"] = df["_oldest_any_ts"].fillna(pd.Timestamp("2099-12-31"))
    return df
