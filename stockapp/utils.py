# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# Hjälpare för robust DataFrame-hantering
# ------------------------------------------------------------

# 1) Lista av "alias" -> kanoniskt namn.
#    Poängen: du har redan *din* rubrikstandard i arket. Vi mappar *till dina namn*.
#    Lägg gärna till fler alias efterhand (vänster = variant; höger = din rubrik).
COLUMN_ALIASES: Dict[str, str] = {
    # identitet/metadata
    "ticker": "Ticker",
    "symbol": "Ticker",
    "bolagsnamn": "Bolagsnamn",
    "company": "Bolagsnamn",
    "company name": "Bolagsnamn",
    "valuta": "Valuta",
    "currency": "Valuta",

    # priser/kap
    "kurs": "Aktuell kurs",
    "pris": "Aktuell kurs",
    "last": "Aktuell kurs",
    "market cap": "Market Cap (SEK)",
    "market cap (nu)": "Market Cap (SEK)",
    "marketcap (sek)": "Market Cap (SEK)",
    "market cap (sek)": "Market Cap (SEK)",
    "market cap (currency)": "Market Cap (valuta)",
    "market cap (valuta)": "Market Cap (valuta)",

    # shares
    "shares outstanding": "Utestående aktier",
    "utestående aktier": "Utestående aktier",

    # P/S
    "p/s": "P/S",
    "ps": "P/S",
    "p/s q1": "P/S Q1",
    "p/s q2": "P/S Q2",
    "p/s q3": "P/S Q3",
    "p/s q4": "P/S Q4",
    "p/s-snitt": "P/S-snitt",
    "p/s 4q-snitt": "P/S-snitt",
    "p/s-snitt (4 kvartal)": "P/S-snitt",

    # revenue / omsättning
    "omsättning idag": "Omsättning idag",
    "omsättning i år": "Omsättning idag",
    "revenue ttm": "Omsättning idag",
    "revenue this year": "Omsättning idag",
    "omsättning nästa år": "Omsättning nästa år",
    "omsättning om 2 år": "Omsättning om 2 år",
    "omsättning om 3 år": "Omsättning om 3 år",

    # riktkurser
    "riktkurs idag": "Riktkurs idag",
    "riktkurs om 1 år": "Riktkurs om 1 år",
    "riktkurs om 2 år": "Riktkurs om 2 år",
    "riktkurs om 3 år": "Riktkurs om 3 år",

    # portfölj
    "antal aktier": "Antal aktier",
    "ägda aktier": "Antal aktier",
    "gav (sek)": "GAV (SEK)",
    "gav": "GAV (SEK)",

    # utdelningsdata
    "årlig utdelning": "Årlig utdelning",
    "dividend yield (%)": "Dividend Yield (%)",
    "payout ratio cf (%)": "Payout Ratio CF (%)",

    # övriga nyckeltal
    "cagr 5 år (%)": "CAGR 5 år (%)",
    "bruttomarginal (%)": "Bruttomarginal (%)",
    "nettomarginal (%)": "Nettomarginal (%)",
    "fcf (m)": "FCF (M)",
    "kassa (m)": "Kassa (M)",
    "runway (kvartal)": "Runway (kvartal)",
    "debt/equity": "Debt/Equity",
    "ev/ebitda": "EV/EBITDA",

    # klassning
    "risklabel": "Risklabel",
    "risk label": "Risklabel",
    "sektor": "Sektor",
    "sector": "Sektor",
    "industri": "Industri",
    "industry": "Industri",

    # tidsstämplar
    "senast manuellt uppdaterad": "Senast manuellt uppdaterad",
    "senast auto-uppdaterad": "Senast auto-uppdaterad",
    "senast uppdaterad källa": "Senast uppdaterad källa",

    # TS-fält
    "ts_utestående aktier": "TS_Utestående aktier",
    "ts_p/s": "TS_P/S",
    "ts_p/s q1": "TS_P/S Q1",
    "ts_p/s q2": "TS_P/S Q2",
    "ts_p/s q3": "TS_P/S Q3",
    "ts_p/s q4": "TS_P/S Q4",
    "ts_omsättning idag": "TS_Omsättning idag",
    "ts_omsättning nästa år": "TS_Omsättning nästa år",
}

# Kolumner som med hög sannolikhet ska vara numeriska (konverteras robust)
LIKELY_NUMERIC: Tuple[str, ...] = (
    "Aktuell kurs",
    "Utestående aktier",
    "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "P/S-snitt",
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
    "Antal aktier",
    "Årlig utdelning",
    "GAV (SEK)",
    "CAGR 5 år (%)",
    "Market Cap (valuta)",
    "Market Cap (SEK)",
    "Bruttomarginal (%)",
    "Nettomarginal (%)",
    "FCF (M)",
    "Debt/Equity",
    "Kassa (M)",
    "Runway (kvartal)",
    "EV/EBITDA",
    "Dividend Yield (%)",
    "Payout Ratio CF (%)",
)

def _clean_header(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()

def canonicalize_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Trim/casefold på rubriker
    - Mappa alias -> dina kanoniska namn (enligt COLUMN_ALIASES)
    - Skapa 'säkra' standardkolumner om de saknas (med vettiga defaultar)
    - Konvertera troliga numeriska kolumner robust (komma -> punkt; tusentals-sep)
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["Ticker", "Bolagsnamn", "Valuta"])

    # 1) bygg nytt columns-lexikon
    mapping: Dict[str, str] = {}
    for c in df.columns:
        key = _clean_header(c)
        mapping[c] = COLUMN_ALIASES.get(key, COLUMN_ALIASES.get(key.strip(), None))
        if mapping[c] is None:
            # Ingen alias-träff: behåll originalnamnet
            mapping[c] = str(c).strip()

    df = df.rename(columns=mapping)

    # 2) se till att grundkolumner finns
    base_defaults = {
        "Ticker": "",
        "Bolagsnamn": "",
        "Valuta": "USD",
        "Antal aktier": 0.0,
    }
    for k, v in base_defaults.items():
        if k not in df.columns:
            df[k] = v

    # 3) trim stringkolumner
    for col in ("Ticker", "Bolagsnamn", "Valuta", "Risklabel", "Sektor", "Industri",
                "Senast manuellt uppdaterad", "Senast auto-uppdaterad", "Senast uppdaterad källa"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # 4) robust numerik-konvertering
    for col in LIKELY_NUMERIC:
        if col in df.columns:
            # ersätt tusentals-separatorer och komma-decimal
            s = (
                df[col]
                .astype(str)
                .str.replace(r"\s", "", regex=True)
                .str.replace(",", ".", regex=False)
                .str.replace(" ", "", regex=False)  # smal NBSP
            )
            df[col] = pd.to_numeric(s, errors="coerce")

    # 5) normalisera ticker & valuta
    df["Ticker"] = df["Ticker"].astype(str).str.upper()
    df["Valuta"] = df["Valuta"].astype(str).str.upper()

    # 6) deduplicera på Ticker (första företräde)
    if "Ticker" in df.columns:
        df = df[~df["Ticker"].duplicated(keep="first")]

    return df.reset_index(drop=True)

def pick_col(df: pd.DataFrame, candidates: Sequence[str], default: Optional[str] = None) -> Optional[str]:
    """Returnera första kolumnen som finns i df av 'candidates' (namnsträng); annars default."""
    for c in candidates:
        if c in df.columns:
            return c
    return default

def format_large_number(x: Union[float, int, None], curr: str = "") -> str:
    """Formatera stort tal med tn/mdr/milj – utan att kasta på NaN."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    n = float(x)
    sign = "-" if n < 0 else ""
    n = abs(n)
    if n >= 1e12:
        s = f"{n/1e12:.2f} tn"
    elif n >= 1e9:
        s = f"{n/1e9:.2f} mdr"
    elif n >= 1e6:
        s = f"{n/1e6:.2f} milj"
    else:
        s = f"{n:.0f}"
    return f"{sign}{s} {curr}".strip()

def debug_df_overview(df: pd.DataFrame, title: str = "Datakoll"):
    """Liten diagnosruta i UI så vi ser vad som faktiskt finns."""
    with st.expander(f"🛠 {title}", expanded=False):
        st.write(f"Rader: **{len(df)}**")
        st.write("Kolumner:", list(df.columns))
        if len(df) > 0:
            st.dataframe(df.head(10), use_container_width=True)
