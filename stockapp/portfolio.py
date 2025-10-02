# -*- coding: utf-8 -*-
"""
stockapp.portfolio
------------------
Portfölj-vy som är robust mot tomma celler:
- Tomma/icke-numeriska "Antal aktier" behandlas som 0.0
- Pris hämtas från Pris/Kurs/Aktuell kurs och konverteras till float
- Valuta antas vara SEK om saknas
- Värde (SEK) och Andel (%) beräknas
"""

from __future__ import annotations

from typing import Dict, List
import numpy as np
import pandas as pd
import streamlit as st

from .config import FINAL_COLS
from .utils import ensure_schema, to_float, format_large_number


# ---- Hjälpare ---------------------------------------------------------------

_QTY_CANDIDATES: List[str] = [
    "Antal aktier",
    "Antal du äger",
    "Antal",
    "Quantity",
]

_PRICE_CANDIDATES: List[str] = [
    "Pris",
    "Kurs",
    "Aktuell kurs",
]

def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Säkerställ minsta schema + normalisera kvantitet/pris.
    - Skapar "Antal aktier" (float) och fyller tomt med 0.0
    - Skapar "Pris" (float) baserat på Kurs/Aktuell kurs vid behov
    - Lägger default 'SEK' i Valuta om saknas
    """
    work = ensure_schema(df.copy(), FINAL_COLS)

    # --- Antal aktier
    qty_col = next((c for c in _QTY_CANDIDATES if c in work.columns), None)
    if qty_col is None:
        work["Antal aktier"] = 0.0
    else:
        # konvertera till float och fyll NA med 0.0
        work["Antal aktier"] = pd.to_numeric(work[qty_col], errors="coerce").fillna(0.0).astype(float)
        if qty_col != "Antal aktier":
            # spegla till standardkolumn för resten av appen
            work["Antal aktier"] = work["Antal aktier"]

    # --- Pris
    price_col = next((c for c in _PRICE_CANDIDATES if c in work.columns), None)
    if price_col is None:
        work["Pris"] = 0.0
    else:
        work["Pris"] = pd.to_numeric(work[price_col], errors="coerce").fillna(0.0).astype(float)
        if price_col != "Pris":
            work["Pris"] = work["Pris"]

    # --- Valuta
    if "Valuta" not in work.columns:
        work["Valuta"] = "SEK"
    work["Valuta"] = work["Valuta"].fillna("SEK").astype(str)

    return work


def _rate_for(ccy: str, user_rates: Dict[str, float]) -> float:
    """
    Hämtar växelkurs (→ SEK) från user_rates. Okända koder = 1.0.
    """
    if not isinstance(user_rates, dict):
        return 1.0
    k = (ccy or "SEK").upper().strip()
    return float(user_rates.get(k, 1.0))


# ---- Publik vy --------------------------------------------------------------

def visa_portfolj(df: pd.DataFrame, user_rates: Dict[str, float]) -> None:
    st.header("💼 Portfölj")

    if df is None or df.empty:
        st.info("Inga bolag i databasen ännu.")
        return

    work = _ensure_cols(df)

    # Beräkna värde i SEK
    work["Värde (SEK)"] = work.apply(
        lambda r: float(r.get("Antal aktier", 0.0)) * float(r.get("Pris", 0.0)) * _rate_for(r.get("Valuta", "SEK"), user_rates),
        axis=1,
    )

    total = float(work["Värde (SEK)"].sum(skipna=True))
    if total > 0:
        work["Andel (%)"] = np.where(total > 0, work["Värde (SEK)"] / total * 100.0, 0.0)
    else:
        work["Andel (%)"] = 0.0

    # Visa summering
    st.markdown(f"**Totalt portföljvärde:** {format_large_number(total, 'SEK')}")

    # Visningskolumner (visa 0 istället för NaN/None)
    show = [
        "Bolagsnamn",
        "Ticker",
        "Valuta",
        "Antal aktier",
        "Pris",
        "Värde (SEK)",
        "Andel (%)",
    ]
    show = [c for c in show if c in work.columns]

    # Format: ersätt NaN/None med 0 för kvantitet/pris
    out = work.copy()
    if "Antal aktier" in out.columns:
        out["Antal aktier"] = pd.to_numeric(out["Antal aktier"], errors="coerce").fillna(0.0)
    if "Pris" in out.columns:
        out["Pris"] = pd.to_numeric(out["Pris"], errors="coerce").fillna(0.0)
    if "Andel (%)" in out.columns:
        out["Andel (%)"] = pd.to_numeric(out["Andel (%)"], errors="coerce").fillna(0.0)

    st.dataframe(
        out[show].sort_values(by="Värde (SEK)", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(f"Rader i databasen: {len(out)}")
