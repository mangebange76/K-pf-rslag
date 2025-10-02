# -*- coding: utf-8 -*-
"""
stockapp.portfolio
------------------
Portfölj-vy som inte klagar om 'Antal aktier' saknas.
Säkerställer kolumner lokalt och räknar värde i SEK baserat på user_rates.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import streamlit as st

from .config import FINAL_COLS
from .utils import ensure_schema, to_float, format_large_number
from .rates import hamta_valutakurs


def _ensure_local_port_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Lägg till kolumner som behövs för portfölj-beräkning."""
    df = ensure_schema(df.copy(), FINAL_COLS)
    if "Antal aktier" not in df.columns:
        df["Antal aktier"] = 0.0
    # Prisfält: använd "Kurs" primärt, annars "Aktuell kurs"
    if "Pris" not in df.columns:
        if "Kurs" in df.columns:
            df["Pris"] = df["Kurs"].apply(to_float)
        elif "Aktuell kurs" in df.columns:
            df["Pris"] = df["Aktuell kurs"].apply(to_float)
        else:
            df["Pris"] = 0.0
    if "Valuta" not in df.columns:
        df["Valuta"] = "SEK"
    return df


def visa_portfolj(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("💼 Portfölj")

    if df is None or df.empty:
        st.info("Inga bolag i databasen ännu.")
        return

    df = _ensure_local_port_cols(df)

    # Beräkna värde (SEK)
    def _row_value_sek(r: pd.Series) -> float:
        pris = to_float(r.get("Pris", 0.0))
        qty = to_float(r.get("Antal aktier", 0.0))
        ccy = str(r.get("Valuta", "SEK")).upper().strip()
        rate = hamta_valutakurs(ccy, user_rates)
        if math.isnan(pris) or qty <= 0:
            return 0.0
        return float(pris) * float(qty) * float(rate)

    df["Värde (SEK)"] = df.apply(_row_value_sek, axis=1)
    total = float(df["Värde (SEK)"].sum())

    st.markdown("**Totalt portföljvärde:** " + format_large_number(total, "SEK"))

    if total > 0:
        df["Andel (%)"] = np.where(total > 0, df["Värde (SEK)"] / total * 100.0, 0.0)

    show = ["Bolagsnamn", "Ticker", "Valuta", "Antal aktier", "Pris", "Värde (SEK)", "Andel (%)"]
    show = [c for c in show if c in df.columns]

    st.dataframe(
        df[show].sort_values(by="Värde (SEK)", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(f"Rader i databasen: {len(df)}")
