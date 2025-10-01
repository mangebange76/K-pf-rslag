# -*- coding: utf-8 -*-
"""
stockapp.portfolio
------------------
Portfölj-vy som inte klagar om 'Antal aktier' saknas.
Den säkerställer kolumnen lokalt (default 0.0).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from .config import FINAL_COLS
from .utils import ensure_schema, to_float


def _ensure_local_port_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Lägg till kolumner som behövs för portfölj-beräkning."""
    df = ensure_schema(df.copy(), FINAL_COLS)
    if "Antal aktier" not in df.columns:
        df["Antal aktier"] = 0.0
    if "Pris" not in df.columns and "Kurs" in df.columns:
        df["Pris"] = df["Kurs"].apply(to_float)
    if "Pris" not in df.columns:
        df["Pris"] = 0.0
    return df


def visa_portfolj(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📊 Portfölj")

    if df is None or df.empty:
        st.info("Inga bolag i databasen ännu.")
        return

    df = _ensure_local_port_cols(df)

    if df["Antal aktier"].sum() == 0:
        st.warning("Kolumnen **Antal aktier** saknar värden; lägg gärna in antal per ticker i **Lägg till/uppdatera**.")
    # beräkna värde (SEK) om Valuta finns, annars i bas
    if "Valuta" not in df.columns:
        df["Valuta"] = "SEK"

    # växelkurs: för enkelhet – hämtas av annan vy; här antar vi 1.0 om saknas
    usd = float(user_rates.get("USD", 1.0))
    eur = float(user_rates.get("EUR", 1.0))
    cad = float(user_rates.get("CAD", 1.0))
    nok = float(user_rates.get("NOK", 1.0))

    def _rate(ccy: str) -> float:
        c = (ccy or "SEK").upper().strip()
        if c == "USD":
            return usd
        if c == "EUR":
            return eur
        if c == "CAD":
            return cad
        if c == "NOK":
            return nok
        return 1.0

    df["Värde (SEK)"] = df.apply(
        lambda r: to_float(r.get("Antal aktier", 0.0)) * to_float(r.get("Pris", 0.0)) * _rate(r.get("Valuta", "SEK")),
        axis=1,
    )

    total = float(df["Värde (SEK)"].sum())
    if total <= 0:
        st.info("Portföljvärdet är 0 SEK. Lägg in antal/kurs för att se fördelning.")
    else:
        df["Andel (%)"] = np.where(total > 0, df["Värde (SEK)"] / total * 100.0, 0.0)

    show = ["Ticker", "Namn", "Valuta", "Antal aktier", "Pris", "Värde (SEK)", "Andel (%)"]
    show = [c for c in show if c in df.columns]

    st.dataframe(
        df[show].sort_values(by="Värde (SEK)", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(f"Rader i databasen: {len(df)}")
