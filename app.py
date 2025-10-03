# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pandas as pd
import streamlit as st

# ----------------- Robust imports -----------------
from stockapp.config import FINAL_COLS, PROPOSALS_PAGE_SIZE
from stockapp.storage import hamta_data
from stockapp.utils import ensure_schema

# Valutor
from stockapp.rates import (
    las_sparade_valutakurser,
    spara_valutakurser,
    hamta_valutakurser_auto,
)

# Vyer (om de finns)
try:
    from stockapp.invest import visa_investeringsforslag
except Exception:
    visa_investeringsforslag = None  # type: ignore

try:
    from stockapp.portfolio import visa_portfolj
except Exception:
    visa_portfolj = None  # type: ignore

try:
    from stockapp.editor import lagg_till_eller_uppdatera
except Exception:
    lagg_till_eller_uppdatera = None  # type: ignore

# Ny manuell-insamlingsvy
from stockapp.manual_collect import manual_collect_view

# ----------------- State init -----------------
def _init_state():
    st.session_state.setdefault("_df_ref", pd.DataFrame(columns=FINAL_COLS))
    # valutakurser seed
    if "_rates_seeded" not in st.session_state:
        saved = las_sparade_valutakurser()
        for k in ("USD", "EUR", "CAD", "NOK", "SEK"):
            st.session_state[f"rate_{k}"] = float(saved.get(k, 1.0))
        st.session_state["_rates_seeded"] = True
    st.session_state.setdefault("page_size", PROPOSALS_PAGE_SIZE)

def _load_df() -> pd.DataFrame:
    try:
        df = hamta_data()
        df = ensure_schema(df, FINAL_COLS)
        return df
    except Exception as e:
        st.error(f"Kunde inte läsa Google Sheet: {e}")
        return pd.DataFrame(columns=FINAL_COLS)

# ----------------- Sidebar: valutakurser -----------------
def _sidebar_rates() -> Dict[str, float]:
    with st.sidebar.expander("💱 Valutakurser (→ SEK)", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Hämta automatiskt"):
                rates, misses, provider = hamta_valutakurser_auto()
                for k, v in rates.items():
                    st.session_state[f"rate_{k}"] = float(v)
                spara_valutakurser(rates)
                if misses:
                    st.warning("Kunde inte hämta: " + ", ".join(misses))
                st.toast(f"Valutor via {provider}.")
                st.rerun()

        usd = st.number_input("USD", key="rate_USD", value=float(st.session_state["rate_USD"]), step=0.01)
        eur = st.number_input("EUR", key="rate_EUR", value=float(st.session_state["rate_EUR"]), step=0.01)
        cad = st.number_input("CAD", key="rate_CAD", value=float(st.session_state["rate_CAD"]), step=0.01)
        nok = st.number_input("NOK", key="rate_NOK", value=float(st.session_state["rate_NOK"]), step=0.01)
        sek = st.number_input("SEK", key="rate_SEK", value=float(st.session_state["rate_SEK"]), step=0.01)

        rates = {"USD": usd, "EUR": eur, "CAD": cad, "NOK": nok, "SEK": sek}
        if st.button("Spara kurser"):
            spara_valutakurser(rates)
            st.toast("Valutakurser sparade.")
    return rates

# ----------------- Main -----------------
def main():
    st.set_page_config(page_title="K-pf-rslag", layout="wide")
    st.title("K-pf-rslag")

    _init_state()
    st.session_state["_df_ref"] = _load_df()

    user_rates = _sidebar_rates()

    view = st.sidebar.radio(
        "Välj vy",
        ["Manuell insamling (4 knappar)", "Investeringsförslag", "Lägg till / uppdatera", "Portfölj"],
        index=0,
    )

    if view == "Manuell insamling (4 knappar)":
        df2 = manual_collect_view(st.session_state["_df_ref"])
        if df2 is not st.session_state["_df_ref"]:
            st.session_state["_df_ref"] = df2

    elif view == "Investeringsförslag":
        if visa_investeringsforslag is None:
            st.info("Investeringsförslag-modulen saknas i denna miljö.")
        else:
            visa_investeringsforslag(st.session_state["_df_ref"], user_rates)

    elif view == "Lägg till / uppdatera":
        if lagg_till_eller_uppdatera is None:
            st.info("Editor-modulen saknas i denna miljö.")
        else:
            df3 = lagg_till_eller_uppdatera(st.session_state["_df_ref"], user_rates)
            if df3 is not st.session_state["_df_ref"]:
                st.session_state["_df_ref"] = df3

    else:  # Portfölj
        if visa_portfolj is None:
            st.info("Portfölj-modulen saknas i denna miljö.")
        else:
            visa_portfolj(st.session_state["_df_ref"], user_rates)


if __name__ == "__main__":
    main()
