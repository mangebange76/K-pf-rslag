# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# --- Våra moduler ----------------------------------------------------
from stockapp.config import (
    FINAL_COLS,
    TS_FIELDS,
    STANDARD_VALUTAKURSER,
    PROPOSALS_PAGE_SIZE,
    BATCH_DEFAULT_SIZE,
)
from stockapp.utils import (
    ensure_schema,
    now_stamp,
    stamp_fields_ts,
    add_oldest_ts_col,
    format_large_number,
)
from stockapp.storage import hamta_data, spara_data
from stockapp.rates import (
    las_sparade_valutakurser,
    spara_valutakurser,
    hamta_valutakurser_auto,
    hamta_valutakurs,
)
from stockapp.fetchers.orchestrator import (
    run_update_full,
    run_update_price_only,
)
from stockapp.batch import sidebar_batch_controls
from stockapp.invest import visa_investeringsforslag
from stockapp.editor import lagg_till_eller_uppdatera
from stockapp.portfolio import visa_portfolj
from stockapp.control import kontrollvy


# ---------------------------------------------------------------------
# Hjälp: valuta-dict -> visar i sidopanelen utan widget-konflikter
# ---------------------------------------------------------------------
def _sidebar_rates() -> Dict[str, float]:
    st.sidebar.subheader("Valutakurser (SEK-bas)")

    # läs sparade kurser som startvärden
    saved = las_sparade_valutakurser()

    # init session_state keys om saknas
    for k in ("USD", "EUR", "CAD", "NOK", "SEK"):
        key = f"rate_{k}_input"
        if key not in st.session_state:
            st.session_state[key] = float(saved.get(k, STANDARD_VALUTAKURSER.get(k, 1.0)))

    cols = st.sidebar.columns(5)
    st.session_state["rate_USD_input"] = cols[0].number_input(
        "USD", min_value=0.0, value=float(st.session_state["rate_USD_input"]), step=0.0001, format="%.6f"
    )
    st.session_state["rate_EUR_input"] = cols[1].number_input(
        "EUR", min_value=0.0, value=float(st.session_state["rate_EUR_input"]), step=0.0001, format="%.6f"
    )
    st.session_state["rate_CAD_input"] = cols[2].number_input(
        "CAD", min_value=0.0, value=float(st.session_state["rate_CAD_input"]), step=0.0001, format="%.6f"
    )
    st.session_state["rate_NOK_input"] = cols[3].number_input(
        "NOK", min_value=0.0, value=float(st.session_state["rate_NOK_input"]), step=0.0001, format="%.6f"
    )
    st.sidebar.write(" ")  # liten luft
    st.session_state["rate_SEK_input"] = 1.0
    cols[4].markdown("**SEK**\n\n`1.0`")

    # Auto-hämtning
    c1, c2, c3 = st.sidebar.columns([1, 1, 2])
    auto = c1.button("Hämta auto")
    spara = c2.button("Spara")
    status_area = st.sidebar.empty()

    if auto:
        rates, misses, provider = hamta_valutakurser_auto()
        # uppdatera inputs
        for k in ("USD", "EUR", "CAD", "NOK"):
            st.session_state[f"rate_{k}_input"] = float(rates.get(k, st.session_state[f"rate_{k}_input"]))
        status_area.info(f"Auto-kurser från: **{provider}**. Missade: {', '.join(misses) if misses else '–'}")

    user_rates = {
        "USD": float(st.session_state["rate_USD_input"]),
        "EUR": float(st.session_state["rate_EUR_input"]),
        "CAD": float(st.session_state["rate_CAD_input"]),
        "NOK": float(st.session_state["rate_NOK_input"]),
        "SEK": 1.0,
    }

    if spara:
        try:
            spara_valutakurser(user_rates)
            status_area.success("Valutakurser sparade.")
        except Exception as e:
            status_area.warning(f"Kunde inte spara valutakurser: {e}")

    # Batch-panel direkt i sidopanelen
    st.sidebar.markdown("---")
    st.sidebar.subheader("Batchkörning")
    if "_runner" not in st.session_state:
        # runner: tar (ticker, user_rates, mode) och returnerar (df_row, logtxt)
        def _runner(tkr: str, rates: Dict[str, float], mode: str) -> Tuple[Dict, str]:
            if mode == "price":
                row, log = run_update_price_only(tkr, rates)
            else:
                row, log = run_update_full(tkr, rates)
            return row, log

        st.session_state["_runner"] = _runner

    # batch-komponenten uppdaterar & sparar via callbacks som vi ger från main()
    return user_rates


# ---------------------------------------------------------------------
# Enkelt skydd – hindra dubbletter
# ---------------------------------------------------------------------
def _ensure_unique_tickers(df: pd.DataFrame) -> pd.DataFrame:
    if "Ticker" not in df.columns:
        return df
    dups = df["Ticker"].duplicated(keep="first")
    if dups.any():
        # behåll första, droppa resten
        df = df[~dups].copy()
    return df


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="K-pf-rslag", layout="wide")
    st.title("K-pf-rslag")

    # Läs data
    df = hamta_data()
    df = ensure_schema(df, FINAL_COLS)
    df = _ensure_unique_tickers(df)

    # Sidopanel (valutor + batch-knappar)
    user_rates = _sidebar_rates()

    # Batch-kontroller (körningar sparar via spara_data-callback)
    def _save_callback(df_new: pd.DataFrame) -> None:
        # stämpla övergripande TS när vi faktiskt skriver
        spara_data(df_new)

    def _recompute_callback(df_current: pd.DataFrame) -> pd.DataFrame:
        # här kan man lägga tunga omräkningar vid behov; just nu pass-through
        return df_current

    df = sidebar_batch_controls(
        df,
        user_rates,
        save_cb=_save_callback,
        recompute_cb=_recompute_callback,
        default_batch_size=BATCH_DEFAULT_SIZE,
        runner=st.session_state["_runner"],
    )

    # Välj vy
    st.markdown("---")
    vy = st.radio(
        "Välj vy",
        ["Investeringsförslag", "Lägg till / uppdatera bolag", "Portfölj", "Kontroll"],
        horizontal=True,
    )

    # Visa vald vy
    if vy == "Investeringsförslag":
        visa_investeringsforslag(df, user_rates, page_size=PROPOSALS_PAGE_SIZE)
    elif vy == "Lägg till / uppdatera bolag":
        df2 = lagg_till_eller_uppdatera(df, user_rates)
        if df2 is not None:
            spara_data(df2)
            st.success("Ändringar sparade.")
            st.session_state["_df_ref"] = df2
    elif vy == "Portfölj":
        visa_portfolj(df, user_rates)
    elif vy == "Kontroll":
        kontrollvy(df)
    else:
        st.info("Ingen vy vald.")

    # Cacha referens i session (för snabbare back-n-forth)
    st.session_state["_df_ref"] = df


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Appfel: {e}")
