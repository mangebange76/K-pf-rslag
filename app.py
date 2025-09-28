# app.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np

# ---- Våra moduler -----------------------------------------------------------
from stockapp.config import FINAL_COLS, TS_FIELDS, STANDARD_VALUTAKURSER
from stockapp.storage import hamta_data, spara_data
from stockapp.rates import (
    las_sparade_valutakurser,
    spara_valutakurser,
    hamta_valutakurser_auto,
    hamta_valutakurs,
)
from stockapp.utils import (
    now_stamp, now_dt,
    säkerställ_kolumner,
    migrera_gamla_riktkurskolumner,
    konvertera_typer,
    uppdatera_berakningar,
)
from stockapp.batch import sidebar_batch_controls
from stockapp.sources import run_update_price_only, run_update_full
from stockapp.views import (
    kontrollvy,
    analysvy,
    lagg_till_eller_uppdatera,
    visa_investeringsforslag,
    visa_portfolj,
)

# -----------------------------------------------------------------------------
# SIDKONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Aktieanalys och investeringsförslag",
    layout="wide",
)

# -----------------------------------------------------------------------------
# Hjälpare: init av session_state
# -----------------------------------------------------------------------------
def _init_session_state():
    # Data-referens
    if "_df_ref" not in st.session_state:
        st.session_state["_df_ref"] = None

    # Runner-val (single-uppdatering & batch)
    if "_single_update_runner" not in st.session_state:
        st.session_state["_single_update_runner"] = run_update_full
    if "_price_update_runner" not in st.session_state:
        st.session_state["_price_update_runner"] = run_update_price_only
    if "runner_mode" not in st.session_state:
        st.session_state["runner_mode"] = "Full auto (Yahoo+SEC)"

    # Valutafält – initera med sparat eller standard, men endast en gång
    saved = las_sparade_valutakurser()
    def _get_rate(code, default):
        try:
            return float(saved.get(code, default))
        except Exception:
            return float(default)

    if "rate_usd_input" not in st.session_state:
        st.session_state.rate_usd_input = _get_rate("USD", STANDARD_VALUTAKURSER["USD"])
    if "rate_nok_input" not in st.session_state:
        st.session_state.rate_nok_input = _get_rate("NOK", STANDARD_VALUTAKURSER["NOK"])
    if "rate_cad_input" not in st.session_state:
        st.session_state.rate_cad_input = _get_rate("CAD", STANDARD_VALUTAKURSER["CAD"])
    if "rate_eur_input" not in st.session_state:
        st.session_state.rate_eur_input = _get_rate("EUR", STANDARD_VALUTAKURSER["EUR"])

    # Batch-UI minne
    if "batch_last_log" not in st.session_state:
        st.session_state["batch_last_log"] = None


# -----------------------------------------------------------------------------
# Sidopanel: valutakurser + batch-kontroller
# -----------------------------------------------------------------------------
def _sidebar_rates() -> dict:
    st.sidebar.header("💱 Valutakurser → SEK")

    # 1) Åtgärdsknappar överst, så att ev. state uppdateras INNAN inputs ritas.
    btn_cols = st.sidebar.columns(3)
    with btn_cols[0]:
        if st.button("🌐 Hämta kurser autom.", key="btn_rates_auto"):
            auto_rates, misses, provider = hamta_valutakurser_auto()
            # Uppdatera state-nycklar FÖRE inputs skapas
            st.session_state.rate_usd_input = float(auto_rates.get("USD", st.session_state.rate_usd_input))
            st.session_state.rate_nok_input = float(auto_rates.get("NOK", st.session_state.rate_nok_input))
            st.session_state.rate_cad_input = float(auto_rates.get("CAD", st.session_state.rate_cad_input))
            st.session_state.rate_eur_input = float(auto_rates.get("EUR", st.session_state.rate_eur_input))
            if misses:
                st.sidebar.warning("Kunde inte hämta alla par:\n- " + "\n- ".join(misses))
            st.sidebar.success(f"Kurser uppdaterade (källa: {provider}).")

    with btn_cols[1]:
        if st.button("💾 Spara kurser", key="btn_rates_save"):
            to_save = {
                "USD": float(st.session_state.rate_usd_input),
                "NOK": float(st.session_state.rate_nok_input),
                "CAD": float(st.session_state.rate_cad_input),
                "EUR": float(st.session_state.rate_eur_input),
                "SEK": 1.0,
            }
            spara_valutakurser(to_save)
            st.sidebar.success("Valutakurser sparade.")

    with btn_cols[2]:
        if st.button("↻ Läs sparade", key="btn_rates_reload"):
            saved = las_sparade_valutakurser()
            st.session_state.rate_usd_input = float(saved.get("USD", st.session_state.rate_usd_input))
            st.session_state.rate_nok_input = float(saved.get("NOK", st.session_state.rate_nok_input))
            st.session_state.rate_cad_input = float(saved.get("CAD", st.session_state.rate_cad_input))
            st.session_state.rate_eur_input = float(saved.get("EUR", st.session_state.rate_eur_input))
            st.sidebar.info("Inlästa sparade kurser.")

    # 2) Själva inputs (nu är session_state redan uppdaterat om knappar trycktes)
    usd = st.sidebar.number_input("USD → SEK", value=float(st.session_state.rate_usd_input), step=0.01, format="%.4f", key="usd_input_widget")
    nok = st.sidebar.number_input("NOK → SEK", value=float(st.session_state.rate_nok_input), step=0.01, format="%.4f", key="nok_input_widget")
    cad = st.sidebar.number_input("CAD → SEK", value=float(st.session_state.rate_cad_input), step=0.01, format="%.4f", key="cad_input_widget")
    eur = st.sidebar.number_input("EUR → SEK", value=float(st.session_state.rate_eur_input), step=0.01, format="%.4f", key="eur_input_widget")

    # 3) Spegla tillbaka till session_state (så att vyer använder de senaste)
    st.session_state.rate_usd_input = float(usd)
    st.session_state.rate_nok_input = float(nok)
    st.session_state.rate_cad_input = float(cad)
    st.session_state.rate_eur_input = float(eur)

    return {
        "USD": float(usd),
        "NOK": float(nok),
        "CAD": float(cad),
        "EUR": float(eur),
        "SEK": 1.0,
    }


def _sidebar_batch_and_actions(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠️ Batch & Snabbåtgärder")

    # Välj runner-läge
    mode = st.sidebar.selectbox(
        "Läge för uppdatering",
        ["Full auto (Yahoo+SEC)", "Endast kurs (Yahoo)"],
        index=0 if st.session_state["runner_mode"] == "Full auto (Yahoo+SEC)" else 1,
    )
    st.session_state["runner_mode"] = mode
    runner = run_update_full if mode.startswith("Full") else run_update_price_only
    st.session_state["_single_update_runner"] = runner  # används även i vyerna

    # Batch-panel (intern progress med 1/X visas i modulen)
    df2 = sidebar_batch_controls(
        df,
        user_rates,
        save_cb=lambda d: spara_data(d),
        recompute_cb=lambda d: uppdatera_berakningar(d, user_rates),
        runner=runner,
    )
    return df2


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    _init_session_state()

    st.title("📊 Aktieanalys och investeringsförslag")

    # 1) Sidopanel kurser
    user_rates = _sidebar_rates()

    # 2) Läs data
    df = hamta_data()
    if df is None or df.empty:
        df = pd.DataFrame({c: [] for c in FINAL_COLS})
    df = säkerställ_kolumner(df)
    df = migrera_gamla_riktkurskolumner(df)
    df = konvertera_typer(df)

    # 3) Batch & actions i sidopanelen
    df = _sidebar_batch_and_actions(df, user_rates)

    # Håll referens i session_state (används av vyerna)
    st.session_state["_df_ref"] = df

    # 4) Meny
    st.sidebar.markdown("---")
    meny = st.sidebar.radio("📌 Välj vy", ["Kontroll","Analys","Lägg till / uppdatera bolag","Investeringsförslag","Portfölj"], index=0)

    # 5) Recompute-ramverk före vy (så att formler håller sig fräscha)
    df_calc = uppdatera_berakningar(df.copy(), user_rates)

    # 6) Vyer
    if meny == "Kontroll":
        kontrollvy(df_calc)
    elif meny == "Analys":
        analysvy(df_calc, user_rates)
    elif meny == "Lägg till / uppdatera bolag":
        df2 = lagg_till_eller_uppdatera(df_calc, user_rates)
        # Om vy returnerar DataFrame (t.ex. efter spar), skriv & uppdatera ref
        if isinstance(df2, pd.DataFrame) and not df2.equals(df):
            spara_data(df2)
            st.session_state["_df_ref"] = df2
    elif meny == "Investeringsförslag":
        visa_investeringsforslag(df_calc, user_rates)
    elif meny == "Portfölj":
        visa_portfolj(df_calc, user_rates)


if __name__ == "__main__":
    main()
