# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd

from stockapp.config import FINAL_COLS, TS_FIELDS, STANDARD_VALUTAKURSER
from stockapp.utils import säkerställ_kolumner, konvertera_typer, auto_rates_fetch
from stockapp.storage import read_dataframe, write_dataframe, backup_snapshot, las_sparade_valutakurser, spara_valutakurser
from stockapp.views import kontrollvy, analysvy, lagg_till_eller_uppdatera, visa_investeringsforslag, visa_portfolj
from stockapp.calc import recompute_all
from stockapp.batch import sidebar_batch_controls

st.set_page_config(page_title="Aktieanalys & Investeringsförslag", layout="wide")

# ---------- Hjälpare ----------
def _init_session():
    for k, v in {
        "rate_usd_input": STANDARD_VALUTAKURSER["USD"],
        "rate_nok_input": STANDARD_VALUTAKURSER["NOK"],
        "rate_cad_input": STANDARD_VALUTAKURSER["CAD"],
        "rate_eur_input": STANDARD_VALUTAKURSER["EUR"],
        "rates_reload": 0,
        "_df_ref": None,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

def _sidebar_rates():
    st.sidebar.header("💱 Valutakurser → SEK")

    # Om auto-payload finns, skriv in i widget-state före render
    if "auto_rates_payload" in st.session_state:
        payload = st.session_state.pop("auto_rates_payload")
        for k, v in payload.items():
            if k=="USD": st.session_state["rate_usd_input"] = float(v)
            if k=="NOK": st.session_state["rate_nok_input"] = float(v)
            if k=="CAD": st.session_state["rate_cad_input"] = float(v)
            if k=="EUR": st.session_state["rate_eur_input"] = float(v)

    usd = st.sidebar.number_input("USD → SEK", key="rate_usd_input", step=0.01, format="%.4f")
    nok = st.sidebar.number_input("NOK → SEK", key="rate_nok_input", step=0.01, format="%.4f")
    cad = st.sidebar.number_input("CAD → SEK", key="rate_cad_input", step=0.01, format="%.4f")
    eur = st.sidebar.number_input("EUR → SEK", key="rate_eur_input", step=0.01, format="%.4f")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🌐 Hämta automatiskt"):
            auto_rates = auto_rates_fetch()
            if auto_rates:
                st.session_state["auto_rates_payload"] = auto_rates
                st.sidebar.success("Kurser hämtade – uppdaterade i fälten.")
            else:
                st.sidebar.warning("Kunde inte hämta kurser.")
    with col2:
        if st.button("💾 Spara kurser"):
            spara_valutakurser({"USD": usd, "NOK": nok, "CAD": cad, "EUR": eur, "SEK": 1.0})
            st.session_state["rates_reload"] += 1
            st.sidebar.success("Valutakurser sparade.")

    if st.sidebar.button("↻ Läs sparade kurser"):
        sr = las_sparade_valutakurser()
        st.session_state["auto_rates_payload"] = sr
        st.sidebar.info("Inlästa sparade kurser.")

    return {"USD": float(usd), "NOK": float(nok), "CAD": float(cad), "EUR": float(eur), "SEK": 1.0}

def _sidebar_batch_and_actions(df: pd.DataFrame, user_rates: dict):
    st.sidebar.markdown("---")
    # Batch (med progress 1/X i text + progressbar hanteras i batch.sidebar)
    def _save(df_to_write):
        write_dataframe(df_to_write)

    def _recompute(df_to_calc, rates):
        return recompute_all(df_to_calc, rates)

    df2 = sidebar_batch_controls(df, user_rates, save_cb=_save, recompute_cb=_recompute)
    return df2

# ---------- MAIN ----------
def main():
    _init_session()
    st.title("📊 Aktieanalys och investeringsförslag")

    # Läs data
    df = read_dataframe()
    if df.empty:
        df = pd.DataFrame({c: [] for c in FINAL_COLS})
    df = säkerställ_kolumner(df, FINAL_COLS)
    df = konvertera_typer(df)
    st.session_state["_df_ref"] = df

    # Sidopanel: kurser
    user_rates = _sidebar_rates()

    # Sidopanel: batch
    df = _sidebar_batch_and_actions(st.session_state["_df_ref"], user_rates)
    st.session_state["_df_ref"] = df

    # Meny
    meny = st.sidebar.radio("📌 Välj vy", ["Kontroll","Analys","Lägg till / uppdatera bolag","Investeringsförslag","Portfölj"])

    if meny == "Kontroll":
        kontrollvy(st.session_state["_df_ref"])
    elif meny == "Analys":
        analysvy(st.session_state["_df_ref"], user_rates)
    elif meny == "Lägg till / uppdatera bolag":
        df2 = lagg_till_eller_uppdatera(st.session_state["_df_ref"], user_rates, save_cb=write_dataframe)
        st.session_state["_df_ref"] = df2
    elif meny == "Investeringsförslag":
        df_calc = recompute_all(st.session_state["_df_ref"], user_rates)
        visa_investeringsforslag(df_calc, user_rates)
    elif meny == "Portfölj":
        df_calc = recompute_all(st.session_state["_df_ref"], user_rates)
        visa_portfolj(df_calc, user_rates)

if __name__ == "__main__":
    main()
