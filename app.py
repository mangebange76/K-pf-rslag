# app.py
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd

# ---- Moduler från ditt paket ----
from stockapp.storage import hamta_data, spara_data
from stockapp.utils import ensure_schema
from stockapp.rates import las_sparade_valutakurser, hamta_valutakurs
from stockapp.views import kontrollvy, analysvy, lagg_till_eller_uppdatera, visa_portfolj
from stockapp.invest import visa_investeringsforslag  # <— viktig ändring: invest.py

st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

def _load_df() -> pd.DataFrame:
    df = hamta_data()
    df = ensure_schema(df)
    return df

def _sidebar_rates() -> dict:
    st.sidebar.header("💱 Valutakurser → SEK")
    saved = las_sparade_valutakurser()
    usd = st.sidebar.number_input("USD → SEK", value=float(saved.get("USD", 10.0)), step=0.01, format="%.4f", key="rate_usd_num")
    nok = st.sidebar.number_input("NOK → SEK", value=float(saved.get("NOK", 1.0)), step=0.01, format="%.4f", key="rate_nok_num")
    cad = st.sidebar.number_input("CAD → SEK", value=float(saved.get("CAD", 7.0)), step=0.01, format="%.4f", key="rate_cad_num")
    eur = st.sidebar.number_input("EUR → SEK", value=float(saved.get("EUR", 11.0)), step=0.01, format="%.4f", key="rate_eur_num")

    user_rates = {"USD": usd, "NOK": nok, "CAD": cad, "EUR": eur, "SEK": 1.0}
    st.sidebar.caption("Tips: Sparade kurser läses automatiskt vid start. Auto-hämtning kan läggas till här om du vill.")
    return user_rates

def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    # 1) Läs data & säkra schema
    df = _load_df()

    # 2) Kurser i sidopanelen
    user_rates = _sidebar_rates()

    # 3) Meny
    meny = st.sidebar.radio(
        "📌 Välj vy",
        ["Kontroll", "Analys", "Lägg till / uppdatera bolag", "Investeringsförslag", "Portfölj"],
        index=0
    )

    if meny == "Kontroll":
        kontrollvy(df)
    elif meny == "Analys":
        analysvy(df, user_rates)
    elif meny == "Lägg till / uppdatera bolag":
        df2 = lagg_till_eller_uppdatera(df, user_rates)
        # Om vyn returnerar uppdaterat df, spara
        if df2 is not None and not df2.equals(df):
            spara_data(df2)
            st.success("Ändringar sparade.")
    elif meny == "Investeringsförslag":
        visa_investeringsforslag(df, user_rates)
    elif meny == "Portfölj":
        visa_portfolj(df, user_rates)

if __name__ == "__main__":
    main()
