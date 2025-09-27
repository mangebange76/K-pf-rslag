# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd

from stockapp.sheets import load_df, save_df, ensure_schema
from stockapp.rates import sidebar_rates
from stockapp.batch import sidebar_batch_controls
from stockapp.calc import recompute_all
from stockapp.views.control import kontrollvy
from stockapp.views.edit import lagg_till_eller_uppdatera
from stockapp.views.analysis import analysvy
from stockapp.views.proposals import visa_investeringsforslag
from stockapp.views.portfolio import visa_portfolj

st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

def main():
    # 1) Valutor (utan experimental_rerun)
    user_rates = sidebar_rates()

    # 2) Läs & säkerställ schema
    df = ensure_schema(load_df())

    # 3) Sidopanel: batch/åtgärder (progress 1/X, sparas via callback)
    df = sidebar_batch_controls(df, user_rates, save_cb=save_df, recompute_cb=recompute_all)

    # 4) Meny
    meny = st.sidebar.radio("📌 Välj vy", ["Kontroll","Analys","Lägg till / uppdatera bolag","Investeringsförslag","Portfölj"])

    if meny == "Kontroll":
        kontrollvy(df)
    elif meny == "Analys":
        analysvy(df, user_rates)
    elif meny == "Lägg till / uppdatera bolag":
        df2 = lagg_till_eller_uppdatera(df, user_rates, save_cb=save_df, recompute_cb=recompute_all)
        if isinstance(df2, pd.DataFrame):
            df = df2
    elif meny == "Investeringsförslag":
        df_calc = recompute_all(df, user_rates)
        visa_investeringsforslag(df_calc, user_rates, save_cb=save_df)
    elif meny == "Portfölj":
        df_calc = recompute_all(df, user_rates)
        visa_portfolj(df_calc, user_rates)

if __name__ == "__main__":
    main()
