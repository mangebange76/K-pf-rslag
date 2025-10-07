# app.py
import streamlit as st
import pandas as pd

from schema_utils import FINAL_COLS, säkerställ_kolumner, migrera_gamla_riktkurskolumner, konvertera_typer
from sheets_utils import hamta_data, spara_data, skapa_snapshot_om_saknas
from views import (hamta_valutakurser_sidebar, massuppdatera, lagg_till_eller_uppdatera,
                   analysvy, visa_investeringsforslag, visa_hamtlogg_panel, spara_logg_till_sheets)
from calc_and_cache import uppdatera_berakningar

st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")
st.title("📊 Aktieanalys och investeringsförslag")

user_rates = hamta_valutakurser_sidebar()

# Hämtlogg-panel + spara-knapp
visa_hamtlogg_panel()
if st.sidebar.button("🧾 Spara hämtlogg till Sheets"):
    spara_logg_till_sheets()

# Läs & normalisera data
df = hamta_data()
if df.empty:
    df = pd.DataFrame({c: [] for c in FINAL_COLS})
    spara_data(df)

df = säkerställ_kolumner(df)
df = migrera_gamla_riktkurskolumner(df)
df = konvertera_typer(df)

# Snapshot vid start
ok, msg = skapa_snapshot_om_saknas(df)
st.sidebar.info(msg)

# Massuppdatering/snapshot-knapp
df = massuppdatera(df, key_prefix="global", user_rates=user_rates)

# Vyer
meny = st.sidebar.radio("📌 Välj vy", ["Analys","Lägg till / uppdatera bolag","Investeringsförslag","Portfölj"])
if meny == "Analys":
    df = uppdatera_berakningar(df, user_rates)
    analysvy(df, user_rates)
elif meny == "Lägg till / uppdatera bolag":
    df = lagg_till_eller_uppdatera(df, user_rates)
elif meny == "Investeringsförslag":
    df = uppdatera_berakningar(df, user_rates)
    visa_investeringsforslag(df, user_rates)
elif meny == "Portfölj":
    df = uppdatera_berakningar(df, user_rates)
    st.header("📦 Min portfölj")
    port = df[df["Antal aktier"] > 0].copy()
    if port.empty:
        st.info("Du äger inga aktier.")
    else:
        from sheets_utils import hamta_valutakurs
        port["Växelkurs"] = port["Valuta"].apply(lambda v: hamta_valutakurs(v, user_rates))
        port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
        total_värde = float(port["Värde (SEK)"].sum())
        port["Andel (%)"] = round(port["Värde (SEK)"] / total_värde * 100.0, 2)
        port["Total årlig utdelning (SEK)"] = port["Antal aktier"] * port["Årlig utdelning"] * port["Växelkurs"]
        tot_utd = float(port["Total årlig utdelning (SEK)"].sum())
        st.markdown(f"**Totalt portföljvärde:** {round(total_värde,2)} SEK")
        st.markdown(f"**Total kommande utdelning:** {round(tot_utd,2)} SEK")
        st.markdown(f"**Ungefärlig månadsutdelning:** {round(tot_utd/12.0,2)} SEK")
        st.dataframe(
            port[["Ticker","Bolagsnamn","Antal aktier","Aktuell kurs","Valuta","Värde (SEK)","Andel (%)","Årlig utdelning","Total årlig utdelning (SEK)"]],
            use_container_width=True
        )
