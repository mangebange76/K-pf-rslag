# app.py (del 1/2)
import streamlit as st
import pandas as pd

# —— Våra moduler
from sheets_utils import (
    hamta_data, spara_data, hamta_valutakurs,
    skapa_snapshot_om_saknas, now_stamp
)
from calc_and_cache import uppdatera_berakningar
from views import (
    hamta_valutakurser_sidebar, massuppdatera,
    lagg_till_eller_uppdatera, analysvy,
    visa_investeringsforslag, visa_hamtlogg_panel, spara_logg_till_sheets
)

st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

# —— Kolumnschema (inkl källor, datumtaggar & auto-tidstämpel)
FINAL_COLS = [
    "Ticker", "Bolagsnamn", "Utestående aktier",
    "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "P/S Q1 datum", "P/S Q2 datum", "P/S Q3 datum", "P/S Q4 datum",
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
    "Antal aktier", "Valuta", "Årlig utdelning", "Aktuell kurs",
    "CAGR 5 år (%)", "P/S-snitt",
    "Senast manuellt uppdaterad", "Senast auto uppdaterad",
    "TS P/S", "TS Omsättning", "TS Utestående aktier",
    "Källa Aktuell kurs", "Källa Utestående aktier",
    "Källa P/S", "Källa P/S Q1", "Källa P/S Q2", "Källa P/S Q3", "Källa P/S Q4"
]

NUM_COLS = {
    "Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
    "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
    "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
    "Antal aktier","Årlig utdelning","Aktuell kurs","CAGR 5 år (%)","P/S-snitt"
}

def säkerställ_kolumner(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in FINAL_COLS:
        if c not in df.columns:
            if c in NUM_COLS:
                df[c] = 0.0
            else:
                df[c] = ""
    # typkonvertering
    for c in df.columns:
        if c in NUM_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        else:
            df[c] = df[c].astype(str)
    return df

def migrera_gamla_riktkurskolumner(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "Riktkurs 2026": "Riktkurs om 1 år",
        "Riktkurs 2027": "Riktkurs om 2 år",
        "Riktkurs 2028": "Riktkurs om 3 år",
        "Riktkurs om idag": "Riktkurs idag",
    }
    for old, new in mapping.items():
        if old in df.columns:
            if new not in df.columns:
                df[new] = 0.0
            new_vals = pd.to_numeric(df[new], errors="coerce").fillna(0.0)
            old_vals = pd.to_numeric(df[old], errors="coerce").fillna(0.0)
            mask = (new_vals == 0.0) & (old_vals > 0.0)
            df.loc[mask, new] = old_vals[mask]
            df = df.drop(columns=[old])
    return df

# —— Portföljvy (lätt vikt, kan ligga här)
def visa_portfolj(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📦 Min portfölj")
    port = df[df["Antal aktier"] > 0].copy()
    if port.empty:
        st.info("Du äger inga aktier.")
        return
    port["Växelkurs"] = port["Valuta"].apply(lambda v: hamta_valutakurserna(v, user_rates))
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
    total_värde = float(port["Värde (SEK)"].sum())
    port["Andel (%)"] = (port["Värde (SEK)"] / total_värde * 100.0).round(2)
    port["Total årlig utdelning (SEK)"] = port["Antal aktier"] * port["Årlig utdelning"] * port["Växelkurs"]
    tot_utd = float(port["Total årlig utdelning (SEK)"].sum())

    st.markdown(f"**Totalt portföljvärde:** {round(total_värde,2)} SEK")
    st.markdown(f"**Total kommande utdelning:** {round(tot_utd,2)} SEK")
    st.markdown(f"**Ungefärlig månadsutdelning:** {round(tot_utd/12.0,2)} SEK")

    st.dataframe(
        port[["Ticker","Bolagsnamn","Antal aktier","Aktuell kurs","Valuta","Värde (SEK)","Andel (%)","Årlig utdelning","Total årlig utdelning (SEK)"]],
        use_container_width=True
    )

def hamta_valutakurserna(v: str, user_rates: dict) -> float:
    try:
        return hamta_valutakurs(v, user_rates)
    except Exception:
        return 1.0

# —— Main
def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    # Sidopanel: valutakurser och hämtlogg
    user_rates = hamta_valutakurser_sidebar()
    with st.sidebar:
        visa_hamtlogg_panel()
        if st.button("🧾 Spara logg till Sheets"):
            spara_logg_till_sheets()

    # Läs data från Sheets
    df = hamta_data()
    if df.empty:
        df = pd.DataFrame({c: [] for c in FINAL_COLS})
        spara_data(df)

    # Säkerställ schema & migrera ev. gamla kolumner
    df = säkerställ_kolumner(df)
    df = migrera_gamla_riktkurskolumner(df)

    # Skapa snapshot vid start (om saknas)
    _ok, _msg = skapa_snapshot_om_saknas(df)

    # Global massuppdateringsknapp i sidopanelen
    df = massuppdatera(df, key_prefix="global", user_rates=user_rates)

    # Meny
    meny = st.sidebar.radio("📌 Välj vy", ["Analys","Lägg till / uppdatera bolag","Investeringsförslag","Portfölj"])
    if meny == "Analys":
        analysvy(df, user_rates)
    elif meny == "Lägg till / uppdatera bolag":
        df = lagg_till_eller_uppdatera(df, user_rates)
    elif meny == "Investeringsförslag":
        df = uppdatera_berakningar(df, user_rates)
        visa_investeringsforslag(df, user_rates)
    elif meny == "Portfölj":
        df = uppdatera_berakningar(df, user_rates)
        visa_portfolj(df, user_rates)

if __name__ == "__main__":
    main()

# app.py (del 2/2)
# (Den här filen innehåller hela appen; del 2 finns bara för att matcha din önskan om uppdelning.
#  Det finns ingen extra kod här – del 1 innehåller allt som körs.)
