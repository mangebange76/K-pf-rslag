# app.py
import streamlit as st
import pandas as pd

from sheets_utils import (
    hamta_data, spara_data, las_sparade_valutakurser, spara_valutakurser,
    hamta_valutakurs, skapa_snapshot_om_saknas, now_stamp
)
from views import (
    hamta_valutakurser_sidebar, massuppdatera, lagg_till_eller_uppdatera,
    analysvy, visa_investeringsforslag, visa_hamtlogg_panel, spara_logg_till_sheets
)
from calc_and_cache import uppdatera_berakningar

# ---------- Sidinställning ----------
st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

# ---------- Kolumnschema ----------
FINAL_COLS = [
    # Bas
    "Ticker", "Bolagsnamn", "Valuta", "Aktuell kurs", "Årlig utdelning",
    "Utestående aktier", "Antal aktier",
    # P/S & kvartal
    "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "P/S Q1 datum", "P/S Q2 datum", "P/S Q3 datum", "P/S Q4 datum",
    "Källa Aktuell kurs", "Källa Utestående aktier", "Källa P/S", "Källa P/S Q1", "Källa P/S Q2", "Källa P/S Q3", "Källa P/S Q4",
    # Omsättning & riktkurser
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
    # Derivat/övrigt
    "CAGR 5 år (%)", "P/S-snitt",
    # Tidsstämplar & meta
    "Senast manuellt uppdaterad", "Senast auto uppdaterad",
    "TS P/S", "TS Utestående aktier", "TS Omsättning",
]

def säkerställ_kolumner(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for kol in FINAL_COLS:
        if kol not in df.columns:
            if any(x in kol.lower() for x in ["kurs","omsättning","p/s","utdelning","cagr","antal","riktkurs","aktier","snitt"]):
                df[kol] = 0.0
            else:
                df[kol] = ""
    return df

def migrera_gamla_riktkurskolumner(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
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

def konvertera_typer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    num_cols = [
        "Utestående aktier","Antal aktier",
        "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
        "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
        "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
        "Aktuell kurs","Årlig utdelning","CAGR 5 år (%)","P/S-snitt",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in [
        "Ticker","Bolagsnamn","Valuta",
        "P/S Q1 datum","P/S Q2 datum","P/S Q3 datum","P/S Q4 datum",
        "Källa Aktuell kurs","Källa Utestående aktier","Källa P/S","Källa P/S Q1","Källa P/S Q2","Källa P/S Q3","Källa P/S Q4",
        "Senast manuellt uppdaterad","Senast auto uppdaterad",
        "TS P/S","TS Utestående aktier","TS Omsättning",
    ]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df

# ---------- Portföljvy (lokal, enkel) ----------
def visa_portfolj(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📦 Min portfölj")
    port = df[df["Antal aktier"] > 0].copy()
    if port.empty:
        st.info("Du äger inga aktier.")
        return
    port["Växelkurs"] = port["Valuta"].apply(lambda v: hamta_valutakurs(v, user_rates))
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
    total_värde = float(port["Värde (SEK)"].sum())
    port["Andel (%)"] = (port["Värde (SEK)"] / total_värde * 100.0).round(2)
    port["Total årlig utdelning (SEK)"] = port["Antal aktier"] * port["Årlig utdelning"] * port["Växelkurs"]

    st.markdown(f"**Totalt portföljvärde:** {round(total_värde,2)} SEK")
    tot_utd = float(port["Total årlig utdelning (SEK)"].sum())
    st.markdown(f"**Total kommande utdelning:** {round(tot_utd,2)} SEK")
    st.markdown(f"**Ungefärlig månadsutdelning:** {round(tot_utd/12.0,2)} SEK")

    st.dataframe(
        port[["Ticker","Bolagsnamn","Antal aktier","Aktuell kurs","Valuta","Värde (SEK)","Andel (%)","Årlig utdelning","Total årlig utdelning (SEK)"]],
        use_container_width=True
    )

# ---------- MAIN ----------
def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    # Valutakurser + hämtlogg i sidopanelen
    user_rates = hamta_valutakurser_sidebar()
    visa_hamtlogg_panel()
    if st.sidebar.button("🗃️ Spara hämtlogg till Sheets"):
        spara_logg_till_sheets()

    # Läs data
    df = hamta_data()
    if df.empty:
        df = pd.DataFrame({c: [] for c in FINAL_COLS})
        spara_data(df)

    # Säkerställ schema och typer
    df = säkerställ_kolumner(df)
    df = migrera_gamla_riktkurskolumner(df)
    df = konvertera_typer(df)

    # 📸 Snapshot vid start (en gång per dag)
    if "did_bootstrap_snapshot" not in st.session_state:
        ok, msg = skapa_snapshot_om_saknas(df)
        st.session_state["did_bootstrap_snapshot"] = True
        if ok:
            st.sidebar.success(msg)
        else:
            st.sidebar.caption(msg)

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
