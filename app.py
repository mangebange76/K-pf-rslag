import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime

st.set_page_config(page_title="Aktieanalys & investeringsförslag", layout="wide")

# ----------------------
# KONFIGURATION
# ----------------------

SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"
SETTINGS_SHEET_NAME = "Inställningar"

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

# ----------------------
# DATAHANTERING
# ----------------------

def skapa_koppling(blad_namn=SHEET_NAME):
    return client.open_by_url(SHEET_URL).worksheet(blad_namn)

def hamta_data():
    sheet = skapa_koppling()
    data = sheet.get_all_records()
    return pd.DataFrame(data)

def spara_data(df):
    sheet = skapa_koppling()
    # Säkerställ rätt kolumner
    godkända_kolumner = [
        "Ticker", "Bolagsnamn", "Aktuell kurs", "Utestående aktier",
        "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning om 1 år", "Omsättning om 2 år",
        "P/S-snitt", "Riktkurs nu", "Riktkurs om 1 år", "Riktkurs om 2 år",
        "Uppsidepotential (%)", "Antal aktier"
    ]
    for kol in godkända_kolumner:
        if kol not in df.columns:
            df[kol] = 0.0 if "kurs" in kol.lower() or "omsättning" in kol.lower() or "p/s" in kol.lower() else ""

    df = df[[kol for kol in godkända_kolumner if kol in df.columns]]
    df = df.fillna("").astype(str)
    sheet.clear()
    sheet.update([df.columns.tolist()] + df.values.tolist())

def konvertera_typer(df):
    numeriska = [
        "Aktuell kurs", "Utestående aktier",
        "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning om 1 år", "Omsättning om 2 år",
        "P/S-snitt", "Riktkurs nu", "Riktkurs om 1 år", "Riktkurs om 2 år",
        "Uppsidepotential (%)", "Antal aktier"
    ]
    for kol in numeriska:
        if kol in df.columns:
            df[kol] = pd.to_numeric(df[kol], errors="coerce").fillna(0.0)
    return df

def sakerstall_kolumner(df):
    godkanda_kolumner = [
        "Ticker", "Bolagsnamn", "Aktuell kurs", "Utestående aktier",
        "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning om 1 år", "Omsättning om 2 år",
        "P/S-snitt", "Riktkurs nu", "Riktkurs om 1 år", "Riktkurs om 2 år",
        "Uppsidepotential (%)", "Antal aktier"
    ]
    befintliga = df.columns.tolist()

    # Ta bort otillåtna kolumner
    df = df[[col for col in befintliga if col in godkanda_kolumner]]

    # Lägg till saknade kolumner
    for col in godkanda_kolumner:
        if col not in df.columns:
            df[col] = 0.0 if "kurs" in col.lower() or "omsättning" in col.lower() or "p/s" in col.lower() else ""

    # Ordna kolumnordning
    df = df[godkanda_kolumner]
    return df

def las_inställningar():
    try:
        sheet = skapa_koppling(SETTINGS_SHEET_NAME)
        data = sheet.get_all_values()
        df = pd.DataFrame(data[1:], columns=data[0])
        inst = dict(zip(df["Inställning"], df["Värde"]))

        valutakurs = float(str(inst.get("Valutakurs", "0")).replace(",", "."))
        max_portf = float(str(inst.get("Max portföljandel", "100")).replace(",", "."))
        max_risk = float(str(inst.get("Max högriskandel", "100")).replace(",", "."))

        return {
            "Valutakurs": valutakurs,
            "Max portföljandel": max_portf,
            "Max högriskandel": max_risk,
            "Senast ändrad": inst.get("Senast ändrad", "")
        }
    except Exception as e:
        st.error(f"Fel vid läsning av inställningar: {e}")
        return {"Valutakurs": 10.0, "Max portföljandel": 100, "Max högriskandel": 100, "Senast ändrad": ""}

def visa_sidopanel(df, inst):
    st.sidebar.subheader("⚙️ Inställningar")

    ny_valutakurs = st.sidebar.number_input("Valutakurs (USD/SEK)", value=inst["Valutakurs"], step=0.01)
    ny_max_portf = st.sidebar.number_input("Max portföljandel (%)", value=inst["Max portföljandel"], step=0.01)
    ny_max_risk = st.sidebar.number_input("Max högriskandel (%)", value=inst["Max högriskandel"], step=0.01)

    if st.sidebar.button("💾 Spara inställningar"):
        try:
            sheet = skapa_koppling(SETTINGS_SHEET_NAME)
            sheet.update("B2", [[str(ny_valutakurs).replace(".", ",")]])
            sheet.update("B3", [[str(ny_max_portf).replace(".", ",")]])
            sheet.update("B4", [[str(ny_max_risk).replace(".", ",")]])
            sheet.update("B5", [[datetime.today().strftime("%Y-%m-%d")]])
            st.sidebar.success("Inställningar uppdaterade.")
        except Exception as e:
            st.sidebar.error(f"Fel vid uppdatering: {e}")

def uppdatera_berakningar(df):
    df["P/S Q1"] = pd.to_numeric(df["P/S Q1"], errors="coerce").fillna(0)
    df["P/S Q2"] = pd.to_numeric(df["P/S Q2"], errors="coerce").fillna(0)
    df["P/S Q3"] = pd.to_numeric(df["P/S Q3"], errors="coerce").fillna(0)
    df["P/S Q4"] = pd.to_numeric(df["P/S Q4"], errors="coerce").fillna(0)

    df["P/S-snitt"] = df[["P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"]].replace(0, np.nan).mean(axis=1).fillna(0)

    df["Omsättning idag"] = pd.to_numeric(df["Omsättning idag"], errors="coerce").fillna(0)
    df["Omsättning om 1 år"] = pd.to_numeric(df["Omsättning om 1 år"], errors="coerce").fillna(0)
    df["Omsättning om 2 år"] = pd.to_numeric(df["Omsättning om 2 år"], errors="coerce").fillna(0)
    df["Utestående aktier"] = pd.to_numeric(df["Utestående aktier"], errors="coerce").fillna(1)

    df["Riktkurs nu"] = round((df["Omsättning idag"] / df["Utestående aktier"]) * df["P/S-snitt"], 2)
    df["Riktkurs om 1 år"] = round((df["Omsättning om 1 år"] / df["Utestående aktier"]) * df["P/S-snitt"], 2)
    df["Riktkurs om 2 år"] = round((df["Omsättning om 2 år"] / df["Utestående aktier"]) * df["P/S-snitt"], 2)

    df["Aktuell kurs"] = pd.to_numeric(df["Aktuell kurs"], errors="coerce").fillna(0)
    df["Uppsidepotential (%)"] = round(((df["Riktkurs nu"] - df["Aktuell kurs"]) / df["Aktuell kurs"]) * 100, 2)
    return df

def visa_tabellrubrik(titel):
    st.markdown(f"<h4 style='margin-top:20px'>{titel}</h4>", unsafe_allow_html=True)

def visa_analysvy(df):
    st.subheader("📊 Aktieanalys")
    df = uppdatera_berakningar(df)
    st.dataframe(df, use_container_width=True)
    return df

def lagg_till_eller_uppdatera_bolag(df):
    st.subheader("➕ Lägg till eller uppdatera bolag")
    existerande_tickers = df["Ticker"].dropna().unique().tolist()
    valt = st.selectbox("Välj bolag att uppdatera (eller lämna tomt för nytt)", [""] + existerande_tickers)

    data = df[df["Ticker"] == valt].iloc[0] if valt else {}

    inmatning = {}
    for kol in [
        "Ticker", "Bolagsnamn", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning om 1 år", "Omsättning om 2 år",
        "Utestående aktier", "Aktuell kurs", "Antal aktier"
    ]:
        standard = data.get(kol, "")
        if kol in ["Bolagsnamn", "Ticker"]:
            inmatning[kol] = st.text_input(kol, value=str(standard))
        else:
            inmatning[kol] = st.number_input(kol, value=float(standard) if standard != "" else 0.0, step=1.0 if "Antal" in kol or "aktier" in kol else 0.01)

    if st.button("💾 Spara bolag"):
        ny_rad = {k: inmatning[k] for k in inmatning}
        df = df[df["Ticker"] != ny_rad["Ticker"]]
        df = pd.concat([df, pd.DataFrame([ny_rad])], ignore_index=True)
        spara_data(df)
        st.success(f"{ny_rad['Ticker']} sparad.")
    return df

def visa_investeringsforslag(df, inst):
    st.subheader("💡 Investeringsförslag & ombalansering")

    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=10000.0, step=100.0)
    valutakurs = inst["Valutakurs"]
    max_portf = inst["Max portföljandel"]
    max_risk = inst["Max högriskandel"]

    df = uppdatera_berakningar(df)
    df["Värde (SEK)"] = df["Antal aktier"] * df["Aktuell kurs"] * valutakurs
    total_varde = df["Värde (SEK)"].sum()
    df["Portföljandel (%)"] = round(df["Värde (SEK)"] / total_varde * 100, 2)

    högrisk = df[(df["Omsättning idag"] < 1000) & (df["Portföljandel (%)"] > max_risk)]
    minska = df[df["Portföljandel (%)"] > max_portf]
    öka = df[(df["Riktkurs om 1 år"] > df["Aktuell kurs"]) & (df["Portföljandel (%)"] < max_portf)]

    if not minska.empty:
        st.markdown("🔻 **Bolag att minska i:**")
        st.dataframe(minska[["Ticker", "Portföljandel (%)", "Värde (SEK)"]])

    if not öka.empty:
        st.markdown("🔼 **Bolag att öka i:**")
        st.dataframe(öka[["Ticker", "Potential", "Portföljandel (%)"]])

    if not högrisk.empty:
        st.markdown("⚠️ **Högriskvarning:**")
        st.dataframe(högrisk[["Ticker", "Omsättning idag", "Portföljandel (%)"]])

    # Investeringsförslag
    df["Potential"] = df["Riktkurs om 1 år"] - df["Aktuell kurs"]
    df = df[df["Potential"] > 0].sort_values(by="Potential", ascending=False)

    if "förslag_index" not in st.session_state:
        st.session_state["förslag_index"] = 0

    i = st.session_state["förslag_index"]
    if i < len(df):
        rad = df.iloc[i]
        antal = int((kapital_sek / valutakurs) // rad["Aktuell kurs"])
        kostnad = round(antal * rad["Aktuell kurs"] * valutakurs, 2)
        st.markdown(
            f"Köp **{antal} st {rad['Ticker']} ({rad['Bolagsnamn']})** för ca **{kostnad} SEK**\n\n"
            f"Potential: {round(rad['Potential'],2)} USD → Riktkurs om 1 år: {round(rad['Riktkurs om 1 år'],2)} USD"
        )
        if st.button("Nästa förslag"):
            st.session_state["förslag_index"] += 1
    else:
        st.info("Inga fler förslag – klicka om för att börja om.")

def visa_portfolj(df, valutakurs):
    st.subheader("💼 Portföljöversikt")
    df = df.copy()
    df["Värde (SEK)"] = df["Antal aktier"] * df["Aktuell kurs"] * valutakurs
    total = df["Värde (SEK)"].sum()
    df["Andel (%)"] = round(df["Värde (SEK)"] / total * 100, 2)
    st.dataframe(df[["Ticker", "Bolagsnamn", "Antal aktier", "Aktuell kurs", "Värde (SEK)", "Andel (%)"]])

def main():
    st.title("📈 Aktieanalys & investeringsförslag – Manuell valutakurs och aktiekurs")

    df = hamta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)

    inställningar = las_inställningar()
    skriv_sidopanel(inställningar)

    meny = st.sidebar.radio("Navigering", ["Analys", "Lägg till / uppdatera bolag", "Investeringsförslag", "Portfölj"])

    if meny == "Analys":
        df = visa_analysvy(df)

    elif meny == "Lägg till / uppdatera bolag":
        df = lagg_till_eller_uppdatera_bolag(df)

    elif meny == "Investeringsförslag":
        visa_investeringsforslag(df, inställningar)

    elif meny == "Portfölj":
        visa_portfolj(df, inställningar["Valutakurs"])

    spara_data(df)

if __name__ == "__main__":
    main()
