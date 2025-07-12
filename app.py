import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime

st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"
SETTINGS_SHEET_NAME = "Inställningar"

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

def skapa_koppling(blad_namn=SHEET_NAME):
    return client.open_by_url(SHEET_URL).worksheet(blad_namn)

def hamta_data():
    sheet = skapa_koppling()
    data = sheet.get_all_records()
    return pd.DataFrame(data)

def spara_data(df):
    df = säkerställ_kolumner(df)
    sheet = skapa_koppling()
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.astype(str).values.tolist())

def konvertera_typer(df):
    kolumner = [
        "Omsättning idag", "Omsättning om 1 år", "Omsättning om 2 år",
        "Aktuell kurs", "Antal aktier", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"
    ]
    for kol in kolumner:
        if kol in df.columns:
            df[kol] = pd.to_numeric(df[kol], errors="coerce").fillna(0.0)
    return df

def säkerställ_kolumner(df):
    godkända_kolumner = [
        "Ticker", "Bolagsnamn", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning om 1 år", "Omsättning om 2 år",
        "Aktuell kurs", "Antal aktier",
        "P/S-snitt", "Riktkurs nu", "Riktkurs om 1 år", "Riktkurs om 2 år",
        "Uppsidepotential (%)"
    ]
    df = df[[col for col in df.columns if col in godkända_kolumner]]
    for kol in godkända_kolumner:
        if kol not in df.columns:
            df[kol] = 0.0 if "P/S" in kol or "Omsättning" in kol or "kurs" in kol else ""
    return df

def las_inställningar():
    try:
        sheet = skapa_koppling(SETTINGS_SHEET_NAME)
        data = sheet.get_all_values()
        df = pd.DataFrame(data[1:], columns=data[0])
        inst = dict(zip(df["Inställning"], df["Värde"]))
        return {
            "Valutakurs": float(str(inst.get("Valutakurs", "10")).replace(",", ".")),
            "Max portföljandel": float(str(inst.get("Max portföljandel", "100")).replace(",", ".")),
            "Max högriskandel": float(str(inst.get("Max högriskandel", "100")).replace(",", ".")),
            "Senast ändrad": inst.get("Senast ändrad", "")
        }
    except Exception as e:
        st.error(f"Fel vid läsning av inställningar: {e}")
        return {"Valutakurs": 10.0, "Max portföljandel": 100, "Max högriskandel": 100, "Senast ändrad": ""}

def skriv_sidopanel(inställningar):
    st.sidebar.header("Inställningar")
    ny_valutakurs = st.sidebar.number_input("Valutakurs (USD till SEK)", value=inställningar["Valutakurs"], step=0.01)
    ny_max_portf = st.sidebar.number_input("Max portföljandel (%)", value=inställningar["Max portföljandel"], step=0.01)
    ny_max_risk = st.sidebar.number_input("Max högriskandel (%)", value=inställningar["Max högriskandel"], step=0.01)

    if st.sidebar.button("Spara inställningar"):
        try:
            sheet = skapa_koppling(SETTINGS_SHEET_NAME)
            sheet.update("B2", [[str(ny_valutakurs).replace(".", ",")]])
            sheet.update("B3", [[str(ny_max_portf).replace(".", ",")]])
            sheet.update("B4", [[str(ny_max_risk).replace(".", ",")]])
            sheet.update("B5", [[datetime.today().strftime("%Y-%m-%d")]])
            st.sidebar.success("Inställningar uppdaterade.")
        except Exception as e:
            st.sidebar.error(f"Fel vid uppdatering av inställningar: {e}")

def uppdatera_berakningar(df):
    df["P/S-snitt"] = df[["P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"]].replace(0, np.nan).mean(axis=1).fillna(0)
    df["Riktkurs nu"] = round((df["Omsättning idag"] / df["Antal aktier"]) * df["P/S-snitt"], 2)
    df["Riktkurs om 1 år"] = round((df["Omsättning om 1 år"] / df["Antal aktier"]) * df["P/S-snitt"], 2)
    df["Riktkurs om 2 år"] = round((df["Omsättning om 2 år"] / df["Antal aktier"]) * df["P/S-snitt"], 2)
    df["Uppsidepotential (%)"] = round(((df["Riktkurs nu"] - df["Aktuell kurs"]) / df["Aktuell kurs"]) * 100, 2)
    return df

def visa_investeringsforslag(df, valutakurs, max_portfoljandel, max_hogriskandel):
    st.subheader("📈 Investeringsförslag")
    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=10000.0, step=100.0, key="kapital")

    if "förslag_index" not in st.session_state:
        st.session_state["förslag_index"] = 0

    df = df[df["Riktkurs om 1 år"] > df["Aktuell kurs"]].copy()
    df["Potential"] = df["Riktkurs om 1 år"] - df["Aktuell kurs"]
    df = df.sort_values(by="Potential", ascending=False)

    if df.empty:
        st.info("Inga bolag har högre riktkurs än aktuell kurs.")
        return

    kapital_usd = kapital_sek / valutakurs
    df["Värde (SEK)"] = df["Antal aktier"] * df["Aktuell kurs"] * valutakurs
    totalvarde = df["Värde (SEK)"].sum()
    df["Portföljandel (%)"] = round(df["Värde (SEK)"] / totalvarde * 100, 2)

    minska = df[df["Portföljandel (%)"] > max_portfoljandel]
    öka = df[(df["Riktkurs om 1 år"] > df["Aktuell kurs"]) & (df["Portföljandel (%)"] < max_portfoljandel)]
    högrisk = df[(df["Omsättning idag"] < 1000) & (df["Portföljandel (%)"] > max_hogriskandel)]

    if not minska.empty:
        st.write("🔻 **Bolag att minska i:**")
        st.dataframe(minska[["Ticker", "Portföljandel (%)", "Värde (SEK)"]])

    if not öka.empty:
        st.write("🔼 **Bolag att öka i:**")
        st.dataframe(öka[["Ticker", "Potential", "Portföljandel (%)"]])

    if not högrisk.empty:
        st.write("⚠️ **Högriskvarning:**")
        st.dataframe(högrisk[["Ticker", "Omsättning idag", "Portföljandel (%)"]])

    st.markdown("### 💡 Bästa investeringsförslag just nu:")
    i = st.session_state["förslag_index"]

    if i < len(df):
        rad = df.iloc[i]
        antal = int(kapital_usd // rad["Aktuell kurs"])
        kostnad_sek = round(antal * rad["Aktuell kurs"] * valutakurs, 2)
        st.markdown(
            f"Köp **{antal} st {rad['Ticker']} ({rad['Bolagsnamn']})** för ca **{kostnad_sek} SEK**\n\n"
            f"Potential: {round(rad['Potential'], 2)} USD → Riktkurs om 1 år: {round(rad['Riktkurs om 1 år'], 2)} USD"
        )
        if st.button("Nästa förslag"):
            st.session_state["förslag_index"] += 1
    else:
        st.info("Inga fler förslag. Starta om för att se från början.")

def main():
    st.set_page_config(page_title="Aktieanalys & investeringsförslag", layout="wide")
    st.title("📈 Aktieanalys & investeringsförslag – Manuell valutakurs och aktiekurs")

    inställningar = las_inställningar()
    skriv_sidopanel(inställningar)

    df = hamta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)
    df = uppdatera_berakningar(df)

    meny = st.sidebar.radio("Välj vy", ["Analys", "Investeringsförslag"])

    if meny == "Analys":
        st.subheader("🔍 Aktier i databasen")
        st.dataframe(df, use_container_width=True)

    elif meny == "Investeringsförslag":
        visa_investeringsforslag(df, inställningar["Valutakurs"], inställningar["Max portföljandel"], inställningar["Max högriskandel"])

    spara_data(df)

if __name__ == "__main__":
    main()
