import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime

st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

# ---------------------------------------
# KONFIGURATION OCH GOOGLE SHEETS-KOPPLING
# ---------------------------------------

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
    sheet = skapa_koppling()
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.astype(str).values.tolist())

def konvertera_typer(df):
    kolumner = [
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år",
        "Utestående aktier", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Aktuell kurs", "Antal aktier"
    ]
    for kol in kolumner:
        if kol in df.columns:
            df[kol] = pd.to_numeric(df[kol], errors="coerce").fillna(0.0)
    return df

def säkerställ_kolumner(df):
    nödvändiga = [
        "Ticker", "Bolagsnamn", "Aktuell kurs", "Utestående aktier",
        "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år",
        "P/S-snitt", "Riktkurs nu", "Riktkurs om 1 år", "Riktkurs om 2 år",
        "Antal aktier"
    ]
    # Om kolumn saknas lägg till med default värde
    for kol in nödvändiga:
        if kol not in df.columns:
            df[kol] = 0.0 if kol not in ["Ticker", "Bolagsnamn"] else ""
    # Ta bort kolumner som inte finns med i nödvändiga
    df = df[[kol for kol in df.columns if kol in nödvändiga] + [kol for kol in nödvändiga if kol not in df.columns]]
    return df

# ---------------------------------------
# BERÄKNINGAR OCH UPPDATERING AV DATAFRAME
# ---------------------------------------

def uppdatera_berakningar(df):
    for i, row in df.iterrows():
        ps_values = [row.get("P/S Q1", 0), row.get("P/S Q2", 0), row.get("P/S Q3", 0), row.get("P/S Q4", 0)]
        ps_values = [ps for ps in ps_values if ps > 0]
        ps_snitt = round(np.mean(ps_values), 2) if ps_values else 0.0
        df.at[i, "P/S-snitt"] = ps_snitt

        if row.get("Utestående aktier", 0) > 0 and ps_snitt > 0:
            df.at[i, "Riktkurs nu"] = round((row.get("Omsättning idag", 0) * ps_snitt) / row["Utestående aktier"], 2)
            df.at[i, "Riktkurs om 1 år"] = round((row.get("Omsättning nästa år", 0) * ps_snitt) / row["Utestående aktier"], 2)
            df.at[i, "Riktkurs om 2 år"] = round((row.get("Omsättning om 2 år", 0) * ps_snitt) / row["Utestående aktier"], 2)
        else:
            df.at[i, "Riktkurs nu"] = 0.0
            df.at[i, "Riktkurs om 1 år"] = 0.0
            df.at[i, "Riktkurs om 2 år"] = 0.0
    return df

# ---------------------------------------
# INVESTERINGSFÖRSLAG, OMBALANSERING OCH HÖGRISKVARNINGAR
# ---------------------------------------

def visa_investeringsforslag(df, valutakurs, max_portfoljandel, max_hogriskandel):
    st.subheader("📈 Investeringsförslag")

    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=10000.0, step=100.0, key="kapital")
    kapital_usd = kapital_sek / valutakurs if valutakurs > 0 else 0

    # Uppdatera portföljvärde och andelar
    df["Värde (SEK)"] = df["Antal aktier"] * df["Aktuell kurs"] * valutakurs
    totalvarde = df["Värde (SEK)"].sum() if not df.empty else 0
    df["Portföljandel (%)"] = round(df["Värde (SEK)"] / totalvarde * 100, 2) if totalvarde > 0 else 0

    # Ombalanseringssektioner
    minska = df[df["Portföljandel (%)"] > max_portfoljandel]
    öka = df[(df["Riktkurs om 1 år"] > df["Aktuell kurs"]) & (df["Portföljandel (%)"] < max_portfoljandel)]
    högrisk = df[(df["Omsättning idag"] < 1000) & (df["Portföljandel (%)"] > max_hogriskandel)]

    if not minska.empty:
        st.write("🔻 **Bolag att minska i:**")
        st.dataframe(minska[["Ticker", "Portföljandel (%)", "Värde (SEK)"]])

    if not öka.empty:
        st.write("🔼 **Bolag att öka i:**")
        st.dataframe(öka[["Ticker", "Uppsidepotential (%)", "Portföljandel (%)"]])

    if not högrisk.empty:
        st.write("⚠️ **Högriskvarning:**")
        st.dataframe(högrisk[["Ticker", "Omsättning idag", "Portföljandel (%)"]])

    # Visa investeringsförslag ett i taget med bläddring
    if "förslag_index" not in st.session_state:
        st.session_state["förslag_index"] = 0

    kandidater = öka.sort_values(by="Uppsidepotential (%)", ascending=False).reset_index(drop=True)

    if kandidater.empty:
        st.info("Inga köpförslag baserat på kriterier och kapital.")
        return

    i = st.session_state["förslag_index"]
    if i >= len(kandidater):
        st.info("Inga fler förslag. Starta om för att bläddra igen.")
        if st.button("Starta om förslag"):
            st.session_state["förslag_index"] = 0
        return

    rad = kandidater.iloc[i]
    pris = rad["Aktuell kurs"]
    antal = int(kapital_usd // pris) if pris > 0 else 0
    kostnad_sek = round(antal * pris * valutakurs, 2)

    st.markdown(f"Köp **{antal} st {rad['Ticker']}** för ca **{kostnad_sek} SEK**")
    st.markdown(f"Potential: {round(rad['Uppsidepotential (%)'], 2)} % | Riktkurs om 1 år: {round(rad['Riktkurs om 1 år'], 2)} USD")

    if st.button("Nästa förslag"):
        st.session_state["förslag_index"] += 1

# ---------------------------------------
# LÄGG TILL / UPPDATERA BOLAG
# ---------------------------------------

def lagg_till_bolag(df):
    st.subheader("➕ Lägg till eller uppdatera bolag")

    alla_bolag = df["Ticker"].tolist() if not df.empty else []
    valt_bolag = st.selectbox("Välj bolag att uppdatera (eller lämna tom för nytt)", [""] + alla_bolag)

    if valt_bolag and valt_bolag in df["Ticker"].values:
        befintlig = df[df["Ticker"] == valt_bolag].iloc[0]
    else:
        befintlig = {}

    kolumner = [
        "Ticker", "Bolagsnamn", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning om 1 år", "Omsättning om 2 år",
        "Aktuell kurs", "Antal aktier"
    ]

    indata = {}
    for kolumn in kolumner:
        standard = befintlig.get(kolumn, 0.0 if kolumn not in ["Ticker", "Bolagsnamn"] else "")
        if kolumn in ["Ticker", "Bolagsnamn"]:
            indata[kolumn] = st.text_input(kolumn, value=str(standard))
        else:
            indata[kolumn] = st.number_input(kolumn, value=float(standard), step=0.01 if kolumn != "Antal aktier" else 1.0)

    if st.button("💾 Spara bolag"):
        ny_rad = {k: (v if k in ["Ticker", "Bolagsnamn"] else float(v)) for k, v in indata.items()}
        df = df[df["Ticker"] != ny_rad["Ticker"]]
        df = pd.concat([df, pd.DataFrame([ny_rad])], ignore_index=True)
        spara_data(df)
        st.success(f"{ny_rad['Ticker']} sparad.")
    return df

# ---------------------------------------
# SIDOPANEL – INSTÄLLNINGAR
# ---------------------------------------

def visa_sidopanel(inställningar):
    st.sidebar.header("⚙️ Inställningar")
    ny_valutakurs = st.sidebar.number_input("Valutakurs (USD till SEK)", value=inställningar["Valutakurs"], step=0.01)
    ny_max_portf = st.sidebar.number_input("Max portföljandel (%)", value=inställningar["Max portföljandel"], step=0.01)
    ny_max_risk = st.sidebar.number_input("Max högriskandel (%)", value=inställningar["Max högriskandel"], step=0.01)

    if st.sidebar.button("💾 Spara inställningar"):
        try:
            sheet = skapa_koppling(SETTINGS_SHEET_NAME)
            sheet.update("B2", [[str(ny_valutakurs).replace(".", ",")]])
            sheet.update("B3", [[str(ny_max_portf).replace(".", ",")]])
            sheet.update("B4", [[str(ny_max_risk).replace(".", ",")]])
            sheet.update("B5", [[datetime.today().strftime("%Y-%m-%d")]])
            st.sidebar.success("Inställningar uppdaterade.")
        except Exception as e:
            st.sidebar.error(f"Fel vid uppdatering av inställningar: {e}")

    return {
        "Valutakurs": ny_valutakurs,
        "Max portföljandel": ny_max_portf,
        "Max högriskandel": ny_max_risk
    }

# ---------------------------------------
# HUVUDFUNKTION & MENYVAL
# ---------------------------------------

def main():
    st.title("📈 Aktieanalys & investeringsförslag – Manuell valutakurs och aktiekurs")

    df = hamta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)

    inställningar = las_inställningar()
    # Visa sidopanel och låt användaren eventuellt ändra inställningar
    uppdaterade_inställningar = visa_sidopanel(inställningar)

    meny = st.sidebar.radio("Navigera", ["Analys", "Lägg till / uppdatera bolag", "Investeringsförslag", "Portfölj"])

    if meny == "Analys":
        df = uppdatera_berakningar(df)
        st.dataframe(df, use_container_width=True)

    elif meny == "Lägg till / uppdatera bolag":
        df = lagg_till_bolag(df)

    elif meny == "Investeringsförslag":
        df = uppdatera_berakningar(df)
        visa_investeringsforslag(
            df,
            uppdaterade_inställningar["Valutakurs"],
            uppdaterade_inställningar["Max portföljandel"],
            uppdaterade_inställningar["Max högriskandel"],
        )

    elif meny == "Portfölj":
        visa_portfolj(df, uppdaterade_inställningar["Valutakurs"])

    spara_data(df)

if __name__ == "__main__":
    main()
