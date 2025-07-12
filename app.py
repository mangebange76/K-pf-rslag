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

@st.cache_data(ttl=60)
def hamta_data_cached(sheet_url, sheet_name, credentials, scope):
    creds = Credentials.from_service_account_info(credentials, scopes=scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_url(sheet_url).worksheet(sheet_name)
    data = sheet.get_all_records()
    return pd.DataFrame(data)

def skapa_koppling(blad_namn=SHEET_NAME):
    return client.open_by_url(SHEET_URL).worksheet(blad_namn)

def spara_data(df):
    sheet = skapa_koppling()
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.astype(str).values.tolist())

# ...resterande kod fortsätter...

# ---------------------------------------
# Databasstruktur – säkerställ kolumner
# ---------------------------------------

GODKÄNDA_KOLUMNER = [
    "Ticker", "Bolagsnamn", "Aktuell kurs", "Utestående aktier",
    "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "Omsättning idag", "Omsättning om 1 år", "Omsättning om 2 år",
    "P/S-snitt", "Riktkurs nu", "Riktkurs om 1 år", "Riktkurs om 2 år",
    "Uppsidepotential (%)", "Antal aktier"
]

def säkerställ_kolumner(df):
    df = df[[col for col in df.columns if col in GODKÄNDA_KOLUMNER]]
    for kolumn in GODKÄNDA_KOLUMNER:
        if kolumn not in df.columns:
            if kolumn in ["Ticker", "Bolagsnamn"]:
                df[kolumn] = ""
            else:
                df[kolumn] = 0.0
    return df[GODKÄNDA_KOLUMNER]

def konvertera_typer(df):
    for kolumn in df.columns:
        if kolumn not in ["Ticker", "Bolagsnamn"]:
            df[kolumn] = pd.to_numeric(df[kolumn], errors="coerce").fillna(0.0)
    return df

# ---------------------------------------
# Inställningar från Google Sheets
# ---------------------------------------

INSTÄLLNINGAR_BLAD = "Inställningar"

def las_inställningar():
    try:
        sheet = skapa_koppling(INSTÄLLNINGAR_BLAD)
        data = sheet.get_all_values()
        df = pd.DataFrame(data[1:], columns=data[0])
        inst = dict(zip(df["Inställning"], df["Värde"]))

        valutakurs = float(str(inst.get("Valutakurs", "10")).replace(",", "."))
        max_port = float(str(inst.get("Max portföljandel (%)", "100")).replace(",", "."))
        max_risk = float(str(inst.get("Max högriskandel (%)", "100")).replace(",", "."))
        senast = inst.get("Senast ändrad", "")

        return {
            "Valutakurs": valutakurs,
            "Max portföljandel": max_port,
            "Max högriskandel": max_risk,
            "Senast ändrad": senast
        }
    except Exception as e:
        st.error(f"Fel vid läsning av inställningar: {e}")
        return {
            "Valutakurs": 10.0,
            "Max portföljandel": 100,
            "Max högriskandel": 100,
            "Senast ändrad": ""
        }

def spara_inställningar(valutakurs, max_port, max_risk):
    try:
        sheet = skapa_koppling(INSTÄLLNINGAR_BLAD)
        sheet.update("B2", [[str(valutakurs).replace(".", ",")]])
        sheet.update("B3", [[str(max_port).replace(".", ",")]])
        sheet.update("B4", [[str(max_risk).replace(".", ",")]])
        sheet.update("B5", [[datetime.today().strftime("%Y-%m-%d")]])
    except Exception as e:
        st.error(f"Fel vid uppdatering av inställningar: {e}")

# ---------------------------------------
# Sidopanel och bolagsformulär
# ---------------------------------------

def visa_sidopanel(inställningar):
    st.sidebar.header("⚙️ Inställningar")
    valutakurs = st.sidebar.number_input("Valutakurs (USD → SEK)", value=inställningar["Valutakurs"], step=0.01)
    max_port = st.sidebar.number_input("Max portföljandel (%)", value=inställningar["Max portföljandel"], step=0.1)
    max_risk = st.sidebar.number_input("Max högriskandel (%)", value=inställningar["Max högriskandel"], step=0.1)

    if st.sidebar.button("💾 Spara inställningar"):
        spara_inställningar(valutakurs, max_port, max_risk)
        st.sidebar.success("Inställningar uppdaterade.")

def lagg_till_bolag(df):
    st.subheader("➕ Lägg till eller uppdatera bolag")

    tickers = df["Ticker"].tolist()
    valt = st.selectbox("Välj bolag (för att uppdatera)", [""] + tickers)

    if valt:
        bef = df[df["Ticker"] == valt].iloc[0]
    else:
        bef = {}

    def inputfält(namn, typ="str"):
        if "P/S Q" in namn or "Omsättning" in namn or "Aktuell kurs" in namn:
            return st.number_input(namn, value=float(bef.get(namn, 0)), step=0.1)
        elif namn == "Antal aktier":
            return st.number_input(namn, value=float(bef.get(namn, 0)), step=1.0)
        else:
            return st.text_input(namn, value=str(bef.get(namn, "")))

    fält = {
        "Ticker": "str", "Bolagsnamn": "str",
        "P/S Q1": "float", "P/S Q2": "float", "P/S Q3": "float", "P/S Q4": "float",
        "Omsättning idag": "float", "Omsättning om 1 år": "float", "Omsättning om 2 år": "float",
        "Aktuell kurs": "float", "Antal aktier": "float"
    }

    indata = {kol: inputfält(kol, typ) for kol, typ in fält.items()}

    if st.button("💾 Spara bolag"):
        ny_rad = {k: float(v) if fält[k] == "float" else v for k, v in indata.items()}
        df = df[df["Ticker"] != ny_rad["Ticker"]]
        df = pd.concat([df, pd.DataFrame([ny_rad])], ignore_index=True)
        spara_data(df)
        st.success(f"{ny_rad['Ticker']} sparad.")
    return df

def uppdatera_berakningar(df):
    df["P/S-snitt"] = df[["P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"]].replace(0, np.nan).mean(axis=1).fillna(0)
    df["Riktkurs nu"] = round((df["Omsättning idag"] / df["Utestående aktier"]) * df["P/S-snitt"], 2)
    df["Riktkurs om 1 år"] = round((df["Omsättning om 1 år"] / df["Utestående aktier"]) * df["P/S-snitt"], 2)
    df["Riktkurs om 2 år"] = round((df["Omsättning om 2 år"] / df["Utestående aktier"]) * df["P/S-snitt"], 2)
    df["Uppsidepotential (%)"] = round(((df["Riktkurs nu"] - df["Aktuell kurs"]) / df["Aktuell kurs"]) * 100, 2)
    return df

# ---------------------------------------
# Investeringsförslag och ombalansering
# ---------------------------------------

def visa_investeringsforslag(df, valutakurs, max_port, max_risk):
    st.subheader("📈 Investeringsförslag")
    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=10000.0, step=100.0)

    if "förslag_index" not in st.session_state:
        st.session_state["förslag_index"] = 0

    df = df[df["Riktkurs om 1 år"] > df["Aktuell kurs"]].copy()
    df["Potential"] = df["Riktkurs om 1 år"] - df["Aktuell kurs"]
    df = df.sort_values(by="Potential", ascending=False)

    if df.empty:
        st.info("Inga bolag har högre riktkurs än aktuell kurs.")
        return

    kapital_usd = kapital_sek / valutakurs

    # OMBALANSERING
    st.markdown("### ⚖️ Ombalansering")
    df["Värde (SEK)"] = df["Antal aktier"] * df["Aktuell kurs"] * valutakurs
    totalvarde = df["Värde (SEK)"].sum()
    df["Portföljandel (%)"] = round(df["Värde (SEK)"] / totalvarde * 100, 2)

    minska = df[df["Portföljandel (%)"] > max_port]
    öka = df[(df["Riktkurs om 1 år"] > df["Aktuell kurs"]) & (df["Portföljandel (%)"] < max_port)]
    högrisk = df[(df["Omsättning idag"] < 1000) & (df["Portföljandel (%)"] >= max_risk)]

    if not minska.empty:
        st.write("🔻 **Bolag att minska i:**")
        st.dataframe(minska[["Ticker", "Portföljandel (%)", "Värde (SEK)"]])

    if not öka.empty:
        st.write("🔼 **Bolag att öka i:**")
        st.dataframe(öka[["Ticker", "Potential", "Portföljandel (%)"]])

    if not högrisk.empty:
        st.write("⚠️ **Högriskvarning:**")
        st.dataframe(högrisk[["Ticker", "Omsättning idag", "Portföljandel (%)"]])

    # VISNING AV FÖRSLAG
    st.markdown("### 💡 Bästa förslag just nu:")
    i = st.session_state["förslag_index"]

    if i < len(df):
        rad = df.iloc[i]
        antal = int(kapital_usd // rad["Aktuell kurs"])
        kostnad_sek = round(antal * rad["Aktuell kurs"] * valutakurs, 2)
        st.markdown(
            f"Köp **{antal} st {rad['Ticker']} ({rad['Bolagsnamn']})** för ca **{kostnad_sek} SEK**  \n"
            f"Potential: {round(rad['Potential'],2)} USD → Riktkurs om 1 år: {round(rad['Riktkurs om 1 år'],2)} USD"
        )
        if st.button("Nästa förslag"):
            st.session_state["förslag_index"] += 1
    else:
        st.info("Inga fler förslag. Starta om appen för att visa från början.")

# ---------------------------------------
# MAIN
# ---------------------------------------

def main():
    st.title("📊 Aktieanalys & investeringsförslag")

    df = hamta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)

    inställningar = las_inställningar()
    visa_sidopanel(inställningar)

    meny = st.sidebar.radio("Meny", ["Analys", "Lägg till / uppdatera bolag", "Investeringsförslag"])

    if meny == "Analys":
        df = uppdatera_berakningar(df)
        st.dataframe(df, use_container_width=True)

    elif meny == "Lägg till / uppdatera bolag":
        df = lagg_till_bolag(df)

    elif meny == "Investeringsförslag":
        df = uppdatera_berakningar(df)
        visa_investeringsforslag(df, inställningar["Valutakurs"], inställningar["Max portföljandel"], inställningar["Max högriskandel"])

    spara_data(df)

if __name__ == "__main__":
    main()
