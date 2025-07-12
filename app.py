import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime

st.set_page_config(page_title="📈 Aktieanalys & investeringsförslag", layout="wide")

# ---------------------------------------
# KONSTANTER OCH AUTENTISERING
# ---------------------------------------
SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"
SETTINGS_SHEET_NAME = "Inställningar"

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

# ---------------------------------------
# DATAHANTERING
# ---------------------------------------

GODKÄNDA_KOLUMNER = [
    "Ticker", "Bolagsnamn", "Aktuell kurs", "Utestående aktier", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "Omsättning idag", "Omsättning om 1 år", "Omsättning om 2 år",
    "P/S-snitt", "Riktkurs nu", "Riktkurs om 1 år", "Riktkurs om 2 år", "Uppsidepotential (%)", "Antal aktier"
]

def skapa_koppling(blad_namn=SHEET_NAME):
    return client.open_by_url(SHEET_URL).worksheet(blad_namn)

def hamta_data():
    sheet = skapa_koppling()
    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    df = säkerställ_kolumner(df)
    return df

def spara_data(df):
    df = säkerställ_kolumner(df)
    sheet = skapa_koppling()
    sheet.clear()
    sheet.update([df.columns.tolist()] + df.astype(str).values.tolist())

def säkerställ_kolumner(df):
    # Ta bort alla kolumner som inte är godkända
    df = df[[col for col in df.columns if col in GODKÄNDA_KOLUMNER]]
    # Lägg till saknade kolumner med defaultvärde
    for col in GODKÄNDA_KOLUMNER:
        if col not in df.columns:
            if col in ["Ticker", "Bolagsnamn"]:
                df[col] = ""
            else:
                df[col] = 0.0
    # Säkerställ rätt kolumnordning
    return df[GODKÄNDA_KOLUMNER]

# ---------------------------------------
# INSTÄLLNINGAR – Läs & skriv från Google Sheets
# ---------------------------------------

def las_inställningar():
    try:
        sheet = skapa_koppling(SETTINGS_SHEET_NAME)
        data = sheet.get_all_values()
        df = pd.DataFrame(data[1:], columns=data[0])
        inst = dict(zip(df["Inställning"], df["Värde"]))

        valutakurs = float(str(inst.get("Valutakurs", "10")).replace(",", "."))
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
        return {
            "Valutakurs": 10.0,
            "Max portföljandel": 100.0,
            "Max högriskandel": 100.0,
            "Senast ändrad": ""
        }

def skriv_sidopanel(inställningar):
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

# ---------------------------------------
# BERÄKNINGAR
# ---------------------------------------

def uppdatera_berakningar(df):
    for i, row in df.iterrows():
        ps_values = [row["P/S Q1"], row["P/S Q2"], row["P/S Q3"], row["P/S Q4"]]
        ps_values = [ps for ps in ps_values if ps > 0]
        ps_snitt = round(np.mean(ps_values), 2) if ps_values else 0.0
        df.at[i, "P/S-snitt"] = ps_snitt

        if row["Utestående aktier"] > 0 and ps_snitt > 0:
            riktkurser = []
            for j, kol in enumerate(["Omsättning idag", "Omsättning om 1 år", "Omsättning om 2 år"]):
                rikt = (row[kol] / row["Utestående aktier"]) * ps_snitt
                riktkurser.append(round(rikt, 2))
            df.at[i, "Riktkurs nu"] = riktkurser[0]
            df.at[i, "Riktkurs om 1 år"] = riktkurser[1]
            df.at[i, "Riktkurs om 2 år"] = riktkurser[2]
        else:
            df.at[i, "Riktkurs nu"] = 0.0
            df.at[i, "Riktkurs om 1 år"] = 0.0
            df.at[i, "Riktkurs om 2 år"] = 0.0

        if row["Aktuell kurs"] > 0:
            df.at[i, "Uppsidepotential (%)"] = round(((df.at[i, "Riktkurs nu"] - row["Aktuell kurs"]) / row["Aktuell kurs"]) * 100, 2)
        else:
            df.at[i, "Uppsidepotential (%)"] = 0.0

    return df

# ---------------------------------------
# Rensning & kontroll av kolumner
# ---------------------------------------

def säkerställ_kolumner(df):
    godkända_kolumner = [
        "Ticker", "Bolagsnamn", "Aktuell kurs", "Utestående aktier",
        "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning om 1 år", "Omsättning om 2 år",
        "P/S-snitt", "Riktkurs nu", "Riktkurs om 1 år", "Riktkurs om 2 år",
        "Uppsidepotential (%)", "Antal aktier"
    ]

    # Ta bort kolumner som inte ska vara med
    df = df[[col for col in df.columns if col in godkända_kolumner]]

    # Lägg till saknade kolumner
    for kol in godkända_kolumner:
        if kol not in df.columns:
            if "kurs" in kol.lower() or "omsättning" in kol.lower() or "p/s" in kol.lower():
                df[kol] = 0.0
            elif "andel" in kol.lower():
                df[kol] = 0.0
            else:
                df[kol] = ""

    return df

# ---------------------------------------
# INVESTERINGSFÖRSLAG & OMBALANSERING
# ---------------------------------------

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

    # OMBALANSERING – Sektioner
    st.markdown("### ⚖️ Ombalansering")
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

    # Visa ett förslag i taget
    st.markdown("### 💡 Bästa investeringsförslag just nu:")
    i = st.session_state["förslag_index"]

    if i < len(df):
        rad = df.iloc[i]
        antal = int(kapital_usd // rad["Aktuell kurs"])
        kostnad_sek = round(antal * rad["Aktuell kurs"] * valutakurs, 2)
        st.markdown(
            f"Köp **{antal} st {rad['Ticker']} ({rad['Bolagsnamn']})** för ca **{kostnad_sek} SEK**\n\n"
            f"Potential: {round(rad['Potential'],2)} USD → Riktkurs om 1 år: {round(rad['Riktkurs om 1 år'], 2)} USD"
        )
        if st.button("Nästa förslag"):
            st.session_state["förslag_index"] += 1
    else:
        st.info("Inga fler förslag. Starta om för att se från början.")

# ---------------------------------------
# LÄGG TILL / UPPDATERA BOLAG
# ---------------------------------------

def lagg_till_bolag(df):
    st.subheader("➕ Lägg till eller uppdatera bolag")

    alla_bolag = df["Ticker"].tolist()
    valt_bolag = st.selectbox("Välj bolag att uppdatera (eller lämna tom för nytt)", [""] + alla_bolag)

    if valt_bolag and valt_bolag in df["Ticker"].values:
        befintlig = df[df["Ticker"] == valt_bolag].iloc[0]
    else:
        befintlig = {}

    kolumner = [
        "Ticker", "Bolagsnamn", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning om 1 år", "Omsättning om 2 år",
        "Aktuell kurs", "Utestående aktier", "Antal aktier"
    ]
    indata = {}
    for kolumn in kolumner:
        standard = befintlig.get(kolumn, 0.0 if "P/S" in kolumn or "Omsättning" in kolumn or "kurs" in kolumn or "aktier" in kolumn.lower() else "")
        if kolumn in ["Aktuell kurs", "Omsättning idag", "Omsättning om 1 år", "Omsättning om 2 år", "Utestående aktier", "Antal aktier", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"]:
            indata[kolumn] = st.number_input(kolumn, value=float(standard), step=0.01)
        else:
            indata[kolumn] = st.text_input(kolumn, value=str(standard))

    if st.button("💾 Spara bolag"):
        ny_rad = {k: float(v) if k not in ["Bolagsnamn", "Ticker"] else v for k, v in indata.items()}
        for kol in ["P/S-snitt", "Riktkurs nu", "Riktkurs om 1 år", "Riktkurs om 2 år", "Uppsidepotential (%)"]:
            ny_rad[kol] = 0.0
        df = df[df["Ticker"] != ny_rad["Ticker"]]
        df = pd.concat([df, pd.DataFrame([ny_rad])], ignore_index=True)
        spara_data(df)
        st.success(f"{ny_rad['Ticker']} sparad.")
    return df

# ---------------------------------------
# HJÄLPFUNKTIONER & MAIN
# ---------------------------------------

def visa_tabellrubrik(titel):
    st.markdown(f"<h4 style='margin-top:20px'>{titel}</h4>", unsafe_allow_html=True)


def main():
    st.title("📈 Aktieanalys & investeringsförslag – Manuell valutakurs och aktiekurs")

    df = hamta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)

    inställningar = las_inställningar()
    skriv_sidopanel(inställningar)

    meny = st.sidebar.radio("Navigera", ["Analys", "Lägg till / uppdatera bolag", "Investeringsförslag"])

    if meny == "Analys":
        df = uppdatera_berakningar(df)
        st.dataframe(df, use_container_width=True)

    elif meny == "Lägg till / uppdatera bolag":
        df = lagg_till_bolag(df)

    elif meny == "Investeringsförslag":
        df = uppdatera_berakningar(df)
        visa_investeringsforslag(
            df,
            inställningar["Valutakurs"],
            inställningar["Max portföljandel"],
            inställningar["Max högriskandel"]
        )

    spara_data(df)


if __name__ == "__main__":
    main()
