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

def säkerställ_kolumner(df):
    nödvändiga = [
        "Ticker", "Bolagsnamn", "Aktuell kurs", "Utestående aktier",
        "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning om 1 år", "Omsättning om 2 år",
        "P/S-snitt", "Riktkurs nu", "Riktkurs om 1 år", "Riktkurs om 2 år",
        "Antal aktier"
    ]
    # Ta bort alla kolumner som inte finns med i listan
    df = df[[col for col in df.columns if col in nödvändiga] + [col for col in nödvändiga if col not in df.columns]]

    for kol in nödvändiga:
        if kol not in df.columns:
            df[kol] = 0.0 if any(x in kol.lower() for x in ["kurs", "omsättning", "p/s"]) else ""

    # Säkerställ ordning på kolumner
    df = df[nödvändiga]
    return df

def konvertera_typer(df):
    kolumner = [
        "Omsättning idag", "Omsättning om 1 år", "Omsättning om 2 år",
        "Utestående aktier", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Aktuell kurs", "Antal aktier"
    ]
    for kol in kolumner:
        if kol in df.columns:
            df[kol] = pd.to_numeric(df[kol], errors="coerce").fillna(0.0)
    return df

def las_inställningar():
    try:
        sheet = skapa_koppling(SETTINGS_SHEET_NAME)
        data = sheet.get_all_values()
        df = pd.DataFrame(data[1:], columns=data[0])
        inst = dict(zip(df["Inställning"], df["Värde"]))

        # Omvandla till rätt typ med punktnotation
        valutakurs = float(str(inst.get("Valutakurs", "10")).replace(",", "."))
        max_portf = float(str(inst.get("Max portföljandel", "20")).replace(",", "."))
        max_risk = float(str(inst.get("Max högriskandel", "2")).replace(",", "."))

        return {
            "Valutakurs": valutakurs,
            "Max portföljandel": max_portf,
            "Max högriskandel": max_risk,
            "Senast ändrad": inst.get("Senast ändrad", "")
        }
    except Exception as e:
        st.error(f"Fel vid läsning av inställningar: {e}")
        # Defaultvärden vid fel
        return {"Valutakurs": 10.0, "Max portföljandel": 20.0, "Max högriskandel": 2.0, "Senast ändrad": ""}

def spara_inställningar(valutakurs, max_portf, max_risk):
    try:
        sheet = skapa_koppling(SETTINGS_SHEET_NAME)
        sheet.update("B2", [[str(valutakurs).replace(".", ",")]])
        sheet.update("B3", [[str(max_portf).replace(".", ",")]])
        sheet.update("B4", [[str(max_risk).replace(".", ",")]])
        sheet.update("B5", [[datetime.today().strftime("%Y-%m-%d")]])
    except Exception as e:
        st.sidebar.error(f"Fel vid uppdatering av inställningar: {e}")

# ---------------------------------------
# BERÄKNINGAR
# ---------------------------------------

def uppdatera_berakningar(df):
    ps_cols = ["P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"]
    df["P/S-snitt"] = df[ps_cols].replace(0, np.nan).mean(axis=1).fillna(0)

    # Beräkna riktkurser
    for idx, row in df.iterrows():
        ps_snitt = row["P/S-snitt"]
        utestående = row["Utestående aktier"]
        if utestående > 0 and ps_snitt > 0:
            df.at[idx, "Riktkurs nu"] = round((row["Omsättning idag"] * ps_snitt) / utestående, 2)
            df.at[idx, "Riktkurs om 1 år"] = round((row["Omsättning om 1 år"] * ps_snitt) / utestående, 2)
            df.at[idx, "Riktkurs om 2 år"] = round((row["Omsättning om 2 år"] * ps_snitt) / utestående, 2)
        else:
            df.at[idx, "Riktkurs nu"] = 0.0
            df.at[idx, "Riktkurs om 1 år"] = 0.0
            df.at[idx, "Riktkurs om 2 år"] = 0.0

    # Uppsidepotential i %
    df["Uppsidepotential (%)"] = np.where(
        df["Aktuell kurs"] > 0,
        round(((df["Riktkurs nu"] - df["Aktuell kurs"]) / df["Aktuell kurs"]) * 100, 2),
        0.0
    )
    return df

# ---------------------------------------
# INVESTERINGSFÖRSLAG & OMBALANSERING
# ---------------------------------------

def visa_investeringsforslag(df, valutakurs, max_portfoljandel, max_hogriskandel):
    st.subheader("📈 Investeringsförslag")
    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=10000.0, step=100.0, key="kapital")

    if "förslag_index" not in st.session_state:
        st.session_state["förslag_index"] = 0

    # Filter för investeringsmöjligheter: riktkurs om 1 år > aktuell kurs
    investerings_df = df[df["Riktkurs om 1 år"] > df["Aktuell kurs"]].copy()
    investerings_df["Potential"] = investerings_df["Riktkurs om 1 år"] - investerings_df["Aktuell kurs"]
    investerings_df = investerings_df.sort_values(by="Potential", ascending=False)

    if investerings_df.empty:
        st.info("Inga bolag har högre riktkurs än aktuell kurs.")
        return

    kapital_usd = kapital_sek / valutakurs

    # Ombalanseringssektioner
    st.markdown("### ⚖️ Ombalansering")
    df["Värde (SEK)"] = df["Antal aktier"] * df["Aktuell kurs"] * valutakurs
    totalvarde = df["Värde (SEK)"].sum()
    if totalvarde > 0:
        df["Portföljandel (%)"] = round(df["Värde (SEK)"] / totalvarde * 100, 2)
    else:
        df["Portföljandel (%)"] = 0.0

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

    # Visa investeringsförslag ett i taget
    st.markdown("### 💡 Bästa investeringsförslag just nu:")
    i = st.session_state["förslag_index"]

    if i < len(investerings_df):
        rad = investerings_df.iloc[i]
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
        "Aktuell kurs", "Antal aktier"
    ]
    indata = {}
    for kolumn in kolumner:
        standard = befintlig.get(kolumn, 0.0 if kolumn != "Bolagsnamn" and kolumn != "Ticker" else "")
        if kolumn == "Antal aktier":
            indata[kolumn] = st.number_input(kolumn, value=float(standard), step=1.0)
        elif kolumn == "Aktuell kurs":
            indata[kolumn] = st.number_input(kolumn, value=float(standard), step=0.01)
        elif kolumn.startswith("Omsättning") or kolumn.startswith("P/S"):
            indata[kolumn] = st.number_input(kolumn, value=float(standard), step=0.01)
        else:
            indata[kolumn] = st.text_input(kolumn, value=str(standard))

    if st.button("💾 Spara bolag"):
        ny_rad = {k: float(v) if k not in ["Bolagsnamn", "Ticker"] else v for k, v in indata.items()}
        df = df[df["Ticker"] != ny_rad["Ticker"]]
        df = pd.concat([df, pd.DataFrame([ny_rad])], ignore_index=True)
        spara_data(df)
        st.success(f"{ny_rad['Ticker']} sparad.")
    return df

# ---------------------------------------
# PORTFÖLJVISNING
# ---------------------------------------

def visa_portfolj(df, valutakurs):
    st.subheader("📦 Min portfölj")
    df = df[df["Antal aktier"] > 0].copy()
    if df.empty:
        st.info("Du äger inga aktier.")
        return
    df["Värde (SEK)"] = df["Antal aktier"] * df["Aktuell kurs"] * valutakurs
    total_varde = df["Värde (SEK)"].sum()
    if total_varde > 0:
        df["Andel (%)"] = round(df["Värde (SEK)"] / total_varde * 100, 2)
    else:
        df["Andel (%)"] = 0.0
    st.dataframe(df[["Ticker", "Bolagsnamn", "Antal aktier", "Aktuell kurs", "Värde (SEK)", "Andel (%)"]], use_container_width=True)

# ---------------------------------------
# SIDOPANELENS INSTÄLLNINGAR
# ---------------------------------------

def visa_sidopanel(inställningar):
    st.sidebar.header("⚙️ Inställningar")

    valutakurs = st.sidebar.number_input("USD/SEK", value=inställningar["Valutakurs"], step=0.01)
    max_andel = st.sidebar.number_input("Max portföljandel (%)", value=inställningar["Max portföljandel"], step=0.01)
    max_risk = st.sidebar.number_input("Max högriskandel (%)", value=inställningar["Max högriskandel"], step=0.01)

    if st.sidebar.button("💾 Spara inställningar"):
        spara_inställningar(valutakurs, max_andel, max_risk)
        st.sidebar.success("Inställningar uppdaterade.")
    return valutakurs, max_andel, max_risk

# ---------------------------------------
# MAIN
# ---------------------------------------

def main():
    st.title("📈 Aktieanalys & investeringsförslag – Manuell valutakurs och aktiekurs")

    df = hamta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)
    df = uppdatera_berakningar(df)

    inställningar = las_inställningar()
    valutakurs, max_portf, max_risk = visa_sidopanel(inställningar)

    meny = st.sidebar.radio("Navigera", ["Analys", "Lägg till / uppdatera bolag", "Investeringsförslag", "Portfölj"])

    if meny == "Analys":
        st.dataframe(df, use_container_width=True)
    elif meny == "Lägg till / uppdatera bolag":
        df = lagg_till_bolag(df)
        spara_data(df)
    elif meny == "Investeringsförslag":
        visa_investeringsforslag(df, valutakurs, max_portf, max_risk)
    elif meny == "Portfölj":
        visa_portfolj(df, valutakurs)

    # Spara alltid datan
    spara_data(df)

if __name__ == "__main__":
    main()
