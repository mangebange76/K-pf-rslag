import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime

st.set_page_config(page_title="Aktieanalys & investeringsförslag", layout="wide")

# -----------------------------
# KONSTANTER OCH SHEET-INFO
# -----------------------------
SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"
SETTINGS_SHEET_NAME = "Inställningar"

RÄTT_KOLUMNER = [
    "Ticker", "Bolagsnamn", "Aktuell kurs", "Utestående aktier",
    "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "Omsättning idag", "Omsättning om 1 år", "Omsättning om 2 år",
    "P/S-snitt", "Riktkurs nu", "Riktkurs om 1 år", "Riktkurs om 2 år",
    "Antal aktier", "Uppsidepotential (%)"
]

# -----------------------------
# AUTENTISERING TILL GOOGLE SHEETS
# -----------------------------
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

def skapa_koppling(blad_namn=SHEET_NAME):
    return client.open_by_url(SHEET_URL).worksheet(blad_namn)

# -----------------------------
# DATAHANTERING
# -----------------------------

def hamta_data():
    try:
        df = pd.DataFrame(skapa_koppling().get_all_records())
        df = df[[kol for kol in df.columns if kol in RÄTT_KOLUMNER]]
        for kol in RÄTT_KOLUMNER:
            if kol not in df.columns:
                df[kol] = 0.0 if "kurs" in kol.lower() or "omsättning" in kol.lower() or "P/S" in kol else ""
        return df[RÄTT_KOLUMNER]
    except:
        return pd.DataFrame(columns=RÄTT_KOLUMNER)

def spara_data(df):
    df = df[RÄTT_KOLUMNER]
    df = df.fillna("").astype(str)
    sheet = skapa_koppling()
    sheet.clear()
    sheet.update([df.columns.tolist()] + df.values.tolist())

# -----------------------------
# INSTÄLLNINGAR – LÄSA OCH SPARA
# -----------------------------

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

def spara_inställningar(valutakurs, max_portf, max_risk):
    try:
        sheet = skapa_koppling(SETTINGS_SHEET_NAME)
        sheet.update("B2", [[str(valutakurs).replace(".", ",")]])
        sheet.update("B3", [[str(max_portf).replace(".", ",")]])
        sheet.update("B4", [[str(max_risk).replace(".", ",")]])
        sheet.update("B5", [[datetime.today().strftime("%Y-%m-%d")]])
    except Exception as e:
        st.error(f"Fel vid uppdatering av inställningar: {e}")

# -----------------------------
# BERÄKNINGAR
# -----------------------------

def uppdatera_berakningar(df):
    df["P/S Q1"] = pd.to_numeric(df["P/S Q1"], errors="coerce")
    df["P/S Q2"] = pd.to_numeric(df["P/S Q2"], errors="coerce")
    df["P/S Q3"] = pd.to_numeric(df["P/S Q3"], errors="coerce")
    df["P/S Q4"] = pd.to_numeric(df["P/S Q4"], errors="coerce")

    df["P/S-snitt"] = df[["P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"]].replace(0, np.nan).mean(axis=1).fillna(0)

    df["Riktkurs nu"] = round((df["Omsättning idag"] / df["Utestående aktier"]) * df["P/S-snitt"], 2)
    df["Riktkurs om 1 år"] = round((df["Omsättning om 1 år"] / df["Utestående aktier"]) * df["P/S-snitt"], 2)
    df["Riktkurs om 2 år"] = round((df["Omsättning om 2 år"] / df["Utestående aktier"]) * df["P/S-snitt"], 2)

    df["Uppsidepotential (%)"] = round(((df["Riktkurs nu"] - df["Aktuell kurs"]) / df["Aktuell kurs"]) * 100, 2)
    return df

def visa_investeringsforslag(df, valutakurs, max_portfoljandel, max_hogriskandel):
    st.subheader("📈 Investeringsförslag")
    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=10000.0, step=100.0, key="kapital")

    if "förslag_index" not in st.session_state:
        st.session_state["förslag_index"] = 0

    # Filtrera bolag med positiv uppsidepotential
    df_filtered = df[df["Riktkurs om 1 år"] > df["Aktuell kurs"]].copy()
    df_filtered["Potential"] = df_filtered["Riktkurs om 1 år"] - df_filtered["Aktuell kurs"]
    df_filtered = df_filtered.sort_values(by="Potential", ascending=False)

    if df_filtered.empty:
        st.info("Inga bolag med positiv uppsidepotential just nu.")
        return

    kapital_usd = kapital_sek / valutakurs

    # Ombalanseringssektioner
    st.markdown("### ⚖️ Ombalansering")
    df["Värde (SEK)"] = df["Antal aktier"] * df["Aktuell kurs"] * valutakurs
    totalvarde = df["Värde (SEK)"].sum()
    df["Portföljandel (%)"] = round(df["Värde (SEK)"] / totalvarde * 100, 2) if totalvarde > 0 else 0

    minska = df[df["Portföljandel (%)"] > max_portfoljandel]
    öka = df_filtered[df_filtered["Ticker"].isin(df["Ticker"]) & (df["Portföljandel (%)"] < max_portfoljandel)]
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

    # Visa investeringsförslag en i taget
    st.markdown("### 💡 Bästa investeringsförslag just nu:")
    i = st.session_state["förslag_index"]

    if i < len(df_filtered):
        rad = df_filtered.iloc[i]
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
        if st.button("Starta om förslag"):
            st.session_state["förslag_index"] = 0

def lagg_till_bolag(df):
    st.subheader("➕ Lägg till eller uppdatera bolag")

    alla_bolag = df["Ticker"].tolist()
    valt_bolag = st.selectbox("Välj bolag att uppdatera (eller lämna tom för nytt)", [""] + sorted(alla_bolag))

    if valt_bolag and valt_bolag in df["Ticker"].values:
        befintlig = df[df["Ticker"] == valt_bolag].iloc[0]
    else:
        befintlig = {}

    with st.form("form_lagg_till_bolag"):
        ticker = st.text_input("Ticker", value=befintlig.get("Ticker", "")).upper()
        namn = st.text_input("Bolagsnamn", value=befintlig.get("Bolagsnamn", ""))
        aktuell_kurs = st.number_input("Aktuell kurs (USD)", value=float(befintlig.get("Aktuell kurs", 0.0)), step=0.01, format="%.2f")
        utestaende_aktier = st.number_input("Utestående aktier (miljoner)", value=float(befintlig.get("Utestående aktier", 0.0)), step=0.01)
        antal_aktier = st.number_input("Antal aktier du äger", value=float(befintlig.get("Antal aktier", 0.0)), step=1.0)

        ps_q1 = st.number_input("P/S Q1", value=float(befintlig.get("P/S Q1", 0.0)), step=0.01)
        ps_q2 = st.number_input("P/S Q2", value=float(befintlig.get("P/S Q2", 0.0)), step=0.01)
        ps_q3 = st.number_input("P/S Q3", value=float(befintlig.get("P/S Q3", 0.0)), step=0.01)
        ps_q4 = st.number_input("P/S Q4", value=float(befintlig.get("P/S Q4", 0.0)), step=0.01)

        oms_idag = st.number_input("Omsättning idag (miljoner USD)", value=float(befintlig.get("Omsättning idag", 0.0)), step=0.01)
        oms_1 = st.number_input("Omsättning om 1 år", value=float(befintlig.get("Omsättning om 1 år", 0.0)), step=0.01)
        oms_2 = st.number_input("Omsättning om 2 år", value=float(befintlig.get("Omsättning om 2 år", 0.0)), step=0.01)

        sparaknapp = st.form_submit_button("💾 Spara")

    if sparaknapp:
        ny_rad = {
            "Ticker": ticker,
            "Bolagsnamn": namn,
            "Aktuell kurs": aktuell_kurs,
            "Utestående aktier": utestaende_aktier,
            "Antal aktier": antal_aktier,
            "P/S Q1": ps_q1,
            "P/S Q2": ps_q2,
            "P/S Q3": ps_q3,
            "P/S Q4": ps_q4,
            "Omsättning idag": oms_idag,
            "Omsättning om 1 år": oms_1,
            "Omsättning om 2 år": oms_2
        }

        df = df[df["Ticker"] != ticker]
        df = pd.concat([df, pd.DataFrame([ny_rad])], ignore_index=True)
        spara_data(df)
        st.success(f"{ticker} sparad/uppdaterad.")
    return df

def visa_sidopanel(inställningar):
    st.sidebar.header("⚙️ Inställningar")

    ny_valutakurs = st.sidebar.number_input("Valutakurs (USD → SEK)", value=inställningar["Valutakurs"], step=0.01)
    ny_max_portf = st.sidebar.number_input("Max portföljandel (%)", value=inställningar["Max portföljandel"], step=0.01)
    ny_max_risk = st.sidebar.number_input("Max högriskandel (%)", value=inställningar["Max högriskandel"], step=0.01)

    if st.sidebar.button("💾 Spara inställningar"):
        try:
            sheet = skapa_koppling(SETTINGS_SHEET_NAME)
            sheet.update("B2", [[str(ny_valutakurs).replace('.', ',')]])
            sheet.update("B3", [[str(ny_max_portf).replace('.', ',')]])
            sheet.update("B4", [[str(ny_max_risk).replace('.', ',')]])
            sheet.update("B5", [[datetime.today().strftime("%Y-%m-%d")]])
            st.sidebar.success("Inställningar sparade!")
        except Exception as e:
            st.sidebar.error(f"Fel vid uppdatering av inställningar: {e}")

def main():
    st.title("📈 Aktieanalys & Investeringsförslag – Manuell valutakurs och aktiekurs")

    df = hamta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)

    inställningar = las_inställningar()
    visa_sidopanel(inställningar)

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
            inställningar["Valutakurs"],
            inställningar["Max portföljandel"],
            inställningar["Max högriskandel"]
        )

    elif meny == "Portfölj":
        visa_portfolj(df, inställningar["Valutakurs"])

    spara_data(df)

if __name__ == "__main__":
    main()

def uppdatera_berakningar(df):
    # Beräkna P/S-snitt utifrån P/S Q1-Q4
    for i, row in df.iterrows():
        ps_values = [row.get("P/S Q1", 0), row.get("P/S Q2", 0), row.get("P/S Q3", 0), row.get("P/S Q4", 0)]
        ps_values = [v for v in ps_values if v > 0]
        ps_snitt = round(np.mean(ps_values), 2) if ps_values else 0.0
        df.at[i, "P/S-snitt"] = ps_snitt

        # Beräkna riktkurser enligt omsättning * P/S-snitt / utestående aktier
        utestående = row.get("Utestående aktier", 0)
        if utestående > 0:
            df.at[i, "Riktkurs nu"] = round((row.get("Omsättning idag", 0) * ps_snitt) / utestående, 2)
            df.at[i, "Riktkurs om 1 år"] = round((row.get("Omsättning om 1 år", 0) * ps_snitt) / utestående, 2)
            df.at[i, "Riktkurs om 2 år"] = round((row.get("Omsättning om 2 år", 0) * ps_snitt) / utestående, 2)
        else:
            df.at[i, "Riktkurs nu"] = 0.0
            df.at[i, "Riktkurs om 1 år"] = 0.0
            df.at[i, "Riktkurs om 2 år"] = 0.0

    # Uppsidepotential i procent baserat på riktkurs nu
    df["Uppsidepotential (%)"] = round(((df["Riktkurs nu"] - df["Aktuell kurs"]) / df["Aktuell kurs"]) * 100, 2)
    df["Uppsidepotential (%)"] = df["Uppsidepotential (%)"].fillna(0)
    return df


def visa_investeringsforslag(df, valutakurs, max_portfoljandel, max_hogriskandel):
    st.subheader("📈 Investeringsförslag")

    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=10000.0, step=100.0, key="kapital")

    if "förslag_index" not in st.session_state:
        st.session_state["förslag_index"] = 0

    # Filtrera bolag med potential
    kandidater = df[df["Riktkurs om 1 år"] > df["Aktuell kurs"]].copy()
    kandidater["Potential"] = kandidater["Riktkurs om 1 år"] - kandidater["Aktuell kurs"]
    kandidater = kandidater.sort_values(by="Potential", ascending=False)

    if kandidater.empty:
        st.info("Inga bolag har högre riktkurs än aktuell kurs.")
        return

    kapital_usd = kapital_sek / valutakurs

    # Ombalanseringssektioner
    st.markdown("### ⚖️ Ombalanseringsförslag")

    kandidater["Värde (SEK)"] = kandidater["Antal aktier"] * kandidater["Aktuell kurs"] * valutakurs
    totalvarde = kandidater["Värde (SEK)"].sum()
    kandidater["Portföljandel (%)"] = round(kandidater["Värde (SEK)"] / totalvarde * 100, 2)

    minska = kandidater[kandidater["Portföljandel (%)"] > max_portfoljandel]
    öka = kandidater[(kandidater["Riktkurs om 1 år"] > kandidater["Aktuell kurs"]) & (kandidater["Portföljandel (%)"] < max_portfoljandel)]
    högrisk = kandidater[(kandidater["Omsättning idag"] < 1000) & (kandidater["Portföljandel (%)"] > max_hogriskandel)]

    if not minska.empty:
        st.write("🔻 **Bolag att minska i:**")
        st.dataframe(minska[["Ticker", "Portföljandel (%)", "Värde (SEK)"]])

    if not öka.empty:
        st.write("🔼 **Bolag att öka i:**")
        st.dataframe(öka[["Ticker", "Potential", "Portföljandel (%)"]])

    if not högrisk.empty:
        st.write("⚠️ **Högriskvarning:**")
        st.dataframe(högrisk[["Ticker", "Omsättning idag", "Portföljandel (%)"]])

    # Visa ett investeringsförslag i taget med bläddringsknapp
    st.markdown("### 💡 Bästa investeringsförslag just nu:")
    i = st.session_state["förslag_index"]

    if i < len(kandidater):
        rad = kandidater.iloc[i]
        antal = int(kapital_usd // rad["Aktuell kurs"])
        kostnad_sek = round(antal * rad["Aktuell kurs"] * valutakurs, 2)
        st.markdown(
            f"Köp **{antal} st {rad['Ticker']} ({rad['Bolagsnamn']})** för ca **{kostnad_sek} SEK**\n\n"
            f"Potential: {round(rad['Potential'], 2)} USD → Riktkurs om 1 år: {round(rad['Riktkurs om 1 år'], 2)} USD"
        )
        if st.button("Nästa förslag"):
            st.session_state["förslag_index"] += 1
    else:
        st.info("Inga fler förslag. Starta om appen för att bläddra igen.")

def visa_portfolj(df, valutakurs):
    st.subheader("📦 Min portfölj")
    df_port = df[df["Antal aktier"] > 0].copy()
    if df_port.empty:
        st.info("Du äger inga aktier.")
        return
    df_port["Värde (SEK)"] = df_port["Antal aktier"] * df_port["Aktuell kurs"] * valutakurs
    df_port["Andel (%)"] = round(df_port["Värde (SEK)"] / df_port["Värde (SEK)"].sum() * 100, 2)
    st.dataframe(df_port[["Ticker", "Bolagsnamn", "Antal aktier", "Aktuell kurs", "Värde (SEK)", "Andel (%)"]], use_container_width=True)
