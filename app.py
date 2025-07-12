# app.py – Del 1: Setup, autentisering och dataladdning

import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# Google Sheets inställningar
SHEET_URL = "https://docs.google.com/spreadsheets/d/1-5JSJpqBB0j7sm3cgEGZmnFoBL_oJDPMpLdleggL0HQ/edit?usp=drivesdk"
SHEET_NAME = "Blad1"
SETTINGS_SHEET = "Inställningar"

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = ServiceAccountCredentials.from_json_keyfile_dict(
    st.secrets["GOOGLE_CREDENTIALS"], scope)
client = gspread.authorize(credentials)

# Ladda data från Google Sheets
def load_data():
    sheet = client.open_by_url(SHEET_URL).worksheet(SHEET_NAME)
    data = sheet.get_all_records()
    return pd.DataFrame(data)

# Spara data till Google Sheets
def save_data(df):
    sheet = client.open_by_url(SHEET_URL).worksheet(SHEET_NAME)
    df_clean = df.copy().fillna("")
    df_clean = df_clean.astype(str)
    sheet.update([df_clean.columns.values.tolist()] + df_clean.values.tolist())

# Skapa inställningsblad om det inte finns
def skapa_inställningsblad():
    spreadsheet = client.open_by_url(SHEET_URL)
    try:
        spreadsheet.worksheet(SETTINGS_SHEET)
    except gspread.WorksheetNotFound:
        sheet = spreadsheet.add_worksheet(title=SETTINGS_SHEET, rows="10", cols="3")
        sheet.update("A1:C1", [["Namn", "Värde", "Senast ändrad"]])
        default_rows = [
            ["Valutakurs", "9.50", datetime.today().strftime("%Y-%m-%d")],
            ["Max portföljandel (%)", "20", datetime.today().strftime("%Y-%m-%d")],
            ["Max högriskandel (%)", "2", datetime.today().strftime("%Y-%m-%d")]
        ]
        sheet.update("A2:C4", default_rows)

# Ladda inställningar
def load_settings():
    skapa_inställningsblad()
    sheet = client.open_by_url(SHEET_URL).worksheet(SETTINGS_SHEET)
    rows = sheet.get_all_records()
    return {row["Namn"]: float(row["Värde"]) for row in rows}

# Spara inställningar
def save_settings(updated_settings):
    sheet = client.open_by_url(SHEET_URL).worksheet(SETTINGS_SHEET)
    rows = sheet.get_all_records()
    df = pd.DataFrame(rows)
    for key, value in updated_settings.items():
        df.loc[df["Namn"] == key, "Värde"] = value
        df.loc[df["Namn"] == key, "Senast ändrad"] = datetime.today().strftime("%Y-%m-%d")
    sheet.update([df.columns.tolist()] + df.astype(str).values.tolist())

# Starta
df = load_data()
settings = load_settings()

# app.py – Del 2: Investeringsförslag och analysfunktioner

def är_högrisk(row):
    try:
        omsättning = float(row.get("Omsättning (miljoner USD)", 0))
        andel = float(row.get("Andel av portfölj (%)", 0))
        return omsättning < 1000 and andel >= settings["Max högriskandel (%)"]
    except:
        return False

def över_maxandel(row):
    try:
        return float(row.get("Andel av portfölj (%)", 0)) > settings["Max portföljandel (%)"]
    except:
        return False

def investeringsforslag(df, kapital_sek, valutakurs):
    df = df.copy()
    df["Aktuell kurs (USD)"] = df["Aktuell kurs (USD)"].astype(str).str.replace(",", ".").astype(float)
    df["Målkurs 2025"] = df["Målkurs 2025"].astype(str).str.replace(",", ".").astype(float)
    df["Målkurs 2026"] = df["Målkurs 2026"].astype(str).str.replace(",", ".").astype(float)

    df["Uppside (%)"] = round((df["Målkurs 2025"] - df["Aktuell kurs (USD)"]) / df["Aktuell kurs (USD)"] * 100, 2)

    bolag_att_oka = df[
        (df["Uppside (%)"] > 0) &
        (df["Andel av portfölj (%)"].astype(float) < settings["Max portföljandel (%)"])
    ].sort_values(by="Uppside (%)", ascending=False)

    bolag_att_minska = df[df["Andel av portfölj (%)"].astype(float) > settings["Max portföljandel (%)"]]

    högrisk = df[df.apply(är_högrisk, axis=1)]

    resultat = []

    kapital_usd = kapital_sek / valutakurs
    tillgängligt = kapital_usd

    for _, row in bolag_att_oka.iterrows():
        if float(row["Andel av portfölj (%)"]) >= settings["Max portföljandel (%)"]:
            continue
        pris = row["Aktuell kurs (USD)"]
        om pris <= 0:
            continue
        antal = int(tillgängligt // pris)
        if antal == 0:
            break
        investering = antal * pris
        tillgängligt -= investering
        resultat.append({
            "Ticker": row["Ticker"],
            "Föreslaget antal": antal,
            "Kostnad (USD)": round(investering, 2),
            "Kostnad (SEK)": round(investering * valutakurs, 2),
            "Uppside (%)": row["Uppside (%)"]
        })

    return resultat, tillgängligt * valutakurs, bolag_att_minska, högrisk

# app.py – Del 3: Vyer för investeringsförslag och inställningar

def visa_investeringsrad(df, valutakurs):
    st.header("📈 Investeringsförslag")

    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", min_value=0, value=1000)
    if valutakurs == 0:
        st.error("Valutakurs kan inte vara 0.")
        return

    try:
        forslag, rest_sek, minska_df, högrisk_df = investeringsforslag(df, kapital_sek, valutakurs)

        st.subheader("🔻 Bolag att minska i")
        if not minska_df.empty:
            st.dataframe(minska_df[["Ticker", "Andel av portfölj (%)"]])
        else:
            st.write("Inga överexponerade innehav.")

        st.subheader("🔼 Bolag att öka i")
        if forslag:
            st.table(forslag)
            if rest_sek < kapital_sek:
                st.success(f"Restkapital: {round(rest_sek)} SEK")
            else:
                st.warning("Kapitalet räcker inte för något förslag. Öka beloppet för att se investeringar.")
        else:
            st.info("Inga köpförslag baserat på kriterierna och tillgängligt kapital.")

        st.subheader("⚠️ Högriskvarningar")
        if not högrisk_df.empty:
            st.dataframe(högrisk_df[["Ticker", "Omsättning (miljoner USD)", "Andel av portfölj (%)"]])
        else:
            st.write("Inga högriskvarningar.")

    except Exception as e:
        st.error(f"Fel vid generering av förslag: {e}")

def inställningar_panel():
    st.sidebar.header("⚙️ Inställningar")
    max_portföljandel = st.sidebar.number_input(
        "Max portföljandel (%)", min_value=1, max_value=100,
        value=settings["Max portföljandel (%)"]
    )
    max_högriskandel = st.sidebar.number_input(
        "Max andel i högriskbolag (%)", min_value=1, max_value=100,
        value=settings["Max högriskandel (%)"]
    )
    valutakurs = st.sidebar.number_input(
        "Valutakurs USD/SEK", min_value=0.01, value=settings["Valutakurs"], format="%.4f"
    )

    if st.sidebar.button("Spara inställningar"):
        new_settings = {
            "Max portföljandel (%)": max_portföljandel,
            "Max högriskandel (%)": max_högriskandel,
            "Valutakurs": valutakurs
        }
        save_settings(new_settings)
        st.sidebar.success("Inställningar sparade!")

    return valutakurs

# app.py – Del 4: Huvudfunktion, formulär, vyval

def main():
    st.set_page_config(page_title="Köpförslag", layout="wide")
    st.title("📊 Aktieanalys & Investeringsförslag")

    global settings
    settings = load_settings()

    valutakurs = inställningar_panel()

    df = ladda_data()
    df = konvertera_till_ratt_typ(df)

    meny = st.sidebar.radio("Välj vy", ["📋 Lista bolag", "➕ Lägg till/Uppdatera bolag", "💡 Investeringsförslag"])

    if meny == "📋 Lista bolag":
        st.header("📋 Bolagslista")
        if not df.empty:
            st.dataframe(df)
        else:
            st.info("Databasen är tom.")
    elif meny == "➕ Lägg till/Uppdatera bolag":
        st.header("➕ Lägg till eller uppdatera ett bolag")

        alla_bolag = df["Ticker"].tolist()
        valt_bolag = st.selectbox("Välj bolag att uppdatera (eller lämna tom för nytt):", [""] + alla_bolag)

        if valt_bolag and valt_bolag in df["Ticker"].values:
            bolagsdata = df[df["Ticker"] == valt_bolag].iloc[0]
        else:
            bolagsdata = {}

        with st.form("lägg_till_form"):
            ticker = st.text_input("Ticker", value=bolagsdata.get("Ticker", ""))
            namn = st.text_input("Bolagsnamn", value=bolagsdata.get("Bolagsnamn", ""))
            aktuell_kurs = st.number_input("Aktuell kurs (USD)", value=float(bolagsdata.get("Aktuell kurs", 0)), step=0.01, format="%.2f")
            mål_2025 = st.number_input("Målkurs 2025", value=float(bolagsdata.get("Målkurs 2025", 0)), step=0.01)
            mål_2026 = st.number_input("Målkurs 2026", value=float(bolagsdata.get("Målkurs 2026", 0)), step=0.01)
            p_s_q1 = st.number_input("P/S Q1", value=float(bolagsdata.get("P/S Q1", 0)), step=0.01)
            p_s_rullande = st.number_input("P/S rullande", value=float(bolagsdata.get("P/S rullande", 0)), step=0.01)
            omsättning = st.number_input("Omsättning (miljoner USD)", value=float(bolagsdata.get("Omsättning (miljoner USD)", 0)), step=0.01)
            portföljandel = st.number_input("Andel av portfölj (%)", value=float(bolagsdata.get("Andel av portfölj (%)", 0)), step=0.01)
            senast_uppdaterad = datetime.today().strftime("%Y-%m-%d")

            submitted = st.form_submit_button("Spara")

        if submitted:
            ny_rad = {
                "Ticker": ticker,
                "Bolagsnamn": namn,
                "Aktuell kurs": aktuell_kurs,
                "Målkurs 2025": mål_2025,
                "Målkurs 2026": mål_2026,
                "P/S Q1": p_s_q1,
                "P/S rullande": p_s_rullande,
                "Omsättning (miljoner USD)": omsättning,
                "Andel av portfölj (%)": portföljandel,
                "Senast uppdaterad": senast_uppdaterad
            }

            df = df[df["Ticker"] != ticker]  # ta bort ev. gammal
            df = pd.concat([df, pd.DataFrame([ny_rad])], ignore_index=True)
            save_data(df)
            st.success(f"{ticker} har sparats.")

    elif meny == "💡 Investeringsförslag":
        visa_investeringsrad(df, valutakurs)


if __name__ == "__main__":
    main()

# app.py – Del 5: Investeringslogik och hjälpfunktioner

def konvertera_till_ratt_typ(df):
    numeriska_kolumner = [
        "Aktuell kurs", "Målkurs 2025", "Målkurs 2026", "P/S Q1", "P/S rullande",
        "Omsättning (miljoner USD)", "Andel av portfölj (%)"
    ]
    for kolumn in numeriska_kolumner:
        if kolumn in df.columns:
            df[kolumn] = pd.to_numeric(df[kolumn], errors="coerce")
    return df

def investeringsforslag(df, kapital_sek, valutakurs):
    kapital_usd = kapital_sek / valutakurs
    max_andel = settings.get("Max portföljandel (%)", 20)
    max_risk = settings.get("Max högriskandel (%)", 2)

    forslag = []
    högrisk_df = pd.DataFrame()
    minska_df = df[df["Andel av portfölj (%)"] > max_andel]

    kandidater = df[
        (df["Andel av portfölj (%)"] < max_andel) &
        (df["Målkurs 2025"] > df["Aktuell kurs"])
    ].copy()

    kandidater["Potential (%)"] = round(((kandidater["Målkurs 2025"] - kandidater["Aktuell kurs"]) / kandidater["Aktuell kurs"]) * 100, 1)
    kandidater = kandidater.sort_values(by="Potential (%)", ascending=False)

    for _, row in kandidater.iterrows():
        ticker = row["Ticker"]
        kurs = row["Aktuell kurs"]
        omsättning = row["Omsättning (miljoner USD)"]
        andel = row["Andel av portfölj (%)"]

        belopp = kapital_usd
        antal = int(belopp // kurs)

        if antal > 0:
            investering_usd = antal * kurs
            investering_sek = investering_usd * valutakurs
            ny_andel = andel + ((investering_usd / (kapital_usd + 1e-6)) * 100)

            högrisk = omsättning < 1000 and andel >= max_risk
            if högrisk:
                högrisk_df = pd.concat([högrisk_df, pd.DataFrame([row])])

            forslag.append({
                "Ticker": ticker,
                "Antal": antal,
                "Kurs (USD)": round(kurs, 2),
                "Investering (SEK)": round(investering_sek, 2),
                "Potential (%)": row["Potential (%)"]
            })

            kapital_usd -= antal * kurs

    return forslag, kapital_usd * valutakurs, minska_df, högrisk_df

# app.py – Del 6: Autentisering, Google Sheets, inställningar

import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

SHEET_URL = "https://docs.google.com/spreadsheets/d/1-5JSJpqBB0j7sm3cgEGZmnFoBL_oJDPMpLdleggL0HQ/edit?usp=drivesdk"
SHEET_NAME = "Blad1"
SETTINGS_SHEET = "Inställningar"

def load_credentials():
    credentials_dict = st.secrets["norremt"]
    credentials = Credentials.from_service_account_info(credentials_dict)
    client = gspread.authorize(credentials)
    return client

def save_data(df):
    df = df.fillna("")
    df = df.astype(str)
    client = load_credentials()
    sheet = client.open_by_url(SHEET_URL).worksheet(SHEET_NAME)
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.values.tolist())

def load_settings():
    try:
        client = load_credentials()
        sheet = client.open_by_url(SHEET_URL)
        try:
            worksheet = sheet.worksheet(SETTINGS_SHEET)
        except gspread.exceptions.WorksheetNotFound:
            worksheet = sheet.add_worksheet(title=SETTINGS_SHEET, rows="10", cols="3")
            worksheet.update("A1:C1", [["Inställning", "Värde", "Senast ändrad"]])
            worksheet.append_row(["Max portföljandel (%)", "20", str(datetime.today().date())])
            worksheet.append_row(["Max högriskandel (%)", "2", str(datetime.today().date())])
            worksheet.append_row(["Valutakurs", "10", str(datetime.today().date())])
        data = worksheet.get_all_records()
        return {row["Inställning"]: float(row["Värde"]) for row in data if row["Värde"] != ""}
    except Exception as e:
        st.error(f"Kunde inte ladda inställningar: {e}")
        return {}

def save_settings(settings_dict):
    try:
        client = load_credentials()
        sheet = client.open_by_url(SHEET_URL).worksheet(SETTINGS_SHEET)
        existing = sheet.get_all_records()
        existing_dict = {row["Inställning"]: i+2 for i, row in enumerate(existing)}

        for key, value in settings_dict.items():
            rad = existing_dict.get(key)
            if rad:
                sheet.update_cell(rad, 2, str(value))
                sheet.update_cell(rad, 3, str(datetime.today().date()))
            else:
                sheet.append_row([key, str(value), str(datetime.today().date())])
    except Exception as e:
        st.error(f"Kunde inte spara inställningar: {e}")

# app.py – Del 7: Huvudfunktion och UI-menyer

import streamlit as st
import pandas as pd

def main():
    st.set_page_config(page_title="Investeringsanalys", layout="wide")
    st.title("📊 Aktieanalys & Investeringsförslag")

    inställningar = load_settings()
    max_andel = inställningar.get("Max portföljandel (%)", 20)
    max_högrisk = inställningar.get("Max högriskandel (%)", 2)
    valutakurs = inställningar.get("Valutakurs", 10)

    # Sidopanel
    st.sidebar.header("Inställningar")
    valutakurs = st.sidebar.number_input("Valutakurs (USD → SEK)", value=valutakurs, step=0.01)
    max_andel = st.sidebar.number_input("Max portföljandel (%)", value=max_andel, step=1)
    max_högrisk = st.sidebar.number_input("Max högriskandel (%)", value=max_högrisk, step=0.1)

    if st.sidebar.button("💾 Spara inställningar"):
        save_settings({
            "Valutakurs": valutakurs,
            "Max portföljandel (%)": max_andel,
            "Max högriskandel (%)": max_högrisk,
        })
        st.sidebar.success("Inställningar sparade.")

    menyval = st.sidebar.radio("Välj vy", ["📈 Investeringsförslag", "🧮 Lägg till / uppdatera bolag", "📋 Databasen", "⚙️ Inställningar"])

    df = hamta_data()
    df = konvertera_till_ratt_typ(df)

    if menyval == "🧮 Lägg till / uppdatera bolag":
        lagg_till_eller_uppdatera_bolag(df)

    elif menyval == "📈 Investeringsförslag":
        visa_investeringsrad(df, valutakurs)

    elif menyval == "📋 Databasen":
        st.subheader("📄 Bolagsdatabas")
        st.dataframe(df)

    elif menyval == "⚙️ Inställningar":
        st.subheader("⚙️ Nuvarande inställningar")
        st.write(pd.DataFrame.from_dict(inställningar, orient="index", columns=["Värde"]))

if __name__ == "__main__":
    main()
