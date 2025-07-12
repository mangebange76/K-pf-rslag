import streamlit as st
import pandas as pd
import gspread
from datetime import datetime
from google.oauth2.service_account import Credentials

# Google Sheets inställningar
SHEET_URL = "https://docs.google.com/spreadsheets/d/1-5JSJpqBB0j7sm3cgEGZmnFoBL_oJDPMpLdleggL0HQ/edit?usp=drivesdk"
SHEET_NAME = "Blad1"
SETTINGS_SHEET = "Inställningar"

def load_credentials():
    credentials_dict = st.secrets["GOOGLE_CREDENTIALS"]
    credentials = Credentials.from_service_account_info(credentials_dict)
    client = gspread.authorize(credentials)
    return client

def load_data():
    client = load_credentials()
    sheet = client.open_by_url(SHEET_URL).worksheet(SHEET_NAME)
    data = sheet.get_all_records()
    return pd.DataFrame(data)

def save_data(df):
    df = df.fillna("")
    df = df.astype(str)
    client = load_credentials()
    sheet = client.open_by_url(SHEET_URL).worksheet(SHEET_NAME)
    sheet.clear()
    sheet.update([df.columns.tolist()] + df.values.tolist())

def skapa_inställningsblad():
    client = load_credentials()
    sheet = client.open_by_url(SHEET_URL)
    try:
        sheet.worksheet(SETTINGS_SHEET)
    except gspread.exceptions.WorksheetNotFound:
        ws = sheet.add_worksheet(title=SETTINGS_SHEET, rows="10", cols="3")
        ws.update("A1:C1", [["Inställning", "Värde", "Senast ändrad"]])
        ws.append_row(["Valutakurs", "10", str(datetime.today().date())])
        ws.append_row(["Max portföljandel (%)", "20", str(datetime.today().date())])
        ws.append_row(["Max högriskandel (%)", "2", str(datetime.today().date())])

def load_settings():
    skapa_inställningsblad()
    client = load_credentials()
    sheet = client.open_by_url(SHEET_URL).worksheet(SETTINGS_SHEET)
    rows = sheet.get_all_records()
    return {row["Inställning"]: float(row["Värde"]) for row in rows if row["Värde"] != ""}

def save_settings(settings_dict):
    client = load_credentials()
    sheet = client.open_by_url(SHEET_URL).worksheet(SETTINGS_SHEET)
    existing = sheet.get_all_records()
    existing_dict = {row["Inställning"]: i + 2 for i, row in enumerate(existing)}

    for key, value in settings_dict.items():
        rad = existing_dict.get(key)
        if rad:
            sheet.update_cell(rad, 2, str(value))
            sheet.update_cell(rad, 3, str(datetime.today().date()))
        else:
            sheet.append_row([key, str(value), str(datetime.today().date())])

def konvertera_till_ratt_typ(df):
    numeriska_kolumner = [
        "Aktuell kurs", "Målkurs 2025", "Målkurs 2026", "P/S Q1", "P/S rullande",
        "Omsättning (miljoner USD)", "Andel av portfölj (%)"
    ]
    for kolumn in numeriska_kolumner:
        if kolumn in df.columns:
            df[kolumn] = pd.to_numeric(df[kolumn], errors="coerce")
    return df

def investeringsforslag(df, kapital_sek, valutakurs, settings):
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

def visa_investeringsrad(df, valutakurs, settings):
    st.header("📈 Investeringsförslag")

    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", min_value=0, value=1000)
    if valutakurs == 0:
        st.error("Valutakurs kan inte vara 0.")
        return

    try:
        forslag, rest_sek, minska_df, högrisk_df = investeringsforslag(df, kapital_sek, valutakurs, settings)

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

def visa_databas(df):
    st.header("📋 Bolagsdatabas")
    if not df.empty:
        st.dataframe(df)
    else:
        st.info("Databasen är tom.")

def lagg_till_eller_uppdatera_bolag(df):
    st.header("➕ Lägg till eller uppdatera ett bolag")

    alla_bolag = df["Ticker"].tolist()
    valt_bolag = st.selectbox("Välj bolag att uppdatera (eller lämna tom för nytt):", [""] + sorted(alla_bolag))

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

        df = df[df["Ticker"] != ticker]
        df = pd.concat([df, pd.DataFrame([ny_rad])], ignore_index=True)
        save_data(df)
        st.success(f"{ticker} har sparats.")

def visa_databas(df):
    st.header("📋 Bolagsdatabas")
    if not df.empty:
        st.dataframe(df)
    else:
        st.info("Databasen är tom.")

def main():
    st.set_page_config(page_title="Investeringsanalys", layout="wide")
    st.title("📊 Aktieanalys & Investeringsförslag")

    inställningar = load_settings()
    max_andel = inställningar.get("Max portföljandel (%)", 20)
    max_högrisk = inställningar.get("Max högriskandel (%)", 2)
    valutakurs = inställningar.get("Valutakurs", 10)

    # Sidopanel – justerbara inställningar
    st.sidebar.header("⚙️ Inställningar")
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

    # Läs datan
    df = load_data()
    df = konvertera_till_ratt_typ(df)

    # Menyval
    menyval = st.sidebar.radio(
        "Välj vy",
        ["📈 Investeringsförslag", "🧮 Lägg till / uppdatera bolag", "📋 Databasen", "⚙️ Inställningar"]
    )

    # Visa vald vy
    if menyval == "📈 Investeringsförslag":
        visa_investeringsrad(df, valutakurs, {
            "Max portföljandel (%)": max_andel,
            "Max högriskandel (%)": max_högrisk
        })
    elif menyval == "🧮 Lägg till / uppdatera bolag":
        lagg_till_eller_uppdatera_bolag(df)
    elif menyval == "📋 Databasen":
        visa_databas(df)
    elif menyval == "⚙️ Inställningar":
        st.subheader("⚙️ Nuvarande inställningar")
        st.write(pd.DataFrame.from_dict(inställningar, orient="index", columns=["Värde"]))

if __name__ == "__main__":
    main()
