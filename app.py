import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# Autentisering till Google Sheets med Streamlit secrets
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["GOOGLE_CREDENTIALS"], scope)
client = gspread.authorize(credentials)

SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"
SETTINGS_SHEET = "Inställningar"

# Hämtar datablad
def hamta_data():
    sheet = client.open_by_url(SHEET_URL).worksheet(SHEET_NAME)
    data = sheet.get_all_records()
    return pd.DataFrame(data)

# Sparar datan
def spara_data(df):
    sheet = client.open_by_url(SHEET_URL).worksheet(SHEET_NAME)
    df = df.astype(str).fillna("")
    sheet.update([df.columns.values.tolist()] + df.values.tolist())

# Inställningar
def hamta_settings():
    try:
        sheet = client.open_by_url(SHEET_URL).worksheet(SETTINGS_SHEET)
        data = sheet.get_all_records()
        return {row["Inställning"]: float(row["Värde"]) for row in data}
    except:
        # Skapar om det saknas
        sheet = client.open_by_url(SHEET_URL).add_worksheet(title=SETTINGS_SHEET, rows="10", cols="3")
        headers = ["Inställning", "Värde", "Senast ändrad"]
        defaults = [
            ["Valutakurs", 10.0, str(datetime.today().date())],
            ["Max portföljandel (%)", 20.0, str(datetime.today().date())],
            ["Max högriskandel (%)", 2.0, str(datetime.today().date())]
        ]
        sheet.append_row(headers)
        for row in defaults:
            sheet.append_row(row)
        return {x[0]: x[1] for x in defaults}

def spara_settings(settings_dict):
    sheet = client.open_by_url(SHEET_URL).worksheet(SETTINGS_SHEET)
    sheet.clear()
    sheet.append_row(["Inställning", "Värde", "Senast ändrad"])
    for key, val in settings_dict.items():
        sheet.append_row([key, val, str(datetime.today().date())])

# Streamlit app
def main():
    st.title("Investeringsförslag App")

    # Inställningar i sidopanel
    st.sidebar.header("Inställningar")
    settings = hamta_settings()

    valutakurs = st.sidebar.number_input("Valutakurs USD/SEK", value=float(settings["Valutakurs"]), step=0.01)
    max_andel = st.sidebar.slider("Max portföljandel (%)", 0.0, 100.0, float(settings["Max portföljandel (%)"]))
    max_risk = st.sidebar.slider("Max högriskandel (%)", 0.0, 100.0, float(settings["Max högriskandel (%)"]))

    if st.sidebar.button("Spara inställningar"):
        spara_settings({
            "Valutakurs": valutakurs,
            "Max portföljandel (%)": max_andel,
            "Max högriskandel (%)": max_risk
        })
        st.sidebar.success("Inställningar sparade.")

    df = hamta_data()
    st.subheader("Databas")
    st.dataframe(df)

    # Lägg till / uppdatera bolag
    st.subheader("Lägg till eller uppdatera bolag")
    tickers = df["Ticker"].unique().tolist() if not df.empty else []
    valt_bolag = st.selectbox("Välj bolag att uppdatera eller skapa nytt", [""] + tickers)

    if valt_bolag and valt_bolag in df["Ticker"].values:
        row = df[df["Ticker"] == valt_bolag].iloc[0]
        namn = row["Namn"]
        aktuell_kurs = float(row["Aktuell kurs"]) / 100
        nuvarande_ps = float(row["P/S"]) / 100
        ps_q1 = float(row["P/S Q1"]) / 100
        omsattning = float(row["Omsättning"]) / 100
        malpris = float(row["Målkurs"]) / 100
    else:
        namn = ""
        aktuell_kurs = 0.0
        nuvarande_ps = 0.0
        ps_q1 = 0.0
        omsattning = 0.0
        malpris = 0.0

    namn = st.text_input("Namn", namn)
    ticker = st.text_input("Ticker", valt_bolag if valt_bolag else "")
    aktuell_kurs = st.number_input("Aktuell kurs (USD)", value=aktuell_kurs, step=0.01)
    nuvarande_ps = st.number_input("P/S", value=nuvarande_ps, step=0.01)
    ps_q1 = st.number_input("P/S Q1", value=ps_q1, step=0.01)
    omsattning = st.number_input("Omsättning (miljoner USD)", value=omsattning, step=0.01)
    malpris = st.number_input("Målkurs (USD)", value=malpris, step=0.01)

    if st.button("Spara bolag"):
        ny_rad = {
            "Namn": namn,
            "Ticker": ticker,
            "Aktuell kurs": round(aktuell_kurs * 100),
            "P/S": round(nuvarande_ps * 100),
            "P/S Q1": round(ps_q1 * 100),
            "Omsättning": round(omsattning * 100),
            "Målkurs": round(malpris * 100)
        }

        if ticker in df["Ticker"].values:
            df.loc[df["Ticker"] == ticker] = ny_rad
        else:
            df = df.append(ny_rad, ignore_index=True)

        spara_data(df)
        st.success("Bolag sparat.")

if __name__ == "__main__":
    main()
