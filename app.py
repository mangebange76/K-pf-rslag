import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# Scope för Google Sheets
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

# Autentisering med credentials från Streamlit secrets
credentials = ServiceAccountCredentials.from_json_keyfile_dict(
    st.secrets["GOOGLE_CREDENTIALS"], scope
)
client = gspread.authorize(credentials)

# Google Sheets inställningar
SHEET_URL = "https://docs.google.com/spreadsheets/d/1-5JSJpqBB0j7sm3cgEGZmnFoBL_oJDPMpLdleggL0HQ/edit?usp=drivesdk"
SHEET_NAME = "Blad1"
SETTINGS_SHEET = "Inställningar"

def load_data():
    sheet = client.open_by_url(SHEET_URL).worksheet(SHEET_NAME)
    data = sheet.get_all_records()
    return pd.DataFrame(data)

def save_data(df):
    sheet = client.open_by_url(SHEET_URL).worksheet(SHEET_NAME)
    df = df.astype(str).fillna("")
    sheet.update([df.columns.values.tolist()] + df.values.tolist())

def ensure_settings_sheet_exists():
    try:
        client.open_by_url(SHEET_URL).worksheet(SETTINGS_SHEET)
    except gspread.exceptions.WorksheetNotFound:
        sheet = client.open_by_url(SHEET_URL)
        settings_sheet = sheet.add_worksheet(title=SETTINGS_SHEET, rows="10", cols="3")
        settings_sheet.update("A1:C1", [["Inställning", "Värde", "Senast ändrad"]])
        settings_sheet.append_row(["Valutakurs", "9.5", datetime.today().strftime("%Y-%m-%d")])
        settings_sheet.append_row(["Max portföljandel (%)", "20", datetime.today().strftime("%Y-%m-%d")])
        settings_sheet.append_row(["Max högriskandel (%)", "2", datetime.today().strftime("%Y-%m-%d")])

def load_settings():
    sheet = client.open_by_url(SHEET_URL).worksheet(SETTINGS_SHEET)
    data = sheet.get_all_records()
    return {row["Inställning"]: row["Värde"] for row in data}

def save_settings(new_settings):
    sheet = client.open_by_url(SHEET_URL).worksheet(SETTINGS_SHEET)
    df = pd.DataFrame(sheet.get_all_records())
    for key, value in new_settings.items():
        df.loc[df["Inställning"] == key, "Värde"] = str(value)
        df.loc[df["Inställning"] == key, "Senast ändrad"] = datetime.today().strftime("%Y-%m-%d")
    sheet.update([df.columns.values.tolist()] + df.values.tolist())

def main():
    st.title("Investeringsapp")
    
    ensure_settings_sheet_exists()
    settings = load_settings()
    
    valutakurs = float(settings.get("Valutakurs", 9.5))
    max_andel = float(settings.get("Max portföljandel (%)", 20))
    max_hogrisk = float(settings.get("Max högriskandel (%)", 2))

    st.sidebar.subheader("Inställningar")
    ny_valutakurs = st.sidebar.number_input("Valutakurs (USD -> SEK)", value=valutakurs, step=0.01)
    ny_maxandel = st.sidebar.slider("Max portföljandel (%)", 5, 100, int(max_andel))
    ny_max_hogrisk = st.sidebar.slider("Max högriskandel (%)", 0, 20, int(max_hogrisk))
    
    if st.sidebar.button("Spara inställningar"):
        save_settings({
            "Valutakurs": ny_valutakurs,
            "Max portföljandel (%)": ny_maxandel,
            "Max högriskandel (%)": ny_max_hogrisk
        })
        st.success("Inställningar sparade")

    try:
        df = load_data()
        st.subheader("Aktuell databas")
        st.dataframe(df)

    except Exception as e:
        st.error(f"Fel vid inläsning av data: {e}")

if __name__ == "__main__":
    main()
