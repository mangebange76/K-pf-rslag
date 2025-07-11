import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import date
import json

# --- KONSTANTER ---
SHEET_URL = "https://docs.google.com/spreadsheets/d/1-5JSJpqBB0j7sm3cgEGZmnFoBL_oJDPMpLdleggL0HQ/edit?usp=drivesdk"
SHEET_NAME = "Blad1"
SETTINGS_SHEET = "Inställningar"

# --- GOOGLE AUTH ---
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = ServiceAccountCredentials.from_json_keyfile_name("secrets/norremt.json", scope)
client = gspread.authorize(credentials)

# --- FUNKTIONER ---

def load_data():
    sheet = client.open_by_url(SHEET_URL).worksheet(SHEET_NAME)
    data = sheet.get_all_records()
    return pd.DataFrame(data)

def save_data(df):
    sheet = client.open_by_url(SHEET_URL).worksheet(SHEET_NAME)
    df = df.fillna("").astype(str)  # Rensa datan
    sheet.update([df.columns.values.tolist()] + df.values.tolist())

def load_settings():
    try:
        sheet = client.open_by_url(SHEET_URL).worksheet(SETTINGS_SHEET)
    except gspread.exceptions.WorksheetNotFound:
        sheet = client.open_by_url(SHEET_URL).add_worksheet(title=SETTINGS_SHEET, rows="10", cols="2")
        sheet.append_row(["Inställning", "Värde", "Senast ändrad"])
        sheet.append_row(["Valutakurs", "9.5", str(date.today())])
        sheet.append_row(["Max portföljandel (%)", "20", str(date.today())])
        sheet.append_row(["Max högriskandel (%)", "2", str(date.today())])
    df = pd.DataFrame(sheet.get_all_records())
    return df

def save_settings(values: dict):
    sheet = client.open_by_url(SHEET_URL).worksheet(SETTINGS_SHEET)
    df = pd.DataFrame(sheet.get_all_records())
    for key, value in values.items():
        df.loc[df["Inställning"] == key, "Värde"] = str(value)
        df.loc[df["Inställning"] == key, "Senast ändrad"] = str(date.today())
    sheet.update([df.columns.values.tolist()] + df.values.tolist())

def get_setting(settings_df, name):
    try:
        return float(settings_df.loc[settings_df["Inställning"] == name, "Värde"].values[0])
    except:
        return None

def investeringsforslag(df, kapital_sek, valutakurs, max_andel, max_hr_andel):
    df["Aktuell kurs"] = df["Aktuell kurs"].astype(str).str.replace(",", ".").astype(float)
    df["P/S"] = df["P/S"].astype(str).str.replace(",", ".").astype(float)
    df["Omsättning"] = df["Omsättning"].astype(str).str.replace(",", ".").astype(float)
    df["Andel i portfölj"] = df["Andel i portfölj"].astype(str).str.replace(",", ".").astype(float)

    kapital_usd = kapital_sek / valutakurs
    df["Prioritet"] = df["P/S"] / df["Aktuell kurs"]
    df_sorted = df.sort_values("Prioritet")

    forslag = []
    rest = kapital_usd

    for _, row in df_sorted.iterrows():
        ticker = row["Ticker"]
        kurs = row["Aktuell kurs"]
        andel = row["Andel i portfölj"]
        oms = row["Omsättning"]

        hr = oms < 1000
        max_andel = max_hr_andel if hr else max_andel
        if andel >= max_andel:
            continue

        antal = int(rest // kurs)
        if antal <= 0:
            continue
        kostnad = antal * kurs
        forslag.append((ticker, antal, kostnad * valutakurs, hr))
        rest -= kostnad
        if rest < kurs:
            break
    return forslag, rest * valutakurs

# --- STREAMLIT APP ---

def main():
    st.set_page_config(page_title="Köpförslag", layout="wide")
    st.title("Investeringsförslag och portföljanalys")

    df = load_data()
    settings_df = load_settings()

    with st.sidebar:
        st.header("Inställningar")
        valutakurs = st.number_input("Valutakurs (USD -> SEK)", value=get_setting(settings_df, "Valutakurs"))
        max_andel = st.number_input("Max portföljandel (%)", value=get_setting(settings_df, "Max portföljandel (%)"))
        max_hr_andel = st.number_input("Max högriskandel (%)", value=get_setting(settings_df, "Max högriskandel (%)"))

        if st.button("Spara inställningar"):
            save_settings({
                "Valutakurs": valutakurs,
                "Max portföljandel (%)": max_andel,
                "Max högriskandel (%)": max_hr_andel
            })
            st.success("Inställningar sparade!")

    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", min_value=0.0, step=100.0)

    if st.button("Visa investeringsförslag"):
        forslag, rest = investeringsforslag(df, kapital_sek, valutakurs, max_andel, max_hr_andel)

        if not forslag:
            st.warning("Kapitalet räcker inte för några köp. Öka beloppet.")
        else:
            st.subheader("Förslag")
            for ticker, antal, kostnad, hr in forslag:
                varning = "⚠️ Högrisk!" if hr else ""
                st.markdown(f"- **{ticker}**: Köp `{antal}` aktier för ca `{kostnad:.0f} SEK` {varning}")

            st.info(f"Kvar efter köp: {rest:.0f} SEK")

if __name__ == "__main__":
    main()
