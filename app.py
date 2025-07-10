import streamlit as st
import pandas as pd
import gspread
import yfinance as yf
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import numpy as np

st.set_page_config(page_title="Köprekommendationer", layout="wide")

# 📌 Inställningar
SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"

ALL_COLUMNS = [
    "Ticker", "Bolagsnamn", "Aktuell kurs", "Valuta", "Valutakurs",
    "Utestående aktier (miljoner)", "Omsättning idag (milj USD)",
    "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "P/S snitt",
    "Omsättning nästa år", "Omsättning om två år", "Omsättning om tre år",
    "Riktkurs idag", "Riktkurs 2026", "Riktkurs 2027",
    "Kommentar", "Senast uppdaterad"
]

import streamlit as st
import gspread
import pandas as pd
from google.oauth2 import service_account

# --- Skapa koppling till Google Sheets ---
def skapa_koppling():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = service_account.Credentials.from_service_account_info(
        st.secrets["GOOGLE_CREDENTIALS"], scopes=scope
    )
    client = gspread.authorize(creds)
    sheet = client.open_by_url(st.secrets["SHEET_URL"]).worksheet("Blad1")
    return sheet

# --- Hämta data från Google Sheet ---
def hamta_data():
    sheet = skapa_koppling()
    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    return df

# --- Spara DataFrame till Google Sheet ---
def spara_data(df):
    sheet = skapa_koppling()
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.values.tolist())

def berakna_riktkurser(df):
    def snitt_ps(row):
        ps_values = [row.get("P/S Q1", 0), row.get("P/S Q2", 0), row.get("P/S Q3", 0), row.get("P/S Q4", 0)]
        giltiga = [v for v in ps_values if v > 0]
        return round(sum(giltiga) / len(giltiga), 2) if giltiga else 0

    df["P/S snitt"] = df.apply(snitt_ps, axis=1)

    df["Riktkurs idag"] = df.apply(lambda row: round(
        (row["Omsättning idag (milj USD)"] * row["P/S snitt"] / row["Utestående aktier (miljoner)"]), 2)
        if row["Omsättning idag (milj USD)"] > 0 else 0, axis=1)

    df["Riktkurs 2026"] = df.apply(lambda row: round(
        (row["Omsättning nästa år"] * row["P/S snitt"] / row["Utestående aktier (miljoner)"]), 2)
        if row["Omsättning nästa år"] > 0 else 0, axis=1)

    df["Riktkurs 2027"] = df.apply(lambda row: round(
        (row["Omsättning om två år"] * row["P/S snitt"] / row["Utestående aktier (miljoner)"]), 2)
        if row["Omsättning om två år"] > 0 else 0, axis=1)

    return df

def berakna_undervardering(df, ar):
    kolumn = f"Riktkurs {ar}"
    if kolumn not in df.columns:
        return df
    df[f"Undervärdering {ar} (%)"] = round((df[kolumn] - df["Aktuell kurs"]) / df["Aktuell kurs"] * 100, 2)
    return df

def filtrera_df(df, sorteringsar):
    riktkurs_kolumn = f"Riktkurs {sorteringsar}"
    undervardering_kolumn = f"Undervärdering {sorteringsar} (%)"

    if riktkurs_kolumn not in df.columns or undervardering_kolumn not in df.columns:
        return df

    df_filtered = df.copy()
    df_filtered = df_filtered[df_filtered[riktkurs_kolumn] > 0]
    df_filtered = df_filtered[df_filtered["Aktuell kurs"] > 0]
    df_filtered = df_filtered[df_filtered["Ticker"] != ""]
    df_filtered = df_filtered[df_filtered["Bolagsnamn"] != ""]

    df_filtered = df_filtered.sort_values(by=undervardering_kolumn, ascending=False)
    df_filtered = df_filtered.reset_index(drop=True)

    return df_filtered

def visa_bolag(df_filtered, index, sorteringsar):
    riktkurs_kolumn = f"Riktkurs {sorteringsar}"
    undervardering_kolumn = f"Undervärdering {sorteringsar} (%)"

    if index < 0 or index >= len(df_filtered):
        st.warning("Inget bolag att visa.")
        return

    rad = df_filtered.iloc[index]
    st.subheader(f"{rad['Bolagsnamn']} ({rad['Ticker']})")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Aktuell kurs (USD)", round(rad["Aktuell kurs"], 2))
        st.metric(f"Riktkurs {sorteringsar} (USD)", round(rad[riktkurs_kolumn], 2))
    with col2:
        st.metric("Valuta", rad["Valuta"])
        st.metric("P/S-snitt", round(rad["P/S-snitt"], 2))

    st.write("**P/S per kvartal:**")
    st.write(
        f"Q1: {rad['P/S Q1']} | Q2: {rad['P/S Q2']} | Q3: {rad['P/S Q3']} | Q4: {rad['P/S Q4']}"
    )

    st.write("**Omsättning (miljoner USD):**")
    st.write(
        f"Idag: {rad['Omsättning idag']}, Nästa år: {rad['Omsättning nästa år']}, Om två år: {rad['Omsättning om två år']}, Om tre år: {rad['Omsättning om tre år']}"
    )

    st.metric(
        f"Undervärdering {sorteringsar}",
        f"{round(rad[undervardering_kolumn], 2)} %",
    )

    st.write("---")

def lagg_till_bolag(df):
    st.subheader("Lägg till nytt bolag manuellt")

    with st.form("nytt_bolag_form"):
        ticker = st.text_input("Ticker").upper()
        bolagsnamn = st.text_input("Bolagsnamn")
        valuta = st.selectbox("Valuta", ["USD", "SEK", "EUR"])
        aktuell_kurs = st.number_input("Aktuell kurs (USD)", min_value=0.0, format="%.2f")
        ps_q1 = st.number_input("P/S Q1", min_value=0.0, format="%.2f")
        ps_q2 = st.number_input("P/S Q2", min_value=0.0, format="%.2f")
        ps_q3 = st.number_input("P/S Q3", min_value=0.0, format="%.2f")
        ps_q4 = st.number_input("P/S Q4", min_value=0.0, format="%.2f")
        oms_idag = st.number_input("Omsättning idag (miljoner USD)", min_value=0.0, format="%.2f")
        oms1 = st.number_input("Omsättning nästa år", min_value=0.0, format="%.2f")
        oms2 = st.number_input("Omsättning om två år", min_value=0.0, format="%.2f")
        oms3 = st.number_input("Omsättning om tre år", min_value=0.0, format="%.2f")
        antal_aktier = st.number_input("Utestående aktier (miljoner)", min_value=0.0, format="%.2f")
        submitted = st.form_submit_button("Spara bolag")

    if submitted:
        ny_rad = {
            "Ticker": ticker,
            "Bolagsnamn": bolagsnamn,
            "Valuta": valuta,
            "Aktuell kurs": aktuell_kurs,
            "P/S Q1": ps_q1,
            "P/S Q2": ps_q2,
            "P/S Q3": ps_q3,
            "P/S Q4": ps_q4,
            "Omsättning idag": oms_idag,
            "Omsättning nästa år": oms1,
            "Omsättning om två år": oms2,
            "Omsättning om tre år": oms3,
            "Utestående aktier": antal_aktier
        }

        df = pd.concat([df, pd.DataFrame([ny_rad])], ignore_index=True)
        df = uppdatera_berakningar(df)
        skriv_data(df)
        st.success(f"{bolagsnamn} har lagts till.")

    return df

def main():
    st.set_page_config(page_title="Aktieanalysapp – Målkurs och värdering", layout="wide")
    st.title("📊 Aktieanalysapp")

    df = hamta_data()

    visa_valutakurs()
    df = konvertera_till_ratt_typ(df)
    df = uppdatera_aktuell_kurs(df)
    df = uppdatera_berakningar(df)
    skriv_data(df)

    visa_bolag(df)
    df = lagg_till_bolag(df)

if __name__ == "__main__":
    main()
