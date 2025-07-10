import streamlit as st
import pandas as pd
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials

# === Google Sheets setup ===
SHEET_URL = "https://docs.google.com/spreadsheets/d/1-5JSJpqBB0j7sm3cgEGZmnFoBL_oJDPMpLdleggL0HQ/edit"
SHEET_NAME = "Blad1"

# === Autentisering ===
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"])
client = gspread.authorize(credentials)
sheet = client.open_by_url(SHEET_URL).worksheet(SHEET_NAME)

# === Ladda data från sheet ===
def load_data():
    data = sheet.get_all_records()
    return pd.DataFrame(data)

# === Spara data till sheet ===
def save_data(df):
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.values.tolist())

import requests

# === Hämta valutakurs USD/SEK ===
def hamta_valutakurs():
    try:
        response = requests.get("https://api.exchangerate.host/latest?base=USD&symbols=SEK")
        data = response.json()
        return round(data["rates"]["SEK"], 2)
    except:
        return 10.0  # fallback

# === Hämta bolagsnamn via yfinance ===
def hamta_bolagsnamn(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info.get("longName") or info.get("shortName") or ticker
    except:
        return ticker

# === Hämta aktuell kurs automatiskt, annars returnera None ===
def hamta_kurs(ticker):
    try:
        data = yf.Ticker(ticker).history(period="1d")
        if not data.empty:
            return round(data["Close"].iloc[-1], 2)
        else:
            return None
    except:
        return None

# === Räkna fram P/S-snitt baserat på 1–4 kvartal ===
def berakna_ps_snitt(q1, q2, q3, q4):
    ps_list = [q for q in [q1, q2, q3, q4] if q > 0]
    if ps_list:
        return round(sum(ps_list) / len(ps_list), 2)
    return 0

# === Beräkna riktkurser ===
def berakna_riktkurser(omsättning, ps, aktier_milj):
    try:
        return round((omsättning * ps) / aktier_milj, 2)
    except:
        return 0

# === Räkna fram innehav i SEK ===
def berakna_innehav_i_sek(antal, kurs_usd, valutakurs):
    try:
        return round(antal * kurs_usd * valutakurs, 2)
    except:
        return 0

def visa_bolagstabell(df):
    st.subheader("📊 Portföljöversikt")
    st.dataframe(df)

def redigera_eller_lagg_till(df, valutakurs):
    st.subheader("➕ Lägg till eller redigera bolag")
    ticker = st.text_input("Ticker (t.ex. OS)", key="ticker")
    if not ticker:
        return df

    bolagsnamn = hamta_bolagsnamn(ticker)
    kurs = hamta_kurs(ticker)

    if kurs is None:
        st.warning(f"Kursen kunde inte hämtas för bolag {ticker}.")
        kurs = st.number_input("Mata in aktuell kurs (USD)", min_value=0.0, step=0.01, key="manuell_kurs")

    antal = st.number_input("Antal aktier", min_value=0, step=1)
    ps_nu = st.number_input("Nuvarande P/S", min_value=0.0, step=0.1)
    ps_q1 = st.number_input("P/S Q1", min_value=0.0, step=0.1)
    ps_q2 = st.number_input("P/S Q2", min_value=0.0, step=0.1)
    ps_q3 = st.number_input("P/S Q3", min_value=0.0, step=0.1)
    ps_q4 = st.number_input("P/S Q4", min_value=0.0, step=0.1)
    ps_snitt = berakna_ps_snitt(ps_q1, ps_q2, ps_q3, ps_q4)

    aktier_milj = st.number_input("Utestående aktier (miljoner)", min_value=0.1, step=0.1)
    oms_idag = st.number_input("Omsättning idag (miljoner USD)", min_value=0.0, step=1.0)
    oms_y1 = st.number_input("Omsättning nästa år", min_value=0.0, step=1.0)
    oms_y2 = st.number_input("Omsättning om två år", min_value=0.0, step=1.0)
    oms_y3 = st.number_input("Omsättning om tre år", min_value=0.0, step=1.0)

    rikt_idag = berakna_riktkurser(oms_idag, ps_snitt, aktier_milj)
    rikt_1 = berakna_riktkurser(oms_y1, ps_snitt, aktier_milj)
    rikt_2 = berakna_riktkurser(oms_y2, ps_snitt, aktier_milj)
    rikt_3 = berakna_riktkurser(oms_y3, ps_snitt, aktier_milj)
    innehav_sek = berakna_innehav_i_sek(antal, kurs, valutakurs)

    if st.button("Spara bolag"):
        ny_rad = {
            "Ticker": ticker,
            "Bolag": bolagsnamn,
            "Antal": antal,
            "Aktuell kurs": kurs,
            "Valutakurs": valutakurs,
            "Innehav i kr": innehav_sek,
            "P/S nu": ps_nu,
            "P/S Q1": ps_q1,
            "P/S Q2": ps_q2,
            "P/S Q3": ps_q3,
            "P/S Q4": ps_q4,
            "P/S snitt": ps_snitt,
            "Utestående aktier": aktier_milj,
            "Omsättning idag": oms_idag,
            "Omsättning nästa år": oms_y1,
            "Omsättning om två år": oms_y2,
            "Omsättning om tre år": oms_y3,
            "Riktkurs idag": rikt_idag,
            "Riktkurs om 1 år": rikt_1,
            "Riktkurs om 2 år": rikt_2,
            "Riktkurs om 3 år": rikt_3
        }
        df = pd.concat([df, pd.DataFrame([ny_rad])], ignore_index=True)
        save_data(df)
        st.success(f"{bolagsnamn} har sparats.")
    return df

def investeringsforslag(df, tillgangligt_belopp):
    st.subheader("📈 Investeringsförslag")

    df = df.copy()
    df["Undervärdering (%)"] = round((df["Riktkurs om 1 år"] - df["Aktuell kurs"]) / df["Aktuell kurs"] * 100, 2)
    df = df.sort_values(by="Undervärdering (%)", ascending=False)

    for i, rad in df.iterrows():
        bolag = rad["Bolag"]
        kurs = rad["Aktuell kurs"]
        undervardering = rad["Undervärdering (%)"]
        innehav_kr = rad["Innehav i kr"]

        if pd.isna(kurs) or kurs <= 0:
            st.warning(f"Kurs saknas för {bolag}")
            continue

        if kurs * 1 > tillgangligt_belopp:
            st.info(
                f"💡 Mest undervärderade bolaget är **{bolag}**, men kursen ({kurs} USD) överstiger ditt tillgängliga kapital ({tillgangligt_belopp} kr). Överväg att spara mer eller omfördela."
            )
        else:
            st.success(f"📌 Köpförslag: **{bolag}** ({kurs} USD), undervärderad med {undervardering} %")

        if innehav_kr > 0.3 * df["Innehav i kr"].sum():
            st.warning(f"📉 Du har en tung position i {bolag} – överväg att vikta ned!")

    st.write("---")

def main():
    st.set_page_config(page_title="Köpvärda Bolag", layout="wide")
    st.title("📊 Aktieanalys – Köpvärda bolag")

    # === Ladda data från Google Sheet
    df = load_data()
    valutakurs = hamta_valutakurs()

    # === Välj vy
    menyval = st.sidebar.selectbox("Välj vy", ["📋 Visa bolag", "➕ Lägg till bolag", "💡 Investeringsråd"])
    tillgangligt = st.sidebar.number_input("Tillgängligt belopp (SEK)", min_value=0, step=100, value=1000)

    if menyval == "📋 Visa bolag":
        visa_bolagstabell(df)
    elif menyval == "➕ Lägg till bolag":
        df = redigera_eller_lagg_till(df, valutakurs)
    elif menyval == "💡 Investeringsråd":
        investeringsforslag(df, tillgangligt)

if __name__ == "__main__":
    main()
