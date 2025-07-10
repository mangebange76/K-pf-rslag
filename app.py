import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

# === Alla kolumner som används ===
ALL_COLUMNS = [
    "Ticker", "Bolagsnamn", "Antal aktier", "Innehav i kr", "Tillgängligt belopp (kr)",
    "Aktuell kurs", "Valutakurs", "P/S nu", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "P/S snitt",
    "Utestående aktier",
    "Omsättning idag", "Omsättning Y1", "Omsättning Y2", "Omsättning Y3",
    "Riktkurs idag", "Riktkurs Y1", "Riktkurs Y2", "Riktkurs Y3"
]

# === Google Sheets-inställningar ===
SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Data"

scope = ["https://www.googleapis.com/auth/spreadsheets"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)
sheet = client.open_by_url(SHEET_URL).worksheet(SHEET_NAME)

# === Läs data från Google Sheets ===
def load_data():
    try:
        data = sheet.get_all_records()
        df = pd.DataFrame(data)
        for col in ALL_COLUMNS:
            if col not in df.columns:
                df[col] = 0
        df = df[ALL_COLUMNS]
    except:
        df = pd.DataFrame(columns=ALL_COLUMNS)
    return df

# === Spara data till Google Sheets ===
def save_data(df):
    df_to_save = df.copy()
    df_to_save = df_to_save.fillna("")
    sheet.clear()
    sheet.update([df_to_save.columns.values.tolist()] + df_to_save.values.tolist())

import yfinance as yf
import requests

# === Hämta valutakurs USD/SEK ===
def hamta_valutakurs():
    try:
        response = requests.get("https://api.exchangerate.host/latest?base=USD&symbols=SEK")
        data = response.json()
        return float(data["rates"]["SEK"])
    except:
        return 10.0  # fallback

# === Hämta bolagsnamn från ticker ===
def hamta_bolagsnamn(ticker):
    try:
        aktie = yf.Ticker(ticker)
        info = aktie.info
        return info.get("longName", "")
    except:
        return ""

# === Räkna ut P/S-snitt baserat på icke-nollvärden ===
def berakna_ps_snitt(row):
    ps_list = [row["P/S Q1"], row["P/S Q2"], row["P/S Q3"], row["P/S Q4"]]
    giltiga = [v for v in ps_list if v > 0]
    if giltiga:
        return round(sum(giltiga) / len(giltiga), 2)
    return 0

# === Räkna ut riktkurs ===
def berakna_riktkurs(oms, ps, aktier):
    try:
        return round((oms * ps) / aktier, 2) if oms > 0 and ps > 0 and aktier > 0 else 0
    except:
        return 0

# === Kör alla beräkningar ===
def berakna_allt(df):
    for i, row in df.iterrows():
        # Valutakurs
        if not row["Valutakurs"] or row["Valutakurs"] == 0:
            df.at[i, "Valutakurs"] = hamta_valutakurs()

        # Bolagsnamn
        if not row["Bolagsnamn"]:
            df.at[i, "Bolagsnamn"] = hamta_bolagsnamn(row["Ticker"])

        # P/S-snitt
        df.at[i, "P/S snitt"] = berakna_ps_snitt(row)

        # Riktkurser
        ps_snitt = df.at[i, "P/S snitt"]
        aktier_milj = df.at[i, "Utestående aktier"]
        df.at[i, "Riktkurs idag"] = berakna_riktkurs(row["Omsättning idag"], ps_snitt, aktier_milj)
        df.at[i, "Riktkurs Y1"] = berakna_riktkurs(row["Omsättning Y1"], ps_snitt, aktier_milj)
        df.at[i, "Riktkurs Y2"] = berakna_riktkurs(row["Omsättning Y2"], ps_snitt, aktier_milj)
        df.at[i, "Riktkurs Y3"] = berakna_riktkurs(row["Omsättning Y3"], ps_snitt, aktier_milj)

        # Innehav i kronor (Antal aktier × Kurs × Valutakurs)
        try:
            innehav = row["Antal aktier"] * row["Aktuell kurs"] * row["Valutakurs"]
            df.at[i, "Innehav i kr"] = round(innehav, 2)
        except:
            df.at[i, "Innehav i kr"] = 0

    return df

# === Visa portföljen i tabellform ===
def visa_portfolj(df):
    st.subheader("📊 Portföljöversikt")

    if df.empty:
        st.info("Inga bolag har lagts till ännu.")
        return df

    visa_df = df[[
        "Ticker", "Bolagsnamn", "Antal aktier", "Innehav i kr", "Tillgängligt belopp (kr)",
        "Aktuell kurs", "Valutakurs", "P/S snitt",
        "Omsättning idag", "Omsättning Y1", "Omsättning Y2", "Omsättning Y3",
        "Riktkurs idag", "Riktkurs Y1", "Riktkurs Y2", "Riktkurs Y3"
    ]].copy()

    visa_df = visa_df.sort_values("Riktkurs Y1", ascending=False)
    st.dataframe(visa_df, use_container_width=True)

    return df

# === Investeringsråd ===
def investeringsrad(df):
    st.subheader("💡 Investeringsråd")

    if df.empty:
        st.warning("Ingen data tillgänglig.")
        return

    total_tillgängligt = df["Tillgängligt belopp (kr)"].sum()
    st.write(f"**Totalt tillgängligt kapital:** {round(total_tillgängligt)} kr")

    df = df[df["Riktkurs Y1"] > 0]
    df["Potential (%)"] = ((df["Riktkurs Y1"] - df["Aktuell kurs"]) / df["Aktuell kurs"]) * 100
    df = df.sort_values("Potential (%)", ascending=False)

    bästa = df.iloc[0]
    ticker = bästa["Ticker"]
    kurs = bästa["Aktuell kurs"]
    rikt = bästa["Riktkurs Y1"]
    namn = bästa["Bolagsnamn"]

    if kurs * bästa["Valutakurs"] > total_tillgängligt:
        st.error(f"💰 Du har för lite kapital för att köpa **{namn} ({ticker})**.\n\n"
                 f"Pris: {round(kurs * bästa['Valutakurs'], 2)} kr\n"
                 f"Tillgängligt: {round(total_tillgängligt, 2)} kr")
        st.markdown(f"👉 **Föreslaget:** Skjut till pengar, spara kapital eller omfördela innehav.")
    else:
        st.success(f"✅ **Köpförslag:** {namn} ({ticker})\n\n"
                   f"Kurs: {kurs} USD → Riktkurs: {rikt} USD → Potential: {round(bästa['Potential (%)'], 1)} %")

# === Lägg till nytt bolag ===
def lagg_till_bolag(df):
    st.subheader("➕ Lägg till nytt bolag")

    with st.form("nytt_bolag"):
        ticker = st.text_input("Ticker").upper()
        antal = st.number_input("Antal aktier", min_value=0, step=1)
        kurs = st.number_input("Aktuell kurs (USD)", min_value=0.0, step=0.01)
        valutakurs = hamta_valutakurs()
        tillgangligt = st.number_input("Tillgängligt belopp (kr)", min_value=0, step=100)
        ps_nu = st.number_input("P/S nu", min_value=0.0, step=0.1)
        ps_q1 = st.number_input("P/S Q1", min_value=0.0, step=0.1)
        ps_q2 = st.number_input("P/S Q2", min_value=0.0, step=0.1)
        ps_q3 = st.number_input("P/S Q3", min_value=0.0, step=0.1)
        ps_q4 = st.number_input("P/S Q4", min_value=0.0, step=0.1)
        aktier_milj = st.number_input("Utestående aktier (miljoner)", min_value=0.0, step=0.1)
        oms_idag = st.number_input("Omsättning idag (miljoner USD)", min_value=0.0, step=1.0)
        oms_y1 = st.number_input("Omsättning nästa år (miljoner USD)", min_value=0.0, step=1.0)
        oms_y2 = st.number_input("Omsättning om två år (miljoner USD)", min_value=0.0, step=1.0)
        oms_y3 = st.number_input("Omsättning om tre år (miljoner USD)", min_value=0.0, step=1.0)

        submit = st.form_submit_button("Lägg till")

    if submit and ticker:
        ny_rad = {
            "Ticker": ticker,
            "Bolagsnamn": hamta_bolagsnamn(ticker),
            "Antal aktier": antal,
            "Aktuell kurs": kurs,
            "Valutakurs": valutakurs,
            "Tillgängligt belopp (kr)": tillgangligt,
            "P/S nu": ps_nu,
            "P/S Q1": ps_q1,
            "P/S Q2": ps_q2,
            "P/S Q3": ps_q3,
            "P/S Q4": ps_q4,
            "Utestående aktier": aktier_milj,
            "Omsättning idag": oms_idag,
            "Omsättning Y1": oms_y1,
            "Omsättning Y2": oms_y2,
            "Omsättning Y3": oms_y3
        }
        df = df.append(ny_rad, ignore_index=True)
        df = berakna_allt(df)
        save_data(df)
        st.success(f"{ticker} tillagt!")

    return df

# === Huvudfunktion ===
def main():
    st.title("📈 Aktieanalys – Köpförslag och Riktkurser")

    df = load_data()
    df = berakna_allt(df)

    # Meny
    val = st.sidebar.radio("Navigera", ["Portfölj", "Investeringsråd", "Lägg till bolag"])

    if val == "Portfölj":
        df = visa_portfolj(df)
    elif val == "Investeringsråd":
        investeringsrad(df)
    elif val == "Lägg till bolag":
        df = lagg_till_bolag(df)

# === Kör appen ===
if __name__ == "__main__":
    main()
