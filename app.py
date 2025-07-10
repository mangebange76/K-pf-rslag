import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

# === Autentisering via Streamlit Secrets (för Cloud) ===
SHEET_URL = st.secrets["SHEET_URL"]
credentials_dict = st.secrets["GOOGLE_CREDENTIALS"]

credentials = Credentials.from_service_account_info(
    credentials_dict,
    scopes=["https://www.googleapis.com/auth/spreadsheets"]
)

client = gspread.authorize(credentials)
spreadsheet = client.open_by_url(SHEET_URL)
worksheet = spreadsheet.sheet1

# === Kolumner som används i appen ===
ALL_COLUMNS = [
    "Ticker", "Bolagsnamn", "Antal aktier", "Innehav i kr", "Tillgängligt belopp (kr)",
    "Aktuell kurs", "Valutakurs", "P/S nu", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "P/S snitt",
    "Utestående aktier", "Omsättning Y1", "Omsättning Y2", "Omsättning Y3",
    "Riktkurs idag", "Riktkurs Y1", "Riktkurs Y2", "Riktkurs Y3"
]

# === Ladda data från Google Sheet ===
def load_data():
    data = worksheet.get_all_records()
    df = pd.DataFrame(data)
    df = ensure_columns_exist(df)
    return df

# === Spara data till Google Sheet ===
def save_data(df):
    worksheet.clear()
    worksheet.append_row(ALL_COLUMNS)
    for _, row in df.iterrows():
        worksheet.append_row([row.get(col, "") for col in ALL_COLUMNS])

# === Säkerställ att alla kolumner finns ===
def ensure_columns_exist(df):
    for col in ALL_COLUMNS:
        if col not in df.columns:
            df[col] = 0
    return df[ALL_COLUMNS]

import yfinance as yf

# === Lägg till eller redigera bolag ===
def lagg_till_bolag(df):
    st.subheader("➕ Lägg till eller redigera bolag")

    with st.form("ny_bolagsform"):
        ticker = st.text_input("Ticker (t.ex. NVDA)").upper()
        antal = st.number_input("Antal aktier", min_value=0, value=0)
        tillgangligt_belopp = st.number_input("Tillgängligt belopp (kr)", min_value=0, value=0)
        valutakurs = st.number_input("Valutakurs USD/SEK", min_value=0.0, value=10.0, format="%.2f")

        ps_nu = st.number_input("Nuvarande P/S", min_value=0.0, value=0.0)
        ps_q1 = st.number_input("P/S Q1", min_value=0.0, value=0.0)
        ps_q2 = st.number_input("P/S Q2", min_value=0.0, value=0.0)
        ps_q3 = st.number_input("P/S Q3", min_value=0.0, value=0.0)
        ps_q4 = st.number_input("P/S Q4", min_value=0.0, value=0.0)

        aktier = st.number_input("Utestående aktier", min_value=0.0, value=0.0, format="%.2f")
        oms1 = st.number_input("Förväntad omsättning nästa år", min_value=0.0, value=0.0)
        oms2 = st.number_input("Förväntad omsättning om 2 år", min_value=0.0, value=0.0)
        oms3 = st.number_input("Förväntad omsättning om 3 år", min_value=0.0, value=0.0)

        submitted = st.form_submit_button("Spara bolag")

    if submitted and ticker:
        try:
            ticker_info = yf.Ticker(ticker)
            kurs = ticker_info.history(period="1d")["Close"].iloc[-1]
            namn = ticker_info.info.get("shortName", ticker)
        except:
            kurs = 0
            namn = "Ej hittat"

        # Spara eller uppdatera i dataframe
        if ticker in df["Ticker"].values:
            idx = df[df["Ticker"] == ticker].index[0]
        else:
            idx = len(df)
            df.loc[idx] = [None] * len(ALL_COLUMNS)

        df.at[idx, "Ticker"] = ticker
        df.at[idx, "Bolagsnamn"] = namn
        df.at[idx, "Antal aktier"] = antal
        df.at[idx, "Tillgängligt belopp (kr)"] = tillgangligt_belopp
        df.at[idx, "Valutakurs"] = valutakurs
        df.at[idx, "Aktuell kurs"] = round(kurs, 2)
        df.at[idx, "P/S nu"] = ps_nu
        df.at[idx, "P/S Q1"] = ps_q1
        df.at[idx, "P/S Q2"] = ps_q2
        df.at[idx, "P/S Q3"] = ps_q3
        df.at[idx, "P/S Q4"] = ps_q4
        df.at[idx, "Utestående aktier"] = aktier
        df.at[idx, "Omsättning Y1"] = oms1
        df.at[idx, "Omsättning Y2"] = oms2
        df.at[idx, "Omsättning Y3"] = oms3

        st.success(f"{ticker} sparat.")
        return df

    return df

# === Beräkna P/S-snitt, riktkurser och innehavsvärden ===
def berakna_allt(df):
    for i, row in df.iterrows():
        try:
            ps_q = [row["P/S Q1"], row["P/S Q2"], row["P/S Q3"], row["P/S Q4"]]
            ps_q = [x for x in ps_q if x > 0]
            ps_snitt = round(sum(ps_q) / len(ps_q), 2) if ps_q else 0
        except:
            ps_snitt = 0

        df.at[i, "P/S snitt"] = ps_snitt

        # Riktkurser
        try:
            if row["Utestående aktier"] > 0:
                df.at[i, "Riktkurs idag"] = round((row["Omsättning Y1"] * ps_snitt) / row["Utestående aktier"], 2)
                df.at[i, "Riktkurs Y1"] = round((row["Omsättning Y2"] * ps_snitt) / row["Utestående aktier"], 2)
                df.at[i, "Riktkurs Y2"] = round((row["Omsättning Y3"] * ps_snitt) / row["Utestående aktier"], 2)
                df.at[i, "Riktkurs Y3"] = df.at[i, "Riktkurs Y2"]
            else:
                df.at[i, "Riktkurs idag"] = 0
                df.at[i, "Riktkurs Y1"] = 0
                df.at[i, "Riktkurs Y2"] = 0
                df.at[i, "Riktkurs Y3"] = 0
        except:
            df.at[i, "Riktkurs idag"] = 0
            df.at[i, "Riktkurs Y1"] = 0
            df.at[i, "Riktkurs Y2"] = 0
            df.at[i, "Riktkurs Y3"] = 0

        # Innehav i kronor
        try:
            df.at[i, "Innehav i kr"] = round(row["Antal aktier"] * row["Aktuell kurs"] * row["Valutakurs"], 2)
        except:
            df.at[i, "Innehav i kr"] = 0

    return df

# === Visa tabell över alla bolag och beräkningar ===
def visa_portfolj(df):
    st.subheader("📊 Din portfölj")

    if df.empty:
        st.info("Inga bolag tillagda ännu.")
        return df

    st.dataframe(df[[
        "Ticker", "Bolagsnamn", "Antal aktier", "Innehav i kr", "Tillgängligt belopp (kr)",
        "Aktuell kurs", "Valutakurs", "P/S snitt",
        "Riktkurs idag", "Riktkurs Y1", "Riktkurs Y2"
    ]].sort_values(by="Riktkurs idag", ascending=False), use_container_width=True)

    if st.button("🔄 Uppdatera alla beräkningar"):
        df = berakna_allt(df)
        st.success("Alla beräkningar uppdaterade.")
    return df

# === Investeringsråd ===
def investeringsrad(df):
    st.subheader("💡 Investeringsråd")

    if df.empty:
        st.info("Inga bolag att analysera.")
        return

    df = df.copy()
    df["Undervärdering (%)"] = ((df["Riktkurs idag"] - df["Aktuell kurs"]) / df["Aktuell kurs"]) * 100
    df = df[df["Aktuell kurs"] > 0].sort_values(by="Undervärdering (%)", ascending=False)

    total_portfoljvarde = df["Innehav i kr"].sum()
    total_tillgangligt = df["Tillgängligt belopp (kr)"].sum()

    st.write(f"💰 Portföljvärde: **{round(total_portfoljvarde):,} kr**")
    st.write(f"📦 Tillgängligt kapital: **{round(total_tillgangligt):,} kr**")

    for _, row in df.iterrows():
        ticker = row["Ticker"]
        namn = row["Bolagsnamn"]
        kurs = row["Aktuell kurs"]
        valuta = row["Valutakurs"]
        riktkurs = row["Riktkurs idag"]
        undervardering = round(row["Undervärdering (%)"], 1)
        tillgangligt = row["Tillgängligt belopp (kr)"]

        pris_sek = kurs * valuta
        max_kopbara = int(tillgangligt // pris_sek) if pris_sek > 0 else 0
        vikt = (row["Innehav i kr"] / total_portfoljvarde * 100) if total_portfoljvarde > 0 else 0

        st.markdown(f"### {namn} ({ticker})")
        st.write(f"- 🎯 Riktkurs: **{riktkurs:.2f} USD**")
        st.write(f"- 📉 Undervärdering: **{undervardering}%**")
        st.write(f"- 💸 Nuvarande vikt: **{vikt:.1f}%**")

        if max_kopbara > 0:
            st.success(f"📈 Rekommenderat: Köp **{max_kopbara} aktier** (~{int(max_kopbara * pris_sek):,} kr)")
        else:
            st.warning(f"💡 {namn} är mest köpvärd, men du saknar medel – **spara eller omfördela**.")

        if vikt > 30:
            st.error(f"⚠️ Portföljvikt {vikt:.1f}% – överexponerad. Överväg att sälja.")

        st.markdown("---")

# === Huvudfunktion ===
def main():
    st.title("📈 Aktieanalys & Investeringsråd")

    # Läs in data från Google Sheets
    df = load_data()

    # Formulär för att lägga till/redigera bolag
    df = lagg_till_bolag(df)

    # Kör alla beräkningar
    df = berakna_allt(df)

    # Visa portföljen
    df = visa_portfolj(df)

    # Visa investeringsråd
    investeringsrad(df)

    # Spara uppdaterad data
    save_data(df)

if __name__ == "__main__":
    main()
