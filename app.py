import streamlit as st
import pandas as pd
import gspread
import yfinance as yf
import time
from google.oauth2.service_account import Credentials
from datetime import datetime

st.set_page_config(page_title="Utdelningsaktier", layout="wide")

SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

def skapa_koppling():
    return client.open_by_url(SHEET_URL).worksheet(SHEET_NAME)

def hamta_data():
    sheet = skapa_koppling()
    data = sheet.get_all_records()
    return pd.DataFrame(data)

def spara_data(df):
    sheet = skapa_koppling()
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.astype(str).values.tolist())

def säkerställ_kolumner(df):
    önskade_kolumner = [
        "Ticker", "Bolagsnamn", "Utdelning", "Valuta", "Äger", "Kurs", "52w High",
        "Direktavkastning (%)", "Riktkurs", "Uppside (%)", "Rekommendation", "Datakälla utdelning",
        "EPS TTM", "EPS om 2 år", "Payout ratio TTM (%)", "Payout ratio 2 år (%)",
        "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "Antal aktier", "CAGR 5 år"
    ]

    df = df[[col for col in df.columns if col in önskade_kolumner]]  # Ta bort alla oönskade
    for kolumn in önskade_kolumner:
        if kolumn not in df.columns:
            df[kolumn] = ""

    # Sortera i korrekt ordning
    df = df[önskade_kolumner]
    return df

import yfinance as yf
import numpy as np

def hamta_yahoo_data(ticker):
    try:
        aktie = yf.Ticker(ticker)
        info = aktie.info

        kurs = info.get("currentPrice")
        high_52w = info.get("fiftyTwoWeekHigh")
        namn = info.get("shortName")
        valuta = info.get("currency")
        utdelning = info.get("dividendRate", 0)
        eps_ttm = info.get("trailingEps")
        eps_forward = info.get("forwardEps")

        utdelning = utdelning if isinstance(utdelning, (int, float)) else 0

        return {
            "Kurs": kurs,
            "52w High": high_52w,
            "Bolagsnamn": namn,
            "Valuta": valuta,
            "Utdelning": utdelning,
            "EPS TTM": eps_ttm,
            "EPS om 2 år": eps_forward,
            "Datakälla utdelning": "Yahoo Finance"
        }

    except Exception:
        return {}

def beräkna_alla_kolumner(df):
    df["Direktavkastning (%)"] = np.where(
        (df["Kurs"].astype(float) > 0) & (df["Utdelning"].astype(float) > 0),
        round((df["Utdelning"].astype(float) / df["Kurs"].astype(float)) * 100, 2),
        ""
    )

    df["Riktkurs"] = np.where(
        df["P/S"].astype(str) != "",
        round((df["Omsättning nästa år"].astype(float) / df["Antal aktier"].astype(float)) * df["P/S"].astype(float), 2),
        ""
    )

    df["Uppside (%)"] = np.where(
        (df["Riktkurs"].astype(str) != "") & (df["Kurs"].astype(str) != ""),
        round((df["Riktkurs"].astype(float) - df["Kurs"].astype(float)) / df["Kurs"].astype(float) * 100, 2),
        ""
    )

    def rekommendation(uppsida):
        if uppsida == "":
            return ""
        if uppsida >= 50:
            return "Köp kraftigt"
        elif uppsida >= 10:
            return "Öka"
        elif uppsida >= 3:
            return "Behåll"
        elif uppsida >= -5:
            return "Pausa"
        else:
            return "Sälj"

    df["Rekommendation"] = df["Uppside (%)"].apply(lambda x: rekommendation(float(x)) if x != "" else "")

    # CAGR → beräkna omsättning om 2 och 3 år
    df["Omsättning om 2 år"] = np.where(
        (df["Omsättning nästa år"].astype(str) != "") & (df["CAGR 5 år"].astype(str) != ""),
        round(df["Omsättning nästa år"].astype(float) * (1 + df["CAGR 5 år"].astype(float)/100), 2),
        ""
    )

    df["Omsättning om 3 år"] = np.where(
        (df["Omsättning om 2 år"].astype(str) != "") & (df["CAGR 5 år"].astype(str) != ""),
        round(df["Omsättning om 2 år"].astype(float) * (1 + df["CAGR 5 år"].astype(float)/100), 2),
        ""
    )

    # Payout ratio
    df["Payout ratio TTM (%)"] = np.where(
        (df["EPS TTM"].astype(float) > 0) & (df["Utdelning"].astype(float) > 0),
        round(df["Utdelning"].astype(float) / df["EPS TTM"].astype(float) * 100, 2),
        ""
    )

    df["Payout ratio 2 år (%)"] = np.where(
        (df["EPS om 2 år"].astype(float) > 0) & (df["Utdelning"].astype(float) > 0),
        round(df["Utdelning"].astype(float) / df["EPS om 2 år"].astype(float) * 100, 2),
        ""
    )

    return df

def lagg_till_eller_uppdatera(df):
    st.header("➕ Lägg till eller uppdatera bolag")

    tickers = df["Ticker"].tolist()
    ticker_lista = [""] + tickers
    index = st.session_state.get("valda_index", 0)

    col1, col2 = st.columns([3, 1])
    with col1:
        vald_ticker = st.selectbox("Välj bolag att uppdatera eller lämna tomt för nytt", ticker_lista, index=index, key="ticker_val")

    with col2:
        if st.button("Nästa →"):
            st.session_state["valda_index"] = min(index + 1, len(ticker_lista) - 1)
            st.experimental_rerun()

    befintlig = df[df["Ticker"] == vald_ticker] if vald_ticker else pd.DataFrame()

    with st.form("lägg_till_uppdatera"):
        ticker = st.text_input("Ticker", value=vald_ticker if vald_ticker else "")
        antal_aktier = st.number_input("Antal aktier (miljoner)", value=float(befintlig["Antal aktier"].values[0]) if not befintlig.empty else 0.0)
        ps = st.number_input("P/S", value=float(befintlig["P/S"].values[0]) if not befintlig.empty else 0.0)
        ps1 = st.number_input("P/S Q1", value=float(befintlig["P/S Q1"].values[0]) if not befintlig.empty else 0.0)
        ps2 = st.number_input("P/S Q2", value=float(befintlig["P/S Q2"].values[0]) if not befintlig.empty else 0.0)
        ps3 = st.number_input("P/S Q3", value=float(befintlig["P/S Q3"].values[0]) if not befintlig.empty else 0.0)
        ps4 = st.number_input("P/S Q4", value=float(befintlig["P/S Q4"].values[0]) if not befintlig.empty else 0.0)
        oms_idag = st.number_input("Omsättning idag (MUSD)", value=float(befintlig["Omsättning idag"].values[0]) if not befintlig.empty else 0.0)
        oms_next = st.number_input("Omsättning nästa år (MUSD)", value=float(befintlig["Omsättning nästa år"].values[0]) if not befintlig.empty else 0.0)

        sparaknapp = st.form_submit_button("💾 Spara bolag")

    if sparaknapp and ticker:
        # Hämta automatisk info
        data = hamta_yahoo_data(ticker)
        ny_rad = {
            "Ticker": ticker,
            "Antal aktier": antal_aktier,
            "P/S": ps,
            "P/S Q1": ps1,
            "P/S Q2": ps2,
            "P/S Q3": ps3,
            "P/S Q4": ps4,
            "Omsättning idag": oms_idag,
            "Omsättning nästa år": oms_next,
            "Bolagsnamn": data.get("Bolagsnamn", ""),
            "Kurs": data.get("Kurs", ""),
            "52w High": data.get("52w High", ""),
            "Valuta": data.get("Valuta", ""),
            "Utdelning": data.get("Utdelning", ""),
            "EPS TTM": data.get("EPS TTM", ""),
            "EPS om 2 år": data.get("EPS om 2 år", ""),
            "Datakälla utdelning": data.get("Datakälla utdelning", ""),
        }

        df = df[df["Ticker"] != ticker]
        df = pd.concat([df, pd.DataFrame([ny_rad])], ignore_index=True)
        df = säkerställ_kolumner(df)
        df = beräkna_alla_kolumner(df)
        spara_data(df)

        st.success(f"{ticker} har sparats och uppdaterats.")
        st.info(f"Följande fält hämtades från Yahoo Finance: {', '.join([k for k in data if data[k] not in [None, '']])}")

    return df

def analysvy(df):
    st.header("📈 Analysvy – filtrering & bläddring")

    st.sidebar.subheader("🔍 Filter")
    rekommendationer = ["Köp kraftigt", "Öka", "Behåll", "Pausa", "Sälj"]
    valda = st.sidebar.multiselect("Rekommendation", rekommendationer, default=rekommendationer)
    min_diravk = st.sidebar.number_input("Min direktavkastning (%)", 0.0, 100.0, 0.0)
    max_diravk = st.sidebar.number_input("Max direktavkastning (%)", 0.0, 100.0, 100.0)
    endast_ager = st.sidebar.checkbox("Visa endast bolag jag äger")

    valuta_kurser = {
        "USD": st.sidebar.number_input("USD → SEK", value=9.50, step=0.01, key="usd"),
        "NOK": st.sidebar.number_input("NOK → SEK", value=1.00, step=0.01, key="nok"),
        "CAD": st.sidebar.number_input("CAD → SEK", value=7.50, step=0.01, key="cad"),
        "EUR": st.sidebar.number_input("EUR → SEK", value=11.00, step=0.01, key="eur"),
    }

    df["Direktavkastning (%)"] = pd.to_numeric(df["Direktavkastning (%)"], errors="coerce")
    df["Äger"] = df["Äger"].astype(str).str.lower() == "ja"
    df_filtrerat = df[
        (df["Rekommendation"].isin(valda)) &
        (df["Direktavkastning (%)"] >= min_diravk) &
        (df["Direktavkastning (%)"] <= max_diravk)
    ]
    if endast_ager:
        df_filtrerat = df_filtrerat[df_filtrerat["Äger"]]

    df_filtrerat = df_filtrerat.copy()
    df_filtrerat["Uppside (%)"] = pd.to_numeric(df_filtrerat["Uppside (%)"], errors="coerce")
    df_filtrerat = df_filtrerat.sort_values(by="Uppside (%)", ascending=False)

    st.write(f"Visar {len(df_filtrerat)} bolag efter filter")

    if not df_filtrerat.empty:
        idx = st.number_input("Bläddra bland filtrerade bolag", min_value=0, max_value=len(df_filtrerat)-1, step=1)
        st.subheader(f"{df_filtrerat.iloc[idx]['Ticker']} – {df_filtrerat.iloc[idx]['Bolagsnamn']}")
        st.write(df_filtrerat.iloc[[idx]].T)

    st.divider()
    st.subheader("📋 Hela databasen")
    st.dataframe(df)

def massuppdatera_alla(df):
    st.header("🔁 Massuppdatera alla bolag")
    start = st.button("Starta massuppdatering")

    if start:
        total = len(df)
        for i, row in df.iterrows():
            st.write(f"Uppdaterar bolag {i+1} av {total}: {row['Ticker']}")
            ticker = row["Ticker"]
            uppdaterade_data = hamta_yahoo_data(ticker)
            for kolumn, värde in uppdaterade_data.items():
                if värde not in [None, ""] and kolumn in df.columns:
                    df.at[i, kolumn] = värde
            time.sleep(1)

        st.success("Massuppdatering klar!")
        spara_data(df)
        st.experimental_rerun()


def main():
    df = hamta_data()
    säkerställ_kolumner(df)

    menyval = st.sidebar.selectbox(
        "📂 Välj vy", ["Lägg till / uppdatera bolag", "Massuppdatera alla bolag", "Analys"]
    )

    if menyval == "Lägg till / uppdatera bolag":
        df = lagg_till_eller_uppdatera(df)
    elif menyval == "Massuppdatera alla bolag":
        massuppdatera_alla(df)
    elif menyval == "Analys":
        analysvy(df)


if __name__ == "__main__":
    main()
