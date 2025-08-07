import streamlit as st
import pandas as pd
import gspread
import yfinance as yf
import time
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

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
    kolumner = [
        "Ticker", "Bolagsnamn", "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "P/S-snitt", "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
        "Antal aktier", "Valuta", "Årlig utdelning", "CAGR 5 år (%)"
    ]
    for kolumn in kolumner:
        if kolumn not in df.columns:
            df[kolumn] = ""
    return df

def hamta_data_fran_yahoo(ticker):
    try:
        aktie = yf.Ticker(ticker)
        info = aktie.info
        namn = info.get("longName", "")
        valuta = info.get("currency", "")
        kurs = info.get("currentPrice", 0.0)
        utdelning = info.get("dividendRate", 0.0)
        cagr = None

        if "financialData" in info and "revenueGrowth" in info["financialData"]:
            cagr = info["financialData"]["revenueGrowth"]

        if cagr is None and "revenueGrowth" in info:
            cagr = info.get("revenueGrowth")

        if isinstance(cagr, (int, float)):
            cagr_procent = round(cagr * 100, 2)
        else:
            cagr_procent = ""

        return namn, kurs, valuta, utdelning, cagr_procent
    except Exception:
        return "", 0.0, "", 0.0, ""

def beräkna_allt(df):
    df["P/S Q1"] = pd.to_numeric(df["P/S Q1"], errors="coerce")
    df["P/S Q2"] = pd.to_numeric(df["P/S Q2"], errors="coerce")
    df["P/S Q3"] = pd.to_numeric(df["P/S Q3"], errors="coerce")
    df["P/S Q4"] = pd.to_numeric(df["P/S Q4"], errors="coerce")
    df["Omsättning nästa år"] = pd.to_numeric(df["Omsättning nästa år"], errors="coerce")
    df["CAGR 5 år (%)"] = pd.to_numeric(df["CAGR 5 år (%)"], errors="coerce")

    df["P/S-snitt"] = df[["P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"]].mean(axis=1)

    df["Omsättning om 2 år"] = df.apply(
        lambda row: round(row["Omsättning nästa år"] * (1 + row["CAGR 5 år (%)"] / 100), 2)
        if pd.notna(row["Omsättning nästa år"]) and pd.notna(row["CAGR 5 år (%)"]) else "", axis=1
    )

    df["Omsättning om 3 år"] = df.apply(
        lambda row: round(row["Omsättning nästa år"] * ((1 + row["CAGR 5 år (%)"] / 100) ** 2), 2)
        if pd.notna(row["Omsättning nästa år"]) and pd.notna(row["CAGR 5 år (%)"]) else "", axis=1
    )

    df["P/S"] = pd.to_numeric(df["P/S"], errors="coerce")
    df["Omsättning idag"] = pd.to_numeric(df["Omsättning idag"], errors="coerce")
    df["Antal aktier"] = pd.to_numeric(df["Antal aktier"], errors="coerce")

    df["Riktkurs idag"] = df.apply(
        lambda row: round((row["Omsättning idag"] / row["Antal aktier"]) * row["P/S"], 2)
        if row["Antal aktier"] and row["P/S"] and row["Omsättning idag"] else "", axis=1
    )

    df["Riktkurs om 1 år"] = df.apply(
        lambda row: round((row["Omsättning nästa år"] / row["Antal aktier"]) * row["P/S-snitt"], 2)
        if row["Antal aktier"] and row["P/S-snitt"] and row["Omsättning nästa år"] else "", axis=1
    )

    df["Riktkurs om 2 år"] = df.apply(
        lambda row: round((row["Omsättning om 2 år"] / row["Antal aktier"]) * row["P/S-snitt"], 2)
        if row["Antal aktier"] and row["P/S-snitt"] and row["Omsättning om 2 år"] else "", axis=1
    )

    df["Riktkurs om 3 år"] = df.apply(
        lambda row: round((row["Omsättning om 3 år"] / row["Antal aktier"]) * row["P/S-snitt"], 2)
        if row["Antal aktier"] and row["P/S-snitt"] and row["Omsättning om 3 år"] else "", axis=1
    )

    return df

def lagg_till_eller_uppdatera(df):
    st.subheader("Lägg till eller uppdatera bolag")
    tickers = df["Ticker"].tolist()
    valt_ticker = st.selectbox("Välj befintligt bolag att uppdatera (eller lämna tomt)", [""] + tickers)

    if valt_ticker:
        befintlig = df[df["Ticker"] == valt_ticker].iloc[0]
    else:
        befintlig = {}

    with st.form("form_bolag"):
        ticker = st.text_input("Ticker", value=befintlig.get("Ticker", "") if befintlig else "")
        ps = st.number_input("P/S", value=float(befintlig.get("P/S", 0.0)) if befintlig else 0.0)
        ps_q1 = st.number_input("P/S Q1", value=float(befintlig.get("P/S Q1", 0.0)) if befintlig else 0.0)
        ps_q2 = st.number_input("P/S Q2", value=float(befintlig.get("P/S Q2", 0.0)) if befintlig else 0.0)
        ps_q3 = st.number_input("P/S Q3", value=float(befintlig.get("P/S Q3", 0.0)) if befintlig else 0.0)
        ps_q4 = st.number_input("P/S Q4", value=float(befintlig.get("P/S Q4", 0.0)) if befintlig else 0.0)
        oms_idag = st.number_input("Omsättning idag", value=float(befintlig.get("Omsättning idag", 0.0)) if befintlig else 0.0)
        oms_next = st.number_input("Omsättning nästa år", value=float(befintlig.get("Omsättning nästa år", 0.0)) if befintlig else 0.0)
        aktier = st.number_input("Antal aktier", value=int(befintlig.get("Antal aktier", 0)) if befintlig else 0)

        sparaknapp = st.form_submit_button("Spara bolag")

    if sparaknapp:
        namn, kurs, valuta, utdelning, cagr = hamta_data_fran_yahoo(ticker)
        if namn:
            st.success(f"Data hämtad från Yahoo Finance för {ticker}: {namn}, kurs {kurs}, valuta {valuta}, utdelning {utdelning}, CAGR {cagr}%")
        else:
            st.warning(f"Kunde inte hämta data för {ticker}. Fyll i manuellt om det saknas.")

        ny_rad = {
            "Ticker": ticker,
            "Bolagsnamn": namn,
            "P/S": ps,
            "P/S Q1": ps_q1,
            "P/S Q2": ps_q2,
            "P/S Q3": ps_q3,
            "P/S Q4": ps_q4,
            "Omsättning idag": oms_idag,
            "Omsättning nästa år": oms_next,
            "Antal aktier": aktier,
            "Valuta": valuta,
            "Aktuell kurs": kurs,
            "Årlig utdelning": utdelning,
            "CAGR 5 år (%)": cagr,
        }

        df = df[df["Ticker"] != ticker]
        df = pd.concat([df, pd.DataFrame([ny_rad])], ignore_index=True)
        df = beräkna_allt(df)
        spara_data(df)
        st.success(f"{ticker} sparat och beräknat!")

    return df

def analysvy(df):
    st.subheader("Analys & investeringsförslag")

    sorteringsval = st.selectbox("Sortera bolag efter uppsida i riktkurs", [
        "Riktkurs idag",
        "Riktkurs om 1 år",
        "Riktkurs om 2 år",
        "Riktkurs om 3 år"
    ])

    sorteringskolumn = {
        "Riktkurs idag": "Uppside (%) idag",
        "Riktkurs om 1 år": "Uppside (%) 1 år",
        "Riktkurs om 2 år": "Uppside (%) 2 år",
        "Riktkurs om 3 år": "Uppside (%) 3 år"
    }[sorteringsval]

    df_filtered = df.copy()
    df_filtered = df_filtered.sort_values(by=sorteringskolumn, ascending=False).reset_index(drop=True)

    st.write(f"{len(df_filtered)} bolag matchar sorteringen")

    if "bläddra_index" not in st.session_state:
        st.session_state.bläddra_index = 0

    def visa_bolag(index):
        if 0 <= index < len(df_filtered):
            bolag = df_filtered.iloc[index]
            st.markdown(f"### {bolag['Bolagsnamn']} ({bolag['Ticker']})")
            st.write(bolag)

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅️ Föregående") and st.session_state.bläddra_index > 0:
            st.session_state.bläddra_index -= 1
    with col2:
        if st.button("Nästa ➡️") and st.session_state.bläddra_index < len(df_filtered) - 1:
            st.session_state.bläddra_index += 1

    visa_bolag(st.session_state.bläddra_index)

    st.markdown("---")
    st.subheader("Uppdatera hela databasen från Yahoo Finance")

    if st.button("Uppdatera alla bolag"):
        for i, rad in df.iterrows():
            ticker = rad["Ticker"]
            namn, kurs, valuta, utdelning, cagr = hamta_data_fran_yahoo(ticker)
            uppdateringar = {}
            if namn: uppdateringar["Bolagsnamn"] = namn
            if kurs: uppdateringar["Aktuell kurs"] = kurs
            if valuta: uppdateringar["Valuta"] = valuta
            if utdelning is not None: uppdateringar["Årlig utdelning"] = utdelning
            if cagr is not None: uppdateringar["CAGR 5 år (%)"] = cagr
            for k, v in uppdateringar.items():
                df.at[i, k] = v
            time.sleep(1)

        df = beräkna_allt(df)
        spara_data(df)
        st.success("Alla bolag uppdaterade från Yahoo Finance.")

    st.markdown("---")
    st.subheader("Databas (alla bolag)")
    st.dataframe(df)

def main():
    df = hamta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)
    df = beräkna_allt(df)

    menyval = st.sidebar.selectbox("Välj vy", [
        "Lägg till / uppdatera bolag",
        "Analys",
        "Visa hela databasen",
        "Uppdatera alla bolag"
    ])

    if menyval == "Lägg till / uppdatera bolag":
        df = lagg_till_eller_uppdatera(df)
    elif menyval == "Analys":
        analysvy(df)
    elif menyval == "Visa hela databasen":
        st.subheader("Hela databasen")
        st.dataframe(df)
    elif menyval == "Uppdatera alla bolag":
        st.subheader("Massuppdatering från Yahoo Finance")
        if st.button("Starta uppdatering"):
            for i, rad in df.iterrows():
                st.write(f"Uppdaterar bolag {i+1} av {len(df)}: {rad['Ticker']}")
                namn, kurs, valuta, utdelning, cagr = hamta_data_fran_yahoo(rad["Ticker"])
                if namn: df.at[i, "Bolagsnamn"] = namn
                if kurs: df.at[i, "Aktuell kurs"] = kurs
                if valuta: df.at[i, "Valuta"] = valuta
                if utdelning is not None: df.at[i, "Årlig utdelning"] = utdelning
                if cagr is not None: df.at[i, "CAGR 5 år (%)"] = cagr
                time.sleep(1)
            df = beräkna_allt(df)
            spara_data(df)
            st.success("Massuppdatering klar.")

if __name__ == "__main__":
    main()
