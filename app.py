import streamlit as st
import pandas as pd
import yfinance as yf
import gspread
import time
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

# Autentisering Google Sheets
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
        "Ticker", "Bolagsnamn", "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "P/S-snitt", "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år", "Antal aktier",
        "Valuta", "Aktuell kurs", "Årlig utdelning", "CAGR 5 år (%)", "Äger"
    ]
    df = df[[col for col in df.columns if col in önskade_kolumner]]
    for kolumn in önskade_kolumner:
        if kolumn not in df.columns:
            df[kolumn] = ""
    return df[önskade_kolumner]

def hamta_yahoo_data(ticker):
    try:
        aktie = yf.Ticker(ticker)
        info = aktie.info
        bolagsnamn = info.get("longName", "")
        kurs = info.get("currentPrice", "")
        valuta = info.get("currency", "")
        utdelning = info.get("dividendRate", "")
        cagr = info.get("fiveYearAvgDividendYield", "")
        if isinstance(cagr, (int, float)):
            cagr = round(cagr, 2)
        return bolagsnamn, kurs, valuta, utdelning, cagr
    except Exception:
        return "", "", "", "", ""

def beräkna_allt(df):
    for index, row in df.iterrows():
        try:
            cagr = float(row.get("CAGR 5 år (%)", 0))
            oms_nästa = float(row.get("Omsättning nästa år", 0))

            # Justera CAGR vid extrema värden
            if cagr > 100:
                justerad_cagr = 0.50
            elif cagr < 0:
                justerad_cagr = 0.02
            else:
                justerad_cagr = cagr / 100

            df.at[index, "Omsättning om 2 år"] = round(oms_nästa * (1 + justerad_cagr), 2)
            df.at[index, "Omsättning om 3 år"] = round(oms_nästa * ((1 + justerad_cagr) ** 2), 2)

            p1 = float(row.get("P/S Q1", 0))
            p2 = float(row.get("P/S Q2", 0))
            p3 = float(row.get("P/S Q3", 0))
            p4 = float(row.get("P/S Q4", 0))
            ps_snitt = round((p1 + p2 + p3 + p4) / 4, 2)
            df.at[index, "P/S-snitt"] = ps_snitt

            df.at[index, "Riktkurs idag"] = round(ps_snitt * float(row.get("Omsättning idag", 0)) / float(row.get("Utestående aktier", 1)), 2)
            df.at[index, "Riktkurs om 1 år"] = round(ps_snitt * oms_nästa / float(row.get("Utestående aktier", 1)), 2)
            df.at[index, "Riktkurs om 2 år"] = round(ps_snitt * df.at[index, "Omsättning om 2 år"] / float(row.get("Utestående aktier", 1)), 2)
            df.at[index, "Riktkurs om 3 år"] = round(ps_snitt * df.at[index, "Omsättning om 3 år"] / float(row.get("Utestående aktier", 1)), 2)
        except:
            continue
    return df

def lagg_till_eller_uppdatera(df):
    st.header("Lägg till / uppdatera bolag")
    tickers = df["Ticker"].dropna().unique().tolist()
    val = st.selectbox("Välj befintligt bolag för att uppdatera eller lämna tomt för nytt", [""] + tickers)

    if val:
        data = df[df["Ticker"] == val].iloc[0]
    else:
        data = pd.Series(dtype=object)

    with st.form("nytt_bolag"):
        ticker = st.text_input("Ticker", data.get("Ticker", ""))
        utestående_aktier = st.number_input("Utestående aktier", value=float(data.get("Utestående aktier", 0)), step=1.0)
        ps = st.number_input("P/S", value=float(data.get("P/S", 0)), step=0.01)
        ps_q1 = st.number_input("P/S Q1", value=float(data.get("P/S Q1", 0)), step=0.01)
        ps_q2 = st.number_input("P/S Q2", value=float(data.get("P/S Q2", 0)), step=0.01)
        ps_q3 = st.number_input("P/S Q3", value=float(data.get("P/S Q3", 0)), step=0.01)
        ps_q4 = st.number_input("P/S Q4", value=float(data.get("P/S Q4", 0)), step=0.01)
        oms_idag = st.number_input("Omsättning idag", value=float(data.get("Omsättning idag", 0)), step=1.0)
        oms_nästa = st.number_input("Omsättning nästa år", value=float(data.get("Omsättning nästa år", 0)), step=1.0)
        antal_aktier = st.number_input("Antal aktier (du äger)", value=float(data.get("Antal aktier", 0)), step=1.0)
        äger = st.selectbox("Äger du aktier i bolaget?", ["Ja", "Nej"], index=0 if data.get("Äger", "Ja") == "Ja" else 1)

        sparaknapp = st.form_submit_button("Spara bolag")

    if sparaknapp:
        bolagsnamn, kurs, valuta, utdelning, cagr = hamta_yahoo_data(ticker)

        ny_rad = {
            "Ticker": ticker,
            "Bolagsnamn": bolagsnamn,
            "Utestående aktier": utestående_aktier,
            "P/S": ps,
            "P/S Q1": ps_q1,
            "P/S Q2": ps_q2,
            "P/S Q3": ps_q3,
            "P/S Q4": ps_q4,
            "Omsättning idag": oms_idag,
            "Omsättning nästa år": oms_nästa,
            "Aktuell kurs": kurs,
            "Valuta": valuta,
            "Årlig utdelning": utdelning,
            "CAGR 5 år (%)": cagr,
            "Antal aktier": antal_aktier,
            "Äger": äger
        }

        df = df[df["Ticker"] != ticker]
        df = pd.concat([df, pd.DataFrame([ny_rad])], ignore_index=True)
        df = beräkna_allt(df)
        spara_data(df)
        st.success(f"{ticker} sparat med data från Yahoo Finance.")

def analysvy(df):
    st.header("Analysvy")

    # Visa enskilt bolag
    val_bolag = st.selectbox("Välj ett bolag att visa", [""] + sorted(df["Ticker"].dropna().unique()))
    if val_bolag:
        bolagsdata = df[df["Ticker"] == val_bolag]
        st.subheader(f"Data för {val_bolag}")
        st.dataframe(bolagsdata)

    # Visa hela databasen
    st.subheader("Hela databasen")
    st.dataframe(df)

def investeringsforslag(df):
    st.header("Investeringsförslag")

    df = df.copy()
    df = df[df["Äger"].str.lower() != "nej"]
    df["Kurs"] = pd.to_numeric(df["Kurs"], errors="coerce")
    df["Antal aktier"] = pd.to_numeric(df["Antal aktier"], errors="coerce")

    riktkursval = st.selectbox("Sortera efter uppsida i riktkurs:", [
        "Riktkurs", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år"
    ])

    kolumn_namn = riktkursval
    df["Uppsida (%)"] = round((df[kolumn_namn] - df["Kurs"]) / df["Kurs"] * 100, 2)
    df = df.sort_values("Uppsida (%)", ascending=False).reset_index(drop=True)

    total_bolag = len(df)
    if total_bolag == 0:
        st.warning("Inga bolag att visa.")
        return

    index = st.number_input("Visa bolag:", min_value=1, max_value=total_bolag, value=1, step=1) - 1
    rad = df.iloc[index]

    st.subheader(f"{rad['Ticker']} – Investeringsdetaljer")
    st.markdown(f"""
    - **Nuvarande kurs:** {rad['Kurs']:.2f} {rad['Valuta']}
    - **Riktkurs nu:** {rad.get('Riktkurs', '–')}
    - **Riktkurs om 1 år:** {rad.get('Riktkurs om 1 år', '–')}
    - **Riktkurs om 2 år:** {rad.get('Riktkurs om 2 år', '–')}
    - **Riktkurs om 3 år:** {rad.get('Riktkurs om 3 år', '–')}
    - **Uppsida (%):** {rad['Uppsida (%)']}%
    """)

    tillgängligt = st.number_input("Ange tillgängligt belopp (SEK):", min_value=0.0, value=0.0, step=100.0)

    växelkurs = 1.0
    if rad["Valuta"] != "SEK":
        växelkurs = st.number_input(f"Växelkurs för {rad['Valuta']} till SEK:", min_value=0.0001, value=10.0)

    kurs_sek = rad["Kurs"] * växelkurs
    antal_köp = int(tillgängligt // kurs_sek)
    befintligt_antal = rad.get("Antal aktier", 0)
    nuvarande_värde = befintligt_antal * kurs_sek
    nytt_värde = antal_köp * kurs_sek
    totalvärde = nuvarande_värde + nytt_värde

    st.markdown(f"""
    #### Investeringsförslag:
    - Du kan köpa **{antal_köp} aktier**
    - Du äger redan **{befintligt_antal} aktier**
    - Nuvarande portföljvärde: **{nuvarande_värde:.2f} SEK**
    - Efter köp skulle värdet bli: **{totalvärde:.2f} SEK**
    """)

    # Bläddringsknappar
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if index > 0:
            st.button("⬅️ Föregående", on_click=st.session_state.update, args=("index", index - 1))
    with col3:
        if index < total_bolag - 1:
            st.button("Nästa ➡️", on_click=st.session_state.update, args=("index", index + 1))

def main():
    st.sidebar.title("Meny")
    menyval = st.sidebar.radio("Gå till:", [
        "Analys", 
        "Portfölj", 
        "Investeringsförslag", 
        "Lägg till / uppdatera", 
        "Massuppdatera"
    ])

    df = hamta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)
    df = beräkna_allt(df)

    if menyval == "Analys":
        analysvy(df)
    elif menyval == "Portfölj":
        visa_portfolj(df)
    elif menyval == "Investeringsförslag":
        investeringsforslag(df)
    elif menyval == "Lägg till / uppdatera":
        lägg_till_eller_uppdatera(df)
    elif menyval == "Massuppdatera":
        massuppdatera_alla(df)

if __name__ == "__main__":
    main()
