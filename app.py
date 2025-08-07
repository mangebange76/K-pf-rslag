import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import gspread
import time
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="📊 Aktieanalys och investeringsförslag", layout="wide")

# Google Sheets-koppling
SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

# Hårdkodade valutakurser till SEK
STANDARD_VALUTAKURSER = {
    "USD": 9.75,
    "NOK": 0.95,
    "CAD": 7.05,
    "EUR": 11.18,
    "SEK": 1.0
}

def skapa_koppling():
    return client.open_by_url(SHEET_URL).worksheet(SHEET_NAME)

def hamta_data():
    return pd.DataFrame(skapa_koppling().get_all_records())

def spara_data(df):
    sheet = skapa_koppling()
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.astype(str).values.tolist())

def säkerställ_kolumner(df):
    behåll_kolumner = [
        "Ticker", "Bolagsnamn", "Aktuell kurs", "Valuta", "Årlig utdelning",
        "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "P/S-snitt", "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
        "Antal aktier", "CAGR 5 år (%)"
    ]

    for kol in behåll_kolumner:
        if kol not in df.columns:
            if "kurs" in kol.lower() or "omsättning" in kol.lower() or "p/s" in kol.lower() or "utdelning" in kol.lower() or "cagr" in kol.lower():
                df[kol] = 0.0
            else:
                df[kol] = ""

    # Ta bort oönskade kolumner
    df = df[behåll_kolumner]
    return df

def konvertera_typer(df):
    num_kolumner = [
        "Aktuell kurs", "Årlig utdelning", "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "P/S-snitt", "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
        "Antal aktier", "CAGR 5 år (%)"
    ]
    for kol in num_kolumner:
        if kol in df.columns:
            df[kol] = pd.to_numeric(df[kol], errors="coerce").fillna(0.0)
    return df

def hamta_data_yahoo(ticker):
    try:
        info = yf.Ticker(ticker).info
        kurs = info.get("currentPrice") or info.get("regularMarketPrice")
        valuta = info.get("currency", "USD")
        utdelning = info.get("dividendRate", 0.0)
        cagr = info.get("fiveYearAvgDividendYield", None)
        if cagr is None:
            cagr = 0.0
        return {
            "Aktuell kurs": kurs or 0.0,
            "Valuta": valuta,
            "Årlig utdelning": utdelning or 0.0,
            "Bolagsnamn": info.get("shortName", ticker),
            "CAGR 5 år (%)": cagr
        }
    except:
        return None

def uppdatera_beräkningar(df):
    for i, rad in df.iterrows():
        ps = [rad["P/S Q1"], rad["P/S Q2"], rad["P/S Q3"], rad["P/S Q4"]]
        ps = [x for x in ps if x > 0]
        ps_snitt = round(np.mean(ps), 2) if ps else 0.0
        df.at[i, "P/S-snitt"] = ps_snitt

        cagr = rad["CAGR 5 år (%)"] / 100
        oms_nasta_ar = rad["Omsättning nästa år"]

        if oms_nasta_ar > 0 and cagr > 0:
            df.at[i, "Omsättning om 2 år"] = round(oms_nasta_ar * (1 + cagr), 2)
            df.at[i, "Omsättning om 3 år"] = round(oms_nasta_ar * ((1 + cagr) ** 2), 2)

        utest_aktier = rad["Utestående aktier"]
        if utest_aktier > 0 and ps_snitt > 0:
            df.at[i, "Riktkurs idag"] = round((rad["Omsättning idag"] * ps_snitt) / utest_aktier, 2)
            df.at[i, "Riktkurs om 1 år"] = round((oms_nasta_ar * ps_snitt) / utest_aktier, 2)
            df.at[i, "Riktkurs om 2 år"] = round((df.at[i, "Omsättning om 2 år"] * ps_snitt) / utest_aktier, 2)
            df.at[i, "Riktkurs om 3 år"] = round((df.at[i, "Omsättning om 3 år"] * ps_snitt) / utest_aktier, 2)
    return df

def hamta_data_yahoo(ticker):
    try:
        info = yf.Ticker(ticker).info
        return {
            "Bolagsnamn": info.get("shortName", ""),
            "Aktuell kurs": info.get("regularMarketPrice", 0.0),
            "Valuta": info.get("currency", "USD"),
            "Årlig utdelning": info.get("dividendRate", 0.0),
            "CAGR 5 år (%)": info.get("revenueGrowth", 0.0) * 100 if info.get("revenueGrowth") else 0.0
        }
    except:
        return {}

def lagg_till_eller_uppdatera(df):
    st.subheader("➕ Lägg till / uppdatera bolag")
    tickers = df["Ticker"].dropna().unique()
    valt = st.selectbox("Välj bolag att uppdatera (eller lämna tom för nytt)", [""] + sorted(tickers))

    befintlig = df[df["Ticker"] == valt].iloc[0] if valt else pd.Series(dtype=object)

    with st.form("form"):
        ticker = st.text_input("Ticker", value=befintlig.get("Ticker", "")).upper()
        klick = st.form_submit_button("Hämta från Yahoo Finance")

        if klick and ticker:
            data = hamta_data_yahoo(ticker)
            st.session_state.yahoo_data = data
            st.success("Data hämtad från Yahoo Finance")

        data = st.session_state.get("yahoo_data", {})

        namn = st.text_input("Bolagsnamn", value=data.get("Bolagsnamn", befintlig.get("Bolagsnamn", "")))
        kurs = st.number_input("Aktuell kurs", value=float(data.get("Aktuell kurs", befintlig.get("Aktuell kurs", 0.0))))
        valuta = st.selectbox("Valuta", ["USD", "SEK", "NOK", "EUR", "CAD"], index=["USD", "SEK", "NOK", "EUR", "CAD"].index(data.get("Valuta", befintlig.get("Valuta", "USD"))))
        utdelning = st.number_input("Årlig utdelning", value=float(data.get("Årlig utdelning", befintlig.get("Årlig utdelning", 0.0))))
        cagr = st.number_input("CAGR 5 år (%)", value=float(data.get("CAGR 5 år (%)", befintlig.get("CAGR 5 år (%)", 0.0))))

        aktier = st.number_input("Utestående aktier", value=float(befintlig.get("Utestående aktier", 0.0)))
        antal = st.number_input("Antal aktier du äger", value=float(befintlig.get("Antal aktier", 0.0)))
        ps = st.number_input("P/S", value=float(befintlig.get("P/S", 0.0)))
        ps_q1 = st.number_input("P/S Q1", value=float(befintlig.get("P/S Q1", 0.0)))
        ps_q2 = st.number_input("P/S Q2", value=float(befintlig.get("P/S Q2", 0.0)))
        ps_q3 = st.number_input("P/S Q3", value=float(befintlig.get("P/S Q3", 0.0)))
        ps_q4 = st.number_input("P/S Q4", value=float(befintlig.get("P/S Q4", 0.0)))

        oms_idag = st.number_input("Omsättning idag", value=float(befintlig.get("Omsättning idag", 0.0)))
        oms_1 = st.number_input("Omsättning nästa år", value=float(befintlig.get("Omsättning nästa år", 0.0)))

        spara = st.form_submit_button("💾 Spara")

    if spara and ticker:
        ny_rad = {
            "Ticker": ticker,
            "Bolagsnamn": namn,
            "Aktuell kurs": kurs,
            "Valuta": valuta,
            "Årlig utdelning": utdelning,
            "CAGR 5 år (%)": cagr,
            "Utestående aktier": aktier,
            "Antal aktier": antal,
            "P/S": ps,
            "P/S Q1": ps_q1,
            "P/S Q2": ps_q2,
            "P/S Q3": ps_q3,
            "P/S Q4": ps_q4,
            "Omsättning idag": oms_idag,
            "Omsättning nästa år": oms_1
        }

        df = df[df["Ticker"] != ticker]
        df = pd.concat([df, pd.DataFrame([ny_rad])], ignore_index=True)
        st.success(f"{ticker} sparat eller uppdaterat.")

    return df

def analysvy(df):
    st.subheader("📊 Analys och investeringsförslag")

    st.markdown("#### Filtrering")
    min_da = st.slider("Min direktavkastning (%)", 0.0, 20.0, 0.0, step=0.1)
    max_da = st.slider("Max direktavkastning (%)", 0.0, 20.0, 20.0, step=0.1)
    cagr_min = st.slider("Min CAGR 5 år (%)", 0.0, 50.0, 0.0, step=0.5)
    endast_med_eps = st.checkbox("Visa endast bolag med växande omsättning (år 2 > år 1)")

    df["Direktavkastning (%)"] = pd.to_numeric(df["Årlig utdelning"], errors="coerce") / pd.to_numeric(df["Aktuell kurs"], errors="coerce") * 100
    df["P/S snitt"] = df[["P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"]].mean(axis=1)

    df["Riktkurs om 1 år"] = df["P/S snitt"] * df["Omsättning nästa år"] / df["Utestående aktier"]
    df["Riktkurs om 2 år"] = df["P/S snitt"] * df["Omsättning om 2 år"] / df["Utestående aktier"]
    df["Riktkurs om 3 år"] = df["P/S snitt"] * df["Omsättning om 3 år"] / df["Utestående aktier"]

    df_filtered = df[
        (df["Direktavkastning (%)"] >= min_da) &
        (df["Direktavkastning (%)"] <= max_da) &
        (df["CAGR 5 år (%)"] >= cagr_min)
    ]

    if endast_med_eps:
        df_filtered = df_filtered[df_filtered["Omsättning om 2 år"] > df_filtered["Omsättning nästa år"]]

    st.write(f"🔎 Visar {len(df_filtered)} bolag")
    st.dataframe(df_filtered)

    st.markdown("---")
    st.markdown("### Uppdatera hela databasen från Yahoo Finance")
    if st.button("🔄 Massuppdatera alla bolag"):
        ny_df = []
        for i, rad in df.iterrows():
            st.info(f"Hämtar data för {rad['Ticker']} ({i+1}/{len(df)})...")
            data = hamta_data_yahoo(rad["Ticker"])
            time.sleep(1)

            rad["Bolagsnamn"] = data.get("Bolagsnamn", rad["Bolagsnamn"])
            rad["Aktuell kurs"] = data.get("Aktuell kurs", rad["Aktuell kurs"])
            rad["Valuta"] = data.get("Valuta", rad["Valuta"])
            rad["Årlig utdelning"] = data.get("Årlig utdelning", rad["Årlig utdelning"])
            rad["CAGR 5 år (%)"] = data.get("CAGR 5 år (%)", rad["CAGR 5 år (%)"])

            if pd.notna(rad["CAGR 5 år (%)"]) and rad["Omsättning nästa år"] > 0:
                tillväxt = 1 + (rad["CAGR 5 år (%)"] / 100)
                rad["Omsättning om 2 år"] = round(rad["Omsättning nästa år"] * tillväxt, 2)
                rad["Omsättning om 3 år"] = round(rad["Omsättning nästa år"] * tillväxt**2, 2)

            ny_df.append(rad)

        ny_df = pd.DataFrame(ny_df)
        spara_data(ny_df)
        st.success("✅ Databasen är uppdaterad")

def main():
    st.set_page_config(page_title="📊 Aktieanalys & investeringsförslag", layout="wide")
    st.title("📊 Aktieanalys & investeringsförslag")

    df = hamta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)

    meny = st.sidebar.radio("📌 Välj vy", [
        "Analys", "Lägg till / uppdatera bolag", "Portfölj", "Massuppdatera från Yahoo"
    ])

    valutakurser = {
        "USD": 9.75,
        "NOK": 0.95,
        "CAD": 7.05,
        "EUR": 11.18,
        "SEK": 1.0
    }

    if meny == "Analys":
        analysvy(df)
    elif meny == "Lägg till / uppdatera bolag":
        df = lagg_till_eller_uppdatera(df)
        spara_data(df)
    elif meny == "Portfölj":
        visa_portfolj(df, valutakurser)
    elif meny == "Massuppdatera från Yahoo":
        st.warning("Flyttad – Använd knappen längst ned i analysvyn istället.")

if __name__ == "__main__":
    main()
