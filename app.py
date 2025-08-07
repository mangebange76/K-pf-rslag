import streamlit as st
import pandas as pd
import numpy as np
import gspread
import yfinance as yf
import time
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="📊 Aktieanalys och investeringsförslag", layout="wide")

SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

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
    data = skapa_koppling().get_all_records()
    return pd.DataFrame(data)

def spara_data(df):
    sheet = skapa_koppling()
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.astype(str).values.tolist())

def konvertera_typer(df):
    kolumner_att_konvertera = [
        "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "P/S-snitt", "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
        "Antal aktier", "Aktuell kurs", "Årlig utdelning", "CAGR 5 år (%)"
    ]
    for kolumn in kolumner_att_konvertera:
        if kolumn in df.columns:
            df[kolumn] = pd.to_numeric(df[kolumn], errors="coerce")
    return df

def säkerställ_kolumner(df):
    nödvändiga_kolumner = [
        "Ticker", "Bolagsnamn", "Utestående aktier", "P/S", "P/S Q1", "P/S Q2",
        "P/S Q3", "P/S Q4", "Omsättning idag", "Omsättning nästa år",
        "Omsättning om 2 år", "Omsättning om 3 år", "P/S-snitt", "Riktkurs idag",
        "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år", "Antal aktier",
        "Valuta", "Aktuell kurs", "Årlig utdelning", "CAGR 5 år (%)"
    ]

    df = df.copy()
    for kolumn in nödvändiga_kolumner:
        if kolumn not in df.columns:
            df[kolumn] = np.nan

    # Ta bort kolumner som inte används
    tillåtna = set(nödvändiga_kolumner)
    df = df[[k for k in df.columns if k in tillåtna]]

    return df

def hamta_yahoo_data(ticker):
    try:
        aktie = yf.Ticker(ticker)
        info = aktie.info

        namn = info.get("longName") or info.get("shortName") or ""
        kurs = info.get("currentPrice") or info.get("regularMarketPrice")
        valuta = info.get("currency") or ""
        utdelning = info.get("dividendRate") or 0.0

        # CAGR 5 år - approximativt via historik
        hist = aktie.history(period="5y")
        if not hist.empty:
            start = hist["Close"].iloc[0]
            end = hist["Close"].iloc[-1]
            år = 5
            cagr = ((end / start) ** (1 / år)) - 1
            cagr_procent = round(cagr * 100, 2)
        else:
            cagr_procent = np.nan

        return namn, kurs, valuta, utdelning, cagr_procent

    except Exception as e:
        return "", np.nan, "", np.nan, np.nan


def beräkna_omsättning_och_riktkurser(rad):
    try:
        cagr = rad.get("CAGR 5 år (%)", np.nan)
        if pd.notna(cagr):
            tillväxtfaktor = 1 + cagr / 100
            rad["Omsättning om 2 år"] = rad.get("Omsättning nästa år", np.nan) * tillväxtfaktor
            rad["Omsättning om 3 år"] = rad.get("Omsättning nästa år", np.nan) * (tillväxtfaktor ** 2)

        ps_siffror = [rad.get(f"P/S Q{i}", np.nan) for i in range(1, 5)]
        ps_siffror = [v for v in ps_siffror if pd.notna(v)]
        rad["P/S-snitt"] = np.mean(ps_siffror) if ps_siffror else np.nan

        for år, kolumn in zip(
            ["idag", "om 1 år", "om 2 år", "om 3 år"],
            ["Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år"]
        ):
            if pd.notna(rad.get(kolumn)) and pd.notna(rad.get("P/S-snitt")) and pd.notna(rad.get("Utestående aktier")):
                riktkurs = (rad["P/S-snitt"] * rad[kolumn]) / rad["Utestående aktier"]
                rad[f"Riktkurs {år}"] = riktkurs
            else:
                rad[f"Riktkurs {år}"] = np.nan

    except Exception as e:
        pass

    return rad

def lagg_till_eller_uppdatera(df):
    st.header("Lägg till / uppdatera bolag")
    tickers = df["Ticker"].dropna().unique().tolist()
    val = st.selectbox("Välj bolag att uppdatera (eller lämna tom för nytt)", [""] + tickers)

    befintlig = df[df["Ticker"] == val].iloc[0] if val else pd.Series(dtype=object)

    with st.form("form_lagg_till"):
        ticker = st.text_input("Ticker", value=befintlig.get("Ticker", ""))
        antal_aktier = st.number_input("Utestående aktier", value=float(befintlig.get("Utestående aktier", 0.0)))
        ps_idag = st.number_input("P/S", value=float(befintlig.get("P/S", 0.0)))
        ps_q1 = st.number_input("P/S Q1", value=float(befintlig.get("P/S Q1", 0.0)))
        ps_q2 = st.number_input("P/S Q2", value=float(befintlig.get("P/S Q2", 0.0)))
        ps_q3 = st.number_input("P/S Q3", value=float(befintlig.get("P/S Q3", 0.0)))
        ps_q4 = st.number_input("P/S Q4", value=float(befintlig.get("P/S Q4", 0.0)))
        oms_idag = st.number_input("Omsättning idag", value=float(befintlig.get("Omsättning idag", 0.0)))
        oms_next = st.number_input("Omsättning nästa år", value=float(befintlig.get("Omsättning nästa år", 0.0)))

        submit = st.form_submit_button("Spara")

    if submit and ticker:
        namn, kurs, valuta, utd, cagr = hamta_yahoo_data(ticker)

        ny_rad = {
            "Ticker": ticker,
            "Bolagsnamn": namn,
            "Aktuell kurs": kurs,
            "Valuta": valuta,
            "Årlig utdelning": utd,
            "CAGR 5 år (%)": cagr,
            "Utestående aktier": antal_aktier,
            "P/S": ps_idag,
            "P/S Q1": ps_q1,
            "P/S Q2": ps_q2,
            "P/S Q3": ps_q3,
            "P/S Q4": ps_q4,
            "Omsättning idag": oms_idag,
            "Omsättning nästa år": oms_next
        }

        ny_rad = beräkna_omsättning_och_riktkurser(ny_rad)

        df = df[df["Ticker"] != ticker]
        df = pd.concat([df, pd.DataFrame([ny_rad])], ignore_index=True)
        spara_data(df)
        st.success("Bolaget har sparats/uppdaterats.")

    return df

def analysvy(df):
    st.header("Analysvy")

    if df.empty:
        st.warning("Databasen är tom.")
        return

    sorteringsval = st.selectbox("Sortera bolag efter uppsida i riktkurs", [
        "Riktkurs idag",
        "Riktkurs om 1 år",
        "Riktkurs om 2 år",
        "Riktkurs om 3 år"
    ])

    kolumn_mapping = {
        "Riktkurs idag": "Riktkurs idag",
        "Riktkurs om 1 år": "Riktkurs om 1 år",
        "Riktkurs om 2 år": "Riktkurs om 2 år",
        "Riktkurs om 3 år": "Riktkurs om 3 år"
    }

    kolumn = kolumn_mapping[sorteringsval]

    df = df.copy()
    df["Uppside (%)"] = round(100 * (df[kolumn] - df["Aktuell kurs"]) / df["Aktuell kurs"], 2)
    df = df.sort_values(by="Uppside (%)", ascending=False)

    st.subheader("Bläddra bland filtrerade bolag")
    if len(df) > 0:
        idx = st.number_input("Visa bolag nummer", min_value=1, max_value=len(df), value=1, step=1)
        valt_bolag = df.iloc[idx - 1]
        st.write(valt_bolag)
    else:
        st.info("Inga bolag matchar filtren.")

    st.subheader("Hela databasen")
    st.dataframe(df)

def massuppdatera_alla(df):
    st.header("Massuppdatera hela databasen")

    if st.button("Uppdatera alla bolag"):
        uppdaterad_df = df.copy()
        for i, rad in uppdaterad_df.iterrows():
            st.write(f"Uppdaterar bolag {i + 1} av {len(df)}: {rad['Ticker']}")
            ticker = rad["Ticker"]
            data = hämta_data_från_yahoo(ticker)
            if data:
                for kolumn, värde in data.items():
                    if värde is not None:
                        uppdaterad_df.at[i, kolumn] = värde
                # Beräkna omsättning år 2 och 3
                try:
                    omsättning_nästa = float(uppdaterad_df.at[i, "Omsättning nästa år"])
                    cagr = float(uppdaterad_df.at[i, "CAGR 5 år (%)"]) / 100
                    uppdaterad_df.at[i, "Omsättning om 2 år"] = round(omsättning_nästa * (1 + cagr)**1, 2)
                    uppdaterad_df.at[i, "Omsättning om 3 år"] = round(omsättning_nästa * (1 + cagr)**2, 2)
                except:
                    pass
            time.sleep(1)

        spara_data(uppdaterad_df)
        st.success("Alla bolag har uppdaterats.")

def main():
    st.sidebar.title("Navigering")
    val = st.sidebar.radio("Välj vy:", [
        "Lägg till / uppdatera bolag",
        "Analysvy",
        "Visa hela databasen",
        "Uppdatera alla bolag"
    ])

    df = hamta_data()
    df = konvertera_typer(df)
    df = säkerställ_kolumner(df)
    df = beräkna_allt(df)

    if val == "Lägg till / uppdatera bolag":
        df = lagg_till_eller_uppdatera(df)
    elif val == "Analysvy":
        analysvy(df)
    elif val == "Visa hela databasen":
        st.dataframe(df)
    elif val == "Uppdatera alla bolag":
        massuppdatera_alla(df)

if __name__ == "__main__":
    main()
