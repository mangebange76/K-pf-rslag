import streamlit as st
import pandas as pd
import yfinance as yf
import time
import gspread
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
        "P/S snitt", "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
        "Antal aktier", "Valuta", "Årlig utdelning", "Aktuell kurs", "CAGR 5 år (%)"
    ]
    df = df.copy()
    for kolumn in kolumner:
        if kolumn not in df.columns:
            df[kolumn] = ""
    df = df[kolumner]
    return df


def hamta_yahoo_data(ticker):
    try:
        info = yf.Ticker(ticker).info
        namn = info.get("longName") or ""
        kurs = info.get("currentPrice") or 0.0
        valuta = info.get("currency") or ""
        utdelning = info.get("dividendRate") or 0.0
        tillvaxt = info.get("revenueGrowth") or None
        cagr = tillvaxt * 100 if tillvaxt is not None else None
        return namn, kurs, valuta, utdelning, cagr
    except Exception:
        return "", 0.0, "", 0.0, None


def beräkna_allt(df):
    df = df.copy()

    df["P/S snitt"] = df[["P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"]].astype(float).mean(axis=1)
    df["Omsättning om 2 år"] = df["Omsättning nästa år"] * (1 + df["CAGR 5 år (%)"] / 100)
    df["Omsättning om 3 år"] = df["Omsättning nästa år"] * ((1 + df["CAGR 5 år (%)"] / 100) ** 2)

    df["Riktkurs idag"] = (df["Omsättning idag"] / df["Utestående aktier"]) * df["P/S snitt"]
    df["Riktkurs om 1 år"] = (df["Omsättning nästa år"] / df["Utestående aktier"]) * df["P/S snitt"]
    df["Riktkurs om 2 år"] = (df["Omsättning om 2 år"] / df["Utestående aktier"]) * df["P/S snitt"]
    df["Riktkurs om 3 år"] = (df["Omsättning om 3 år"] / df["Utestående aktier"]) * df["P/S snitt"]

    return df


def lagg_till_eller_uppdatera(df):
    st.header("Lägg till eller uppdatera bolag")
    tickers = df["Ticker"].tolist()
    valt = st.selectbox("Välj bolag att uppdatera eller skapa nytt", [""] + tickers)
    ny = valt == ""
    befintlig = df[df["Ticker"] == valt].iloc[0] if not ny else pd.Series(dtype=str)

    with st.form(key="bolagsformulär"):
        ticker = st.text_input("Ticker", value=befintlig.get("Ticker", ""))
        ps = st.number_input("P/S", value=float(befintlig.get("P/S", 0.0)) if not ny else 0.0)
        ps_q = [st.number_input(f"P/S Q{i}", value=float(befintlig.get(f"P/S Q{i}", 0.0)) if not ny else 0.0) for i in range(1, 5)]
        oms_idag = st.number_input("Omsättning idag", value=float(befintlig.get("Omsättning idag", 0.0)) if not ny else 0.0)
        oms_nasta = st.number_input("Omsättning nästa år", value=float(befintlig.get("Omsättning nästa år", 0.0)) if not ny else 0.0)
        antal_aktier = st.number_input("Utestående aktier", value=float(befintlig.get("Utestående aktier", 0.0)) if not ny else 0.0)
        ant_innehav = st.number_input("Antal aktier i portfölj", value=float(befintlig.get("Antal aktier", 0.0)) if not ny else 0.0)

        spar = st.form_submit_button("Spara")

    if spar:
        namn, kurs, valuta, utdelning, cagr = hamta_yahoo_data(ticker)

        ny_rad = {
            "Ticker": ticker,
            "Bolagsnamn": namn,
            "Utestående aktier": antal_aktier,
            "P/S": ps,
            "P/S Q1": ps_q[0],
            "P/S Q2": ps_q[1],
            "P/S Q3": ps_q[2],
            "P/S Q4": ps_q[3],
            "Omsättning idag": oms_idag,
            "Omsättning nästa år": oms_nasta,
            "Antal aktier": ant_innehav,
            "Valuta": valuta,
            "Aktuell kurs": kurs,
            "Årlig utdelning": utdelning,
            "CAGR 5 år (%)": cagr or 0.0,
        }

        for key in ["Omsättning om 2 år", "Omsättning om 3 år", "P/S snitt", "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år"]:
            ny_rad[key] = None

        df = df[df["Ticker"] != ticker]
        df = pd.concat([df, pd.DataFrame([ny_rad])], ignore_index=True)
        df = beräkna_allt(df)
        st.success(f"{ticker} sparat och uppdaterat.")
    return df


def analysvy(df):
    st.header("Analysvy")

    sorteringsval = st.selectbox("Sortera efter uppsida i:", [
        "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år"
    ])

    df = beräkna_allt(df)
    df["Uppside (%)"] = ((df[sorteringsval] - df["Aktuell kurs"]) / df["Aktuell kurs"]) * 100
    df = df.sort_values("Uppside (%)", ascending=False).reset_index(drop=True)

    st.write(f"{len(df)} bolag matchar dina val.")
    index = st.number_input("Visa bolag", min_value=0, max_value=len(df)-1, step=1)
    st.dataframe(df.iloc[[index]])

    st.subheader("Hela databasen")
    st.dataframe(df)


def portfoljvy(df):
    st.header("Portföljvy")
    df = beräkna_allt(df)
    df["Innehavsvärde (SEK)"] = df["Antal aktier"] * df["Aktuell kurs"]
    st.dataframe(df[["Ticker", "Antal aktier", "Aktuell kurs", "Innehavsvärde (SEK)"]])


def investeringsforslagvy(df):
    st.header("Investeringsförslag")
    df = beräkna_allt(df)
    kapital = st.number_input("Ange tillgängligt kapital (SEK)", min_value=0)
    sorterade = df.sort_values("Riktkurs om 3 år", ascending=False)
    st.dataframe(sorterade.head(10))


def massuppdatera_alla(df):
    st.header("Massuppdatering")
    for i, row in df.iterrows():
        ticker = row["Ticker"]
        st.write(f"Uppdaterar {ticker}...")
        namn, kurs, valuta, utdelning, cagr = hamta_yahoo_data(ticker)

        df.at[i, "Bolagsnamn"] = namn
        df.at[i, "Aktuell kurs"] = kurs
        df.at[i, "Valuta"] = valuta
        df.at[i, "Årlig utdelning"] = utdelning
        df.at[i, "CAGR 5 år (%)"] = cagr or 0.0

        time.sleep(1)

    df = beräkna_allt(df)
    st.success("Alla bolag har uppdaterats.")
    return df


def konvertera_typer(df):
    numeriska_kolumner = [
        "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "P/S snitt", "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
        "Antal aktier", "Årlig utdelning", "Aktuell kurs", "CAGR 5 år (%)"
    ]
    for kolumn in numeriska_kolumner:
        if kolumn in df.columns:
            df[kolumn] = pd.to_numeric(df[kolumn], errors="coerce")
    return df


def main():
    st.title("Aktieanalys och investeringsförslag")

    df = hamta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)

    meny = st.sidebar.radio("Meny", [
        "Analys", "Portfölj", "Investeringsförslag", "Lägg till / uppdatera", "Uppdatera alla bolag"
    ])

    if meny == "Analys":
        analysvy(df)
    elif meny == "Portfölj":
        portfoljvy(df)
    elif meny == "Investeringsförslag":
        investeringsforslagvy(df)
    elif meny == "Lägg till / uppdatera":
        df = lagg_till_eller_uppdatera(df)
    elif meny == "Uppdatera alla bolag":
        df = massuppdatera_alla(df)

    spara_data(df)


if __name__ == "__main__":
    main()
