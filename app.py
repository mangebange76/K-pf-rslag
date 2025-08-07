import streamlit as st
import pandas as pd
import yfinance as yf
import time
import gspread
from google.oauth2.service_account import Credentials
import math

st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

def skapa_koppling():
    return client.open_by_url(SHEET_URL).worksheet(SHEET_NAME)

def hämta_data():
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
        "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
        "Antal aktier", "Valuta", "Årlig utdelning", "Aktuell kurs", "CAGR 5 år (%)", "P/S-snitt"
    ]
    for kolumn in kolumner:
        if kolumn not in df.columns:
            df[kolumn] = ""
    return df[kolumner]

def konvertera_typer(df):
    num_kolumner = [
        "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
        "Antal aktier", "Aktuell kurs", "Årlig utdelning", "CAGR 5 år (%)", "P/S-snitt"
    ]
    for kol in num_kolumner:
        df[kol] = pd.to_numeric(df[kol], errors="coerce")
    return df

def hämta_yahoo_data(ticker):
    try:
        aktie = yf.Ticker(ticker)
        info = aktie.info

        namn = info.get("longName", "")
        kurs = info.get("currentPrice", "")
        valuta = info.get("currency", "")
        utdelning = info.get("dividendRate", "")

        # Hämta omsättning år 1 och 5 (för CAGR)
        income_stmt = aktie.financials
        oms_year1 = None
        oms_year5 = None
        if not income_stmt.empty:
            income_stmt = income_stmt.T
            kolumner = income_stmt.columns
            if "Total Revenue" in kolumner:
                revenues = income_stmt["Total Revenue"].dropna()
                if len(revenues) >= 2:
                    oms_year1 = revenues.iloc[0]
                    oms_year5 = revenues.iloc[min(4, len(revenues) - 1)]
        return namn, kurs, valuta, utdelning, oms_year1, oms_year5
    except Exception as e:
        return "", "", "", "", None, None

def beräkna_cagr(oms1, oms5):
    try:
        if oms1 is None or oms5 is None or oms1 <= 0 or oms5 <= 0:
            return None
        cagr = ((oms5 / oms1) ** (1/4)) - 1
        return round(cagr * 100, 2)
    except:
        return None

def beräkna_kolumner(df):
    df = konvertera_typer(df)

    # Beräkna CAGR 5 år (om inte redan angiven)
    for i, row in df.iterrows():
        if pd.isna(row["CAGR 5 år (%)"]):
            oms1 = row["Omsättning idag"]
            oms5 = row["Omsättning nästa år"]
            cagr = beräkna_cagr(oms1, oms5)
            df.at[i, "CAGR 5 år (%)"] = cagr

    # Justera extrema värden
    for i, row in df.iterrows():
        cagr = row["CAGR 5 år (%)"]
        oms1 = row["Omsättning idag"]
        if pd.notna(cagr) and pd.notna(oms1):
            if cagr > 100:
                oms2 = oms1 * 1.5
                oms3 = oms2 * 1.5
            elif cagr < 0:
                oms2 = oms1 * 1.02
                oms3 = oms2 * 1.02
            else:
                faktor = 1 + cagr / 100
                oms2 = oms1 * faktor
                oms3 = oms2 * faktor
            df.at[i, "Omsättning om 2 år"] = round(oms2, 2)
            df.at[i, "Omsättning om 3 år"] = round(oms3, 2)

    # Räkna ut snitt P/S
    ps_kol = ["P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"]
    df["P/S-snitt"] = df[ps_kol].mean(axis=1, skipna=True)

    # Beräkna riktkurser
    for i, row in df.iterrows():
        snitt_ps = row["P/S-snitt"]
        aktier = row["Utestående aktier"]
        valuta = row["Valuta"]
        oms = {
            "Riktkurs idag": row["Omsättning idag"],
            "Riktkurs om 1 år": row["Omsättning nästa år"],
            "Riktkurs om 2 år": row["Omsättning om 2 år"],
            "Riktkurs om 3 år": row["Omsättning om 3 år"]
        }
        if pd.notna(snitt_ps) and pd.notna(aktier):
            for kol, oms_val in oms.items():
                if pd.notna(oms_val):
                    riktkurs = (oms_val * snitt_ps) / aktier
                    df.at[i, kol] = round(riktkurs, 2)

    return df

def formulär(df):
    st.subheader("Lägg till eller uppdatera bolag")

    tickers = df["Ticker"].dropna().unique().tolist()
    valt_ticker = st.selectbox("Välj befintligt bolag att uppdatera (eller lämna tomt för nytt)", [""] + tickers)

    if valt_ticker:
        data = df[df["Ticker"] == valt_ticker].iloc[0].to_dict()
    else:
        data = {}

    with st.form("nytt_bolag"):
        ticker = st.text_input("Ticker", value=data.get("Ticker", ""))
        bolagsnamn = st.text_input("Bolagsnamn", value=data.get("Bolagsnamn", ""))
        aktier = st.number_input("Utestående aktier", value=float(data.get("Utestående aktier", 0)))
        ps = st.number_input("P/S", value=float(data.get("P/S", 0)))
        ps_q1 = st.number_input("P/S Q1", value=float(data.get("P/S Q1", 0)))
        ps_q2 = st.number_input("P/S Q2", value=float(data.get("P/S Q2", 0)))
        ps_q3 = st.number_input("P/S Q3", value=float(data.get("P/S Q3", 0)))
        ps_q4 = st.number_input("P/S Q4", value=float(data.get("P/S Q4", 0)))
        oms_idag = st.number_input("Omsättning idag", value=float(data.get("Omsättning idag", 0)))
        oms_next = st.number_input("Omsättning nästa år", value=float(data.get("Omsättning nästa år", 0)))
        antal = st.number_input("Antal aktier", value=float(data.get("Antal aktier", 0)))

        submit = st.form_submit_button("Spara bolag")

    if submit:
        ny_rad = {
            "Ticker": ticker,
            "Bolagsnamn": bolagsnamn,
            "Utestående aktier": aktier,
            "P/S": ps,
            "P/S Q1": ps_q1,
            "P/S Q2": ps_q2,
            "P/S Q3": ps_q3,
            "P/S Q4": ps_q4,
            "Omsättning idag": oms_idag,
            "Omsättning nästa år": oms_next,
            "Antal aktier": antal
        }

        # Hämta automatiska värden från Yahoo Finance
        data_yahoo = hämta_från_yahoo(ticker)

        if data_yahoo:
            ny_rad.update(data_yahoo)
            st.success("Data hämtad från Yahoo Finance:")
            st.write(data_yahoo)
        else:
            st.warning("Kunde inte hämta data från Yahoo Finance – komplettera manuellt")

        # Uppdatera dataframe
        if ticker in df["Ticker"].values:
            df.update(pd.DataFrame([ny_rad]))
        else:
            df = pd.concat([df, pd.DataFrame([ny_rad])], ignore_index=True)

        df = beräkna_kolumner(df)
        spara_data(df)
        st.success("Bolag sparat!")

def hämta_från_yahoo(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info

        kurs = info.get("currentPrice")
        valuta = info.get("currency")
        utdelning = info.get("dividendRate")
        namn = info.get("longName")

        # Hämta omsättning för år 1 och år 5
        fin_data = ticker_obj.financials
        if fin_data is None or fin_data.empty:
            return {}

        # Omsättning år 1 = senaste kolumnen, år 5 = femte bakåt
        omsättningar = fin_data.loc["Total Revenue"].dropna().values
        if len(omsättningar) < 5:
            return {}

        oms_år1 = omsättningar[0]
        oms_år5 = omsättningar[4]

        # CAGR = ((år5 / år1) ** (1/4)) - 1
        if oms_år1 > 0 and oms_år5 > 0:
            cagr = ((oms_år5 / oms_år1) ** (1 / 4)) - 1
            cagr_procent = round(cagr * 100, 2)
        else:
            cagr_procent = 0.0

        return {
            "Aktuell kurs": kurs,
            "Valuta": valuta,
            "Årlig utdelning": utdelning,
            "Bolagsnamn": namn,
            "Omsättning år 1": oms_år1,
            "Omsättning år 5": oms_år5,
            "CAGR 5 år (%)": cagr_procent
        }

    except Exception as e:
        st.error(f"Fel vid hämtning från Yahoo Finance: {e}")
        return {}

def beräkna_kolumner(df):
    df = konvertera_typer(df)

    # Beräkna omsättning om 2 och 3 år baserat på CAGR
    def beräkna_omsättning(oms1, cagr, år):
        if pd.isna(oms1) or pd.isna(cagr):
            return None
        if cagr > 100:
            justerad_cagr = 0.5
        elif cagr < 0:
            justerad_cagr = 0.02
        else:
            justerad_cagr = cagr / 100
        return round(oms1 * ((1 + justerad_cagr) ** år), 0)

    df["Omsättning om 2 år"] = df.apply(
        lambda row: beräkna_omsättning(row["Omsättning år 1"], row["CAGR 5 år (%)"], 2), axis=1
    )
    df["Omsättning om 3 år"] = df.apply(
        lambda row: beräkna_omsättning(row["Omsättning år 1"], row["CAGR 5 år (%)"], 3), axis=1
    )

    # Beräkna P/S-snitt
    df["P/S-snitt"] = df[["P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"]].mean(axis=1, skipna=True)

    # Beräkna riktkurser
    def räkna_riktkurs(omsättning, ps, aktier, valuta):
        try:
            if pd.notna(omsättning) and pd.notna(ps) and pd.notna(aktier) and aktier > 0:
                värde = (omsättning * ps) / aktier
                return round(värde, 2)
        except:
            pass
        return None

    df["Riktkurs idag"] = df.apply(
        lambda row: räkna_riktkurs(row["Omsättning idag"], row["P/S-snitt"], row["Utestående aktier"], row["Valuta"]), axis=1
    )
    df["Riktkurs om 1 år"] = df.apply(
        lambda row: räkna_riktkurs(row["Omsättning nästa år"], row["P/S-snitt"], row["Utestående aktier"], row["Valuta"]), axis=1
    )
    df["Riktkurs om 2 år"] = df.apply(
        lambda row: räkna_riktkurs(row["Omsättning om 2 år"], row["P/S-snitt"], row["Utestående aktier"], row["Valuta"]), axis=1
    )
    df["Riktkurs om 3 år"] = df.apply(
        lambda row: räkna_riktkurs(row["Omsättning om 3 år"], row["P/S-snitt"], row["Utestående aktier"], row["Valuta"]), axis=1
    )

    return df

def konvertera_typer(df):
    numeriska_kolumner = [
        "Utestående aktier",
        "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning år 1", "Omsättning år 5",
        "Omsättning om 2 år", "Omsättning om 3 år",
        "Antal aktier", "Aktuell kurs", "Årlig utdelning",
        "CAGR 5 år (%)", "P/S-snitt",
        "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år"
    ]
    for kolumn in numeriska_kolumner:
        if kolumn in df.columns:
            df[kolumn] = pd.to_numeric(df[kolumn], errors="coerce")
    return df


def main():
    st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")
    st.title("📈 Aktieanalys och investeringsförslag")

    df = hämta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)
    df = beräkna_kolumner(df)

    meny = st.sidebar.radio("Navigering", [
        "Lägg till / uppdatera bolag",
        "Analys",
        "Portfölj",
        "Investeringsförslag",
        "Uppdatera alla bolag"
    ])

    if meny == "Lägg till / uppdatera bolag":
        formulär(df)

    elif meny == "Analys":
        analysvy(df)

    elif meny == "Portfölj":
        visa_portfolj(df)

    elif meny == "Investeringsförslag":
        investeringsförslag(df)

    elif meny == "Uppdatera alla bolag":
        massuppdatera(df)


if __name__ == "__main__":
    main()
