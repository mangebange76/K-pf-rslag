import streamlit as st
import pandas as pd
import yfinance as yf
import gspread
import time
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

@st.cache_data(ttl=60)
def hämta_data():
    sheet = client.open_by_url(SHEET_URL).worksheet(SHEET_NAME)
    data = sheet.get_all_records()
    return pd.DataFrame(data)

def spara_data(df):
    sheet = client.open_by_url(SHEET_URL).worksheet(SHEET_NAME)
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.astype(str).values.tolist())

def säkerställ_kolumner(df):
    kolumner = [
        "Ticker", "Bolagsnamn", "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år", "Antal aktier",
        "Valuta", "Årlig utdelning", "Aktuell kurs", "CAGR 5 år (%)", "P/S-snitt"
    ]
    for kolumn in kolumner:
        if kolumn not in df.columns:
            df[kolumn] = ""
    return df[kolumner]

def konvertera_typer(df):
    numeriska_kolumner = [
        "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
        "Antal aktier", "Årlig utdelning", "Aktuell kurs", "CAGR 5 år (%)", "P/S-snitt"
    ]
    for kolumn in numeriska_kolumner:
        df[kolumn] = pd.to_numeric(df[kolumn], errors="coerce")
    return df

def hämta_yahoo_data(ticker):
    try:
        aktie = yf.Ticker(ticker)
        info = aktie.info

        namn = info.get("longName") or ""
        valuta = info.get("currency") or ""
        kurs = info.get("currentPrice") or info.get("regularMarketPrice") or None
        utdelning = info.get("dividendRate") or 0

        # Hämta omsättning år 1 och 5
        oms = aktie.financials
        if "Total Revenue" in aktie.income_stmt:
            oms1 = aktie.income_stmt.loc["Total Revenue"].values[0]
            oms5 = aktie.income_stmt.loc["Total Revenue"].values[-1]
        else:
            oms1 = oms5 = None

        return {
            "Bolagsnamn": namn,
            "Valuta": valuta,
            "Aktuell kurs": kurs,
            "Årlig utdelning": utdelning,
            "Omsättning nästa år": oms1,
            "Omsättning om 5 år": oms5
        }
    except Exception as e:
        st.warning(f"Kunde inte hämta data för {ticker}: {e}")
        return {}

def beräkna_cagr(oms1, oms5, år=5):
    try:
        if pd.notna(oms1) and pd.notna(oms5) and oms1 > 0:
            return (oms5 / oms1) ** (1 / år) - 1
    except:
        pass
    return None

def beräkna_omsättning(bas_oms, cagr, år):
    try:
        if pd.notna(bas_oms) and pd.notna(cagr):
            return bas_oms * ((1 + cagr) ** år)
    except:
        pass
    return None

def beräkna_riktkurs(omsättning, p_s, aktier, valuta):
    try:
        if pd.notna(omsättning) and pd.notna(p_s) and pd.notna(aktier) and aktier > 0:
            riktkurs = (omsättning * p_s) / aktier
            return riktkurs
    except:
        pass
    return None

def beräkna_kolumner(df):
    df["CAGR 5 år (%)"] = df.apply(lambda row: beräkna_cagr(row["Omsättning nästa år"], row["Omsättning om 5 år"]), axis=1)
    df["Omsättning om 2 år"] = df.apply(lambda row: beräkna_omsättning(row["Omsättning nästa år"], row["CAGR 5 år (%)"], 1), axis=1)
    df["Omsättning om 3 år"] = df.apply(lambda row: beräkna_omsättning(row["Omsättning nästa år"], row["CAGR 5 år (%)"], 2), axis=1)

    df["P/S-snitt"] = df[["P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"]].mean(axis=1)

    df["Riktkurs idag"] = df.apply(lambda row: beräkna_riktkurs(row["Omsättning idag"], row["P/S-snitt"], row["Utestående aktier"], row["Valuta"]), axis=1)
    df["Riktkurs om 1 år"] = df.apply(lambda row: beräkna_riktkurs(row["Omsättning nästa år"], row["P/S-snitt"], row["Utestående aktier"], row["Valuta"]), axis=1)
    df["Riktkurs om 2 år"] = df.apply(lambda row: beräkna_riktkurs(row["Omsättning om 2 år"], row["P/S-snitt"], row["Utestående aktier"], row["Valuta"]), axis=1)
    df["Riktkurs om 3 år"] = df.apply(lambda row: beräkna_riktkurs(row["Omsättning om 3 år"], row["P/S-snitt"], row["Utestående aktier"], row["Valuta"]), axis=1)
    return df

def formulär(df):
    st.header("Lägg till / uppdatera bolag")
    tickers = df["Ticker"].dropna().unique().tolist()
    vald_ticker = st.selectbox("Välj bolag att uppdatera eller lämna tomt för nytt", [""] + tickers)

    if vald_ticker:
        data = df[df["Ticker"] == vald_ticker].iloc[0]
    else:
        data = pd.Series(dtype="object")

    with st.form("bolagsformulär"):
        ticker = st.text_input("Ticker", value=data.get("Ticker", ""))
        antal_aktier = st.number_input("Antal aktier (ägda)", min_value=0, value=int(data.get("Antal aktier", 0) or 0))
        utestående = st.number_input("Utestående aktier", min_value=0, value=int(data.get("Utestående aktier", 0) or 0))
        ps = st.number_input("P/S", value=float(data.get("P/S", 0) or 0))
        ps_q1 = st.number_input("P/S Q1", value=float(data.get("P/S Q1", 0) or 0))
        ps_q2 = st.number_input("P/S Q2", value=float(data.get("P/S Q2", 0) or 0))
        ps_q3 = st.number_input("P/S Q3", value=float(data.get("P/S Q3", 0) or 0))
        ps_q4 = st.number_input("P/S Q4", value=float(data.get("P/S Q4", 0) or 0))
        oms_idag = st.number_input("Omsättning idag", value=float(data.get("Omsättning idag", 0) or 0))
        oms_next = st.number_input("Omsättning nästa år", value=float(data.get("Omsättning nästa år", 0) or 0))

        spara = st.form_submit_button("Spara bolag")

    if spara:
        ny_rad = {
            "Ticker": ticker,
            "Antal aktier": antal_aktier,
            "Utestående aktier": utestående,
            "P/S": ps,
            "P/S Q1": ps_q1,
            "P/S Q2": ps_q2,
            "P/S Q3": ps_q3,
            "P/S Q4": ps_q4,
            "Omsättning idag": oms_idag,
            "Omsättning nästa år": oms_next,
        }

        # Hämta från Yahoo Finance
        try:
            aktie = yf.Ticker(ticker)
            info = aktie.info
            ny_rad["Bolagsnamn"] = info.get("longName") or info.get("shortName") or ""
            ny_rad["Aktuell kurs"] = info.get("currentPrice")
            ny_rad["Valuta"] = info.get("currency")
            ny_rad["Årlig utdelning"] = info.get("dividendRate")

            ny_rad["Omsättning om 5 år"] = aktie.fast_info.get("revenue_five_years_forward")
        except:
            st.warning("Kunde inte hämta all data från Yahoo Finance.")

        df = df[df["Ticker"] != ticker]
        df = pd.concat([df, pd.DataFrame([ny_rad])], ignore_index=True)
        df = beräkna_kolumner(df)
        df = konvertera_typer(df)
        spara_data(df)
        st.success("Bolaget har sparats och beräkningar har uppdaterats.")

def analysvy(df):
    st.header("Analys")
    sorteringsval = st.selectbox("Sortera efter riktkurs om ...", ["1 år", "2 år", "3 år"])
    kolumn = f"Riktkurs om {sorteringsval}"
    df["Uppside (%)"] = ((df[kolumn] - df["Aktuell kurs"]) / df["Aktuell kurs"]) * 100
    df = df.sort_values(by="Uppside (%)", ascending=False)

    st.write(f"Visar {len(df)} bolag, sorterade efter uppsida baserat på {kolumn}")
    st.dataframe(df)

    if st.button("Massuppdatera från Yahoo Finance"):
        massuppdatera(df)

def portfölj(df):
    st.header("Portfölj")
    df_ägda = df[df["Antal aktier"] > 0].copy()
    df_ägda["Värde (SEK)"] = df_ägda["Antal aktier"] * df_ägda["Aktuell kurs"]
    df_ägda["Utdelning (SEK)"] = df_ägda["Antal aktier"] * df_ägda["Årlig utdelning"]

    st.subheader("Bolag du äger")
    st.dataframe(df_ägda[["Ticker", "Antal aktier", "Aktuell kurs", "Värde (SEK)", "Utdelning (SEK)"]])

    st.subheader("Portföljsammanställning")
    totalt_värde = df_ägda["Värde (SEK)"].sum()
    total_utdelning = df_ägda["Utdelning (SEK)"].sum()
    st.metric("Totalt portföljvärde (SEK)", f"{totalt_värde:,.0f}")
    st.metric("Total årlig utdelning (SEK)", f"{total_utdelning:,.0f}")
    st.metric("Månadsutdelning (SEK)", f"{total_utdelning / 12:,.0f}")

def investeringsförslag(df):
    st.header("Investeringsförslag")
    kolumner = {
        "1 år": "Riktkurs om 1 år",
        "2 år": "Riktkurs om 2 år",
        "3 år": "Riktkurs om 3 år"
    }
    val = st.selectbox("Filtrera efter uppsida baserat på riktkurs om ...", list(kolumner.keys()))
    kol = kolumner[val]

    df["Uppside (%)"] = ((df[kol] - df["Aktuell kurs"]) / df["Aktuell kurs"]) * 100
    df_förslag = df.sort_values("Uppside (%)", ascending=False).reset_index(drop=True)

    st.subheader("Bläddra bland bolag med störst uppsida")
    index = st.number_input("Visa bolag nr:", min_value=1, max_value=len(df_förslag), value=1, step=1) - 1
    row = df_förslag.iloc[index]

    st.write(f"### {row['Bolagsnamn']} ({row['Ticker']})")
    st.write(f"Aktuell kurs: {row['Aktuell kurs']} {row['Valuta']}")
    st.write(f"Riktkurs {val}: {row[kol]} {row['Valuta']}")
    st.write(f"Uppside: {row['Uppside (%)']:.2f} %")

    tillgängligt_belopp = st.number_input("Tillgängligt belopp (SEK)", min_value=0, value=0)
    if tillgängligt_belopp > 0:
        pris = row["Aktuell kurs"]
        antal_köp = int(tillgängligt_belopp // pris)
        st.write(f"Möjliga att köpa: {antal_köp} aktier")

def main():
    df = hämta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)
    df = beräkna_kolumner(df)

    meny = st.sidebar.radio("Meny", ["Analys", "Portfölj", "Investeringsförslag", "Lägg till / uppdatera"])
    if meny == "Analys":
        analysvy(df)
    elif meny == "Portfölj":
        portfölj(df)
    elif meny == "Investeringsförslag":
        investeringsförslag(df)
    elif meny == "Lägg till / uppdatera":
        formulär(df)

if __name__ == "__main__":
    main()
