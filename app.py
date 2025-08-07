import streamlit as st
import pandas as pd
import yfinance as yf
import gspread
import time
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

# Google Sheets-koppling
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
        "Antal aktier", "Valuta", "Årlig utdelning", "CAGR 5 år (%)", "Äger"
    ]
    df = df.copy()
    for kolumn in kolumner:
        if kolumn not in df.columns:
            df[kolumn] = ""
    # Ta bort oönskade kolumner
    oönskade = ["P/S idag", "Riktkurs 2026", "Riktkurs 2027", "Riktkurs 2028", "Omsättning om 4 år", "P/S metod", "Max andel", "Yahoo ticker", "Bolag", "Initierad", "P/S-snitt.1", "21"]
    df.drop(columns=[k for k in oönskade if k in df.columns], inplace=True)
    return df

def hämta_yahoo_data(ticker):
    try:
        aktie = yf.Ticker(ticker)
        info = aktie.info
        namn = info.get("longName", "")
        kurs = info.get("currentPrice", "")
        valuta = info.get("currency", "")
        utdelning = info.get("dividendRate", "")
        cagr = info.get("revenueGrowth", "")
        return {
            "Bolagsnamn": namn,
            "Kurs": kurs,
            "Valuta": valuta,
            "Årlig utdelning": utdelning,
            "CAGR 5 år (%)": round(cagr * 100, 2) if cagr is not None else ""
        }
    except Exception:
        return {}

def konvertera_typer(df):
    kolumner_float = [
        "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "Omsättning idag",
        "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "P/S-snitt", "Riktkurs idag", "Riktkurs om 1 år",
        "Riktkurs om 2 år", "Riktkurs om 3 år", "Antal aktier",
        "Årlig utdelning", "CAGR 5 år (%)", "Utestående aktier"
    ]
    for kolumn in kolumner_float:
        if kolumn in df.columns:
            df[kolumn] = pd.to_numeric(df[kolumn], errors="coerce")
    return df

def beräkna_allt(df):
    df = df.copy()
    df = konvertera_typer(df)

    df["P/S-snitt"] = df[["P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"]].mean(axis=1)

    for i, row in df.iterrows():
        cagr = row.get("CAGR 5 år (%)", "")
        oms_kommande = row.get("Omsättning nästa år", "")
        if pd.notna(cagr) and pd.notna(oms_kommande):
            if cagr > 100:
                tillväxtfaktor = 1.5
            elif cagr < 0:
                tillväxtfaktor = 1.02
            else:
                tillväxtfaktor = 1 + (cagr / 100)
            df.at[i, "Omsättning om 2 år"] = oms_kommande * tillväxtfaktor
            df.at[i, "Omsättning om 3 år"] = oms_kommande * (tillväxtfaktor ** 2)

    for i, row in df.iterrows():
        for år in ["", " om 1 år", " om 2 år", " om 3 år"]:
            oms_kol = f"Omsättning{år}" if år else "Omsättning idag"
            ps_snitt = row.get("P/S-snitt", None)
            aktier = row.get("Utestående aktier", None)
            oms = row.get(oms_kol, None)
            if pd.notna(oms) and pd.notna(ps_snitt) and pd.notna(aktier):
                riktkurs = (oms * ps_snitt) / aktier
                df.at[i, f"Riktkurs{år}"] = riktkurs

    return df

def formulär(df):
    st.header("Lägg till / uppdatera bolag")
    tickers = df["Ticker"].tolist()
    val = st.selectbox("Välj ett bolag att uppdatera eller skapa nytt", [""] + tickers)

    if val:
        data = df[df["Ticker"] == val].iloc[0].to_dict()
    else:
        data = {}

    with st.form("nytt_bolag"):
        ticker = st.text_input("Ticker", value=data.get("Ticker", ""))
        utestående = st.number_input("Utestående aktier", value=float(data.get("Utestående aktier", 0)))
        ps = st.number_input("P/S", value=float(data.get("P/S", 0)))
        ps_q = [st.number_input(f"P/S Q{i+1}", value=float(data.get(f"P/S Q{i+1}", 0))) for i in range(4)]
        oms_idag = st.number_input("Omsättning idag", value=float(data.get("Omsättning idag", 0)))
        oms_next = st.number_input("Omsättning nästa år", value=float(data.get("Omsättning nästa år", 0)))
        antal = st.number_input("Antal aktier", value=float(data.get("Antal aktier", 0)))
        äger = st.selectbox("Äger du aktier?", ["Ja", "Nej"], index=["Ja", "Nej"].index(data.get("Äger", "Nej")))

        sparknapp = st.form_submit_button("Spara")

    if sparknapp:
        info = hämta_yahoo_data(ticker)
        ny_rad = {
            "Ticker": ticker,
            "Utestående aktier": utestående,
            "P/S": ps,
            "P/S Q1": ps_q[0],
            "P/S Q2": ps_q[1],
            "P/S Q3": ps_q[2],
            "P/S Q4": ps_q[3],
            "Omsättning idag": oms_idag,
            "Omsättning nästa år": oms_next,
            "Antal aktier": antal,
            "Äger": äger,
            "Bolagsnamn": info.get("Bolagsnamn", data.get("Bolagsnamn", "")),
            "Valuta": info.get("Valuta", data.get("Valuta", "")),
            "Årlig utdelning": info.get("Årlig utdelning", data.get("Årlig utdelning", 0)),
            "CAGR 5 år (%)": info.get("CAGR 5 år (%)", data.get("CAGR 5 år (%)", ""))
        }

        df = df[df["Ticker"] != ticker]
        df = pd.concat([df, pd.DataFrame([ny_rad])], ignore_index=True)
        df = säkerställ_kolumner(df)
        df = beräkna_allt(df)
        spara_data(df)
        st.success("Bolaget har sparats och uppdaterats.")

def analysvy(df):
    st.header("Analys")
    unika_bolag = df["Ticker"].dropna().unique()
    valt_bolag = st.selectbox("Välj ett bolag för att se detaljer", [""] + list(unika_bolag))

    if valt_bolag:
        bolagsdata = df[df["Ticker"] == valt_bolag]
        st.write(f"### Detaljer för {valt_bolag}")
        st.dataframe(bolagsdata)

    st.write("### Hela databasen")
    st.dataframe(df)

def visa_portfolj(df):
    st.header("Portfölj")
    if "Äger" not in df.columns:
        st.warning("Kolumnen 'Äger' saknas i databasen.")
        return

    df = df[df["Äger"].str.lower() == "ja"]
    df = konvertera_typer(df)

    df["Värde (SEK)"] = df["Antal aktier"] * df["Riktkurs idag"]
    df["Utdelning (SEK)"] = df["Antal aktier"] * df["Årlig utdelning"]

    total_värde = df["Värde (SEK)"].sum()
    total_utdelning = df["Utdelning (SEK)"].sum()
    utdelning_per_månad = total_utdelning / 12

    st.metric("Totalt portföljvärde (SEK)", f"{total_värde:,.0f}")
    st.metric("Total kommande utdelning (SEK)", f"{total_utdelning:,.0f}")
    st.metric("Utdelning per månad (snitt)", f"{utdelning_per_månad:,.0f}")

def investeringsförslag_vy(df):
    st.header("Investeringsförslag")

    val = st.selectbox("Filtrera efter uppsida", ["Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år"])
    df = konvertera_typer(df)
    df["Uppsida (%)"] = ((df[val] - df["Riktkurs idag"]) / df["Riktkurs idag"]) * 100
    df = df.sort_values(by="Uppsida (%)", ascending=False).reset_index(drop=True)

    if len(df) == 0:
        st.info("Inga bolag att visa.")
        return

    index = st.number_input("Visa bolag nr", min_value=1, max_value=len(df), value=1) - 1
    rad = df.iloc[index]

    st.subheader(rad["Bolagsnamn"])
    st.write(f"**Nuvarande kurs:** {rad['Riktkurs idag']:.2f}")
    st.write(f"**Riktkurs nu:** {rad['Riktkurs idag']:.2f}")
    st.write(f"**Riktkurs om 1 år:** {rad['Riktkurs om 1 år']:.2f}")
    st.write(f"**Riktkurs om 2 år:** {rad['Riktkurs om 2 år']:.2f}")
    st.write(f"**Riktkurs om 3 år:** {rad['Riktkurs om 3 år']:.2f}")
    st.write(f"**Uppsida enligt valt riktkurs:** {rad[val]:.2f}")

    tillgängligt = st.number_input("Tillgängligt belopp (SEK)", min_value=0)
    if tillgängligt > 0:
        kurs = rad["Riktkurs idag"]
        antal_köp = tillgängligt // kurs
        befintligt_antal = rad["Antal aktier"]
        totalt_före = befintligt_antal * kurs
        totalt_efter = (befintligt_antal + antal_köp) * kurs
        st.write(f"Du kan köpa **{int(antal_köp)}** aktier.")
        st.write(f"Du äger redan **{int(befintligt_antal)}** aktier.")
        st.write(f"Andel av portföljvärde före köp: {totalt_före:,.0f} SEK")
        st.write(f"Andel av portföljvärde efter köp: {totalt_efter:,.0f} SEK")

def massuppdatera(df):
    st.header("Massuppdatering från Yahoo Finance")
    om_kör = st.button("Uppdatera alla bolag")

    if not om_kör:
        return df

    df = df.copy()
    total = len(df)
    for i, ticker in enumerate(df["Ticker"]):
        st.write(f"Uppdaterar {ticker} ({i+1}/{total})...")
        info = hämta_yahoo_data(ticker)
        for kol in ["Bolagsnamn", "Valuta", "Årlig utdelning", "CAGR 5 år (%)"]:
            if kol in info:
                df.loc[df["Ticker"] == ticker, kol] = info[kol]
        time.sleep(1)

    df = beräkna_allt(df)
    spara_data(df)
    st.success("Alla bolag har uppdaterats.")
    return df

def main():
    st.set_page_config("Aktieanalys och investeringsförslag", layout="wide")
    df = hämta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)
    df = beräkna_allt(df)

    meny = st.sidebar.radio("Meny", [
        "Lägg till / uppdatera bolag",
        "Analys",
        "Portfölj",
        "Investeringsförslag",
        "Massuppdatera alla bolag"
    ])

    if meny == "Lägg till / uppdatera bolag":
        formulär(df)
    elif meny == "Analys":
        analysvy(df)
    elif meny == "Portfölj":
        visa_portfolj(df)
    elif meny == "Investeringsförslag":
        investeringsförslag_vy(df)
    elif meny == "Massuppdatera alla bolag":
        df = massuppdatera(df)

if __name__ == "__main__":
    main()
