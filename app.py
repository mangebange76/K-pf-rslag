import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import gspread
import time
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

# Google Sheets-inställningar
SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Data"

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
        "Ticker", "Bolagsnamn", "Aktuell kurs", "Valuta", "Utestående aktier", "P/S", 
        "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "P/S-snitt",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
        "Antal aktier", "Årlig utdelning", "Äger", "CAGR 5 år (%)", "Uppside (%)"
    ]
    df_columns = df.columns.tolist()
    for kol in kolumner:
        if kol not in df_columns:
            df[kol] = ""

    # Ta bort oönskade kolumner
    df = df[kolumner]
    return df

def formulär(df):
    st.header("Lägg till eller uppdatera bolag")

    tickers = df["Ticker"].dropna().unique().tolist()
    val = st.selectbox("Välj befintligt bolag eller skriv nytt", [""] + tickers)

    nytt = val == ""

    with st.form("nytt_bolag"):
        ticker = st.text_input("Ticker", value=val if not nytt else "").upper()
        antal_aktier = st.number_input("Antal aktier", min_value=0, step=1)

        utestående = st.number_input("Utestående aktier", min_value=0, step=1000)
        ps = st.number_input("P/S", min_value=0.0)
        ps_q1 = st.number_input("P/S Q1", min_value=0.0)
        ps_q2 = st.number_input("P/S Q2", min_value=0.0)
        ps_q3 = st.number_input("P/S Q3", min_value=0.0)
        ps_q4 = st.number_input("P/S Q4", min_value=0.0)
        omsättning_idag = st.number_input("Omsättning idag", min_value=0.0)
        omsättning_nästa = st.number_input("Omsättning nästa år", min_value=0.0)

        äger = st.selectbox("Äger?", ["Ja", "Nej"])

        sparaknapp = st.form_submit_button("Spara bolag")

    if sparaknapp and ticker:
        data = {
            "Ticker": ticker,
            "Utestående aktier": utestående,
            "P/S": ps,
            "P/S Q1": ps_q1,
            "P/S Q2": ps_q2,
            "P/S Q3": ps_q3,
            "P/S Q4": ps_q4,
            "Omsättning idag": omsättning_idag,
            "Omsättning nästa år": omsättning_nästa,
            "Antal aktier": antal_aktier,
            "Äger": äger
        }

        try:
            info = yf.Ticker(ticker).info
            kurs = info.get("currentPrice")
            bolagsnamn = info.get("longName") or ""
            valuta = info.get("currency")
            utdelning = info.get("dividendRate")
            cagr = info.get("revenueGrowth")

            if kurs:
                data["Aktuell kurs"] = kurs
            if bolagsnamn:
                data["Bolagsnamn"] = bolagsnamn
            if valuta:
                data["Valuta"] = valuta
            if utdelning:
                data["Årlig utdelning"] = utdelning
            if cagr is not None:
                data["CAGR 5 år (%)"] = round(cagr * 100, 2)
            else:
                data["CAGR 5 år (%)"] = ""

        except Exception as e:
            st.error(f"Kunde inte hämta data från Yahoo Finance: {e}")

        # Uppdatera df med ny eller uppdaterad rad
        df = df[df["Ticker"] != ticker]
        df = df.append(data, ignore_index=True)
        df = beräkna_allt(df)
        df = säkerställ_kolumner(df)
        spara_data(df)
        st.success(f"{ticker} sparat.")

def beräkna_allt(df):
    df = df.copy()

    # Konvertera nödvändiga kolumner till numeriska
    kolumner = [
        "Omsättning nästa år", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Aktuell kurs", "Utestående aktier", "CAGR 5 år (%)", "Årlig utdelning"
    ]
    for kolumn in kolumner:
        if kolumn in df.columns:
            df[kolumn] = pd.to_numeric(df[kolumn], errors="coerce")

    # Beräkna P/S-snitt
    df["P/S-snitt"] = df[["P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"]].mean(axis=1)

    # CAGR-justering
    cagr = df["CAGR 5 år (%)"].fillna(0) / 100
    justerad_cagr = cagr.copy()
    justerad_cagr[cagr > 1] = 0.5  # max 50% ökning om > 100%
    justerad_cagr[cagr < 0] = 0.02  # ersätt med 2% inflation om negativ

    df["Omsättning om 2 år"] = df["Omsättning nästa år"] * (1 + justerad_cagr)
    df["Omsättning om 3 år"] = df["Omsättning nästa år"] * (1 + justerad_cagr) ** 2

    # Riktkurser
    df["Riktkurs idag"] = df["Omsättning idag"] * df["P/S-snitt"] / df["Utestående aktier"]
    df["Riktkurs om 1 år"] = df["Omsättning nästa år"] * df["P/S-snitt"] / df["Utestående aktier"]
    df["Riktkurs om 2 år"] = df["Omsättning om 2 år"] * df["P/S-snitt"] / df["Utestående aktier"]
    df["Riktkurs om 3 år"] = df["Omsättning om 3 år"] * df["P/S-snitt"] / df["Utestående aktier"]

    # Räkna ut direktavkastning
    df["Direktavkastning (%)"] = (df["Årlig utdelning"] / df["Aktuell kurs"]) * 100

    # Uppside mot nuvarande kurs (idag som default)
    df["Uppside (%)"] = ((df["Riktkurs idag"] - df["Aktuell kurs"]) / df["Aktuell kurs"]) * 100

    return df

def visa_portfolj(df):
    st.subheader("📊 Portfölj")

    if "Äger" not in df.columns or "Antal aktier" not in df.columns:
        st.warning("Kolumnerna 'Äger' eller 'Antal aktier' saknas i databasen.")
        return

    # Filtrera bara bolag man äger
    df_portfolj = df[df["Äger"].str.lower() == "ja"].copy()

    if df_portfolj.empty:
        st.info("Du äger inga bolag just nu.")
        return

    df_portfolj["Antal aktier"] = pd.to_numeric(df_portfolj["Antal aktier"], errors="coerce").fillna(0)
    df_portfolj["Aktuell kurs"] = pd.to_numeric(df_portfolj["Aktuell kurs"], errors="coerce").fillna(0)
    df_portfolj["Årlig utdelning"] = pd.to_numeric(df_portfolj["Årlig utdelning"], errors="coerce").fillna(0)

    # Beräkna totalt värde och utdelning
    df_portfolj["Värde (SEK)"] = df_portfolj["Antal aktier"] * df_portfolj["Aktuell kurs"]
    df_portfolj["Total utdelning (SEK)"] = df_portfolj["Antal aktier"] * df_portfolj["Årlig utdelning"]

    totalt_varde = df_portfolj["Värde (SEK)"].sum()
    total_utdelning = df_portfolj["Total utdelning (SEK)"].sum()
    utdelning_per_manad = total_utdelning / 12

    st.metric("💰 Totalt portföljvärde (SEK)", f"{totalt_varde:,.0f}")
    st.metric("📈 Kommande årlig utdelning (SEK)", f"{total_utdelning:,.0f}")
    st.metric("📆 Genomsnittlig utdelning per månad", f"{utdelning_per_manad:,.0f} SEK")

    st.markdown("### Bolag i din portfölj")
    st.dataframe(df_portfolj[[
        "Ticker", "Bolagsnamn", "Antal aktier", "Aktuell kurs", "Värde (SEK)",
        "Årlig utdelning", "Total utdelning (SEK)"
    ]].sort_values(by="Värde (SEK)", ascending=False), use_container_width=True)

def investeringsförslag(df):
    st.subheader("💡 Investeringsförslag")

    if df.empty or "Aktuell kurs" not in df.columns:
        st.warning("Databasen är tom eller saknar kolumnen 'Aktuell kurs'.")
        return

    riktkursval = st.selectbox("Sortera efter uppsida i riktkurs:", [
        "Riktkurs", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år"
    ])

    if riktkursval not in df.columns:
        st.warning(f"Kolumnen '{riktkursval}' finns inte i databasen.")
        return

    df["Aktuell kurs"] = pd.to_numeric(df["Aktuell kurs"], errors="coerce")
    df[riktkursval] = pd.to_numeric(df[riktkursval], errors="coerce")
    df = df.dropna(subset=["Aktuell kurs", riktkursval])

    df["Uppside (%)"] = ((df[riktkursval] - df["Aktuell kurs"]) / df["Aktuell kurs"]) * 100
    df = df.sort_values(by="Uppside (%)", ascending=False).reset_index(drop=True)

    if df.empty:
        st.info("Inga bolag med giltig uppsida.")
        return

    st.markdown(f"### {len(df)} bolag med positiv uppsida ({riktkursval})")

    index = st.number_input("Visa bolag:", min_value=0, max_value=len(df) - 1, step=1, value=0)

    bolag = df.iloc[index]
    st.markdown(f"## {bolag['Bolagsnamn']} ({bolag['Ticker']})")
    st.write(f"Aktuell kurs: {bolag['Aktuell kurs']:.2f} {bolag.get('Valuta', '')}")
    st.write(f"Riktkurs nu: {bolag.get('Riktkurs', '')}")
    st.write(f"Riktkurs om 1 år: {bolag.get('Riktkurs om 1 år', '')}")
    st.write(f"Riktkurs om 2 år: {bolag.get('Riktkurs om 2 år', '')}")
    st.write(f"Riktkurs om 3 år: {bolag.get('Riktkurs om 3 år', '')}")
    st.metric("Uppside (%)", f"{bolag['Uppside (%)']:.1f}%")

    tillgängligt_belopp = st.number_input("Tillgängligt belopp (SEK):", min_value=0, value=0)

    if tillgängligt_belopp > 0 and bolag["Aktuell kurs"] > 0:
        antal_köpbara = int(tillgängligt_belopp // bolag["Aktuell kurs"])
        äger = str(bolag.get("Äger", "")).lower() == "ja"
        befintliga = int(bolag.get("Antal aktier", 0)) if äger else 0
        ny_total = befintliga + antal_köpbara
        nuvärde = befintliga * bolag["Aktuell kurs"]
        framtida_värde = ny_total * bolag["Aktuell kurs"]

        st.markdown(f"**Köpbara aktier:** {antal_köpbara}")
        st.markdown(f"**Äger redan:** {befintliga}")
        st.markdown(f"**Nuvarande andel av portföljen:** {nuvärde:.0f} SEK")
        st.markdown(f"**Efter köp (potentiellt):** {framtida_värde:.0f} SEK")

    st.write("---")
    st.markdown("Visa nästa bolag med bläddringsfunktion:")

    kol1, kol2 = st.columns(2)
    with kol1:
        if st.button("⬅️ Föregående", key="föregående") and index > 0:
            st.experimental_set_query_params(index=index - 1)
            st.rerun()
    with kol2:
        if st.button("➡️ Nästa", key="nästa") and index < len(df) - 1:
            st.experimental_set_query_params(index=index + 1)
            st.rerun()

def main():
    st.title("📊 Aktieanalys och investeringsförslag")
    menyval = st.sidebar.radio("Meny", [
        "Lägg till / uppdatera bolag",
        "Analys",
        "Portfölj",
        "Investeringsförslag",
        "Uppdatera alla bolag"
    ])

    df = hämta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)
    df = beräkna_allt(df)

    if menyval == "Lägg till / uppdatera bolag":
        formulär(df)

    elif menyval == "Analys":
        visa_analys(df)

    elif menyval == "Portfölj":
        visa_portfolj(df)

    elif menyval == "Investeringsförslag":
        investeringsförslag(df)

    elif menyval == "Uppdatera alla bolag":
        massuppdatera(df)


if __name__ == "__main__":
    main()
