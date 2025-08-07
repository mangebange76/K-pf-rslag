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
        "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
        "Antal aktier", "Valuta", "Årlig utdelning", "Äger", "CAGR 5 år (%)"
    ]

    onödiga_kolumner = [
        "P/S idag", "Riktkurs 2026", "Riktkurs 2027", "Riktkurs 2028",
        "Yahoo ticker", "Bolag", "Max andel", "Omsättning om 4 år", "P/S metod", "Initierad", "21"
    ]

    for kolumn in önskade_kolumner:
        if kolumn not in df.columns:
            df[kolumn] = ""

    for kolumn in df.columns:
        if kolumn not in önskade_kolumner:
            if kolumn in onödiga_kolumner:
                df.drop(columns=[kolumn], inplace=True)

    return df

def konvertera_typer(df):
    numeriska_kolumner = [
        "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "P/S-snitt",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
        "Antal aktier", "Årlig utdelning", "CAGR 5 år (%)"
    ]
    for kolumn in numeriska_kolumner:
        if kolumn in df.columns:
            df[kolumn] = pd.to_numeric(df[kolumn], errors="coerce")
    return df

def beräkna_allt(df):
    df = konvertera_typer(df)

    # P/S-snitt
    df["P/S-snitt"] = df[["P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"]].mean(axis=1)

    # Justerad CAGR
    justerad_cagr = []
    for cagr in df["CAGR 5 år (%)"]:
        if pd.isna(cagr):
            justerad_cagr.append(None)
        elif cagr > 100:
            justerad_cagr.append(50 / 100)
        elif cagr < 0:
            justerad_cagr.append(0.02)
        else:
            justerad_cagr.append(cagr / 100)
    df["Justerad CAGR"] = justerad_cagr

    # Omsättning om 2 år och 3 år
    df["Omsättning om 2 år"] = df["Omsättning nästa år"] * (1 + df["Justerad CAGR"])
    df["Omsättning om 3 år"] = df["Omsättning nästa år"] * (1 + df["Justerad CAGR"]) ** 2

    # Riktkurser
    df["Riktkurs idag"] = df["Omsättning idag"] * df["P/S-snitt"] / df["Utestående aktier"]
    df["Riktkurs om 1 år"] = df["Omsättning nästa år"] * df["P/S-snitt"] / df["Utestående aktier"]
    df["Riktkurs om 2 år"] = df["Omsättning om 2 år"] * df["P/S-snitt"] / df["Utestående aktier"]
    df["Riktkurs om 3 år"] = df["Omsättning om 3 år"] * df["P/S-snitt"] / df["Utestående aktier"]

    return df

def visa_portfolj(df):
    st.subheader("📦 Portföljsammanställning")

    df = df[df["Äger"].str.lower() == "ja"]

    if df.empty:
        st.info("Du äger inga bolag just nu.")
        return

    # Aktuell kurs i SEK (kurs * valutakurs) – just nu ej omräkning
    df["Värde (SEK)"] = df["Kurs"] * df["Antal aktier"]

    # Kommande utdelning
    df["Kommande utdelning (SEK)"] = df["Årlig utdelning"] * df["Antal aktier"]

    totalt_värde = df["Värde (SEK)"].sum()
    total_utdelning = df["Kommande utdelning (SEK)"].sum()
    utdelning_per_månad = total_utdelning / 12

    col1, col2, col3 = st.columns(3)
    col1.metric("Totalt portföljvärde", f"{totalt_värde:,.0f} SEK")
    col2.metric("Total kommande utdelning", f"{total_utdelning:,.0f} SEK")
    col3.metric("Utdelning per månad (snitt)", f"{utdelning_per_månad:,.0f} SEK")

    st.dataframe(df[["Ticker", "Bolagsnamn", "Antal aktier", "Kurs", "Värde (SEK)", "Årlig utdelning", "Kommande utdelning (SEK)"]])

def analysvy(df):
    st.subheader("📈 Analys")

    sorteringsval = st.selectbox("Sortera bolag efter uppsida i riktkurs:", ["Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år"])

    kolumn_map = {
        "Riktkurs idag": ("Riktkurs idag", "Uppside (%) idag"),
        "Riktkurs om 1 år": ("Riktkurs om 1 år", "Uppside (%) 1 år"),
        "Riktkurs om 2 år": ("Riktkurs om 2 år", "Uppside (%) 2 år"),
        "Riktkurs om 3 år": ("Riktkurs om 3 år", "Uppside (%) 3 år")
    }

    riktkurs_kol, uppsida_kol = kolumn_map[sorteringsval]

    df = df.copy()
    df[uppsida_kol] = (df[riktkurs_kol] - df["Kurs"]) / df["Kurs"] * 100
    df = df.sort_values(by=uppsida_kol, ascending=False)

    if not df.empty:
        index = st.session_state.get("bolags_index", 0)
        max_index = len(df) - 1

        if st.button("⬅️ Föregående") and index > 0:
            index -= 1
        if st.button("➡️ Nästa") and index < max_index:
            index += 1

        st.session_state["bolags_index"] = index
        bolag = df.iloc[index]

        st.markdown(f"### {bolag['Ticker']} – {bolag['Bolagsnamn']}")
        st.write(f"Aktuell kurs: {bolag['Kurs']:.2f}")
        st.write(f"Riktkurs idag: {bolag['Riktkurs idag']:.2f}")
        st.write(f"Riktkurs om 1 år: {bolag['Riktkurs om 1 år']:.2f}")
        st.write(f"Riktkurs om 2 år: {bolag['Riktkurs om 2 år']:.2f}")
        st.write(f"Riktkurs om 3 år: {bolag['Riktkurs om 3 år']:.2f}")
        st.write(f"Uppside enligt {sorteringsval}: {bolag[uppsida_kol]:.1f} %")

        tillgängligt_belopp = st.number_input("Tillgängligt belopp (SEK)", value=0)
        if tillgängligt_belopp > 0 and bolag["Kurs"] > 0:
            antal_köp = int(tillgängligt_belopp // bolag["Kurs"])
            st.write(f"Du kan köpa **{antal_köp} st** aktier för det beloppet.")

            if bolag["Antal aktier"] > 0:
                portföljvärde = (df["Kurs"] * df["Antal aktier"]).sum()
                nuvarande_andel = bolag["Kurs"] * bolag["Antal aktier"] / portföljvärde * 100 if portföljvärde else 0
                ny_andel = bolag["Kurs"] * (bolag["Antal aktier"] + antal_köp) / portföljvärde * 100 if portföljvärde else 0
                st.write(f"Nuvarande portföljandel: {nuvarande_andel:.2f} %")
                st.write(f"Andel efter köp: {ny_andel:.2f} %")

    st.markdown("---")
    st.subheader("📋 Hela databasen")
    st.dataframe(df)

def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    df = hamta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)
    df = beräkna_allt(df)

    meny = st.sidebar.radio("Meny", ["Lägg till / uppdatera bolag", "Analys", "Portfölj", "Investeringsförslag", "Massuppdatering"])

    if meny == "Lägg till / uppdatera bolag":
        formulär(df)
    elif meny == "Analys":
        analysvy(df)
    elif meny == "Portfölj":
        visa_portfolj(df)
    elif meny == "Investeringsförslag":
        investeringsvy(df)
    elif meny == "Massuppdatering":
        massuppdatera(df)

if __name__ == "__main__":
    main()
