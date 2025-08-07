import streamlit as st
import pandas as pd
import yfinance as yf
import gspread
import time
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

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
    nödvändiga_kolumner = [
        "Ticker", "Bolagsnamn", "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "P/S-snitt", "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år", "Antal aktier",
        "Valuta", "Aktuell kurs", "Årlig utdelning", "CAGR 5 år (%)", "Uppside (%)", "Äger"
    ]
    for kolumn in nödvändiga_kolumner:
        if kolumn not in df.columns:
            df[kolumn] = ""

    # Ta bort oönskade kolumner
    oönskade_kolumner = ["P/S idag", "Riktkurs 2026", "Riktkurs 2027", "Riktkurs 2028",
                         "Omsättning om 4 år", "Yahoo ticker", "Max andel", "P/S metod",
                         "Initierad", "Bolag", 21]
    df = df.drop(columns=[c for c in oönskade_kolumner if c in df.columns], errors="ignore")

    # Undvik dubletter
    df = df.loc[:, ~df.columns.duplicated()]
    return df

def hämta_data_från_yahoo(ticker):
    try:
        aktie = yf.Ticker(ticker)
        info = aktie.info
        namn = info.get("longName", "")
        kurs = info.get("currentPrice", "")
        valuta = info.get("currency", "")
        utdelning = info.get("dividendRate", "")
        cagr_5år = info.get("revenueGrowth", "")
        return {
            "Bolagsnamn": namn,
            "Aktuell kurs": kurs,
            "Valuta": valuta,
            "Årlig utdelning": utdelning,
            "CAGR 5 år (%)": round(cagr_5år * 100, 2) if cagr_5år else ""
        }
    except:
        return {}

def konvertera_typer(df):
    numeriska = [
        "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "Omsättning idag", "Omsättning nästa år",
        "Omsättning om 2 år", "Omsättning om 3 år", "P/S-snitt", "Riktkurs idag", "Riktkurs om 1 år",
        "Riktkurs om 2 år", "Riktkurs om 3 år", "Antal aktier", "Aktuell kurs", "Årlig utdelning",
        "CAGR 5 år (%)", "Uppside (%)", "Utestående aktier"
    ]
    for kolumn in numeriska:
        if kolumn in df.columns:
            df[kolumn] = pd.to_numeric(df[kolumn], errors="coerce")
    return df

def beräkna_allt(df):
    df = konvertera_typer(df)

    # Beräkna P/S-snitt
    df["P/S-snitt"] = df[["P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"]].mean(axis=1)

    # Justera CAGR och beräkna framtida omsättning
    justerad_cagr = df["CAGR 5 år (%)"].copy()
    justerad_cagr[df["CAGR 5 år (%)"] > 100] = 50
    justerad_cagr[df["CAGR 5 år (%)"] < 0] = 2

    df["Omsättning om 2 år"] = df["Omsättning nästa år"] * (1 + justerad_cagr / 100)
    df["Omsättning om 3 år"] = df["Omsättning nästa år"] * ((1 + justerad_cagr / 100) ** 2)

    # Beräkna riktkurser
    df["Riktkurs idag"] = df["Omsättning idag"] * df["P/S-snitt"] / df["Utestående aktier"]
    df["Riktkurs om 1 år"] = df["Omsättning nästa år"] * df["P/S-snitt"] / df["Utestående aktier"]
    df["Riktkurs om 2 år"] = df["Omsättning om 2 år"] * df["P/S-snitt"] / df["Utestående aktier"]
    df["Riktkurs om 3 år"] = df["Omsättning om 3 år"] * df["P/S-snitt"] / df["Utestående aktier"]

    return df

def formulär(df):
    st.subheader("Lägg till eller uppdatera bolag")
    with st.form("nytt_bolag"):
        ticker = st.text_input("Ticker (obligatorisk)").upper()
        utestående_aktier = st.number_input("Utestående aktier", min_value=0.0, step=1.0)
        ps = st.number_input("P/S", min_value=0.0, step=0.1)
        ps_q1 = st.number_input("P/S Q1", min_value=0.0, step=0.1)
        ps_q2 = st.number_input("P/S Q2", min_value=0.0, step=0.1)
        ps_q3 = st.number_input("P/S Q3", min_value=0.0, step=0.1)
        ps_q4 = st.number_input("P/S Q4", min_value=0.0, step=0.1)
        omsättning_idag = st.number_input("Omsättning idag", min_value=0.0, step=1.0)
        omsättning_nästa = st.number_input("Omsättning nästa år", min_value=0.0, step=1.0)
        antal_aktier = st.number_input("Antal aktier du äger", min_value=0.0, step=1.0)
        äger = st.selectbox("Äger du aktien?", ["Ja", "Nej"])
        spara = st.form_submit_button("Spara")

    if spara and ticker:
        data = {
            "Ticker": ticker,
            "Utestående aktier": utestående_aktier,
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
            data["Bolagsnamn"] = info.get("longName", "")
            data["Aktuell kurs"] = info.get("currentPrice", 0.0)
            data["Valuta"] = info.get("currency", "")
            data["Årlig utdelning"] = info.get("dividendRate", 0.0)

            # CAGR hämtas om möjligt
            cagr = info.get("fiveYearAvgDividendYield")
            if cagr is not None:
                data["CAGR 5 år (%)"] = round(cagr, 2)
            else:
                data["CAGR 5 år (%)"] = 0.0

            data["Datakälla"] = "Yahoo Finance"
        except:
            data["Datakälla"] = "Manuell inmatning"

        # Uppdatera eller lägg till
        if ticker in df["Ticker"].values:
            df.loc[df["Ticker"] == ticker, data.keys()] = data.values()
        else:
            df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)

        df = beräkna_allt(df)
        spara_data(df)
        st.success(f"{ticker} har sparats.")

    return df

def visa_portfolj(df):
    st.subheader("📊 Portföljöversikt")

    df = df[df["Äger"].str.lower() == "ja"]

    if df.empty:
        st.info("Du äger inga aktier just nu.")
        return

    df["Värde (SEK)"] = df["Antal aktier"] * df["Aktuell kurs"]
    df["Utdelning totalt (SEK)"] = df["Antal aktier"] * df["Årlig utdelning"]

    totalt_värde = df["Värde (SEK)"].sum()
    total_utdelning = df["Utdelning totalt (SEK)"].sum()
    utdelning_per_månad = total_utdelning / 12

    st.metric("Totalt portföljvärde (SEK)", f"{totalt_värde:,.0f}")
    st.metric("Total årlig utdelning (SEK)", f"{total_utdelning:,.0f}")
    st.metric("Utdelning per månad (snitt)", f"{utdelning_per_månad:,.0f}")

    st.dataframe(df[["Ticker", "Bolagsnamn", "Antal aktier", "Aktuell kurs", "Värde (SEK)", "Årlig utdelning", "Utdelning totalt (SEK)"]])

def analysvy(df):
    st.subheader("🔍 Analys")

    val = st.selectbox("Välj bolag att visa", df["Ticker"].unique())
    vald_rad = df[df["Ticker"] == val]

    if not vald_rad.empty:
        st.write("**Data för valt bolag:**")
        st.dataframe(vald_rad)

    st.write("**Databasen i sin helhet:**")
    st.dataframe(df)

def investeringsförslag(df):
    st.subheader("💡 Investeringsförslag")

    riktkursval = st.selectbox("Filtrera efter riktkurs", ["Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år"])
    df = df.copy()
    df = df[df["Aktuell kurs"] > 0]
    df["Uppside (%)"] = ((df[riktkursval] - df["Aktuell kurs"]) / df["Aktuell kurs"]) * 100
    df = df.sort_values("Uppside (%)", ascending=False).reset_index(drop=True)

    if df.empty:
        st.info("Inga bolag matchar.")
        return

    index = st.number_input("Visa bolag nummer", min_value=0, max_value=len(df)-1, step=1)

    rad = df.iloc[index]
    st.markdown(f"### {rad['Bolagsnamn']} ({rad['Ticker']})")
    st.write(f"Aktuell kurs: {rad['Aktuell kurs']:.2f} {rad['Valuta']}")
    st.write(f"{riktkursval}: {rad[riktkursval]:.2f} {rad['Valuta']}")
    st.write(f"Uppside: {rad['Uppside (%)']:.1f} %")

    tillgängligt_belopp = st.number_input("Tillgängligt belopp (SEK)", min_value=0.0, step=100.0)
    if tillgängligt_belopp > 0:
        köpbara = tillgängligt_belopp // rad["Aktuell kurs"]
        redan_äger = rad["Antal aktier"]
        framtida_antal = redan_äger + köpbara
        nuvarande_andel = (rad["Antal aktier"] * rad["Aktuell kurs"]) / df["Aktuell kurs"].mul(df["Antal aktier"]).sum() * 100
        framtida_andel = (framtida_antal * rad["Aktuell kurs"]) / df["Aktuell kurs"].mul(df["Antal aktier"]).sum() * 100

        st.write(f"Du kan köpa **{int(köpbara)}** aktier.")
        st.write(f"Du äger redan **{int(redan_äger)}** aktier.")
        st.write(f"Nuvarande portföljandel: **{nuvarande_andel:.1f}%**")
        st.write(f"Portföljandel efter köp: **{framtida_andel:.1f}%**")

def main():
    st.title("📈 Aktieanalys och investeringsförslag")
    df = hämta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)
    df = beräkna_allt(df)

    menyval = st.sidebar.radio("Meny", [
        "Lägg till / uppdatera bolag",
        "Analys",
        "Portfölj",
        "Investeringsförslag",
        "Massuppdatera alla"
    ])

    if menyval == "Lägg till / uppdatera bolag":
        formulär(df)

    elif menyval == "Analys":
        analysvy(df)

    elif menyval == "Portfölj":
        visa_portfolj(df)

    elif menyval == "Investeringsförslag":
        investeringsförslag(df)

    elif menyval == "Massuppdatera alla":
        if st.button("Starta massuppdatering"):
            df = massuppdatera_alla(df)
            spara_data(df)
            st.success("Massuppdatering klar.")

if __name__ == "__main__":
    main()
