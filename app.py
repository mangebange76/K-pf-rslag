import streamlit as st
import pandas as pd
import gspread
import yfinance as yf
import time
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

SHEET_URL = st.secrets["SHEET_URL"]
ARKNAMN = "Blad1"

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

def skapa_koppling():
    return client.open_by_url(SHEET_URL).worksheet(ARKNAMN)

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
    for kol in kolumner:
        if kol not in df.columns:
            df[kol] = ""
    df = df[kolumner]
    return df

def hämta_data_fran_yahoo(ticker):
    try:
        aktie = yf.Ticker(ticker)
        info = aktie.info
        namn = info.get("longName") or info.get("shortName") or ""
        kurs = info.get("currentPrice") or info.get("regularMarketPrice") or ""
        valuta = info.get("currency") or ""
        utdelning = info.get("dividendRate") or 0.0

        # Omsättning år 1 och 5 från revenue forecast (för att beräkna CAGR)
        revenue_forecast = aktie.financials
        cagr = None

        try:
            revenue_df = aktie.financials.T
            oms_years = revenue_df["Total Revenue"].dropna()
            if len(oms_years) >= 5:
                year_1 = oms_years.iloc[-5]
                year_5 = oms_years.iloc[-1]
                if year_1 > 0 and year_5 > 0:
                    cagr = round(((year_5 / year_1) ** (1 / 5) - 1) * 100, 2)
        except:
            pass

        return {
            "Bolagsnamn": namn,
            "Aktuell kurs": kurs,
            "Valuta": valuta,
            "Årlig utdelning": utdelning,
            "CAGR 5 år (%)": cagr
        }

    except Exception as e:
        return {}

def beräkna_kolumner(df):
    for i, row in df.iterrows():
        try:
            ps_varden = []
            for kolumn in ["P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"]:
                try:
                    ps_varden.append(float(row[kolumn]))
                except:
                    pass
            ps_snitt = sum(ps_varden) / len(ps_varden) if ps_varden else 0

            try:
                cagr = float(row["CAGR 5 år (%)"])
                if cagr > 100:
                    tillväxt = 0.5
                elif cagr < 0:
                    tillväxt = 0.02
                else:
                    tillväxt = cagr / 100
            except:
                tillväxt = 0.02

            oms_idag = float(row["Omsättning idag"]) if str(row["Omsättning idag"]).replace(",", ".").replace(" ", "").replace("-", "").replace("nan", "") else 0
            oms_2 = oms_idag * (1 + tillväxt) ** 2
            oms_3 = oms_idag * (1 + tillväxt) ** 3

            riktkurs_idag = oms_idag * ps_snitt
            riktkurs_1 = float(row["Omsättning nästa år"]) * ps_snitt if str(row["Omsättning nästa år"]).replace(",", ".").replace(" ", "").replace("-", "").replace("nan", "") else 0
            riktkurs_2 = oms_2 * ps_snitt
            riktkurs_3 = oms_3 * ps_snitt

            df.at[i, "P/S-snitt"] = round(ps_snitt, 2)
            df.at[i, "Omsättning om 2 år"] = round(oms_2, 0)
            df.at[i, "Omsättning om 3 år"] = round(oms_3, 0)
            df.at[i, "Riktkurs idag"] = round(riktkurs_idag, 2)
            df.at[i, "Riktkurs om 1 år"] = round(riktkurs_1, 2)
            df.at[i, "Riktkurs om 2 år"] = round(riktkurs_2, 2)
            df.at[i, "Riktkurs om 3 år"] = round(riktkurs_3, 2)
        except:
            continue
    return df

def formulär(df):
    st.subheader("Lägg till eller uppdatera bolag")

    alla_tickers = df["Ticker"].tolist()
    valt_bolag = st.selectbox("Välj bolag att uppdatera eller lämna tomt för nytt", [""] + alla_tickers)

    if valt_bolag:
        data = df[df["Ticker"] == valt_bolag].iloc[0].to_dict()
    else:
        data = {kolumn: "" for kolumn in df.columns}

    with st.form("nytt_bolag"):
        ticker = st.text_input("Ticker", value=data.get("Ticker", ""))
        utestående = st.number_input("Utestående aktier (miljoner)", value=float(data.get("Utestående aktier", 0)), step=1.0)
        ps = st.number_input("P/S", value=float(data.get("P/S", 0)), step=0.1)
        ps_q1 = st.number_input("P/S Q1", value=float(data.get("P/S Q1", 0)), step=0.1)
        ps_q2 = st.number_input("P/S Q2", value=float(data.get("P/S Q2", 0)), step=0.1)
        ps_q3 = st.number_input("P/S Q3", value=float(data.get("P/S Q3", 0)), step=0.1)
        ps_q4 = st.number_input("P/S Q4", value=float(data.get("P/S Q4", 0)), step=0.1)
        oms_idag = st.number_input("Omsättning idag", value=float(data.get("Omsättning idag", 0)), step=1000000.0)
        oms_nästa = st.number_input("Omsättning nästa år", value=float(data.get("Omsättning nästa år", 0)), step=1000000.0)
        antal_aktier = st.number_input("Antal aktier", value=float(data.get("Antal aktier", 0)), step=1.0)

        sparaknapp = st.form_submit_button("Spara bolag")

    if sparaknapp:
        ny_rad = {
            "Ticker": ticker,
            "Utestående aktier": utestående,
            "P/S": ps,
            "P/S Q1": ps_q1,
            "P/S Q2": ps_q2,
            "P/S Q3": ps_q3,
            "P/S Q4": ps_q4,
            "Omsättning idag": oms_idag,
            "Omsättning nästa år": oms_nästa,
            "Antal aktier": antal_aktier,
        }

        # Hämta från Yahoo Finance
        yahoo_data = hämta_data_fran_yahoo(ticker)
        for nyckel, värde in yahoo_data.items():
            ny_rad[nyckel] = värde

        # Fyll i tomma kolumner
        for kolumn in df.columns:
            if kolumn not in ny_rad:
                ny_rad[kolumn] = ""

        # Lägg till eller ersätt rad
        df = df[df["Ticker"] != ticker]
        df = pd.concat([df, pd.DataFrame([ny_rad])], ignore_index=True)
        df = beräkna_kolumner(df)
        spara_data(df)
        st.success("Bolaget har sparats och beräkningar uppdaterats!")

    return df

def analysvy(df):
    st.subheader("Analys")

    sorteringsval = st.selectbox("Sortera efter uppsida i riktkurs", ["Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år"])
    df["Uppside (%)"] = ((df[sorteringsval] - df["Aktuell kurs"]) / df["Aktuell kurs"]) * 100
    df = df.sort_values("Uppside (%)", ascending=False)

    index = st.number_input("Visa bolag nr", min_value=1, max_value=len(df), value=1)
    rad = df.iloc[index - 1]

    st.write(f"**{rad['Ticker']} – {rad['Bolagsnamn']}**")
    st.write(f"Aktuell kurs: {rad['Aktuell kurs']}")
    st.write(f"Riktkurs idag: {rad['Riktkurs idag']}")
    st.write(f"Riktkurs om 1 år: {rad['Riktkurs om 1 år']}")
    st.write(f"Riktkurs om 2 år: {rad['Riktkurs om 2 år']}")
    st.write(f"Riktkurs om 3 år: {rad['Riktkurs om 3 år']}")
    st.write(f"Uppside: {rad['Uppside (%)']:.1f}%")

    st.write("—" * 50)
    st.write("**Hela databasen:**")
    st.dataframe(df)


def visa_portfolj(df):
    st.subheader("Portfölj")

    df_portfölj = df[df["Antal aktier"] > 0].copy()
    if df_portfölj.empty:
        st.info("Du har inga bolag i portföljen.")
        return

    df_portfölj["Värde (SEK)"] = df_portfölj["Antal aktier"] * df_portfölj["Aktuell kurs"]
    df_portfölj["Utdelning (SEK)"] = df_portfölj["Antal aktier"] * df_portfölj["Årlig utdelning"]

    total_värde = df_portfölj["Värde (SEK)"].sum()
    total_utdelning = df_portfölj["Utdelning (SEK)"].sum()

    st.write(f"**Totalt portföljvärde:** {total_värde:,.0f} SEK")
    st.write(f"**Total utdelning per år:** {total_utdelning:,.0f} SEK")
    st.write(f"**Utdelning per månad (snitt):** {total_utdelning/12:,.0f} SEK")

    st.dataframe(df_portfölj[["Ticker", "Antal aktier", "Aktuell kurs", "Värde (SEK)", "Årlig utdelning", "Utdelning (SEK)"]])


def investeringsförslag(df):
    st.subheader("Investeringsförslag")

    riktval = st.selectbox("Välj riktkurs att utgå från", ["Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år"])
    tillgängligt = st.number_input("Tillgängligt belopp att investera (SEK)", min_value=0.0, step=100.0)

    df["Uppside (%)"] = ((df[riktval] - df["Aktuell kurs"]) / df["Aktuell kurs"]) * 100
    df_sorted = df.sort_values("Uppside (%)", ascending=False)

    index = st.number_input("Visa förslag nr", min_value=1, max_value=len(df_sorted), value=1)
    rad = df_sorted.iloc[index - 1]

    st.write(f"**{rad['Ticker']} – {rad['Bolagsnamn']}**")
    st.write(f"Aktuell kurs: {rad['Aktuell kurs']}")
    st.write(f"Vald riktkurs: {rad[riktval]}")
    st.write(f"Uppside: {rad['Uppside (%)']:.1f}%")

    if tillgängligt > 0:
        köpbara = int(tillgängligt // rad["Aktuell kurs"])
        äger = rad["Antal aktier"]
        nuvärde = äger * rad["Aktuell kurs"]
        framtida_värde = (äger + köpbara) * rad[riktval]

        st.write(f"Du kan köpa **{köpbara} aktier**.")
        st.write(f"Nuvarande portföljandel: {nuvärde:,.0f} SEK")
        st.write(f"Portföljandel efter köp: {framtida_värde:,.0f} SEK")

    st.write("—" * 50)
    st.dataframe(df_sorted[["Ticker", "Aktuell kurs", riktval, "Uppside (%)"]])

def main():
    st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")
    st.title("📊 Aktieanalys och investeringsförslag")

    df = hämta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)
    df = beräkna_kolumner(df)

    meny = st.sidebar.radio("Meny", ["Lägg till / uppdatera bolag", "Analys", "Portfölj", "Investeringsförslag", "Uppdatera alla bolag"])

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
