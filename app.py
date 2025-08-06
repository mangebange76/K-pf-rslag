import streamlit as st
import pandas as pd
import numpy as np
import gspread
import yfinance as yf
import requests
import time
from bs4 import BeautifulSoup
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
    data = skapa_koppling().get_all_records()
    return pd.DataFrame(data)

def spara_data(df):
    sheet = skapa_koppling()
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.astype(str).values.tolist())

def konvertera_typer(df):
    kolumner = [
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Aktuell kurs", "Antal aktier"
    ]
    for kol in kolumner:
        if kol in df.columns:
            df[kol] = pd.to_numeric(df[kol], errors="coerce").fillna(0.0)
    return df

def säkerställ_kolumner(df):
    nödvändiga = [
        "Ticker", "Bolagsnamn", "Valuta", "Aktuell kurs", "Utestående aktier",
        "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "P/S-snitt", "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år", "Antal aktier"
    ]
    for kol in nödvändiga:
        if kol not in df.columns:
            if kol in ["Ticker", "Bolagsnamn", "Valuta"]:
                df[kol] = ""
            else:
                df[kol] = 0.0
    return df

def parse_yahoo_number(val):
    try:
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return float(val)
        val = val.replace(",", "").replace(" ", "").upper()
        mult = 1
        if val.endswith("T"):
            mult = 1_000_000_000_000
            val = val[:-1]
        elif val.endswith("B"):
            mult = 1_000_000_000
            val = val[:-1]
        elif val.endswith("M"):
            mult = 1_000_000
            val = val[:-1]
        elif val.endswith("K"):
            mult = 1_000
            val = val[:-1]
        return float(val) * mult
    except:
        return None


def scrape_yahoo_finance(ticker):
    headers = {"User-Agent": "Mozilla/5.0"}
    result = {"Aktuell kurs": None, "Valuta": None, "P/S": None, "Omsättning idag": None, "Omsättning nästa år": None}

    try:
        main_html = requests.get(f"https://finance.yahoo.com/quote/{ticker}", headers=headers, timeout=10).text
        soup_main = BeautifulSoup(main_html, "html.parser")

        price_tag = soup_main.find("fin-streamer", {"data-field": "regularMarketPrice"})
        if price_tag:
            try:
                kurs = float(price_tag.get_text(strip=True).replace(",", ""))
                if 0.01 < kurs < 5000:
                    result["Aktuell kurs"] = kurs
            except:
                pass

        currency_tag = soup_main.find("span", string=lambda x: x and "Currency" in x)
        if currency_tag:
            result["Valuta"] = currency_tag.get_text(strip=True).split()[-1]

        stats_html = requests.get(f"https://finance.yahoo.com/quote/{ticker}/key-statistics", headers=headers, timeout=10).text
        soup_stats = BeautifulSoup(stats_html, "html.parser")
        for row in soup_stats.find_all("tr"):
            cols = [c.get_text(strip=True) for c in row.find_all("td")]
            if len(cols) == 2 and "Price/Sales" in cols[0]:
                ps_val = parse_yahoo_number(cols[1])
                if ps_val is not None:
                    result["P/S"] = min(ps_val, 100)  # mjuk tak

        analysis_html = requests.get(f"https://finance.yahoo.com/quote/{ticker}/analysis", headers=headers, timeout=10).text
        soup_analysis = BeautifulSoup(analysis_html, "html.parser")
        for table in soup_analysis.find_all("table"):
            if "Revenue Estimate" in table.get_text():
                for row in table.find_all("tr"):
                    cols = [c.get_text(strip=True) for c in row.find_all("td")]
                    if len(cols) >= 3:
                        name = cols[0].lower()
                        if any(key in name for key in ["current year", "this year", "current yr"]):
                            result["Omsättning idag"] = parse_yahoo_number(cols[1])
                        elif any(key in name for key in ["next year", "next yr"]):
                            result["Omsättning nästa år"] = parse_yahoo_number(cols[1])
                break
    except:
        pass

    return result


def fallback_yfinance(ticker, data):
    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info

        if data.get("Aktuell kurs") is None:
            kurs = info.get("regularMarketPrice")
            if kurs and 0.01 < kurs < 5000:
                data["Aktuell kurs"] = kurs

        if data.get("Valuta") is None:
            data["Valuta"] = info.get("currency", "USD")

        if data.get("P/S") is None:
            mc = parse_yahoo_number(info.get("marketCap"))
            rev = parse_yahoo_number(info.get("totalRevenue"))
            if mc and rev:
                data["P/S"] = min(mc / rev, 100)

        if data.get("Omsättning idag") is None:
            data["Omsättning idag"] = parse_yahoo_number(info.get("totalRevenue"))

    except:
        pass
    return data


def uppdatera_berakningar(df):
    for i, rad in df.iterrows():
        ps_values = [min(v, 100) for v in [rad["P/S"], rad["P/S Q1"], rad["P/S Q2"], rad["P/S Q3"], rad["P/S Q4"]] if v > 0]
        ps_snitt = round(np.mean(ps_values), 2) if ps_values else 0
        df.at[i, "P/S-snitt"] = ps_snitt

        if rad["Utestående aktier"] > 0:
            df.at[i, "Riktkurs idag"] = round((rad["Omsättning idag"] * ps_snitt) / rad["Utestående aktier"], 2)
            df.at[i, "Riktkurs om 1 år"] = round((rad["Omsättning nästa år"] * ps_snitt) / rad["Utestående aktier"], 2)
            df.at[i, "Riktkurs om 2 år"] = round((rad["Omsättning om 2 år"] * ps_snitt) / rad["Utestående aktier"], 2)
            df.at[i, "Riktkurs om 3 år"] = round((rad["Omsättning om 3 år"] * ps_snitt) / rad["Utestående aktier"], 2)
    return df


def analysvy(df, valutakurser):
    st.subheader("📈 Analysläge")
    df = uppdatera_berakningar(df)

    if st.button("🔄 Uppdatera alla data från Yahoo"):
        fel_lista = []
        total = len(df)
        status = st.empty()

        for i, row in df.iterrows():
            ticker = str(row["Ticker"]).strip()
            status.text(f"🔄 Uppdaterar {i+1}/{total} ({ticker})...")

            data = scrape_yahoo_finance(ticker)
            data = fallback_yfinance(ticker, data)

            if data.get("Aktuell kurs") is not None:
                df.at[i, "Aktuell kurs"] = data["Aktuell kurs"]
            if data.get("Valuta"):
                df.at[i, "Valuta"] = data["Valuta"]
            if data.get("P/S") is not None:
                df.at[i, "P/S"] = data["P/S"]
            if data.get("Omsättning idag") is not None:
                df.at[i, "Omsättning idag"] = data["Omsättning idag"]
            if data.get("Omsättning nästa år") is not None:
                df.at[i, "Omsättning nästa år"] = data["Omsättning nästa år"]

            if df.at[i, "Omsättning idag"] > 0 and df.at[i, "Omsättning nästa år"] > 0:
                oms0 = df.at[i, "Omsättning idag"]
                oms1 = df.at[i, "Omsättning nästa år"]
                cagr = (oms1 / oms0) - 1
                cagr = min(max(cagr, 0.02), 0.50)
                df.at[i, "Omsättning om 2 år"] = oms1 * (1 + cagr)
                df.at[i, "Omsättning om 3 år"] = df.at[i, "Omsättning om 2 år"] * (1 + cagr)

            time.sleep(1)

        df = uppdatera_berakningar(df)
        spara_data(df)

    st.markdown("### 📋 Alla bolag i databasen")
    st.dataframe(df, use_container_width=True)

def lagg_till_eller_uppdatera(df):
    st.subheader("➕ Lägg till / uppdatera bolag")
    namn_map = {f"{rad['Bolagsnamn']} ({rad['Ticker']})": rad['Ticker'] for _, rad in df.iterrows()}
    valt = st.selectbox("Välj bolag att uppdatera (eller lämna tom för nytt)", [""] + sorted(namn_map.keys()))

    if valt:
        ticker_vald = namn_map[valt]
        befintlig = df[df["Ticker"] == ticker_vald].iloc[0]
    else:
        befintlig = pd.Series(dtype=object)

    with st.form("form"):
        ticker = st.text_input("Ticker", value=befintlig.get("Ticker", "") if not befintlig.empty else "").upper()
        namn = st.text_input("Bolagsnamn", value=befintlig.get("Bolagsnamn", "") if not befintlig.empty else "")
        valuta = st.selectbox("Valuta", ["USD", "NOK", "CAD", "SEK", "EUR"], 
                               index=0 if befintlig.empty else ["USD", "NOK", "CAD", "SEK", "EUR"].index(befintlig.get("Valuta", "USD")))
        kurs = st.number_input("Aktuell kurs", value=float(befintlig.get("Aktuell kurs", 0.0)) if not befintlig.empty else 0.0)
        aktier = st.number_input("Utestående aktier (miljoner)", value=float(befintlig.get("Utestående aktier", 0.0)) if not befintlig.empty else 0.0)
        antal_aktier = st.number_input("Antal aktier du äger", value=float(befintlig.get("Antal aktier", 0.0)) if not befintlig.empty else 0.0)

        ps_idag = st.number_input("P/S idag", value=float(befintlig.get("P/S", 0.0)) if not befintlig.empty else 0.0)
        ps1 = st.number_input("P/S Q1", value=float(befintlig.get("P/S Q1", 0.0)) if not befintlig.empty else 0.0)
        ps2 = st.number_input("P/S Q2", value=float(befintlig.get("P/S Q2", 0.0)) if not befintlig.empty else 0.0)
        ps3 = st.number_input("P/S Q3", value=float(befintlig.get("P/S Q3", 0.0)) if not befintlig.empty else 0.0)
        ps4 = st.number_input("P/S Q4", value=float(befintlig.get("P/S Q4", 0.0)) if not befintlig.empty else 0.0)

        oms_idag = st.number_input("Omsättning idag (miljoner)", value=float(befintlig.get("Omsättning idag", 0.0)) if not befintlig.empty else 0.0)
        oms_1 = st.number_input("Omsättning nästa år", value=float(befintlig.get("Omsättning nästa år", 0.0)) if not befintlig.empty else 0.0)
        oms_2 = st.number_input("Omsättning om 2 år", value=float(befintlig.get("Omsättning om 2 år", 0.0)) if not befintlig.empty else 0.0)
        oms_3 = st.number_input("Omsättning om 3 år", value=float(befintlig.get("Omsättning om 3 år", 0.0)) if not befintlig.empty else 0.0)

        sparaknapp = st.form_submit_button("💾 Spara")

    if sparaknapp and ticker:
        ny_rad = {
            "Ticker": ticker, "Bolagsnamn": namn, "Valuta": valuta, "Aktuell kurs": kurs,
            "Utestående aktier": aktier, "Antal aktier": antal_aktier,
            "P/S": ps_idag, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms_1,
            "Omsättning om 2 år": oms_2, "Omsättning om 3 år": oms_3
        }

        if ticker in df["Ticker"].values:
            df.loc[df["Ticker"] == ticker, ny_rad.keys()] = ny_rad.values()
            st.success(f"{ticker} uppdaterat.")
        else:
            df = pd.concat([df, pd.DataFrame([ny_rad])], ignore_index=True)
            st.success(f"{ticker} tillagt.")
    return df


def visa_investeringsforslag(df, valutakurser):
    st.subheader("💡 Investeringsförslag")
    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=500.0, step=500.0)
    riktkurs_val = st.selectbox("Vilken riktkurs ska användas?", ["Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år"], index=1)
    filterval = st.radio("Visa förslag för:", ["Alla bolag", "Endast portföljen"])

    df_portfolj = df[df["Antal aktier"] > 0].copy()
    df_portfolj["Värde (SEK)"] = df_portfolj.apply(lambda r: r["Antal aktier"] * r["Aktuell kurs"] * valutakurser.get(r["Valuta"], 1), axis=1)
    portfoljvarde = df_portfolj["Värde (SEK)"].sum()

    if filterval == "Endast portföljen":
        df_forslag = df_portfolj[df_portfolj[riktkurs_val] > df_portfolj["Aktuell kurs"]].copy()
    else:
        df_forslag = df[df[riktkurs_val] > df["Aktuell kurs"]].copy()

    df_forslag["Potential (%)"] = ((df_forslag[riktkurs_val] - df_forslag["Aktuell kurs"]) / df_forslag["Aktuell kurs"]) * 100
    df_forslag = df_forslag.sort_values(by="Potential (%)", ascending=False).reset_index(drop=True)

    if not df_forslag.empty:
        if 'forslags_index' not in st.session_state:
            st.session_state.forslags_index = 0
        index = st.session_state.forslags_index

        rad = df_forslag.iloc[index]
        antal = int((kapital_sek / valutakurser.get(rad["Valuta"], 1)) // rad["Aktuell kurs"])
        investering_sek = antal * rad["Aktuell kurs"] * valutakurser.get(rad["Valuta"], 1)
        nuvarande_innehav = df_portfolj[df_portfolj["Ticker"] == rad["Ticker"]]["Värde (SEK)"].sum()
        nuvarande_andel = round((nuvarande_innehav / portfoljvarde) * 100, 2) if portfoljvarde > 0 else 0
        ny_andel = round(((nuvarande_innehav + investering_sek) / portfoljvarde) * 100, 2) if portfoljvarde > 0 else 0

        st.markdown(f"""
            ### 💰 Förslag {index+1} av {len(df_forslag)}
            - **Bolag:** {rad['Bolagsnamn']} ({rad['Ticker']})
            - **Aktuell kurs:** {round(rad['Aktuell kurs'], 2)} {rad['Valuta']}
            - **{riktkurs_val}:** {round(rad[riktkurs_val], 2)} {rad['Valuta']}
            - **Potential:** {round(rad['Potential (%)'], 2)}%
            - **Antal att köpa:** {antal} st
            - **Beräknad investering:** {round(investering_sek, 2)} SEK
            - **Nuvarande andel i portföljen:** {nuvarande_andel}%
            - **Andel efter köp:** {ny_andel}%
        """)

        if st.button("➡️ Nästa förslag"):
            st.session_state.forslags_index += 1
    else:
        st.info("Inga bolag matchar kriterierna just nu.")

def visa_portfolj(df, valutakurser):
    st.subheader("📦 Min portfölj")
    df = df[df["Antal aktier"] > 0].copy()
    if df.empty:
        st.info("Du äger inga aktier.")
        return
    df["Värde (SEK)"] = df.apply(lambda r: r["Antal aktier"] * r["Aktuell kurs"] * valutakurser.get(r["Valuta"], 1), axis=1)
    df["Andel (%)"] = round(df["Värde (SEK)"] / df["Värde (SEK)"].sum() * 100, 2)
    total_varde = df["Värde (SEK)"].sum()
    st.markdown(f"**Totalt portföljvärde:** {round(total_varde, 2)} SEK")
    st.dataframe(df[["Ticker", "Bolagsnamn", "Valuta", "Antal aktier", "Aktuell kurs", "Värde (SEK)", "Andel (%)"]], use_container_width=True)


def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    df = hamta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)

    # Standardvalutakurser till SEK
    valutakurser = {
        "USD": st.sidebar.number_input("USD → SEK", value=9.50, step=0.01),
        "NOK": st.sidebar.number_input("NOK → SEK", value=0.93, step=0.01),
        "EUR": st.sidebar.number_input("EUR → SEK", value=11.10, step=0.01),
        "CAD": st.sidebar.number_input("CAD → SEK", value=7.00, step=0.01),
        "SEK": 1.0
    }

    meny = st.sidebar.radio("Meny", ["Analys", "Lägg till / uppdatera bolag", "Investeringsförslag", "Portfölj"])

    if meny == "Analys":
        analysvy(df, valutakurser)
    elif meny == "Lägg till / uppdatera bolag":
        df = lagg_till_eller_uppdatera(df)
        spara_data(df)
    elif meny == "Investeringsförslag":
        df = uppdatera_berakningar(df)
        visa_investeringsforslag(df, valutakurser)
    elif meny == "Portfölj":
        df = uppdatera_berakningar(df)
        visa_portfolj(df, valutakurser)


if __name__ == "__main__":
    main()
