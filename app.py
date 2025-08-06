import streamlit as st
import pandas as pd
import numpy as np
import gspread
import yfinance as yf
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

# ------------------------------------------------
# Google Sheets-konfiguration
# ------------------------------------------------
SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"  # Anpassat till din faktiska flik i Google Sheet

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
        "Ticker", "Bolagsnamn", "Antal aktier",
        "P/S idag", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "CAGR 5 år (%)", "Aktuell kurs", "Valuta",
        "P/S-snitt", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år"
    ]
    for kol in kolumner:
        if kol not in df.columns:
            df[kol] = ""
    return df

def konvertera_typer(df):
    numeriska = [
        "Antal aktier", "P/S idag", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "CAGR 5 år (%)", "Aktuell kurs", "P/S-snitt",
        "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år"
    ]
    for kol in numeriska:
        if kol in df.columns:
            df[kol] = pd.to_numeric(df[kol], errors="coerce")
    return df

# ------------------------------------------------
# Datahämtning och beräkningar
# ------------------------------------------------

def hamta_cagr_5ar(ticker):
    """Hämtar 5-års CAGR från Yahoo Finance historik."""
    try:
        hist = yf.Ticker(ticker).history(period="5y", interval="3mo")
        if hist.empty:
            return None
        start_val = hist["Close"].iloc[0]
        slut_val = hist["Close"].iloc[-1]
        år = 5
        cagr = ((slut_val / start_val) ** (1/år) - 1) * 100
        return round(cagr, 2)
    except Exception:
        return None

def berakna_framtida_omsattning(oms_next, cagr):
    """Beräknar omsättning om 2 och 3 år baserat på CAGR."""
    if oms_next in (None, "", 0) or cagr is None:
        return "", ""
    tillv = cagr / 100
    # År 2
    oms_ar2 = round(oms_next * (1 + tillv), 2)
    # År 3
    oms_ar3 = round(oms_ar2 * (1 + tillv), 2)
    return oms_ar2, oms_ar3

def hamta_bolagsinfo_yahoo(ticker):
    """Hämtar bolagsnamn, aktuell kurs och valuta från Yahoo Finance."""
    try:
        info = yf.Ticker(ticker).info
        namn = info.get("shortName", "")
        kurs = info.get("regularMarketPrice", "")
        valuta = info.get("currency", "")
        return namn, kurs, valuta
    except Exception:
        return "", "", ""

def lagg_till_eller_uppdatera(df):
    st.subheader("➕ Lägg till / uppdatera bolag")

    # Bläddringsfunktion + rullista
    if "bolags_index" not in st.session_state:
        st.session_state.bolags_index = 0

    tickers = df["Ticker"].tolist()
    if tickers:
        valt_ticker = st.selectbox(
            "Välj bolag att uppdatera",
            options=tickers,
            index=st.session_state.bolags_index
        )
        befintlig = df[df["Ticker"] == valt_ticker].iloc[0]
    else:
        valt_ticker = ""
        befintlig = pd.Series({}, dtype=object)

    with st.form("bolag_form"):
        ticker = st.text_input("Ticker", value=str(befintlig.get("Ticker", "")) if not befintlig.empty else "").upper()
        antal_aktier = st.number_input("Antal aktier du äger", value=float(befintlig.get("Antal aktier", 0) or 0), step=1.0)
        ps_idag = st.number_input("P/S idag", value=float(befintlig.get("P/S idag", 0) or 0), step=0.01)
        ps1 = st.number_input("P/S Q1", value=float(befintlig.get("P/S Q1", 0) or 0), step=0.01)
        ps2 = st.number_input("P/S Q2", value=float(befintlig.get("P/S Q2", 0) or 0), step=0.01)
        ps3 = st.number_input("P/S Q3", value=float(befintlig.get("P/S Q3", 0) or 0), step=0.01)
        ps4 = st.number_input("P/S Q4", value=float(befintlig.get("P/S Q4", 0) or 0), step=0.01)
        oms_idag = st.number_input("Omsättning idag (miljoner)", value=float(befintlig.get("Omsättning idag", 0) or 0), step=0.01)
        oms_next = st.number_input("Omsättning nästa år (miljoner)", value=float(befintlig.get("Omsättning nästa år", 0) or 0), step=0.01)

        sparaknapp = st.form_submit_button("💾 Spara")

    if sparaknapp:
        bolagsnamn, kurs, valuta = hamta_bolagsinfo_yahoo(ticker)
        cagr = hamta_cagr_5ar(ticker)
        oms2, oms3 = berakna_framtida_omsattning(oms_next, cagr)

        ny_rad = {
            "Ticker": ticker,
            "Bolagsnamn": bolagsnamn,
            "Antal aktier": antal_aktier,
            "P/S idag": ps_idag, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms_next,
            "Omsättning om 2 år": oms2, "Omsättning om 3 år": oms3,
            "CAGR 5 år (%)": cagr,
            "Aktuell kurs": kurs, "Valuta": valuta
        }

        if ticker in df["Ticker"].values:
            df.loc[df["Ticker"] == ticker, ny_rad.keys()] = ny_rad.values()
            st.success(f"{ticker} uppdaterat med data från Yahoo Finance.")
        else:
            df = pd.concat([df, pd.DataFrame([ny_rad])], ignore_index=True)
            st.success(f"{ticker} tillagt med data från Yahoo Finance.")

        spara_data(df)

    # Bläddringsknappar
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Föregående"):
            if st.session_state.bolags_index > 0:
                st.session_state.bolags_index -= 1
                st.experimental_rerun()
    with col2:
        if st.button("➡️ Nästa"):
            if st.session_state.bolags_index < len(tickers) - 1:
                st.session_state.bolags_index += 1
                st.experimental_rerun()

    return df

def uppdatera_berakningar(df):
    for i, rad in df.iterrows():
        ps = [rad.get("P/S Q1", 0), rad.get("P/S Q2", 0), rad.get("P/S Q3", 0), rad.get("P/S Q4", 0)]
        ps = [x for x in ps if x and x > 0]
        ps_snitt = round(np.mean(ps), 2) if ps else ""
        df.at[i, "P/S-snitt"] = ps_snitt

        if ps_snitt and rad.get("Utestående aktier", 0) > 0:
            if rad.get("Omsättning idag"):
                df.at[i, "Riktkurs om 1 år"] = round((rad["Omsättning idag"] * ps_snitt) / rad["Utestående aktier"], 2)
            if rad.get("Omsättning om 2 år"):
                df.at[i, "Riktkurs om 2 år"] = round((rad["Omsättning om 2 år"] * ps_snitt) / rad["Utestående aktier"], 2)
            if rad.get("Omsättning om 3 år"):
                df.at[i, "Riktkurs om 3 år"] = round((rad["Omsättning om 3 år"] * ps_snitt) / rad["Utestående aktier"], 2)
    return df

def analysvy(df):
    st.subheader("📊 Analysvy")
    df = uppdatera_berakningar(df)
    st.dataframe(df, use_container_width=True)

def visa_investeringsforslag(df, valutakurs_usd):
    st.subheader("💡 Investeringsförslag")
    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=500.0, step=100.0)

    riktkurs_val = st.selectbox("Välj riktkurs", ["Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år"], index=0)

    df_forslag = df.copy()
    df_forslag = df_forslag[df_forslag[riktkurs_val] != ""]
    df_forslag["Potential (%)"] = ((df_forslag[riktkurs_val] - df_forslag["Aktuell kurs"]) / df_forslag["Aktuell kurs"]) * 100
    df_forslag = df_forslag.sort_values(by="Potential (%)", ascending=False)

    for _, rad in df_forslag.iterrows():
        antal_kop = int((kapital_sek / valutakurs_usd) // rad["Aktuell kurs"])
        st.markdown(f"""
        **{rad['Bolagsnamn']} ({rad['Ticker']})**  
        Aktuell kurs: {rad['Aktuell kurs']} {rad['Valuta']}  
        {riktkurs_val}: {rad[riktkurs_val]} {rad['Valuta']}  
        Potential: {round(rad['Potential (%)'], 2)}%  
        Köpförslag: {antal_kop} aktier
        """)

def visa_portfolj(df, valutakurs_usd):
    st.subheader("📦 Portfölj")
    df_port = df[df["Antal aktier"] > 0].copy()
    if df_port.empty:
        st.info("Inga aktier i portföljen.")
        return
    df_port["Värde (SEK)"] = df_port["Antal aktier"] * df_port["Aktuell kurs"] * valutakurs_usd
    total = df_port["Värde (SEK)"].sum()
    st.markdown(f"**Totalt portföljvärde:** {round(total, 2)} SEK")
    st.dataframe(df_port, use_container_width=True)

def main():
    st.title("📊 Aktieanalys och investeringsförslag")
    df = hamta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)

    meny = st.sidebar.radio("Meny", ["Analys", "Lägg till / uppdatera bolag", "Investeringsförslag", "Portfölj"])

    if meny == "Analys":
        analysvy(df)
    elif meny == "Lägg till / uppdatera bolag":
        df = lagg_till_eller_uppdatera(df)
    elif meny == "Investeringsförslag":
        valutakurs_usd = st.sidebar.number_input("Valutakurs USD → SEK", value=9.50, step=0.01)
        df = uppdatera_berakningar(df)
        visa_investeringsforslag(df, valutakurs_usd)
    elif meny == "Portfölj":
        valutakurs_usd = st.sidebar.number_input("Valutakurs USD → SEK", value=9.50, step=0.01)
        df = uppdatera_berakningar(df)
        visa_portfolj(df, valutakurs_usd)

if __name__ == "__main__":
    main()
