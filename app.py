import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import yfinance as yf
import math

st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Bolag"

scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
]
credentials = Credentials.from_service_account_info(
    st.secrets["GOOGLE_CREDENTIALS"], scopes=scope
)
client = gspread.authorize(credentials)

# ------------------------------------------------
# Hjälpfunktioner för Google Sheets
# ------------------------------------------------
def skapa_koppling():
    return client.open_by_url(SHEET_URL).worksheet(SHEET_NAME)

def hamta_data():
    sheet = skapa_koppling()
    data = sheet.get_all_records()
    df = pd.DataFrame(data)

    # Säkerställ kolumner – ta bort gamla riktkurskolumner
    kolumner = [
        "Ticker", "Bolagsnamn", "P/S idag", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "CAGR 5 år (%)", "Antal aktier", "Aktuell kurs", "Valuta", "Årlig utdelning",
        "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år"
    ]
    for col in kolumner:
        if col not in df.columns:
            df[col] = ""

    # Ta bort gamla 2026-2028 om de finns
    for col in ["Riktkurs 2026", "Riktkurs 2027", "Riktkurs 2028"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    return df[kolumner]

def spara_data(df):
    sheet = skapa_koppling()
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.astype(str).values.tolist())

# ------------------------------------------------
# Datahämtning och beräkningar
# ------------------------------------------------
def hamta_cagr_5ar(ticker):
    """Hämta CAGR (5 år) för omsättning från Yahoo Finance."""
    try:
        hist = yf.Ticker(ticker).financials
        if "Total Revenue" in hist.index:
            oms_hist = hist.loc["Total Revenue"].dropna().values
            if len(oms_hist) >= 5:
                start_val = oms_hist[-1]
                slut_val = oms_hist[0]
                if start_val > 0 and slut_val > 0:
                    cagr = (slut_val / start_val) ** (1 / (len(oms_hist) - 1)) - 1
                    return round(cagr * 100, 2)  # Procent
    except Exception:
        pass
    return None  # Om vi inte kan hämta

def berakna_framtida_omsattning(oms_next, cagr_procent):
    """Beräkna omsättning om 2 och 3 år baserat på CAGR."""
    if oms_next is None or oms_next == "" or pd.isna(oms_next):
        return "", ""

    try:
        oms_next = float(oms_next)
    except ValueError:
        return "", ""

    if cagr_procent is None or pd.isna(cagr_procent):
        return oms_next, oms_next

    # Omvandla från procent till decimal
    cagr_decimal = cagr_procent / 100.0

    # Begränsa till max ±50%
    if cagr_decimal > 0.5:
        cagr_decimal = 0.5
    elif cagr_decimal < -0.5:
        cagr_decimal = -0.5

    oms_ar2 = round(oms_next * (1 + cagr_decimal), 2)
    oms_ar3 = round(oms_ar2 * (1 + cagr_decimal), 2)
    return oms_ar2, oms_ar3

def lagg_till_eller_uppdatera(df):
    st.subheader("➕ Lägg till / uppdatera bolag")

    namn_map = {f"{rad['Bolagsnamn']} ({rad['Ticker']})": idx for idx, rad in df.iterrows()}
    valt = st.selectbox("Välj bolag att uppdatera (eller lämna tom för nytt)",
                        [""] + list(namn_map.keys()))

    if valt:
        idx = namn_map[valt]
        befintlig = df.iloc[idx]
    else:
        befintlig = pd.Series(dtype=object)

    with st.form("form"):
        # Manuella fält
        ticker = st.text_input("Ticker", value=befintlig.get("Ticker", "")).upper()
        antal_aktier = st.number_input("Antal aktier", value=float(befintlig.get("Antal aktier", 0.0)) if not befintlig.empty else 0.0, step=1.0)
        ps_idag = st.number_input("P/S idag", value=float(befintlig.get("P/S idag", 0.0)) if not befintlig.empty else 0.0)
        ps1 = st.number_input("P/S Q1", value=float(befintlig.get("P/S Q1", 0.0)) if not befintlig.empty else 0.0)
        ps2 = st.number_input("P/S Q2", value=float(befintlig.get("P/S Q2", 0.0)) if not befintlig.empty else 0.0)
        ps3 = st.number_input("P/S Q3", value=float(befintlig.get("P/S Q3", 0.0)) if not befintlig.empty else 0.0)
        ps4 = st.number_input("P/S Q4", value=float(befintlig.get("P/S Q4", 0.0)) if not befintlig.empty else 0.0)
        oms_idag = st.number_input("Omsättning idag (miljoner)", value=float(befintlig.get("Omsättning idag", 0.0)) if not befintlig.empty else 0.0)
        oms_next = st.number_input("Omsättning nästa år (miljoner)", value=float(befintlig.get("Omsättning nästa år", 0.0)) if not befintlig.empty else 0.0)

        sparaknapp = st.form_submit_button("💾 Spara")

    if sparaknapp and ticker:
        # Hämta automatiska fält
        try:
            info = yf.Ticker(ticker).info
            namn = info.get("shortName", befintlig.get("Bolagsnamn", ""))
            kurs = info.get("regularMarketPrice", befintlig.get("Aktuell kurs", ""))
            valuta = info.get("currency", befintlig.get("Valuta", ""))
        except Exception:
            namn, kurs, valuta = befintlig.get("Bolagsnamn", ""), befintlig.get("Aktuell kurs", ""), befintlig.get("Valuta", "")

        # Hämta CAGR och beräkna omsättning år 2 och 3
        cagr = hamta_cagr_5ar(ticker)
        oms_ar2, oms_ar3 = berakna_framtida_omsattning(oms_next, cagr)

        ny_rad = {
            "Ticker": ticker,
            "Bolagsnamn": namn,
            "Antal aktier": antal_aktier,
            "P/S idag": ps_idag,
            "P/S Q1": ps1,
            "P/S Q2": ps2,
            "P/S Q3": ps3,
            "P/S Q4": ps4,
            "Omsättning idag": oms_idag,
            "Omsättning nästa år": oms_next,
            "Omsättning om 2 år": oms_ar2,
            "Omsättning om 3 år": oms_ar3,
            "CAGR 5 år (%)": cagr if cagr is not None else "",
            "Aktuell kurs": kurs,
            "Valuta": valuta
        }

        if ticker in df["Ticker"].values:
            df.loc[df["Ticker"] == ticker, ny_rad.keys()] = ny_rad.values()
            st.success(f"{ticker} uppdaterat.")
        else:
            df = pd.concat([df, pd.DataFrame([ny_rad])], ignore_index=True)
            st.success(f"{ticker} tillagt.")

        spara_data(df)

    return df

def uppdatera_berakningar(df):
    for i, rad in df.iterrows():
        # Räkna P/S-snitt bara på värden > 0
        ps_values = [rad.get("P/S Q1", 0), rad.get("P/S Q2", 0), rad.get("P/S Q3", 0), rad.get("P/S Q4", 0)]
        ps_values = [x for x in ps_values if isinstance(x, (int, float)) and x > 0]
        ps_snitt = round(sum(ps_values) / len(ps_values), 2) if ps_values else ""

        df.at[i, "P/S-snitt"] = ps_snitt

        # Riktkurs om 1 år
        if ps_snitt != "" and rad.get("Omsättning idag") not in ("", None) and rad.get("Utestående aktier", 0) > 0:
            df.at[i, "Riktkurs om 1 år"] = round((rad["Omsättning idag"] * ps_snitt) / rad["Utestående aktier"], 2)
        else:
            df.at[i, "Riktkurs om 1 år"] = ""

        # Riktkurs om 2 år
        if ps_snitt != "" and rad.get("Omsättning om 2 år") not in ("", None) and rad.get("Utestående aktier", 0) > 0:
            df.at[i, "Riktkurs om 2 år"] = round((rad["Omsättning om 2 år"] * ps_snitt) / rad["Utestående aktier"], 2)
        else:
            df.at[i, "Riktkurs om 2 år"] = ""

        # Riktkurs om 3 år
        if ps_snitt != "" and rad.get("Omsättning om 3 år") not in ("", None) and rad.get("Utestående aktier", 0) > 0:
            df.at[i, "Riktkurs om 3 år"] = round((rad["Omsättning om 3 år"] * ps_snitt) / rad["Utestående aktier"], 2)
        else:
            df.at[i, "Riktkurs om 3 år"] = ""

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
