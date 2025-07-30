import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import gspread
import time
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Tillväxtaktieverktyg", layout="wide")

# ---- AUTHENTISERING ----
SHEET_URL = st.secrets["SHEET_URL"]

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(
    st.secrets["GOOGLE_CREDENTIALS"], scopes=scope
)
client = gspread.authorize(credentials)

# ---- HÄMTA & SPARA DATA ----
def connect_to_gsheet(url):
    return client.open_by_url(url).sheet1

def get_database(sheet):
    data = sheet.get_all_records()
    df = pd.DataFrame(data)

    for col in ["Aktuell kurs", "Antal aktier", "Årlig utdelning", "Max andel", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
                "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
                "Riktkurs idag", "Riktkurs 2026", "Riktkurs 2027", "Riktkurs 2028"]:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)

    for col in ["Bolag", "Ticker", "Yahoo-ticker", "Valuta"]:
        if col not in df.columns:
            df[col] = ""

    return df

def spara_df(sheet, df):
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.astype(str).values.tolist())

def lagg_till_eller_uppdatera(df):
    st.header("➕ Lägg till / uppdatera bolag")

    val = st.selectbox("Välj bolag att uppdatera", [""] + sorted(df["Bolag"].unique()))
    bef = df[df["Bolag"] == val].iloc[0] if val else {}

    with st.form("form"):
        bolag = st.text_input("Bolagsnamn", value=bef.get("Bolag", ""))
        ticker = st.text_input("Ticker", value=bef.get("Ticker", "")).upper()
        yahoo = st.text_input("Yahoo-ticker", value=bef.get("Yahoo-ticker", ""))
        valuta = st.selectbox("Valuta", ["USD", "NOK", "CAD", "SEK", "EUR"],
                              index=0 if not bef else ["USD", "NOK", "CAD", "SEK", "EUR"].index(bef.get("Valuta", "USD")))
        kurs = st.number_input("Aktuell kurs", value=float(bef.get("Aktuell kurs", 0.0)))
        antal = st.number_input("Antal aktier", value=float(bef.get("Antal aktier", 0.0)))
        utdelning = st.number_input("Årlig utdelning per aktie", value=float(bef.get("Årlig utdelning", 0.0)))
        maxandel = st.number_input("Max andel av portfölj (%)", value=float(bef.get("Max andel", 0.0)))

        ps = st.number_input("P/S", value=float(bef.get("P/S", 0.0)))
        ps1 = st.number_input("P/S Q1", value=float(bef.get("P/S Q1", 0.0)))
        ps2 = st.number_input("P/S Q2", value=float(bef.get("P/S Q2", 0.0)))
        ps3 = st.number_input("P/S Q3", value=float(bef.get("P/S Q3", 0.0)))
        ps4 = st.number_input("P/S Q4", value=float(bef.get("P/S Q4", 0.0)))
        oms_idag = st.number_input("Omsättning idag", value=float(bef.get("Omsättning idag", 0.0)))
        oms1 = st.number_input("Omsättning nästa år", value=float(bef.get("Omsättning nästa år", 0.0)))
        oms2 = st.number_input("Omsättning om 2 år", value=float(bef.get("Omsättning om 2 år", 0.0)))
        oms3 = st.number_input("Omsättning om 3 år", value=float(bef.get("Omsättning om 3 år", 0.0)))
        rikt1 = st.number_input("Riktkurs idag", value=float(bef.get("Riktkurs idag", 0.0)))
        rikt2 = st.number_input("Riktkurs 2026", value=float(bef.get("Riktkurs 2026", 0.0)))
        rikt3 = st.number_input("Riktkurs 2027", value=float(bef.get("Riktkurs 2027", 0.0)))
        rikt4 = st.number_input("Riktkurs 2028", value=float(bef.get("Riktkurs 2028", 0.0)))

        sparaknapp = st.form_submit_button("💾 Spara")

    if sparaknapp:
        ny = {
            "Bolag": bolag, "Ticker": ticker, "Yahoo-ticker": yahoo, "Valuta": valuta,
            "Aktuell kurs": kurs, "Antal aktier": antal, "Årlig utdelning": utdelning, "Max andel": maxandel,
            "P/S": ps, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms1, "Omsättning om 2 år": oms2, "Omsättning om 3 år": oms3,
            "Riktkurs idag": rikt1, "Riktkurs 2026": rikt2, "Riktkurs 2027": rikt3, "Riktkurs 2028": rikt4
        }
        if bolag in df["Bolag"].values:
            df.loc[df["Bolag"] == bolag, ny.keys()] = ny.values()
            st.success("Bolaget uppdaterat.")
        else:
            df = pd.concat([df, pd.DataFrame([ny])], ignore_index=True)
            st.success("Bolaget tillagt.")
    return df

def hamta_valutakurser():
    return {
        "USD": 1.0,
        "NOK": st.sidebar.number_input("Valutakurs NOK→USD", value=0.094),
        "CAD": st.sidebar.number_input("Valutakurs CAD→USD", value=0.73),
        "SEK": st.sidebar.number_input("Valutakurs SEK→USD", value=0.095),
        "EUR": st.sidebar.number_input("Valutakurs EUR→USD", value=1.10),
    }

def uppdatera_kurser(df):
    st.subheader("🔄 Uppdatera aktiekurser")
    valutakurser = hamta_valutakurser()
    tickers = df["Yahoo-ticker"].fillna("").tolist()
    total = len(tickers)

    bar = st.progress(0)
    uppdaterade = 0

    for i, (index, rad) in enumerate(df.iterrows()):
        yahoo = rad.get("Yahoo-ticker", "")
        valuta = rad.get("Valuta", "USD")
        if not yahoo:
            continue
        try:
            data = yf.Ticker(yahoo).history(period="1d")
            kurs = round(data["Close"].iloc[-1], 2)
            df.at[index, "Aktuell kurs"] = kurs * valutakurser.get(valuta, 1)
            uppdaterade += 1
        except Exception:
            pass
        bar.progress((i + 1) / total)

        time.sleep(2)  # paus mellan varje anrop

    st.success(f"{uppdaterade} av {total} bolag uppdaterades.")
    return df

def visa_investeringsforslag(df, valutakurs):
    st.header("💡 Investeringsförslag")
    kapital = st.number_input("Tillgängligt kapital (SEK)", value=500.0)
    riktkurs_val = st.selectbox("Jämför med riktkurs för år", ["2026", "2027", "2028", "idag"])
    portfolj_only = st.checkbox("Visa bara bolag du redan äger", value=False)

    riktkolumn = f"Riktkurs {riktkurs_val}" if riktkurs_val != "idag" else "Riktkurs idag"

    df["Värde (SEK)"] = df["Antal aktier"] * df["Aktuell kurs"] * valutakurs
    portfoljvarde = df["Värde (SEK)"].sum()

    filtrerad = df.copy()
    if portfolj_only:
        filtrerad = filtrerad[filtrerad["Antal aktier"] > 0]

    filtrerad = filtrerad[filtrerad[riktkolumn] > filtrerad["Aktuell kurs"]]
    filtrerad["Potential (%)"] = ((filtrerad[riktkolumn] - filtrerad["Aktuell kurs"]) / filtrerad["Aktuell kurs"]) * 100
    filtrerad = filtrerad.sort_values(by="Potential (%)", ascending=False).reset_index(drop=True)

    if "inv_index" not in st.session_state:
        st.session_state.inv_index = 0

    if st.button("➡️ Nästa förslag"):
        st.session_state.inv_index += 1

    index = st.session_state.inv_index
    if index >= len(filtrerad):
        st.warning("Inga fler förslag.")
        return

    rad = filtrerad.iloc[index]
    yahoo_valuta = rad.get("Valuta", "USD")
    antal = int((kapital / valutakurs) // rad["Aktuell kurs"])
    total_sek = antal * rad["Aktuell kurs"] * valutakurs
    befintligt_varde = rad["Antal aktier"] * rad["Aktuell kurs"] * valutakurs
    andel = (befintligt_varde / portfoljvarde) * 100 if portfoljvarde > 0 else 0
    maxandel = rad.get("Max andel", 100)
    tillåtet = max(0, maxandel - andel)
    max_köp_sek = (tillåtet / 100) * portfoljvarde

    st.markdown(f"""
        ### 🎯 Förslag {index + 1} av {len(filtrerad)}
        - **Bolag:** {rad['Bolag']} ({rad['Ticker']})
        - **Aktuell kurs:** {round(rad['Aktuell kurs'], 2)} USD
        - **Riktkurs ({riktkurs_val}):** {round(rad[riktkolumn], 2)} USD
        - **Potential:** {round(rad['Potential (%)'], 1)}%
        - **Befintligt innehav:** {round(andel, 2)}% av portföljen
        - **Max andel:** {maxandel}%
        - **Rekommenderat köp:** {antal} st aktier
        - **Köp för max:** {round(max_köp_sek, 2)} SEK
    """)

def visa_portfolj(df, valutakurs):
    st.header("📦 Min portfölj")
    df = df[df["Antal aktier"] > 0].copy()
    if df.empty:
        st.info("Du äger inga aktier.")
        return

    df["Värde (SEK)"] = df["Antal aktier"] * df["Aktuell kurs"] * valutakurs
    df["Andel (%)"] = round(df["Värde (SEK)"] / df["Värde (SEK)"].sum() * 100, 2)

    df["Årlig utdelning (SEK)"] = df["Årlig utdelning"] * df["Antal aktier"] * valutakurs
    total_utdelning = df["Årlig utdelning (SEK)"].sum()
    manadsutdelning = total_utdelning / 12

    totalt = round(df["Värde (SEK)"].sum(), 2)
    st.metric("💰 Totalt portföljvärde", f"{totalt:,.0f} SEK")
    st.metric("📈 Förväntad årlig utdelning", f"{total_utdelning:,.0f} SEK")
    st.metric("📆 Snitt månadsutdelning", f"{manadsutdelning:,.0f} SEK")

    st.dataframe(df[[
        "Bolag", "Ticker", "Antal aktier", "Aktuell kurs", "Värde (SEK)", "Andel (%)", "Årlig utdelning (SEK)"
    ]], use_container_width=True)

def main():
    st.set_page_config(page_title="📊 Aktieanalys & Portfölj", layout="wide")
    st.title("📊 Aktieanalys & investeringsförslag")

    df = hamta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)

    meny = st.sidebar.radio("📋 Välj vy", ["Portfölj", "Lägg till / uppdatera bolag", "Investeringsförslag", "Uppdatera kurser", "Data"])

    valutakurs = st.sidebar.number_input("Valutakurs USD → SEK", value=10.0)
    
    if meny == "Portfölj":
        df = uppdatera_berakningar(df)
        visa_portfolj(df, valutakurs)

    elif meny == "Lägg till / uppdatera bolag":
        df = lagg_till_eller_uppdatera(df)
        spara_data(df)

    elif meny == "Investeringsförslag":
        df = uppdatera_berakningar(df)
        visa_investeringsforslag(df, valutakurs)

    elif meny == "Uppdatera kurser":
        df = uppdatera_kurser(df)
        spara_data(df)

    elif meny == "Data":
        st.subheader("📊 Rådata")
        st.dataframe(df)

if __name__ == "__main__":
    main()
