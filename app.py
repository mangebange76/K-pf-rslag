import streamlit as st
import pandas as pd
import numpy as np
import time
import gspread
from google.oauth2.service_account import Credentials
import yfinance as yf

st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

SHEET_URL = st.secrets["GOOGLE_CREDENTIALS"]["SHEET_URL"]
SHEET_NAME = "Blad1"
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

def skapa_koppling():
    sheet = client.open_by_url(SHEET_URL).worksheet(SHEET_NAME)
    return sheet

def hamta_data():
    sheet = skapa_koppling()
    data = sheet.get_all_records()
    return pd.DataFrame(data)

def spara_data(df):
    sheet = skapa_koppling()
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.astype(str).values.tolist())

def konvertera_typer(df):
    kolumner = [
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Aktuell kurs", "Antal aktier", "Årlig utdelning"
    ]
    for kol in kolumner:
        if kol in df.columns:
            df[kol] = pd.to_numeric(df[kol], errors="coerce").fillna(0.0)
    return df

def säkerställ_kolumner(df):
    nödvändiga = [
        "Ticker", "Yahoo-ticker", "Bolagsnamn", "Valuta",
        "Aktuell kurs", "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "P/S-snitt", "Riktkurs idag", "Riktkurs 2026", "Riktkurs 2027", "Riktkurs 2028",
        "Antal aktier", "Årlig utdelning", "Max andel av portfölj (%)"
    ]
    for kol in nödvändiga:
        if kol not in df.columns:
            if "kurs" in kol.lower() or "omsättning" in kol.lower() or "p/s" in kol.lower() or "andel" in kol.lower():
                df[kol] = 0.0
            else:
                df[kol] = ""
    return df

def uppdatera_berakningar(df):
    for i, rad in df.iterrows():
        ps = [rad["P/S Q1"], rad["P/S Q2"], rad["P/S Q3"], rad["P/S Q4"]]
        ps = [x for x in ps if x > 0]
        ps_snitt = round(np.mean(ps), 2) if ps else 0
        df.at[i, "P/S-snitt"] = ps_snitt

        if rad["Utestående aktier"] > 0:
            df.at[i, "Riktkurs idag"] = round((rad["Omsättning idag"] * ps_snitt) / rad["Utestående aktier"], 2)
            df.at[i, "Riktkurs 2026"] = round((rad["Omsättning nästa år"] * ps_snitt) / rad["Utestående aktier"], 2)
            df.at[i, "Riktkurs 2027"] = round((rad["Omsättning om 2 år"] * ps_snitt) / rad["Utestående aktier"], 2)
            df.at[i, "Riktkurs 2028"] = round((rad["Omsättning om 3 år"] * ps_snitt) / rad["Utestående aktier"], 2)
    return df

def uppdatera_kurser(df):
    if "Yahoo-ticker" not in df.columns:
        df["Yahoo-ticker"] = ""

    st.subheader("🔄 Uppdaterar kurser från Yahoo Finance")
    if st.button("Starta uppdatering"):
        progress = st.empty()
        for i, rad in df.iterrows():
            ticker = rad["Yahoo-ticker"] if rad["Yahoo-ticker"] else rad["Ticker"]
            try:
                aktie = yf.Ticker(ticker)
                ny_kurs = aktie.info.get("regularMarketPrice")
                if ny_kurs:
                    df.at[i, "Aktuell kurs"] = ny_kurs
            except Exception:
                pass

            procent = round((i + 1) / len(df) * 100)
            progress.write(f"🔄 Uppdaterar {ticker} ({procent}%)")
            time.sleep(2)
        st.success("✅ Alla kurser uppdaterade.")
    return df

def lagg_till_eller_uppdatera(df):
    st.subheader("➕ Lägg till / uppdatera bolag")
    tickers = df["Ticker"].tolist()
    namnlista = df["Bolagsnamn"].tolist()
    namn_till_ticker = dict(zip(namnlista, tickers))
    valt_namn = st.selectbox("Välj bolag (eller lämna tom för nytt)", [""] + sorted(namnlista))
    valt = namn_till_ticker.get(valt_namn, "")

    if valt:
        bef = df[df["Ticker"] == valt].iloc[0]
    else:
        bef = {}

    with st.form("form"):
        ticker = st.text_input("Ticker", value=bef.get("Ticker", "")).upper()
        yahoo = st.text_input("Yahoo-ticker", value=bef.get("Yahoo-ticker", ""))
        namn = st.text_input("Bolagsnamn", value=bef.get("Bolagsnamn", ""))
        valuta = st.selectbox("Valuta", ["USD", "SEK", "NOK", "CAD", "EUR"], index=0 if not bef else ["USD", "SEK", "NOK", "CAD", "EUR"].index(bef.get("Valuta", "USD")))

        kurs = st.number_input("Aktuell kurs", value=bef.get("Aktuell kurs", 0.0), step=0.01)
        aktier = st.number_input("Utestående aktier", value=bef.get("Utestående aktier", 0.0), step=0.01)
        antal = st.number_input("Antal aktier du äger", value=bef.get("Antal aktier", 0.0), step=1.0)
        utdelning = st.number_input("Årlig utdelning per aktie", value=bef.get("Årlig utdelning", 0.0), step=0.01)
        maxandel = st.number_input("Max andel av portfölj (%)", value=bef.get("Max andel av portfölj (%)", 0.0), step=0.1)

        ps = st.number_input("P/S", value=bef.get("P/S", 0.0))
        ps1 = st.number_input("P/S Q1", value=bef.get("P/S Q1", 0.0))
        ps2 = st.number_input("P/S Q2", value=bef.get("P/S Q2", 0.0))
        ps3 = st.number_input("P/S Q3", value=bef.get("P/S Q3", 0.0))
        ps4 = st.number_input("P/S Q4", value=bef.get("P/S Q4", 0.0))

        oms0 = st.number_input("Omsättning idag", value=bef.get("Omsättning idag", 0.0))
        oms1 = st.number_input("Omsättning nästa år", value=bef.get("Omsättning nästa år", 0.0))
        oms2 = st.number_input("Omsättning om 2 år", value=bef.get("Omsättning om 2 år", 0.0))
        oms3 = st.number_input("Omsättning om 3 år", value=bef.get("Omsättning om 3 år", 0.0))

        sparaknapp = st.form_submit_button("💾 Spara")

    if sparaknapp and ticker:
        ny = {
            "Ticker": ticker, "Yahoo-ticker": yahoo, "Bolagsnamn": namn, "Valuta": valuta,
            "Aktuell kurs": kurs, "Utestående aktier": aktier, "Antal aktier": antal, "Årlig utdelning": utdelning,
            "Max andel av portfölj (%)": maxandel,
            "P/S": ps, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms0, "Omsättning nästa år": oms1, "Omsättning om 2 år": oms2, "Omsättning om 3 år": oms3
        }

        if ticker in df["Ticker"].values:
            df.loc[df["Ticker"] == ticker, ny.keys()] = ny.values()
            st.success(f"{ticker} uppdaterat.")
        else:
            df = pd.concat([df, pd.DataFrame([ny])], ignore_index=True)
            st.success(f"{ticker} tillagt.")
    return df

def visa_investeringsforslag(df, valutakurser, filter_portfolj=False):
    st.subheader("💡 Investeringsförslag")
    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=500.0, step=100.0)
    riktval = st.selectbox("Filtrera efter riktkurs:", ["Riktkurs 2026", "Riktkurs idag", "Riktkurs 2027", "Riktkurs 2028"])

    df = uppdatera_berakningar(df)

    df = df.copy()
    df["Valutakurs"] = df["Valuta"].map(valutakurser).fillna(0.0)
    df["Värde (SEK)"] = df["Antal aktier"] * df["Aktuell kurs"] * df["Valutakurs"]

    portvarde = df["Värde (SEK)"].sum()

    if filter_portfolj:
        df_forslag = df[df["Antal aktier"] > 0].copy()
    else:
        df_forslag = df.copy()

    df_forslag = df_forslag[df_forslag[riktval] > df_forslag["Aktuell kurs"]]
    df_forslag["Potential (%)"] = ((df_forslag[riktval] - df_forslag["Aktuell kurs"]) / df_forslag["Aktuell kurs"]) * 100
    df_forslag = df_forslag.sort_values(by="Potential (%)", ascending=False).reset_index(drop=True)

    if df_forslag.empty:
        st.info("Inga förslag att visa.")
        return

    if 'forslags_index' not in st.session_state:
        st.session_state.forslags_index = 0

    index = st.session_state.forslags_index
    if index >= len(df_forslag):
        st.info("Inga fler förslag.")
        return

    rad = df_forslag.iloc[index]
    kurs_usd = rad["Aktuell kurs"]
    valutakurs = rad["Valutakurs"]
    if valutakurs == 0:
        st.warning("Valutakurs saknas.")
        return

    kapital = kapital_sek / valutakurs
    antal = int(kapital // kurs_usd)
    investering_sek = antal * kurs_usd * valutakurs
    andel_nu = round((rad["Värde (SEK)"] / portvarde) * 100, 2) if portvarde > 0 else 0
    andel_efter = round(((rad["Värde (SEK)"] + investering_sek) / portvarde) * 100, 2) if portvarde > 0 else 0

    st.markdown(f"""
    ### 💼 Förslag {index+1} av {len(df_forslag)}
    - **Bolag:** {rad['Bolagsnamn']} ({rad['Ticker']})
    - **Aktuell kurs:** {round(kurs_usd, 2)} {rad['Valuta']}
    - **{riktval}:** {round(rad[riktval], 2)} {rad['Valuta']}
    - **Potential:** {round(rad['Potential (%)'], 2)}%
    - **Andel nu:** {andel_nu}%
    - **Andel efter köp:** {andel_efter}%
    - **Antal att köpa:** {antal} st
    - **Beräknad investering:** {round(investering_sek, 2)} SEK
    """)

    if st.button("➡️ Nästa förslag"):
        st.session_state.forslags_index += 1

def visa_portfolj(df, valutakurser):
    st.subheader("📦 Min portfölj")
    df = df[df["Antal aktier"] > 0].copy()
    if df.empty:
        st.info("Du äger inga aktier.")
        return
    df["Valutakurs"] = df["Valuta"].map(valutakurser).fillna(0.0)
    df["Värde (SEK)"] = df["Antal aktier"] * df["Aktuell kurs"] * df["Valutakurs"]
    df["Andel (%)"] = round(df["Värde (SEK)"] / df["Värde (SEK)"].sum() * 100, 2)

    df["Total utdelning (SEK)"] = df["Årlig utdelning"] * df["Antal aktier"] * df["Valutakurs"]
    total_utdelning = df["Total utdelning (SEK)"].sum()
    månadsutdelning = round(total_utdelning / 12, 2)

    st.metric("Totalt portföljvärde", f"{round(df['Värde (SEK)'].sum(), 2)} SEK")
    st.metric("Förväntad årlig utdelning", f"{round(total_utdelning, 2)} SEK")
    st.metric("Snittutdelning per månad", f"{månadsutdelning} SEK")

    st.dataframe(df[["Bolagsnamn", "Ticker", "Antal aktier", "Aktuell kurs", "Värde (SEK)", "Andel (%)", "Årlig utdelning"]], use_container_width=True)

def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    df = hamta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)

    menyval = st.sidebar.selectbox("Välj vy", ["Lägg till / uppdatera bolag", "Uppdatera kurser", "Investeringsförslag", "Investeringsförslag (portföljinnehav)", "Portfölj"])

    st.sidebar.subheader("Valutakurser (till USD)")
    valutakurser = {}
    for valuta in ["USD", "SEK", "NOK", "CAD", "EUR"]:
        valutakurser[valuta] = st.sidebar.number_input(f"{valuta} → USD", value=1.0 if valuta == "USD" else 0.10, step=0.01)

    if menyval == "Lägg till / uppdatera bolag":
        df = lagg_till_eller_uppdatera(df)
    elif menyval == "Uppdatera kurser":
        df = uppdatera_kurser(df)
    elif menyval == "Investeringsförslag":
        visa_investeringsforslag(df, valutakurser, filter_portfolj=False)
    elif menyval == "Investeringsförslag (portföljinnehav)":
        visa_investeringsforslag(df, valutakurser, filter_portfolj=True)
    elif menyval == "Portfölj":
        visa_portfolj(df, valutakurser)

    spara_data(df)

if __name__ == "__main__":
    main()
