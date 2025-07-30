import streamlit as st
import pandas as pd
import numpy as np
import gspread
import yfinance as yf
import time
from google.oauth2.service_account import Credentials

# ---------------------------------------
# KONFIGURATION OCH GOOGLE SHEETS
# ---------------------------------------

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
        "Bolag", "Ticker", "Yahoo-ticker", "Valuta", "Aktuell kurs", "Utestående aktier",
        "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "Omsättning idag",
        "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "P/S-snitt", "Riktkurs idag", "Riktkurs 2026", "Riktkurs 2027", "Riktkurs 2028",
        "Antal aktier", "Årlig utdelning", "Max andel"
    ]
    for kol in kolumner:
        if kol not in df.columns:
            df[kol] = 0.0 if "kurs" in kol.lower() or "omsättning" in kol.lower() or "p/s" in kol.lower() else ""
    return df

def konvertera_typer(df):
    for kol in df.columns:
        if kol.startswith(("P/S", "Omsättning", "Aktuell kurs", "Riktkurs", "Antal", "Årlig utdelning", "Max andel")):
            df[kol] = pd.to_numeric(df[kol], errors="coerce").fillna(0.0)
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

def lagg_till_eller_uppdatera(df):
    st.subheader("➕ Lägg till / uppdatera bolag")
    tickers = df["Ticker"].tolist()
    bolagsmap = {row["Bolag"]: row["Ticker"] for _, row in df.iterrows()}
    bolagsnamn_lista = list(bolagsmap.keys())
    valt = st.selectbox("Välj existerande bolag att uppdatera", [""] + bolagsnamn_lista)

    if valt:
        befintlig = df[df["Bolag"] == valt].iloc[0]
    else:
        befintlig = pd.Series()

    with st.form("form"):
        bolag = st.text_input("Bolagsnamn", value=befintlig.get("Bolag", ""))
        ticker = st.text_input("Ticker", value=befintlig.get("Ticker", "")).upper()
        yahoo = st.text_input("Yahoo-ticker", value=befintlig.get("Yahoo-ticker", ""))
        valuta = st.selectbox("Valuta", ["USD", "NOK", "CAD", "SEK", "EUR"], index=0 if befintlig.empty else ["USD","NOK","CAD","SEK","EUR"].index(befintlig.get("Valuta", "USD")))

        kurs = st.number_input("Aktuell kurs (USD)", value=0.0 if befintlig.empty else float(befintlig.get("Aktuell kurs", 0.0)))
        aktier = st.number_input("Utestående aktier (miljoner)", value=0.0 if befintlig.empty else float(befintlig.get("Utestående aktier", 0.0)))
        antal = st.number_input("Antal aktier du äger", value=0.0 if befintlig.empty else float(befintlig.get("Antal aktier", 0.0)))
        utdelning = st.number_input("Årlig utdelning per aktie", value=0.0 if befintlig.empty else float(befintlig.get("Årlig utdelning", 0.0)))
        max_andel = st.number_input("Max andel i portfölj (%)", value=0.0 if befintlig.empty else float(befintlig.get("Max andel", 0.0)))

        ps_idag = st.number_input("P/S idag", value=0.0 if befintlig.empty else float(befintlig.get("P/S", 0.0)))
        ps1 = st.number_input("P/S Q1", value=0.0 if befintlig.empty else float(befintlig.get("P/S Q1", 0.0)))
        ps2 = st.number_input("P/S Q2", value=0.0 if befintlig.empty else float(befintlig.get("P/S Q2", 0.0)))
        ps3 = st.number_input("P/S Q3", value=0.0 if befintlig.empty else float(befintlig.get("P/S Q3", 0.0)))
        ps4 = st.number_input("P/S Q4", value=0.0 if befintlig.empty else float(befintlig.get("P/S Q4", 0.0)))

        oms_idag = st.number_input("Omsättning idag (miljoner USD)", value=0.0 if befintlig.empty else float(befintlig.get("Omsättning idag", 0.0)))
        oms1 = st.number_input("Omsättning nästa år", value=0.0 if befintlig.empty else float(befintlig.get("Omsättning nästa år", 0.0)))
        oms2 = st.number_input("Omsättning om 2 år", value=0.0 if befintlig.empty else float(befintlig.get("Omsättning om 2 år", 0.0)))
        oms3 = st.number_input("Omsättning om 3 år", value=0.0 if befintlig.empty else float(befintlig.get("Omsättning om 3 år", 0.0)))

        spara = st.form_submit_button("💾 Spara")

    if spara and ticker:
        ny_rad = {
            "Bolag": bolag, "Ticker": ticker, "Yahoo-ticker": yahoo, "Valuta": valuta,
            "Aktuell kurs": kurs, "Utestående aktier": aktier, "Antal aktier": antal,
            "Årlig utdelning": utdelning, "Max andel": max_andel,
            "P/S": ps_idag, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms1, "Omsättning om 2 år": oms2, "Omsättning om 3 år": oms3
        }

        if ticker in df["Ticker"].values:
            df.loc[df["Ticker"] == ticker, ny_rad.keys()] = ny_rad.values()
            st.success(f"{ticker} uppdaterat.")
        else:
            df = pd.concat([df, pd.DataFrame([ny_rad])], ignore_index=True)
            st.success(f"{ticker} tillagt.")
    return df

def uppdatera_kurser(df, valutakurser):
    st.subheader("📈 Uppdatera kurser från Yahoo Finance")
    if st.button("🔄 Uppdatera alla kurser"):
        lyckade = 0
        totalt = len(df)
        for i, row in df.iterrows():
            yticker = row.get("Yahoo-ticker", "")
            valuta = row.get("Valuta", "USD")
            if yticker:
                try:
                    data = yf.Ticker(yticker).history(period="1d")
                    if not data.empty:
                        pris = data["Close"].iloc[-1]
                        pris_usd = pris / valutakurser.get(valuta, 1.0)
                        df.at[i, "Aktuell kurs"] = round(pris_usd, 2)
                        lyckade += 1
                except:
                    pass
            percent = int((i + 1) / totalt * 100)
            st.progress(percent)
            time.sleep(2)
        spara_data(df)
        st.success(f"{lyckade} av {totalt} kurser uppdaterade.")
    return df

def visa_investeringsforslag(df, valutakurser):
    st.subheader("💡 Investeringsförslag")
    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=500.0, step=100.0)
    årval = st.selectbox("Välj riktkurs-år", ["Riktkurs 2026", "Riktkurs 2027", "Riktkurs 2028"])
    filtrera_portfolj = st.checkbox("Visa bara bolag jag redan äger")

    df = uppdatera_berakningar(df)
    df["Potential (%)"] = ((df[årval] - df["Aktuell kurs"]) / df["Aktuell kurs"]) * 100
    df = df[df["Potential (%)"] > 0]

    if filtrera_portfolj:
        df = df[df["Antal aktier"] > 0]

    df = df.sort_values(by="Potential (%)", ascending=False).reset_index(drop=True)

    if 'forslags_index' not in st.session_state:
        st.session_state.forslags_index = 0

    if df.empty:
        st.info("Inga bolag matchar kriterierna just nu.")
        return

    index = st.session_state.forslags_index
    if index >= len(df):
        st.info("Inga fler förslag att visa.")
        return

    rad = df.iloc[index]
    ticker = rad["Ticker"]
    valuta = rad.get("Valuta", "USD")
    valutakurs = valutakurser.get(valuta, 1.0)
    kurs_usd = rad["Aktuell kurs"]

    if kurs_usd <= 0 or valutakurs == 0:
        st.warning(f"{ticker}: ogiltig kurs eller valutakurs.")
        return

    kapital_usd = kapital_sek / valutakurs
    antal = int(kapital_usd // kurs_usd)
    total_sek = antal * kurs_usd * valutakurs

    portfolj = df[df["Antal aktier"] > 0]
    portfolj["Värde (SEK)"] = portfolj["Antal aktier"] * portfolj["Aktuell kurs"] * portfolj["Valuta"].map(valutakurser)
    total_portfolj = portfolj["Värde (SEK)"].sum()
    innehav_sek = rad["Antal aktier"] * kurs_usd * valutakurs
    andel_nu = innehav_sek / total_portfolj * 100 if total_portfolj > 0 else 0
    max_andel = rad.get("Max andel", 100)

    if andel_nu >= max_andel:
        st.info(f"{ticker} har redan {round(andel_nu,2)} % av portföljen – över maxgränsen ({max_andel} %).")
        st.session_state.forslags_index += 1
        return

    if antal == 0:
        st.info(f"För lite kapital för att köpa {ticker}.")
        st.session_state.forslags_index += 1
        return

    ny_andel = (innehav_sek + total_sek) / total_portfolj * 100 if total_portfolj > 0 else 0

    st.markdown(f"""
        ### 💰 Förslag {index+1} av {len(df)}
        - **Bolag:** {rad['Bolag']} ({ticker})
        - **Aktuell kurs:** {round(kurs_usd, 2)} USD ({valuta})
        - **{årval}:** {round(rad[årval], 2)} USD
        - **Potential:** {round(rad["Potential (%)"], 2)} %
        - **Antal att köpa:** {antal} st
        - **Beräknad investering:** {round(total_sek, 2)} SEK
        - **Nuvarande andel:** {round(andel_nu, 2)} %
        - **Efter köp:** {round(ny_andel, 2)} %
        - **Max tillåten andel:** {max_andel} %
    """)

    if st.button("➡️ Nästa förslag"):
        st.session_state.forslags_index += 1

def visa_portfolj(df, valutakurser):
    st.subheader("📦 Min portfölj")
    df = df[df["Antal aktier"] > 0].copy()
    if df.empty:
        st.info("Du äger inga aktier.")
        return
    df["Värde (SEK)"] = df["Antal aktier"] * df["Aktuell kurs"] * df["Valuta"].map(valutakurser)
    total_varde = df["Värde (SEK)"].sum()
    df["Andel (%)"] = round(df["Värde (SEK)"] / total_varde * 100, 2)

    df["Total utdelning (SEK)"] = df["Årlig utdelning"] * df["Antal aktier"] * df["Valuta"].map(valutakurser)
    total_utdelning = df["Total utdelning (SEK)"].sum()
    månadsutdelning = total_utdelning / 12

    st.metric("📈 Totalt portföljvärde", f"{round(total_varde):,} SEK")
    st.metric("💸 Förväntad utdelning / år", f"{round(total_utdelning):,} SEK")
    st.metric("📆 Månadsutdelning (snitt)", f"{round(månadsutdelning):,} SEK")

    st.dataframe(df[["Bolag", "Ticker", "Antal aktier", "Aktuell kurs", "Värde (SEK)", "Andel (%)"]], use_container_width=True)

def main():
    st.title("📊 Aktieanalys och investeringsförslag")
    valutakurser = {
        "USD": 10.0,
        "NOK": st.sidebar.number_input("Valutakurs NOK → SEK", value=1.0, step=0.01),
        "CAD": st.sidebar.number_input("Valutakurs CAD → SEK", value=8.0, step=0.1),
        "SEK": 1.0,
        "EUR": st.sidebar.number_input("Valutakurs EUR → SEK", value=11.0, step=0.1)
    }

    meny = st.sidebar.radio("Meny", ["Analys", "Lägg till / uppdatera bolag", "Investeringsförslag", "Portfölj", "Uppdatera kurser"])
    df = hamta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)

    if meny == "Analys":
        df = uppdatera_berakningar(df)
        st.dataframe(df, use_container_width=True)

    elif meny == "Lägg till / uppdatera bolag":
        df = lagg_till_eller_uppdatera(df)
        spara_data(df)

    elif meny == "Investeringsförslag":
        visa_investeringsforslag(df, valutakurser)

    elif meny == "Portfölj":
        visa_portfolj(df, valutakurser)

    elif meny == "Uppdatera kurser":
        df = uppdatera_kurser(df, valutakurser)

if __name__ == "__main__":
    main()
