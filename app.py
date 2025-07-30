import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time
import gspread
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

# ---------------------------------------
# KONFIGURATION OCH GOOGLE SHEETS
# ---------------------------------------

SHEET_URL = st.secrets["SHEET_URL"]
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
        "Ticker", "Yahoo-ticker", "Bolagsnamn", "Aktuell kurs", "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "P/S-snitt", "Riktkurs idag", "Riktkurs 2026", "Riktkurs 2027", "Riktkurs 2028",
        "Antal aktier", "Valuta", "Årlig utdelning"
    ]
    for kol in nödvändiga:
        if kol not in df.columns:
            if "kurs" in kol.lower() or "omsättning" in kol.lower() or "p/s" in kol.lower() or "utdelning" in kol.lower():
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
    st.subheader("🔄 Uppdatera aktiekurser automatiskt")
    
    yahoo_tickers = df["Yahoo-ticker"].fillna("")
    tickers_att_använda = [
        yt if yt.strip() else df.loc[i, "Ticker"]
        for i, yt in enumerate(yahoo_tickers)
    ]

    misslyckade = []
    total = len(tickers_att_använda)

    with st.spinner("Hämtar kurser..."):
        for i, (idx, ticker) in enumerate(zip(df.index, tickers_att_använda)):
            try:
                aktie = yf.Ticker(ticker)
                pris = aktie.info.get("currentPrice", None)
                if pris is None or not isinstance(pris, (int, float)):
                    misslyckade.append(ticker)
                    continue
                df.at[idx, "Aktuell kurs"] = pris
            except Exception:
                misslyckade.append(ticker)
                continue
            time.sleep(2)
            st.progress((i + 1) / total)

    st.success("✅ Kurser uppdaterade.")
    if misslyckade:
        st.warning(f"Kunde inte uppdatera {len(misslyckade)} tickers: {', '.join(misslyckade)}")

    return df

def lagg_till_eller_uppdatera(df):
    st.subheader("➕ Lägg till / uppdatera bolag")
    tickers = df["Bolagsnamn"].tolist()
    valt = st.selectbox("Välj bolag att uppdatera (eller lämna tom för nytt)", [""] + sorted(tickers))

    if valt:
        bef = df[df["Bolagsnamn"] == valt].iloc[0]
    else:
        bef = pd.Series(dtype=object)

    with st.form("form"):
        bolagsnamn = st.text_input("Bolagsnamn", value=bef.get("Bolagsnamn", ""))
        ticker = st.text_input("Ticker", value=bef.get("Ticker", "")).upper()
        yahoo_ticker = st.text_input("Yahoo-ticker (valfritt)", value=bef.get("Yahoo-ticker", ""))
        valuta = st.selectbox("Valuta", ["USD", "NOK", "CAD", "SEK", "EUR"],
                              index=0 if bef.empty else ["USD", "NOK", "CAD", "SEK", "EUR"].index(bef.get("Valuta", "USD")))

        kurs = st.number_input("Aktuell kurs", value=0.0 if bef.empty else float(bef.get("Aktuell kurs", 0.0)))
        utdelning = st.number_input("Årlig utdelning per aktie", value=0.0 if bef.empty else float(bef.get("Årlig utdelning", 0.0)))
        antal_aktier = st.number_input("Antal aktier du äger", value=0.0 if bef.empty else float(bef.get("Antal aktier", 0.0)))
        aktier = st.number_input("Utestående aktier (miljoner)", value=0.0 if bef.empty else float(bef.get("Utestående aktier", 0.0)))

        ps_idag = st.number_input("P/S idag", value=0.0 if bef.empty else float(bef.get("P/S", 0.0)))
        ps1 = st.number_input("P/S Q1", value=0.0 if bef.empty else float(bef.get("P/S Q1", 0.0)))
        ps2 = st.number_input("P/S Q2", value=0.0 if bef.empty else float(bef.get("P/S Q2", 0.0)))
        ps3 = st.number_input("P/S Q3", value=0.0 if bef.empty else float(bef.get("P/S Q3", 0.0)))
        ps4 = st.number_input("P/S Q4", value=0.0 if bef.empty else float(bef.get("P/S Q4", 0.0)))

        oms_idag = st.number_input("Omsättning idag (miljoner)", value=0.0 if bef.empty else float(bef.get("Omsättning idag", 0.0)))
        oms_1 = st.number_input("Omsättning nästa år", value=0.0 if bef.empty else float(bef.get("Omsättning nästa år", 0.0)))
        oms_2 = st.number_input("Omsättning om 2 år", value=0.0 if bef.empty else float(bef.get("Omsättning om 2 år", 0.0)))
        oms_3 = st.number_input("Omsättning om 3 år", value=0.0 if bef.empty else float(bef.get("Omsättning om 3 år", 0.0)))

        spara = st.form_submit_button("💾 Spara")

    if spara and ticker:
        ny_rad = {
            "Bolagsnamn": bolagsnamn, "Ticker": ticker, "Yahoo-ticker": yahoo_ticker, "Valuta": valuta,
            "Aktuell kurs": kurs, "Årlig utdelning": utdelning, "Antal aktier": antal_aktier,
            "Utestående aktier": aktier,
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
    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=500.0, step=100.0)

    riktkursval = st.selectbox("Välj riktkurs att jämföra mot", ["Riktkurs idag", "Riktkurs 2026", "Riktkurs 2027", "Riktkurs 2028"])
    endast_portfolj = st.checkbox("Visa endast innehav i portföljen")

    df = df.copy()
    df["Valutakurs"] = df["Valuta"].map(valutakurser).fillna(0.0)
    df["Värde (SEK)"] = df["Antal aktier"] * df["Aktuell kurs"] * df["Valutakurs"]

    df_port = df[df["Antal aktier"] > 0]
    portfoljvarde = df_port["Värde (SEK)"].sum()

    df["Potential (%)"] = ((df[riktkursval] - df["Aktuell kurs"]) / df["Aktuell kurs"]) * 100
    df = df[df["Potential (%)"] > 0].copy()

    if endast_portfolj:
        df = df[df["Antal aktier"] > 0].copy()

    df = df[df["Valutakurs"] > 0].sort_values(by="Potential (%)", ascending=False).reset_index(drop=True)

    if 'forslags_index' not in st.session_state:
        st.session_state.forslags_index = 0

    index = st.session_state.forslags_index
    if df.empty:
        st.info("Inga bolag matchar kriterierna.")
        return
    if index >= len(df):
        st.info("Inga fler förslag att visa.")
        return

    rad = df.iloc[index]
    if rad["Aktuell kurs"] <= 0:
        st.warning("Felaktig kurs.")
        return

    kapital_usd = kapital_sek / rad["Valutakurs"]
    antal = int(kapital_usd // rad["Aktuell kurs"])
    total_sek = antal * rad["Aktuell kurs"] * rad["Valutakurs"]
    andel_efter = round((rad["Värde (SEK)"] + total_sek) / (portfoljvarde + total_sek) * 100, 2)
    andel_nu = round(rad["Värde (SEK)"] / portfoljvarde * 100, 2) if portfoljvarde > 0 else 0.0

    st.markdown(f"""
        ### 💰 Förslag {index+1} av {len(df)}
        - **Bolag:** {rad['Bolagsnamn']} ({rad['Ticker']})
        - **Aktuell kurs:** {round(rad['Aktuell kurs'], 2)} {rad['Valuta']}
        - **{riktkursval}:** {round(rad[riktkursval], 2)} {rad['Valuta']}
        - **Potential:** {round(rad["Potential (%)"], 2)}%
        - **Antal att köpa:** {antal} st
        - **Beräknad investering:** {round(total_sek, 2)} SEK
        - **Andel nuvarande portfölj:** {andel_nu}%
        - **Andel efter köp:** {andel_efter}%
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
    df["Årlig utdelning (SEK)"] = df["Årlig utdelning"] * df["Antal aktier"] * df["Valutakurs"]

    total_utdelning = df["Årlig utdelning (SEK)"].sum()
    snitt_per_manad = total_utdelning / 12

    st.metric("Total portföljvärde", f"{int(df['Värde (SEK)'].sum()):,} SEK")
    st.metric("Förväntad årlig utdelning", f"{round(total_utdelning, 2):,} SEK")
    st.metric("Snittutdelning per månad", f"{round(snitt_per_manad, 2):,} SEK")

    st.dataframe(df[[
        "Bolagsnamn", "Ticker", "Antal aktier", "Aktuell kurs", "Valuta",
        "Värde (SEK)", "Andel (%)", "Årlig utdelning", "Årlig utdelning (SEK)"
    ]], use_container_width=True)

def main():
    st.set_page_config(page_title="📊 Aktieanalys och investeringsförslag", layout="wide")
    st.title("📊 Aktieanalys och investeringsförslag")

    df = hamta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)

    # Manuell valutainmatning
    st.sidebar.header("📉 Valutakurser")
    valutakurser = {}
    for valuta in ["USD", "NOK", "CAD", "SEK", "EUR"]:
        valutakurser[valuta] = st.sidebar.number_input(f"{valuta} → SEK", value=10.0 if valuta == "USD" else 1.0, step=0.01)

    meny = st.sidebar.radio("Meny", ["Analys", "Lägg till / uppdatera bolag", "Investeringsförslag", "Portfölj", "Uppdatera kurser"])

    if meny == "Analys":
        df = uppdatera_berakningar(df)
        st.dataframe(df, use_container_width=True)

    elif meny == "Lägg till / uppdatera bolag":
        df = lagg_till_eller_uppdatera(df)
        spara_data(df)

    elif meny == "Investeringsförslag":
        df = uppdatera_berakningar(df)
        visa_investeringsforslag(df, valutakurser)

    elif meny == "Portfölj":
        df = uppdatera_berakningar(df)
        visa_portfolj(df, valutakurser)

    elif meny == "Uppdatera kurser":
        df = uppdatera_kurser(df)
        spara_data(df)

if __name__ == "__main__":
    main()
