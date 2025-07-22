import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
import yfinance as yf

st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

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
        "Aktuell kurs", "Antal aktier"
    ]
    for kol in kolumner:
        if kol in df.columns:
            df[kol] = pd.to_numeric(df[kol], errors="coerce").fillna(0.0)
    return df

def säkerställ_kolumner(df):
    nödvändiga = [
        "Ticker", "Bolagsnamn", "Aktuell kurs", "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "P/S-snitt", "Riktkurs idag", "Riktkurs 2026", "Riktkurs 2027", "Riktkurs 2028", "Antal aktier"
    ]
    for kol in nödvändiga:
        if kol not in df.columns:
            df[kol] = 0.0 if "kurs" in kol.lower() or "omsättning" in kol.lower() or "p/s" in kol.lower() else ""
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

def hamta_kurs_och_valuta(ticker):
    try:
        data = yf.Ticker(ticker)
        info = data.info
        pris = info.get("currentPrice") or info.get("regularMarketPrice")
        valuta = info.get("currency", "USD")
        if pris is None:
            return None, None
        return pris, valuta
    except Exception:
        return None, None

def hamta_valutakurs(fran_valuta):
    try:
        if fran_valuta == "USD":
            return 1.0
        valutapar = f"{fran_valuta}USD=X"
        kursinfo = yf.Ticker(valutapar).info
        return kursinfo.get("regularMarketPrice", 1.0)
    except Exception:
        return 1.0

def lagg_till_eller_uppdatera(df):
    st.subheader("➕ Lägg till / uppdatera bolag")

    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["Bolagsnamn"] = df["Bolagsnamn"].astype(str).str.strip()
    alternativ = sorted([f"{namn} ({ticker})" for namn, ticker in zip(df["Bolagsnamn"], df["Ticker"])])

    valt = st.selectbox("Välj existerande bolag att uppdatera (eller lämna tom för nytt)", [""] + alternativ)

    if valt:
        ticker_vald = valt.split("(")[-1].replace(")", "").strip()
        befintlig = df[df["Ticker"] == ticker_vald].iloc[0]
    else:
        befintlig = {
            "Ticker": "",
            "Bolagsnamn": "",
            "Aktuell kurs": 0.0,
            "Utestående aktier": 0.0,
            "Antal aktier": 0.0,
            "P/S": 0.0,
            "P/S Q1": 0.0,
            "P/S Q2": 0.0,
            "P/S Q3": 0.0,
            "P/S Q4": 0.0,
            "Omsättning idag": 0.0,
            "Omsättning nästa år": 0.0,
            "Omsättning om 2 år": 0.0,
            "Omsättning om 3 år": 0.0
        }

    with st.form("form"):
        ticker = st.text_input("Ticker (Yahoo Finance-format)", value=befintlig.get("Ticker", "")).upper()
        st.caption("ℹ️ Ange Yahoo Finance-ticker, t.ex. `EQNR.OL` för Oslo, `SHOP.TO` för Toronto, `BMW.DE` för Frankfurt.")

        if st.form_submit_button("🔄 Hämta aktuell kurs från Yahoo"):
            pris, valuta = hamta_kurs_och_valuta(ticker)
            if pris is None:
                st.warning("Kunde inte hämta kurs – kontrollera ticker.")
            else:
                växelkurs = hamta_valutakurs(valuta)
                kurs_usd = pris * växelkurs
                st.session_state["hamtad_kurs"] = round(kurs_usd, 2)
                st.success(f"Hämtad kurs: {pris} {valuta} → {kurs_usd:.2f} USD")

        kurs = st.number_input("Aktuell kurs (USD)", value=st.session_state.get("hamtad_kurs", float(befintlig.get("Aktuell kurs", 0.0))))
        namn = st.text_input("Bolagsnamn", value=befintlig.get("Bolagsnamn", ""))
        aktier = st.number_input("Utestående aktier (miljoner)", value=float(befintlig.get("Utestående aktier", 0.0)))
        antal_aktier = st.number_input("Antal aktier du äger", value=float(befintlig.get("Antal aktier", 0.0)))

        ps_idag = st.number_input("P/S idag", value=float(befintlig.get("P/S", 0.0)))
        ps1 = st.number_input("P/S Q1", value=float(befintlig.get("P/S Q1", 0.0)))
        ps2 = st.number_input("P/S Q2", value=float(befintlig.get("P/S Q2", 0.0)))
        ps3 = st.number_input("P/S Q3", value=float(befintlig.get("P/S Q3", 0.0)))
        ps4 = st.number_input("P/S Q4", value=float(befintlig.get("P/S Q4", 0.0)))

        oms_idag = st.number_input("Omsättning idag (miljoner USD)", value=float(befintlig.get("Omsättning idag", 0.0)))
        oms_1 = st.number_input("Omsättning nästa år", value=float(befintlig.get("Omsättning nästa år", 0.0)))
        oms_2 = st.number_input("Omsättning om 2 år", value=float(befintlig.get("Omsättning om 2 år", 0.0)))
        oms_3 = st.number_input("Omsättning om 3 år", value=float(befintlig.get("Omsättning om 3 år", 0.0)))

        sparaknapp = st.form_submit_button("💾 Spara")

    if sparaknapp and ticker:
        ny_rad = {
            "Ticker": ticker, "Bolagsnamn": namn, "Aktuell kurs": kurs, "Utestående aktier": aktier, "Antal aktier": antal_aktier,
            "P/S": ps_idag, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms_1, "Omsättning om 2 år": oms_2, "Omsättning om 3 år": oms_3
        }

        df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
        ticker = ticker.strip().upper()

        if ticker in df["Ticker"].values:
            df.loc[df["Ticker"] == ticker, ny_rad.keys()] = ny_rad.values()
            st.success(f"{ticker} uppdaterat.")
        else:
            df = pd.concat([df, pd.DataFrame([ny_rad])], ignore_index=True)
            st.success(f"{ticker} tillagt.")
    return df

def visa_investeringsforslag(df, valutakurs):
    st.subheader("💡 Investeringsförslag")
    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=10000.0, step=500.0)

    riktkurs_val = st.selectbox("Välj riktkurs att basera förslagen på:", ["Riktkurs 2026", "Riktkurs 2027", "Riktkurs 2028"])

    df_portfolj = df[df["Antal aktier"] > 0].copy()
    df_portfolj["Värde (SEK)"] = df_portfolj["Antal aktier"] * df_portfolj["Aktuell kurs"] * valutakurs
    portfoljvarde = df_portfolj["Värde (SEK)"].sum()

    df_forslag = df[df[riktkurs_val] > df["Aktuell kurs"]].copy()
    df_forslag["Potential (%)"] = ((df_forslag[riktkurs_val] - df_forslag["Aktuell kurs"]) / df_forslag["Aktuell kurs"]) * 100
    df_forslag = df_forslag.sort_values(by="Potential (%)", ascending=False).reset_index(drop=True)

    if valutakurs == 0:
        st.warning("Valutakursen får inte vara 0.")
        return

    kapital_usd = kapital_sek / valutakurs

    if 'forslags_index' not in st.session_state:
        st.session_state.forslags_index = 0

    if df_forslag.empty:
        st.info("Inga bolag matchar kriterierna just nu.")
        return

    index = st.session_state.forslags_index
    if index >= len(df_forslag):
        st.info("Inga fler förslag att visa.")
        return

    rad = df_forslag.iloc[index]
    if rad["Aktuell kurs"] <= 0:
        st.warning("Felaktig aktiekurs – kan inte visa förslag.")
        return

    antal = int(kapital_usd // rad["Aktuell kurs"])
    total_sek = antal * rad["Aktuell kurs"] * valutakurs
    andel_procent = round((total_sek / portfoljvarde) * 100, 2) if portfoljvarde > 0 else 0

    st.markdown(f"""
        ### 💰 Förslag {index+1} av {len(df_forslag)}
        - **Bolag:** {rad['Bolagsnamn']} ({rad['Ticker']})
        - **Aktuell kurs:** {round(rad['Aktuell kurs'], 2)} USD
        - **{riktkurs_val}:** {round(rad[riktkurs_val], 2)} USD
        - **Potential:** {round(rad['Potential (%)'], 2)}%
        - **Antal att köpa:** {antal} st
        - **Beräknad investering:** {round(total_sek, 2)} SEK
        - **Andel av nuvarande portföljvärde:** {andel_procent}%
    """)

    if st.button("➡️ Nästa förslag"):
        st.session_state.forslags_index += 1


def visa_portfolj(df, valutakurs):
    st.subheader("📦 Min portfölj")
    df = df[df["Antal aktier"] > 0].copy()
    if df.empty:
        st.info("Du äger inga aktier.")
        return
    df["Värde (SEK)"] = df["Antal aktier"] * df["Aktuell kurs"] * valutakurs
    totalt = df["Värde (SEK)"].sum()
    df["Andel (%)"] = round(df["Värde (SEK)"] / totalt * 100, 2)

    st.markdown(f"### 💰 Totalt portföljvärde: {totalt:,.2f} SEK")
    st.dataframe(df[["Ticker", "Bolagsnamn", "Antal aktier", "Aktuell kurs", "Värde (SEK)", "Andel (%)"]], use_container_width=True)

def analysvy(df):
    st.subheader("📈 Analysläge")
    df = uppdatera_berakningar(df)

    if st.button("🔄 Uppdatera alla aktuella kurser från Yahoo"):
        misslyckade = []
        uppdaterade = 0

        for i, rad in df.iterrows():
            ticker = str(rad["Ticker"]).strip().upper()
            try:
                pris, valuta = hamta_kurs_och_valuta(ticker)
                if pris is None:
                    misslyckade.append(ticker)
                    continue
                växelkurs = hamta_valutakurs(valuta)
                kurs_usd = pris * växelkurs
                df.at[i, "Aktuell kurs"] = round(kurs_usd, 2)
                uppdaterade += 1
            except Exception:
                misslyckade.append(ticker)

        spara_data(df)
        st.success(f"{uppdaterade} tickers uppdaterade.")
        if misslyckade:
            st.warning("Kunde inte uppdatera följande tickers:\n" + ", ".join(misslyckade))

    st.dataframe(df, use_container_width=True)


def main():
    st.title("📊 Aktieanalys och investeringsförslag")
    df = hamta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)

    valutakurs = st.sidebar.number_input("Valutakurs USD → SEK", value=10.0, step=0.1)
    meny = st.sidebar.radio("Meny", ["Analys", "Lägg till / uppdatera bolag", "Investeringsförslag", "Portfölj"])

    if meny == "Analys":
        analysvy(df)
    elif meny == "Lägg till / uppdatera bolag":
        df = lagg_till_eller_uppdatera(df)
        spara_data(df)
    elif meny == "Investeringsförslag":
        df = uppdatera_berakningar(df)
        visa_investeringsforslag(df, valutakurs)
    elif meny == "Portfölj":
        df = uppdatera_berakningar(df)
        visa_portfolj(df, valutakurs)

if __name__ == "__main__":
    main()
