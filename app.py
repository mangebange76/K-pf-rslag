import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import gspread
import time
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
        "P/S-snitt", "Riktkurs idag", "Riktkurs 2026", "Riktkurs 2027", "Riktkurs 2028", "Antal aktier", "Valuta", "Årlig utdelning"
    ]
    for kol in nödvändiga:
        if kol not in df.columns:
            df[kol] = "" if kol in ["Ticker", "Yahoo-ticker", "Bolagsnamn", "Valuta"] else 0.0
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

def uppdatera_kurser(df, valutakurser):
    st.subheader("🔄 Uppdatera aktiekurser")
    if st.button("Hämta aktuella kurser för alla bolag"):
        misslyckade = []
        with st.spinner("Hämtar kurser..."):
            for i, rad in df.iterrows():
                ticker = rad["Yahoo-ticker"] if rad["Yahoo-ticker"] else rad["Ticker"]
                try:
                    aktie = yf.Ticker(ticker)
                    info = aktie.history(period="1d")
                    ny_kurs = info["Close"].iloc[-1]
                    df.at[i, "Aktuell kurs"] = round(ny_kurs, 2)
                    time.sleep(2)
                except:
                    misslyckade.append(ticker)
        st.success("Kurser uppdaterade.")
        if misslyckade:
            st.warning(f"Kunde inte hämta kurs för: {', '.join(misslyckade)}")
    return df

def visa_portfolj(df, valutakurser):
    st.subheader("📦 Min portfölj")
    df = df[df["Antal aktier"] > 0].copy()
    if df.empty:
        st.info("Du äger inga aktier.")
        return
    df["Värde (SEK)"] = df.apply(lambda x: x["Antal aktier"] * x["Aktuell kurs"] * valutakurser.get(x["Valuta"], 1), axis=1)
    df["Andel (%)"] = round(df["Värde (SEK)"] / df["Värde (SEK)"].sum() * 100, 2)
    df["Utdelning (SEK)"] = df.apply(lambda x: x["Antal aktier"] * x["Årlig utdelning"] * valutakurser.get(x["Valuta"], 1), axis=1)

    totalvärde = round(df["Värde (SEK)"].sum(), 2)
    totalutdelning = round(df["Utdelning (SEK)"].sum(), 2)
    månadsutdelning = round(totalutdelning / 12, 2)

    st.metric("Total portföljvärde (SEK)", f"{totalvärde:,.2f}")
    st.metric("Årlig utdelning (SEK)", f"{totalutdelning:,.2f}")
    st.metric("Snitt månadsutdelning (SEK)", f"{månadsutdelning:,.2f}")

    st.dataframe(df[["Ticker", "Bolagsnamn", "Antal aktier", "Aktuell kurs", "Valuta", "Värde (SEK)", "Andel (%)", "Årlig utdelning", "Utdelning (SEK)"]], use_container_width=True)

# OBS: Övriga funktioner t.ex. lagg_till_eller_uppdatera() skickas i nästa svar pga utrymme.

def lagg_till_eller_uppdatera(df):
    st.subheader("➕ Lägg till / uppdatera bolag")
    tickers = df["Bolagsnamn"].tolist()
    valt = st.selectbox("Välj bolag att uppdatera (eller lämna tom för nytt)", [""] + tickers)

    if valt:
        bef = df[df["Bolagsnamn"] == valt].iloc[0]
    else:
        bef = pd.Series()

    with st.form("form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            ticker = st.text_input("Ticker", value=bef.get("Ticker", "") if not bef.empty else "").upper()
            yahoo = st.text_input("Yahoo-ticker (om annan)", value=bef.get("Yahoo-ticker", "") if not bef.empty else "")
            namn = st.text_input("Bolagsnamn", value=bef.get("Bolagsnamn", "") if not bef.empty else "")
        with col2:
            valuta = st.selectbox("Valuta", ["USD", "NOK", "CAD", "SEK", "EUR"], index=0 if bef.empty else ["USD", "NOK", "CAD", "SEK", "EUR"].index(bef.get("Valuta", "USD")))
            kurs = st.number_input("Aktuell kurs", value=bef.get("Aktuell kurs", 0.0) if not bef.empty else 0.0)
            aktier = st.number_input("Utestående aktier (miljoner)", value=bef.get("Utestående aktier", 0.0) if not bef.empty else 0.0)
            egna = st.number_input("Antal aktier du äger", value=bef.get("Antal aktier", 0.0) if not bef.empty else 0.0)
        with col3:
            utdelning = st.number_input("Årlig utdelning (i aktiens valuta)", value=bef.get("Årlig utdelning", 0.0) if not bef.empty else 0.0)

        st.markdown("#### P/S och Omsättning")
        p1, p2, p3, p4, p5 = st.columns(5)
        with p1: ps = st.number_input("P/S idag", value=bef.get("P/S", 0.0) if not bef.empty else 0.0)
        with p2: ps1 = st.number_input("P/S Q1", value=bef.get("P/S Q1", 0.0) if not bef.empty else 0.0)
        with p3: ps2 = st.number_input("P/S Q2", value=bef.get("P/S Q2", 0.0) if not bef.empty else 0.0)
        with p4: ps3 = st.number_input("P/S Q3", value=bef.get("P/S Q3", 0.0) if not bef.empty else 0.0)
        with p5: ps4 = st.number_input("P/S Q4", value=bef.get("P/S Q4", 0.0) if not bef.empty else 0.0)

        o1, o2, o3, o4 = st.columns(4)
        with o1: oms1 = st.number_input("Omsättning idag", value=bef.get("Omsättning idag", 0.0) if not bef.empty else 0.0)
        with o2: oms2 = st.number_input("Omsättning nästa år", value=bef.get("Omsättning nästa år", 0.0) if not bef.empty else 0.0)
        with o3: oms3 = st.number_input("Omsättning om 2 år", value=bef.get("Omsättning om 2 år", 0.0) if not bef.empty else 0.0)
        with o4: oms4 = st.number_input("Omsättning om 3 år", value=bef.get("Omsättning om 3 år", 0.0) if not bef.empty else 0.0)

        sparaknapp = st.form_submit_button("💾 Spara")

    if sparaknapp and ticker:
        ny_rad = {
            "Ticker": ticker, "Yahoo-ticker": yahoo, "Bolagsnamn": namn,
            "Valuta": valuta, "Aktuell kurs": kurs, "Utestående aktier": aktier, "Antal aktier": egna,
            "P/S": ps, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms1, "Omsättning nästa år": oms2, "Omsättning om 2 år": oms3, "Omsättning om 3 år": oms4,
            "Årlig utdelning": utdelning
        }

        if namn in df["Bolagsnamn"].values:
            df.loc[df["Bolagsnamn"] == namn, ny_rad.keys()] = ny_rad.values()
            st.success(f"{namn} uppdaterat.")
        else:
            df = pd.concat([df, pd.DataFrame([ny_rad])], ignore_index=True)
            st.success(f"{namn} tillagt.")
    return df

def visa_investeringsforslag(df, valutakurser):
    st.subheader("💡 Investeringsförslag")

    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=500.0, step=100.0)
    riktkurs_val = st.selectbox("Välj riktkurs att basera förslag på", ["Riktkurs idag", "Riktkurs 2026", "Riktkurs 2027", "Riktkurs 2028"])
    filter_typ = st.radio("Visa förslag baserat på:", ["Hela databasen", "Endast min portfölj"])

    df = uppdatera_berakningar(df)

    df["Värde (SEK)"] = df["Antal aktier"] * df["Aktuell kurs"] * df["Valuta"].map(valutakurser).fillna(0)

    portfolj_df = df[df["Antal aktier"] > 0].copy()
    portfoljvarde = portfolj_df["Värde (SEK)"].sum()

    if filter_typ == "Endast min portfölj":
        df_forslag = portfolj_df.copy()
    else:
        df_forslag = df.copy()

    df_forslag = df_forslag[df_forslag[riktkurs_val] > df_forslag["Aktuell kurs"]].copy()
    df_forslag["Potential (%)"] = ((df_forslag[riktkurs_val] - df_forslag["Aktuell kurs"]) / df_forslag["Aktuell kurs"]) * 100
    df_forslag = df_forslag.sort_values(by="Potential (%)", ascending=False).reset_index(drop=True)

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
    valuta = rad["Valuta"]
    valutakurs = valutakurser.get(valuta, 0)
    if valutakurs == 0 or rad["Aktuell kurs"] <= 0:
        st.warning(f"Kan inte visa förslag: valutakurs ({valuta}) saknas eller aktiekurs ogiltig.")
        return

    kapital_usd = kapital_sek / valutakurs
    antal = int(kapital_usd // rad["Aktuell kurs"])
    total_sek = antal * rad["Aktuell kurs"] * valutakurs
    nuvarande_värde = rad["Antal aktier"] * rad["Aktuell kurs"] * valutakurs
    andel_procent = round((nuvarande_värde / portfoljvarde) * 100, 2) if portfoljvarde > 0 else 0

    st.markdown(f"""
        ### 💰 Förslag {index+1} av {len(df_forslag)}
        - **Bolag:** {rad['Bolagsnamn']} ({rad['Ticker']})
        - **Aktuell kurs:** {round(rad['Aktuell kurs'], 2)} {valuta}
        - **{riktkurs_val}:** {round(rad[riktkurs_val], 2)} {valuta}
        - **Potential:** {round(rad["Potential (%)"], 2)}%
        - **Antal att köpa:** {antal} st
        - **Beräknad investering:** {round(total_sek, 2)} SEK
        - **Nuvarande andel av portföljvärde:** {andel_procent}%
    """)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("➡️ Nästa förslag"):
            st.session_state.forslags_index += 1
    with col2:
        if st.button("🔁 Börja om"):
            st.session_state.forslags_index = 0

def visa_portfolj(df, valutakurser):
    st.subheader("📦 Min portfölj")
    df = df[df["Antal aktier"] > 0].copy()
    if df.empty:
        st.info("Du äger inga aktier.")
        return

    df["Valutakurs"] = df["Valuta"].map(valutakurser).fillna(0)
    df["Värde (SEK)"] = df["Antal aktier"] * df["Aktuell kurs"] * df["Valutakurs"]
    df["Årlig utdelning (SEK)"] = df["Årlig utdelning"] * df["Antal aktier"] * df["Valutakurs"]
    df["Andel (%)"] = round(df["Värde (SEK)"] / df["Värde (SEK)"].sum() * 100, 2)

    totalvärde = df["Värde (SEK)"].sum()
    total_utdelning = df["Årlig utdelning (SEK)"].sum()
    månadsutdelning = total_utdelning / 12

    st.info(f"**Totalt portföljvärde:** {round(totalvärde, 2)} SEK")
    st.success(f"**Förväntad årlig utdelning:** {round(total_utdelning, 2)} SEK  \n**Månadsutdelning (snitt):** {round(månadsutdelning, 2)} SEK")

    st.dataframe(df[["Ticker", "Bolagsnamn", "Antal aktier", "Aktuell kurs", "Valuta", "Värde (SEK)", "Årlig utdelning (SEK)", "Andel (%)"]], use_container_width=True)
   import time
import yfinance as yf

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
            st.progress((i+1)/total)

    st.success("✅ Kurser uppdaterade.")
    if misslyckade:
        st.warning(f"Kunde inte uppdatera {len(misslyckade)} tickers: {', '.join(misslyckade)}")
    
    return df 

# -------------------------------------
# MAIN-KÖRNING
# -------------------------------------
def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    df = hamta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)

    valutakurser = {}
    st.sidebar.subheader("💱 Valutakurser (för portfölj)")
    for valuta in ["USD", "NOK", "CAD", "SEK", "EUR"]:
        valutakurser[valuta] = st.sidebar.number_input(f"{valuta} → SEK", value=10.0 if valuta == "USD" else 1.0, step=0.1)

    meny = st.sidebar.radio("📌 Meny", ["Analys", "Lägg till / uppdatera bolag", "Uppdatera kurser", "Investeringsförslag", "Portfölj"])

    if meny == "Analys":
        df = uppdatera_berakningar(df)
        st.dataframe(df, use_container_width=True)

    elif meny == "Lägg till / uppdatera bolag":
        df = lagg_till_eller_uppdatera(df)
        spara_data(df)

    elif meny == "Uppdatera kurser":
        df = uppdatera_kurser(df)
        spara_data(df)

    elif meny == "Investeringsförslag":
        visa_investeringsforslag(df, valutakurser)

    elif meny == "Portfölj":
        visa_portfolj(df, valutakurser)

if __name__ == "__main__":
    main()
