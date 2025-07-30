import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import gspread
import time
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Tillväxtaktieverktyg", layout="wide")

SHEET_URL = st.secrets["GOOGLE_CREDENTIALS"]["SHEET_URL"]

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(
    st.secrets["GOOGLE_CREDENTIALS"], scopes=scope
)
client = gspread.authorize(credentials)

@st.cache_data(ttl=60)
def load_data():
    sheet = client.open_by_url(SHEET_URL).sheet1
    data = sheet.get_all_records()
    return pd.DataFrame(data)

def save_data(df):
    sheet = client.open_by_url(SHEET_URL).sheet1
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.values.tolist())

def lagg_till_eller_uppdatera(df):
    st.header("➕ Lägg till / uppdatera bolag")
    bolagsnamn_lista = df["Bolag"].dropna().unique().tolist()
    valt_bolag = st.selectbox("Välj bolag att uppdatera (eller lämna tomt för nytt)", [""] + bolagsnamn_lista)

    befintlig = df[df["Bolag"] == valt_bolag].iloc[0] if valt_bolag else {}

    with st.form("bolagsformulär"):
        kol1, kol2 = st.columns(2)

        with kol1:
            bolag = st.text_input("Bolagsnamn", value=befintlig.get("Bolag", ""))
            ticker = st.text_input("Ticker", value=befintlig.get("Ticker", ""))
            yahoo = st.text_input("Yahoo-ticker", value=befintlig.get("Yahoo-ticker", ""))
            valuta = st.selectbox("Valuta", ["USD", "NOK", "CAD", "SEK", "EUR"],
                                  index=["USD", "NOK", "CAD", "SEK", "EUR"].index(befintlig.get("Valuta", "USD")) if befintlig else 0)
            max_andel = st.number_input("Max andel av portfölj (%)", min_value=0.0, value=befintlig.get("Max andel", 0.0), step=0.1)
            antal = st.number_input("Antal aktier", min_value=0, value=int(befintlig.get("Antal aktier", 0)), step=1)
            aktuell_kurs = st.number_input("Aktuell kurs", min_value=0.0, value=befintlig.get("Aktuell kurs", 0.0), step=0.01)
            utdelning = st.number_input("Årlig utdelning", min_value=0.0, value=befintlig.get("Årlig utdelning", 0.0), step=0.01)

        with kol2:
            rikt_idag = st.number_input("Riktkurs idag", value=befintlig.get("Riktkurs idag", 0.0), step=0.1)
            rikt_2026 = st.number_input("Riktkurs 2026", value=befintlig.get("Riktkurs 2026", 0.0), step=0.1)
            rikt_2027 = st.number_input("Riktkurs 2027", value=befintlig.get("Riktkurs 2027", 0.0), step=0.1)
            rikt_2028 = st.number_input("Riktkurs 2028", value=befintlig.get("Riktkurs 2028", 0.0), step=0.1)
            ps1 = st.number_input("P/S 1", value=befintlig.get("P/S 1", 0.0), step=0.1)
            ps2 = st.number_input("P/S 2", value=befintlig.get("P/S 2", 0.0), step=0.1)
            ps3 = st.number_input("P/S 3", value=befintlig.get("P/S 3", 0.0), step=0.1)
            oms1 = st.number_input("Omsättning 1", value=befintlig.get("Omsättning 1", 0.0), step=0.01)
            oms2 = st.number_input("Omsättning 2", value=befintlig.get("Omsättning 2", 0.0), step=0.01)
            oms3 = st.number_input("Omsättning 3", value=befintlig.get("Omsättning 3", 0.0), step=0.01)

        spara = st.form_submit_button("💾 Spara")

    if spara:
        ny_rad = {
            "Bolag": bolag,
            "Ticker": ticker,
            "Yahoo-ticker": yahoo,
            "Valuta": valuta,
            "Max andel": max_andel,
            "Antal aktier": antal,
            "Aktuell kurs": aktuell_kurs,
            "Årlig utdelning": utdelning,
            "Riktkurs idag": rikt_idag,
            "Riktkurs 2026": rikt_2026,
            "Riktkurs 2027": rikt_2027,
            "Riktkurs 2028": rikt_2028,
            "P/S 1": ps1,
            "P/S 2": ps2,
            "P/S 3": ps3,
            "Omsättning 1": oms1,
            "Omsättning 2": oms2,
            "Omsättning 3": oms3
        }

        df = df[df["Bolag"] != bolag]
        df = pd.concat([df, pd.DataFrame([ny_rad])], ignore_index=True)
        save_data(df)
        st.success(f"{bolag} sparad / uppdaterad!")

    return df

def portfoljvy(df):
    st.header("📊 Portfölj")

    usd_sek = st.number_input("Valutakurs USD → SEK", value=10.0, step=0.01)
    nok_usd = st.number_input("Valutakurs NOK → USD", value=0.095, step=0.001)
    cad_usd = st.number_input("Valutakurs CAD → USD", value=0.73, step=0.001)
    sek_usd = st.number_input("Valutakurs SEK → USD", value=0.094, step=0.001)
    eur_usd = st.number_input("Valutakurs EUR → USD", value=1.1, step=0.001)

    valuta_map = {
        "USD": 1,
        "NOK": nok_usd,
        "CAD": cad_usd,
        "SEK": sek_usd,
        "EUR": eur_usd
    }

    df["Valuta"] = df["Valuta"].fillna("USD")
    df["Växelkurs"] = df["Valuta"].map(valuta_map)
    df["USD-kurs"] = df["Aktuell kurs"] * df["Växelkurs"]
    df["Värde (USD)"] = df["Antal aktier"] * df["USD-kurs"]

    total_portfolio_value = df["Värde (USD)"].sum()
    df["Andel av portfölj (%)"] = df["Värde (USD)"] / total_portfolio_value * 100

    df["Total utdelning (USD)"] = df["Årlig utdelning"] * df["Antal aktier"]
    df["Total utdelning (SEK)"] = df["Total utdelning (USD)"] * usd_sek
    total_utdelning = df["Total utdelning (SEK)"].sum()
    utdelning_per_manad = total_utdelning / 12

    st.metric("📦 Totalt portföljvärde (USD)", f"{total_portfolio_value:,.0f} USD")
    st.metric("📈 Förväntad årlig utdelning (SEK)", f"{total_utdelning:,.0f} SEK")
    st.metric("📅 Månadsutdelning (snitt, SEK)", f"{utdelning_per_manad:,.0f} SEK")

    st.dataframe(df[[
        "Bolag", "Antal aktier", "Aktuell kurs", "Valuta", "Värde (USD)",
        "Andel av portfölj (%)", "Årlig utdelning", "Total utdelning (SEK)"
    ]].sort_values(by="Andel av portfölj (%)", ascending=False), use_container_width=True)

def investeringsforslag(df):
    st.header("💡 Investeringsförslag")

    kapital = st.number_input("Tillgängligt kapital (SEK)", value=500, step=100)

    riktkursval = st.selectbox("Använd riktkurs från:", ["Riktkurs idag", "Riktkurs 2026", "Riktkurs 2027", "Riktkurs 2028"])
    visa_endast_portfolj = st.checkbox("Visa endast innehav i portföljen", value=False)

    usd_sek = st.number_input("Valutakurs USD → SEK (för konvertering)", value=10.0, step=0.01)

    valuta_map = {
        "USD": 1,
        "NOK": st.session_state.get("nok_usd", 0.095),
        "CAD": st.session_state.get("cad_usd", 0.73),
        "SEK": st.session_state.get("sek_usd", 0.094),
        "EUR": st.session_state.get("eur_usd", 1.1)
    }

    df["Valuta"] = df["Valuta"].fillna("USD")
    df["Växelkurs"] = df["Valuta"].map(valuta_map)
    df["USD-kurs"] = df["Aktuell kurs"] * df["Växelkurs"]
    df["Värde (USD)"] = df["Antal aktier"] * df["USD-kurs"]
    total_portfolio_value = df["Värde (USD)"].sum()

    df["Riktkurs"] = df[riktkursval]
    df["Potential (%)"] = (df["Riktkurs"] - df["Aktuell kurs"]) / df["Aktuell kurs"] * 100
    df["Max andel"] = df["Max andel"].fillna(0)

    df = df[df["Riktkurs"] > 0]
    if visa_endast_portfolj:
        df = df[df["Antal aktier"] > 0]

    forslag = []
    for _, row in df.iterrows():
        nuvarande_andel = (row["Värde (USD)"] / total_portfolio_value) * 100 if total_portfolio_value > 0 else 0
        max_andel = row["Max andel"]
        usd_kurs = row["USD-kurs"]
        riktkurs = row["Riktkurs"]

        # Filtrera bort om innehavet redan överstiger max
        if max_andel > 0 and nuvarande_andel >= max_andel:
            continue

        # Beräkna hur många man kan köpa för att nå max
        max_varde_usd = (max_andel / 100) * total_portfolio_value
        diff_usd = max_varde_usd - row["Värde (USD)"]
        pris_per_aktie_usd = usd_kurs
        antal_kop = int((kapital / usd_sek) // pris_per_aktie_usd)

        if max_andel > 0 and diff_usd < kapital / usd_sek:
            antal_kop = int(diff_usd // pris_per_aktie_usd)

        if antal_kop <= 0:
            continue

        nytt_varde = row["Värde (USD)"] + antal_kop * pris_per_aktie_usd
        ny_andel = nytt_varde / total_portfolio_value * 100 if total_portfolio_value > 0 else 0

        forslag.append({
            "Bolag": row["Bolag"],
            "Aktuell kurs": row["Aktuell kurs"],
            "Valuta": row["Valuta"],
            "Riktkurs": riktkurs,
            "Potential (%)": round(row["Potential (%)"], 1),
            "Nuvarande andel (%)": round(nuvarande_andel, 2),
            "Efter köp (%)": round(ny_andel, 2),
            "Köp antal": antal_kop
        })

    st.subheader("📌 Förslag")
    if not forslag:
        st.info("Inga förslag matchar kriterierna.")
    else:
        df_forslag = pd.DataFrame(forslag).sort_values(by="Potential (%)", ascending=False)
        st.dataframe(df_forslag, use_container_width=True)

def uppdatera_kurser(df):
    st.header("🔄 Uppdatera kurser från Yahoo Finance")

    tickers = df["Yahoo-ticker"].fillna(df["Ticker"]).tolist()
    total = len(tickers)
    status = st.empty()
    bar = st.progress(0)

    nya_kurser = []
    for i, (index, rad) in enumerate(df.iterrows()):
        ticker = rad["Yahoo-ticker"] if pd.notna(rad["Yahoo-ticker"]) else rad["Ticker"]
        try:
            aktie = yf.Ticker(ticker)
            kurs = aktie.history(period="1d")["Close"].iloc[-1]
            nya_kurser.append(kurs)
        except:
            nya_kurser.append(rad["Aktuell kurs"])  # behåll tidigare om fel
        procent = int((i + 1) / total * 100)
        status.text(f"{procent}% klart ({i + 1}/{total})")
        bar.progress(procent)
        time.sleep(2)

    df["Aktuell kurs"] = nya_kurser
    st.success("Kurserna är uppdaterade!")
    return df

def main():
    st.set_page_config(page_title="Aktieanalys", layout="wide")
    st.title("📈 Aktieanalys & Investeringsverktyg")

    sheet = connect_to_gsheet(SHEET_URL)
    df = get_database(sheet)

    menyval = st.sidebar.radio("Välj vy", ["📦 Portfölj", "💡 Investeringsförslag", "➕ Lägg till / uppdatera bolag", "🔄 Uppdatera kurser"])

    if menyval == "📦 Portfölj":
        portfoljvy(df)

    elif menyval == "💡 Investeringsförslag":
        investeringsforslag(df)

    elif menyval == "➕ Lägg till / uppdatera bolag":
        df = lagg_till_eller_uppdatera(df)
        spara_df(sheet, df)

    elif menyval == "🔄 Uppdatera kurser":
        df = uppdatera_kurser(df)
        spara_df(sheet, df)

if __name__ == "__main__":
    main()
