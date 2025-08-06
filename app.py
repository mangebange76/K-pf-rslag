import streamlit as st
import pandas as pd
import numpy as np
import gspread
import yfinance as yf
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

# --- Google Sheets konfiguration ---
SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

def skapa_koppling():
    return client.open_by_url(SHEET_URL).worksheet(SHEET_NAME)

def hamta_data():
    return pd.DataFrame(skapa_koppling().get_all_records())

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
        "Ticker", "Bolagsnamn", "Aktuell kurs", "Valuta", "Utestående aktier",
        "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "P/S-snitt", "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
        "Antal aktier", "Årlig utdelning"
    ]
    for kol in nödvändiga:
        if kol not in df.columns:
            if any(x in kol.lower() for x in ["kurs", "omsättning", "p/s", "antal", "utdelning"]):
                df[kol] = 0.0
            else:
                df[kol] = ""
    return df

def uppdatera_berakningar(df):
    for i, rad in df.iterrows():
        ps = [rad["P/S Q1"], rad["P/S Q2"], rad["P/S Q3"], rad["P/S Q4"]]
        ps = [x for x in ps if isinstance(x, (int, float)) and x > 0]
        ps_snitt = round(np.mean(ps), 2) if ps else 0
        df.at[i, "P/S-snitt"] = ps_snitt

        if rad["Utestående aktier"] > 0:
            df.at[i, "Riktkurs idag"] = round((rad["Omsättning idag"] * ps_snitt) / rad["Utestående aktier"], 2)
            df.at[i, "Riktkurs om 1 år"] = round((rad["Omsättning nästa år"] * ps_snitt) / rad["Utestående aktier"], 2)
            df.at[i, "Riktkurs om 2 år"] = round((rad["Omsättning om 2 år"] * ps_snitt) / rad["Utestående aktier"], 2)
            df.at[i, "Riktkurs om 3 år"] = round((rad["Omsättning om 3 år"] * ps_snitt) / rad["Utestående aktier"], 2)
    return df

def hamta_kurs_valuta_ps_oms(ticker):
    """Hämtar kurs, valuta, P/S och omsättningar från Yahoo."""
    try:
        info = yf.Ticker(ticker).info
        data = {
            "Aktuell kurs": info.get("regularMarketPrice", None),
            "Valuta": info.get("currency", None),
            "P/S": info.get("priceToSalesTrailing12Months", None),
            "Omsättning idag": info.get("totalRevenue", None)
        }
        return data
    except Exception:
        return {}

def lagg_till_eller_uppdatera(df):
    st.subheader("➕ Lägg till / uppdatera bolag")

    # Sortera tickers i alfabetisk ordning
    tickers = sorted(df["Ticker"].unique())
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0

    # Rullista för att välja bolag
    selected_ticker = st.selectbox(
        "Välj bolag att uppdatera (eller lämna tom för nytt)",
        [""] + tickers,
        index=(st.session_state.current_index + 1 if tickers else 0)
    )

    # Om man väljer i rullistan, synka index
    if selected_ticker:
        st.session_state.current_index = tickers.index(selected_ticker)

    # Hämta befintlig rad
    if selected_ticker:
        befintlig = df[df["Ticker"] == selected_ticker].iloc[0]
    else:
        befintlig = pd.Series(dtype=object)

    with st.form("form"):
        ticker = st.text_input("Ticker", value=befintlig.get("Ticker", "") if not befintlig.empty else "").upper()
        namn = st.text_input("Bolagsnamn", value=befintlig.get("Bolagsnamn", "") if not befintlig.empty else "")

        kurs = st.number_input("Aktuell kurs", value=float(befintlig.get("Aktuell kurs", "") or 0.0), step=0.01, format="%.2f")
        valuta = st.text_input("Valuta", value=befintlig.get("Valuta", "") if not befintlig.empty else "")

        aktier_utest = st.number_input("Utestående aktier (miljoner)", value=float(befintlig.get("Utestående aktier", "") or 0.0), step=0.01, format="%.2f")
        antal_aktier = st.number_input("Antal aktier du äger", value=float(befintlig.get("Antal aktier", "") or 0.0), step=1.0, format="%.0f")

        ps_idag = st.number_input("P/S idag", value=float(befintlig.get("P/S", "") or 0.0), step=0.01, format="%.2f")
        ps1 = st.number_input("P/S Q1", value=float(befintlig.get("P/S Q1", "") or 0.0), step=0.01, format="%.2f")
        ps2 = st.number_input("P/S Q2", value=float(befintlig.get("P/S Q2", "") or 0.0), step=0.01, format="%.2f")
        ps3 = st.number_input("P/S Q3", value=float(befintlig.get("P/S Q3", "") or 0.0), step=0.01, format="%.2f")
        ps4 = st.number_input("P/S Q4", value=float(befintlig.get("P/S Q4", "") or 0.0), step=0.01, format="%.2f")

        oms_idag = st.number_input("Omsättning idag", value=float(befintlig.get("Omsättning idag", "") or 0.0), step=0.01, format="%.2f")
        oms_1 = st.number_input("Omsättning nästa år", value=float(befintlig.get("Omsättning nästa år", "") or 0.0), step=0.01, format="%.2f")
        oms_2 = st.number_input("Omsättning om 2 år", value=float(befintlig.get("Omsättning om 2 år", "") or 0.0), step=0.01, format="%.2f")
        oms_3 = st.number_input("Omsättning om 3 år", value=float(befintlig.get("Omsättning om 3 år", "") or 0.0), step=0.01, format="%.2f")

        utd = st.number_input("Årlig utdelning", value=float(befintlig.get("Årlig utdelning", "") or 0.0), step=0.01, format="%.2f")

        sparaknapp = st.form_submit_button("💾 Spara")

    if sparaknapp and ticker:
        # Skapa ny rad-data
        ny_rad = {
            "Ticker": ticker, "Bolagsnamn": namn, "Aktuell kurs": kurs, "Valuta": valuta,
            "Utestående aktier": aktier_utest, "Antal aktier": antal_aktier,
            "P/S": ps_idag, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms_1,
            "Omsättning om 2 år": oms_2, "Omsättning om 3 år": oms_3,
            "Årlig utdelning": utd
        }

        # Lägg till eller uppdatera i df
        if ticker in df["Ticker"].values:
            for kol, val in ny_rad.items():
                df.loc[df["Ticker"] == ticker, kol] = val
            st.success(f"{ticker} uppdaterat.")
        else:
            df = pd.concat([df, pd.DataFrame([ny_rad])], ignore_index=True)
            st.success(f"{ticker} tillagt.")

        # Hämta färsk data från Yahoo
        yahoo_data = hamta_kurs_valuta_ps_oms(ticker)
        uppdaterade_falt = []
        for kol, val in yahoo_data.items():
            if val is not None:
                df.loc[df["Ticker"] == ticker, kol] = val
                uppdaterade_falt.append(kol)

        if uppdaterade_falt:
            st.info(f"Från Yahoo uppdaterades: {', '.join(uppdaterade_falt)}")

        spara_data(df)

    # Bläddringsknappar
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅ Föregående") and tickers:
            st.session_state.current_index = (st.session_state.current_index - 1) % len(tickers)
            st.experimental_rerun()
    with col2:
        if st.button("Nästa ➡") and tickers:
            st.session_state.current_index = (st.session_state.current_index + 1) % len(tickers)
            st.experimental_rerun()

    return df

def berakna_cagr(start, slut, ar):
    if start <= 0 or slut <= 0 or ar <= 0:
        return None
    return (slut / start) ** (1 / ar) - 1

def räkna_omsättning_cagr(df):
    for i, rad in df.iterrows():
        oms_idag = rad["Omsättning idag"]
        oms_nasta = rad["Omsättning nästa år"]

        if oms_idag > 0 and oms_nasta > 0:
            tillväxt = berakna_cagr(oms_idag, oms_nasta, 1)
            if tillväxt is not None:
                tillväxt = max(min(tillväxt, 0.50), -0.02)  # Tak 50%, golv -2%
                df.at[i, "Omsättning om 2 år"] = round(oms_nasta * (1 + tillväxt), 2)
                df.at[i, "Omsättning om 3 år"] = round(df.at[i, "Omsättning om 2 år"] * (1 + tillväxt), 2)
    return df

def hantera_valutakurser():
    st.sidebar.subheader("💱 Valutakurser till SEK")
    valutakurser = {
        "USD": st.sidebar.number_input("USD → SEK", value=9.50, step=0.01),
        "NOK": st.sidebar.number_input("NOK → SEK", value=0.93, step=0.01),
        "EUR": st.sidebar.number_input("EUR → SEK", value=11.10, step=0.01),
        "CAD": st.sidebar.number_input("CAD → SEK", value=7.00, step=0.01)
    }
    return valutakurser

def visa_investeringsforslag(df, valutakurser):
    st.subheader("💡 Investeringsförslag")
    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=500.0, step=500.0)

    riktkurs_val = st.selectbox(
        "Vilken riktkurs ska användas?",
        ["Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år"],
        index=1
    )

    filterval = st.radio("Visa förslag för:", ["Alla bolag", "Endast portföljen"])

    df_portfolj = df[df["Antal aktier"] > 0].copy()
    df_portfolj["Värde (SEK)"] = df_portfolj.apply(
        lambda r: r["Antal aktier"] * r["Aktuell kurs"] * valutakurser.get(r["Valuta"], 1), axis=1
    )
    portfoljvarde = df_portfolj["Värde (SEK)"].sum()

    if filterval == "Endast portföljen":
        df_forslag = df_portfolj[df_portfolj[riktkurs_val] > df_portfolj["Aktuell kurs"]].copy()
    else:
        df_forslag = df[df[riktkurs_val] > df["Aktuell kurs"]].copy()

    df_forslag["Potential (%)"] = ((df_forslag[riktkurs_val] - df_forslag["Aktuell kurs"]) / df_forslag["Aktuell kurs"]) * 100
    df_forslag = df_forslag.sort_values(by="Potential (%)", ascending=False).reset_index(drop=True)

    kapital_usd = kapital_sek / valutakurser.get("USD", 1)

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

def visa_portfolj(df, valutakurser):
    st.subheader("📦 Min portfölj")
    df = df[df["Antal aktier"] > 0].copy()
    if df.empty:
        st.info("Du äger inga aktier.")
        return

    df["Värde (SEK)"] = df.apply(
        lambda r: r["Antal aktier"] * r["Aktuell kurs"] * valutakurser.get(r["Valuta"], 1), axis=1
    )
    df["Andel (%)"] = round(df["Värde (SEK)"] / df["Värde (SEK)"].sum() * 100, 2)
    total_varde = df["Värde (SEK)"].sum()

    total_utdelning = (df["Årlig utdelning"] * df["Antal aktier"] *
                       df["Valuta"].map(valutakurser).fillna(1)).sum()
    manadsutdelning = total_utdelning / 12

    st.markdown(f"**Totalt portföljvärde:** {round(total_varde, 2)} SEK")
    st.markdown(f"**Förväntad årlig utdelning:** {round(total_utdelning, 2)} SEK")
    st.markdown(f"**Förväntad månadsutdelning:** {round(manadsutdelning, 2)} SEK")

    st.dataframe(df[[
        "Ticker", "Bolagsnamn", "Antal aktier", "Aktuell kurs", "Valuta",
        "Värde (SEK)", "Andel (%)"
    ]], use_container_width=True)

def analysvy(df):
    st.subheader("📈 Analysläge")
    valutakurser = hantera_valutakurser()

    if st.button("🔄 Uppdatera alla från Yahoo"):
        misslyckade = {}
        total = len(df)
        status = st.empty()

        for i, row in df.iterrows():
            ticker = str(row["Ticker"]).strip()
            status.text(f"🔄 {i+1}/{total} – {ticker}")

            yahoo_data = hamta_kurs_valuta_ps_oms(ticker)
            for kol, val in yahoo_data.items():
                if val is not None:
                    df.at[i, kol] = val
                else:
                    misslyckade.setdefault(ticker, []).append(kol)

        spara_data(df)
        status.text("✅ Uppdatering slutförd.")
        if misslyckade:
            miss_str = "\n".join(f"{t}: {', '.join(fel)}" for t, fel in misslyckade.items())
            st.text_area("Misslyckade uppdateringar", miss_str)

    st.markdown("### 📄 Hela databasen")
    st.dataframe(df, use_container_width=True)

def main():
    st.title("📊 Aktieanalys och investeringsförslag")
    df = hamta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)

    meny = st.sidebar.radio("Meny", [
        "Analys", "Lägg till / uppdatera bolag", "Investeringsförslag", "Portfölj"
    ])

    if meny == "Analys":
        analysvy(df)

    elif meny == "Lägg till / uppdatera bolag":
        df = lagg_till_eller_uppdatera(df)
        spara_data(df)

    elif meny == "Investeringsförslag":
        df = räkna_omsättning_cagr(df)
        df = uppdatera_berakningar(df)
        valutakurser = hantera_valutakurser()
        visa_investeringsforslag(df, valutakurser)

    elif meny == "Portfölj":
        df = uppdatera_berakningar(df)
        valutakurser = hantera_valutakurser()
        visa_portfolj(df, valutakurser)

if __name__ == "__main__":
    main()
