import streamlit as st
import pandas as pd
import numpy as np
import gspread
import yfinance as yf
import time
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

STANDARD_VALUTAKURSER = {
    "USD": 9.75,
    "NOK": 0.95,
    "CAD": 7.05,
    "EUR": 11.18,
    "SEK": 1.0
}

KOLUMNER = [
    "Ticker", "Bolagsnamn", "Aktuell kurs", "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
    "P/S-snitt", "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
    "Antal aktier", "Valuta", "Årlig utdelning", "CAGR 5 år (%)"
]

KOLUMNER_ATT_TA_BORT = [
    "P/S idag", "Riktkurs 2026", "Riktkurs 2027", "Riktkurs 2028", "Yahoo ticker", "Bolag",
    "Max andel", "Omsättning om 4 år", "P/S metod", "Initierad", "21"
]

def skapa_koppling():
    return client.open_by_url(SHEET_URL).worksheet(SHEET_NAME)

def hamta_data():
    sheet = skapa_koppling()
    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    df = df.drop(columns=[col for col in df.columns if col in KOLUMNER_ATT_TA_BORT], errors="ignore")
    for kol in KOLUMNER:
        if kol not in df.columns:
            df[kol] = 0.0 if "kurs" in kol.lower() or "omsättning" in kol.lower() or "p/s" in kol.lower() or "utdelning" in kol.lower() or "CAGR" in kol else ""
    return df[KOLUMNER]

def spara_data(df):
    sheet = skapa_koppling()
    sheet.clear()
    sheet.update([df.columns.tolist()] + df.astype(str).values.tolist())

def konvertera_typer(df):
    num_kol = [
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Aktuell kurs", "Antal aktier", "Årlig utdelning", "P/S-snitt",
        "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år", "CAGR 5 år (%)"
    ]
    for kol in num_kol:
        if kol in df.columns:
            df[kol] = pd.to_numeric(df[kol], errors="coerce").fillna(0.0)
    return df

def beräkna_alla_kolumner(df):
    for i, rad in df.iterrows():
        ps_värden = [rad["P/S Q1"], rad["P/S Q2"], rad["P/S Q3"], rad["P/S Q4"]]
        ps = [x for x in ps_värden if x > 0]
        ps_snitt = round(np.mean(ps), 2) if ps else 0.0
        df.at[i, "P/S-snitt"] = ps_snitt

        # Beräkna omsättning om 2 och 3 år
        cagr = rad["CAGR 5 år (%)"] / 100
        oms1 = rad["Omsättning nästa år"]
        oms2 = oms1 * (1 + cagr)
        oms3 = oms1 * (1 + cagr)**2
        df.at[i, "Omsättning om 2 år"] = round(oms2, 2)
        df.at[i, "Omsättning om 3 år"] = round(oms3, 2)

        if rad["Utestående aktier"] > 0:
            df.at[i, "Riktkurs idag"] = round((rad["Omsättning idag"] * ps_snitt) / rad["Utestående aktier"], 2)
            df.at[i, "Riktkurs om 1 år"] = round((oms1 * ps_snitt) / rad["Utestående aktier"], 2)
            df.at[i, "Riktkurs om 2 år"] = round((oms2 * ps_snitt) / rad["Utestående aktier"], 2)
            df.at[i, "Riktkurs om 3 år"] = round((oms3 * ps_snitt) / rad["Utestående aktier"], 2)
    return df

def hamta_data_yahoo(ticker):
    try:
        yf_obj = yf.Ticker(ticker)
        info = yf_obj.info
        namn = info.get("longName") or info.get("shortName") or ""
        kurs = info.get("currentPrice") or info.get("regularMarketPrice") or 0.0
        valuta = info.get("currency", "USD")
        utd = info.get("dividendRate", 0.0)
        cagr = None

        try:
            growth = yf_obj.analysis
            if "Revenue Estimate" in growth.index and "Next 5 Years (per annum)" in growth.columns:
                cagr_str = growth.loc["Revenue Estimate", "Next 5 Years (per annum)"]
                if isinstance(cagr_str, str) and "%" in cagr_str:
                    cagr = float(cagr_str.replace("%", "").strip())
        except:
            pass

        return {
            "Bolagsnamn": namn,
            "Aktuell kurs": kurs,
            "Valuta": valuta,
            "Årlig utdelning": utd,
            "CAGR 5 år (%)": cagr if cagr is not None else 0.0
        }
    except Exception:
        return {
            "Bolagsnamn": "", "Aktuell kurs": 0.0, "Valuta": "USD",
            "Årlig utdelning": 0.0, "CAGR 5 år (%)": 0.0
        }

def lagg_till_eller_uppdatera(df):
    st.subheader("➕ Lägg till / uppdatera bolag")
    namn_map = {f"{rad['Bolagsnamn']} ({rad['Ticker']})": rad['Ticker'] for _, rad in df.iterrows()}
    valt = st.selectbox("Välj bolag att uppdatera (eller lämna tom för nytt)", [""] + sorted(namn_map.keys()))

    if valt:
        ticker_vald = namn_map[valt]
        befintlig = df[df["Ticker"] == ticker_vald].iloc[0]
    else:
        befintlig = pd.Series(dtype=object)

    with st.form("form"):
        ticker = st.text_input("Ticker", value=befintlig.get("Ticker", "") if not befintlig.empty else "").upper()
        utestående = st.number_input("Utestående aktier (miljoner)", value=float(befintlig.get("Utestående aktier", 0.0)) if not befintlig.empty else 0.0)
        antal = st.number_input("Antal aktier du äger", value=float(befintlig.get("Antal aktier", 0.0)) if not befintlig.empty else 0.0)

        ps = st.number_input("P/S", value=float(befintlig.get("P/S", 0.0)) if not befintlig.empty else 0.0)
        ps_q1 = st.number_input("P/S Q1", value=float(befintlig.get("P/S Q1", 0.0)) if not befintlig.empty else 0.0)
        ps_q2 = st.number_input("P/S Q2", value=float(befintlig.get("P/S Q2", 0.0)) if not befintlig.empty else 0.0)
        ps_q3 = st.number_input("P/S Q3", value=float(befintlig.get("P/S Q3", 0.0)) if not befintlig.empty else 0.0)
        ps_q4 = st.number_input("P/S Q4", value=float(befintlig.get("P/S Q4", 0.0)) if not befintlig.empty else 0.0)

        oms_idag = st.number_input("Omsättning idag", value=float(befintlig.get("Omsättning idag", 0.0)) if not befintlig.empty else 0.0)
        oms_1 = st.number_input("Omsättning nästa år", value=float(befintlig.get("Omsättning nästa år", 0.0)) if not befintlig.empty else 0.0)

        sparaknapp = st.form_submit_button("💾 Hämta och spara")

    if sparaknapp and ticker:
        yahoo_data = hamta_data_yahoo(ticker)
        ny_rad = {
            "Ticker": ticker,
            "Bolagsnamn": yahoo_data["Bolagsnamn"],
            "Aktuell kurs": yahoo_data["Aktuell kurs"],
            "Valuta": yahoo_data["Valuta"],
            "Årlig utdelning": yahoo_data["Årlig utdelning"],
            "CAGR 5 år (%)": yahoo_data["CAGR 5 år (%)"],
            "Utestående aktier": utestående,
            "Antal aktier": antal,
            "P/S": ps,
            "P/S Q1": ps_q1,
            "P/S Q2": ps_q2,
            "P/S Q3": ps_q3,
            "P/S Q4": ps_q4,
            "Omsättning idag": oms_idag,
            "Omsättning nästa år": oms_1,
        }

        if ticker in df["Ticker"].values:
            df.loc[df["Ticker"] == ticker, ny_rad.keys()] = ny_rad.values()
            st.success(f"{ticker} uppdaterat med data från Yahoo Finance.")
        else:
            df = pd.concat([df, pd.DataFrame([ny_rad])], ignore_index=True)
            st.success(f"{ticker} tillagt med data från Yahoo Finance.")
    return df

def analysvy(df):
    st.subheader("📈 Analysläge")
    df = uppdatera_berakningar(df)

    if st.button("🔄 Uppdatera hela databasen från Yahoo Finance"):
        misslyckade = []
        uppdaterade = 0
        total = len(df)
        status = st.empty()
        bar = st.progress(0)

        with st.spinner("Uppdaterar..."):
            for i, row in df.iterrows():
                ticker = str(row["Ticker"]).strip().upper()
                status.text(f"🔄 Uppdaterar {i + 1} av {total} ({ticker})...")

                try:
                    yahoo_data = hamta_data_yahoo(ticker)

                    df.at[i, "Bolagsnamn"] = yahoo_data["Bolagsnamn"]
                    df.at[i, "Aktuell kurs"] = yahoo_data["Aktuell kurs"]
                    df.at[i, "Valuta"] = yahoo_data["Valuta"]
                    df.at[i, "Årlig utdelning"] = yahoo_data["Årlig utdelning"]
                    df.at[i, "CAGR 5 år (%)"] = yahoo_data["CAGR 5 år (%)"]
                    uppdaterade += 1
                except Exception:
                    misslyckade.append(ticker)

                bar.progress((i + 1) / total)
                time.sleep(1)

        spara_data(df)
        status.text("✅ Uppdatering klar.")
        st.success(f"{uppdaterade} tickers uppdaterade.")
        if misslyckade:
            st.warning("Kunde inte uppdatera följande tickers:\n" + ", ".join(misslyckade))

    st.dataframe(df, use_container_width=True)

def visa_portfolj(df, valutakurser):
    st.subheader("📦 Min portfölj")
    df = df[df["Antal aktier"] > 0].copy()
    if df.empty:
        st.info("Du äger inga aktier.")
        return

    df["Växelkurs"] = df["Valuta"].map(valutakurser).fillna(1.0)
    df["Värde (SEK)"] = df["Antal aktier"] * df["Aktuell kurs"] * df["Växelkurs"]
    df["Andel (%)"] = round(df["Värde (SEK)"] / df["Värde (SEK)"].sum() * 100, 2)

    df["Total årlig utdelning"] = df["Antal aktier"] * df["Årlig utdelning"] * df["Växelkurs"]
    total_utdelning = df["Total årlig utdelning"].sum()
    total_varde = df["Värde (SEK)"].sum()

    st.markdown(f"**Totalt portföljvärde:** {round(total_varde, 2)} SEK")
    st.markdown(f"**Förväntad årlig utdelning:** {round(total_utdelning, 2)} SEK")
    st.markdown(f"**Genomsnittlig månadsutdelning:** {round(total_utdelning / 12, 2)} SEK")

    st.dataframe(
        df[["Ticker", "Bolagsnamn", "Antal aktier", "Aktuell kurs", "Valuta", "Värde (SEK)", "Andel (%)", "Årlig utdelning", "Total årlig utdelning"]],
        use_container_width=True
    )


def visa_investeringsforslag(df, valutakurser):
    st.subheader("💡 Investeringsförslag")
    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=500.0, step=500.0)

    riktkurs_val = st.selectbox(
        "Vilken riktkurs ska användas?",
        ["Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år"],
        index=1
    )

    filterval = st.radio("Visa förslag för:", ["Alla bolag", "Endast portföljen"])

    df["Växelkurs"] = df["Valuta"].map(valutakurser).fillna(1.0)
    df_portfolj = df[df["Antal aktier"] > 0].copy()
    df_portfolj["Värde (SEK)"] = df_portfolj["Antal aktier"] * df_portfolj["Aktuell kurs"] * df_portfolj["Växelkurs"]
    portfoljvarde = df_portfolj["Värde (SEK)"].sum()

    if filterval == "Endast portföljen":
        df_forslag = df_portfolj[df_portfolj[riktkurs_val] > df_portfolj["Aktuell kurs"]].copy()
    else:
        df_forslag = df[df[riktkurs_val] > df["Aktuell kurs"]].copy()

    df_forslag["Potential (%)"] = ((df_forslag[riktkurs_val] - df_forslag["Aktuell kurs"]) / df_forslag["Aktuell kurs"]) * 100
    df_forslag = df_forslag.sort_values(by="Potential (%)", ascending=False).reset_index(drop=True)

    if df_forslag.empty:
        st.info("Inga bolag matchar kriterierna just nu.")
        return

    if "forslags_index" not in st.session_state:
        st.session_state.forslags_index = 0

    index = st.session_state.forslags_index
    if index >= len(df_forslag):
        st.info("Inga fler förslag att visa.")
        return

    rad = df_forslag.iloc[index]
    kurs_sek = rad["Aktuell kurs"] * rad["Växelkurs"]
    antal = int(kapital_sek // kurs_sek)
    investering_sek = antal * kurs_sek

    nuvarande_innehav = df_portfolj[df_portfolj["Ticker"] == rad["Ticker"]]["Värde (SEK)"].sum()
    ny_total = nuvarande_innehav + investering_sek
    nuvarande_andel = round((nuvarande_innehav / portfoljvarde) * 100, 2) if portfoljvarde > 0 else 0
    ny_andel = round((ny_total / portfoljvarde) * 100, 2) if portfoljvarde > 0 else 0

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


def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    df = hamta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)
    df = uppdatera_berakningar(df)

    st.sidebar.header("💱 Valutakurser till SEK")
    valutakurser = {
        "USD": st.sidebar.number_input("USD → SEK", value=9.75, step=0.01),
        "NOK": st.sidebar.number_input("NOK → SEK", value=0.95, step=0.01),
        "CAD": st.sidebar.number_input("CAD → SEK", value=7.05, step=0.01),
        "EUR": st.sidebar.number_input("EUR → SEK", value=11.18, step=0.01),
    }

    meny = st.sidebar.radio("📌 Välj vy", ["Analys", "Lägg till / uppdatera bolag", "Investeringsförslag", "Portfölj"])

    if meny == "Analys":
        analysvy(df)
    elif meny == "Lägg till / uppdatera bolag":
        df = lagg_till_eller_uppdatera(df)
        spara_data(df)
    elif meny == "Investeringsförslag":
        visa_investeringsforslag(df, valutakurser)
    elif meny == "Portfölj":
        visa_portfolj(df, valutakurser)


if __name__ == "__main__":
    main()
