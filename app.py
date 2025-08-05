import streamlit as st
import pandas as pd
import numpy as np
import gspread
import yfinance as yf
import time
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

# Google Sheets-konfiguration
SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

# --- Grundläggande funktioner för Sheets ---
def skapa_koppling():
    return client.open_by_url(SHEET_URL).worksheet(SHEET_NAME)

def hamta_data():
    data = skapa_koppling().get_all_records()
    return pd.DataFrame(data)

def spara_data(df):
    sheet = skapa_koppling()
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.astype(str).values.tolist())

# --- Hjälpfunktion för konvertering av numeriska kolumner ---
def konvertera_typer(df):
    kolumner = [
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "Omsättning om 4 år", "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Aktuell kurs", "Antal aktier"
    ]
    for kol in kolumner:
        if kol in df.columns:
            df[kol] = pd.to_numeric(df[kol], errors="coerce").fillna(0.0)
    return df

# --- Säkerställ att alla nödvändiga kolumner finns ---
def säkerställ_kolumner(df):
    nödvändiga = [
        "Ticker", "Bolagsnamn", "Aktuell kurs", "Valuta", "Utestående aktier",
        "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år",
        "Omsättning om 3 år", "Omsättning om 4 år",
        "P/S-snitt", "Riktkurs idag", "Riktkurs om 1 år",
        "Riktkurs om 2 år", "Riktkurs om 3 år", "Antal aktier"
    ]
    for kol in nödvändiga:
        if kol not in df.columns:
            if any(x in kol.lower() for x in ["kurs", "omsättning", "p/s", "antal"]):
                df[kol] = 0.0
            else:
                df[kol] = ""
    return df

# --- Tolka Yahoo Finance-siffror med B/M/T ---
def parse_yahoo_number(value):
    if value is None or value == "":
        return None
    if isinstance(value, str):
        value = value.replace(",", "").strip()
        try:
            if value.endswith("B"):
                return float(value[:-1]) * 1_000_000_000
            elif value.endswith("M"):
                return float(value[:-1]) * 1_000_000
            elif value.endswith("T"):
                return float(value[:-1]) * 1_000_000_000_000
            else:
                return float(value)
        except:
            return None
    try:
        return float(value)
    except:
        return None

# --- Hämta kurs, P/S och omsättning från Yahoo Finance ---
def hamta_ps_och_kurs(ticker):
    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info

        data = {}

        # Kurs
        pris = info.get("regularMarketPrice")
        data["Aktuell kurs"] = pris if pris is not None else None

        # Valuta
        valuta = info.get("currency", "USD")
        data["Valuta"] = valuta

        # Market Cap
        marketcap = parse_yahoo_number(info.get("marketCap"))

        # P/S idag
        oms_ttm = parse_yahoo_number(info.get("totalRevenue"))
        if oms_ttm and marketcap:
            data["P/S"] = marketcap / oms_ttm
        else:
            data["P/S"] = None

        # P/S Q1–Q4 (om kvartalsdata finns)
        try:
            kvartalsoms = yf_ticker.quarterly_financials.loc["Total Revenue"].values
            kvartalsoms = [parse_yahoo_number(x) for x in kvartalsoms if parse_yahoo_number(x)]
            for idx, oms in enumerate(kvartalsoms[:4], start=1):
                data[f"P/S Q{idx}"] = (marketcap / (oms * 4)) if oms and marketcap else None
        except Exception:
            for idx in range(1, 5):
                data[f"P/S Q{idx}"] = None

        # Omsättning idag, nästa år, om 2 år
        try:
            analysis = yf_ticker.analysis
            if "Revenue Estimate" in analysis.index:
                oms_idag = parse_yahoo_number(analysis.loc["Revenue Estimate"]["Avg"])
                oms_next = parse_yahoo_number(analysis.loc["Revenue Estimate"].iloc[1]["Avg"])
                oms_2y = parse_yahoo_number(analysis.loc["Revenue Estimate"].iloc[2]["Avg"])
                data["Omsättning idag"] = oms_idag
                data["Omsättning nästa år"] = oms_next
                data["Omsättning om 2 år"] = oms_2y
        except Exception:
            data["Omsättning idag"] = None
            data["Omsättning nästa år"] = None
            data["Omsättning om 2 år"] = None

        # --- CAGR för år 3 och år 4 ---
        try:
            hist_oms = yf_ticker.financials.loc["Total Revenue"].values
            hist_oms = [parse_yahoo_number(x) for x in hist_oms if parse_yahoo_number(x)]
            if len(hist_oms) >= 5 and data.get("Omsättning om 2 år"):
                oms_start = hist_oms[-1]  # 5 år sedan
                oms_slut = hist_oms[0]    # senaste året
                if oms_start > 0:
                    cagr = (oms_slut / oms_start) ** (1 / 5) - 1
                    cagr = min(max(cagr, 0.02), 0.50)  # golv 2%, tak 50%
                    data["Omsättning om 3 år"] = data["Omsättning om 2 år"] * (1 + cagr)
                    data["Omsättning om 4 år"] = data["Omsättning om 3 år"] * (1 + cagr)
                else:
                    data["Omsättning om 3 år"] = None
                    data["Omsättning om 4 år"] = None
            else:
                data["Omsättning om 3 år"] = None
                data["Omsättning om 4 år"] = None
        except Exception:
            data["Omsättning om 3 år"] = None
            data["Omsättning om 4 år"] = None

        return data
    except Exception:
        return None


# --- Analysvy med uppdatering och felrapport ---
def analysvy(df, valutakurser):
    st.subheader("📈 Analysläge")
    df = uppdatera_berakningar(df)

    if st.button("🔄 Uppdatera alla aktuella kurser och data från Yahoo"):
        fel_lista = []
        total = len(df)
        status = st.empty()

        with st.spinner("Uppdaterar..."):
            for i, row in df.iterrows():
                ticker = str(row["Ticker"]).strip().upper()
                status.text(f"🔄 Uppdaterar {i+1} av {total} ({ticker})...")
                data = hamta_ps_och_kurs(ticker)

                if not data:
                    fel_lista.append({"Ticker": ticker, "Fel": ["Hämtning misslyckades"]})
                    continue

                fel_falt = []

                # Uppdatera fält, logga fel
                for kolumn in [
                    "Aktuell kurs", "Valuta", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
                    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år",
                    "Omsättning om 3 år", "Omsättning om 4 år"
                ]:
                    if data.get(kolumn) is not None:
                        df.at[i, kolumn] = data[kolumn]
                    else:
                        fel_falt.append(kolumn)

                if fel_falt:
                    fel_lista.append({"Ticker": ticker, "Fel": fel_falt})

                time.sleep(1)  # väntetid mellan anropen

        spara_data(df)
        status.text("✅ Uppdatering slutförd.")

        if fel_lista:
            st.error("Vissa tickers kunde inte uppdateras helt.")
            df_fel = pd.DataFrame(fel_lista)
            st.dataframe(df_fel, use_container_width=True)

            # Kopieringsbar lista
            lista_text = "\n".join([f"{item['Ticker']} – {', '.join(item['Fel'])}" for item in fel_lista])
            st.text_area("Kopiera lista över misslyckade uppdateringar:", value=lista_text, height=200)

    st.dataframe(df, use_container_width=True)

def uppdatera_berakningar(df):
    for i, rad in df.iterrows():
        ps = [rad["P/S Q1"], rad["P/S Q2"], rad["P/S Q3"], rad["P/S Q4"]]
        ps = [x for x in ps if x > 0]
        ps_snitt = round(np.mean(ps), 2) if ps else 0
        df.at[i, "P/S-snitt"] = ps_snitt

        if rad["Utestående aktier"] > 0:
            df.at[i, "Riktkurs idag"] = round((rad["Omsättning idag"] * ps_snitt) / rad["Utestående aktier"], 2)
            df.at[i, "Riktkurs om 1 år"] = round((rad["Omsättning nästa år"] * ps_snitt) / rad["Utestående aktier"], 2)
            df.at[i, "Riktkurs om 2 år"] = round((rad["Omsättning om 2 år"] * ps_snitt) / rad["Utestående aktier"], 2)
            df.at[i, "Riktkurs om 3 år"] = round((rad["Omsättning om 3 år"] * ps_snitt) / rad["Utestående aktier"], 2)
    return df


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
        namn = st.text_input("Bolagsnamn", value=befintlig.get("Bolagsnamn", "") if not befintlig.empty else "")
        kurs = st.number_input("Aktuell kurs", value=float(befintlig.get("Aktuell kurs", 0.0)) if not befintlig.empty else 0.0)
        valuta = st.selectbox(
            "Valuta",
            ["USD", "NOK", "EUR", "CAD"],
            index=0 if befintlig.get("Valuta", "") == "" else ["USD", "NOK", "EUR", "CAD"].index(befintlig.get("Valuta", "USD"))
        )
        aktier = st.number_input("Utestående aktier (miljoner)", value=float(befintlig.get("Utestående aktier", 0.0)) if not befintlig.empty else 0.0)
        antal_aktier = st.number_input("Antal aktier du äger", value=float(befintlig.get("Antal aktier", 0.0)) if not befintlig.empty else 0.0)

        ps_idag = st.number_input("P/S idag", value=float(befintlig.get("P/S", 0.0)) if not befintlig.empty else 0.0)
        ps1 = st.number_input("P/S Q1", value=float(befintlig.get("P/S Q1", 0.0)) if not befintlig.empty else 0.0)
        ps2 = st.number_input("P/S Q2", value=float(befintlig.get("P/S Q2", 0.0)) if not befintlig.empty else 0.0)
        ps3 = st.number_input("P/S Q3", value=float(befintlig.get("P/S Q3", 0.0)) if not befintlig.empty else 0.0)
        ps4 = st.number_input("P/S Q4", value=float(befintlig.get("P/S Q4", 0.0)) if not befintlig.empty else 0.0)

        oms_idag = st.number_input("Omsättning idag (miljoner)", value=float(befintlig.get("Omsättning idag", 0.0)) if not befintlig.empty else 0.0)
        oms_1 = st.number_input("Omsättning nästa år", value=float(befintlig.get("Omsättning nästa år", 0.0)) if not befintlig.empty else 0.0)
        oms_2 = st.number_input("Omsättning om 2 år", value=float(befintlig.get("Omsättning om 2 år", 0.0)) if not befintlig.empty else 0.0)
        oms_3 = st.number_input("Omsättning om 3 år", value=float(befintlig.get("Omsättning om 3 år", 0.0)) if not befintlig.empty else 0.0)
        oms_4 = st.number_input("Omsättning om 4 år", value=float(befintlig.get("Omsättning om 4 år", 0.0)) if not befintlig.empty else 0.0)

        sparaknapp = st.form_submit_button("💾 Spara")

    if sparaknapp and ticker:
        ny_rad = {
            "Ticker": ticker, "Bolagsnamn": namn, "Aktuell kurs": kurs, "Valuta": valuta,
            "Utestående aktier": aktier, "Antal aktier": antal_aktier,
            "P/S": ps_idag, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms_1,
            "Omsättning om 2 år": oms_2, "Omsättning om 3 år": oms_3, "Omsättning om 4 år": oms_4
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

    if df_forslag.empty:
        st.info("Inga bolag matchar kriterierna just nu.")
        return

    if 'forslags_index' not in st.session_state:
        st.session_state.forslags_index = 0

    index = st.session_state.forslags_index
    if index >= len(df_forslag):
        st.info("Inga fler förslag att visa.")
        return

    rad = df_forslag.iloc[index]
    kapital_valuta = kapital_sek / valutakurser.get(rad["Valuta"], 1)

    antal = int(kapital_valuta // rad["Aktuell kurs"])
    investering_sek = antal * rad["Aktuell kurs"] * valutakurser.get(rad["Valuta"], 1)

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
    total = df["Värde (SEK)"].sum()
    st.markdown(f"**Totalt portföljvärde:** {round(total, 2)} SEK")
    st.dataframe(df[["Ticker", "Bolagsnamn", "Antal aktier", "Valuta", "Aktuell kurs", "Värde (SEK)", "Andel (%)"]], use_container_width=True)


def main():
    st.title("📊 Aktieanalys och investeringsförslag")
    df = hamta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)

    st.sidebar.header("Valutakurser")
    valutakurser = {
        "USD": st.sidebar.number_input("USD → SEK", value=9.50, step=0.01),
        "NOK": st.sidebar.number_input("NOK → SEK", value=0.93, step=0.01),
        "EUR": st.sidebar.number_input("EUR → SEK", value=11.10, step=0.01),
        "CAD": st.sidebar.number_input("CAD → SEK", value=7.00, step=0.01)
    }

    meny = st.sidebar.radio("Meny", ["Analys", "Lägg till / uppdatera bolag", "Investeringsförslag", "Portfölj"])

    if meny == "Analys":
        analysvy(df, valutakurser)
    elif meny == "Lägg till / uppdatera bolag":
        df = lagg_till_eller_uppdatera(df)
        spara_data(df)
    elif meny == "Investeringsförslag":
        df = uppdatera_berakningar(df)
        visa_investeringsforslag(df, valutakurser)
    elif meny == "Portfölj":
        df = uppdatera_berakningar(df)
        visa_portfolj(df, valutakurser)


if __name__ == "__main__":
    main()
