import streamlit as st
import pandas as pd
import numpy as np
import gspread
import yfinance as yf
import time
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

# ------------------------------
# KONFIGURATION OCH GOOGLE SHEETS
# ------------------------------
SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

def skapa_koppling():
    return client.open_by_url(SHEET_URL).worksheet(SHEET_NAME)

def hamta_data():
    data = skapa_koppling().get_all_records()
    return pd.DataFrame(data)

def spara_data(df):
    sheet = skapa_koppling()
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.astype(str).values.tolist())

# ------------------------------
# DATAHANTERING
# ------------------------------
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
        "Ticker", "Bolagsnamn", "Valuta", "Aktuell kurs", "Utestående aktier",
        "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "P/S-snitt", "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
        "Antal aktier"
    ]
    for kol in nödvändiga:
        if kol not in df.columns:
            if kol in ["Ticker", "Bolagsnamn", "Valuta"]:
                df[kol] = ""
            else:
                df[kol] = 0.0
    return df

def uppdatera_berakningar(df):
    for i, rad in df.iterrows():
        ps = [rad["P/S Q1"], rad["P/S Q2"], rad["P/S Q3"], rad["P/S Q4"]]
        ps = [min(x, 100) for x in ps if x > 0]  # Mjukt tak på 100
        ps_snitt = round(np.mean(ps), 2) if ps else 0
        df.at[i, "P/S-snitt"] = ps_snitt

        if rad["Utestående aktier"] > 0 and ps_snitt > 0:
            for år, kol in enumerate(["Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år"]):
                oms_kol = ["Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år"][år]
                df.at[i, kol] = round((rad[oms_kol] * ps_snitt) / rad["Utestående aktier"], 2)
    return df

# ------------------------------
# YAHOO FINANCE HÄMTNING
# ------------------------------
def hamta_kurs_och_valuta(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info.get("regularMarketPrice", None), info.get("currency", "USD")
    except Exception:
        return None, "USD"

def hamta_valutakurs(valuta):
    if valuta == "USD":
        return 1.0
    elif valuta == "NOK":
        return 0.093
    elif valuta == "CAD":
        return 0.74
    elif valuta == "EUR":
        return 1.05
    elif valuta == "SEK":
        return 0.095
    else:
        return 1.0

def lagg_till_eller_uppdatera(df):
    st.subheader("➕ Lägg till / uppdatera bolag")

    # Visa bolag i listan som "Namn (Ticker)"
    namn_map = {f"{rad['Bolagsnamn']} ({rad['Ticker']})": rad['Ticker'] for _, rad in df.iterrows()}
    valt = st.selectbox("Välj bolag att uppdatera (eller lämna tom för nytt)", [""] + sorted(namn_map.keys()))

    if valt:
        ticker_vald = namn_map[valt]
        befintlig = df[df["Ticker"] == ticker_vald].iloc[0]
    else:
        befintlig = pd.Series(dtype=object)

    with st.form("form"):
        # --- Manuella huvudfält ---
        ticker = st.text_input("Ticker", value=befintlig.get("Ticker", "") if not befintlig.empty else "").upper()
        utest_aktier = st.number_input(
            "Utestående aktier (miljoner)",
            value=float(befintlig.get("Utestående aktier", 0.0)) if not befintlig.empty else 0.0
        )
        oms_idag = st.number_input(
            "Omsättning idag (miljoner, i bolagets valuta)",
            value=float(befintlig.get("Omsättning idag", 0.0)) if not befintlig.empty else 0.0
        )

        # --- Avancerade fält ---
        with st.expander("Visa avancerade fält"):
            namn = st.text_input("Bolagsnamn", value=befintlig.get("Bolagsnamn", "") if not befintlig.empty else "")
            valuta = st.text_input("Valuta", value=befintlig.get("Valuta", "") if not befintlig.empty else "")

            kurs = st.number_input("Aktuell kurs", value=float(befintlig.get("Aktuell kurs", 0.0)) if not befintlig.empty else 0.0)
            antal_aktier = st.number_input("Antal aktier du äger", value=float(befintlig.get("Antal aktier", 0.0)) if not befintlig.empty else 0.0)

            ps_idag = st.number_input("P/S idag", value=float(befintlig.get("P/S", 0.0)) if not befintlig.empty else 0.0)
            ps1 = st.number_input("P/S Q1", value=float(befintlig.get("P/S Q1", 0.0)) if not befintlig.empty else 0.0)
            ps2 = st.number_input("P/S Q2", value=float(befintlig.get("P/S Q2", 0.0)) if not befintlig.empty else 0.0)
            ps3 = st.number_input("P/S Q3", value=float(befintlig.get("P/S Q3", 0.0)) if not befintlig.empty else 0.0)
            ps4 = st.number_input("P/S Q4", value=float(befintlig.get("P/S Q4", 0.0)) if not befintlig.empty else 0.0)

            oms_1 = st.number_input("Omsättning nästa år", value=float(befintlig.get("Omsättning nästa år", 0.0)) if not befintlig.empty else 0.0)
            oms_2 = st.number_input("Omsättning om 2 år", value=float(befintlig.get("Omsättning om 2 år", 0.0)) if not befintlig.empty else 0.0)
            oms_3 = st.number_input("Omsättning om 3 år", value=float(befintlig.get("Omsättning om 3 år", 0.0)) if not befintlig.empty else 0.0)

        sparaknapp = st.form_submit_button("💾 Spara")

    if sparaknapp and ticker:
        # Skapa ny rad från manuella inmatningar
        ny_rad = {
            "Ticker": ticker,
            "Bolagsnamn": namn if 'namn' in locals() else "",
            "Valuta": valuta if 'valuta' in locals() else "",
            "Aktuell kurs": kurs if 'kurs' in locals() else 0.0,
            "Utestående aktier": utest_aktier,
            "Antal aktier": antal_aktier if 'antal_aktier' in locals() else 0.0,
            "P/S": ps_idag if 'ps_idag' in locals() else 0.0,
            "P/S Q1": ps1 if 'ps1' in locals() else 0.0,
            "P/S Q2": ps2 if 'ps2' in locals() else 0.0,
            "P/S Q3": ps3 if 'ps3' in locals() else 0.0,
            "P/S Q4": ps4 if 'ps4' in locals() else 0.0,
            "Omsättning idag": oms_idag,
            "Omsättning nästa år": oms_1 if 'oms_1' in locals() else 0.0,
            "Omsättning om 2 år": oms_2 if 'oms_2' in locals() else 0.0,
            "Omsättning om 3 år": oms_3 if 'oms_3' in locals() else 0.0
        }

        # --- Automatisk hämtning från Yahoo för tomma fält ---
        pris, y_valuta = hamta_kurs_och_valuta(ticker)
        if pris is not None and ny_rad["Aktuell kurs"] == 0.0:
            ny_rad["Aktuell kurs"] = round(pris, 2)
        if ny_rad["Valuta"] == "" and y_valuta:
            ny_rad["Valuta"] = y_valuta

        # (Här kan man lägga till extra hämtning för P/S och omsättning nästa år om vi utökar funktionen senare)

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

    riktkurs_val = st.selectbox(
        "Vilken riktkurs ska användas?",
        ["Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år"],
        index=1
    )

    filterval = st.radio("Visa förslag för:", ["Alla bolag", "Endast portföljen"])

    # Beräkna portföljvärde
    df_portfolj = df[df["Antal aktier"] > 0].copy()
    df_portfolj["Värde (SEK)"] = df_portfolj.apply(
        lambda r: r["Antal aktier"] * r["Aktuell kurs"] * valutakurser.get(r["Valuta"], 1), axis=1
    )
    portfoljvarde = df_portfolj["Värde (SEK)"].sum()

    # Filtrera bolag
    if filterval == "Endast portföljen":
        df_forslag = df_portfolj[df_portfolj[riktkurs_val] > df_portfolj["Aktuell kurs"]].copy()
    else:
        df_forslag = df[df[riktkurs_val] > df["Aktuell kurs"]].copy()

    df_forslag["Potential (%)"] = ((df_forslag[riktkurs_val] - df_forslag["Aktuell kurs"]) / df_forslag["Aktuell kurs"]) * 100
    df_forslag = df_forslag.sort_values(by="Potential (%)", ascending=False).reset_index(drop=True)

    if valutakurser.get("USD", 0) == 0:
        st.warning("Valutakurs USD → SEK får inte vara 0.")
        return

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


def hantera_valutakurser(df):
    st.sidebar.subheader("Valutakurser → SEK")
    valutakurser = {}
    for valuta in sorted(df["Valuta"].dropna().unique()):
        # Sätt standardvärden
        if valuta == "USD":
            default = 9.50
        elif valuta == "NOK":
            default = 0.93
        elif valuta == "EUR":
            default = 11.10
        elif valuta == "CAD":
            default = 7.00
        else:
            default = 1.0

        valutakurser[valuta] = st.sidebar.number_input(
            f"{valuta} → SEK",
            value=default,
            step=0.01
        )
    return valutakurser

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

    # Utdelningssummering om kolumn finns
    if "Årlig utdelning" in df.columns:
        df["Utdelning (SEK/år)"] = df.apply(
            lambda r: r["Antal aktier"] * r["Årlig utdelning"] * valutakurser.get(r["Valuta"], 1), axis=1
        )
        total_utdelning = df["Utdelning (SEK/år)"].sum()
        st.markdown(f"**Totalt portföljvärde:** {round(total, 2)} SEK  \n"
                    f"**Förväntad årlig utdelning:** {round(total_utdelning, 2)} SEK  \n"
                    f"**Förväntad genomsnittlig månadsutdelning:** {round(total_utdelning/12, 2)} SEK")
    else:
        st.markdown(f"**Totalt portföljvärde:** {round(total, 2)} SEK")

    st.dataframe(df[["Ticker", "Bolagsnamn", "Antal aktier", "Aktuell kurs", "Värde (SEK)", "Andel (%)"]], use_container_width=True)


def analysvy(df):
    st.subheader("📈 Analysläge")
    df = uppdatera_berakningar(df)
    st.dataframe(df, use_container_width=True)


def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    df = hamta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)

    # Dynamisk valutahantering
    valutakurser = hantera_valutakurser(df)

    meny = st.sidebar.radio("Meny", ["Analys", "Lägg till / uppdatera bolag", "Investeringsförslag", "Portfölj"])

    if meny == "Analys":
        analysvy(df)

    elif meny == "Lägg till / uppdatera bolag":
        df = lagg_till_eller_uppdatera(df)
        df = uppdatera_berakningar(df)
        spara_data(df)

    elif meny == "Investeringsförslag":
        df = uppdatera_berakningar(df)
        visa_investeringsforslag(df, valutakurser)

    elif meny == "Portfölj":
        df = uppdatera_berakningar(df)
        visa_portfolj(df, valutakurser)


if __name__ == "__main__":
    main()
