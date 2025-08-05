import streamlit as st
import pandas as pd
import numpy as np
import gspread
import yfinance as yf
import time
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

# ---------------------------------------
# GOOGLE SHEETS KONFIG
# ---------------------------------------
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

# ---------------------------------------
# STANDARDVALUTOR
# ---------------------------------------
STANDARD_VALUTAKURSER = {
    "USD": 9.75,
    "NOK": 0.95,
    "CAD": 7.05,
    "SEK": 1.0,
    "EUR": 11.18
}

# ---------------------------------------
# PARSA SIFFROR FRÅN YAHOO (B/M/T)
# ---------------------------------------
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

# ---------------------------------------
# SÄKERSTÄLL KOLUMNER
# ---------------------------------------
def säkerställ_kolumner(df):
    nödvändiga = [
        "Ticker", "Bolagsnamn", "Aktuell kurs", "Valuta", "Utestående aktier",
        "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "P/S-metod",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "P/S-snitt", "Riktkurs idag", "Riktkurs 2026", "Riktkurs 2027", "Riktkurs 2028",
        "Antal aktier", "Årlig utdelning"
    ]
    for kol in nödvändiga:
        if kol not in df.columns:
            if kol in ["P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "Omsättning idag",
                       "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
                       "P/S-snitt", "Riktkurs idag", "Riktkurs 2026", "Riktkurs 2027",
                       "Riktkurs 2028", "Antal aktier", "Årlig utdelning"]:
                df[kol] = 0.0
            else:
                df[kol] = ""
    return df

# ---------------------------------------
# KONVERTERA TYPER
# ---------------------------------------
def konvertera_typer(df):
    kolumner = [
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "P/S-snitt", "Riktkurs idag", "Riktkurs 2026", "Riktkurs 2027", "Riktkurs 2028",
        "Antal aktier", "Årlig utdelning"
    ]
    for kol in kolumner:
        if kol in df.columns:
            df[kol] = pd.to_numeric(df[kol], errors="coerce").fillna(0.0)
    return df

# ---------------------------------------
# BERÄKNA RIKTKURSER OCH LOGGA P/S-SNITT
# ---------------------------------------
def uppdatera_berakningar(df, logga=False):
    ps_logg = []
    for i, rad in df.iterrows():
        ps_varden = [rad["P/S Q1"], rad["P/S Q2"], rad["P/S Q3"], rad["P/S Q4"]]
        ps_filtrerade = [x for x in ps_varden if x > 0]
        ps_snitt = round(np.mean(ps_filtrerade), 2) if ps_filtrerade else 0
        df.at[i, "P/S-snitt"] = ps_snitt

        if logga:
            ps_logg.append(f"{rad['Bolagsnamn']} ({rad['Ticker']}) – P/S-värden: {ps_filtrerade} → Snitt: {ps_snitt}")

        if rad["Utestående aktier"] > 0:
            df.at[i, "Riktkurs idag"] = round((rad["Omsättning idag"] * ps_snitt) / rad["Utestående aktier"], 2)
            df.at[i, "Riktkurs 2026"] = round((rad["Omsättning nästa år"] * ps_snitt) / rad["Utestående aktier"], 2)
            df.at[i, "Riktkurs 2027"] = round((rad["Omsättning om 2 år"] * ps_snitt) / rad["Utestående aktier"], 2)
            df.at[i, "Riktkurs 2028"] = round((rad["Omsättning om 3 år"] * ps_snitt) / rad["Utestående aktier"], 2)

    return df, ps_logg if logga else df

# ---------------------------------------
# FORMULÄR: LÄGG TILL / UPPDATERA BOLAG
# ---------------------------------------
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
        valuta = st.selectbox("Valuta", ["USD", "NOK", "CAD", "SEK", "EUR"],
                              index=0 if befintlig.empty else ["USD", "NOK", "CAD", "SEK", "EUR"].index(befintlig.get("Valuta", "USD")))
        kurs = st.number_input("Aktuell kurs", value=float(befintlig.get("Aktuell kurs", 0.0)) if not befintlig.empty else 0.0)
        aktier = st.number_input("Utestående aktier (miljoner)", value=float(befintlig.get("Utestående aktier", 0.0)) if not befintlig.empty else 0.0)
        antal_aktier = st.number_input("Antal aktier du äger", value=float(befintlig.get("Antal aktier", 0.0)) if not befintlig.empty else 0.0)
        årlig_utdelning = st.number_input("Årlig utdelning per aktie", value=float(befintlig.get("Årlig utdelning", 0.0)) if not befintlig.empty else 0.0)

        ps_idag = st.number_input("P/S idag", value=float(befintlig.get("P/S", 0.0)) if not befintlig.empty else 0.0)
        ps1 = st.number_input("P/S Q1", value=float(befintlig.get("P/S Q1", 0.0)) if not befintlig.empty else 0.0)
        ps2 = st.number_input("P/S Q2", value=float(befintlig.get("P/S Q2", 0.0)) if not befintlig.empty else 0.0)
        ps3 = st.number_input("P/S Q3", value=float(befintlig.get("P/S Q3", 0.0)) if not befintlig.empty else 0.0)
        ps4 = st.number_input("P/S Q4", value=float(befintlig.get("P/S Q4", 0.0)) if not befintlig.empty else 0.0)

        oms_idag = st.number_input("Omsättning idag (miljoner)", value=float(befintlig.get("Omsättning idag", 0.0)) if not befintlig.empty else 0.0)
        oms_1 = st.number_input("Omsättning nästa år", value=float(befintlig.get("Omsättning nästa år", 0.0)) if not befintlig.empty else 0.0)
        oms_2 = st.number_input("Omsättning om 2 år", value=float(befintlig.get("Omsättning om 2 år", 0.0)) if not befintlig.empty else 0.0)
        oms_3 = st.number_input("Omsättning om 3 år", value=float(befintlig.get("Omsättning om 3 år", 0.0)) if not befintlig.empty else 0.0)

        sparaknapp = st.form_submit_button("💾 Spara")

    if sparaknapp and ticker:
        ny_rad = {
            "Ticker": ticker, "Bolagsnamn": namn, "Valuta": valuta,
            "Aktuell kurs": kurs, "Utestående aktier": aktier, "Antal aktier": antal_aktier,
            "Årlig utdelning": årlig_utdelning,
            "P/S": ps_idag, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms_1, "Omsättning om 2 år": oms_2, "Omsättning om 3 år": oms_3
        }
        if ticker in df["Ticker"].values:
            df.loc[df["Ticker"] == ticker, ny_rad.keys()] = ny_rad.values()
            st.success(f"{ticker} uppdaterat.")
        else:
            df = pd.concat([df, pd.DataFrame([ny_rad])], ignore_index=True)
            st.success(f"{ticker} tillagt.")
    return df

def analysvy(df):
    st.subheader("📈 Analysläge")

    if st.button("🔄 Uppdatera alla aktuella kurser och P/S från Yahoo"):
        misslyckade = []
        uppdaterade = 0
        total = len(df)
        status = st.empty()
        bar = st.progress(0)
        ps_logg_lista = []

        with st.spinner("Uppdaterar data från Yahoo Finance..."):
            for i, row in df.iterrows():
                ticker = str(row["Ticker"]).strip().upper()
                if not ticker:
                    continue

                status.text(f"🔄 ({i+1}/{total}) Uppdaterar {ticker}...")

                try:
                    pris, valuta, ps_idag, ps_hist = hamta_ps_och_kurs(ticker)

                    # Kurs och valuta
                    if pris:
                        df.at[i, "Aktuell kurs"] = round(pris, 2)
                        df.at[i, "Valuta"] = valuta

                    # P/S idag
                    if ps_idag and ps_idag > 0:
                        df.at[i, "P/S"] = round(ps_idag, 2)
                        df.at[i, "P/S-metod"] = "Yahoo"
                    else:
                        df.at[i, "P/S-metod"] = "Saknas"

                    # P/S kvartal
                    if ps_hist and len(ps_hist) >= 4:
                        df.at[i, "P/S Q1"] = round(ps_hist[0], 2)
                        df.at[i, "P/S Q2"] = round(ps_hist[1], 2)
                        df.at[i, "P/S Q3"] = round(ps_hist[2], 2)
                        df.at[i, "P/S Q4"] = round(ps_hist[3], 2)

                    uppdaterade += 1

                except Exception:
                    misslyckade.append(ticker)

                bar.progress((i+1)/total)
                time.sleep(1)  # paus för att undvika blockering

            # Efter att alla rader är uppdaterade, beräkna riktkurser och logga P/S
            df, ps_logg_lista = uppdatera_berakningar(df, logga=True)

        spara_data(df)
        st.success(f"{uppdaterade} tickers uppdaterade.")

        if misslyckade:
            st.warning("Kunde inte uppdatera följande tickers:\n" + ", ".join(misslyckade))

        # Visa P/S-logg i expander
        with st.expander("📄 Detaljerad P/S-logg"):
            for rad in ps_logg_lista:
                st.write(rad)

    st.dataframe(df, use_container_width=True)

def visa_investeringsforslag(df, valutakurser):
    st.subheader("💡 Investeringsförslag")
    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=500.0, step=500.0)
    riktkurs_val = st.selectbox("Vilken riktkurs ska användas?", ["Riktkurs idag", "Riktkurs 2026", "Riktkurs 2027", "Riktkurs 2028"], index=1)
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


def visa_portfolj(df, valutakurser):
    st.subheader("📦 Min portfölj")
    df = df[df["Antal aktier"] > 0].copy()
    if df.empty:
        st.info("Du äger inga aktier.")
        return
    df["Värde (SEK)"] = df["Antal aktier"] * df["Aktuell kurs"] * df["Valuta"].map(valutakurser).fillna(1.0)
    df["Andel (%)"] = round(df["Värde (SEK)"] / df["Värde (SEK)"].sum() * 100, 2)
    total = df["Värde (SEK)"].sum()
    total_utdelning = (df["Antal aktier"] * df["Årlig utdelning"] * df["Valuta"].map(valutakurser).fillna(1.0)).sum()
    st.markdown(f"**Totalt portföljvärde:** {round(total, 2)} SEK")
    st.markdown(f"**Förväntad årlig utdelning:** {round(total_utdelning, 2)} SEK")
    st.markdown(f"**Förväntad månadsutdelning (snitt):** {round(total_utdelning/12, 2)} SEK")
    st.dataframe(df[["Ticker", "Bolagsnamn", "Antal aktier", "Aktuell kurs", "Värde (SEK)", "Andel (%)", "Årlig utdelning"]], use_container_width=True)


def main():
    st.title("📊 Aktieanalys och investeringsförslag")
    df = hamta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)

    valutakurser = {}
    st.sidebar.subheader("💱 Valutakurser (till SEK)")
    for valuta in ["USD", "NOK", "CAD", "SEK", "EUR"]:
        valutakurser[valuta] = st.sidebar.number_input(f"{valuta} → SEK", value=float(STANDARD_VALUTAKURSER[valuta]), step=0.01)

    meny = st.sidebar.radio("Meny", ["Analys", "Lägg till / uppdatera bolag", "Investeringsförslag", "Portfölj"])

    if meny == "Analys":
        analysvy(df)
    elif meny == "Lägg till / uppdatera bolag":
        df = lagg_till_eller_uppdatera(df)
        spara_data(df)
    elif meny == "Investeringsförslag":
        df, _ = uppdatera_berakningar(df, logga=False)
        visa_investeringsforslag(df, valutakurser)
    elif meny == "Portfölj":
        df, _ = uppdatera_berakningar(df, logga=False)
        visa_portfolj(df, valutakurser)


if __name__ == "__main__":
    main()
