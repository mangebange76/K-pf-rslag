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
    data = skapa_koppling().get_all_records()
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
        "Ticker", "Bolagsnamn", "Aktuell kurs", "Valuta", "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "P/S-snitt", "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år", "Antal aktier", "Årlig utdelning"
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

    # Visa bolag i listan som "Namn (Ticker)"
    namn_map = {f"{rad['Bolagsnamn']} ({rad['Ticker']})": rad['Ticker'] for _, rad in df.iterrows()}
    valt = st.selectbox("Välj bolag att uppdatera (eller lämna tom för nytt)", [""] + sorted(namn_map.keys()))

    if valt:
        ticker_vald = namn_map[valt]
        befintlig = df[df["Ticker"] == ticker_vald].iloc[0]
    else:
        befintlig = pd.Series(dtype=object)

    with st.form("form"):
        ticker = st.text_input("Ticker", value=befintlig.get("Ticker", "") if not befintlig.empty else "").upper()
        utest_aktier = st.number_input(
            "Utestående aktier (miljoner)",
            value=float(befintlig.get("Utestående aktier", 0.0)) if not befintlig.empty else 0.0
        )
        oms_idag = st.number_input(
            "Omsättning idag (miljoner, i bolagets valuta)",
            value=float(befintlig.get("Omsättning idag", 0.0)) if not befintlig.empty else 0.0
        )

        antal_aktier = st.number_input(
            "Antal aktier du äger",
            value=float(befintlig.get("Antal aktier", 0.0)) if not befintlig.empty else 0.0
        )

        sparaknapp = st.form_submit_button("💾 Spara och hämta från Yahoo")

    if sparaknapp and ticker:
        uppdaterade_falt = {}

        # Börja med de manuella fälten
        ny_rad = {
            "Ticker": ticker,
            "Utestående aktier": utest_aktier,
            "Omsättning idag": oms_idag,
            "Antal aktier": antal_aktier
        }

        # --- Hämtning från Yahoo ---
        try:
            yf_data = yf.Ticker(ticker)

            # Kurs och valuta
            info = yf_data.info
            ny_rad["Aktuell kurs"] = round(info.get("regularMarketPrice", 0.0), 2)
            ny_rad["Valuta"] = info.get("currency", "USD")
            uppdaterade_falt["Aktuell kurs"] = ny_rad["Aktuell kurs"]
            uppdaterade_falt["Valuta"] = ny_rad["Valuta"]

            # P/S och Market Cap per kvartal (om tillgängligt)
            for i, kol in enumerate(["P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"]):
                ps_value = None
                try:
                    ps_value = info.get("priceToSalesTrailing12Months", None) if i == 0 else None
                except:
                    pass
                ny_rad[kol] = round(ps_value, 2) if ps_value else 0.0
                uppdaterade_falt[kol] = ny_rad[kol]

            # Omsättning nästa år (från earnings estimate)
            try:
                cal = yf_data.get_financials(freq="yearly")
                if "Total Revenue" in cal.index:
                    oms_next_year = cal.loc["Total Revenue"].iloc[0] / 1_000_000
                    ny_rad["Omsättning nästa år"] = round(oms_next_year, 2)
                    uppdaterade_falt["Omsättning nästa år"] = ny_rad["Omsättning nästa år"]
            except:
                ny_rad["Omsättning nästa år"] = 0.0

            # Beräkna omsättning om 2 och 3 år
            if ny_rad["Omsättning idag"] > 0 and ny_rad["Omsättning nästa år"] > 0:
                tillvaxt = (ny_rad["Omsättning nästa år"] / ny_rad["Omsättning idag"]) - 1
                if tillvaxt < 0:
                    tillvaxt = 0.02  # inflation
                tillvaxt = min(tillvaxt, 0.5)  # tak 50%
                ny_rad["Omsättning om 2 år"] = round(ny_rad["Omsättning nästa år"] * (1 + tillvaxt), 2)
                ny_rad["Omsättning om 3 år"] = round(ny_rad["Omsättning om 2 år"] * (1 + tillvaxt), 2)
                uppdaterade_falt["Omsättning om 2 år"] = ny_rad["Omsättning om 2 år"]
                uppdaterade_falt["Omsättning om 3 år"] = ny_rad["Omsättning om 3 år"]

        except Exception as e:
            st.error(f"Kunde inte hämta data från Yahoo Finance för {ticker}: {e}")

        # --- Spara i DataFrame ---
        if ticker in df["Ticker"].values:
            for kol, värde in ny_rad.items():
                df.loc[df["Ticker"] == ticker, kol] = värde
            st.success(f"{ticker} uppdaterat från Yahoo Finance.")
        else:
            # Lägg till nytt bolag
            for kol in säkerställ_kolumner(pd.DataFrame()).columns:
                if kol not in ny_rad:
                    ny_rad[kol] = 0.0 if "kurs" in kol.lower() or "omsättning" in kol.lower() or "p/s" in kol.lower() else ""
            df = pd.concat([df, pd.DataFrame([ny_rad])], ignore_index=True)
            st.success(f"{ticker} tillagt och uppdaterat från Yahoo Finance.")

        # Visa vilka fält som uppdaterats
        if uppdaterade_falt:
            st.info("**Följande fält uppdaterades:**\n" + "\n".join([f"- {k}: {v}" for k, v in uppdaterade_falt.items()]))

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

    st.dataframe(
        df[["Ticker", "Bolagsnamn", "Antal aktier", "Aktuell kurs", "Värde (SEK)", "Andel (%)"]],
        use_container_width=True
    )


def analysvy(df):
    st.subheader("📈 Analysläge")
    df = uppdatera_berakningar(df)
    st.dataframe(df, use_container_width=True)


def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    df = hamta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)

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
