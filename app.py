import streamlit as st
import pandas as pd
import numpy as np
import gspread
import yfinance as yf
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
    return client.open_by_url(SHEET_URL).worksheet(SHEET_NAME)

def hamta_data():
    data = skapa_koppling().get_all_records()
    return pd.DataFrame(data)

def spara_data(df):
    sheet = skapa_koppling()
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.astype(str).values.tolist())

# ---------------------------------------
# DATARAM-HANTERING
# ---------------------------------------
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
        "Ticker", "Bolagsnamn", "Aktuell kurs", "Valuta", "Årlig utdelning",
        "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "P/S-snitt", "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år", "Antal aktier"
    ]
    for kol in nödvändiga:
        if kol not in df.columns:
            if kol in ["Ticker", "Bolagsnamn", "Valuta"]:
                df[kol] = ""
            elif "Riktkurs" in kol or "Omsättning" in kol or "P/S" in kol or kol in ["Aktuell kurs", "Årlig utdelning", "Utestående aktier", "Antal aktier"]:
                df[kol] = 0.0
    return df

# ---------------------------------------
# HÄMTA DATA FRÅN YAHOO
# ---------------------------------------
def hamta_kurs_valuta_ps_oms(ticker):
    """Hämtar aktuell kurs, valuta, P/S-värden och omsättning."""
    data = {}
    try:
        info = yf.Ticker(ticker).info
        data["Aktuell kurs"] = info.get("currentPrice")
        data["Valuta"] = info.get("currency")

        # Hämta P/S nuvarande och bakåt (om tillgängligt)
        if "priceToSalesTrailing12Months" in info:
            data["P/S"] = info["priceToSalesTrailing12Months"]

        # Omsättning idag och nästa år från 'financialData' eller 'earnings'
        if "totalRevenue" in info and info["totalRevenue"] is not None:
            data["Omsättning idag"] = info["totalRevenue"] / 1_000_000  # miljoner

    except Exception:
        pass

    return data

# ---------------------------------------
# BERÄKNINGAR
# ---------------------------------------
def uppdatera_berakningar(df):
    for i, rad in df.iterrows():
        ps = [rad["P/S Q1"], rad["P/S Q2"], rad["P/S Q3"], rad["P/S Q4"]]
        ps = [x for x in ps if x > 0 and x <= 100]  # mjukt tak på 100
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
        aktier = st.number_input("Utestående aktier (miljoner)", value=float(befintlig.get("Utestående aktier", 0.0)) if not befintlig.empty else 0.0)
        antal_aktier = st.number_input("Antal aktier du äger", value=float(befintlig.get("Antal aktier", 0.0)) if not befintlig.empty else 0.0)
        valuta = st.selectbox("Valuta", ["USD", "NOK", "CAD", "SEK", "EUR"],
                               index=0 if befintlig.get("Valuta", "") == "" else ["USD", "NOK", "CAD", "SEK", "EUR"].index(befintlig.get("Valuta", "USD")))
        arlig_utdelning = st.number_input("Årlig utdelning per aktie", value=float(befintlig.get("Årlig utdelning", 0.0)) if not befintlig.empty else 0.0)

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
            "Ticker": ticker, "Bolagsnamn": namn, "Aktuell kurs": kurs, "Utestående aktier": aktier,
            "Antal aktier": antal_aktier, "Valuta": valuta, "Årlig utdelning": arlig_utdelning,
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

        # ➡ Hämta data från Yahoo och uppdatera automatiskt
        yahoo_data = hamta_kurs_valuta_ps_oms(ticker)
        uppdaterade_falt = []
        misslyckade_falt = []

        for kol, val in yahoo_data.items():
            if val is not None:
                df.loc[df["Ticker"] == ticker, kol] = val
                uppdaterade_falt.append(kol)
            else:
                misslyckade_falt.append(kol)

        if uppdaterade_falt:
            st.info(f"Från Yahoo uppdaterades: {', '.join(uppdaterade_falt)}")

        if misslyckade_falt:
            miss_str = f"{ticker}: {', '.join(misslyckade_falt)}"
            st.text_area("Misslyckade uppdateringar", miss_str, height=100)
            st.download_button(
                label="📋 Kopiera/Exportera lista",
                data=miss_str,
                file_name=f"misslyckade_{ticker}.txt",
                mime="text/plain"
            )

        # ➡ Uppdatera omsättning år 2 & år 3 direkt vid sparning
        df = räkna_omsättning_cagr(df)
        spara_data(df)

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
        lambda x: x["Antal aktier"] * x["Aktuell kurs"] * valutakurser.get(x["Valuta"], 1),
        axis=1
    )
    portfoljvarde = df_portfolj["Värde (SEK)"].sum()

    if filterval == "Endast portföljen":
        df_forslag = df_portfolj[df_portfolj[riktkurs_val] > df_portfolj["Aktuell kurs"]].copy()
    else:
        df_forslag = df[df[riktkurs_val] > df["Aktuell kurs"]].copy()

    df_forslag["Potential (%)"] = ((df_forslag[riktkurs_val] - df_forslag["Aktuell kurs"]) / df_forslag["Aktuell kurs"]) * 100
    df_forslag = df_forslag.sort_values(by="Potential (%)", ascending=False).reset_index(drop=True)

    if not df_forslag.empty:
        if 'forslags_index' not in st.session_state:
            st.session_state.forslags_index = 0

        index = st.session_state.forslags_index
        if index < len(df_forslag):
            rad = df_forslag.iloc[index]
            bolags_valuta = rad["Valuta"]
            valutakurs = valutakurser.get(bolags_valuta, 1)
            antal = int((kapital_sek / valutakurs) // rad["Aktuell kurs"])
            investering_sek = antal * rad["Aktuell kurs"] * valutakurs

            nuvarande_innehav = df_portfolj[df_portfolj["Ticker"] == rad["Ticker"]]["Värde (SEK)"].sum()
            ny_total = nuvarande_innehav + investering_sek
            nuvarande_andel = round((nuvarande_innehav / portfoljvarde) * 100, 2) if portfoljvarde > 0 else 0
            ny_andel = round((ny_total / portfoljvarde) * 100, 2) if portfoljvarde > 0 else 0

            st.markdown(f"""
                ### 💰 Förslag {index+1} av {len(df_forslag)}
                - **Bolag:** {rad['Bolagsnamn']} ({rad['Ticker']})
                - **Aktuell kurs:** {round(rad['Aktuell kurs'], 2)} {bolags_valuta}
                - **{riktkurs_val}:** {round(rad[riktkurs_val], 2)} {bolags_valuta}
                - **Potential:** {round(rad['Potential (%)'], 2)}%
                - **Antal att köpa:** {antal} st
                - **Beräknad investering:** {round(investering_sek, 2)} SEK
                - **Nuvarande andel i portföljen:** {nuvarande_andel}%
                - **Andel efter köp:** {ny_andel}%
            """)

            if st.button("➡️ Nästa förslag"):
                st.session_state.forslags_index += 1
        else:
            st.info("Inga fler förslag att visa.")
    else:
        st.info("Inga bolag matchar kriterierna just nu.")

def visa_portfolj(df, valutakurser):
    st.subheader("📦 Min portfölj")
    df = df[df["Antal aktier"] > 0].copy()
    if df.empty:
        st.info("Du äger inga aktier.")
        return
    df["Värde (SEK)"] = df.apply(
        lambda x: x["Antal aktier"] * x["Aktuell kurs"] * valutakurser.get(x["Valuta"], 1),
        axis=1
    )
    df["Andel (%)"] = round(df["Värde (SEK)"] / df["Värde (SEK)"].sum() * 100, 2)
    total = df["Värde (SEK)"].sum()

    df["Årlig utdelning (SEK)"] = df.apply(
        lambda x: x["Årlig utdelning"] * x["Antal aktier"] * valutakurser.get(x["Valuta"], 1),
        axis=1
    )
    total_utdelning = df["Årlig utdelning (SEK)"].sum()
    manadsutdelning = total_utdelning / 12

    st.markdown(f"**Totalt portföljvärde:** {round(total, 2)} SEK")
    st.markdown(f"**Förväntad årlig utdelning:** {round(total_utdelning, 2)} SEK")
    st.markdown(f"**Förväntad genomsnittlig månadsutdelning:** {round(manadsutdelning, 2)} SEK")

    st.dataframe(df[[
        "Ticker", "Bolagsnamn", "Antal aktier", "Aktuell kurs", "Valuta",
        "Värde (SEK)", "Andel (%)", "Årlig utdelning (SEK)"
    ]], use_container_width=True)

def analysvy(df, valutakurser):
    st.subheader("📈 Analysläge")
    df = uppdatera_berakningar(df)

    if st.button("🔄 Uppdatera alla aktuella kurser från Yahoo"):
        misslyckade = {}
        uppdaterade = 0
        total = len(df)
        status = st.empty()

        with st.spinner("Uppdaterar kurser..."):
            for i, row in df.iterrows():
                ticker = str(row["Ticker"]).strip().upper()
                status.text(f"🔄 Uppdaterar {i+1} av {total} tickers... ({ticker})")

                try:
                    yahoo_data = hamta_kurs_valuta_ps_oms(ticker)
                    for kol, val in yahoo_data.items():
                        if val is not None:
                            df.at[i, kol] = val
                        else:
                            misslyckade.setdefault(ticker, []).append(kol)
                    uppdaterade += 1
                except Exception:
                    misslyckade.setdefault(ticker, []).append("Kunde inte uppdateras alls")

                time.sleep(2)

        df = räkna_omsättning_cagr(df)
        spara_data(df)
        status.text("✅ Uppdatering slutförd.")
        st.success(f"{uppdaterade} tickers uppdaterade.")

        if misslyckade:
            miss_str = "\n".join([f"{t}: {', '.join(fel)}" for t, fel in misslyckade.items()])
            st.text_area("Misslyckade uppdateringar", miss_str, height=150)
            st.download_button(
                label="📋 Kopiera/Exportera lista",
                data=miss_str,
                file_name="misslyckade_lista.txt",
                mime="text/plain"
            )

    st.dataframe(df, use_container_width=True)

def räkna_omsättning_cagr(df):
    for i, rad in df.iterrows():
        if rad["Omsättning nästa år"] > 0 and rad["Omsättning om 2 år"] == 0:
            try:
                hist = yf.Ticker(rad["Ticker"]).history(period="5y", interval="1y")
                oms_hist = hist["Close"].tolist()
                if len(oms_hist) >= 2:
                    start_val = oms_hist[0]
                    slut_val = oms_hist[-1]
                    cagr = ((slut_val / start_val) ** (1 / (len(oms_hist) - 1))) - 1
                    cagr = max(min(cagr, 0.50), -0.02)

                    år2 = rad["Omsättning nästa år"] * (1 + cagr)
                    år3 = år2 * (1 + cagr)
                    df.at[i, "Omsättning om 2 år"] = round(år2, 2)
                    df.at[i, "Omsättning om 3 år"] = round(år3, 2)
            except Exception:
                pass
    return df

def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    df = hamta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)

    valutakurser = {
        "USD": st.sidebar.number_input("USD → SEK", value=9.50, step=0.01),
        "NOK": st.sidebar.number_input("NOK → SEK", value=0.93, step=0.01),
        "CAD": st.sidebar.number_input("CAD → SEK", value=7.00, step=0.01),
        "SEK": 1.0,
        "EUR": st.sidebar.number_input("EUR → SEK", value=11.10, step=0.01)
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
