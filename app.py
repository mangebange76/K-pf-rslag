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
        "Ticker", "Bolagsnamn", "Aktuell kurs", "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "P/S-snitt", "Riktkurs idag", "Riktkurs 2026", "Riktkurs 2027", "Riktkurs 2028", "Antal aktier",
        "Valuta", "Årlig utdelning"
    ]
    for kol in nödvändiga:
        if kol not in df.columns:
            df[kol] = 0.0 if "kurs" in kol.lower() or "omsättning" in kol.lower() or "p/s" in kol.lower() or "utdelning" in kol.lower() else ""
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
        info = yf.Ticker(ticker).info
        return info.get("regularMarketPrice", None), info.get("currency", "USD")
    except Exception:
        return None, "USD"

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

        ps_idag = st.number_input("P/S idag", value=float(befintlig.get("P/S", 0.0)) if not befintlig.empty else 0.0)
        ps1 = st.number_input("P/S Q1", value=float(befintlig.get("P/S Q1", 0.0)) if not befintlig.empty else 0.0)
        ps2 = st.number_input("P/S Q2", value=float(befintlig.get("P/S Q2", 0.0)) if not befintlig.empty else 0.0)
        ps3 = st.number_input("P/S Q3", value=float(befintlig.get("P/S Q3", 0.0)) if not befintlig.empty else 0.0)
        ps4 = st.number_input("P/S Q4", value=float(befintlig.get("P/S Q4", 0.0)) if not befintlig.empty else 0.0)

        oms_idag = st.number_input("Omsättning idag", value=float(befintlig.get("Omsättning idag", 0.0)) if not befintlig.empty else 0.0)
        oms_1 = st.number_input("Omsättning nästa år", value=float(befintlig.get("Omsättning nästa år", 0.0)) if not befintlig.empty else 0.0)
        oms_2 = st.number_input("Omsättning om 2 år", value=float(befintlig.get("Omsättning om 2 år", 0.0)) if not befintlig.empty else 0.0)
        oms_3 = st.number_input("Omsättning om 3 år", value=float(befintlig.get("Omsättning om 3 år", 0.0)) if not befintlig.empty else 0.0)

        valuta = st.selectbox("Valuta (den valuta aktien handlas i)", ["USD", "NOK", "CAD", "SEK", "EUR"],
                              index=0 if befintlig.empty else ["USD","NOK","CAD","SEK","EUR"].index(befintlig.get("Valuta", "USD")))

        utdelning = st.number_input("Årlig utdelning per aktie", value=float(befintlig.get("Årlig utdelning", 0.0)) if not befintlig.empty else 0.0)

        sparaknapp = st.form_submit_button("💾 Spara")

    if sparaknapp and ticker:
        ny_rad = {
            "Ticker": ticker,
            "Bolagsnamn": namn,
            "Aktuell kurs": kurs,
            "Utestående aktier": aktier,
            "Antal aktier": antal_aktier,
            "P/S": ps_idag,
            "P/S Q1": ps1,
            "P/S Q2": ps2,
            "P/S Q3": ps3,
            "P/S Q4": ps4,
            "Omsättning idag": oms_idag,
            "Omsättning nästa år": oms_1,
            "Omsättning om 2 år": oms_2,
            "Omsättning om 3 år": oms_3,
            "Valuta": valuta,
            "Årlig utdelning": utdelning
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
        ["Riktkurs idag", "Riktkurs 2026", "Riktkurs 2027", "Riktkurs 2028"],
        index=1
    )

    filterval = st.radio("Visa förslag för:", ["Alla bolag", "Endast portföljen"])

    df = df.copy()
    df = df[df["Aktuell kurs"] > 0]

    df["Valutakurs"] = df["Valuta"].apply(lambda v: valutakurser.get(v, 1.0))
    df["Kurs i SEK"] = df["Aktuell kurs"] * df["Valutakurs"]
    df["Värde (SEK)"] = df["Antal aktier"] * df["Kurs i SEK"]

    df_portfolj = df[df["Antal aktier"] > 0].copy()
    portfoljvarde = df_portfolj["Värde (SEK)"].sum()

    if filterval == "Endast portföljen":
        df_forslag = df_portfolj[df_portfolj[riktkurs_val] > df_portfolj["Aktuell kurs"]]
    else:
        df_forslag = df[df[riktkurs_val] > df["Aktuell kurs"]]

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
    växelkurs = valutakurser.get(valuta, 1.0)
    aktuell_kurs = rad["Aktuell kurs"]
    riktkurs = rad[riktkurs_val]
    kurs_sek = aktuell_kurs * växelkurs
    riktkurs_sek = riktkurs * växelkurs

    kapital_usd = kapital_sek / växelkurs
    antal = int(kapital_usd // aktuell_kurs)
    investering_sek = antal * kurs_sek

    nuvarande_innehav = df_portfolj[df_portfolj["Ticker"] == rad["Ticker"]]["Värde (SEK)"].sum()
    ny_total = nuvarande_innehav + investering_sek
    nuvarande_andel = round((nuvarande_innehav / portfoljvarde) * 100, 2) if portfoljvarde > 0 else 0
    ny_andel = round((ny_total / portfoljvarde) * 100, 2) if portfoljvarde > 0 else 0

    st.markdown(f"""
        ### 💰 Förslag {index+1} av {len(df_forslag)}
        - **Bolag:** {rad['Bolagsnamn']} ({rad['Ticker']})
        - **Aktuell kurs:** {round(aktuell_kurs, 2)} {valuta} ({round(kurs_sek, 2)} SEK)
        - **{riktkurs_val}:** {round(riktkurs, 2)} {valuta} ({round(riktkurs_sek, 2)} SEK)
        - **Potential:** {round(rad['Potential (%)'], 2)}%
        - **Antal att köpa:** {antal} st
        - **Beräknad investering:** {round(investering_sek, 2)} SEK
        - **Nuvarande andel i portföljen:** {nuvarande_andel}%
        - **Andel efter köp:** {ny_andel}%
    """)

    if st.button("➡️ Nästa förslag"):
        st.session_state.forslags_index += 1


def analysvy(df, valutakurser):
    st.subheader("📈 Analysläge")
    df = uppdatera_berakningar(df)

    if st.button("🔄 Uppdatera alla aktuella kurser från Yahoo"):
        misslyckade = []
        uppdaterade = 0
        total = len(df)
        status = st.empty()
        progress = st.progress(0)

        with st.spinner("Uppdaterar kurser..."):
            for i, row in df.iterrows():
                ticker = str(row["Ticker"]).strip().upper()
                status.text(f"🔄 Uppdaterar {i+1}/{total} ({ticker})")
                try:
                    pris, valuta = hamta_kurs_och_valuta(ticker)
                    if pris is None:
                        misslyckade.append(ticker)
                        continue
                    växelkurs = valutakurser.get(valuta, 1.0)
                    kurs_usd = pris / växelkurs  # Konvertera till USD
                    df.at[i, "Aktuell kurs"] = round(kurs_usd, 2)
                    df.at[i, "Valuta"] = valuta
                    uppdaterade += 1
                except Exception:
                    misslyckade.append(ticker)

                progress.progress((i+1) / total)
                time.sleep(2)

        spara_data(df)
        status.text("✅ Uppdatering klar")
        st.success(f"{uppdaterade} tickers uppdaterade.")
        if misslyckade:
            st.warning("Kunde inte uppdatera:\n" + ", ".join(misslyckade))

    st.dataframe(df, use_container_width=True)

def visa_portfolj(df, valutakurser):
    st.subheader("📦 Min portfölj")
    df = df[df["Antal aktier"] > 0].copy()
    if df.empty:
        st.info("Du äger inga aktier.")
        return

    df["Valutakurs"] = df["Valuta"].apply(lambda v: valutakurser.get(v, 1.0))
    df["Kurs i SEK"] = df["Aktuell kurs"] * df["Valutakurs"]
    df["Värde (SEK)"] = df["Antal aktier"] * df["Kurs i SEK"]
    total_värde = df["Värde (SEK)"].sum()
    df["Andel (%)"] = round(df["Värde (SEK)"] / total_värde * 100, 2)

    if "Årlig utdelning" in df.columns:
        df["Utdelning (SEK)"] = df["Årlig utdelning"] * df["Antal aktier"] * df["Valutakurs"]
        total_utdelning = df["Utdelning (SEK)"].sum()
        månadsutdelning = total_utdelning / 12
        st.markdown(f"**Förväntad årlig utdelning:** {round(total_utdelning,2)} SEK")
        st.markdown(f"**Förväntad genomsnittlig månadsutdelning:** {round(månadsutdelning,2)} SEK")

    st.markdown(f"**Totalt portföljvärde:** {round(total_värde, 2)} SEK")

    st.dataframe(df[[
        "Ticker", "Bolagsnamn", "Antal aktier", "Aktuell kurs", "Valuta", "Kurs i SEK",
        "Värde (SEK)", "Andel (%)"
    ] + (["Årlig utdelning", "Utdelning (SEK)"] if "Årlig utdelning" in df.columns else [])], use_container_width=True)

def main():
    st.title("📊 Aktieanalys och investeringsförslag")
    df = hamta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)

    # 🔁 Valutakurser – manuellt inmatade av användaren
    st.sidebar.markdown("### 💱 Valutakurser till SEK")
    valutakurser = {
        "USD": st.sidebar.number_input("USD", value=1.0),
        "NOK": st.sidebar.number_input("NOK", value=0.93),
        "CAD": st.sidebar.number_input("CAD", value=7.40),
        "SEK": st.sidebar.number_input("SEK", value=1.0),
        "EUR": st.sidebar.number_input("EUR", value=11.20)
    }

    meny = st.sidebar.radio("📌 Meny", [
        "Analys", "Lägg till / uppdatera bolag",
        "Investeringsförslag", "Portfölj"
    ])

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
