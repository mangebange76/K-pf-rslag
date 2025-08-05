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
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år", "Omsättning om 4 år",
        "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Aktuell kurs", "Antal aktier", "Årlig utdelning"
    ]
    for kol in kolumner:
        if kol in df.columns:
            df[kol] = pd.to_numeric(df[kol], errors="coerce").fillna(0.0)
    return df

def säkerställ_kolumner(df):
    nödvändiga = [
        "Ticker", "Bolagsnamn", "Aktuell kurs", "Utestående aktier",
        "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år",
        "Omsättning om 2 år", "Omsättning om 3 år", "Omsättning om 4 år",
        "P/S-snitt", "Riktkurs idag", "Riktkurs 2026", "Riktkurs 2027", "Riktkurs 2028",
        "Antal aktier", "Valuta", "Årlig utdelning"
    ]
    for kol in nödvändiga:
        if kol not in df.columns:
            if ("kurs" in kol.lower() or "omsättning" in kol.lower() or
                "p/s" in kol.lower() or "utdelning" in kol.lower()):
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
            df.at[i, "Riktkurs 2026"] = round((rad["Omsättning nästa år"] * ps_snitt) / rad["Utestående aktier"], 2)
            df.at[i, "Riktkurs 2027"] = round((rad["Omsättning om 2 år"] * ps_snitt) / rad["Utestående aktier"], 2)
            df.at[i, "Riktkurs 2028"] = round((rad["Omsättning om 3 år"] * ps_snitt) / rad["Utestående aktier"], 2)
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
                              index=0 if befintlig.empty else ["USD", "NOK", "CAD", "SEK", "EUR"].index(befintlig.get("Valuta", "USD")))
        utdelning = st.number_input("Årlig utdelning per aktie", value=float(befintlig.get("Årlig utdelning", 0.0)) if not befintlig.empty else 0.0)

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
            "Ticker": ticker, "Bolagsnamn": namn, "Aktuell kurs": kurs, "Utestående aktier": aktier,
            "Antal aktier": antal_aktier, "Valuta": valuta, "Årlig utdelning": utdelning,
            "P/S": ps_idag, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms_1, "Omsättning om 2 år": oms_2,
            "Omsättning om 3 år": oms_3, "Omsättning om 4 år": oms_4
        }

        if ticker in df["Ticker"].values:
            df.loc[df["Ticker"] == ticker, ny_rad.keys()] = ny_rad.values()
            st.success(f"{ticker} uppdaterat.")
        else:
            df = pd.concat([df, pd.DataFrame([ny_rad])], ignore_index=True)
            st.success(f"{ticker} tillagt.")

    return df


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
        df[[
            "Ticker", "Bolagsnamn", "Antal aktier", "Aktuell kurs", "Valuta",
            "Värde (SEK)", "Andel (%)", "Årlig utdelning", "Total årlig utdelning"
        ]],
        use_container_width=True
    )

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


def analysvy(df, valutakurser):
    st.subheader("📈 Analysläge")

    if st.button("🔄 Uppdatera alla aktuella kurser och fundamenta från Yahoo"):
        misslyckade, uppdaterade = [], 0
        ps_yahoo, ps_manuell, ps_saknas = [], [], []
        total = len(df)
        status = st.empty()
        bar = st.progress(0)

        with st.spinner("Uppdaterar data..."):
            for i, row in df.iterrows():
                ticker = str(row["Ticker"]).strip().upper()
                status.text(f"🔄 Uppdaterar {i+1} av {total} ({ticker})...")

                try:
                    yticker = yf.Ticker(ticker)

                    # 1️⃣ Kurs och valuta
                    info = yticker.info
                    pris = info.get("regularMarketPrice", None)
                    valuta = info.get("currency", "USD")
                    if pris:
                        df.at[i, "Aktuell kurs"] = round(pris, 2)
                        df.at[i, "Valuta"] = valuta

                    # 2️⃣ P/S idag
                    ps_ttm = info.get("priceToSalesTrailing12Months", None)
                    if ps_ttm is not None:
                        df.at[i, "P/S"] = round(ps_ttm, 2)
                        ps_yahoo.append(ticker)
                    else:
                        marketcap = info.get("marketCap", None)
                        if marketcap and "Total Revenue" in yticker.financials.index:
                            revenues = yticker.financials.loc["Total Revenue"].dropna().tolist()
                            if revenues:
                                senaste_oms = revenues[0]
                                if senaste_oms > 0:
                                    ps_ttm = marketcap / senaste_oms
                                    df.at[i, "P/S"] = round(ps_ttm, 2)
                                    ps_manuell.append(ticker)
                        if ps_ttm is None:
                            ps_saknas.append(ticker)

                    # 3️⃣ P/S Q1–Q4 (endast om data finns)
                    aktier_utest = row.get("Utestående aktier", 0)
                    if aktier_utest > 0 and not yticker.quarterly_financials.empty:
                        if "Total Revenue" in yticker.quarterly_financials.index:
                            kursdata = yticker.history(period="1y", interval="3mo")["Close"].dropna().tolist()
                            kvartalsoms = yticker.quarterly_financials.loc["Total Revenue"].dropna().tolist()
                            antal_kvartal = min(len(kursdata), len(kvartalsoms))
                            for idx in range(min(4, antal_kvartal)):
                                kurs_q = kursdata[idx]
                                oms_q = kvartalsoms[idx]
                                if kurs_q and oms_q and oms_q > 0:
                                    marketcap_q = kurs_q * aktier_utest
                                    ps_varde = round(marketcap_q / oms_q, 2)
                                    df.at[i, f"P/S Q{idx+1}"] = ps_varde

                    # 4️⃣ Omsättning nästa år och om 2 år
                    oms1, oms2 = None, None
                    try:
                        analysis = yticker.analysis
                        if "Revenue Estimate" in analysis.index:
                            revenue_estimates = analysis.loc["Revenue Estimate"]
                            oms1 = revenue_estimates.iloc[0, 1] if pd.notna(revenue_estimates.iloc[0, 1]) else None
                            oms2 = revenue_estimates.iloc[1, 1] if pd.notna(revenue_estimates.iloc[1, 1]) else None
                            if oms1:
                                df.at[i, "Omsättning nästa år"] = oms1
                            if oms2:
                                df.at[i, "Omsättning om 2 år"] = oms2
                    except Exception:
                        pass

                    # 5️⃣ Historisk omsättning för CAGR
                    cagr = None
                    try:
                        if "Total Revenue" in yticker.financials.index:
                            revenues = yticker.financials.loc["Total Revenue"].dropna().tolist()
                            revenues = revenues[::-1]
                            if len(revenues) >= 2:
                                start_val = revenues[0]
                                end_val = revenues[-1]
                                years = len(revenues) - 1
                                if start_val > 0 and years > 0:
                                    cagr = (end_val / start_val) ** (1 / years) - 1
                    except Exception:
                        pass

                    # 6️⃣ Omsättning om 3 och 4 år
                    if oms1 and oms2:
                        growth_rate = cagr if cagr is not None else (oms2 - oms1) / oms1
                        if growth_rate < 0:
                            growth_rate = 0.02
                        oms3 = oms2 * (1 + growth_rate)
                        oms4 = oms3 * (1 + growth_rate)
                        df.at[i, "Omsättning om 3 år"] = oms3
                        if "Omsättning om 4 år" in df.columns:
                            df.at[i, "Omsättning om 4 år"] = oms4

                    uppdaterade += 1

                except Exception:
                    misslyckade.append(ticker)

                bar.progress((i + 1) / total)
                time.sleep(1)

        # ✅ Spara alla ändringar direkt
        spara_data(df)

        status.text("✅ Uppdatering klar.")
        st.success(f"{uppdaterade} tickers uppdaterade.")
        if misslyckade:
            st.warning("Kunde inte uppdatera följande tickers:\n" + ", ".join(misslyckade))
        if ps_yahoo:
            st.info(f"P/S hämtades direkt från Yahoo för: {', '.join(ps_yahoo)}")
        if ps_manuell:
            st.info(f"P/S beräknades manuellt för: {', '.join(ps_manuell)}")
        if ps_saknas:
            st.warning(f"Inget P/S kunde sättas för: {', '.join(ps_saknas)}")

    st.dataframe(df, use_container_width=True)

def main():
    st.title("📊 Aktieanalys och investeringsförslag")
    df = hamta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)

    valutakurser = {}
    st.sidebar.subheader("💱 Valutakurser")
    for valuta in ["USD", "NOK", "CAD", "SEK", "EUR"]:
        valutakurser[valuta] = st.sidebar.number_input(
            f"{valuta} → SEK",
            value=float(STANDARD_VALUTAKURSER[valuta]),
            step=0.01
        )

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
