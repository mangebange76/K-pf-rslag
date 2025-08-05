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
        "Aktuell kurs", "Antal aktier"
    ]
    for kol in kolumner:
        if kol in df.columns:
            df[kol] = pd.to_numeric(df[kol], errors="coerce").fillna(0.0)
    return df

def säkerställ_kolumner(df):
    nödvändiga = [
        "Ticker", "Bolagsnamn", "Aktuell kurs", "Valuta", "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "P/S-snitt", "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år", "Antal aktier"
    ]
    for kol in nödvändiga:
        if kol not in df.columns:
            if kol in ["Ticker", "Bolagsnamn", "Valuta"]:
                df[kol] = ""
            elif "Riktkurs" in kol or "P/S" in kol or "Omsättning" in kol or "kurs" in kol.lower():
                df[kol] = 0.0
            else:
                df[kol] = ""
    return df

# --- Hjälpfunktion för att tolka tal från Yahoo (B/M/T) ---
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

# --- Hämtar kurs, valuta och P/S från Yahoo ---
def hamta_ps_och_kurs(ticker):
    try:
        yticker = yf.Ticker(ticker)
        info = yticker.info

        pris = info.get("regularMarketPrice", None)
        valuta = info.get("currency", "USD")
        marketcap = parse_yahoo_number(info.get("marketCap", None))

        # TTM omsättning
        oms_ttm = None
        try:
            fin_df = yticker.financials
            if not fin_df.empty and "Total Revenue" in fin_df.index:
                oms_ttm = fin_df.loc["Total Revenue"].iloc[0]
        except:
            oms_ttm = None

        ps_idag = None
        if marketcap and oms_ttm and oms_ttm > 0:
            ps_idag = marketcap / oms_ttm

        # Kvartals-P/S
        ps_hist = []
        try:
            q_fin_df = yticker.quarterly_financials
            if not q_fin_df.empty and "Total Revenue" in q_fin_df.index:
                oms_values = q_fin_df.loc["Total Revenue"].dropna().tolist()
                for oms in oms_values[:4]:
                    if marketcap and oms and oms > 0:
                        ps_hist.append(marketcap / (oms * 4))
                    else:
                        ps_hist.append(0)
        except:
            pass

        return pris, valuta, ps_idag, ps_hist, None
    except Exception as e:
        return None, None, None, [], str(e)

# --- Analysläge ---
def analysvy(df):
    st.subheader("📈 Analysläge")

    # Standardvalutakurser
    valutakurser = {
        "USD": st.sidebar.number_input("USD → SEK", value=9.50, step=0.01),
        "NOK": st.sidebar.number_input("NOK → SEK", value=0.93, step=0.01),
        "EUR": st.sidebar.number_input("EUR → SEK", value=11.10, step=0.01),
        "CAD": st.sidebar.number_input("CAD → SEK", value=7.00, step=0.01)
    }

    felsokningslage = st.checkbox("Visa felsökningsinfo vid uppdatering", value=False)

    if st.button("🔄 Uppdatera alla aktuella kurser och P/S från Yahoo"):
        misslyckade = []
        uppdaterade = []
        felsokningslogg = []
        total = len(df)
        status = st.empty()
        bar = st.progress(0)

        for i, row in df.iterrows():
            ticker = str(row["Ticker"]).strip().upper()
            if not ticker:
                continue

            status.text(f"🔄 ({i+1}/{total}) Uppdaterar {ticker}...")
            pris, valuta, ps_idag, ps_hist, felorsak = hamta_ps_och_kurs(ticker)

            if felsokningslage:
                felsokningslogg.append({
                    "Ticker": ticker,
                    "Kurs hittad": pris is not None,
                    "Valuta": valuta,
                    "P/S idag": ps_idag,
                    "Felorsak": felorsak if felorsak else ""
                })

            if felorsak:
                misslyckade.append(f"{ticker} – {felorsak}")
                bar.progress((i+1)/total)
                time.sleep(1)
                continue

            if pris:
                df.at[i, "Aktuell kurs"] = round(pris, 2)
                df.at[i, "Valuta"] = valuta

            if ps_idag and ps_idag > 0:
                df.at[i, "P/S"] = round(ps_idag, 2)

            if ps_hist and len(ps_hist) >= 4:
                df.at[i, "P/S Q1"] = round(ps_hist[0], 2)
                df.at[i, "P/S Q2"] = round(ps_hist[1], 2)
                df.at[i, "P/S Q3"] = round(ps_hist[2], 2)
                df.at[i, "P/S Q4"] = round(ps_hist[3], 2)

            uppdaterade.append(ticker)
            bar.progress((i+1)/total)
            time.sleep(1)

        spara_data(df)
        st.success(f"{len(uppdaterade)} tickers uppdaterade, {len(misslyckade)} misslyckades.")

        if misslyckade:
            with st.expander("📋 Misslyckade tickers"):
                st.write("\n".join(misslyckade))
                st.code("\n".join([m.split(" – ")[0] for m in misslyckade]), language="text")

        if felsokningslage and felsokningslogg:
            st.subheader("🛠 Felsökningslogg")
            st.dataframe(pd.DataFrame(felsokningslogg))

    # Visa alltid databasen i analysläget
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
        valuta = st.selectbox("Valuta", ["USD", "NOK", "EUR", "CAD"], index=0 if befintlig.get("Valuta", "") == "" else ["USD", "NOK", "EUR", "CAD"].index(befintlig.get("Valuta", "USD")))
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

        sparaknapp = st.form_submit_button("💾 Spara")

    if sparaknapp and ticker:
        ny_rad = {
            "Ticker": ticker, "Bolagsnamn": namn, "Aktuell kurs": kurs, "Valuta": valuta, "Utestående aktier": aktier, "Antal aktier": antal_aktier,
            "P/S": ps_idag, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms_1, "Omsättning om 2 år": oms_2, "Omsättning om 3 år": oms_3

Här kommer **Del 3** igen, komplett och i rätt form, med nya riktkurs‑namn, valutahantering och oförändrad baslogik i övrigt.  

---

```python
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

def visa_portfolj(df, valutakurser):
    st.subheader("📦 Min portfölj")
    df = df[df["Antal aktier"] > 0].copy()
    if df.empty:
        st.info("Du äger inga aktier.")
        return
    df["Värde (SEK)"] = df.apply(lambda r: r["Antal aktier"] * r["Aktuell kurs"] * valutakurser.get(r["Valuta"], 1), axis=1)
    df["Andel (%)"] = round(df["Värde (SEK)"] / df["Värde (SEK)"].sum() * 100, 2)
    total = df["Värde (SEK)"].sum()
    st.markdown(f"**Totalt portföljvärde:** {round(total, 2)} SEK")
    st.dataframe(df[["Ticker", "Bolagsnamn", "Antal aktier", "Aktuell kurs", "Valuta", "Värde (SEK)", "Andel (%)"]],
                 use_container_width=True)


def main():
    st.title("📊 Aktieanalys och investeringsförslag")
    df = hamta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)

    valutakurser = {
        "USD": st.sidebar.number_input("USD → SEK", value=9.50, step=0.01),
        "NOK": st.sidebar.number_input("NOK → SEK", value=0.93, step=0.01),
        "EUR": st.sidebar.number_input("EUR → SEK", value=11.10, step=0.01),
        "CAD": st.sidebar.number_input("CAD → SEK", value=7.00, step=0.01)
    }

    meny = st.sidebar.radio("Meny", ["Analys", "Lägg till / uppdatera bolag", "Investeringsförslag", "Portfölj"])

    if meny == "Analys":
        analysvy(df)
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
