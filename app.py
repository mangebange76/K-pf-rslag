import streamlit as st
import pandas as pd
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
    sheet = skapa_koppling()
    data = sheet.get_all_records()
    return pd.DataFrame(data)

def spara_data(df):
    sheet = skapa_koppling()
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.astype(str).values.tolist())

def säkerställ_kolumner(df):
    kolumner = [
        "Ticker", "Bolagsnamn", "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "P/S-snitt", "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år", "Antal aktier",
        "Valuta", "Aktuell kurs", "Årlig utdelning", "CAGR 5 år (%)"
    ]
    for kolumn in kolumner:
        if kolumn not in df.columns:
            df[kolumn] = ""

    # Rensa bort överflödiga kolumner
    tillåtna = set(kolumner)
    df = df[[k for k in df.columns if k in tillåtna]]

    return df

def justera_omsättning_med_cagr(df):
    for i, row in df.iterrows():
        try:
            omsättning = float(row.get("Omsättning nästa år", 0))
            cagr = float(row.get("CAGR 5 år (%)", 0)) / 100
            if pd.isna(omsättning) or omsättning == 0:
                continue

            if cagr > 1:
                tillväxt = 0.5
            elif cagr < 0:
                tillväxt = 0.02
            else:
                tillväxt = cagr

            df.at[i, "Omsättning om 2 år"] = round(omsättning * (1 + tillväxt), 2)
            df.at[i, "Omsättning om 3 år"] = round(omsättning * (1 + tillväxt) ** 2, 2)
        except Exception:
            continue
    return df

def hamta_yahoo_data(ticker):
    try:
        aktie = yf.Ticker(ticker)
        info = aktie.info

        namn = info.get("longName") or info.get("shortName") or ""
        kurs = info.get("currentPrice") or info.get("regularMarketPrice") or 0.0
        valuta = info.get("currency") or ""
        utdelning = info.get("dividendRate") or 0.0
        cagr = info.get("revenueGrowth") or 0.0  # Kommer som 0.12 = 12%
        cagr = round(cagr * 100, 2)

        return {
            "Bolagsnamn": namn,
            "Aktuell kurs": kurs,
            "Valuta": valuta,
            "Årlig utdelning": utdelning,
            "CAGR 5 år (%)": cagr
        }
    except Exception:
        return {}

def beräkna_allt(df):
    df = säkerställ_kolumner(df)

    for i, row in df.iterrows():
        try:
            # P/S-snitt
            ps_list = [float(row.get(k, 0)) for k in ["P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"]]
            ps_snitt = sum(ps_list) / len([x for x in ps_list if x != 0])
            df.at[i, "P/S-snitt"] = round(ps_snitt, 2)

            # Riktkurser
            kursdata = {
                "idag": "Omsättning idag",
                "1 år": "Omsättning nästa år",
                "2 år": "Omsättning om 2 år",
                "3 år": "Omsättning om 3 år",
            }
            for nyckel, kolumn in kursdata.items():
                try:
                    omsättning = float(row.get(kolumn, 0))
                    aktier = float(row.get("Utestående aktier", 0))
                    riktkurs = (omsättning * ps_snitt) / aktier if aktier > 0 else 0
                    df.at[i, f"Riktkurs om {nyckel}"] = round(riktkurs, 2)
                except:
                    df.at[i, f"Riktkurs om {nyckel}"] = ""
        except:
            continue

    df = justera_omsättning_med_cagr(df)
    return df

def beräkna_portföljvärde(df):
    try:
        df["Antal aktier"] = pd.to_numeric(df["Antal aktier"], errors="coerce")
        df["Aktuell kurs"] = pd.to_numeric(df["Aktuell kurs"], errors="coerce")
        df["Årlig utdelning"] = pd.to_numeric(df["Årlig utdelning"], errors="coerce")

        df["Värde (SEK)"] = df["Antal aktier"] * df["Aktuell kurs"]
        df["Utdelning (SEK)"] = df["Antal aktier"] * df["Årlig utdelning"]

        total_värde = df["Värde (SEK)"].sum()
        total_utdelning = df["Utdelning (SEK)"].sum()
        utdelning_per_månad = round(total_utdelning / 12, 2)

        return total_värde, total_utdelning, utdelning_per_månad
    except:
        return 0, 0, 0

def lagg_till_eller_uppdatera(df):
    st.subheader("Lägg till eller uppdatera bolag")

    tickers = df["Ticker"].dropna().unique().tolist()
    valt_befintligt = st.selectbox("Välj befintligt bolag (för uppdatering)", [""] + tickers)

    if valt_befintligt:
        befintlig = df[df["Ticker"] == valt_befintligt].iloc[0]
    else:
        befintlig = {}

    with st.form("bolagsformulär"):
        ticker = st.text_input("Ticker", value=befintlig.get("Ticker", ""))
        antal_aktier = st.number_input("Antal aktier", value=float(befintlig.get("Antal aktier", 0.0)))
        utestående = st.number_input("Utestående aktier", value=float(befintlig.get("Utestående aktier", 0.0)))
        ps = st.number_input("P/S", value=float(befintlig.get("P/S", 0.0)))
        ps_q1 = st.number_input("P/S Q1", value=float(befintlig.get("P/S Q1", 0.0)))
        ps_q2 = st.number_input("P/S Q2", value=float(befintlig.get("P/S Q2", 0.0)))
        ps_q3 = st.number_input("P/S Q3", value=float(befintlig.get("P/S Q3", 0.0)))
        ps_q4 = st.number_input("P/S Q4", value=float(befintlig.get("P/S Q4", 0.0)))
        oms_idag = st.number_input("Omsättning idag", value=float(befintlig.get("Omsättning idag", 0.0)))
        oms_nasta = st.number_input("Omsättning nästa år", value=float(befintlig.get("Omsättning nästa år", 0.0)))

        sparaknapp = st.form_submit_button("Spara bolag")

    if sparaknapp and ticker:
        ny_data = {
            "Ticker": ticker,
            "Antal aktier": antal_aktier,
            "Utestående aktier": utestående,
            "P/S": ps,
            "P/S Q1": ps_q1,
            "P/S Q2": ps_q2,
            "P/S Q3": ps_q3,
            "P/S Q4": ps_q4,
            "Omsättning idag": oms_idag,
            "Omsättning nästa år": oms_nasta,
        }

        yahoo_data = hamta_yahoo_data(ticker)
        ny_data.update(yahoo_data)

        # CAGR-baserad omsättning om 2 och 3 år
        cagr = yahoo_data.get("CAGR 5 år (%)", 0.0)
        if cagr > 100:
            justerad_cagr = 50
        elif cagr < 0:
            justerad_cagr = 2
        else:
            justerad_cagr = cagr

        if oms_nasta > 0:
            oms2 = oms_nasta * (1 + justerad_cagr / 100)
            oms3 = oms_nasta * ((1 + justerad_cagr / 100) ** 2)
        else:
            oms2, oms3 = 0, 0

        ny_data["Omsättning om 2 år"] = round(oms2, 2)
        ny_data["Omsättning om 3 år"] = round(oms3, 2)

        df = df[df["Ticker"] != ticker]
        df = pd.concat([df, pd.DataFrame([ny_data])], ignore_index=True)

        df = säkerställ_kolumner(df)
        df = beräkna_allt(df)
        spara_data(df)
        st.success(f"{ticker} sparades och uppdaterades med Yahoo Finance-data.")

    return df

def analysvy(df):
    st.subheader("Analys av bolag")

    alla_tickers = df["Ticker"].dropna().unique().tolist()
    valt_ticker = st.selectbox("Välj ett bolag att visa detaljer för", [""] + alla_tickers)

    if valt_ticker:
        bolag = df[df["Ticker"] == valt_ticker].iloc[0]
        st.write("### Detaljerad information")
        for kolumn in df.columns:
            st.write(f"**{kolumn}**: {bolag.get(kolumn, '')}")

    st.write("### Hela databasen")
    st.dataframe(df)

def investeringsforslag(df):
    st.subheader("Investeringsförslag")

    belopp = st.number_input("Tillgängligt belopp (SEK)", min_value=0, value=0)

    riktkurs_val = st.selectbox(
        "Sortera efter uppsida i riktkurs...",
        ("Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år")
    )

    kolumn_mapping = {
        "Riktkurs idag": "Riktkurs idag",
        "Riktkurs om 1 år": "Riktkurs om 1 år",
        "Riktkurs om 2 år": "Riktkurs om 2 år",
        "Riktkurs om 3 år": "Riktkurs om 3 år"
    }

    kolumn = kolumn_mapping[riktkurs_val]
    df = df.copy()
    df["Uppsida (%)"] = ((df[kolumn] - df["Aktuell kurs"]) / df["Aktuell kurs"]) * 100
    df = df.sort_values(by="Uppsida (%)", ascending=False).reset_index(drop=True)

    if len(df) == 0:
        st.warning("Inga bolag att visa.")
        return

    if "position" not in st.session_state:
        st.session_state["position"] = 0

    pos = st.session_state["position"]
    max_pos = len(df) - 1
    bolag = df.iloc[pos]

    st.write(f"### {bolag['Bolagsnamn']} ({bolag['Ticker']})")

    st.metric("Nuvarande kurs", f"{bolag['Aktuell kurs']:.2f} {bolag['Valuta']}")
    st.metric("Riktkurs nu", f"{bolag['Riktkurs idag']:.2f} {bolag['Valuta']}")
    st.metric("Riktkurs om 1 år", f"{bolag['Riktkurs om 1 år']:.2f} {bolag['Valuta']}")
    st.metric("Riktkurs om 2 år", f"{bolag['Riktkurs om 2 år']:.2f} {bolag['Valuta']}")
    st.metric("Riktkurs om 3 år", f"{bolag['Riktkurs om 3 år']:.2f} {bolag['Valuta']}")
    st.metric("Uppsida (%)", f"{bolag['Uppsida (%)']:.2f}%")

    if belopp > 0:
        pris = bolag["Aktuell kurs"]
        antal_kop = int(belopp // pris)
        antal_ager = int(bolag.get("Antal aktier", 0))
        portfoljvarde = df["Aktuell kurs"] * df["Antal aktier"]
        totalt_varde = portfoljvarde.sum()
        andel_fore = (antal_ager * pris) / totalt_varde * 100 if totalt_varde > 0 else 0
        andel_efter = ((antal_ager + antal_kop) * pris) / totalt_varde * 100 if totalt_varde > 0 else 0

        st.write(f"**Antal att köpa:** {antal_kop}")
        st.write(f"**Du äger redan:** {antal_ager} aktier")
        st.write(f"**Andel av portfölj före köp:** {andel_fore:.2f}%")
        st.write(f"**Andel av portfölj efter köp:** {andel_efter:.2f}%")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("⬅️ Föregående", disabled=pos == 0):
            st.session_state["position"] = max(0, pos - 1)
    with col2:
        st.write(f"**{pos + 1} / {len(df)}**")
    with col3:
        if st.button("➡️ Nästa", disabled=pos == max_pos):
            st.session_state["position"] = min(max_pos, pos + 1)

def portfoljvy(df):
    st.subheader("📊 Portföljsammanställning")

    df = df.copy()

    # Ta bort rader utan kurs eller antal aktier
    df = df[(df["Aktuell kurs"] > 0) & (df["Antal aktier"] > 0)]

    if df.empty:
        st.info("Ingen portföljdata att visa.")
        return

    df["Värde (SEK)"] = df["Aktuell kurs"] * df["Antal aktier"]
    df["Utdelning (SEK)"] = df["Årlig utdelning"] * df["Antal aktier"]

    totalt_varde = df["Värde (SEK)"].sum()
    total_utdelning = df["Utdelning (SEK)"].sum()
    utdelning_per_manad = total_utdelning / 12

    st.metric("Totalt portföljvärde", f"{totalt_varde:,.0f} SEK")
    st.metric("Total kommande utdelning", f"{total_utdelning:,.0f} SEK")
    st.metric("Utdelning per månad (snitt)", f"{utdelning_per_manad:,.0f} SEK")

def main():
    st.title("📈 Aktieanalys och investeringsförslag")

    df = hamta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)
    df = beräkna_allt(df)

    menyval = st.sidebar.radio("Meny", [
        "Lägg till / uppdatera bolag",
        "Analys",
        "Investeringsförslag",
        "Portfölj",
        "Uppdatera alla bolag"
    ])

    if menyval == "Lägg till / uppdatera bolag":
        df = lagg_till_eller_uppdatera(df)
    elif menyval == "Analys":
        analysvy(df)
    elif menyval == "Investeringsförslag":
        investeringsforslag_vy(df)
    elif menyval == "Portfölj":
        portfoljvy(df)
    elif menyval == "Uppdatera alla bolag":
        df = massuppdatera_alla(df)

    spara_data(df)

if __name__ == "__main__":
    main()
