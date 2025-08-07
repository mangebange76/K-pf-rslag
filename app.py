import streamlit as st
import pandas as pd
import yfinance as yf
import time
import gspread
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

def skapa_koppling():
    return client.open_by_url(SHEET_URL).worksheet(SHEET_NAME)

def hämta_data():
    sheet = skapa_koppling()
    data = sheet.get_all_records()
    return pd.DataFrame(data)

def spara_data(df):
    sheet = skapa_koppling()
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.astype(str).values.tolist())

def säkerställ_kolumner(df):
    önskade_kolumner = [
        "Ticker", "Bolagsnamn", "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "P/S-snitt", "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
        "Aktuell kurs", "Antal aktier", "Valuta", "Årlig utdelning", "Äger", "CAGR 5 år (%)"
    ]
    for kolumn in önskade_kolumner:
        if kolumn not in df.columns:
            df[kolumn] = ""
    df = df[önskade_kolumner]
    return df

def konvertera_typer(df):
    kolumner_att_konvertera = [
        "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "Aktuell kurs", "Antal aktier", "Årlig utdelning", "CAGR 5 år (%)"
    ]
    for kolumn in kolumner_att_konvertera:
        if kolumn in df.columns:
            df[kolumn] = pd.to_numeric(df[kolumn], errors="coerce")
    return df

def beräkna_allt(df):
    df = konvertera_typer(df)

    df["P/S-snitt"] = df[["P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"]].mean(axis=1)

    # Justering för CAGR
    justerad_cagr = []
    for val in df["CAGR 5 år (%)"]:
        if pd.isna(val):
            justerad_cagr.append(None)
        elif val > 100:
            justerad_cagr.append(0.50)
        elif val < 0:
            justerad_cagr.append(0.02)
        else:
            justerad_cagr.append(val / 100)
    df["justerad_cagr"] = justerad_cagr

    df["Omsättning om 2 år"] = df["Omsättning nästa år"] * (1 + df["justerad_cagr"])
    df["Omsättning om 3 år"] = df["Omsättning nästa år"] * (1 + df["justerad_cagr"])**2

    df["Riktkurs idag"] = df["Omsättning idag"] * df["P/S-snitt"] / df["Utestående aktier"]
    df["Riktkurs om 1 år"] = df["Omsättning nästa år"] * df["P/S-snitt"] / df["Utestående aktier"]
    df["Riktkurs om 2 år"] = df["Omsättning om 2 år"] * df["P/S-snitt"] / df["Utestående aktier"]
    df["Riktkurs om 3 år"] = df["Omsättning om 3 år"] * df["P/S-snitt"] / df["Utestående aktier"]

    df["Utdelning (SEK)"] = df["Årlig utdelning"] * df["Antal aktier"]
    df["Värde (SEK)"] = df["Aktuell kurs"] * df["Antal aktier"]
    return df

def hämta_från_yahoo(ticker):
    try:
        aktie = yf.Ticker(ticker)
        info = aktie.info

        namn = info.get("longName", "")
        kurs = info.get("currentPrice", None)
        valuta = info.get("currency", "")
        utdelning = info.get("dividendRate", 0.0)
        tillväxt = info.get("fiveYearAvgDividendYield", None)

        if tillväxt is None:
            tillväxt = info.get("revenueGrowth", None)
        cagr = tillväxt * 100 if tillväxt is not None else None

        return {
            "Bolagsnamn": namn,
            "Aktuell kurs": kurs,
            "Valuta": valuta,
            "Årlig utdelning": utdelning,
            "CAGR 5 år (%)": cagr
        }
    except Exception:
        return {}

def formulär(df):
    with st.form("nytt_bolag"):
        st.subheader("Lägg till eller uppdatera ett bolag")
        ticker = st.text_input("Ticker").upper()

        col1, col2 = st.columns(2)
        with col1:
            utestående = st.number_input("Utestående aktier", min_value=0.0)
            ps = st.number_input("P/S", min_value=0.0)
            ps_q1 = st.number_input("P/S Q1", min_value=0.0)
            ps_q2 = st.number_input("P/S Q2", min_value=0.0)
        with col2:
            ps_q3 = st.number_input("P/S Q3", min_value=0.0)
            ps_q4 = st.number_input("P/S Q4", min_value=0.0)
            oms_idag = st.number_input("Omsättning idag", min_value=0.0)
            oms_next = st.number_input("Omsättning nästa år", min_value=0.0)

        antal_aktier = st.number_input("Antal aktier", min_value=0)
        äger = st.selectbox("Äger du aktien?", ["Ja", "Nej"])

        sparaknapp = st.form_submit_button("Spara bolag")

    if sparaknapp:
        data = {
            "Ticker": ticker,
            "Utestående aktier": utestående,
            "P/S": ps,
            "P/S Q1": ps_q1,
            "P/S Q2": ps_q2,
            "P/S Q3": ps_q3,
            "P/S Q4": ps_q4,
            "Omsättning idag": oms_idag,
            "Omsättning nästa år": oms_next,
            "Antal aktier": antal_aktier,
            "Äger": äger
        }

        hämtad = hämta_från_yahoo(ticker)
        data.update(hämtad)

        for kolumn in df.columns:
            if kolumn not in data:
                data[kolumn] = None

        ny_df = pd.DataFrame([data])
        ny_df = beräkna_allt(ny_df)

        df = df[df["Ticker"] != ticker]
        df = pd.concat([df, ny_df], ignore_index=True)

        spara_data(df)
        st.success(f"{ticker} har sparats och uppdaterats.")

def analysvy(df):
    st.subheader("Analysvy")

    sorteringsval = st.selectbox(
        "Sortera bolag efter uppsida i riktkurs:",
        ["Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år"]
    )

    sort_kolumn = {
        "Riktkurs idag": "Riktkurs idag",
        "Riktkurs om 1 år": "Riktkurs om 1 år",
        "Riktkurs om 2 år": "Riktkurs om 2 år",
        "Riktkurs om 3 år": "Riktkurs om 3 år",
    }[sorteringsval]

    df["Uppsida (%)"] = ((df[sort_kolumn] - df["Aktuell kurs"]) / df["Aktuell kurs"]) * 100
    df = df.sort_values(by="Uppsida (%)", ascending=False).reset_index(drop=True)

    st.markdown("### Välj bolag att visa:")
    valda_ticker = st.selectbox("Välj bolag:", df["Ticker"].unique())

    valt_bolag = df[df["Ticker"] == valda_ticker]
    st.dataframe(valt_bolag)

    st.markdown("### Hela databasen")
    st.dataframe(df)

    st.markdown("### Uppdatera enskilt bolag")
    enskild = st.selectbox("Välj bolag att uppdatera från Yahoo", df["Ticker"].unique(), key="enskild")
    if st.button("Uppdatera valt bolag"):
        ny_data = hämta_från_yahoo(enskild)
        for k, v in ny_data.items():
            df.loc[df["Ticker"] == enskild, k] = v

        df.loc[df["Ticker"] == enskild] = beräkna_allt(df[df["Ticker"] == enskild])
        spara_data(df)
        st.success(f"{enskild} har uppdaterats.")

def visa_portfolj(df):
    st.subheader("Portfölj")

    df = df.copy()
    if "Äger" not in df.columns:
        st.error("Kolumnen 'Äger' saknas i databasen.")
        return

    df = df[df["Äger"].str.lower() == "ja"]

    if df.empty:
        st.info("Inga bolag markerade som 'Äger'.")
        return

    df["Antal aktier"] = pd.to_numeric(df["Antal aktier"], errors="coerce").fillna(0)
    df["Aktuell kurs"] = pd.to_numeric(df["Aktuell kurs"], errors="coerce").fillna(0)
    df["Årlig utdelning"] = pd.to_numeric(df["Årlig utdelning"], errors="coerce").fillna(0)

    df["Värde (SEK)"] = df["Antal aktier"] * df["Aktuell kurs"]
    df["Utdelning (SEK)"] = df["Antal aktier"] * df["Årlig utdelning"]

    totalt_värde = df["Värde (SEK)"].sum()
    total_utdelning = df["Utdelning (SEK)"].sum()
    utdelning_per_månad = total_utdelning / 12

    st.metric("Totalt portföljvärde (SEK)", f"{totalt_värde:,.0f}")
    st.metric("Total kommande utdelning (SEK)", f"{total_utdelning:,.0f}")
    st.metric("Utdelning per månad (SEK)", f"{utdelning_per_månad:,.0f}")

    st.dataframe(df[
        ["Ticker", "Bolagsnamn", "Antal aktier", "Aktuell kurs", "Värde (SEK)", "Årlig utdelning", "Utdelning (SEK)"]
    ])

def investeringsförslag(df):
    st.subheader("Investeringsförslag")

    df = df.copy()
    df = konvertera_typer(df)

    riktkursval = st.selectbox(
        "Sortera efter uppsida baserat på:",
        ["Riktkurs", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år"]
    )

    df["Uppside (%)"] = ((df[riktkursval] - df["Aktuell kurs"]) / df["Aktuell kurs"]) * 100
    df = df.sort_values("Uppside (%)", ascending=False).reset_index(drop=True)

    if df.empty:
        st.warning("Inga bolag med data att visa.")
        return

    total = len(df)
    if "inv_index" not in st.session_state:
        st.session_state.inv_index = 0

    kol1, kol2 = st.columns([1, 4])
    with kol1:
        if st.button("◀️ Föregående", use_container_width=True) and st.session_state.inv_index > 0:
            st.session_state.inv_index -= 1
    with kol2:
        if st.button("Nästa ▶️", use_container_width=True) and st.session_state.inv_index < total - 1:
            st.session_state.inv_index += 1

    rad = df.iloc[st.session_state.inv_index]
    st.markdown(f"### {rad['Ticker']} – {rad['Bolagsnamn']}")
    st.write(f"Aktuell kurs: **{rad['Aktuell kurs']}** {rad['Valuta']}")
    st.write(f"Riktkurs: **{rad['Riktkurs']}** ({round((rad['Riktkurs'] - rad['Aktuell kurs']) / rad['Aktuell kurs'] * 100, 1)} % uppsida)")
    st.write(f"Riktkurs om 1 år: **{rad['Riktkurs om 1 år']}**")
    st.write(f"Riktkurs om 2 år: **{rad['Riktkurs om 2 år']}**")
    st.write(f"Riktkurs om 3 år: **{rad['Riktkurs om 3 år']}**")
    st.write(f"CAGR 5 år: **{rad['CAGR 5 år (%)']} %**")

    belopp = st.number_input("Tillgängligt belopp (SEK)", value=0.0, step=100.0)
    if belopp > 0 and rad["Aktuell kurs"] > 0:
        möjliga = int(belopp // rad["Aktuell kurs"])
        st.write(f"För detta belopp kan du köpa **{möjliga} aktier**.")

        antal_ägda = rad["Antal aktier"] if pd.notna(rad["Antal aktier"]) else 0
        totalt_värde = antal_ägda * rad["Aktuell kurs"]
        efter_köp = (antal_ägda + möjliga) * rad["Aktuell kurs"]

        st.write(f"Du äger redan **{antal_ägda} aktier** (värde: {totalt_värde:,.0f} SEK)")
        st.write(f"Efter köp skulle du ha aktier för **{efter_köp:,.0f} SEK**")

def analysvy(df):
    st.subheader("Analysvy")

    df = df.copy()
    df = konvertera_typer(df)

    # Rullista för att välja ett bolag
    alla_bolag = df["Ticker"].dropna().unique().tolist()
    valt_bolag = st.selectbox("Välj ett bolag för detaljerad visning:", alla_bolag)

    if valt_bolag:
        bolagsdata = df[df["Ticker"] == valt_bolag]
        st.markdown(f"### Detaljer för {valt_bolag}")
        st.dataframe(bolagsdata.T, use_container_width=True)

    st.markdown("---")
    st.markdown("### Hela databasen")
    st.dataframe(df, use_container_width=True)

def main():
    st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")
    st.title("📈 Aktieanalys och investeringsförslag")

    df = hämta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)
    df = beräkna_allt(df)

    meny = st.sidebar.radio("Meny", ["Lägg till / uppdatera", "Analys", "Investeringsförslag", "Portfölj", "Massuppdatering"])

    if meny == "Lägg till / uppdatera":
        formulär(df)
    elif meny == "Analys":
        analysvy(df)
    elif meny == "Investeringsförslag":
        investeringsförslag(df)
    elif meny == "Portfölj":
        visa_portfolj(df)
    elif meny == "Massuppdatering":
        massuppdatera_alla(df)

if __name__ == "__main__":
    main()
