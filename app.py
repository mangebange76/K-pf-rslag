import streamlit as st
import pandas as pd
import yfinance as yf
import gspread
import time
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

# Google Sheets-inställningar
SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Data"

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
    kolumner = [
        "Ticker", "Bolagsnamn", "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "P/S-snitt", "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
        "Aktuell kurs", "Antal aktier", "Valuta", "Årlig utdelning", "CAGR 5 år (%)", "Äger"
    ]
    for kolumn in kolumner:
        if kolumn not in df.columns:
            df[kolumn] = ""
    # Ta bort oanvända kolumner
    tillåtna = set(kolumner)
    df = df[[k for k in df.columns if k in tillåtna]]
    return df

def hämta_data_från_yahoo(ticker):
    try:
        aktie = yf.Ticker(ticker)
        info = aktie.info

        namn = info.get("longName") or ""
        kurs = info.get("currentPrice") or ""
        valuta = info.get("currency") or ""
        utdelning = info.get("dividendRate") or ""
        cagr = info.get("fiveYearAvgDividendYield") or ""

        if cagr and cagr > 0:
            cagr = round(cagr / 100, 4)  # Om värdet är i %, gör om till faktor

        return {
            "Bolagsnamn": namn,
            "Aktuell kurs": kurs,
            "Valuta": valuta,
            "Årlig utdelning": utdelning,
            "CAGR 5 år (%)": cagr
        }
    except Exception:
        return {}

def konvertera_typer(df):
    kolumner = [
        "Aktuell kurs", "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "Antal aktier", "Årlig utdelning", "CAGR 5 år (%)"
    ]
    for kol in kolumner:
        df[kol] = pd.to_numeric(df[kol], errors="coerce").fillna(0)
    return df

def beräkna_allt(df):
    df = konvertera_typer(df)

    df["P/S-snitt"] = df[["P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"]].mean(axis=1)

    def justera_cagr(row):
        cagr = row["CAGR 5 år (%)"]
        if cagr > 1.0:
            return 0.5
        elif cagr < 0:
            return 0.02
        return cagr

    df["justerad_cagr"] = df.apply(justera_cagr, axis=1)

    df["Omsättning om 2 år"] = df["Omsättning nästa år"] * (1 + df["justerad_cagr"])
    df["Omsättning om 3 år"] = df["Omsättning nästa år"] * (1 + df["justerad_cagr"])**2

    df["Riktkurs idag"] = df["P/S-snitt"] * df["Omsättning idag"] / df["Utestående aktier"]
    df["Riktkurs om 1 år"] = df["P/S-snitt"] * df["Omsättning nästa år"] / df["Utestående aktier"]
    df["Riktkurs om 2 år"] = df["P/S-snitt"] * df["Omsättning om 2 år"] / df["Utestående aktier"]
    df["Riktkurs om 3 år"] = df["P/S-snitt"] * df["Omsättning om 3 år"] / df["Utestående aktier"]

    return df

def formulär(df):
    with st.form("nytt_bolag"):
        st.subheader("Lägg till eller uppdatera bolag")
        ticker = st.text_input("Ticker (ex. AMD)").upper()
        utestående = st.number_input("Utestående aktier", min_value=0.0, step=1.0)
        ps = st.number_input("P/S", min_value=0.0, step=0.1)
        ps_q1 = st.number_input("P/S Q1", min_value=0.0, step=0.1)
        ps_q2 = st.number_input("P/S Q2", min_value=0.0, step=0.1)
        ps_q3 = st.number_input("P/S Q3", min_value=0.0, step=0.1)
        ps_q4 = st.number_input("P/S Q4", min_value=0.0, step=0.1)
        oms_idag = st.number_input("Omsättning idag", min_value=0.0, step=1.0)
        oms_next = st.number_input("Omsättning nästa år", min_value=0.0, step=1.0)
        antal_aktier = st.number_input("Antal aktier i portföljen", min_value=0, step=1)
        äger = st.selectbox("Äger du aktien?", ["Ja", "Nej"])

        sparaknapp = st.form_submit_button("Spara bolag")

    if sparaknapp and ticker:
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
            "Äger": äger,
        }

        yahoo_data = hämta_data_från_yahoo(ticker)
        data.update(yahoo_data)

        index = df[df["Ticker"] == ticker].index
        if len(index) > 0:
            for key, value in data.items():
                df.at[index[0], key] = value
        else:
            df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)

        df = beräkna_allt(df)
        spara_data(df)
        st.success(f"{ticker} sparat!")

    return df

def visa_portfolj(df):
    st.subheader("📊 Portföljsammanställning")

    df = df[df["Äger"].str.lower() == "ja"]
    df = df.copy()

    df["Portföljvärde (SEK)"] = df["Aktuell kurs"] * df["Antal aktier"]
    df["Utdelning total (SEK)"] = df["Årlig utdelning"] * df["Antal aktier"]

    total_värde = df["Portföljvärde (SEK)"].sum()
    total_utdelning = df["Utdelning total (SEK)"].sum()
    utdelning_per_månad = total_utdelning / 12

    st.metric("Totalt portföljvärde (SEK)", f"{total_värde:,.0f}")
    st.metric("Total kommande utdelning (SEK)", f"{total_utdelning:,.0f}")
    st.metric("Utdelning per månad (SEK)", f"{utdelning_per_månad:,.0f}")


def investeringsförslag(df):
    st.subheader("💡 Investeringsförslag")

    riktkursval = st.selectbox("Sortera efter uppsida i:", [
        "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år"
    ])

    tillgängligt_belopp = st.number_input("Tillgängligt belopp (SEK)", min_value=0, step=100)

    df = df.copy()
    df = df[df["Aktuell kurs"] > 0]
    df["Uppside (%)"] = ((df[riktkursval] - df["Aktuell kurs"]) / df["Aktuell kurs"]) * 100
    df = df.sort_values("Uppside (%)", ascending=False).reset_index(drop=True)

    if len(df) == 0:
        st.warning("Inga bolag med data att visa.")
        return

    i = st.session_state.get("inv_index", 0)
    if st.button("⬅️ Föregående") and i > 0:
        i -= 1
    if st.button("➡️ Nästa") and i < len(df) - 1:
        i += 1
    st.session_state.inv_index = i

    row = df.iloc[i]
    st.markdown(f"### {row['Ticker']}")
    st.markdown(f"**Nuvarande kurs:** {row['Aktuell kurs']}")
    st.markdown(f"**Riktkurs idag:** {row['Riktkurs idag']}")
    st.markdown(f"**Riktkurs om 1 år:** {row['Riktkurs om 1 år']}")
    st.markdown(f"**Riktkurs om 2 år:** {row['Riktkurs om 2 år']}")
    st.markdown(f"**Riktkurs om 3 år:** {row['Riktkurs om 3 år']}")
    st.markdown(f"**Uppside (%):** {row['Uppside (%)']:.2f}%")

    if tillgängligt_belopp > 0:
        aktier_köp = int(tillgängligt_belopp // row["Aktuell kurs"])
        nuvarande = row["Antal aktier"]
        framtida = nuvarande + aktier_köp
        kurs = row["Aktuell kurs"]
        nuvärde = nuvarande * kurs
        framtida_värde = framtida * kurs
        total_värde = df[df["Äger"].str.lower() == "ja"]["Aktuell kurs"] * df[df["Äger"].str.lower() == "ja"]["Antal aktier"]
        total_portföljvärde = total_värde.sum() + tillgängligt_belopp

        andel_nu = (nuvärde / total_portföljvärde) * 100 if total_portföljvärde else 0
        andel_sen = (framtida_värde / total_portföljvärde) * 100 if total_portföljvärde else 0

        st.markdown(f"- Aktier du kan köpa: **{aktier_köp}**")
        st.markdown(f"- Aktier du redan äger: **{nuvärde / kurs:.0f}**")
        st.markdown(f"- Nuvarande portföljandel: **{andel_nu:.2f}%**")
        st.markdown(f"- Portföljandel efter köp: **{andel_sen:.2f}%**")

def analysvy(df):
    st.subheader("🔎 Analys")

    val = st.selectbox("Välj bolag att visa detaljer för", df["Ticker"].unique())

    valt_bolag = df[df["Ticker"] == val]
    st.write("**Detaljer för valt bolag:**")
    st.dataframe(valt_bolag, use_container_width=True)

    st.write("---")
    st.write("**Hela databasen:**")
    st.dataframe(df, use_container_width=True)


def main():
    st.title("📈 Aktieanalys och investeringsförslag")

    menyval = st.sidebar.radio("Meny", [
        "Lägg till / uppdatera bolag",
        "Analys",
        "Portfölj",
        "Uppdatera alla bolag",
        "Investeringsförslag"
    ])

    df = hämta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)
    df = beräkna_allt(df)

    if menyval == "Lägg till / uppdatera bolag":
        formulär(df)

    elif menyval == "Analys":
        analysvy(df)

    elif menyval == "Portfölj":
        visa_portfolj(df)

    elif menyval == "Uppdatera alla bolag":
        massuppdatera(df)

    elif menyval == "Investeringsförslag":
        investeringsförslag(df)


if __name__ == "__main__":
    main()
