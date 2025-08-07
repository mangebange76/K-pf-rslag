import streamlit as st
import pandas as pd
import gspread
import yfinance as yf
import time
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

# Google Sheets-koppling
SHEET_URL = "https://docs.google.com/spreadsheets/d/1-5JSJpqBB0j7sm3cgEGZmnFoBL_oJDPMpLdleggL0HQ/edit"
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
    kolumner = [
        "Ticker", "Bolagsnamn", "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
        "Antal aktier", "Valuta", "Årlig utdelning", "Aktuell kurs", "CAGR 5 år (%)", "P/S-snitt"
    ]
    for kolumn in kolumner:
        if kolumn not in df.columns:
            df[kolumn] = ""

    # Radera överflödiga kolumner som inte används
    oanvända_kolumner = [col for col in df.columns if col not in kolumner]
    df.drop(columns=oanvända_kolumner, inplace=True)

    return df

def konvertera_typer(df):
    num_kolumner = [
        "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
        "Antal aktier", "Aktuell kurs", "Årlig utdelning", "CAGR 5 år (%)", "P/S-snitt"
    ]
    for kolumn in num_kolumner:
        df[kolumn] = pd.to_numeric(df[kolumn], errors="coerce")
    return df

def hämta_och_beräkna_data(ticker):
    try:
        aktie = yf.Ticker(ticker)
        info = aktie.info

        namn = info.get("longName") or info.get("shortName") or ""
        kurs = info.get("currentPrice")
        valuta = info.get("currency")
        utdelning = info.get("dividendRate")

        # Hämta historisk omsättning
        fin_data = aktie.financials
        oms = aktie.income_stmt
        omsättning = None
        if not oms.empty:
            omsättning = oms.loc["Total Revenue"].sort_index(ascending=True)

        cagr = None
        oms1 = oms5 = None
        if omsättning is not None and len(omsättning) >= 5:
            oms1 = omsättning.iloc[0]
            oms5 = omsättning.iloc[4]
            if oms1 and oms5:
                cagr = ((oms5 / oms1) ** (1 / 4) - 1) * 100

        return namn, kurs, valuta, utdelning, cagr
    except Exception:
        return "", None, None, None, None

def formulär(df):
    st.subheader("Lägg till eller uppdatera bolag")

    befintliga_tickers = df["Ticker"].dropna().unique().tolist()
    valt_bolag = st.selectbox("Välj ett bolag att uppdatera eller lämna tomt för nytt", [""] + befintliga_tickers)

    if valt_bolag:
        bolagsdata = df[df["Ticker"] == valt_bolag].iloc[0]
    else:
        bolagsdata = pd.Series(dtype=object)

    with st.form("nytt_bolag"):
        ticker = st.text_input("Ticker", bolagsdata.get("Ticker", ""))
        utestående = st.number_input("Utestående aktier", value=bolagsdata.get("Utestående aktier", 0.0))
        ps = st.number_input("P/S", value=bolagsdata.get("P/S", 0.0))
        ps_q1 = st.number_input("P/S Q1", value=bolagsdata.get("P/S Q1", 0.0))
        ps_q2 = st.number_input("P/S Q2", value=bolagsdata.get("P/S Q2", 0.0))
        ps_q3 = st.number_input("P/S Q3", value=bolagsdata.get("P/S Q3", 0.0))
        ps_q4 = st.number_input("P/S Q4", value=bolagsdata.get("P/S Q4", 0.0))
        oms_idag = st.number_input("Omsättning idag", value=bolagsdata.get("Omsättning idag", 0.0))
        oms_next = st.number_input("Omsättning nästa år", value=bolagsdata.get("Omsättning nästa år", 0.0))
        antal = st.number_input("Antal aktier", value=bolagsdata.get("Antal aktier", 0.0))

        submitted = st.form_submit_button("Spara")

    if submitted and ticker:
        namn, kurs, valuta, utdelning, cagr = hämta_och_beräkna_data(ticker)

        ny_rad = {
            "Ticker": ticker,
            "Bolagsnamn": namn,
            "Utestående aktier": utestående,
            "P/S": ps,
            "P/S Q1": ps_q1,
            "P/S Q2": ps_q2,
            "P/S Q3": ps_q3,
            "P/S Q4": ps_q4,
            "Omsättning idag": oms_idag,
            "Omsättning nästa år": oms_next,
            "Antal aktier": antal,
            "Aktuell kurs": kurs,
            "Valuta": valuta,
            "Årlig utdelning": utdelning,
            "CAGR 5 år (%)": cagr
        }

        if cagr:
            if cagr > 100:
                tillväxt = 0.5
            elif cagr < 0:
                tillväxt = 0.02
            else:
                tillväxt = cagr / 100

            ny_rad["Omsättning om 2 år"] = oms_idag * ((1 + tillväxt) ** 2)
            ny_rad["Omsättning om 3 år"] = oms_idag * ((1 + tillväxt) ** 3)

        for nyckel in ["Omsättning om 2 år", "Omsättning om 3 år"]:
            if nyckel not in ny_rad:
                ny_rad[nyckel] = None

        ny_rad = beräkna_riktkurser(ny_rad)

        df = df[df["Ticker"] != ticker]
        df = pd.concat([df, pd.DataFrame([ny_rad])], ignore_index=True)
        df = säkerställ_kolumner(df)
        spara_data(df)
        st.success("Bolaget har sparats med uppdaterade beräkningar.")

def beräkna_riktkurser(rad):
    ps_värden = [rad.get("P/S"), rad.get("P/S Q1"), rad.get("P/S Q2"),
                 rad.get("P/S Q3"), rad.get("P/S Q4")]
    ps_värden = [v for v in ps_värden if isinstance(v, (int, float)) and v > 0]

    if not ps_värden or not rad.get("Utestående aktier"):
        return rad

    ps_snitt = sum(ps_värden) / len(ps_värden)
    rad["P/S-snitt"] = ps_snitt

    aktier = rad.get("Utestående aktier")

    for år, omsättning in {
        "idag": rad.get("Omsättning idag"),
        "om 1 år": rad.get("Omsättning nästa år"),
        "om 2 år": rad.get("Omsättning om 2 år"),
        "om 3 år": rad.get("Omsättning om 3 år"),
    }.items():
        if isinstance(omsättning, (int, float)) and omsättning > 0:
            riktkurs = (ps_snitt * omsättning) / aktier
            rad[f"Riktkurs {år}"] = riktkurs
        else:
            rad[f"Riktkurs {år}"] = None

    return rad

def analysvy(df):
    st.subheader("📊 Analysvy")

    sorteringsval = st.selectbox("Sortera efter uppsida i riktkurs:", [
        "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år"
    ])

    df = df.copy()
    if sorteringsval in df.columns:
        df["Uppside (%)"] = ((df[sorteringsval] - df["Aktuell kurs"]) / df["Aktuell kurs"]) * 100
        df = df.sort_values("Uppside (%)", ascending=False)

    st.markdown("### Välj bolag att visa")
    tickers = df["Ticker"].dropna().unique().tolist()
    valt_bolag = st.selectbox("Välj ett bolag", tickers)

    if valt_bolag:
        bolag = df[df["Ticker"] == valt_bolag]
        st.dataframe(bolag.transpose())

    st.markdown("### Alla bolag i databasen")
    st.dataframe(df)

def investeringsförslag(df):
    st.subheader("📈 Investeringsförslag")

    riktkursval = st.selectbox("Välj riktkurs att filtrera efter:", [
        "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år"
    ])

    df = df.copy()
    if riktkursval in df.columns:
        df["Uppside (%)"] = ((df[riktkursval] - df["Aktuell kurs"]) / df["Aktuell kurs"]) * 100
        df = df.sort_values("Uppside (%)", ascending=False)

        index = st.session_state.get("inv_index", 0)

        if not df.empty:
            bolag = df.iloc[index]
            st.markdown(f"### {bolag['Ticker']} – {bolag['Bolagsnamn']}")
            st.write(f"**Aktuell kurs:** {bolag['Aktuell kurs']}")
            st.write(f"**Riktkurs (val):** {bolag[riktkursval]}")
            st.write(f"**Uppside (%):** {round(bolag['Uppside (%)'], 2)} %")
            st.write(f"**Årlig utdelning:** {bolag['Årlig utdelning']} {bolag['Valuta']}")

            belopp = st.number_input("Tillgängligt belopp (SEK)", value=0)
            if belopp > 0 and bolag["Aktuell kurs"] > 0:
                antal_köp = int(belopp / bolag["Aktuell kurs"])
                antal_äger = bolag["Antal aktier"] if "Antal aktier" in bolag else 0
                portföljvärde = df["Antal aktier"] * df["Aktuell kurs"]
                total_portfölj = portföljvärde.sum()
                andel_nu = round((antal_äger * bolag["Aktuell kurs"]) / total_portfölj * 100, 2) if total_portfölj > 0 else 0
                andel_efter = round(((antal_äger + antal_köp) * bolag["Aktuell kurs"]) / total_portfölj * 100, 2) if total_portfölj > 0 else 0

                st.write(f"**Köpbara aktier:** {antal_köp}")
                st.write(f"**Äger redan:** {antal_äger}")
                st.write(f"**Andel av portfölj före köp:** {andel_nu} %")
                st.write(f"**Andel efter köp:** {andel_efter} %")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Föregående"):
                    st.session_state["inv_index"] = max(0, index - 1)
            with col2:
                if st.button("Nästa"):
                    st.session_state["inv_index"] = min(len(df) - 1, index + 1)
        else:
            st.warning("Inga bolag hittades.")

def main():
    df = hämta_data()
    df = konvertera_typer(df)
    df = beräkna_kolumner(df)

    menyval = st.sidebar.radio("📌 Meny", ["Lägg till / uppdatera bolag", "Analys", "Portfölj", "Investeringsförslag", "Uppdatera alla bolag"])

    if menyval == "Lägg till / uppdatera bolag":
        formulär(df)
    elif menyval == "Analys":
        analysvy(df)
    elif menyval == "Portfölj":
        visa_portfolj(df)
    elif menyval == "Investeringsförslag":
        investeringsförslag(df)
    elif menyval == "Uppdatera alla bolag":
        massuppdatera(df)

if __name__ == "__main__":
    main()
