import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import requests

# Google Sheets-konfiguration
SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"

# Funktion: skapa koppling till Google Sheet
def skapa_koppling():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_service_account_info(
        st.secrets["GOOGLE_CREDENTIALS"], scopes=scope
    )
    client = gspread.authorize(creds)
    sheet = client.open_by_url(SHEET_URL).worksheet(SHEET_NAME)
    return sheet

# Funktion: hämta data från Google Sheet
def hamta_data():
    sheet = skapa_koppling()
    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    return df

# Funktion: spara DataFrame till Google Sheet
def spara_data(df):
    sheet = skapa_koppling()
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.values.tolist())

# Funktion: säkerställ att nödvändiga kolumner finns
def saknade_kolumner(df):
    nödvändiga = [
        "Ticker", "Bolagsnamn", "Aktuell kurs", "Valuta", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om två år", "Omsättning om tre år",
        "Utestående aktier", "Riktkurs idag", "Riktkurs 2026", "Riktkurs 2027", "Kommentar"
    ]
    for kolumn in nödvändiga:
        if kolumn not in df.columns:
            df[kolumn] = ""
    return df

import streamlit as st
import pandas as pd
import gspread
from google.oauth2 import service_account

# Skapa koppling till Google Sheet
def skapa_koppling():
    scope = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = service_account.Credentials.from_service_account_info(
        st.secrets["GOOGLE_CREDENTIALS"],
        scopes=scope
    )
    client = gspread.authorize(creds)
    sheet = client.open_by_url(st.secrets["SHEET_URL"]).worksheet("Blad1")
    return sheet

# Läs in data från Google Sheet till DataFrame
def hamta_data():
    sheet = skapa_koppling()
    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    return df

# Skriv DataFrame till Google Sheet
def spara_data(df):
    sheet = skapa_koppling()
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.values.tolist())

# Kontrollera att alla nödvändiga kolumner finns
def saknade_kolumner(df, obligatoriska):
    return [col for col in obligatoriska if col not in df.columns]

def konvertera_till_ratt_typ(df):
    numeriska_kolumner = [
        "Aktuell kurs", "Valutakurs", "Utestående aktier", "Omsättning idag",
        "Omsättning 2025", "Omsättning 2026", "Omsättning 2027",
        "P/S nu", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "P/S-snitt",
        "Riktkurs nu", "Riktkurs 2026", "Riktkurs 2027"
    ]
    for kolumn in numeriska_kolumner:
        if kolumn in df.columns:
            df[kolumn] = pd.to_numeric(df[kolumn], errors="coerce")
    return df

# Funktion: räkna fram P/S-snitt (exkludera nollor)
def berakna_ps_snitt(rad):
    ps_varden = [rad.get("P/S Q1", 0), rad.get("P/S Q2", 0), rad.get("P/S Q3", 0), rad.get("P/S Q4", 0)]
    giltiga = [v for v in ps_varden if v and v > 0]
    if not giltiga:
        return None
    return sum(giltiga) / len(giltiga)

# Funktion: uppdatera beräkningar för alla rader
def uppdatera_berakningar(df):
    for i, rad in df.iterrows():
        ps_snitt = berakna_ps_snitt(rad)
        df.at[i, "P/S snitt"] = round(ps_snitt, 2) if ps_snitt else None

        try:
            # Riktkurser
            aktier = rad["Utestående aktier"]
            if aktier and ps_snitt:
                if rad["Omsättning idag"]:
                    df.at[i, "Riktkurs idag"] = round((rad["Omsättning idag"] * ps_snitt) / aktier, 2)
                if rad["Omsättning nästa år"]:
                    df.at[i, "Riktkurs 2026"] = round((rad["Omsättning nästa år"] * ps_snitt) / aktier, 2)
                if rad["Omsättning om två år"]:
                    df.at[i, "Riktkurs 2027"] = round((rad["Omsättning om två år"] * ps_snitt) / aktier, 2)
        except:
            pass
    return df

# Funktion: investeringsförslag baserat på målkurs och kapital
def investeringsforslag(df, tillgangligt_kapital):
    df["Undervärdering (%)"] = ((df["Riktkurs 2026"] - df["Aktuell kurs"]) / df["Aktuell kurs"]) * 100
    df = df.sort_values(by="Undervärdering (%)", ascending=False)

    forslag = []
    kapital = tillgangligt_kapital

    for _, rad in df.iterrows():
        if rad["Aktuell kurs"] <= 0:
            continue

        om_kursen_ar_billig = rad["Riktkurs 2026"] > rad["Aktuell kurs"]
        if om_kursen_ar_billig:
            antal = int(kapital // rad["Aktuell kurs"])
            if antal > 0:
                forslag.append({
                    "Ticker": rad["Ticker"],
                    "Köp antal": antal,
                    "Pris per aktie": rad["Aktuell kurs"],
                    "Totalt": antal * rad["Aktuell kurs"]
                })
                kapital -= antal * rad["Aktuell kurs"]

    return forslag, kapital

# Formulär: lägg till eller uppdatera bolagsdata manuellt
def visa_formular(df):
    with st.form("Lägg till/uppdatera bolag"):
        ticker = st.text_input("Ticker")
        namn = st.text_input("Bolagsnamn")
        kurs = st.number_input("Aktuell kurs (USD)", min_value=0.0, value=0.0)
        ps_idag = st.number_input("P/S idag", min_value=0.0, value=0.0)
        ps_q1 = st.number_input("P/S Q1", min_value=0.0, value=0.0)
        ps_q2 = st.number_input("P/S Q2", min_value=0.0, value=0.0)
        ps_q3 = st.number_input("P/S Q3", min_value=0.0, value=0.0)
        ps_q4 = st.number_input("P/S Q4", min_value=0.0, value=0.0)
        oms_idag = st.number_input("Omsättning idag (miljoner USD)", min_value=0.0, value=0.0)
        oms_2026 = st.number_input("Omsättning nästa år (miljoner USD)", min_value=0.0, value=0.0)
        oms_2027 = st.number_input("Omsättning om två år (miljoner USD)", min_value=0.0, value=0.0)
        aktier = st.number_input("Utestående aktier (miljoner)", min_value=0.0, value=0.0)

        submit = st.form_submit_button("Spara")

    if submit and ticker:
        ny_rad = {
            "Ticker": ticker,
            "Bolagsnamn": namn,
            "Aktuell kurs": kurs,
            "P/S idag": ps_idag,
            "P/S Q1": ps_q1,
            "P/S Q2": ps_q2,
            "P/S Q3": ps_q3,
            "P/S Q4": ps_q4,
            "Omsättning idag": oms_idag,
            "Omsättning nästa år": oms_2026,
            "Omsättning om två år": oms_2027,
            "Utestående aktier": aktier
        }

        index = df[df["Ticker"] == ticker].index
        if not index.empty:
            for kolumn, varde in ny_rad.items():
                df.at[index[0], kolumn] = varde
        else:
            df = pd.concat([df, pd.DataFrame([ny_rad])], ignore_index=True)

        st.success(f"{ticker} har sparats.")
        spara_data(df)
        st.experimental_rerun()

# Snitt-P/S baserat på angivna kvartal
def beräkna_snitt_ps(rad):
    ps_varden = [rad.get(f"P/S Q{i}", 0) for i in range(1, 5)]
    giltiga = [v for v in ps_varden if v > 0]
    return sum(giltiga) / len(giltiga) if giltiga else 0

# Riktkursberäkning
def beräkna_riktkurs(omsättning, snitt_ps, aktier):
    if omsättning > 0 and snitt_ps > 0 and aktier > 0:
        return (omsättning * snitt_ps) / aktier
    return 0

# Undervärdering i procent
def beräkna_undervärdering(riktkurs, aktuell_kurs):
    if riktkurs > 0 and aktuell_kurs > 0:
        return ((riktkurs - aktuell_kurs) / aktuell_kurs) * 100
    return 0

# Hämta aktuell kurs (via yfinance)
def uppdatera_aktuell_kurs(df):
    for i, rad in df.iterrows():
        ticker = rad["Ticker"]
        try:
            info = yf.Ticker(ticker).info
            ny_kurs = info.get("currentPrice", None)
            if ny_kurs:
                df.at[i, "Aktuell kurs"] = ny_kurs
        except Exception:
            st.warning(f"Kursen kunde inte hämtas för {ticker}")
    return df

# Konvertera kolumner till rätt typ
def konvertera_till_ratt_typ(df):
    kolumner = [
        "Aktuell kurs", "P/S idag", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om två år", "Utestående aktier"
    ]
    for kolumn in kolumner:
        df[kolumn] = pd.to_numeric(df[kolumn], errors="coerce").fillna(0.0)
    return df

# Uppdatera värderingar och riktkurser
def uppdatera_beräkningar(df):
    for i, rad in df.iterrows():
        snitt_ps = beräkna_snitt_ps(rad)
        df.at[i, "P/S snitt"] = round(snitt_ps, 2)

        df.at[i, "Riktkurs idag"] = round(beräkna_riktkurs(
            rad["Omsättning idag"], snitt_ps, rad["Utestående aktier"]
        ), 2)

        df.at[i, "Riktkurs 2026"] = round(beräkna_riktkurs(
            rad["Omsättning nästa år"], snitt_ps, rad["Utestående aktier"]
        ), 2)

        df.at[i, "Riktkurs 2027"] = round(beräkna_riktkurs(
            rad["Omsättning om två år"], snitt_ps, rad["Utestående aktier"]
        ), 2)

        df.at[i, "Undervärdering idag"] = round(beräkna_undervärdering(
            df.at[i, "Riktkurs idag"], rad["Aktuell kurs"]
        ), 2)

        df.at[i, "Undervärdering 2026"] = round(beräkna_undervärdering(
            df.at[i, "Riktkurs 2026"], rad["Aktuell kurs"]
        ), 2)

        df.at[i, "Undervärdering 2027"] = round(beräkna_undervärdering(
            df.at[i, "Riktkurs 2027"], rad["Aktuell kurs"]
        ), 2)
    return df

# Spara DataFrame till Google Sheet
def spara_data(sheet, df):
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.values.tolist())

# Export till Excel (valfritt)
def exportera_excel(df):
    buffer = BytesIO()
    df.to_excel(buffer, index=False)
    st.download_button("Ladda ner som Excel", buffer.getvalue(), file_name="aktiedata.xlsx")

def main():
    st.title("📈 Aktieanalys & Investeringsråd")

    # Anslut och hämta data
    sheet = skapa_koppling()
    df = hamta_data()
    df = konvertera_till_ratt_typ(df)

    # Visa aktuell valutakurs
    visa_valutakurs()

    # Sidomeny
    menyval = st.sidebar.radio("Välj vy", ["📊 Analys", "➕ Lägg till bolag", "🔁 Uppdatera värderingar", "💼 Investeringsråd", "📤 Export"])

    if menyval == "📊 Analys":
        visa_tabell(df)

    elif menyval == "➕ Lägg till bolag":
        df = lagg_till_bolag(df)
        spara_data(sheet, df)
        st.success("Bolag tillagt!")

    elif menyval == "🔁 Uppdatera värderingar":
        df = uppdatera_beräkningar(df)
        spara_data(sheet, df)
        st.success("Alla värderingar uppdaterade!")

    elif menyval == "💼 Investeringsråd":
        visa_investeringsrad(df)

    elif menyval == "📤 Export":
        exportera_excel(df)

if __name__ == "__main__":
    main()
